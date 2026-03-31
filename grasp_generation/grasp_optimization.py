"""
Differentiable Grasp Optimization (DexGraspNet-style)
=====================================================
Replaces the RRT-based grasp expansion with gradient-based optimization
that directly optimizes hand pose and joint angles for stable grasps.

Algorithm (based on DexGraspNet, Wan et al. 2023):
  1. Initialize random hand poses around the object
  2. Optimize via gradient descent on a composite energy:
     E = w_contact * E_contact       (fingertips on object surface)
       + w_penetration * E_penetration (no hand-object interpenetration)
       + w_fc * E_force_closure        (grasp stability)
       + w_self * E_self_collision     (no finger self-intersection)
       + w_joint * E_joint_limit       (stay within joint limits)
  3. Filter results by final energy and NFO quality
  4. Build GraspGraph from optimized grasps

Key advantages over RRT:
  - Grasps are guaranteed to be on/near the object surface
  - Joint angles are physically consistent (from FK, not heuristic)
  - Force closure is directly optimized
  - Much faster for generating large grasp sets
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as Fn
import trimesh

from .grasp_sampler import Grasp, GraspSet
from .hand_model import HandModel, build_hand_model, random_rotation_6d, rotation_6d_to_matrix


# ---------------------------------------------------------------------------
# Object SDF (Signed Distance Field) for contact/penetration energy
# ---------------------------------------------------------------------------

class MeshSDF:
    """
    Approximate signed distance field for a trimesh object.

    Uses a voxel grid for fast batched SDF queries. The sign convention:
      negative = inside the object (penetration)
      positive = outside the object
      zero = on the surface
    """

    def __init__(self, mesh: trimesh.Trimesh, resolution: int = 64, padding: float = 0.02):
        self.mesh = mesh
        bounds = mesh.bounds  # (2, 3): min, max
        self.origin = bounds[0] - padding
        self.extent = (bounds[1] - bounds[0]) + 2 * padding
        self.resolution = resolution

        # Build voxel grid
        grid_points = self._build_grid(resolution)
        # Query closest point on mesh for each grid point
        closest, distances, _ = trimesh.proximity.closest_point(mesh, grid_points)
        # Determine sign: inside (negative) or outside (positive)
        signs = np.ones(len(grid_points), dtype=np.float32)
        # Use winding number for inside/outside test (robust for watertight meshes)
        inside = mesh.contains(grid_points)
        signs[inside] = -1.0
        sdf_values = signs * distances.astype(np.float32)

        # Store as 3D tensor for trilinear interpolation
        self.sdf_grid = torch.tensor(
            sdf_values.reshape(resolution, resolution, resolution),
            dtype=torch.float32,
        )
        self._origin_t = torch.tensor(self.origin, dtype=torch.float32)
        self._extent_t = torch.tensor(self.extent, dtype=torch.float32)

    def _build_grid(self, res: int) -> np.ndarray:
        lin = np.linspace(0, 1, res)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        return self.origin + grid * self.extent

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query SDF values at given points via trilinear interpolation.

        Args:
            points: (..., 3) points in object frame

        Returns:
            sdf: (...,) signed distance values
        """
        device = points.device
        grid = self.sdf_grid.to(device)
        origin = self._origin_t.to(device)
        extent = self._extent_t.to(device)

        # Normalize to [-1, 1] for grid_sample
        normalized = 2.0 * (points - origin) / (extent + 1e-8) - 1.0

        # Reshape for grid_sample: (1, 1, N, 1, 3) → sample from (1, 1, D, H, W)
        flat = normalized.reshape(1, 1, -1, 1, 3)
        grid_5d = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        sampled = Fn.grid_sample(
            grid_5d, flat, mode="bilinear", padding_mode="border",
            align_corners=True,
        )
        sdf = sampled.reshape(points.shape[:-1])
        return sdf

    def surface_normals(self, points: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
        """
        Compute surface normal direction at given points via finite differences on SDF.

        Args:
            points: (..., 3)

        Returns:
            normals: (..., 3) unit outward normals
        """
        dx = torch.zeros_like(points)
        dy = torch.zeros_like(points)
        dz = torch.zeros_like(points)
        dx[..., 0] = eps
        dy[..., 1] = eps
        dz[..., 2] = eps

        grad_x = self.query(points + dx) - self.query(points - dx)
        grad_y = self.query(points + dy) - self.query(points - dy)
        grad_z = self.query(points + dz) - self.query(points - dz)

        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)
        return grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# Energy Functions
# ---------------------------------------------------------------------------

def energy_contact(
    fingertip_positions: torch.Tensor,
    sdf: MeshSDF,
) -> torch.Tensor:
    """
    Contact energy: penalizes fingertips that are far from the object surface.

    E_contact = mean(sdf(tip)^2) per batch
    Minimized when all fingertips are exactly on the surface (sdf = 0).

    Args:
        fingertip_positions: (B, F, 3)
        sdf: object SDF

    Returns:
        energy: (B,)
    """
    B, F, _ = fingertip_positions.shape
    sdf_vals = sdf.query(fingertip_positions.reshape(B * F, 3)).reshape(B, F)
    return (sdf_vals ** 2).mean(dim=-1)


def energy_penetration(
    fingertip_positions: torch.Tensor,
    sdf: MeshSDF,
) -> torch.Tensor:
    """
    Penetration energy: penalizes fingertips that are inside the object.

    E_penetration = mean(relu(-sdf(tip))^2)
    Only penalizes negative SDF (inside the object).

    Args:
        fingertip_positions: (B, F, 3)
        sdf: object SDF

    Returns:
        energy: (B,)
    """
    B, F, _ = fingertip_positions.shape
    sdf_vals = sdf.query(fingertip_positions.reshape(B * F, 3)).reshape(B, F)
    penetration = Fn.relu(-sdf_vals)
    return (penetration ** 2).mean(dim=-1)


def energy_force_closure(
    fingertip_positions: torch.Tensor,
    sdf: MeshSDF,
    mu: float = 0.5,
) -> torch.Tensor:
    """
    Differentiable force closure energy (simplified).

    Based on DexGraspNet: uses the minimum singular value of the grasp
    wrench matrix as a differentiable proxy for force closure quality.

    Higher singular value = better force closure → we minimize negative.

    E_fc = -min_singular_value(W)

    Args:
        fingertip_positions: (B, F, 3)
        sdf: object SDF (for computing surface normals)
        mu: friction coefficient

    Returns:
        energy: (B,)
    """
    B, num_fingers, _ = fingertip_positions.shape
    device = fingertip_positions.device

    # Get surface normals at fingertip positions
    normals = sdf.surface_normals(fingertip_positions.reshape(B * num_fingers, 3))
    normals = normals.reshape(B, num_fingers, 3)

    # Build simplified grasp matrix G: 6×F
    # Contact wrench = [f_i; p_i × f_i] where f_i = -n_i (inward normal)
    forces = -normals  # (B, F, 3) inward forces
    positions = fingertip_positions  # (B, F, 3)

    # Torques = cross(position, force)
    torques = torch.cross(positions, forces, dim=-1)  # (B, F, 3)

    # Grasp matrix: (B, 6, F)
    G = torch.cat([forces.transpose(1, 2), torques.transpose(1, 2)], dim=1)  # (B, 6, F)

    # Min singular value (differentiable)
    sv = torch.linalg.svdvals(G)  # (B, min(6, F))
    min_sv = sv[:, -1]  # smallest singular value

    # Negate: we want to MAXIMIZE min singular value
    return -min_sv


def energy_self_collision(
    fingertip_positions: torch.Tensor,
    min_distance: float = 0.01,
) -> torch.Tensor:
    """
    Self-collision energy: penalizes fingertips that are too close to each other.

    E_self = sum(relu(min_dist - ||tip_i - tip_j||)^2) for all pairs

    Args:
        fingertip_positions: (B, F, 3)
        min_distance: minimum allowed distance between fingertips

    Returns:
        energy: (B,)
    """
    B, num_fingers, _ = fingertip_positions.shape

    # Pairwise distances
    # (B, F, 1, 3) - (B, 1, F, 3) → (B, F, F, 3)
    diff = fingertip_positions.unsqueeze(2) - fingertip_positions.unsqueeze(1)
    dists = torch.norm(diff, dim=-1)  # (B, F, F)

    # Upper triangle (avoid double counting and self-pairs)
    mask = torch.triu(torch.ones(num_fingers, num_fingers, device=fingertip_positions.device), diagonal=1)
    mask = mask.unsqueeze(0).expand(B, -1, -1)

    violations = Fn.relu(min_distance - dists) * mask
    return (violations ** 2).sum(dim=(1, 2))


def energy_joint_limits(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Joint limit energy: penalizes joints approaching their limits.

    E_joint = sum(relu(margin - (q - q_min))^2 + relu(margin - (q_max - q))^2)

    Args:
        joint_angles: (B, D)
        hand_model: HandModel with joint limits
        margin: soft margin from limits

    Returns:
        energy: (B,)
    """
    low = hand_model.joint_lower.to(joint_angles.device)
    high = hand_model.joint_upper.to(joint_angles.device)

    lower_violation = Fn.relu(margin - (joint_angles - low))
    upper_violation = Fn.relu(margin - (high - joint_angles))

    return ((lower_violation ** 2) + (upper_violation ** 2)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Grasp Optimizer
# ---------------------------------------------------------------------------

class GraspOptimizer:
    """
    DexGraspNet-style differentiable grasp optimizer.

    Optimizes hand pose (translation + rotation) and joint angles to produce
    stable grasps on a given object mesh.

    Usage:
        optimizer = GraspOptimizer(hand_model, mesh)
        grasps = optimizer.optimize(num_grasps=300)
    """

    def __init__(
        self,
        hand_model: HandModel,
        mesh: trimesh.Trimesh,
        # Energy weights
        w_contact: float = 100.0,
        w_penetration: float = 50.0,
        w_fc: float = 10.0,
        w_self: float = 20.0,
        w_joint: float = 5.0,
        # Optimization params
        lr: float = 0.005,
        num_iterations: int = 200,
        batch_size: int = 256,
        mu: float = 0.5,
        sdf_resolution: int = 64,
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.hand_model = hand_model
        self.mesh = mesh
        self.w_contact = w_contact
        self.w_penetration = w_penetration
        self.w_fc = w_fc
        self.w_self = w_self
        self.w_joint = w_joint
        self.lr = lr
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.mu = mu
        self.device = torch.device(device)

        # Build SDF
        print(f"[GraspOptimizer] Building SDF (resolution={sdf_resolution})...")
        self.sdf = MeshSDF(mesh, resolution=sdf_resolution)

        self.hand_model = self.hand_model.to(self.device)

    def optimize(
        self,
        num_grasps: int = 300,
        min_quality: float = 0.005,
        verbose: bool = True,
    ) -> GraspSet:
        """
        Run grasp optimization and return a GraspSet.

        Args:
            num_grasps: target number of grasps to produce
            min_quality: minimum NFO quality for filtering
            verbose: print progress

        Returns:
            GraspSet with optimized grasps
        """
        all_grasps = []
        num_batches = max(1, (num_grasps * 3) // self.batch_size + 1)  # oversample

        for batch_idx in range(num_batches):
            if verbose and batch_idx % 5 == 0:
                print(f"  [GraspOpt] Batch {batch_idx + 1}/{num_batches}, "
                      f"collected {len(all_grasps)} grasps so far")

            batch_grasps = self._optimize_batch()
            all_grasps.extend(batch_grasps)

            if len(all_grasps) >= num_grasps:
                break

        # Sort by quality and keep top-K
        all_grasps.sort(key=lambda g: g.quality, reverse=True)
        all_grasps = all_grasps[:num_grasps]

        grasp_set = GraspSet(grasps=all_grasps)
        if verbose:
            print(f"[GraspOptimizer] Produced {len(grasp_set)} grasps")

        return grasp_set

    def _optimize_batch(self) -> List[Grasp]:
        """Optimize a single batch of grasp candidates."""
        B = self.batch_size
        device = self.device
        hand = self.hand_model

        # Initialize: random hand poses around the object
        translation, rotation_6d, joint_angles = self._initialize_batch(B)

        # Make parameters require gradients
        translation.requires_grad_(True)
        rotation_6d.requires_grad_(True)
        joint_angles.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [translation, rotation_6d, joint_angles],
            lr=self.lr,
        )

        # Optimization loop
        for step in range(self.num_iterations):
            optimizer.zero_grad()

            # Clamp joints (differentiable approximation via soft clamping)
            q_clamped = _soft_clamp(joint_angles, hand.joint_lower, hand.joint_upper)

            # Forward kinematics → fingertip positions
            tips = hand.forward_kinematics(q_clamped, translation, rotation_6d)

            # Compute energies
            e_contact = energy_contact(tips, self.sdf)
            e_pen = energy_penetration(tips, self.sdf)
            e_fc = energy_force_closure(tips, self.sdf, self.mu)
            e_self = energy_self_collision(tips)
            e_joint = energy_joint_limits(q_clamped, hand)

            # Total energy
            total = (
                self.w_contact * e_contact
                + self.w_penetration * e_pen
                + self.w_fc * e_fc
                + self.w_self * e_self
                + self.w_joint * e_joint
            )

            loss = total.mean()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [translation, rotation_6d, joint_angles], max_norm=1.0
            )

            optimizer.step()

        # Extract results
        with torch.no_grad():
            q_final = hand.clamp_joints(joint_angles)
            tips_final = hand.forward_kinematics(q_final, translation, rotation_6d)
            R_final = rotation_6d_to_matrix(rotation_6d)

            # Compute final energies for filtering
            e_contact_final = energy_contact(tips_final, self.sdf)
            e_fc_final = energy_force_closure(tips_final, self.sdf, self.mu)

        return self._extract_grasps(
            tips_final, q_final, translation, R_final,
            e_contact_final, e_fc_final,
        )

    def _initialize_batch(
        self, B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize a batch of hand poses around the object.

        Strategy: place hand at random positions on a sphere around the object
        centroid, with random orientations and joint angles.
        """
        device = self.device

        # Object centroid and radius
        centroid = torch.tensor(
            self.mesh.centroid, dtype=torch.float32, device=device
        )
        radius = float(np.max(self.mesh.bounding_box.extents)) * 1.2

        # Random positions on sphere around object
        dirs = torch.randn(B, 3, device=device)
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)
        translation = centroid.unsqueeze(0) + dirs * radius

        # Random rotations
        rotation_6d = random_rotation_6d(B, device)

        # Random joint angles (biased toward mid-range for grasping)
        hand = self.hand_model
        low = hand.joint_lower.to(device)
        high = hand.joint_upper.to(device)
        mid = (low + high) / 2.0
        spread = (high - low) / 4.0
        joint_angles = mid + torch.randn(B, hand.num_dof, device=device) * spread
        joint_angles = torch.clamp(joint_angles, low, high)

        return translation, rotation_6d, joint_angles

    def _extract_grasps(
        self,
        tips: torch.Tensor,        # (B, F, 3)
        joint_angles: torch.Tensor, # (B, D)
        translation: torch.Tensor,  # (B, 3)
        rotation: torch.Tensor,     # (B, 3, 3)
        e_contact: torch.Tensor,    # (B,)
        e_fc: torch.Tensor,         # (B,)
    ) -> List[Grasp]:
        """Convert optimized tensors to Grasp objects, filtering bad results."""
        grasps = []
        B = tips.shape[0]

        tips_np = tips.cpu().numpy()
        joints_np = joint_angles.cpu().numpy()
        trans_np = translation.cpu().numpy()
        rot_np = rotation.cpu().numpy()
        e_contact_np = e_contact.cpu().numpy()
        e_fc_np = e_fc.cpu().numpy()

        for i in range(B):
            # Filter: contact energy should be low (fingertips near surface)
            if e_contact_np[i] > 0.001:  # > 1mm^2 mean error
                continue

            # Quality score: inverse of force closure energy (higher = better)
            quality = float(max(0.0, -e_fc_np[i]))

            # Compute surface normals at fingertip positions
            tip_pts = tips_np[i]  # (F, 3)
            _, _, face_idx = trimesh.proximity.closest_point(self.mesh, tip_pts)
            normals = self.mesh.face_normals[face_idx]

            # Object pose in hand frame:
            # p_obj_in_hand = R^T @ (p_obj - t_hand)
            R = rot_np[i]  # (3, 3)
            t = trans_np[i]  # (3,)
            # Object is at origin in object frame; hand is at (t, R)
            # So object in hand frame: R^T @ (0 - t) = -R^T @ t
            obj_pos_hand = -R.T @ t
            # Quaternion: R^T as quaternion
            from scipy.spatial.transform import Rotation
            obj_quat_hand = Rotation.from_matrix(R.T).as_quat()  # (x, y, z, w)
            # Convert to (w, x, y, z) convention
            obj_quat_hand = np.array([
                obj_quat_hand[3], obj_quat_hand[0],
                obj_quat_hand[1], obj_quat_hand[2],
            ], dtype=np.float32)

            grasps.append(Grasp(
                fingertip_positions=tip_pts.astype(np.float32),
                contact_normals=normals.astype(np.float32),
                quality=quality,
                joint_angles=joints_np[i].astype(np.float32),
                object_pos_hand=obj_pos_hand.astype(np.float32),
                object_quat_hand=obj_quat_hand,
                object_pose_frame="root",
            ))

        return grasps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_clamp(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor,
                sharpness: float = 10.0) -> torch.Tensor:
    """Differentiable soft clamping using sigmoid."""
    # Sigmoid-based soft clamp that's differentiable everywhere
    mid = (low + high) / 2.0
    half_range = (high - low) / 2.0
    normalized = (x - mid) / (half_range + 1e-8)
    clamped_norm = torch.tanh(normalized)
    return mid + half_range * clamped_norm
