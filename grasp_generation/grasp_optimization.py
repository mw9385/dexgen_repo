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

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as Fn
import trimesh

from .grasp_sampler import Grasp, GraspSet
from .hand_model import HandModel, build_hand_model, random_rotation_6d, rotation_6d_to_matrix


# ---------------------------------------------------------------------------
# Analytical SDF for primitive shapes (exact, fast, no mesh queries)
# ---------------------------------------------------------------------------

class AnalyticalSDF:
    """
    Exact signed distance field for primitive shapes.

    Fully differentiable through PyTorch, no trimesh dependency at runtime.
    Much faster and more robust than mesh-based SDF.

    Supported shapes: cube (box), sphere, cylinder.
    """

    def __init__(self, shape_type: str, size: float, mesh: trimesh.Trimesh):
        self.shape_type = shape_type
        self.size = size
        self.mesh = mesh  # kept for normal queries in _extract_grasps

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance from points to the primitive surface.

        Negative = inside, Positive = outside, Zero = on surface.

        Args:
            points: (..., 3)

        Returns:
            sdf: (...,)
        """
        if self.shape_type == "sphere":
            return self._sdf_sphere(points, self.size / 2.0)
        elif self.shape_type == "cube":
            return self._sdf_box(points, self.size / 2.0)
        elif self.shape_type == "cylinder":
            return self._sdf_cylinder(points, self.size / 2.0, self.size / 2.0)
        else:
            # Fallback: approximate SDF using distance from centroid
            return self._sdf_sphere(points, self.size / 2.0)

    def surface_normals(self, points: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
        """Compute surface normals via finite differences on the SDF."""
        grad_x = self.query(points + _eps_vec(points, 0, eps)) - self.query(points - _eps_vec(points, 0, eps))
        grad_y = self.query(points + _eps_vec(points, 1, eps)) - self.query(points - _eps_vec(points, 1, eps))
        grad_z = self.query(points + _eps_vec(points, 2, eps)) - self.query(points - _eps_vec(points, 2, eps))
        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)
        return grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _sdf_sphere(points: torch.Tensor, radius: float) -> torch.Tensor:
        """SDF of a sphere centered at origin."""
        return torch.norm(points, dim=-1) - radius

    @staticmethod
    def _sdf_box(points: torch.Tensor, half_extent: float) -> torch.Tensor:
        """SDF of an axis-aligned box centered at origin."""
        h = half_extent
        q = torch.abs(points) - h
        outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        return outside + inside

    @staticmethod
    def _sdf_cylinder(points: torch.Tensor, radius: float, half_height: float) -> torch.Tensor:
        """SDF of a Z-aligned cylinder centered at origin."""
        xy = points[..., :2]
        z = points[..., 2]
        d_radial = torch.norm(xy, dim=-1) - radius
        d_height = torch.abs(z) - half_height
        outside = torch.norm(
            torch.stack([torch.clamp(d_radial, min=0.0), torch.clamp(d_height, min=0.0)], dim=-1),
            dim=-1,
        )
        inside = torch.clamp(torch.max(d_radial, d_height), max=0.0)
        return outside + inside


def _eps_vec(points: torch.Tensor, axis: int, eps: float) -> torch.Tensor:
    """Create an epsilon offset vector along the given axis."""
    v = torch.zeros_like(points)
    v[..., axis] = eps
    return v


# ---------------------------------------------------------------------------
# Mesh-based SDF fallback (for custom meshes only)
# ---------------------------------------------------------------------------

class MeshSDF:
    """
    Approximate signed distance field for arbitrary trimesh objects.

    Uses a voxel grid with chunked construction to avoid OOM.
    Only used as fallback when AnalyticalSDF can't be used.
    """

    def __init__(self, mesh: trimesh.Trimesh, resolution: int = 32, padding: float = 0.02):
        self.mesh = mesh
        bounds = mesh.bounds
        self.origin = bounds[0] - padding
        self.extent = (bounds[1] - bounds[0]) + 2 * padding
        self.resolution = resolution

        print(f"  [MeshSDF] Building voxel grid ({resolution}^3 = {resolution**3} points)...")
        sdf_values = self._build_sdf_grid(resolution)

        self.sdf_grid = torch.tensor(
            sdf_values.reshape(resolution, resolution, resolution),
            dtype=torch.float32,
        )
        self._origin_t = torch.tensor(self.origin, dtype=torch.float32)
        self._extent_t = torch.tensor(self.extent, dtype=torch.float32)
        print(f"  [MeshSDF] Done.")

    def _build_sdf_grid(self, res: int) -> np.ndarray:
        """Build SDF grid in chunks to avoid memory issues."""
        lin = np.linspace(0, 1, res)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        all_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        all_points = self.origin + all_points * self.extent

        total = len(all_points)
        chunk_size = min(10000, total)
        sdf_values = np.zeros(total, dtype=np.float32)

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = all_points[start:end]

            # Closest point distance
            try:
                _, distances, _ = trimesh.proximity.closest_point(self.mesh, chunk)
            except Exception:
                distances = np.full(len(chunk), 0.01, dtype=np.float32)

            # Inside/outside test
            try:
                inside = self.mesh.contains(chunk)
            except Exception:
                # Fallback: estimate via ray casting from centroid
                centroid = self.mesh.centroid
                dists_to_center = np.linalg.norm(chunk - centroid, axis=-1)
                radius_est = np.max(self.mesh.bounding_box.extents) / 2.0
                inside = dists_to_center < radius_est

            signs = np.ones(len(chunk), dtype=np.float32)
            signs[inside] = -1.0
            sdf_values[start:end] = signs * distances.astype(np.float32)

        return sdf_values

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """Query SDF values via trilinear interpolation."""
        device = points.device
        grid = self.sdf_grid.to(device)
        origin = self._origin_t.to(device)
        extent = self._extent_t.to(device)

        normalized = 2.0 * (points - origin) / (extent + 1e-8) - 1.0
        flat = normalized.reshape(1, 1, -1, 1, 3)
        grid_5d = grid.unsqueeze(0).unsqueeze(0)

        sampled = Fn.grid_sample(
            grid_5d, flat, mode="bilinear", padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(points.shape[:-1])

    def surface_normals(self, points: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
        """Compute surface normals via finite differences."""
        grad_x = self.query(points + _eps_vec(points, 0, eps)) - self.query(points - _eps_vec(points, 0, eps))
        grad_y = self.query(points + _eps_vec(points, 1, eps)) - self.query(points - _eps_vec(points, 1, eps))
        grad_z = self.query(points + _eps_vec(points, 2, eps)) - self.query(points - _eps_vec(points, 2, eps))
        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)
        return grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)


def build_sdf(mesh: trimesh.Trimesh, shape_type: str = "custom",
              size: float = 0.06, sdf_resolution: int = 32):
    """
    Factory function: use AnalyticalSDF for primitives, MeshSDF for custom meshes.
    """
    if shape_type in ("cube", "sphere", "cylinder"):
        print(f"  [SDF] Using analytical SDF for {shape_type} (exact, fast)")
        return AnalyticalSDF(shape_type, size, mesh)
    else:
        print(f"  [SDF] Using mesh-based SDF for custom shape (resolution={sdf_resolution})")
        return MeshSDF(mesh, resolution=sdf_resolution)


# ---------------------------------------------------------------------------
# Energy Functions
# ---------------------------------------------------------------------------

def energy_contact(fingertip_positions: torch.Tensor, sdf) -> torch.Tensor:
    """
    Contact energy: penalizes fingertips far from the object surface.
    E_contact = mean(sdf(tip)^2)
    """
    B, F, _ = fingertip_positions.shape
    sdf_vals = sdf.query(fingertip_positions.reshape(B * F, 3)).reshape(B, F)
    return (sdf_vals ** 2).mean(dim=-1)


def energy_penetration(fingertip_positions: torch.Tensor, sdf) -> torch.Tensor:
    """
    Penetration energy: penalizes fingertips inside the object.
    E_pen = mean(relu(-sdf)^2)
    """
    B, F, _ = fingertip_positions.shape
    sdf_vals = sdf.query(fingertip_positions.reshape(B * F, 3)).reshape(B, F)
    penetration = Fn.relu(-sdf_vals)
    return (penetration ** 2).mean(dim=-1)


def energy_force_closure(fingertip_positions: torch.Tensor, sdf,
                         mu: float = 0.5) -> torch.Tensor:
    """
    Differentiable force closure energy.
    E_fc = -min_singular_value(grasp_wrench_matrix)
    """
    B, num_fingers, _ = fingertip_positions.shape

    normals = sdf.surface_normals(fingertip_positions.reshape(B * num_fingers, 3))
    normals = normals.reshape(B, num_fingers, 3)

    forces = -normals
    torques = torch.cross(fingertip_positions, forces, dim=-1)

    # Grasp matrix: (B, 6, F)
    G = torch.cat([forces.transpose(1, 2), torques.transpose(1, 2)], dim=1)

    sv = torch.linalg.svdvals(G)
    min_sv = sv[:, -1]
    return -min_sv


def energy_self_collision(fingertip_positions: torch.Tensor,
                          min_distance: float = 0.01) -> torch.Tensor:
    """
    Self-collision energy: penalizes fingertips too close to each other.
    """
    B, num_fingers, _ = fingertip_positions.shape

    diff = fingertip_positions.unsqueeze(2) - fingertip_positions.unsqueeze(1)
    dists = torch.norm(diff, dim=-1)

    mask = torch.triu(torch.ones(num_fingers, num_fingers,
                                 device=fingertip_positions.device), diagonal=1)
    mask = mask.unsqueeze(0).expand(B, -1, -1)

    violations = Fn.relu(min_distance - dists) * mask
    return (violations ** 2).sum(dim=(1, 2))


def energy_joint_limits(joint_angles: torch.Tensor, hand_model: HandModel,
                        margin: float = 0.05) -> torch.Tensor:
    """Joint limit energy: penalizes joints near their limits."""
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
    """

    def __init__(
        self,
        hand_model: HandModel,
        mesh: trimesh.Trimesh,
        shape_type: str = "custom",
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
        sdf_resolution: int = 32,
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

        # Build SDF (analytical for primitives, mesh-based for custom)
        obj_size = float(np.max(mesh.bounding_box.extents))
        print(f"[GraspOptimizer] Building SDF for {shape_type} "
              f"(size={obj_size:.3f}m)...")
        self.sdf = build_sdf(mesh, shape_type, obj_size, sdf_resolution)
        print(f"[GraspOptimizer] SDF ready.")

        self.hand_model = self.hand_model.to(self.device)

    def optimize(
        self,
        num_grasps: int = 300,
        min_quality: float = 0.005,
        verbose: bool = True,
    ) -> GraspSet:
        """Run grasp optimization and return a GraspSet."""
        all_grasps = []
        num_batches = max(1, (num_grasps * 3) // self.batch_size + 1)

        for batch_idx in range(num_batches):
            if verbose:
                print(f"  [GraspOpt] Batch {batch_idx + 1}/{num_batches}, "
                      f"collected {len(all_grasps)} grasps so far")

            try:
                batch_grasps = self._optimize_batch()
                all_grasps.extend(batch_grasps)
            except Exception as e:
                print(f"  [GraspOpt] WARNING: batch {batch_idx + 1} failed: {e}")
                continue

            if len(all_grasps) >= num_grasps:
                break

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

        translation, rotation_6d, joint_angles = self._initialize_batch(B)
        translation.requires_grad_(True)
        rotation_6d.requires_grad_(True)
        joint_angles.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [translation, rotation_6d, joint_angles],
            lr=self.lr,
        )

        for step in range(self.num_iterations):
            optimizer.zero_grad()

            q_clamped = _soft_clamp(joint_angles, hand.joint_lower, hand.joint_upper)
            tips = hand.forward_kinematics(q_clamped, translation, rotation_6d)

            e_contact = energy_contact(tips, self.sdf)
            e_pen = energy_penetration(tips, self.sdf)
            e_fc = energy_force_closure(tips, self.sdf, self.mu)
            e_self = energy_self_collision(tips)
            e_joint = energy_joint_limits(q_clamped, hand)

            total = (
                self.w_contact * e_contact
                + self.w_penetration * e_pen
                + self.w_fc * e_fc
                + self.w_self * e_self
                + self.w_joint * e_joint
            )

            loss = total.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [translation, rotation_6d, joint_angles], max_norm=1.0
            )
            optimizer.step()

            # Log progress every 50 steps
            if step == 0 or (step + 1) % 50 == 0:
                print(f"    step {step+1}/{self.num_iterations}: "
                      f"loss={loss.item():.4f} "
                      f"contact={e_contact.mean().item():.6f} "
                      f"fc={e_fc.mean().item():.4f}")

        with torch.no_grad():
            q_final = hand.clamp_joints(joint_angles)
            tips_final = hand.forward_kinematics(q_final, translation, rotation_6d)
            R_final = rotation_6d_to_matrix(rotation_6d)
            e_contact_final = energy_contact(tips_final, self.sdf)
            e_fc_final = energy_force_closure(tips_final, self.sdf, self.mu)

        return self._extract_grasps(
            tips_final, q_final, translation, R_final,
            e_contact_final, e_fc_final,
        )

    def _initialize_batch(self, B: int):
        """Initialize random hand poses around the object."""
        device = self.device

        centroid = torch.tensor(
            self.mesh.centroid, dtype=torch.float32, device=device
        )
        radius = float(np.max(self.mesh.bounding_box.extents)) * 1.2

        dirs = torch.randn(B, 3, device=device)
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)
        translation = centroid.unsqueeze(0) + dirs * radius

        rotation_6d = random_rotation_6d(B, device)

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
        tips: torch.Tensor,
        joint_angles: torch.Tensor,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        e_contact: torch.Tensor,
        e_fc: torch.Tensor,
    ) -> List[Grasp]:
        """Convert optimized tensors to Grasp objects, filtering bad results."""
        from scipy.spatial.transform import Rotation as R_scipy

        grasps = []
        B = tips.shape[0]

        tips_np = tips.cpu().numpy()
        joints_np = joint_angles.cpu().numpy()
        trans_np = translation.cpu().numpy()
        rot_np = rotation.cpu().numpy()
        e_contact_np = e_contact.cpu().numpy()
        e_fc_np = e_fc.cpu().numpy()

        # Contact energy threshold: allow grasps within 3mm mean surface distance
        contact_threshold = 0.003 ** 2  # squared distance threshold

        for i in range(B):
            if e_contact_np[i] > contact_threshold:
                continue

            quality = float(max(0.0, -e_fc_np[i]))

            tip_pts = tips_np[i]

            # Compute surface normals at fingertip positions
            try:
                _, _, face_idx = trimesh.proximity.closest_point(self.mesh, tip_pts)
                normals = self.mesh.face_normals[face_idx].astype(np.float32)
            except Exception:
                # Fallback: normals pointing away from centroid
                centroid = self.mesh.centroid
                normals = tip_pts - centroid
                norms = np.linalg.norm(normals, axis=-1, keepdims=True)
                normals = (normals / (norms + 1e-8)).astype(np.float32)

            # Object pose in hand frame
            rot_mat = rot_np[i]
            t = trans_np[i]
            obj_pos_hand = (-rot_mat.T @ t).astype(np.float32)

            obj_quat_scipy = R_scipy.from_matrix(rot_mat.T).as_quat()  # (x,y,z,w)
            obj_quat_hand = np.array([
                obj_quat_scipy[3], obj_quat_scipy[0],
                obj_quat_scipy[1], obj_quat_scipy[2],
            ], dtype=np.float32)

            grasps.append(Grasp(
                fingertip_positions=tip_pts.astype(np.float32),
                contact_normals=normals,
                quality=quality,
                joint_angles=joints_np[i].astype(np.float32),
                object_pos_hand=obj_pos_hand,
                object_quat_hand=obj_quat_hand,
                object_pose_frame="root",
            ))

        return grasps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_clamp(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor,
                sharpness: float = 10.0) -> torch.Tensor:
    """Differentiable soft clamping using tanh."""
    mid = (low + high) / 2.0
    half_range = (high - low) / 2.0
    normalized = (x - mid) / (half_range + 1e-8)
    clamped_norm = torch.tanh(normalized)
    return mid + half_range * clamped_norm
