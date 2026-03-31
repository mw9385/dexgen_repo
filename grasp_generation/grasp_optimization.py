"""
Grasp Optimization (DexGraspNet Integration)
=============================================
Uses DexGraspNet's exact algorithm for grasp generation:
  - Energy: E_fc (wrench Jacobian) + E_dis + E_pen + E_spen + E_joints
  - Optimizer: Simulated Annealing with RMSProp
  - Hand model: MJCF-based FK via pytorch_kinematics
  - Object model: Analytical SDF for primitives

References:
  - DexGraspNet (Wang et al., ICRA 2023)
  - PKU-EPIC/DexGraspNet on GitHub
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import trimesh

from .grasp_sampler import Grasp, GraspSet
from .hand_model import (
    DexGraspNetHandModel,
    PrimitiveObjectModel,
    build_hand_model,
    build_object_model,
    robust_compute_rotation_matrix_from_ortho6d,
    rotation_matrix_to_ortho6d,
    random_rotation_6d,
)


# ---------------------------------------------------------------------------
# Energy functions (from DexGraspNet energy.py)
# ---------------------------------------------------------------------------

def cal_energy(hand_model: DexGraspNetHandModel,
               object_model: PrimitiveObjectModel,
               w_dis=100.0, w_pen=100.0, w_spen=10.0, w_joints=1.0):
    """
    Compute composite grasp energy (exact DexGraspNet formulation).

    Returns:
        total_energy, E_fc, E_dis, E_pen, E_spen, E_joints — all (B,)
    """
    device = hand_model.device
    batch_size, n_contact, _ = hand_model.contact_points.shape

    # E_dis: contact distance energy
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc: force closure energy (wrench Jacobian)
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor(
        [[0, 0, 0, 0, 0, -1, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, -1, 0, 0],
         [0, -1, 0, 1, 0, 0, 0, 0, 0]],
        dtype=torch.float, device=device,
    )
    g = torch.cat([
        torch.eye(3, dtype=torch.float, device=device)
            .expand(batch_size, n_contact, 3, 3)
            .reshape(batch_size, 3 * n_contact, 3),
        (hand_model.contact_points @ transformation_matrix)
            .view(batch_size, 3 * n_contact, 3),
    ], dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm

    # E_joints: joint limit violation
    E_joints = (
        torch.sum(
            (hand_model.hand_pose[:, 9:] > hand_model.joints_upper)
            * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1)
        + torch.sum(
            (hand_model.hand_pose[:, 9:] < hand_model.joints_lower)
            * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)
    )

    # E_pen: reverse penetration (object surface points → hand SDF)
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale
    # Expand surface points to batch size
    if object_surface_points.shape[0] == 1:
        object_surface_points = object_surface_points.expand(batch_size, -1, -1)
    distances = hand_model.cal_distance(object_surface_points)
    distances = distances.clone()
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    # E_spen: self-penetration
    E_spen = hand_model.self_penetration()

    total = E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints
    return total, E_fc, E_dis, E_pen, E_spen, E_joints


# ---------------------------------------------------------------------------
# Simulated Annealing Optimizer (from DexGraspNet optimizer.py)
# ---------------------------------------------------------------------------

class AnnealingOptimizer:
    """
    DexGraspNet's Simulated Annealing + RMSProp optimizer.

    Features:
      - Random resampling of contact point indices
      - RMSProp adaptive learning rate
      - Metropolis acceptance criterion with temperature decay
    """

    def __init__(self, hand_model: DexGraspNetHandModel,
                 switch_possibility=0.5,
                 starting_temperature=18,
                 temperature_decay=0.95,
                 annealing_period=30,
                 step_size=0.005,
                 stepsize_period=50,
                 mu=0.98,
                 device="cuda"):
        self.hand_model = hand_model
        self.device = device
        self.switch_possibility = switch_possibility
        self.starting_temperature = torch.tensor(starting_temperature, dtype=torch.float, device=device)
        self.temperature_decay = torch.tensor(temperature_decay, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(annealing_period, dtype=torch.long, device=device)
        self.step_size = torch.tensor(step_size, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(stepsize_period, dtype=torch.long, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float, device=device)
        self.step = 0

        self.old_hand_pose = None
        self.old_contact_point_indices = None
        self.old_global_transformation = None
        self.old_global_rotation = None
        self.old_current_status = None
        self.old_contact_points = None
        self.old_grad_hand_pose = None
        self.ema_grad_hand_pose = torch.zeros(
            self.hand_model.n_dofs + 9, dtype=torch.float, device=device
        )

    def try_step(self):
        s = self.step_size * self.temperature_decay ** torch.div(
            self.step, self.step_size_period, rounding_mode='floor')
        step_size = torch.zeros(
            *self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device
        ) + s

        self.ema_grad_hand_pose = (
            self.mu * (self.hand_model.hand_pose.grad ** 2).mean(0)
            + (1 - self.mu) * self.ema_grad_hand_pose
        )

        hand_pose = self.hand_model.hand_pose - (
            step_size * self.hand_model.hand_pose.grad
            / (torch.sqrt(self.ema_grad_hand_pose) + 1e-6)
        )

        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = (
            torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device)
            < self.switch_possibility
        )
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(
            self.hand_model.n_contact_candidates,
            size=[switch_mask.sum()], device=self.device,
        )

        self.old_hand_pose = self.hand_model.hand_pose
        self.old_contact_point_indices = self.hand_model.contact_point_indices
        self.old_global_transformation = self.hand_model.global_translation
        self.old_global_rotation = self.hand_model.global_rotation
        self.old_current_status = self.hand_model.current_status
        self.old_contact_points = self.hand_model.contact_points
        self.old_grad_hand_pose = self.hand_model.hand_pose.grad
        self.hand_model.set_parameters(hand_pose, contact_point_indices)

        self.step += 1
        return s

    def accept_step(self, energy, new_energy):
        batch_size = energy.shape[0]
        temperature = self.starting_temperature * self.temperature_decay ** torch.div(
            self.step, self.annealing_period, rounding_mode='floor')

        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)
        accept = alpha < torch.exp((energy - new_energy) / temperature)

        with torch.no_grad():
            reject = ~accept
            self.hand_model.hand_pose[reject] = self.old_hand_pose[reject]
            self.hand_model.contact_point_indices[reject] = self.old_contact_point_indices[reject]
            self.hand_model.global_translation[reject] = self.old_global_transformation[reject]
            self.hand_model.global_rotation[reject] = self.old_global_rotation[reject]
            self.hand_model.current_status = self.hand_model.chain.forward_kinematics(
                self.hand_model.hand_pose[:, 9:]
            )
            self.hand_model.contact_points[reject] = self.old_contact_points[reject]
            self.hand_model.hand_pose.grad[reject] = self.old_grad_hand_pose[reject]

        return accept, temperature

    def zero_grad(self):
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()


# ---------------------------------------------------------------------------
# Initialization (based on DexGraspNet initializations.py)
# ---------------------------------------------------------------------------

def initialize_grasp_poses(hand_model: DexGraspNetHandModel,
                           object_model: PrimitiveObjectModel,
                           batch_size: int,
                           n_contact: int = 4,
                           distance_range=(0.12, 0.25),
                           jitter_strength: float = 0.3,
                           device: str = "cuda"):
    """
    Initialize hand poses around the object for Simulated Annealing.

    Adapted from DexGraspNet's initialize_convex_hull for primitive shapes.
    """
    mesh = object_model.mesh

    # Sample approach points from inflated convex hull
    hull = mesh.convex_hull
    vertices = hull.vertices.copy()
    vertices += 0.15 * vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8)
    inflated = trimesh.Trimesh(vertices=vertices, faces=hull.faces).convex_hull

    # Sample approach directions using FPS-like spreading
    hull_points, _ = trimesh.sample.sample_surface(inflated, batch_size * 10)
    # Subsample uniformly
    indices = np.random.choice(len(hull_points), size=batch_size, replace=len(hull_points) < batch_size)
    p = torch.tensor(hull_points[indices], dtype=torch.float, device=device)

    # Find closest point on original mesh → approach normal
    closest, _, _ = trimesh.proximity.closest_point(mesh, p.cpu().numpy())
    closest = torch.tensor(closest, dtype=torch.float, device=device)
    n = (closest - p) / (torch.norm(closest - p, dim=1, keepdim=True) + 1e-8)

    # Sample random parameters
    distance = (
        distance_range[0]
        + (distance_range[1] - distance_range[0])
        * torch.rand(batch_size, dtype=torch.float, device=device)
    )
    deviate_theta = (-math.pi / 6 + math.pi / 3 * torch.rand(batch_size, device=device))
    process_theta = 2 * math.pi * torch.rand(batch_size, device=device)
    rotate_theta = 2 * math.pi * torch.rand(batch_size, device=device)

    # Build rotation matrices
    import transforms3d
    rotation = torch.zeros(batch_size, 3, 3, dtype=torch.float, device=device)
    translation = torch.zeros(batch_size, 3, dtype=torch.float, device=device)

    rotation_hand = torch.tensor(
        transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'),
        dtype=torch.float, device=device,
    )

    for j in range(batch_size):
        rot_local = torch.tensor(
            transforms3d.euler.euler2mat(
                float(process_theta[j]), float(deviate_theta[j]),
                float(rotate_theta[j]), axes='rzxz'),
            dtype=torch.float, device=device,
        )
        rot_global = torch.tensor(
            transforms3d.euler.euler2mat(
                math.atan2(float(n[j, 1]), float(n[j, 0])) - math.pi / 2,
                -math.acos(float(torch.clamp(n[j, 2], -1.0, 1.0))),
                0, axes='rzxz'),
            dtype=torch.float, device=device,
        )
        rotation[j] = rot_global @ rot_local @ rotation_hand
        approach_dir = (rot_global @ rot_local @ torch.tensor(
            [0, 0, 1], dtype=torch.float, device=device))
        translation[j] = p[j] - distance[j] * approach_dir

    # Initialize joint angles (DexGraspNet canonical pose + jitter)
    joint_angles_mu = torch.tensor(
        [0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0,
         0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0],
        dtype=torch.float, device=device,
    )
    joint_angles_sigma = jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros(batch_size, hand_model.n_dofs, dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            float(joint_angles_mu[i]), float(joint_angles_sigma[i]),
            float(hand_model.joints_lower[i]) - 1e-6,
            float(hand_model.joints_upper[i]) + 1e-6,
        )

    # Assemble hand_pose: [translation(3) | rot6d(6) | joints(22)]
    rot6d = rotation.transpose(1, 2)[:, :2].reshape(-1, 6)
    hand_pose = torch.cat([translation, rot6d, joint_angles], dim=1)
    hand_pose.requires_grad_()

    # Random contact point indices
    contact_point_indices = torch.randint(
        hand_model.n_contact_candidates,
        size=[batch_size, n_contact], device=device,
    )

    hand_model.set_parameters(hand_pose, contact_point_indices)


# ---------------------------------------------------------------------------
# Main Grasp Optimizer
# ---------------------------------------------------------------------------

class GraspOptimizer:
    """
    DexGraspNet-based grasp optimizer for our pipeline.

    Integrates:
      - DexGraspNet's energy functions (E_fc, E_dis, E_pen, E_spen, E_joints)
      - Simulated Annealing + RMSProp
      - Initialization from inflated convex hull
      - Filtering by energy thresholds
      - Conversion to our Grasp/GraspSet format
    """

    def __init__(
        self,
        hand_model: DexGraspNetHandModel,
        mesh: trimesh.Trimesh,
        shape_type: str = "custom",
        size: float = 0.06,
        # Energy weights (DexGraspNet defaults)
        w_dis: float = 100.0,
        w_pen: float = 100.0,
        w_spen: float = 10.0,
        w_joints: float = 1.0,
        # SA optimizer params
        n_iter: int = 2000,
        batch_size: int = 128,
        n_contact: int = 4,
        step_size: float = 0.005,
        starting_temperature: float = 18.0,
        temperature_decay: float = 0.95,
        # Filtering thresholds
        thres_fc: float = 0.3,
        thres_dis: float = 0.005,
        thres_pen: float = 0.02,
        # Device
        device: str = "cuda",
        # Unused params (for backward compat with CLI)
        **kwargs,
    ):
        self.hand_model = hand_model
        self.mesh = mesh
        self.shape_type = shape_type
        self.size = size
        self.w_dis = w_dis
        self.w_pen = w_pen
        self.w_spen = w_spen
        self.w_joints = w_joints
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_contact = n_contact
        self.step_size = step_size
        self.starting_temperature = starting_temperature
        self.temperature_decay = temperature_decay
        self.thres_fc = thres_fc
        self.thres_dis = thres_dis
        self.thres_pen = thres_pen
        self.device = device

        # Build object model
        self.object_model = build_object_model(
            mesh, shape_type, size, device=device,
        )
        print(f"[GraspOptimizer] Ready: {shape_type} (size={size:.3f}m), "
              f"batch={batch_size}, iters={n_iter}")

    def optimize(self, num_grasps: int = 300, verbose: bool = True, **kwargs) -> GraspSet:
        """Run grasp optimization and return a GraspSet."""
        all_grasps = []
        # Oversample to account for filtering
        num_batches = max(1, (num_grasps * 4) // self.batch_size + 1)

        for batch_idx in range(num_batches):
            if verbose:
                print(f"  [GraspOpt] Batch {batch_idx + 1}/{num_batches}, "
                      f"collected {len(all_grasps)} grasps")

            try:
                batch_grasps = self._optimize_batch()
                all_grasps.extend(batch_grasps)
            except Exception as e:
                print(f"  [GraspOpt] WARNING: batch {batch_idx+1} failed: {e}")
                import traceback; traceback.print_exc()
                continue

            if len(all_grasps) >= num_grasps:
                break

        all_grasps.sort(key=lambda g: g.quality, reverse=True)
        all_grasps = all_grasps[:num_grasps]

        grasp_set = GraspSet(grasps=all_grasps)
        if verbose:
            print(f"[GraspOptimizer] Produced {len(grasp_set)} grasps "
                  f"(target={num_grasps})")

        return grasp_set

    def _optimize_batch(self) -> List[Grasp]:
        """Run one batch of Simulated Annealing optimization."""
        B = self.batch_size
        device = self.device

        # Initialize hand poses
        initialize_grasp_poses(
            self.hand_model, self.object_model,
            batch_size=B, n_contact=self.n_contact,
            device=device,
        )

        # Create SA optimizer
        optimizer = AnnealingOptimizer(
            self.hand_model,
            step_size=self.step_size,
            starting_temperature=self.starting_temperature,
            temperature_decay=self.temperature_decay,
            device=device,
        )

        # Compute initial energy
        energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(
            self.hand_model, self.object_model,
            self.w_dis, self.w_pen, self.w_spen, self.w_joints,
        )
        energy.sum().backward(retain_graph=True)

        # Main SA loop
        for step in range(1, self.n_iter + 1):
            optimizer.try_step()
            optimizer.zero_grad()

            new_energy, new_fc, new_dis, new_pen, new_spen, new_joints = cal_energy(
                self.hand_model, self.object_model,
                self.w_dis, self.w_pen, self.w_spen, self.w_joints,
            )
            new_energy.sum().backward(retain_graph=True)

            accept, temperature = optimizer.accept_step(energy, new_energy)

            # Update accepted energies
            energy[accept] = new_energy[accept]
            E_fc[accept] = new_fc[accept]
            E_dis[accept] = new_dis[accept]
            E_pen[accept] = new_pen[accept]
            E_spen[accept] = new_spen[accept]
            E_joints[accept] = new_joints[accept]

            if step % 500 == 0 or step == 1:
                n_good = int(((E_fc < self.thres_fc) & (E_dis < self.thres_dis)).sum())
                print(f"    step {step}/{self.n_iter}: "
                      f"E={energy.mean():.2f} fc={E_fc.mean():.4f} "
                      f"dis={E_dis.mean():.5f} pen={E_pen.mean():.4f} "
                      f"T={temperature:.2f} good={n_good}/{B}")

        # Extract successful grasps
        return self._extract_grasps(E_fc, E_dis, E_pen)

    def _extract_grasps(self, E_fc, E_dis, E_pen) -> List[Grasp]:
        """Filter and convert optimized grasps to our Grasp format."""
        from scipy.spatial.transform import Rotation as R_scipy

        # Success criteria (DexGraspNet thresholds)
        success = (
            (E_fc < self.thres_fc)
            & (E_dis < self.thres_dis)
            & (E_pen < self.thres_pen)
        )

        grasps = []
        hand_pose = self.hand_model.hand_pose.detach()
        contact_pts = self.hand_model.contact_points.detach()

        for i in range(hand_pose.shape[0]):
            if not success[i]:
                continue

            # Extract hand pose components
            t = hand_pose[i, :3].cpu().numpy()
            rot6d = hand_pose[i, 3:9]
            R = robust_compute_rotation_matrix_from_ortho6d(
                rot6d.unsqueeze(0)
            )[0].cpu().numpy()
            joint_angles = hand_pose[i, 9:].cpu().numpy()
            tips = contact_pts[i].cpu().numpy()  # (n_contact, 3)

            # Quality: inverse of E_fc (lower E_fc = better force closure)
            quality = float(max(0.0, self.thres_fc - E_fc[i].item()))

            # Surface normals at contact points
            _, _, face_idx = trimesh.proximity.closest_point(self.mesh, tips)
            normals = self.mesh.face_normals[face_idx].astype(np.float32)

            # Object pose in hand frame: R^T @ (0 - t) = -R^T @ t
            obj_pos_hand = (-R.T @ t).astype(np.float32)
            obj_quat_scipy = R_scipy.from_matrix(R.T).as_quat()  # (x,y,z,w)
            obj_quat_hand = np.array([
                obj_quat_scipy[3], obj_quat_scipy[0],
                obj_quat_scipy[1], obj_quat_scipy[2],
            ], dtype=np.float32)

            grasps.append(Grasp(
                fingertip_positions=tips.astype(np.float32),
                contact_normals=normals,
                quality=quality,
                joint_angles=joint_angles.astype(np.float32),
                object_pos_hand=obj_pos_hand,
                object_quat_hand=obj_quat_hand,
                object_pose_frame="root",
            ))

        return grasps
