"""
Isaac Sim Physics-based Grasp Sampler
=====================================
Generates collision-free grasps by sampling directly in Isaac Sim.
Uses PhysX physics settle to validate — guarantees no penetration.

Key advantages over DexGraspNet optimization:
  - FK computed by Isaac Sim directly → no MJCF-Isaac FK mismatch
  - Collision detection by PhysX → no analytical SDF approximation error
  - Generated grasps are physics-validated by construction

Algorithm:
  1. Sample random joint configurations (pre-shape centered + noise)
  2. Set joints in N parallel environments (object moved away)
  3. Step physics to compute FK
  4. Place objects at fingertip centroids
  5. Run physics settle (multiple steps)
  6. Filter by stability:
     a. Object velocity < threshold (no ejection from penetration)
     b. Object height > minimum (didn't fall)
     c. Object drift from centroid < maximum
  7. Filter by contact:
     a. At least min_contact_fingers fingertips near object surface
     b. No deep finger-object penetration
  8. Optionally score by NFO quality metric
  9. Collect valid grasps and repeat until target count reached

Usage:
  Called from scripts/run_sim_grasp_generation.py with a live Isaac Lab env.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh

from .grasp_sampler import Grasp, GraspSet


# ---------------------------------------------------------------------------
# Mesh construction (shared with run_grasp_optimization.py)
# ---------------------------------------------------------------------------

def make_primitive_mesh(shape: str, size: float) -> trimesh.Trimesh:
    if shape == "cube":
        return trimesh.creation.box(extents=[size, size, size])
    elif shape == "sphere":
        return trimesh.creation.icosphere(radius=size / 2.0, subdivisions=3)
    elif shape == "cylinder":
        return trimesh.creation.cylinder(radius=size / 2.0, height=size)
    else:
        raise ValueError(f"Unknown shape: {shape}")


# ---------------------------------------------------------------------------
# SimGraspSampler
# ---------------------------------------------------------------------------

class SimGraspSampler:
    """
    Batched grasp sampler running entirely inside Isaac Sim.

    Uses N parallel environments to sample N joint configurations per round,
    validates via PhysX physics settle, and collects stable grasps.

    Args:
        env: Live Isaac Lab ManagerBasedRLEnv instance.
        object_name: Name for the generated grasps (e.g. "cube_060_f5").
        object_shape: Primitive shape type ("cube", "sphere", "cylinder").
        object_size: Object size in metres.
        num_fingers: Number of fingers (default 5 for Shadow Hand).
        settle_steps: Physics steps for settle test.
        vel_threshold: Max object linear velocity after settle (m/s).
        min_height: Min object z-position after settle.
        max_drift: Max object drift from fingertip centroid after settle.
        contact_threshold: Max fingertip-to-surface distance for contact (m).
        min_contact_fingers: Minimum number of fingertips in contact.
        penetration_margin: Max allowed finger-mesh penetration depth (m).
        noise_std: Joint noise standard deviation (fraction of range).
        nfo_min_quality: Minimum NFO quality score (0 = disabled).
        seed: Random seed.
    """

    # Shadow Hand 24-DOF: indices for wrist joints (keep fixed)
    _WRIST_JOINTS = [0, 1]
    # Passive spread joints that should have minimal noise
    _PASSIVE_JOINTS = [2, 6, 10, 14]

    # Pre-shape biases for grasping configurations.
    # These shift the sampling center from the joint midpoint toward
    # configurations more likely to produce valid grasps.
    # Format: (joint_index, bias_fraction_of_range)
    # Positive = toward upper limit (more flexion for finger joints)
    _GRASP_PRESHAPE_BIASES = {
        # Thumb opposition
        19: 0.15,   # THJ4: more opposition
        20: 0.10,   # THJ3: slight flexion
        # Finger flexion (MCP joints)
        3: 0.10,    # FFJ3
        7: 0.10,    # MFJ3
        11: 0.10,   # RFJ3
        16: 0.10,   # LFJ3
        # PIP joints
        4: 0.15,    # FFJ2
        8: 0.15,    # MFJ2
        12: 0.15,   # RFJ2
        17: 0.15,   # LFJ2
    }

    def __init__(
        self,
        env,
        object_name: str,
        object_shape: str,
        object_size: float,
        num_fingers: int = 5,
        settle_steps: int = 8,
        vel_threshold: float = 0.25,
        min_height: float = 0.15,
        max_drift: float = 0.04,
        contact_threshold: float = 0.015,
        min_contact_fingers: int = 3,
        penetration_margin: float = 0.005,
        noise_std: float = 0.25,
        nfo_min_quality: float = 0.0,
        seed: int = 42,
    ):
        self.env = env
        self.object_name = object_name
        self.object_shape = object_shape
        self.object_size = object_size
        self.num_fingers = num_fingers
        self.settle_steps = settle_steps
        self.vel_threshold = vel_threshold
        self.min_height = min_height
        self.max_drift = max_drift
        self.contact_threshold = contact_threshold
        self.min_contact_fingers = min_contact_fingers
        self.penetration_margin = penetration_margin
        self.noise_std = noise_std
        self.nfo_min_quality = nfo_min_quality
        self.rng = np.random.default_rng(seed)

        # Env references
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.num_envs = env.num_envs

        # Fingertip body IDs
        from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
        self.ft_ids = get_fingertip_body_ids_from_env(self.robot, env)

        # All env IDs
        self.all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long,
        )

        # Joint limits from sim
        self.num_dof = self.robot.data.joint_pos.shape[-1]
        self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()
        self.q_range = self.q_high - self.q_low

        # Build sampling center: midpoint + grasp biases
        self.q_center = (self.q_low + self.q_high) / 2.0
        self.q_center[self._WRIST_JOINTS] = 0.0
        for jidx, bias in self._GRASP_PRESHAPE_BIASES.items():
            if jidx < self.num_dof:
                self.q_center[jidx] += bias * self.q_range[jidx]
        self.q_center = torch.clamp(self.q_center, self.q_low, self.q_high)

        # Noise scale per joint
        self.noise_scale = self.q_range.clone() * self.noise_std
        self.noise_scale[self._WRIST_JOINTS] = 0.0
        # Passive joints: reduced noise
        for j in self._PASSIVE_JOINTS:
            if j < self.num_dof:
                self.noise_scale[j] *= 0.3

        # Object mesh for contact check and NFO
        self.mesh = make_primitive_mesh(object_shape, object_size)

        # NFO evaluator (optional)
        self.nfo = None
        if nfo_min_quality > 0:
            from .net_force_optimization import NetForceOptimizer
            self.nfo = NetForceOptimizer(
                mu=0.5, num_edges=8, min_quality=nfo_min_quality,
            )

        # Finger link body IDs for penetration check (all finger links, not just tips)
        self.finger_body_ids = self._resolve_finger_body_ids()

    def _resolve_finger_body_ids(self) -> list:
        """Get body IDs for all finger links (for penetration check)."""
        _FINGER_LINKS = [
            "robot0_ffknuckle", "robot0_ffproximal", "robot0_ffmiddle", "robot0_ffdistal",
            "robot0_mfknuckle", "robot0_mfproximal", "robot0_mfmiddle", "robot0_mfdistal",
            "robot0_rfknuckle", "robot0_rfproximal", "robot0_rfmiddle", "robot0_rfdistal",
            "robot0_lfmetacarpal", "robot0_lfknuckle", "robot0_lfproximal",
            "robot0_lfmiddle", "robot0_lfdistal",
            "robot0_thbase", "robot0_thproximal", "robot0_thhub",
            "robot0_thmiddle", "robot0_thdistal",
        ]
        ids = []
        for name in _FINGER_LINKS:
            try:
                found = self.robot.find_bodies(name)[0]
                if len(found) > 0:
                    ids.append(int(found[0]))
            except Exception:
                pass
        return ids

    # ------------------------------------------------------------------
    # Main sampling interface
    # ------------------------------------------------------------------

    def sample(
        self,
        num_grasps: int = 300,
        max_rounds: int = 500,
        verbose: bool = True,
    ) -> GraspSet:
        """
        Generate grasps via batched simulation sampling.

        Args:
            num_grasps: Target number of grasps to collect.
            max_rounds: Maximum sampling rounds (each tests num_envs candidates).
            verbose: Print progress.

        Returns:
            GraspSet with validated grasps (24-DOF Isaac format).
        """
        all_grasps: List[Grasp] = []
        total_tested = 0
        perturb_pool: List[torch.Tensor] = []  # valid joint configs for perturbation

        if verbose:
            print(f"[SimGraspSampler] Target: {num_grasps} grasps for '{self.object_name}'")
            print(f"  num_envs={self.num_envs}, settle_steps={self.settle_steps}, "
                  f"vel_thresh={self.vel_threshold}, contact_thresh={self.contact_threshold}, "
                  f"min_contact={self.min_contact_fingers}/{self.num_fingers}")

        for round_idx in range(max_rounds):
            if len(all_grasps) >= num_grasps:
                break

            # Use perturbation strategy for ~30% of rounds after collecting seeds
            use_perturb = (
                len(perturb_pool) >= 5
                and self.rng.random() < 0.3
            )

            if use_perturb:
                q_batch = self._sample_joints_perturb(perturb_pool)
            else:
                q_batch = self._sample_joints_preshape()

            new_grasps = self._evaluate_batch(q_batch)
            all_grasps.extend(new_grasps)
            total_tested += self.num_envs

            # Add valid joint configs to perturbation pool
            for g in new_grasps:
                if g.joint_angles is not None:
                    perturb_pool.append(
                        torch.tensor(g.joint_angles, device=self.device)
                    )
                    # Cap pool size
                    if len(perturb_pool) > 500:
                        # Keep random subset
                        idx = self.rng.choice(len(perturb_pool), 300, replace=False)
                        perturb_pool = [perturb_pool[i] for i in idx]

            if verbose and (round_idx + 1) % 10 == 0:
                rate = len(all_grasps) / total_tested * 100 if total_tested > 0 else 0
                print(f"  round {round_idx+1}/{max_rounds}: "
                      f"{len(all_grasps)}/{num_grasps} grasps "
                      f"({rate:.1f}% acceptance, {total_tested} tested)")

        # Sort by quality (descending) and truncate
        all_grasps.sort(key=lambda g: g.quality, reverse=True)
        all_grasps = all_grasps[:num_grasps]

        if verbose:
            rate = len(all_grasps) / total_tested * 100 if total_tested > 0 else 0
            print(f"[SimGraspSampler] Done: {len(all_grasps)} grasps from "
                  f"{total_tested} candidates ({rate:.1f}% acceptance)")

        return GraspSet(grasps=all_grasps, object_name=self.object_name)

    # ------------------------------------------------------------------
    # Joint sampling strategies
    # ------------------------------------------------------------------

    def _sample_joints_preshape(self) -> torch.Tensor:
        """Sample joints around grasp-biased midpoint with noise."""
        N = self.num_envs
        noise = torch.randn(N, self.num_dof, device=self.device)
        q = self.q_center.unsqueeze(0) + noise * self.noise_scale.unsqueeze(0)
        return torch.clamp(q, self.q_low, self.q_high)

    def _sample_joints_perturb(
        self, pool: List[torch.Tensor], perturb_std: float = 0.08,
    ) -> torch.Tensor:
        """Sample by perturbing previously valid joint configurations."""
        N = self.num_envs
        # Pick random seeds from pool
        idx = self.rng.integers(0, len(pool), size=N)
        seeds = torch.stack([pool[i] for i in idx])  # (N, num_dof)
        noise = torch.randn(N, self.num_dof, device=self.device)
        noise[:, self._WRIST_JOINTS] = 0.0
        q = seeds + noise * self.q_range.unsqueeze(0) * perturb_std
        return torch.clamp(q, self.q_low, self.q_high)

    # ------------------------------------------------------------------
    # Batch evaluation (core simulation loop)
    # ------------------------------------------------------------------

    def _evaluate_batch(self, joint_pos: torch.Tensor) -> List[Grasp]:
        """
        Evaluate a batch of joint configurations via physics simulation.

        Args:
            joint_pos: (num_envs, num_dof) joint positions.

        Returns:
            List of valid Grasp objects.
        """
        N = self.num_envs
        env_ids = self.all_env_ids
        robot = self.robot
        obj = self.obj
        env = self.env

        # --- 1. Set wrist at default pose ---
        root_state = robot.data.default_root_state[env_ids, :7].clone()
        root_state[:, :3] += env.scene.env_origins[env_ids]
        robot.write_root_pose_to_sim(root_state, env_ids=env_ids)

        # --- 2. Set joint angles (object far away to prevent collision) ---
        temp_obj_state = obj.data.default_root_state[env_ids].clone()
        temp_obj_state[:, :3] = (
            env.scene.env_origins[env_ids]
            + torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        )
        temp_obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(temp_obj_state, env_ids=env_ids)
        obj.update(0.0)

        robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids,
        )
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # --- 3. Step physics for FK computation ---
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        # --- 4. Read fingertip positions and place objects ---
        ft_pos_w = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]  # (N, F, 3)
        obj_pos_w = ft_pos_w.mean(dim=1)  # (N, 3) — fingertip centroid

        # Object quaternion: identity (aligned with world)
        obj_quat_w = torch.zeros(N, 4, device=self.device)
        obj_quat_w[:, 0] = 1.0

        obj_state = obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos_w
        obj_state[:, 3:7] = obj_quat_w
        obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        obj.update(0.0)

        # --- 5. Physics settle ---
        for _ in range(self.settle_steps):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

        # --- 6. Stability check ---
        obj_vel = obj.data.root_lin_vel_w[env_ids]  # (N, 3)
        speed = torch.norm(obj_vel, dim=-1)          # (N,)
        obj_pos_after = obj.data.root_pos_w[env_ids]  # (N, 3)
        obj_z = obj_pos_after[:, 2]

        # Fingertip positions after settle
        ft_pos_after = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
        centroid_after = ft_pos_after.mean(dim=1)
        obj_drift = torch.norm(obj_pos_after - centroid_after, dim=-1)

        stable = (
            (speed < self.vel_threshold)
            & (obj_z > self.min_height)
            & (obj_drift < self.max_drift)
        )

        # --- 7. Build grasps for stable environments ---
        grasps = []
        stable_indices = torch.where(stable)[0]

        if len(stable_indices) == 0:
            return grasps

        # Batch fingertip contact and penetration check on CPU
        # Use the pre-settle fingertip positions (before physics moved things)
        for idx_t in stable_indices:
            i = int(idx_t.item())
            grasp = self._build_grasp_single(
                env_idx=i,
                joint_pos=joint_pos[i],
                ft_pos_w=ft_pos_w[i],        # (F, 3) pre-settle
                obj_pos_w=obj_pos_w[i],       # (3,) pre-settle
                obj_quat_w=obj_quat_w[i],     # (4,)
            )
            if grasp is not None:
                grasps.append(grasp)

        return grasps

    # ------------------------------------------------------------------
    # Single grasp construction + validation
    # ------------------------------------------------------------------

    def _build_grasp_single(
        self,
        env_idx: int,
        joint_pos: torch.Tensor,     # (num_dof,)
        ft_pos_w: torch.Tensor,      # (F, 3)
        obj_pos_w: torch.Tensor,     # (3,)
        obj_quat_w: torch.Tensor,    # (4,)
    ) -> Optional[Grasp]:
        """
        Build a Grasp object from a validated env state.

        Checks fingertip contact distance and optional NFO quality.
        """
        # Transform fingertip positions to object frame
        # Object at (obj_pos_w) with identity quaternion → object frame = world shifted
        ft_obj = (ft_pos_w - obj_pos_w).cpu().numpy()  # (F, 3) in object frame

        # Closest point on mesh surface
        closest, dists, face_idx = trimesh.proximity.closest_point(
            self.mesh, ft_obj,
        )

        # Contact check: at least min_contact_fingers within threshold
        in_contact = dists <= self.contact_threshold
        n_contact = int(in_contact.sum())
        if n_contact < self.min_contact_fingers:
            return None

        # Penetration check on finger links
        if not self._check_finger_penetration(env_idx, obj_pos_w):
            return None

        # Surface normals at closest points
        normals = self.mesh.face_normals[face_idx].astype(np.float32)

        # Fingertip positions in object frame (relative to centroid)
        # Use closest points on surface for contacting fingers,
        # actual FK positions for non-contacting fingers
        fingertip_positions = closest.astype(np.float32)

        # NFO quality
        quality = 0.0
        if self.nfo is not None and n_contact >= 2:
            contact_pts = closest[in_contact].astype(np.float32)
            contact_nrm = normals[in_contact]
            quality = self.nfo.evaluate(Grasp(
                fingertip_positions=contact_pts,
                contact_normals=contact_nrm,
            ))
            if quality < self.nfo.min_quality:
                return None
        elif n_contact >= 3:
            # Simple quality: fraction of fingers in contact
            quality = n_contact / self.num_fingers

        # Compute object pose in hand root frame
        pos_hand, quat_hand = self._compute_obj_pose_hand(
            env_idx, obj_pos_w, obj_quat_w,
        )

        return Grasp(
            fingertip_positions=fingertip_positions,
            contact_normals=normals,
            quality=quality,
            object_name=self.object_name,
            object_scale=self.object_size,
            joint_angles=joint_pos.cpu().numpy().copy(),
            object_pos_hand=pos_hand,
            object_quat_hand=quat_hand,
            object_pose_frame="hand_root",
        )

    def _check_finger_penetration(
        self, env_idx: int, obj_pos_w: torch.Tensor,
    ) -> bool:
        """Check that finger links don't deeply penetrate the object mesh."""
        fl_world = self.robot.data.body_pos_w[env_idx, self.finger_body_ids, :]
        fl_obj = (fl_world - obj_pos_w).cpu().numpy()  # (num_links, 3)

        closest, dists, face_idx = trimesh.proximity.closest_point(
            self.mesh, fl_obj,
        )
        # Check penetration direction via surface normals
        to_pt = fl_obj - closest
        normals = self.mesh.face_normals[face_idx]
        sign = np.sum(to_pt * normals, axis=-1)
        # Negative sign = inside mesh = penetration
        penetration = np.where(sign < 0, dists, 0.0)

        return not np.any(penetration > self.penetration_margin)

    def _compute_obj_pose_hand(
        self,
        env_idx: int,
        obj_pos_w: torch.Tensor,  # (3,)
        obj_quat_w: torch.Tensor,  # (4,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute object pose in hand root frame."""
        from isaaclab.utils.math import quat_apply_inverse
        from envs.mdp.math_utils import quat_multiply, quat_conjugate

        rp = self.robot.data.root_pos_w[env_idx]    # (3,)
        rq = self.robot.data.root_quat_w[env_idx]   # (4,)

        rel = obj_pos_w - rp
        pos_hand = quat_apply_inverse(
            rq.unsqueeze(0), rel.unsqueeze(0),
        )[0]

        quat_hand = quat_multiply(
            quat_conjugate(rq.unsqueeze(0)),
            obj_quat_w.unsqueeze(0),
        )[0]

        return (
            pos_hand.cpu().numpy().copy().astype(np.float32),
            quat_hand.cpu().numpy().copy().astype(np.float32),
        )
