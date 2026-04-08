"""
Isaac Sim Physics-based Grasp Sampler
=====================================
Generates collision-free grasps by simulating active grasping in Isaac Sim.

Algorithm — active grasp-by-closing:
  1. Orient hand palm-up (+Z) so the object can rest on the palm
  2. Set fingers to an open pre-shape
  3. Place object above the palm center
  4. Set finger joint targets to a sampled *closing* configuration
  5. Step physics for closing_steps — actuators drive fingers toward
     targets, PhysX handles contact/collision naturally.  Fingers stop
     where they meet the object surface.
  6. Step physics for settle_steps — verify the grasp is stable
  7. Validate: object velocity, height, drift, fingertip contact
  8. Extract the *actual* post-contact joint positions as the grasp

This guarantees:
  - FK is from Isaac Sim directly → no MJCF mismatch
  - Collision resolved by PhysX → no analytical SDF error
  - Fingers physically interact with the object → real grasps, not just
    analytically-close configurations

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
# Mesh construction
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
    Batched grasp-by-closing sampler in Isaac Sim.

    Each round:
      - N parallel envs start with palm-up open hand + object on palm
      - Finger targets are set to a sampled closing configuration
      - Physics runs for closing_steps → fingers close around object
      - Settle for settle_steps → check stability
      - Valid grasps are collected with their actual (post-contact) joints

    Args:
        env: Live Isaac Lab ManagerBasedRLEnv instance.
        object_name: Identifier for the generated grasps.
        object_shape: Primitive shape type ("cube", "sphere", "cylinder").
        object_size: Object size in metres.
        num_fingers: Number of fingers (default 5 for Shadow Hand).
        closing_steps: Physics steps for finger closing phase.
        settle_steps: Physics steps for stability verification.
        vel_threshold: Max object velocity after settle (m/s).
        min_height: Min object z after settle (detects dropped objects).
        max_drift: Max object displacement from initial placement.
        contact_threshold: Max fingertip-to-surface distance for "in contact".
        min_contact_fingers: Min fingertips touching the object.
        penetration_margin: Max finger-mesh penetration depth (m).
        noise_std: Joint target noise (fraction of range).
        nfo_min_quality: Min NFO quality score (0 = disabled).
        render: Render each physics step (for visualization).
        seed: Random seed.
    """

    # Shadow Hand 24-DOF: wrist joints (keep fixed)
    _WRIST_JOINTS = [0, 1]
    # Passive spread joints (reduced noise)
    _PASSIVE_JOINTS = [2, 6, 10, 14]

    # Closing target biases — shift sampling center toward flexed/grasping poses.
    # These are applied ON TOP of joint midpoint to produce targets that
    # actively close fingers around objects.
    _CLOSING_TARGET_BIASES = {
        # Thumb opposition (strong)
        19: 0.20,   # THJ4: opposition
        20: 0.15,   # THJ3: flexion
        21: 0.10,   # THJ2: flexion
        # MCP flexion — drives proximal phalanges toward object
        3: 0.15,    # FFJ3
        7: 0.15,    # MFJ3
        11: 0.15,   # RFJ3
        16: 0.15,   # LFJ3
        # PIP flexion — curls fingers around object
        4: 0.25,    # FFJ2
        8: 0.25,    # MFJ2
        12: 0.25,   # RFJ2
        17: 0.25,   # LFJ2
        # DIP flexion
        5: 0.15,    # FFJ1
        9: 0.15,    # MFJ1
        13: 0.15,   # RFJ1
        18: 0.15,   # LFJ1
    }

    def __init__(
        self,
        env,
        object_name: str,
        object_shape: str,
        object_size: float,
        num_fingers: int = 5,
        closing_steps: int = 40,
        settle_steps: int = 10,
        vel_threshold: float = 0.25,
        min_height: float = 0.15,
        max_drift: float = 0.04,
        contact_threshold: float = 0.015,
        min_contact_fingers: int = 3,
        penetration_margin: float = 0.005,
        noise_std: float = 0.25,
        nfo_min_quality: float = 0.0,
        render: bool = False,
        seed: int = 42,
    ):
        self.env = env
        self.object_name = object_name
        self.object_shape = object_shape
        self.object_size = object_size
        self.num_fingers = num_fingers
        self.closing_steps = closing_steps
        self.settle_steps = settle_steps
        self.vel_threshold = vel_threshold
        self.min_height = min_height
        self.max_drift = max_drift
        self.contact_threshold = contact_threshold
        self.min_contact_fingers = min_contact_fingers
        self.penetration_margin = penetration_margin
        self.noise_std = noise_std
        self.nfo_min_quality = nfo_min_quality
        self.render = render
        self.rng = np.random.default_rng(seed)

        # Env references
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.num_envs = env.num_envs

        # Fingertip body IDs
        from envs.mdp.sim_utils import (
            get_fingertip_body_ids_from_env,
            get_palm_body_id_from_env,
        )
        self.ft_ids = get_fingertip_body_ids_from_env(self.robot, env)
        self.palm_body_id = get_palm_body_id_from_env(self.robot, env)

        # All env IDs
        self.all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long,
        )

        # Joint limits from sim
        self.num_dof = self.robot.data.joint_pos.shape[-1]
        self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()
        self.q_range = self.q_high - self.q_low
        self.q_mid = (self.q_low + self.q_high) / 2.0

        # Open hand pre-shape: fingers mostly extended (15% into range)
        self.q_open = self.q_low + 0.15 * self.q_range
        self.q_open[self._WRIST_JOINTS] = 0.0

        # Closing target center: midpoint + flexion biases
        self.q_close_center = self.q_mid.clone()
        self.q_close_center[self._WRIST_JOINTS] = 0.0
        for jidx, bias in self._CLOSING_TARGET_BIASES.items():
            if jidx < self.num_dof:
                self.q_close_center[jidx] += bias * self.q_range[jidx]
        self.q_close_center = torch.clamp(
            self.q_close_center, self.q_low, self.q_high,
        )

        # Noise scale per joint
        self.noise_scale = self.q_range.clone() * self.noise_std
        self.noise_scale[self._WRIST_JOINTS] = 0.0
        for j in self._PASSIVE_JOINTS:
            if j < self.num_dof:
                self.noise_scale[j] *= 0.3

        # Precompute palm-up wrist pose (do a quick FK to resolve palm normal)
        self._palm_up_wrist_pos, self._palm_up_wrist_quat = \
            self._compute_palm_up_pose()

        # Object mesh for contact check and NFO
        self.mesh = make_primitive_mesh(object_shape, object_size)

        # NFO evaluator (optional)
        self.nfo = None
        if nfo_min_quality > 0:
            from .net_force_optimization import NetForceOptimizer
            self.nfo = NetForceOptimizer(
                mu=0.5, num_edges=8, min_quality=nfo_min_quality,
            )

        # Finger link body IDs for penetration check
        self.finger_body_ids = self._resolve_finger_body_ids()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _compute_palm_up_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute wrist pose with palm facing +Z.

        Sets a temporary joint state, resolves the palm normal,
        then computes the corrected wrist orientation.
        Returns (wrist_pos, wrist_quat) both shape (num_envs, *).
        """
        from envs.mdp.sim_utils import align_wrist_palm_up

        env = self.env
        robot = self.robot
        env_ids = self.all_env_ids
        N = self.num_envs

        # Start from default root state
        root_local = robot.data.default_root_state[env_ids, :7].clone()
        wrist_pos = root_local[:, :3] + env.scene.env_origins[env_ids]
        wrist_quat = root_local[:, 3:7]

        # Set default joints so palm normal can be computed from body positions
        default_q = robot.data.default_joint_pos[env_ids].clone()
        robot.write_root_pose_to_sim(
            torch.cat([wrist_pos, wrist_quat], dim=-1), env_ids=env_ids,
        )
        robot.write_joint_state_to_sim(
            default_q, torch.zeros_like(default_q), env_ids=env_ids,
        )
        robot.update(0.0)

        # Align palm upward
        wrist_quat = align_wrist_palm_up(env, env_ids, wrist_quat)

        return wrist_pos, wrist_quat

    def _resolve_finger_body_ids(self) -> list:
        """Get body IDs for all finger links (for penetration check)."""
        _FINGER_LINKS = [
            "robot0_ffknuckle", "robot0_ffproximal", "robot0_ffmiddle",
            "robot0_ffdistal",
            "robot0_mfknuckle", "robot0_mfproximal", "robot0_mfmiddle",
            "robot0_mfdistal",
            "robot0_rfknuckle", "robot0_rfproximal", "robot0_rfmiddle",
            "robot0_rfdistal",
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
        Generate grasps via batched grasp-by-closing simulation.

        Returns:
            GraspSet with validated grasps (24-DOF Isaac format).
        """
        all_grasps: List[Grasp] = []
        total_tested = 0
        perturb_pool: List[torch.Tensor] = []

        if verbose:
            print(f"[SimGraspSampler] Target: {num_grasps} grasps "
                  f"for '{self.object_name}'")
            print(f"  num_envs={self.num_envs}, "
                  f"closing_steps={self.closing_steps}, "
                  f"settle_steps={self.settle_steps}, "
                  f"vel_thresh={self.vel_threshold}, "
                  f"contact_thresh={self.contact_threshold}, "
                  f"min_contact={self.min_contact_fingers}/{self.num_fingers}, "
                  f"render={self.render}")

        for round_idx in range(max_rounds):
            if len(all_grasps) >= num_grasps:
                break

            # Use perturbation (~30%) after collecting some seeds
            use_perturb = (
                len(perturb_pool) >= 5
                and self.rng.random() < 0.3
            )
            if use_perturb:
                q_targets = self._sample_closing_targets_perturb(perturb_pool)
            else:
                q_targets = self._sample_closing_targets()

            new_grasps = self._evaluate_batch(q_targets)
            all_grasps.extend(new_grasps)
            total_tested += self.num_envs

            # Add valid actual joint states to perturbation pool
            for g in new_grasps:
                if g.joint_angles is not None:
                    perturb_pool.append(
                        torch.tensor(g.joint_angles, device=self.device)
                    )
                    if len(perturb_pool) > 500:
                        idx = self.rng.choice(
                            len(perturb_pool), 300, replace=False,
                        )
                        perturb_pool = [perturb_pool[i] for i in idx]

            if verbose and (round_idx + 1) % 5 == 0:
                rate = (len(all_grasps) / total_tested * 100
                        if total_tested > 0 else 0)
                print(f"  round {round_idx+1}/{max_rounds}: "
                      f"{len(all_grasps)}/{num_grasps} grasps "
                      f"({rate:.1f}% acceptance, {total_tested} tested)")

        all_grasps.sort(key=lambda g: g.quality, reverse=True)
        all_grasps = all_grasps[:num_grasps]

        if verbose:
            rate = (len(all_grasps) / total_tested * 100
                    if total_tested > 0 else 0)
            print(f"[SimGraspSampler] Done: {len(all_grasps)} grasps from "
                  f"{total_tested} candidates ({rate:.1f}% acceptance)")

        return GraspSet(grasps=all_grasps, object_name=self.object_name)

    # ------------------------------------------------------------------
    # Joint target sampling
    # ------------------------------------------------------------------

    def _sample_closing_targets(self) -> torch.Tensor:
        """Sample closing target joints (biased toward flexion)."""
        N = self.num_envs
        noise = torch.randn(N, self.num_dof, device=self.device)
        q = (self.q_close_center.unsqueeze(0)
             + noise * self.noise_scale.unsqueeze(0))
        return torch.clamp(q, self.q_low, self.q_high)

    def _sample_closing_targets_perturb(
        self, pool: List[torch.Tensor], perturb_std: float = 0.08,
    ) -> torch.Tensor:
        """Perturb previously valid joint configurations."""
        N = self.num_envs
        idx = self.rng.integers(0, len(pool), size=N)
        seeds = torch.stack([pool[i] for i in idx])
        noise = torch.randn(N, self.num_dof, device=self.device)
        noise[:, self._WRIST_JOINTS] = 0.0
        q = seeds + noise * self.q_range.unsqueeze(0) * perturb_std
        return torch.clamp(q, self.q_low, self.q_high)

    # ------------------------------------------------------------------
    # Batch evaluation — active grasp-by-closing simulation
    # ------------------------------------------------------------------

    def _evaluate_batch(
        self, joint_pos_target: torch.Tensor,
    ) -> List[Grasp]:
        """
        Run one round of grasp-by-closing for all N environments.

        Phase 1: Setup — palm-up wrist, open hand, object on palm
        Phase 2: Close — drive fingers toward targets (closing_steps)
        Phase 3: Settle — hold targets, check stability (settle_steps)
        Phase 4: Extract — validate and build Grasp objects
        """
        N = self.num_envs
        env_ids = self.all_env_ids
        robot = self.robot
        obj = self.obj
        env = self.env

        # ── Phase 1: Setup ──────────────────────────────────────────

        # 1a. Set wrist palm-up
        from envs.mdp.sim_utils import set_robot_root_pose
        set_robot_root_pose(
            env, env_ids,
            self._palm_up_wrist_pos, self._palm_up_wrist_quat,
        )

        # 1b. Open hand (fingers extended)
        q_open = self.q_open.unsqueeze(0).expand(N, -1).clone()
        robot.write_joint_state_to_sim(
            q_open, torch.zeros_like(q_open), env_ids=env_ids,
        )
        robot.set_joint_position_target(q_open, env_ids=env_ids)

        # 1c. Move object far away during FK resolution
        temp_state = obj.data.default_root_state[env_ids].clone()
        temp_state[:, :3] = (
            env.scene.env_origins[env_ids]
            + torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        )
        temp_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(temp_state, env_ids=env_ids)
        obj.update(0.0)

        # 1d. Step physics to resolve FK (palm position now valid)
        env.sim.step(render=self.render)
        env.scene.update(dt=env.physics_dt)

        # 1e. Place object slightly above palm center
        palm_pos_w = robot.data.body_pos_w[
            env_ids, self.palm_body_id, :
        ].clone()  # (N, 3)
        obj_pos_w = palm_pos_w.clone()
        obj_pos_w[:, 2] += 0.02 + self.object_size * 0.3  # above palm

        obj_quat_w = torch.zeros(N, 4, device=self.device)
        obj_quat_w[:, 0] = 1.0  # identity

        obj_state = obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos_w
        obj_state[:, 3:7] = obj_quat_w
        obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        obj.update(0.0)

        # ── Phase 2: Close fingers ──────────────────────────────────
        # Set target to the sampled closing configuration.
        # The position controller drives fingers toward target over time.
        # PhysX resolves contact — fingers stop where they meet the object.
        robot.set_joint_position_target(joint_pos_target, env_ids=env_ids)

        for _ in range(self.closing_steps):
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

        # ── Phase 3: Settle ─────────────────────────────────────────
        # Hold targets steady and check that the object is stable.
        for _ in range(self.settle_steps):
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

        # ── Phase 4: Extract and validate ───────────────────────────

        # Stability check
        obj_vel = obj.data.root_lin_vel_w[env_ids]       # (N, 3)
        speed = torch.norm(obj_vel, dim=-1)               # (N,)
        obj_pos_after = obj.data.root_pos_w[env_ids]      # (N, 3)
        obj_z = obj_pos_after[:, 2]

        ft_pos_after = robot.data.body_pos_w[env_ids][
            :, self.ft_ids, :
        ]  # (N, F, 3)
        centroid_after = ft_pos_after.mean(dim=1)         # (N, 3)
        obj_drift = torch.norm(
            obj_pos_after - centroid_after, dim=-1,
        )

        stable = (
            (speed < self.vel_threshold)
            & (obj_z > self.min_height)
            & (obj_drift < self.max_drift)
        )

        # Build grasps for stable environments
        grasps = []
        stable_indices = torch.where(stable)[0]
        if len(stable_indices) == 0:
            return grasps

        # Actual joint positions after contact (NOT the targets)
        actual_q = robot.data.joint_pos[env_ids].clone()
        # Actual object pose after settle
        obj_pos_final = obj.data.root_pos_w[env_ids].clone()
        obj_quat_final = obj.data.root_quat_w[env_ids].clone()

        for idx_t in stable_indices:
            i = int(idx_t.item())
            grasp = self._build_grasp_single(
                env_idx=i,
                actual_joint_pos=actual_q[i],
                ft_pos_w=ft_pos_after[i],
                obj_pos_w=obj_pos_final[i],
                obj_quat_w=obj_quat_final[i],
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
        actual_joint_pos: torch.Tensor,  # (num_dof,) — post-contact
        ft_pos_w: torch.Tensor,          # (F, 3)
        obj_pos_w: torch.Tensor,         # (3,)
        obj_quat_w: torch.Tensor,        # (4,)
    ) -> Optional[Grasp]:
        """Build and validate a Grasp from the post-interaction state."""
        # Fingertip positions in object frame
        ft_obj = (ft_pos_w - obj_pos_w).cpu().numpy()  # (F, 3)

        # Closest point on mesh surface
        closest, dists, face_idx = trimesh.proximity.closest_point(
            self.mesh, ft_obj,
        )

        # Contact check
        in_contact = dists <= self.contact_threshold
        n_contact = int(in_contact.sum())
        if n_contact < self.min_contact_fingers:
            return None

        # Penetration check on finger links
        if not self._check_finger_penetration(env_idx, obj_pos_w):
            return None

        # Surface normals at closest points
        normals = self.mesh.face_normals[face_idx].astype(np.float32)
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
            quality = n_contact / self.num_fingers

        # Object pose in hand root frame
        pos_hand, quat_hand = self._compute_obj_pose_hand(
            env_idx, obj_pos_w, obj_quat_w,
        )

        return Grasp(
            fingertip_positions=fingertip_positions,
            contact_normals=normals,
            quality=quality,
            object_name=self.object_name,
            object_scale=self.object_size,
            joint_angles=actual_joint_pos.cpu().numpy().copy(),
            object_pos_hand=pos_hand,
            object_quat_hand=quat_hand,
            object_pose_frame="hand_root",
        )

    def _check_finger_penetration(
        self, env_idx: int, obj_pos_w: torch.Tensor,
    ) -> bool:
        """Check that finger links don't deeply penetrate the object."""
        fl_world = self.robot.data.body_pos_w[
            env_idx, self.finger_body_ids, :
        ]
        fl_obj = (fl_world - obj_pos_w).cpu().numpy()

        closest, dists, face_idx = trimesh.proximity.closest_point(
            self.mesh, fl_obj,
        )
        to_pt = fl_obj - closest
        normals = self.mesh.face_normals[face_idx]
        sign = np.sum(to_pt * normals, axis=-1)
        penetration = np.where(sign < 0, dists, 0.0)
        return not np.any(penetration > self.penetration_margin)

    def _compute_obj_pose_hand(
        self,
        env_idx: int,
        obj_pos_w: torch.Tensor,
        obj_quat_w: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute object pose in hand root frame."""
        from isaaclab.utils.math import quat_apply_inverse
        from envs.mdp.math_utils import quat_multiply, quat_conjugate

        rp = self.robot.data.root_pos_w[env_idx]
        rq = self.robot.data.root_quat_w[env_idx]

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
