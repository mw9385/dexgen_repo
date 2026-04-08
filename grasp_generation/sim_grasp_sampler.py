"""
Isaac Sim Batched Grasp Sampler with Physics Validation
========================================================
Batched version of the existing HeuristicSampler + PhysX settle test.

Follows the same approach as HeuristicSampler in grasp_sampler.py:
  - Default wrist pose (palm-up is applied later during training reset)
  - Object placed at fingertip centroid of midpoint joints
  - Joint configs sampled as midpoint ± noise
  - Contact + penetration check via FK + trimesh
  - NFO quality scoring

Additions over HeuristicSampler:
  - Batched across N parallel environments (not just env 0)
  - PhysX physics settle validation as final filter
  - Perturbation-based diversity expansion from valid seeds

The palm-up transform is NOT applied here — that's done at training
time by apply_palm_up_transform() in events.py, consistent with the
existing pipeline.

Usage:
  Called from scripts/run_sim_grasp_generation.py with a live Isaac Lab env.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh

from .grasp_sampler import Grasp, GraspSet, _resolve_finger_body_ids, _compute_obj_pose_hand


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
    Batched HeuristicSampler + PhysX settle validation.

    Two-phase pipeline:
      Phase 1 — FK validation (fast, kinematics only, batched across N envs):
        Same logic as HeuristicSampler._evaluate():
        set joints → robot.update(0.0) → check contact distance + penetration
      Phase 2 — Physics settle (for FK-valid candidates only):
        Set joints → place object → step physics → check velocity/height/drift

    Args:
        env: Live Isaac Lab ManagerBasedRLEnv.
        object_name: Identifier for output grasps.
        object_shape / object_size: Primitive mesh params.
        num_fingers: Number of fingers (5 for Shadow Hand).
        noise_std: Joint noise as fraction of range.
        contact_threshold: Max fingertip-to-surface distance for contact.
        min_contact_fingers: Min fingertips in contact.
        penetration_margin: Max finger link penetration depth.
        nfo_min_quality: Min NFO score (0 = disabled).
        settle_steps: Physics steps for settle validation.
        vel_threshold: Max object velocity after settle.
        render: Render physics steps (for visualization).
        seed: Random seed.
    """

    _WRIST_JOINTS = [0, 1]

    def __init__(
        self,
        env,
        object_name: str,
        object_shape: str,
        object_size: float,
        num_fingers: int = 5,
        noise_std: float = 0.3,
        contact_threshold: float = 0.02,
        min_contact_fingers: int = 3,
        penetration_margin: float = 0.008,
        nfo_min_quality: float = 0.0,
        settle_steps: int = 15,
        vel_threshold: float = 0.3,
        min_height: float = 0.15,
        max_drift: float = 0.05,
        render: bool = False,
        seed: int = 42,
    ):
        self.env = env
        self.object_name = object_name
        self.object_shape = object_shape
        self.object_size = object_size
        self.num_fingers = num_fingers
        self.noise_std = noise_std
        self.contact_threshold = contact_threshold
        self.min_contact_fingers = min_contact_fingers
        self.penetration_margin = penetration_margin
        self.nfo_min_quality = nfo_min_quality
        self.settle_steps = settle_steps
        self.vel_threshold = vel_threshold
        self.min_height = min_height
        self.max_drift = max_drift
        self.render = render
        self.rng = np.random.default_rng(seed)

        # Env references
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.num_envs = env.num_envs

        from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
        self.ft_ids = get_fingertip_body_ids_from_env(self.robot, env)
        self.finger_body_ids = _resolve_finger_body_ids(self.robot)

        self.all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long,
        )

        # Joint limits
        self.num_dof = self.robot.data.joint_pos.shape[-1]
        self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()
        self.q_range = self.q_high - self.q_low
        self.q_mid = (self.q_low + self.q_high) / 2.0
        self.q_mid[self._WRIST_JOINTS] = 0.0

        # Object mesh (for contact/penetration check)
        self.mesh = make_primitive_mesh(object_shape, object_size)

        # NFO evaluator
        self.nfo = None
        if nfo_min_quality > 0:
            from .net_force_optimization import NetForceOptimizer
            self.nfo = NetForceOptimizer(
                mu=0.5, num_edges=8, min_quality=nfo_min_quality,
            )

        # Setup: place object at fingertip centroid of midpoint pose
        # (same approach as HeuristicSampler._setup_object)
        self.obj_pos_w = self._setup_object_at_centroid()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_object_at_centroid(self) -> torch.Tensor:
        """
        Place object at fingertip centroid of midpoint joint configuration.

        Follows HeuristicSampler._setup_object() exactly:
        set midpoint joints → FK → centroid → place object.
        This is done for ALL envs simultaneously.

        Returns:
            obj_pos_w: (num_envs, 3) object position in world frame.
        """
        env_ids = self.all_env_ids
        N = self.num_envs

        # Set midpoint joints in all envs
        q_mid_batch = self.q_mid.unsqueeze(0).expand(N, -1)
        self.robot.write_joint_state_to_sim(
            q_mid_batch, torch.zeros_like(q_mid_batch), env_ids=env_ids,
        )
        self.robot.set_joint_position_target(q_mid_batch, env_ids=env_ids)
        self.robot.update(0.0)

        # Fingertip centroid
        ft_pos = self.robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
        obj_pos_w = ft_pos.mean(dim=1)  # (N, 3)

        # Place object at centroid in all envs
        obj_state = self.obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos_w
        obj_state[:, 3:7] = torch.tensor(
            [[1, 0, 0, 0]], device=self.device, dtype=torch.float32,
        ).expand(N, -1)
        obj_state[:, 7:] = 0.0
        self.obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        self.obj.update(0.0)

        print(f"    Object placed at centroid: {obj_pos_w[0].tolist()}")
        return obj_pos_w

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
        Phase 1: FK validation (batched) → Phase 2: Physics settle.

        Returns GraspSet with physics-validated grasps.
        """
        fk_valid: List[Grasp] = []
        total_tested = 0
        perturb_pool: List[torch.Tensor] = []

        # We need more FK-valid candidates than num_grasps because
        # physics settle will filter some out.
        fk_target = int(num_grasps * 1.5)

        if verbose:
            print(f"[SimGraspSampler] Target: {num_grasps} grasps "
                  f"for '{self.object_name}'")
            print(f"  Phase 1: FK validation (batched, {self.num_envs} envs)")
            print(f"    noise_std={self.noise_std}, "
                  f"contact_thresh={self.contact_threshold}m, "
                  f"min_contact={self.min_contact_fingers}/{self.num_fingers}")

        # ── Phase 1: FK validation ──────────────────────────────────
        for round_idx in range(max_rounds):
            if len(fk_valid) >= fk_target:
                break

            # Perturbation strategy after collecting seeds
            use_perturb = (
                len(perturb_pool) >= 5 and self.rng.random() < 0.3
            )
            if use_perturb:
                q_batch = self._sample_joints_perturb(perturb_pool)
            else:
                q_batch = self._sample_joints()

            new_grasps = self._fk_validate_batch(q_batch)
            fk_valid.extend(new_grasps)
            total_tested += self.num_envs

            for g in new_grasps:
                if g.joint_angles is not None:
                    perturb_pool.append(
                        torch.tensor(g.joint_angles, device=self.device),
                    )
                    if len(perturb_pool) > 500:
                        idx = self.rng.choice(
                            len(perturb_pool), 300, replace=False,
                        )
                        perturb_pool = [perturb_pool[i] for i in idx]

            if verbose and (round_idx + 1) % 10 == 0:
                rate = (len(fk_valid) / total_tested * 100
                        if total_tested > 0 else 0)
                print(f"    round {round_idx+1}: {len(fk_valid)} FK-valid "
                      f"({rate:.1f}%, {total_tested} tested)")

        if verbose:
            print(f"  Phase 1 done: {len(fk_valid)} FK-valid grasps")

        if len(fk_valid) == 0:
            print(f"  WARNING: 0 FK-valid grasps")
            return GraspSet(object_name=self.object_name)

        # ── Phase 2: Physics settle validation ──────────────────────
        if verbose:
            print(f"  Phase 2: Physics settle ({self.settle_steps} steps, "
                  f"vel_thresh={self.vel_threshold})")

        physics_valid = self._physics_validate(fk_valid, verbose=verbose)

        # Sort by quality, truncate
        physics_valid.sort(key=lambda g: g.quality, reverse=True)
        physics_valid = physics_valid[:num_grasps]

        if verbose:
            print(f"[SimGraspSampler] Done: {len(physics_valid)} grasps "
                  f"(FK: {len(fk_valid)}, physics: {len(physics_valid)})")

        return GraspSet(grasps=physics_valid, object_name=self.object_name)

    # ------------------------------------------------------------------
    # Joint sampling
    # ------------------------------------------------------------------

    def _sample_joints(self) -> torch.Tensor:
        """Sample joints: midpoint + noise (same as HeuristicSampler)."""
        N = self.num_envs
        noise = torch.randn(N, self.num_dof, device=self.device) * self.noise_std
        noise[:, self._WRIST_JOINTS] = 0.0
        q = self.q_mid.unsqueeze(0) + noise * self.q_range.unsqueeze(0)
        return torch.clamp(q, self.q_low, self.q_high)

    def _sample_joints_perturb(
        self, pool: List[torch.Tensor], perturb_std: float = 0.08,
    ) -> torch.Tensor:
        """Perturb previously valid joint configs for diversity."""
        N = self.num_envs
        idx = self.rng.integers(0, len(pool), size=N)
        seeds = torch.stack([pool[i] for i in idx])
        noise = torch.randn(N, self.num_dof, device=self.device)
        noise[:, self._WRIST_JOINTS] = 0.0
        q = seeds + noise * self.q_range.unsqueeze(0) * perturb_std
        return torch.clamp(q, self.q_low, self.q_high)

    # ------------------------------------------------------------------
    # Phase 1: FK validation (batched, kinematics only)
    # ------------------------------------------------------------------

    def _fk_validate_batch(
        self, joint_pos: torch.Tensor,
    ) -> List[Grasp]:
        """
        Batched FK validation — same logic as HeuristicSampler._evaluate()
        but for all N envs simultaneously.

        Sets joints → robot.update(0.0) → check contact + penetration.
        No physics stepping needed (kinematics only).
        """
        N = self.num_envs
        env_ids = self.all_env_ids

        # Set joints in all envs (kinematics only, no physics)
        self.robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids,
        )
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.update(0.0)

        # Read fingertip positions for all envs: (N, F, 3)
        ft_pos_w = self.robot.data.body_pos_w[env_ids][:, self.ft_ids, :]

        grasps = []
        for i in range(N):
            grasp = self._fk_validate_single(
                i, joint_pos[i], ft_pos_w[i], self.obj_pos_w[i],
            )
            if grasp is not None:
                grasps.append(grasp)

        return grasps

    def _fk_validate_single(
        self,
        env_idx: int,
        q: torch.Tensor,           # (num_dof,)
        ft_pos_w: torch.Tensor,    # (F, 3)
        obj_pos_w: torch.Tensor,   # (3,)
    ) -> Optional[Grasp]:
        """
        Single-env FK validation.
        Mirrors HeuristicSampler._evaluate() exactly.
        """
        # -- Penetration check (finger links vs object mesh) --
        fl_world = self.robot.data.body_pos_w[env_idx, self.finger_body_ids, :]
        fl_obj = (fl_world - obj_pos_w).cpu().numpy()
        fl_closest, fl_dists, fl_face = trimesh.proximity.closest_point(
            self.mesh, fl_obj,
        )
        to_pt = fl_obj - fl_closest
        fl_normals = self.mesh.face_normals[fl_face]
        sign = np.sum(to_pt * fl_normals, axis=-1)
        penetration = np.where(sign < 0, fl_dists, 0.0)
        if np.any(penetration > self.penetration_margin):
            return None

        # -- Partial contact check --
        ft_obj = (ft_pos_w - obj_pos_w).cpu().numpy()
        closest, dists, face_idx = trimesh.proximity.closest_point(
            self.mesh, ft_obj,
        )
        in_contact = dists <= self.contact_threshold
        n_contact = int(in_contact.sum())
        if n_contact < self.min_contact_fingers:
            return None

        # -- NFO quality --
        normals = self.mesh.face_normals[face_idx].astype(np.float32)
        contact_pts = closest[in_contact].astype(np.float32)
        contact_nrm = normals[in_contact]
        if self.nfo is not None and len(contact_pts) >= 2:
            quality = self.nfo.evaluate(Grasp(
                fingertip_positions=contact_pts,
                contact_normals=contact_nrm,
            ))
            if quality < self.nfo.min_quality:
                return None
        else:
            quality = n_contact / self.num_fingers

        # -- Object pose in hand frame --
        obj_quat_w = self.obj.data.root_quat_w[env_idx]
        pos_hand, quat_hand = _compute_obj_pose_hand(
            self.robot, obj_pos_w, self.device, obj_quat_w=obj_quat_w,
        )

        # Fingertip positions: use closest surface points (same as HeuristicSampler)
        grasp_centroid = ft_obj.mean(axis=0)
        fp_local = (closest - grasp_centroid).astype(np.float32)

        return Grasp(
            fingertip_positions=fp_local,
            contact_normals=normals,
            quality=quality,
            object_name=self.object_name,
            object_scale=self.object_size,
            joint_angles=q.cpu().numpy().copy(),
            object_pos_hand=pos_hand,
            object_quat_hand=quat_hand,
            object_pose_frame="hand_root",
        )

    # ------------------------------------------------------------------
    # Phase 2: Physics settle validation
    # ------------------------------------------------------------------

    def _physics_validate(
        self, grasps: List[Grasp], verbose: bool = True,
    ) -> List[Grasp]:
        """
        Validate FK-valid grasps via PhysX physics settle.

        Same approach as clean_grasp_graph.py:
        set joints → place object → step physics → check velocity.

        Processes grasps in batches of num_envs.
        """
        from envs.mdp.sim_utils import (
            set_robot_joints_direct,
            set_robot_root_pose,
            get_fingertip_body_ids_from_env,
        )

        env = self.env
        robot = self.robot
        obj = self.obj
        valid = []

        # Process in batches of num_envs
        for batch_start in range(0, len(grasps), self.num_envs):
            batch = grasps[batch_start:batch_start + self.num_envs]
            batch_size = len(batch)
            env_ids = self.all_env_ids[:batch_size]

            # 1. Set default wrist pose
            root_state = robot.data.default_root_state[env_ids, :7].clone()
            root_state[:, :3] += env.scene.env_origins[env_ids]
            robot.write_root_pose_to_sim(root_state, env_ids=env_ids)

            # 2. Set joint angles per env
            joint_list = [g.joint_angles for g in batch]
            set_robot_joints_direct(env, env_ids, joint_list)

            # 3. Move object far away for FK step
            temp_state = obj.data.default_root_state[env_ids].clone()
            temp_state[:, :3] = (
                env.scene.env_origins[env_ids]
                + torch.tensor([[0, 0, -10.0]], device=self.device)
            )
            temp_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(temp_state, env_ids=env_ids)
            obj.update(0.0)

            # 4. FK step (physics needed to update body_pos_w properly)
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

            # 5. Place object at fingertip centroid (Isaac FK)
            ft_pos = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
            obj_pos = ft_pos.mean(dim=1)  # (batch_size, 3)

            obj_state = obj.data.default_root_state[env_ids].clone()
            obj_state[:, :3] = obj_pos
            obj_state[:, 3:7] = torch.tensor(
                [[1, 0, 0, 0]], device=self.device, dtype=torch.float32,
            ).expand(batch_size, -1)
            obj_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
            obj.update(0.0)

            # 6. Physics settle
            for _ in range(self.settle_steps):
                env.sim.step(render=self.render)
                env.scene.update(dt=env.physics_dt)

            # 7. Check stability
            speed = torch.norm(
                obj.data.root_lin_vel_w[env_ids], dim=-1,
            )
            obj_z = obj.data.root_pos_w[env_ids, 2]
            ft_after = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
            centroid_after = ft_after.mean(dim=1)
            obj_drift = torch.norm(
                obj.data.root_pos_w[env_ids, :3] - centroid_after, dim=-1,
            )

            for j in range(batch_size):
                if (speed[j] < self.vel_threshold
                        and obj_z[j] > self.min_height
                        and obj_drift[j] < self.max_drift):
                    valid.append(batch[j])

            if verbose and batch_start % (self.num_envs * 5) == 0:
                print(f"    physics: {batch_start+batch_size}/{len(grasps)} "
                      f"tested, {len(valid)} passed")

        return valid
