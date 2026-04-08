"""
Isaac Sim Batched Grasp Sampler — Close-from-Open Strategy
===========================================================
Generates grasps by simulating active finger closing in Isaac Sim.

Pipeline per round:
  1. (Init, once) Apply palm-up transform so palm faces +Z
  2. Set all envs to OPEN hand (fingers extended)
  3. Place object above palm (at open-hand fingertip centroid)
  4. Set sampled CLOSING targets as joint position targets
  5. Step physics for hold_steps — actuators drive fingers toward
     targets, PhysX handles contact. Fingers stop where they meet
     the object surface.
  6. Validate after holding: velocity, height, contact, NFO

This "close-from-open" approach ensures fingers actually wrap around
the object, unlike random joint sampling where fingers might never
touch the object.
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
    Batched grasp sampler with physics-based holding.

    Each round:
      - N envs get different sampled joint targets
      - Object placed at fingertip centroid
      - Physics runs for hold_steps with position control active
      - After hold: check stability + contact + NFO

    Args:
        env: Live Isaac Lab ManagerBasedRLEnv.
        object_name: Identifier for output grasps.
        object_shape / object_size: Primitive mesh params.
        hold_steps: Physics steps to hold the grasp (actuators active).
        vel_threshold: Max object velocity after holding.
        contact_threshold: Max fingertip-to-surface distance for contact.
        min_contact_fingers: Min fingertips in contact.
        penetration_margin: Max finger link penetration depth.
        nfo_min_quality: Min NFO score (0 = disabled).
        render: Render physics steps.
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
        hold_steps: int = 40,
        vel_threshold: float = 0.3,
        min_height: float = 0.15,
        max_drift: float = 0.05,
        contact_threshold: float = 0.02,
        min_contact_fingers: int = 3,
        penetration_margin: float = 0.008,
        nfo_min_quality: float = 0.0,
        render: bool = False,
        seed: int = 42,
    ):
        self.env = env
        self.object_name = object_name
        self.object_shape = object_shape
        self.object_size = object_size
        self.num_fingers = num_fingers
        self.noise_std = noise_std
        self.hold_steps = hold_steps
        self.vel_threshold = vel_threshold
        self.min_height = min_height
        self.max_drift = max_drift
        self.contact_threshold = contact_threshold
        self.min_contact_fingers = min_contact_fingers
        self.penetration_margin = penetration_margin
        self.nfo_min_quality = nfo_min_quality
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

        # Open hand pre-shape: fingers mostly extended (20% into range)
        self.q_open = self.q_low + 0.20 * self.q_range
        self.q_open[self._WRIST_JOINTS] = 0.0

        # Closing target center: biased toward flexion so fingers
        # actually close around the object
        # Shadow Hand 24-DOF joint indices:
        #   [0,1]=wrist, [2]=FFJ4(passive), [3-5]=FFJ3/2/1,
        #   [6]=MFJ4(passive), [7-9]=MFJ3/2/1,
        #   [10]=RFJ4(passive), [11-13]=RFJ3/2/1,
        #   [14]=LFJ5(passive), [15-18]=LFJ4/3/2/1,
        #   [19-23]=THJ4/3/2/1/0
        _CLOSING_BIASES = {
            # Thumb opposition + flexion (critical for grasping)
            19: 0.25,   # THJ4: opposition
            20: 0.20,   # THJ3: flexion
            21: 0.15,   # THJ2: flexion
            # MCP flexion — curls fingers toward palm
            3: 0.20,    # FFJ3
            7: 0.20,    # MFJ3
            11: 0.20,   # RFJ3
            16: 0.20,   # LFJ3
            # PIP flexion — wraps fingers around object
            4: 0.30,    # FFJ2
            8: 0.30,    # MFJ2
            12: 0.30,   # RFJ2
            17: 0.30,   # LFJ2
            # DIP flexion
            5: 0.15,    # FFJ1
            9: 0.15,    # MFJ1
            13: 0.15,   # RFJ1
            18: 0.15,   # LFJ1
        }
        self.q_close_center = self.q_mid.clone()
        self.q_close_center[self._WRIST_JOINTS] = 0.0
        for jidx, bias in _CLOSING_BIASES.items():
            if jidx < self.num_dof:
                self.q_close_center[jidx] += bias * self.q_range[jidx]
        self.q_close_center = torch.clamp(
            self.q_close_center, self.q_low, self.q_high,
        )

        # Noise scale for closing targets
        self.noise_scale = self.q_range.clone() * self.noise_std
        self.noise_scale[self._WRIST_JOINTS] = 0.0
        # Passive joints: less noise
        for j in [2, 6, 10, 14]:  # passive spread joints
            if j < self.num_dof:
                self.noise_scale[j] *= 0.3

        # Object mesh
        self.mesh = make_primitive_mesh(object_shape, object_size)

        # NFO evaluator (always create for debugging, filter only if min_quality > 0)
        from .net_force_optimization import NetForceOptimizer
        self.nfo = NetForceOptimizer(
            mu=0.5, num_edges=8,
            min_quality=nfo_min_quality if nfo_min_quality > 0 else 0.0,
        )
        self.nfo_min_quality = nfo_min_quality

        # ── Palm-up initialisation ─────────────────────────────────
        # 1. Set midpoint joints → step physics → apply palm-up
        # 2. Store the palm-up wrist pose for all subsequent rounds
        self._palm_up_wrist_pos, self._palm_up_wrist_quat = \
            self._init_palm_up()

    # ------------------------------------------------------------------
    # Init: palm-up wrist pose
    # ------------------------------------------------------------------

    def _init_palm_up(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute palm-up wrist pose using the same approach as
        apply_palm_up_transform() in events.py.

        Steps:
          1. Set default wrist + midpoint joints
          2. Step physics (so body_pos_w is accurate)
          3. Compute palm normal and correction rotation
          4. Rotate wrist around fingertip centroid
        """
        from envs.mdp.sim_utils import (
            set_robot_root_pose,
            get_local_palm_normal,
            get_fingertip_body_ids_from_env,
        )
        from envs.mdp.math_utils import quat_from_two_vectors, quat_multiply
        from isaaclab.utils.math import quat_apply

        env = self.env
        robot = self.robot
        obj = self.obj
        env_ids = self.all_env_ids
        N = self.num_envs

        # 1. Default wrist pose
        root_state = robot.data.default_root_state[env_ids, :7].clone()
        root_state[:, :3] += env.scene.env_origins[env_ids]
        robot.write_root_pose_to_sim(root_state, env_ids=env_ids)

        # Midpoint joints
        q_mid = self.q_mid.unsqueeze(0).expand(N, -1)
        robot.write_joint_state_to_sim(
            q_mid, torch.zeros_like(q_mid), env_ids=env_ids,
        )
        robot.set_joint_position_target(q_mid, env_ids=env_ids)

        # Move object away
        temp = obj.data.default_root_state[env_ids].clone()
        temp[:, :3] = env.scene.env_origins[env_ids] + torch.tensor(
            [[0, 0, -10.0]], device=self.device,
        )
        temp[:, 7:] = 0.0
        obj.write_root_state_to_sim(temp, env_ids=env_ids)
        obj.update(0.0)

        # 2. Step physics so body positions are accurate
        env.sim.step(render=self.render)
        env.scene.update(dt=env.physics_dt)

        # 3. Palm normal → correction quaternion
        wrist_pos = robot.data.root_pos_w[env_ids].clone()
        wrist_quat = robot.data.root_quat_w[env_ids].clone()

        palm_normal_local = get_local_palm_normal(robot, env)
        palm_normal_local = palm_normal_local.unsqueeze(0).expand(N, 3)
        palm_normal_world = quat_apply(wrist_quat, palm_normal_local)

        target_up = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device,
        ).expand(N, 3)
        correction = quat_from_two_vectors(palm_normal_world, target_up)

        # 4. Rotate wrist around fingertip centroid (same as apply_palm_up_transform)
        ft_ids = get_fingertip_body_ids_from_env(robot, env)
        ft_world = robot.data.body_pos_w[env_ids][:, ft_ids, :]
        pivot = ft_world.mean(dim=1)

        new_quat = quat_multiply(correction, wrist_quat)
        new_quat = new_quat / (torch.norm(new_quat, dim=-1, keepdim=True) + 1e-8)
        wrist_rel = wrist_pos - pivot
        new_pos = quat_apply(correction, wrist_rel) + pivot

        # Apply and verify
        set_robot_root_pose(env, env_ids, new_pos, new_quat)
        env.sim.step(render=self.render)
        env.scene.update(dt=env.physics_dt)

        print(f"    [Palm-up] wrist_pos={new_pos[0].tolist()}")
        print(f"    [Palm-up] wrist_quat={new_quat[0].tolist()}")

        # Verify: palm normal should now point +Z
        palm_w_after = quat_apply(
            robot.data.root_quat_w[env_ids],
            palm_normal_local,
        )
        print(f"    [Palm-up] palm_normal_world={palm_w_after[0].tolist()} "
              f"(should be ~[0,0,1])")

        return new_pos.clone(), new_quat.clone()

    # ------------------------------------------------------------------
    # Main sampling interface
    # ------------------------------------------------------------------

    def sample(
        self,
        num_grasps: int = 300,
        max_rounds: int = 500,
        verbose: bool = True,
    ) -> GraspSet:
        all_grasps: List[Grasp] = []
        total_tested = 0
        perturb_pool: List[torch.Tensor] = []

        # Rejection counters for debugging
        reject_counts = {
            "penetration": 0,
            "contact": 0,
            "nfo": 0,
            "velocity": 0,
            "height": 0,
            "drift": 0,
        }

        if verbose:
            print(f"\n[SimGraspSampler] Target: {num_grasps} grasps "
                  f"for '{self.object_name}'")
            print(f"  hold_steps={self.hold_steps}, "
                  f"noise_std={self.noise_std}, "
                  f"contact_thresh={self.contact_threshold}m, "
                  f"min_contact={self.min_contact_fingers}/{self.num_fingers}, "
                  f"nfo_min={self.nfo_min_quality}")

        for round_idx in range(max_rounds):
            if len(all_grasps) >= num_grasps:
                break

            use_perturb = (
                len(perturb_pool) >= 5 and self.rng.random() < 0.3
            )
            if use_perturb:
                q_batch = self._sample_joints_perturb(perturb_pool)
            else:
                q_batch = self._sample_joints()

            new_grasps = self._evaluate_round(
                q_batch, reject_counts, verbose=(round_idx == 0),
            )
            all_grasps.extend(new_grasps)
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

            if verbose and (round_idx + 1) % 5 == 0:
                rate = (len(all_grasps) / total_tested * 100
                        if total_tested > 0 else 0)
                print(f"  round {round_idx+1}: {len(all_grasps)}/{num_grasps} "
                      f"({rate:.1f}%, tested={total_tested})")

        all_grasps.sort(key=lambda g: g.quality, reverse=True)
        all_grasps = all_grasps[:num_grasps]

        if verbose:
            rate = (len(all_grasps) / total_tested * 100
                    if total_tested > 0 else 0)
            print(f"\n[SimGraspSampler] Result: {len(all_grasps)} grasps "
                  f"({rate:.1f}% of {total_tested})")
            print(f"  Rejections: {reject_counts}")

        return GraspSet(grasps=all_grasps, object_name=self.object_name)

    # ------------------------------------------------------------------
    # Joint sampling
    # ------------------------------------------------------------------

    def _sample_joints(self) -> torch.Tensor:
        """Sample closing targets: biased toward flexion + noise."""
        N = self.num_envs
        noise = torch.randn(N, self.num_dof, device=self.device)
        q = self.q_close_center.unsqueeze(0) + noise * self.noise_scale.unsqueeze(0)
        return torch.clamp(q, self.q_low, self.q_high)

    def _sample_joints_perturb(
        self, pool: List[torch.Tensor], perturb_std: float = 0.08,
    ) -> torch.Tensor:
        N = self.num_envs
        idx = self.rng.integers(0, len(pool), size=N)
        seeds = torch.stack([pool[i] for i in idx])
        noise = torch.randn(N, self.num_dof, device=self.device)
        noise[:, self._WRIST_JOINTS] = 0.0
        q = seeds + noise * self.q_range.unsqueeze(0) * perturb_std
        return torch.clamp(q, self.q_low, self.q_high)

    # ------------------------------------------------------------------
    # Per-round evaluation (physics-based)
    # ------------------------------------------------------------------

    def _evaluate_round(
        self,
        closing_targets: torch.Tensor,
        reject_counts: dict,
        verbose: bool = False,
    ) -> List[Grasp]:
        """
        Close-from-open grasp evaluation.

        1. Palm-up wrist + OPEN hand
        2. Place object at open-hand fingertip centroid
        3. Switch to CLOSING targets — actuators drive fingers closed
        4. Hold for hold_steps — fingers wrap around object
        5. Validate
        """
        from envs.mdp.sim_utils import set_robot_root_pose

        N = self.num_envs
        env_ids = self.all_env_ids
        env = self.env
        robot = self.robot
        obj = self.obj

        # ── 1. Palm-up wrist + OPEN hand ─────────────────────────────
        set_robot_root_pose(
            env, env_ids,
            self._palm_up_wrist_pos, self._palm_up_wrist_quat,
        )

        q_open = self.q_open.unsqueeze(0).expand(N, -1)
        robot.write_joint_state_to_sim(
            q_open, torch.zeros_like(q_open), env_ids=env_ids,
        )
        robot.set_joint_position_target(q_open, env_ids=env_ids)

        # Move object away for FK step
        temp = obj.data.default_root_state[env_ids].clone()
        temp[:, :3] = (
            env.scene.env_origins[env_ids]
            + torch.tensor([[0, 0, -10.0]], device=self.device)
        )
        temp[:, 7:] = 0.0
        obj.write_root_state_to_sim(temp, env_ids=env_ids)
        obj.update(0.0)

        # FK step — resolve open hand fingertip positions
        env.sim.step(render=self.render)
        env.scene.update(dt=env.physics_dt)

        # ── 2. Place object at open-hand fingertip centroid ──────────
        ft_open = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
        obj_pos_w = ft_open.mean(dim=1)  # (N, 3)

        obj_state = obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos_w
        obj_state[:, 3:7] = torch.tensor(
            [[1, 0, 0, 0]], device=self.device, dtype=torch.float32,
        ).expand(N, -1)
        obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        obj.update(0.0)

        # ── 3. Switch to closing targets ─────────────────────────────
        # Now fingers will drive toward the closing configuration.
        # PhysX handles contact — fingers stop at the object surface.
        for step in range(self.hold_steps):
            robot.set_joint_position_target(closing_targets, env_ids=env_ids)
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

        # ── 5. Validate ─────────────────────────────────────────────
        # Stability
        speed = torch.norm(obj.data.root_lin_vel_w[env_ids], dim=-1)
        obj_pos_after = obj.data.root_pos_w[env_ids]
        obj_z = obj_pos_after[:, 2]
        ft_after = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
        centroid_after = ft_after.mean(dim=1)
        obj_drift = torch.norm(obj_pos_after[:, :3] - centroid_after, dim=-1)

        # Read actual joint positions (post-contact, may differ from targets)
        actual_q = robot.data.joint_pos[env_ids].clone()

        grasps = []
        for i in range(N):
            # Stability checks
            if speed[i] >= self.vel_threshold:
                reject_counts["velocity"] += 1
                continue
            if obj_z[i] < self.min_height:
                reject_counts["height"] += 1
                continue
            if obj_drift[i] > self.max_drift:
                reject_counts["drift"] += 1
                continue

            grasp = self._validate_and_build(
                i, actual_q[i], ft_after[i],
                obj_pos_after[i], reject_counts,
                verbose=(verbose and i == 0),
            )
            if grasp is not None:
                grasps.append(grasp)

        return grasps

    # ------------------------------------------------------------------
    # Per-env validation + grasp construction
    # ------------------------------------------------------------------

    def _validate_and_build(
        self,
        env_idx: int,
        actual_q: torch.Tensor,
        ft_pos_w: torch.Tensor,
        obj_pos_w: torch.Tensor,
        reject_counts: dict,
        verbose: bool = False,
    ) -> Optional[Grasp]:
        """Validate contact/penetration/NFO and build Grasp."""

        # Fingertip positions in object frame
        ft_obj = (ft_pos_w - obj_pos_w[:3]).cpu().numpy()

        # Closest point on mesh
        closest, dists, face_idx = trimesh.proximity.closest_point(
            self.mesh, ft_obj,
        )
        normals = self.mesh.face_normals[face_idx].astype(np.float32)
        in_contact = dists <= self.contact_threshold
        n_contact = int(in_contact.sum())

        if verbose:
            print(f"\n    [Debug env={env_idx}] Fingertip distances to surface:")
            for fi in range(len(dists)):
                status = "CONTACT" if in_contact[fi] else "far"
                print(f"      finger {fi}: {dists[fi]*1000:.1f}mm ({status})")
            print(f"      → {n_contact}/{self.num_fingers} in contact "
                  f"(need {self.min_contact_fingers})")

        # Contact check
        if n_contact < self.min_contact_fingers:
            reject_counts["contact"] += 1
            return None

        # Penetration check
        fl_world = self.robot.data.body_pos_w[
            env_idx, self.finger_body_ids, :
        ]
        fl_obj = (fl_world - obj_pos_w[:3]).cpu().numpy()
        fl_closest, fl_dists, fl_face = trimesh.proximity.closest_point(
            self.mesh, fl_obj,
        )
        to_pt = fl_obj - fl_closest
        fl_normals = self.mesh.face_normals[fl_face]
        sign = np.sum(to_pt * fl_normals, axis=-1)
        penetration = np.where(sign < 0, fl_dists, 0.0)
        max_pen = float(penetration.max())

        if verbose:
            print(f"    [Debug] Max penetration: {max_pen*1000:.1f}mm "
                  f"(limit {self.penetration_margin*1000:.1f}mm)")

        if max_pen > self.penetration_margin:
            reject_counts["penetration"] += 1
            return None

        # NFO quality
        contact_pts = closest[in_contact].astype(np.float32)
        contact_nrm = normals[in_contact]

        if len(contact_pts) >= 2:
            quality = self.nfo.evaluate(Grasp(
                fingertip_positions=contact_pts,
                contact_normals=contact_nrm,
            ))
        else:
            quality = 0.0

        if verbose:
            print(f"    [Debug NFO] n_contact={n_contact}, quality={quality:.4f}")
            print(f"      contact_pts shape: {contact_pts.shape}")
            for ci in range(len(contact_pts)):
                print(f"      pt[{ci}]={contact_pts[ci].tolist()}, "
                      f"nrm[{ci}]={contact_nrm[ci].tolist()}")
            if self.nfo_min_quality > 0:
                print(f"      threshold={self.nfo_min_quality}, "
                      f"pass={'YES' if quality >= self.nfo_min_quality else 'NO'}")

        if self.nfo_min_quality > 0 and quality < self.nfo_min_quality:
            reject_counts["nfo"] += 1
            return None

        # If NFO is disabled, use contact fraction as quality
        if self.nfo_min_quality <= 0:
            quality = max(quality, n_contact / self.num_fingers)

        # Object pose in hand frame
        obj_quat_w = self.obj.data.root_quat_w[env_idx]
        pos_hand, quat_hand = _compute_obj_pose_hand(
            self.robot, obj_pos_w[:3], self.device, obj_quat_w=obj_quat_w,
        )

        grasp_centroid = ft_obj.mean(axis=0)
        fp_local = (closest - grasp_centroid).astype(np.float32)

        return Grasp(
            fingertip_positions=fp_local,
            contact_normals=normals,
            quality=quality,
            object_name=self.object_name,
            object_scale=self.object_size,
            joint_angles=actual_q.cpu().numpy().copy(),
            object_pos_hand=pos_hand,
            object_quat_hand=quat_hand,
            object_pose_frame="hand_root",
        )
