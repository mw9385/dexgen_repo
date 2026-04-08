"""
DexterityGen Algorithm 3 — Grasp Set Generation + Physics Validation
=====================================================================
Implements the exact grasp generation pipeline from the DexterityGen paper
(Appendix, Algorithm 3):

  1. Sample M candidate contact points & normals on object surface
  2. GraspAnalysis: Evaluate Net Force Optimization (NFO) quality
  3. Random Pose: Sample object pose in hand frame
  4. Assign: Solve IK to find joint angles (q)
  5. Collision: Reject if hand penetrates object

Then validates in Isaac Sim physics:
  6. Physics settle: force joints, check object stays stable

Key difference from previous HeuristicSampler:
  - HeuristicSampler: random JOINTS → FK → check if near surface
  - Algorithm 3: sample ON SURFACE → NFO → IK → collision check
  Algorithm 3 guarantees fingertips are on the surface by construction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh

from .grasp_sampler import Grasp, GraspSet, _resolve_finger_body_ids, _compute_obj_pose_hand
from .net_force_optimization import NetForceOptimizer


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
# Algorithm 3, Step 1-2: Surface Sampling + NFO
# ---------------------------------------------------------------------------

class SurfaceGraspSampler:
    """
    Sample candidate grasps by picking contact points on the object surface
    and evaluating NFO quality.

    Algorithm 3, Steps 1-2:
      1. Sample M sets of num_fingers contact points on mesh surface
      2. For each set, compute surface normals
      3. Evaluate NFO force-closure quality
      4. Keep grasps with quality >= min_quality
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        nfo: NetForceOptimizer,
        num_fingers: int = 5,
        min_quality: float = 0.03,
        min_finger_spacing: float = 0.01,
        seed: int = 42,
    ):
        self.mesh = mesh
        self.nfo = nfo
        self.num_fingers = num_fingers
        self.min_quality = min_quality
        self.min_finger_spacing = min_finger_spacing
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        num_candidates: int = 10000,
        num_grasps: int = 500,
        verbose: bool = True,
    ) -> List[Grasp]:
        """
        Sample contact point sets on the mesh surface and filter by NFO.

        Returns grasps with fingertip_positions (on surface) + contact_normals
        + quality. No joint_angles yet (IK will fill those).
        """
        if verbose:
            print(f"    [SurfaceSampler] Sampling {num_grasps} grasps "
                  f"from {num_candidates} candidates "
                  f"(NFO >= {self.min_quality})")

        valid = []
        nfo_fail = 0
        spacing_fail = 0

        for attempt in range(num_candidates):
            if len(valid) >= num_grasps:
                break

            # Step 1: Sample num_fingers points on surface
            points, face_idx = trimesh.sample.sample_surface(
                self.mesh, self.num_fingers,
            )
            points = points.astype(np.float32)
            normals = self.mesh.face_normals[face_idx].astype(np.float32)

            # Check pairwise finger spacing
            if not self._check_spacing(points):
                spacing_fail += 1
                continue

            # Step 2: NFO quality evaluation
            grasp = Grasp(
                fingertip_positions=points,
                contact_normals=normals,
            )
            quality = self.nfo.evaluate(grasp)

            if quality < self.min_quality:
                nfo_fail += 1
                continue

            grasp.quality = quality
            valid.append(grasp)

            if verbose and len(valid) % 100 == 0:
                print(f"      {len(valid)} valid (attempt {attempt+1}, "
                      f"spacing_fail={spacing_fail}, nfo_fail={nfo_fail})")

        if verbose:
            print(f"    [SurfaceSampler] {len(valid)} grasps from "
                  f"{min(attempt+1, num_candidates)} attempts "
                  f"(spacing_fail={spacing_fail}, nfo_fail={nfo_fail})")

        return valid

    def _check_spacing(self, points: np.ndarray) -> bool:
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(points[i] - points[j]) < self.min_finger_spacing:
                    return False
        return True


# ---------------------------------------------------------------------------
# Algorithm 3, Steps 3-5: Random Pose + IK + Collision
# ---------------------------------------------------------------------------

def assign_ik_and_check_collision(
    env,
    grasps: List[Grasp],
    mesh: trimesh.Trimesh,
    object_size: float,
    penetration_margin: float = 0.008,
    render: bool = False,
    verbose: bool = True,
) -> List[Grasp]:
    """
    Algorithm 3, Steps 3-5:
      3. Object is placed at fingertip centroid (defines object pose)
      4. IK (refine_hand_to_start_grasp) to find joint angles
      5. Collision check: reject if hand penetrates object

    Processes grasps one at a time on env 0 (IK is serial).
    """
    from envs.mdp.sim_utils import (
        set_robot_root_pose,
        set_adaptive_joint_pose,
        get_fingertip_body_ids_from_env,
        refine_hand_to_start_grasp,
    )

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    finger_body_ids = _resolve_finger_body_ids(robot)

    solved = []
    ik_fail = 0
    collision_fail = 0

    if verbose:
        print(f"    [IK+Collision] Processing {len(grasps)} grasps")

    for i, grasp in enumerate(grasps):
        fp = grasp.fingertip_positions  # (F, 3) on object surface
        fp_tensor = torch.tensor(
            fp, device=device, dtype=torch.float32,
        ).unsqueeze(0)  # (1, F, 3)

        # Step 3: Set default wrist + midpoint joints
        root_state = robot.data.default_root_state[env_ids, :7].clone()
        root_state[:, :3] += env.scene.env_origins[env_ids]
        set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])

        set_adaptive_joint_pose(env, env_ids, object_size)
        robot.update(0.0)

        # Place object at fingertip centroid (object frame = world shifted)
        ft_pos = robot.data.body_pos_w[env_ids][:, ft_ids, :]
        obj_pos = ft_pos.mean(dim=1)  # (1, 3)
        obj_quat = torch.tensor(
            [[1, 0, 0, 0]], device=device, dtype=torch.float32,
        )

        obj_state = obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos
        obj_state[:, 3:7] = obj_quat
        obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        obj.update(0.0)

        # Step 4: IK — solve joint angles to reach surface contact points
        refine_hand_to_start_grasp(env, env_ids, fp_tensor)
        robot.update(0.0)

        # Check IK quality: are fingertips actually near target?
        ft_after = robot.data.body_pos_w[0, ft_ids, :]
        ft_obj_after = (ft_after - obj_pos[0]).cpu().numpy()
        _, ik_dists, _ = trimesh.proximity.closest_point(mesh, ft_obj_after)
        ik_mean_err = float(ik_dists.mean())

        if ik_mean_err > 0.03:  # IK failed to converge within 3cm
            ik_fail += 1
            continue

        # Read solved joint angles
        q = robot.data.joint_pos[0].cpu().numpy().copy()

        # Step 5: Collision check — finger links vs object mesh
        fl_world = robot.data.body_pos_w[0, finger_body_ids, :]
        fl_obj = (fl_world - obj_pos[0]).cpu().numpy()
        fl_closest, fl_dists, fl_face = trimesh.proximity.closest_point(
            mesh, fl_obj,
        )
        to_pt = fl_obj - fl_closest
        fl_normals = mesh.face_normals[fl_face]
        sign = np.sum(to_pt * fl_normals, axis=-1)
        penetration = np.where(sign < 0, fl_dists, 0.0)

        if np.any(penetration > penetration_margin):
            collision_fail += 1
            continue

        # Compute object pose in hand frame
        pos_hand, quat_hand = _compute_obj_pose_hand(
            robot, obj_pos[0], device, obj_quat_w=obj_quat[0],
        )

        # Use actual IK-solved fingertip positions (on mesh surface)
        _, _, face_idx_final = trimesh.proximity.closest_point(
            mesh, ft_obj_after,
        )
        normals_final = mesh.face_normals[face_idx_final].astype(np.float32)

        grasp.joint_angles = q
        grasp.object_pos_hand = pos_hand
        grasp.object_quat_hand = quat_hand
        grasp.object_pose_frame = "hand_root"
        grasp.contact_normals = normals_final
        solved.append(grasp)

        if verbose and (i + 1) % 50 == 0:
            print(f"      [{i+1}/{len(grasps)}] solved={len(solved)} "
                  f"(ik_fail={ik_fail}, collision={collision_fail})")

        if verbose and i == 0:
            print(f"      [Debug grasp 0] ik_mean_err={ik_mean_err*1000:.1f}mm, "
                  f"max_pen={float(penetration.max())*1000:.1f}mm, "
                  f"quality={grasp.quality:.4f}")
            print(f"      joint_angles (first 10): {q[:10].tolist()}")

    if verbose:
        print(f"    [IK+Collision] {len(solved)}/{len(grasps)} passed "
              f"(ik_fail={ik_fail}, collision={collision_fail})")

    return solved


# ---------------------------------------------------------------------------
# Step 6: Physics Validation
# ---------------------------------------------------------------------------

class SimGraspValidator:
    """
    Validates grasps via PhysX physics settle.

    Forces joints to stored values each step (write_joint_state_to_sim)
    to keep the hand configuration fixed while PhysX handles object collision.
    """

    def __init__(
        self,
        env,
        settle_steps: int = 40,
        vel_threshold: float = 0.01,
        min_height: float = 0.15,
        max_drift: float = 0.05,
        render: bool = False,
    ):
        self.env = env
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.num_envs = env.num_envs
        self.settle_steps = settle_steps
        self.vel_threshold = vel_threshold
        self.min_height = min_height
        self.max_drift = max_drift
        self.render = render

        from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
        self.ft_ids = get_fingertip_body_ids_from_env(self.robot, env)
        self.all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long,
        )

    def validate(
        self, grasps: List[Grasp], verbose: bool = True,
    ) -> List[Grasp]:
        from envs.mdp.sim_utils import (
            set_robot_joints_direct,
            set_robot_root_pose,
        )

        env = self.env
        robot = self.robot
        obj = self.obj
        valid = []
        reject_counts = {"velocity": 0, "height": 0, "drift": 0}

        grasps = [g for g in grasps if g.joint_angles is not None]

        if verbose:
            print(f"\n    [Validator] {len(grasps)} grasps, "
                  f"settle={self.settle_steps}, vel_thresh={self.vel_threshold}")

        for batch_start in range(0, len(grasps), self.num_envs):
            batch = grasps[batch_start:batch_start + self.num_envs]
            bs = len(batch)
            env_ids = self.all_env_ids[:bs]
            debug = (batch_start == 0 and verbose)

            # 1. Wrist + joints
            root_state = robot.data.default_root_state[env_ids, :7].clone()
            root_state[:, :3] += env.scene.env_origins[env_ids]
            set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])

            set_robot_joints_direct(env, env_ids, [g.joint_angles for g in batch])

            q_batch = torch.stack([
                torch.tensor(g.joint_angles, device=self.device,
                             dtype=torch.float32)
                for g in batch
            ])

            # 2. Object far away → FK → re-write joints → FK
            temp = obj.data.default_root_state[env_ids].clone()
            temp[:, :3] = (
                env.scene.env_origins[env_ids]
                + torch.tensor([[0, 0, -10.0]], device=self.device)
            )
            temp[:, 7:] = 0.0
            obj.write_root_state_to_sim(temp, env_ids=env_ids)
            obj.update(0.0)

            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)
            robot.write_joint_state_to_sim(
                q_batch, torch.zeros_like(q_batch), env_ids=env_ids,
            )
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

            # 3. Place object at fingertip centroid
            ft_pos = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
            obj_pos = ft_pos.mean(dim=1)

            obj_state = obj.data.default_root_state[env_ids].clone()
            obj_state[:, :3] = obj_pos
            obj_state[:, 3:7] = torch.tensor(
                [[1, 0, 0, 0]], device=self.device, dtype=torch.float32,
            ).expand(bs, -1)
            obj_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
            obj.update(0.0)

            # 4. Hold: force joints each step
            for step_i in range(self.settle_steps):
                robot.write_joint_state_to_sim(
                    q_batch, torch.zeros_like(q_batch), env_ids=env_ids,
                )
                env.sim.step(render=self.render)
                env.scene.update(dt=env.physics_dt)

                if debug and step_i in (0, 19, self.settle_steps - 1):
                    obj_v = torch.norm(
                        obj.data.root_lin_vel_w[env_ids[0]],
                    ).item()
                    obj_p = obj.data.root_pos_w[env_ids[0]].tolist()
                    print(f"      step {step_i:3d}: vel={obj_v:.4f} "
                          f"pos={[f'{x:.3f}' for x in obj_p]}")

            # 5. Check stability
            speed = torch.norm(
                obj.data.root_lin_vel_w[env_ids], dim=-1,
            )
            obj_z = obj.data.root_pos_w[env_ids, 2]
            ft_after = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
            centroid_after = ft_after.mean(dim=1)
            obj_drift = torch.norm(
                obj.data.root_pos_w[env_ids, :3] - centroid_after, dim=-1,
            )

            for j in range(bs):
                if speed[j] >= self.vel_threshold:
                    reject_counts["velocity"] += 1
                elif obj_z[j] < self.min_height:
                    reject_counts["height"] += 1
                elif obj_drift[j] > self.max_drift:
                    reject_counts["drift"] += 1
                else:
                    valid.append(batch[j])

            if verbose:
                done = min(batch_start + bs, len(grasps))
                print(f"    [{done}/{len(grasps)}] passed: {len(valid)}")

        if verbose:
            print(f"    [Validator] {len(valid)}/{len(grasps)} passed. "
                  f"Rejects: {reject_counts}")

        return valid


# ---------------------------------------------------------------------------
# Full pipeline: Algorithm 3 + Physics Validation
# ---------------------------------------------------------------------------

def generate_and_validate(
    env,
    object_name: str,
    object_shape: str,
    object_size: float,
    num_grasps: int = 300,
    # Step 1-2: Surface sampling + NFO
    num_candidates: int = 50000,
    num_surface_grasps: int = 1000,
    nfo_min_quality: float = 0.03,
    min_finger_spacing: float = 0.01,
    # Step 3-5: IK + Collision
    penetration_margin: float = 0.008,
    # Step 6: Physics validation
    settle_steps: int = 40,
    vel_threshold: float = 0.01,
    # Misc
    render: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> GraspSet:
    """
    DexterityGen Algorithm 3 + Physics Validation.

    Phase 1 (Steps 1-2): Sample contact points on surface + NFO quality
    Phase 2 (Steps 3-5): IK to find joints + collision check
    Phase 3 (Step 6): Physics settle validation
    """
    mesh = make_primitive_mesh(object_shape, object_size)

    nfo = NetForceOptimizer(
        mu=0.5, num_edges=8, min_quality=nfo_min_quality,
    )

    # ── Phase 1: Surface sampling + NFO (Steps 1-2) ─────────────
    if verbose:
        print(f"\n  Phase 1: Surface sampling + NFO (Algorithm 3, Steps 1-2)")

    surface_sampler = SurfaceGraspSampler(
        mesh=mesh,
        nfo=nfo,
        num_fingers=5,
        min_quality=nfo_min_quality,
        min_finger_spacing=min_finger_spacing,
        seed=seed,
    )
    surface_grasps = surface_sampler.sample(
        num_candidates=num_candidates,
        num_grasps=num_surface_grasps,
        verbose=verbose,
    )

    if len(surface_grasps) == 0:
        print("  WARNING: 0 surface grasps")
        return GraspSet(object_name=object_name)

    if verbose:
        print(f"  → {len(surface_grasps)} NFO-valid surface grasps")

    # ── Phase 2: IK + Collision (Steps 3-5) ──────────────────────
    if verbose:
        print(f"\n  Phase 2: IK + Collision check (Algorithm 3, Steps 3-5)")

    ik_grasps = assign_ik_and_check_collision(
        env=env,
        grasps=surface_grasps,
        mesh=mesh,
        object_size=object_size,
        penetration_margin=penetration_margin,
        render=render,
        verbose=verbose,
    )

    if len(ik_grasps) == 0:
        print("  WARNING: 0 grasps survived IK + collision")
        return GraspSet(object_name=object_name)

    if verbose:
        print(f"  → {len(ik_grasps)} IK-solved grasps")

    # ── Phase 3: Physics validation (Step 6) ─────────────────────
    if verbose:
        print(f"\n  Phase 3: Physics validation (settle test)")

    validator = SimGraspValidator(
        env=env,
        settle_steps=settle_steps,
        vel_threshold=vel_threshold,
        render=render,
    )
    valid = validator.validate(ik_grasps, verbose=verbose)

    valid.sort(key=lambda g: g.quality, reverse=True)
    valid = valid[:num_grasps]

    if verbose:
        print(f"\n  Final: {len(valid)} physics-validated grasps "
              f"(surface={len(surface_grasps)}, IK={len(ik_grasps)}, "
              f"physics={len(valid)})")

    # Set object name
    for g in valid:
        g.object_name = object_name
        g.object_scale = object_size

    return GraspSet(grasps=valid, object_name=object_name)
