"""
Grasp Generation Pipeline: Seed → Surface RRT → IK → Physics Validation
========================================================================
1. HeuristicSampler: FK-based seed generation (contact + NFO quality)
2. Surface RRT Expansion: perturb fingertip positions on object surface,
   check finger spacing + NFO quality → build grasp graph
3. IK (refine_hand_to_start_grasp): solve joint angles for each grasp
4. Physics Validation: Isaac Sim PhysX settle test

Uses config parameters from configs/grasp_generation.yaml [surface_rrt].
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import trimesh

from .grasp_sampler import Grasp, GraspSet, HeuristicSampler
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
# Surface RRT Expansion
# ---------------------------------------------------------------------------

class SurfaceRRTExpander:
    """
    Expand seed grasps by perturbing fingertip positions on object surface.

    Algorithm:
      1. Start with seed grasps (fingertip_positions on mesh surface)
      2. Pick a random seed
      3. Add Gaussian noise (±delta_pos) to each fingertip
      4. Project back to mesh surface (closest point)
      5. Check: min pairwise finger spacing, NFO quality
      6. If valid, add to graph with edge to parent
      7. Repeat until target_size reached

    Config keys (from grasp_generation.yaml [surface_rrt]):
      delta_pos, delta_max, min_quality, max_attempts_per_step,
      collision_threshold, min_finger_spacing
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        nfo: NetForceOptimizer,
        delta_pos: float = 0.008,
        delta_max: float = 0.04,
        min_quality: float = 0.03,
        max_attempts_per_step: int = 30,
        min_finger_spacing: float = 0.01,
        num_fingers: int = 5,
        seed: int = 42,
    ):
        self.mesh = mesh
        self.nfo = nfo
        self.delta_pos = delta_pos
        self.delta_max = delta_max
        self.min_quality = min_quality
        self.max_attempts_per_step = max_attempts_per_step
        self.min_finger_spacing = min_finger_spacing
        self.num_fingers = num_fingers
        self.rng = np.random.default_rng(seed)

    def expand(
        self,
        seeds: List[Grasp],
        target_size: int = 300,
        verbose: bool = True,
    ) -> List[Grasp]:
        """
        Expand seeds via surface-projected RRT.

        Returns list of grasps with fingertip_positions + contact_normals
        but WITHOUT joint_angles (those come from IK later).
        """
        if len(seeds) == 0:
            return []

        graph = list(seeds)  # start with seeds
        fps_array = np.stack([g.fingertip_positions for g in graph])

        stall_count = 0
        max_stall = 200  # give up if no progress for 200 rounds

        if verbose:
            print(f"    [RRT] Seeds: {len(seeds)}, target: {target_size}, "
                  f"delta_pos={self.delta_pos}, min_spacing={self.min_finger_spacing}")

        while len(graph) < target_size and stall_count < max_stall:
            added_this_round = False

            for _ in range(self.max_attempts_per_step):
                # Pick random parent
                parent_idx = self.rng.integers(0, len(graph))
                parent_fp = graph[parent_idx].fingertip_positions.copy()

                # Perturb each fingertip
                noise = self.rng.normal(0, self.delta_pos, parent_fp.shape)
                new_fp = parent_fp + noise.astype(np.float32)

                # Project to surface
                closest, dists, face_idx = trimesh.proximity.closest_point(
                    self.mesh, new_fp,
                )
                new_fp = closest.astype(np.float32)
                normals = self.mesh.face_normals[face_idx].astype(np.float32)

                # Check pairwise finger spacing
                if not self._check_spacing(new_fp):
                    continue

                # Check NFO quality
                quality = self.nfo.evaluate(Grasp(
                    fingertip_positions=new_fp,
                    contact_normals=normals,
                ))
                if quality < self.min_quality:
                    continue

                # Check distance to parent (for graph edge)
                dist_to_parent = float(np.linalg.norm(
                    new_fp.flatten() - parent_fp.flatten(),
                ))
                if dist_to_parent > self.delta_max * self.num_fingers:
                    continue

                # Valid expansion
                new_grasp = Grasp(
                    fingertip_positions=new_fp,
                    contact_normals=normals,
                    quality=quality,
                    joint_angles=None,  # IK will fill this later
                )
                graph.append(new_grasp)
                fps_array = np.vstack([fps_array, new_fp[np.newaxis]])
                added_this_round = True
                break

            if added_this_round:
                stall_count = 0
            else:
                stall_count += 1

            if verbose and len(graph) % 50 == 0:
                print(f"    [RRT] {len(graph)}/{target_size} grasps")

        if verbose:
            print(f"    [RRT] Final: {len(graph)} grasps "
                  f"(seeds={len(seeds)}, expanded={len(graph)-len(seeds)})")

        return graph

    def _check_spacing(self, fp: np.ndarray) -> bool:
        """Check min pairwise distance between all fingertips."""
        n = len(fp)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(fp[i] - fp[j]) < self.min_finger_spacing:
                    return False
        return True


# ---------------------------------------------------------------------------
# IK Solver (uses refine_hand_to_start_grasp from sim_utils)
# ---------------------------------------------------------------------------

def solve_ik_for_grasps(
    env,
    grasps: List[Grasp],
    render: bool = False,
    verbose: bool = True,
) -> List[Grasp]:
    """
    Solve IK for grasps that have fingertip_positions but no joint_angles.

    Uses refine_hand_to_start_grasp (per-finger differential IK).
    Processes one grasp at a time on env 0.

    Grasps that already have joint_angles are passed through unchanged.
    """
    from envs.mdp.sim_utils import (
        set_robot_root_pose,
        set_adaptive_joint_pose,
        get_fingertip_body_ids_from_env,
        refine_hand_to_start_grasp,
    )
    from .grasp_sampler import _compute_obj_pose_hand

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)

    solved = []
    skipped = 0

    for i, grasp in enumerate(grasps):
        # Already has joint angles → pass through
        if grasp.joint_angles is not None:
            solved.append(grasp)
            continue

        fp = grasp.fingertip_positions  # (F, 3) in object frame
        fp_tensor = torch.tensor(
            fp, device=device, dtype=torch.float32,
        ).unsqueeze(0)  # (1, F, 3)

        # 1. Default wrist pose
        root_state = robot.data.default_root_state[env_ids, :7].clone()
        root_state[:, :3] += env.scene.env_origins[env_ids]
        set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])

        # 2. Midpoint joints as IK starting point
        set_adaptive_joint_pose(env, env_ids, 0.06)
        robot.update(0.0)

        # 3. Place object at fingertip centroid
        ft_pos = robot.data.body_pos_w[env_ids][:, ft_ids, :]
        obj_pos = ft_pos.mean(dim=1)
        obj_quat = torch.tensor(
            [[1, 0, 0, 0]], device=device, dtype=torch.float32,
        )

        obj_state = obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos
        obj_state[:, 3:7] = obj_quat
        obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        obj.update(0.0)

        # 4. IK refinement
        refine_hand_to_start_grasp(env, env_ids, fp_tensor)
        robot.update(0.0)

        # 5. Read solved joint angles
        q = robot.data.joint_pos[0].cpu().numpy().copy()

        # 6. Compute object pose in hand frame
        pos_hand, quat_hand = _compute_obj_pose_hand(
            robot, obj_pos[0], device, obj_quat_w=obj_quat[0],
        )

        grasp.joint_angles = q
        grasp.object_pos_hand = pos_hand
        grasp.object_quat_hand = quat_hand
        grasp.object_pose_frame = "hand_root"
        solved.append(grasp)

        if verbose and (i + 1) % 50 == 0:
            print(f"    [IK] {i+1}/{len(grasps)} solved")

    if verbose:
        print(f"    [IK] Done: {len(solved)} solved "
              f"({skipped} skipped, had joint_angles)")

    return solved


# ---------------------------------------------------------------------------
# Physics Validator
# ---------------------------------------------------------------------------

class SimGraspValidator:
    """
    Validates grasps via PhysX physics settle.

    Forces joints to stored values each step (write_joint_state_to_sim)
    to prevent actuator drift, while PhysX handles object collision.
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
        reject_counts = {"velocity": 0, "height": 0, "drift": 0, "no_joints": 0}

        # Filter out grasps without joint angles
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

            joint_list = [g.joint_angles for g in batch]
            set_robot_joints_direct(env, env_ids, joint_list)

            q_batch = torch.stack([
                torch.tensor(g.joint_angles, device=self.device,
                             dtype=torch.float32)
                for g in batch
            ])

            # 2. Object far away → FK → re-write joints → FK again
            temp = obj.data.default_root_state[env_ids].clone()
            temp[:, :3] = (
                env.scene.env_origins[env_ids]
                + torch.tensor([[0, 0, -10.0]], device=self.device)
            )
            temp[:, 7:] = 0.0
            obj.write_root_state_to_sim(temp, env_ids=env_ids)
            obj.update(0.0)

            # FK step + re-write to fix actuator drift
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)
            robot.write_joint_state_to_sim(
                q_batch, torch.zeros_like(q_batch), env_ids=env_ids,
            )
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

            if debug:
                q_now = robot.data.joint_pos[env_ids[0]]
                diff = torch.abs(q_batch[0] - q_now).max().item()
                print(f"    Joint diff after setup: {diff:.6f}")

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
                    print(f"    step {step_i:3d}: obj_vel={obj_v:.4f} "
                          f"obj_pos={[f'{x:.3f}' for x in obj_p]}")

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
# Full pipeline: Seed → RRT → IK → Physics
# ---------------------------------------------------------------------------

def generate_and_validate(
    env,
    object_name: str,
    object_shape: str,
    object_size: float,
    num_grasps: int = 300,
    # HeuristicSampler (Phase 1)
    num_seed_candidates: int = 20000,
    num_seeds: int = 500,
    noise_std: float = 0.3,
    contact_threshold: float = 0.03,
    min_contact_fingers: int = 3,
    penetration_margin: float = 0.008,
    # Surface RRT (Phase 2)
    rrt_target_size: int = 300,
    delta_pos: float = 0.008,
    delta_max: float = 0.04,
    nfo_min_quality: float = 0.03,
    max_attempts_per_step: int = 30,
    min_finger_spacing: float = 0.01,
    # Physics validation (Phase 4)
    settle_steps: int = 40,
    vel_threshold: float = 0.01,
    # Misc
    render: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> GraspSet:
    """
    Full pipeline:
      Phase 1: HeuristicSampler → seed grasps
      Phase 2: Surface RRT expansion → expanded grasps
      Phase 3: IK → joint angles for expanded grasps
      Phase 4: Physics validation → final grasps
    """
    from envs.mdp.sim_utils import get_fingertip_body_ids_from_env

    robot = env.scene["robot"]
    device = env.device
    mesh = make_primitive_mesh(object_shape, object_size)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)

    nfo = NetForceOptimizer(
        mu=0.5, num_edges=8, min_quality=nfo_min_quality,
    )

    # ── Phase 1: Seed generation ─────────────────────────────────
    if verbose:
        print(f"\n  Phase 1: HeuristicSampler → seeds")

    sampler = HeuristicSampler(
        mesh=mesh,
        object_name=object_name,
        object_scale=object_size,
        num_candidates=num_seed_candidates,
        num_grasps=num_seeds,
        num_fingers=5,
        nfo=nfo,
        env=env,
        ft_ids=ft_ids,
        noise_std=noise_std,
        contact_threshold=contact_threshold,
        min_contact_fingers=min_contact_fingers,
        penetration_margin=penetration_margin,
        seed=seed,
    )
    seeds = sampler.sample()

    if len(seeds) == 0:
        print("  WARNING: 0 seeds")
        return GraspSet(object_name=object_name)

    if verbose:
        print(f"  → {len(seeds)} seeds")

    # ── Phase 2: Surface RRT expansion ───────────────────────────
    if verbose:
        print(f"\n  Phase 2: Surface RRT expansion → target {rrt_target_size}")

    expander = SurfaceRRTExpander(
        mesh=mesh,
        nfo=nfo,
        delta_pos=delta_pos,
        delta_max=delta_max,
        min_quality=nfo_min_quality,
        max_attempts_per_step=max_attempts_per_step,
        min_finger_spacing=min_finger_spacing,
        num_fingers=5,
        seed=seed,
    )
    expanded = expander.expand(
        seeds.grasps, target_size=rrt_target_size, verbose=verbose,
    )

    if verbose:
        with_joints = sum(1 for g in expanded if g.joint_angles is not None)
        without_joints = len(expanded) - with_joints
        print(f"  → {len(expanded)} total "
              f"({with_joints} with joints, {without_joints} need IK)")

    # ── Phase 3: IK for grasps without joint angles ──────────────
    needs_ik = [g for g in expanded if g.joint_angles is None]
    if needs_ik and verbose:
        print(f"\n  Phase 3: IK for {len(needs_ik)} grasps")

    if needs_ik:
        expanded = solve_ik_for_grasps(
            env, expanded, render=render, verbose=verbose,
        )

    # Filter out any that still don't have joints
    expanded = [g for g in expanded if g.joint_angles is not None]

    if verbose:
        print(f"  → {len(expanded)} with joint angles")

    # ── Phase 4: Physics validation ──────────────────────────────
    if verbose:
        print(f"\n  Phase 4: Physics validation")

    validator = SimGraspValidator(
        env=env,
        settle_steps=settle_steps,
        vel_threshold=vel_threshold,
        render=render,
    )
    valid = validator.validate(expanded, verbose=verbose)

    valid.sort(key=lambda g: g.quality, reverse=True)
    valid = valid[:num_grasps]

    if verbose:
        print(f"\n  Final: {len(valid)} physics-validated grasps")

    return GraspSet(grasps=valid, object_name=object_name)
