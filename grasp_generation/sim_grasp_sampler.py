"""
Grasp Generation + Physics Validation Pipeline
===============================================
1. HeuristicSampler generates grasp candidates (FK + contact + NFO)
   — uses Isaac Sim FK directly, no MJCF mismatch
2. SimGraspValidator filters them via PhysX physics settle
   — same logic as clean_grasp_graph.py

This reuses the existing proven code:
  - HeuristicSampler (grasp_sampler.py) for candidate generation
  - NetForceOptimizer (net_force_optimization.py) for quality scoring
  - Physics settle (clean_grasp_graph.py logic) for collision validation
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
# Physics Validator (same logic as clean_grasp_graph.py)
# ---------------------------------------------------------------------------

class SimGraspValidator:
    """
    Validates grasp candidates via PhysX physics settle.

    For each grasp:
      1. Set wrist pose (palm-up)
      2. Set stored joint angles
      3. Step physics (FK update, object far away)
      4. Place object at Isaac FK fingertip centroid
      5. Step physics for settle_steps
      6. Check: object velocity < threshold, height > min, drift < max

    Processes grasps in batches of num_envs for efficiency.
    """

    def __init__(
        self,
        env,
        settle_steps: int = 40,
        vel_threshold: float = 0.0,
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
        """
        Filter grasps by physics settle test.
        Batched across num_envs for speed.
        """
        from envs.mdp.sim_utils import (
            set_robot_joints_direct,
            set_robot_root_pose,
            align_wrist_palm_up,
        )

        env = self.env
        robot = self.robot
        obj = self.obj
        valid = []
        reject_counts = {"velocity": 0, "height": 0, "drift": 0}

        if verbose:
            print(f"\n  [Validator] {len(grasps)} candidates, "
                  f"settle_steps={self.settle_steps}, "
                  f"vel_thresh={self.vel_threshold}")

        # imports used in loop
        from envs.mdp.sim_utils import (
            get_local_palm_normal,
            get_fingertip_body_ids_from_env,
        )
        from envs.mdp.math_utils import quat_from_two_vectors, quat_multiply
        from isaaclab.utils.math import quat_apply

        is_first_batch = True

        for batch_start in range(0, len(grasps), self.num_envs):
            batch = grasps[batch_start:batch_start + self.num_envs]
            bs = len(batch)
            env_ids = self.all_env_ids[:bs]
            debug = is_first_batch and verbose
            is_first_batch = False

            # ── 1. Set wrist pose + joints ───────────────────────────
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

            if debug:
                print(f"\n    ── DEBUG: Grasp 0 ──")
                print(f"    Stored joint_angles (first 10): "
                      f"{batch[0].joint_angles[:10].tolist()}")
                q_in_sim = robot.data.joint_pos[env_ids[0]]
                print(f"    Sim joint_pos BEFORE physics (first 10): "
                      f"{q_in_sim[:10].tolist()}")

            # Move object far away
            temp = obj.data.default_root_state[env_ids].clone()
            temp[:, :3] = (
                env.scene.env_origins[env_ids]
                + torch.tensor([[0, 0, -10.0]], device=self.device)
            )
            temp[:, 7:] = 0.0
            obj.write_root_state_to_sim(temp, env_ids=env_ids)
            obj.update(0.0)

            # FK step
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

            if debug:
                q_after_fk = robot.data.joint_pos[env_ids[0]]
                print(f"    Sim joint_pos AFTER FK step (first 10): "
                      f"{q_after_fk[:10].tolist()}")
                diff = torch.abs(q_batch[0] - q_after_fk).max().item()
                print(f"    Max joint diff (stored vs sim): {diff:.6f}")

            # ── 2. Palm-up rotation ──────────────────────────────────
            cur_wrist_pos = robot.data.root_pos_w[env_ids].clone()
            cur_wrist_quat = robot.data.root_quat_w[env_ids].clone()
            palm_n_local = get_local_palm_normal(robot, env)
            palm_n_local = palm_n_local.unsqueeze(0).expand(bs, 3)
            palm_n_world = quat_apply(cur_wrist_quat, palm_n_local)
            target_up = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device,
            ).expand(bs, 3)
            correction = quat_from_two_vectors(palm_n_world, target_up)

            ft_world = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
            pivot = ft_world.mean(dim=1)

            new_wrist_quat = quat_multiply(correction, cur_wrist_quat)
            new_wrist_quat = new_wrist_quat / (
                torch.norm(new_wrist_quat, dim=-1, keepdim=True) + 1e-8
            )
            wrist_rel = cur_wrist_pos - pivot
            new_wrist_pos = quat_apply(correction, wrist_rel) + pivot

            set_robot_root_pose(env, env_ids, new_wrist_pos, new_wrist_quat)
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

            if debug:
                palm_after = quat_apply(
                    robot.data.root_quat_w[env_ids[:1]],
                    palm_n_local[:1],
                )
                print(f"    Palm normal after rotation: "
                      f"{palm_after[0].tolist()} (want ~[0,0,1])")
                q_after_palmup = robot.data.joint_pos[env_ids[0]]
                print(f"    Sim joint_pos AFTER palm-up (first 10): "
                      f"{q_after_palmup[:10].tolist()}")

            # ── 3. Place object at fingertip centroid ────────────────
            ft_pos = robot.data.body_pos_w[env_ids][:, self.ft_ids, :]
            obj_pos = ft_pos.mean(dim=1)

            if debug:
                print(f"    Fingertip positions (env 0):")
                for fi in range(len(self.ft_ids)):
                    print(f"      ft[{fi}]: {ft_pos[0, fi].tolist()}")
                print(f"    Object placed at centroid: {obj_pos[0].tolist()}")

            obj_state = obj.data.default_root_state[env_ids].clone()
            obj_state[:, :3] = obj_pos
            obj_state[:, 3:7] = torch.tensor(
                [[1, 0, 0, 0]], device=self.device, dtype=torch.float32,
            ).expand(bs, -1)
            obj_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
            obj.update(0.0)

            # ── 4. Hold grasp ────────────────────────────────────────
            for step_i in range(self.settle_steps):
                robot.set_joint_position_target(q_batch, env_ids=env_ids)
                env.sim.step(render=self.render)
                env.scene.update(dt=env.physics_dt)

                if debug and step_i in (0, 9, 19, self.settle_steps - 1):
                    q_now = robot.data.joint_pos[env_ids[0]]
                    obj_v = torch.norm(obj.data.root_lin_vel_w[env_ids[0]]).item()
                    obj_p = obj.data.root_pos_w[env_ids[0]].tolist()
                    q_diff = torch.abs(q_batch[0] - q_now).max().item()
                    print(f"    step {step_i:3d}: obj_vel={obj_v:.4f} "
                          f"obj_pos={[f'{x:.3f}' for x in obj_p]} "
                          f"max_joint_diff={q_diff:.4f}")

            # ── 5. Check stability ───────────────────────────────────
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
                passed = True
                reason = ""
                if speed[j] >= self.vel_threshold:
                    reject_counts["velocity"] += 1
                    reason = f"vel={speed[j].item():.4f}"
                    passed = False
                elif obj_z[j] < self.min_height:
                    reject_counts["height"] += 1
                    reason = f"z={obj_z[j].item():.4f}"
                    passed = False
                elif obj_drift[j] > self.max_drift:
                    reject_counts["drift"] += 1
                    reason = f"drift={obj_drift[j].item():.4f}"
                    passed = False

                if passed:
                    valid.append(batch[j])

                if debug and j < 3:
                    status = "PASS" if passed else f"FAIL({reason})"
                    print(f"    grasp {batch_start+j}: vel={speed[j].item():.4f} "
                          f"z={obj_z[j].item():.3f} "
                          f"drift={obj_drift[j].item():.4f} → {status}")

            if verbose:
                done = min(batch_start + bs, len(grasps))
                print(f"    [{done}/{len(grasps)}] passed: {len(valid)}")

        if verbose:
            print(f"  [Validator] {len(valid)}/{len(grasps)} passed. "
                  f"Rejects: {reject_counts}")

        return valid


# ---------------------------------------------------------------------------
# Integrated pipeline: generate + validate
# ---------------------------------------------------------------------------

def generate_and_validate(
    env,
    object_name: str,
    object_shape: str,
    object_size: float,
    num_grasps: int = 300,
    # HeuristicSampler params
    num_candidates: int = 20000,
    noise_std: float = 0.3,
    contact_threshold: float = 0.03,
    min_contact_fingers: int = 3,
    penetration_margin: float = 0.008,
    nfo_min_quality: float = 0.03,
    # Validator params
    settle_steps: int = 40,
    vel_threshold: float = 0.0,
    # Misc
    render: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> GraspSet:
    """
    Full pipeline:
      1. HeuristicSampler generates candidates (FK + contact + NFO)
      2. SimGraspValidator filters via PhysX physics settle
    """
    from envs.mdp.sim_utils import get_fingertip_body_ids_from_env

    robot = env.scene["robot"]
    device = env.device
    mesh = make_primitive_mesh(object_shape, object_size)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)

    # NFO evaluator
    nfo = NetForceOptimizer(
        mu=0.5, num_edges=8, min_quality=nfo_min_quality,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Phase 1: HeuristicSampler (FK + NFO)")
        print(f"  candidates={num_candidates}, noise={noise_std}, "
              f"nfo_min={nfo_min_quality}")
        print(f"{'='*60}")

    # Generate ~1.5x target to account for physics filtering
    gen_target = int(num_grasps * 1.5)

    sampler = HeuristicSampler(
        mesh=mesh,
        object_name=object_name,
        object_scale=object_size,
        num_candidates=num_candidates,
        num_grasps=gen_target,
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

    candidates = sampler.sample()

    if len(candidates) == 0:
        print("  WARNING: HeuristicSampler produced 0 candidates")
        return GraspSet(object_name=object_name)

    # Debug: inspect first candidate from Phase 1
    if verbose and len(candidates) > 0:
        g0 = candidates[0]
        print(f"\n  [Phase 1 Debug] First candidate:")
        print(f"    quality={g0.quality:.4f}")
        print(f"    joint_angles ({len(g0.joint_angles)} DOF, first 10): "
              f"{g0.joint_angles[:10].tolist()}")
        print(f"    fingertip_positions:\n      {g0.fingertip_positions}")
        print(f"    object_pos_hand={g0.object_pos_hand}")
        print(f"    object_quat_hand={g0.object_quat_hand}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Phase 2: Physics Validation")
        print(f"  settle_steps={settle_steps}, vel_thresh={vel_threshold}")
        print(f"{'='*60}")

    validator = SimGraspValidator(
        env=env,
        settle_steps=settle_steps,
        vel_threshold=vel_threshold,
        render=render,
    )

    valid_grasps = validator.validate(
        candidates.grasps, verbose=verbose,
    )

    # Sort by quality, truncate
    valid_grasps.sort(key=lambda g: g.quality, reverse=True)
    valid_grasps = valid_grasps[:num_grasps]

    if verbose:
        print(f"\n  Final: {len(valid_grasps)} grasps "
              f"(generated {len(candidates)}, validated {len(valid_grasps)})")

    return GraspSet(grasps=valid_grasps, object_name=object_name)
