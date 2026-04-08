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
        settle_steps: int = 15,
        vel_threshold: float = 0.3,
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

        for batch_start in range(0, len(grasps), self.num_envs):
            batch = grasps[batch_start:batch_start + self.num_envs]
            bs = len(batch)
            env_ids = self.all_env_ids[:bs]

            # 1. Default wrist pose → palm-up
            root_state = robot.data.default_root_state[env_ids, :7].clone()
            root_state[:, :3] += env.scene.env_origins[env_ids]
            wrist_pos = root_state[:, :3]
            wrist_quat = root_state[:, 3:7]
            set_robot_root_pose(env, env_ids, wrist_pos, wrist_quat)

            # 2. Set stored joint angles
            joint_list = [g.joint_angles for g in batch]
            set_robot_joints_direct(env, env_ids, joint_list)

            # 3. Move object far away → FK step
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

            # 4. Place object at Isaac FK fingertip centroid
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

            # 5. Physics settle — re-apply joint targets each step
            q_batch = torch.stack([
                torch.tensor(g.joint_angles, device=self.device,
                             dtype=torch.float32)
                for g in batch
            ])
            for _ in range(self.settle_steps):
                robot.set_joint_position_target(q_batch, env_ids=env_ids)
                env.sim.step(render=self.render)
                env.scene.update(dt=env.physics_dt)

            # 6. Check stability
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
    settle_steps: int = 15,
    vel_threshold: float = 0.3,
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
