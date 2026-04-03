"""
Clean Grasp Graph — Filter out grasps that cause penetration in simulation.

Launches Isaac Lab with a single env, then for each grasp:
  1. Teleport hand+object into the grasp pose (same as training reset)
  2. Step physics for a few frames
  3. Measure object velocity — high velocity = penetration ejection
  4. Keep only grasps where object stays stable

Usage:
    python scripts/clean_grasp_graph.py [--vel-threshold 0.3] [--settle-steps 5]
"""

import argparse
import copy
import math
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Filter penetrating grasps via sim test")
    p.add_argument("--input", type=str, default="data/grasp_graph.pkl")
    p.add_argument("--output", type=str, default="data/grasp_graph_clean.pkl")
    p.add_argument("--vel-threshold", type=float, default=0.3,
                   help="Max object linear velocity (m/s) after settling. Above = bad grasp.")
    p.add_argument("--settle-steps", type=int, default=5,
                   help="Sim steps after placement before measuring velocity")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num_envs", type=int, default=1)
    # Isaac Lab AppLauncher compatibility
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from envs.anygrasp_env import AnyGraspEnvCfg
    from grasp_generation.graph_io import load_merged_graph
    from grasp_generation.grasp_sampler import GraspSet

    print(f"[CleanGraph] Loading: {args.input}")
    graph = load_merged_graph(args.input)
    if graph is None:
        print("ERROR: Could not load grasp graph")
        sim_app.close()
        return

    # Create a minimal env for sim testing
    env_cfg = AnyGraspEnvCfg(num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.grasp_graph_path = args.input
    # Override depenetration velocity to HIGH so penetrating grasps
    # produce visible ejection velocity (easy to detect).
    # Training uses 0.05 to suppress ejection, but here we WANT it.
    import isaaclab.sim as sim_utils
    env_cfg.scene.object.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, max_depenetration_velocity=5.0,
    )

    env = ManagerBasedRLEnv(env_cfg)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device

    # Import placement helpers from events
    from envs.mdp.events import (
        _align_wrist_palm_up,
        _sample_wrist_pose_world,
        _set_robot_root_pose,
        _set_robot_joints_direct,
        _place_object_in_hand,
        _get_fingertip_body_ids_from_env,
        _pad_fingertip_positions,
        _expand_grasp_joint_vector,
    )

    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    print(f"[CleanGraph] Objects: {graph.object_names}")
    print(f"[CleanGraph] vel_threshold={args.vel_threshold} m/s, settle_steps={args.settle_steps}")
    total_before = 0
    total_after = 0

    for obj_name in graph.object_names:
        subgraph = graph.graphs[obj_name]
        grasp_set = subgraph.grasp_set
        n_grasps = len(grasp_set)
        total_before += n_grasps
        env_num_fingers = subgraph.num_fingers

        print(f"\n[CleanGraph] Testing {obj_name}: {n_grasps} grasps...")

        clean_indices = []
        for idx in range(n_grasps):
            grasp = grasp_set[idx]

            # Skip if missing essential data
            if grasp.fingertip_positions is None or grasp.joint_angles is None:
                continue

            fp = _pad_fingertip_positions(grasp.fingertip_positions, env_num_fingers)
            fp_tensor = torch.tensor(fp, device=device, dtype=torch.float32).unsqueeze(0)  # (1, F, 3)

            # Step 1: Set wrist palm-up
            wrist_pos, wrist_quat = _sample_wrist_pose_world(env, env_ids, apply_noise=False)
            wrist_quat = _align_wrist_palm_up(env, env_ids, wrist_quat)
            _set_robot_root_pose(env, env_ids, wrist_pos, wrist_quat)

            # Step 2: Set joint angles
            _set_robot_joints_direct(env, env_ids, [grasp.joint_angles])

            # Step 3: FK update
            robot.update(0.0)

            # Step 4: Place object via rigid alignment
            _place_object_in_hand(env, env_ids, fp_tensor)

            # Step 5: Settle — step physics
            for _ in range(args.settle_steps):
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

            # Step 6: Measure object velocity
            obj_vel = obj.data.root_lin_vel_w[0]  # (3,)
            speed = float(torch.norm(obj_vel).item())

            # Also check if object fell
            obj_z = float(obj.data.root_pos_w[0, 2].item())

            if speed < args.vel_threshold and obj_z > 0.15:
                clean_indices.append(idx)

            if (idx + 1) % 50 == 0 or idx == n_grasps - 1:
                print(f"  [{idx+1}/{n_grasps}] clean so far: {len(clean_indices)}")

        # Rebuild grasp set
        old_to_new = {old: new for new, old in enumerate(clean_indices)}
        new_grasps = [grasp_set[i] for i in clean_indices]

        # Rebuild edges
        new_edges = []
        for i, j in subgraph.edges:
            if i in old_to_new and j in old_to_new:
                new_edges.append((old_to_new[i], old_to_new[j]))

        subgraph.grasp_set = GraspSet(grasps=new_grasps, object_name=obj_name)
        subgraph.edges = new_edges

        n_clean = len(new_grasps)
        total_after += n_clean
        removed = n_grasps - n_clean
        print(f"[CleanGraph] {obj_name}: {n_clean}/{n_grasps} clean "
              f"({removed} removed, {len(new_edges)} edges)")

    print(f"\n[CleanGraph] Total: {total_after}/{total_before} grasps kept "
          f"({total_before - total_after} removed)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"[CleanGraph] Saved to: {output_path}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
