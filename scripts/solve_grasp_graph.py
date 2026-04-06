"""
Solve Grasp Graph — Fill joint_angles + object_pose via Isaac Sim IK.

For each grasp in the graph:
  1. Place object at fixed position
  2. Compute wrist pose from fingertip arrangement
  3. Set joints to adaptive initial pose
  4. Run per-finger differential IK (null-space + SVD fallback)
  5. Measure fingertip error → reject if too high
  6. Settle physics → reject if object ejected/dropped
  7. Store joint_angles + object_pos_hand + object_quat_hand + error

This converts a "contact-set graph" (fingertip positions only) into a
"robot-state graph" (joint_angles + object_pose) usable for RL reset.

Usage:
    /isaac-sim/python.sh scripts/solve_grasp_graph.py \\
        --input data/grasp_graph.pkl \\
        --output data/grasp_graph_solved.pkl \\
        --error-threshold 0.008 \\
        --vel-threshold 0.3
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Solve IK for grasp graph nodes")
    p.add_argument("--input", type=str, default="data/grasp_graph.pkl")
    p.add_argument("--output", type=str, default="data/grasp_graph_solved.pkl")
    p.add_argument("--error-threshold", type=float, default=0.008,
                   help="Max mean fingertip error (m) to accept a grasp")
    p.add_argument("--vel-threshold", type=float, default=0.3,
                   help="Max object velocity (m/s) after settling")
    p.add_argument("--settle-steps", type=int, default=5,
                   help="Physics steps after placement before measuring")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num_envs", type=int, default=1)
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
    from isaaclab.utils.math import quat_apply_inverse

    from envs.anygrasp_env import AnyGraspEnvCfg
    from grasp_generation.graph_io import load_merged_graph
    from grasp_generation.grasp_sampler import GraspSet

    from envs.mdp.sim_utils import (
        place_object_fixed,
        compute_wrist_from_fingertips,
        set_robot_root_pose,
        set_adaptive_joint_pose,
        refine_hand_to_start_grasp,
        get_fingertip_body_ids_from_env,
        pad_fingertip_positions,
    )
    from envs.mdp.math_utils import (
        local_to_world_points,
        quat_multiply,
        quat_conjugate,
    )

    print(f"[SolveGraph] Loading: {args.input}")
    graph = load_merged_graph(args.input)
    if graph is None:
        print("ERROR: Could not load grasp graph")
        sim_app.close()
        return

    # Create a minimal env
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.grasp_graph_path = args.input

    import isaaclab.sim as sim_cfg_utils
    env_cfg.scene.object.spawn.rigid_props = sim_cfg_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, max_depenetration_velocity=5.0,
    )

    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    ft_ids = get_fingertip_body_ids_from_env(robot, env)

    print(f"[SolveGraph] Objects: {graph.object_names}")
    print(f"[SolveGraph] error_threshold={args.error_threshold}m, "
          f"vel_threshold={args.vel_threshold}m/s, settle_steps={args.settle_steps}")

    total_before = 0
    total_solved = 0

    for obj_name in graph.object_names:
        subgraph = graph.graphs[obj_name]
        grasp_set = subgraph.grasp_set
        n_grasps = len(grasp_set)
        total_before += n_grasps
        env_num_fingers = subgraph.num_fingers

        # Get object size for adaptive joint pose
        obj_size = 0.06
        if hasattr(graph, "object_specs"):
            spec = graph.object_specs.get(obj_name, {})
            if "size" in spec:
                obj_size = float(spec["size"])

        print(f"\n[SolveGraph] Solving {obj_name}: {n_grasps} grasps (size={obj_size:.3f}m)...")

        solved_indices = []
        for idx in range(n_grasps):
            grasp = grasp_set[idx]
            if grasp.fingertip_positions is None:
                continue

            fp = pad_fingertip_positions(grasp.fingertip_positions.copy(), env_num_fingers)
            fp_tensor = torch.tensor(
                fp, device=device, dtype=torch.float32,
            ).unsqueeze(0)  # (1, F, 3)

            # ── Step 1: Place object at fixed position ──
            obj_pos_w, obj_quat_w = place_object_fixed(env, env_ids)
            obj.update(0.0)

            # ── Step 2: Fingertip targets in world frame ──
            start_world = local_to_world_points(fp_tensor, obj_pos_w, obj_quat_w)

            # ── Step 3: Compute wrist from fingertips ──
            wrist_pos, wrist_quat = compute_wrist_from_fingertips(
                env, env_ids, start_world,
            )
            set_robot_root_pose(env, env_ids, wrist_pos, wrist_quat)

            # ── Step 4: Adaptive initial joint pose ──
            set_adaptive_joint_pose(env, env_ids, obj_size)
            robot.update(0.0)

            # Re-fix object
            obj_root_state = obj.data.default_root_state[env_ids].clone()
            obj_root_state[:, :3] = obj_pos_w
            obj_root_state[:, 3:7] = obj_quat_w
            obj_root_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_root_state, env_ids=env_ids)
            obj.update(0.0)

            # ── Step 5: Per-finger IK ──
            refine_hand_to_start_grasp(env, env_ids, fp_tensor)
            robot.update(0.0)

            # ── Step 6: Measure fingertip error ──
            actual_tips = robot.data.body_pos_w[env_ids][:, ft_ids, :]
            target_world = local_to_world_points(fp_tensor, obj_pos_w, obj_quat_w)
            tip_err = torch.norm(actual_tips - target_world, dim=-1)  # (1, F)
            mean_err = float(tip_err.mean().item())
            max_err = float(tip_err.max().item())

            if mean_err > args.error_threshold:
                continue  # IK failed

            # ── Step 7: Settle physics ──
            for _ in range(args.settle_steps):
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

            # ── Step 8: Check object stability ──
            obj_vel = obj.data.root_lin_vel_w[0]
            speed = float(torch.norm(obj_vel).item())
            obj_z = float(obj.data.root_pos_w[0, 2].item())

            if speed > args.vel_threshold or obj_z < 0.15:
                continue  # Object ejected or dropped

            # ── Step 9: Store results ──
            joint_pos = robot.data.joint_pos[0].cpu().numpy()
            grasp.joint_angles = joint_pos.copy()

            # Compute object pose in hand frame
            rp_w = robot.data.root_pos_w[0]
            rq_w = robot.data.root_quat_w[0]
            op_w = obj.data.root_pos_w[0]
            oq_w = obj.data.root_quat_w[0]
            rel = op_w - rp_w
            obj_pos_hand = quat_apply_inverse(
                rq_w.unsqueeze(0), rel.unsqueeze(0),
            )[0]
            obj_quat_hand = quat_multiply(
                quat_conjugate(rq_w.unsqueeze(0)), oq_w.unsqueeze(0),
            )[0]

            grasp.object_pos_hand = obj_pos_hand.cpu().numpy().copy()
            grasp.object_quat_hand = obj_quat_hand.cpu().numpy().copy()
            grasp.object_pose_frame = "hand_root"
            grasp.reset_contact_error = mean_err
            grasp.reset_contact_error_max = max_err

            solved_indices.append(idx)

            if (idx + 1) % 50 == 0 or idx == n_grasps - 1:
                print(f"  [{idx+1}/{n_grasps}] solved: {len(solved_indices)} "
                      f"(last err: {mean_err:.4f}m)")

        # Rebuild grasp set with only solved grasps
        old_to_new = {old: new for new, old in enumerate(solved_indices)}
        new_grasps = [grasp_set[i] for i in solved_indices]

        # Rebuild edges
        new_edges = []
        for i, j in subgraph.edges:
            if i in old_to_new and j in old_to_new:
                new_edges.append((old_to_new[i], old_to_new[j]))

        subgraph.grasp_set = GraspSet(grasps=new_grasps, object_name=obj_name)
        subgraph.edges = new_edges

        n_solved = len(new_grasps)
        total_solved += n_solved
        print(f"[SolveGraph] {obj_name}: {n_solved}/{n_grasps} solved "
              f"({n_grasps - n_solved} rejected, {len(new_edges)} edges)")

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  SOLVE SUMMARY")
    print("=" * 60)
    for obj_name in graph.object_names:
        sg = graph.graphs[obj_name]
        n = len(sg.grasp_set)
        e = len(sg.edges)
        print(f"  {obj_name:30s}  {n:4d} grasps  {e:4d} edges")
    print("-" * 60)
    print(f"  {'TOTAL':30s}  {total_solved:4d}/{total_before} solved "
          f"({total_before - total_solved} rejected)")
    print("=" * 60)

    if total_solved == 0:
        print("\n[WARNING] 0 grasps solved! Check:")
        print("  - Object size may be too small for the hand")
        print("  - Try relaxing --error-threshold (current: "
              f"{args.error_threshold}m)")
        print("  - Try increasing --settle-steps (current: "
              f"{args.settle_steps})")
        print("  - Ensure grasp_graph.pkl has grasps "
              "(run scripts/run_grasp_generation.py first)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"[SolveGraph] Saved to: {output_path}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
