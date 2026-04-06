"""
Solve Grasp Graph — Fill joint_angles + object_pose via Isaac Sim IK.

This script converts a "contact-set graph" (fingertip positions only) into a
"robot-state graph" (joint_angles + object_pose) usable for RL reset.

Minimal-change improvements for Shadow Hand:
1. Initialize wrist from grasp-specific fingertip targets instead of default root pose.
2. Use stored object_pos_hand/object_quat_hand if available; otherwise fall back to centroid + identity quat.
3. Relax default acceptance thresholds so valid grasps are less likely to all get rejected.

Usage:
    /isaac-sim/python.sh scripts/solve_grasp_graph.py \
        --input data/grasp_graph.pkl \
        --output data/grasp_graph_solved.pkl \
        --error-threshold 0.015 \
        --vel-threshold 0.8 \
        --settle-steps 2
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
    p.add_argument(
        "--error-threshold",
        type=float,
        default=0.015,
        help="Max mean fingertip error (m) to accept a grasp",
    )
    p.add_argument(
        "--vel-threshold",
        type=float,
        default=0.8,
        help="Max object linear speed (m/s) after settling",
    )
    p.add_argument(
        "--settle-steps",
        type=int,
        default=2,
        help="Physics steps after placement before measuring stability",
    )
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    return p.parse_args()


def _normalize_quat_torch(q, eps: float = 1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def main():
    args = parse_args()

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_cfg_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from envs.anygrasp_env import AnyGraspEnvCfg
    from envs.mdp.math_utils import (
        local_to_world_points,
        quat_conjugate,
        quat_multiply,
    )
    from envs.mdp.sim_utils import (
        compute_wrist_from_fingertips,
        get_fingertip_body_ids_from_env,
        pad_fingertip_positions,
        place_object_fixed,
        refine_hand_to_start_grasp,
        set_adaptive_joint_pose,
        set_robot_root_pose,
    )
    from grasp_generation.graph_io import load_merged_graph
    from grasp_generation.grasp_sampler import GraspSet

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
    env_cfg.scene.object.spawn.rigid_props = sim_cfg_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )

    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)

    print(f"[SolveGraph] Objects: {graph.object_names}")
    print(
        f"[SolveGraph] error_threshold={args.error_threshold}m, "
        f"vel_threshold={args.vel_threshold}m/s, "
        f"settle_steps={args.settle_steps}"
    )

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

        print(
            f"\n[SolveGraph] Solving {obj_name}: {n_grasps} grasps "
            f"(size={obj_size:.3f}m)..."
        )

        solved_indices = []
        reject_no_fp = 0
        reject_ik = 0
        reject_physics = 0

        ik_errors = []
        settle_speeds = []
        settle_heights = []

        for idx in range(n_grasps):
            grasp = grasp_set[idx]

            if grasp.fingertip_positions is None:
                reject_no_fp += 1
                continue

            fp = pad_fingertip_positions(
                grasp.fingertip_positions.copy(),
                env_num_fingers,
            )
            fp_tensor = torch.tensor(
                fp,
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0)  # (1, F, 3)

            # ------------------------------------------------------------------
            # Step 1. Seed an object pose just to compute world-space fingertip
            # targets for wrist initialization.
            # ------------------------------------------------------------------
            seed_obj_pos_w, seed_obj_quat_w = place_object_fixed(env, env_ids)
            obj.update(0.0)

            start_world = local_to_world_points(fp_tensor, seed_obj_pos_w, seed_obj_quat_w)

            # ------------------------------------------------------------------
            # Step 2. Compute wrist pose from the grasp-specific fingertip layout.
            # ------------------------------------------------------------------
            wrist_pos, wrist_quat = compute_wrist_from_fingertips(
                env,
                env_ids,
                start_world,
            )
            set_robot_root_pose(env, env_ids, wrist_pos, wrist_quat)

            # ------------------------------------------------------------------
            # Step 3. Adaptive initial joint pose
            # ------------------------------------------------------------------
            set_adaptive_joint_pose(env, env_ids, obj_size)
            robot.update(0.0)

            # ------------------------------------------------------------------
            # Step 4. Place object.
            # Prefer the stored hand-relative pose when available.
            # Otherwise fall back to fingertip centroid + identity quaternion.
            # ------------------------------------------------------------------
            hand_pos_w = robot.data.root_pos_w[env_ids].clone()
            hand_quat_w = robot.data.root_quat_w[env_ids].clone()

            has_stored_pose = (
                getattr(grasp, "object_pos_hand", None) is not None
                and getattr(grasp, "object_quat_hand", None) is not None
            )

            if has_stored_pose:
                obj_pos_hand_t = torch.tensor(
                    np.asarray(grasp.object_pos_hand),
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                obj_quat_hand_t = torch.tensor(
                    np.asarray(grasp.object_quat_hand),
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)

                obj_pos_w = hand_pos_w + quat_apply(hand_quat_w, obj_pos_hand_t)
                obj_quat_w = quat_multiply(hand_quat_w, obj_quat_hand_t)
                obj_quat_w = _normalize_quat_torch(obj_quat_w)
            else:
                ft_pos_w = robot.data.body_pos_w[env_ids][:, ft_ids, :]  # (1, F, 3)
                obj_pos_w = ft_pos_w.mean(dim=1)  # (1, 3)
                obj_quat_w = torch.zeros(1, 4, device=device)
                obj_quat_w[:, 0] = 1.0

            obj_root_state = obj.data.default_root_state[env_ids].clone()
            obj_root_state[:, :3] = obj_pos_w
            obj_root_state[:, 3:7] = obj_quat_w
            obj_root_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_root_state, env_ids=env_ids)
            obj.update(0.0)

            # ------------------------------------------------------------------
            # Step 5. Per-finger IK to match grasp-specific targets
            # ------------------------------------------------------------------
            # Debug: print state BEFORE IK for first grasp
            if idx == 0:
                _jac = robot.root_physx_view.get_jacobians()
                print(f"[DEBUG] robot.is_fixed_base = {robot.is_fixed_base}")
                print(f"[DEBUG] Jacobian shape = {tuple(_jac.shape)}")
                print(f"[DEBUG] joint_pos shape = {tuple(robot.data.joint_pos.shape)}")
                print(f"[DEBUG] num_bodies = {robot.data.body_pos_w.shape[1]}")
                print(f"[DEBUG] ft_ids = {ft_ids}")
                print(f"[DEBUG] wrist_pos = {robot.data.root_pos_w[env_ids][0].tolist()}")
                print(f"[DEBUG] wrist_quat = {robot.data.root_quat_w[env_ids][0].tolist()}")
                print(f"[DEBUG] obj_pos_w = {obj_pos_w[0].tolist()}")
                print(f"[DEBUG] obj_quat_w = {obj_quat_w[0].tolist()}")
                _ft_before = robot.data.body_pos_w[env_ids][:, ft_ids, :]
                _tgt = local_to_world_points(fp_tensor, obj_pos_w, obj_quat_w)
                _err_before = torch.norm(_ft_before - _tgt, dim=-1)
                print(f"[DEBUG] fingertip positions (before IK):")
                for fi in range(len(ft_ids)):
                    print(f"  finger {fi}: actual={_ft_before[0,fi].tolist()}  "
                          f"target={_tgt[0,fi].tolist()}  "
                          f"err={float(_err_before[0,fi]):.4f}m")
                print(f"[DEBUG] fp_tensor (object frame) = {fp_tensor[0].tolist()}")
                print(f"[DEBUG] mean_err BEFORE IK = {float(_err_before.mean()):.4f}m")

            refine_hand_to_start_grasp(env, env_ids, fp_tensor)
            robot.update(0.0)

            # ------------------------------------------------------------------
            # Step 6. Measure fingertip error against the actual object pose used
            # in this solve attempt.
            # ------------------------------------------------------------------
            actual_tips = robot.data.body_pos_w[env_ids][:, ft_ids, :]
            target_world = local_to_world_points(fp_tensor, obj_pos_w, obj_quat_w)

            # Debug: print state AFTER IK for first grasp
            if idx == 0:
                _err_after = torch.norm(actual_tips - target_world, dim=-1)
                print(f"[DEBUG] mean_err AFTER IK = {float(_err_after.mean()):.4f}m")
                for fi in range(len(ft_ids)):
                    print(f"  finger {fi}: actual={actual_tips[0,fi].tolist()}  "
                          f"target={target_world[0,fi].tolist()}  "
                          f"err={float(_err_after[0,fi]):.4f}m")

            tip_err = torch.norm(actual_tips - target_world, dim=-1)  # (1, F)

            mean_err = float(tip_err.mean().item())
            max_err = float(tip_err.max().item())
            ik_errors.append(mean_err)

            # Progress every 50 grasps
            if (idx + 1) % 50 == 0:
                print(
                    f"  [{idx + 1}/{n_grasps}] solved={len(solved_indices)} "
                    f"ik_reject={reject_ik} phys_reject={reject_physics} "
                    f"last_err={mean_err:.4f}m"
                )

            if mean_err > args.error_threshold:
                reject_ik += 1
                continue

            # ------------------------------------------------------------------
            # Step 7. Settle physics
            # ------------------------------------------------------------------
            for _ in range(args.settle_steps):
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

            # ------------------------------------------------------------------
            # Step 8. Check object stability
            # ------------------------------------------------------------------
            obj_vel = obj.data.root_lin_vel_w[0]
            speed = float(torch.norm(obj_vel).item())
            obj_z = float(obj.data.root_pos_w[0, 2].item())

            settle_speeds.append(speed)
            settle_heights.append(obj_z)

            if speed > args.vel_threshold or obj_z < 0.15:
                reject_physics += 1
                continue

            # ------------------------------------------------------------------
            # Step 9. Store results
            # ------------------------------------------------------------------
            joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
            grasp.joint_angles = joint_pos.copy()

            # Compute object pose in hand frame from final sim state
            rp_w = robot.data.root_pos_w[0]
            rq_w = robot.data.root_quat_w[0]
            op_w = obj.data.root_pos_w[0]
            oq_w = obj.data.root_quat_w[0]

            rel = op_w - rp_w
            obj_pos_hand = quat_apply_inverse(
                rq_w.unsqueeze(0),
                rel.unsqueeze(0),
            )[0]
            obj_quat_hand = quat_multiply(
                quat_conjugate(rq_w.unsqueeze(0)),
                oq_w.unsqueeze(0),
            )[0]

            grasp.object_pos_hand = obj_pos_hand.detach().cpu().numpy().copy()
            grasp.object_quat_hand = obj_quat_hand.detach().cpu().numpy().copy()
            grasp.object_pose_frame = "hand_root"
            grasp.reset_contact_error = mean_err
            grasp.reset_contact_error_max = max_err

            solved_indices.append(idx)

        # Per-object rejection breakdown
        print(f"\n  Rejection breakdown for {obj_name}:")
        print(f"    no fingertip data: {reject_no_fp}")
        print(f"    IK error > {args.error_threshold}m: {reject_ik}")
        print(f"    physics unstable: {reject_physics}")

        if ik_errors:
            ik_arr = np.array(ik_errors)
            print(
                f"    IK error stats: mean={ik_arr.mean():.4f}m "
                f"median={np.median(ik_arr):.4f}m "
                f"min={ik_arr.min():.4f}m "
                f"max={ik_arr.max():.4f}m"
            )
        if settle_speeds:
            sp_arr = np.array(settle_speeds)
            print(
                f"    settle speed stats: mean={sp_arr.mean():.4f}m/s "
                f"median={np.median(sp_arr):.4f}m/s "
                f"min={sp_arr.min():.4f}m/s "
                f"max={sp_arr.max():.4f}m/s"
            )
        if settle_heights:
            z_arr = np.array(settle_heights)
            print(
                f"    settle height stats: mean={z_arr.mean():.4f}m "
                f"median={np.median(z_arr):.4f}m "
                f"min={z_arr.min():.4f}m "
                f"max={z_arr.max():.4f}m"
            )

        # Keep only solved grasps for this object
        if len(solved_indices) == 0:
            print(f"  WARNING: no solved grasps for {obj_name}")
            subgraph.grasp_set = GraspSet([])
            if hasattr(subgraph, "graph"):
                try:
                    import networkx as nx
                    subgraph.graph = nx.Graph()
                except Exception:
                    pass
            continue

        solved_grasps = [grasp_set[i] for i in solved_indices]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(solved_indices)}

        # Filter graph edges to solved subset only
        new_nx_graph = None
        if hasattr(subgraph, "graph") and subgraph.graph is not None:
            try:
                import networkx as nx
                new_nx_graph = nx.Graph()
                for old_idx in solved_indices:
                    new_nx_graph.add_node(old_to_new[old_idx])
                for u, v, data in subgraph.graph.edges(data=True):
                    if u in old_to_new and v in old_to_new:
                        new_nx_graph.add_edge(
                            old_to_new[u],
                            old_to_new[v],
                            **data,
                        )
            except Exception:
                new_nx_graph = None

        subgraph.grasp_set = GraspSet(solved_grasps)
        if new_nx_graph is not None:
            subgraph.graph = new_nx_graph

        num_solved = len(solved_grasps)
        total_solved += num_solved
        print(
            f"[SolveGraph] {obj_name}: kept {num_solved}/{n_grasps} grasps "
            f"({100.0 * num_solved / max(n_grasps, 1):.1f}%)"
        )

    # Save solved graph
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(graph, f)

    print("\n[SolveGraph] Done.")
    print(
        f"[SolveGraph] Total kept: {total_solved}/{total_before} grasps "
        f"({100.0 * total_solved / max(total_before, 1):.1f}%)"
    )
    print(f"[SolveGraph] Saved to: {out_path}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
