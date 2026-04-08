"""
Visualize stored grasps from the grasp graph directly.

Loads each grasp's joint_angles, sets the robot, places the object
at fingertip centroid, and renders without physics stepping.
Cycles through grasps every few seconds.

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/visualize_grasps.py \
        --grasp_graph data/grasp_graph_clean.pkl --num_envs 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser(description="Visualize stored grasps from grasp graph")
    p.add_argument("--grasp_graph", action="append", default=None)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--interval", type=float, default=3.0,
                   help="Seconds between grasp switches")
    p.add_argument("--object_name", type=str, default=None,
                   help="Specific object name to visualize (default: first in graph)")
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"))
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.grasp_graph is None:
        args.grasp_graph = ["data/grasp_graph_clean.pkl"]

    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from envs import AnyGraspEnvCfg, register_anygrasp_env
    from envs.anygrasp_env import _build_object_spawner
    from grasp_generation.graph_io import load_merged_graph
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph
    from envs.mdp.sim_utils import (
        set_robot_joints_direct,
        set_robot_root_pose,
        get_fingertip_body_ids_from_env,
    )

    register_anygrasp_env()

    merged_graph = load_merged_graph(args.grasp_graph)
    if merged_graph is None:
        print("ERROR: Could not load grasp graph")
        sim_app.close()
        sys.exit(1)

    # Pick object
    if isinstance(merged_graph, MultiObjectGraspGraph):
        if args.object_name and args.object_name in merged_graph.graphs:
            obj_name = args.object_name
        else:
            obj_name = next(iter(merged_graph.graphs.keys()))
        g = merged_graph.graphs[obj_name]
        specs = list(merged_graph.object_specs.values())
    else:
        obj_name = merged_graph.object_name
        g = merged_graph
        specs = None

    print(f"[GraspViz] Object: {obj_name}, {len(g)} grasps")

    # Build env
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = args.grasp_graph
    if specs:
        env_cfg.scene.object = env_cfg.scene.object.replace(spawn=_build_object_spawner(specs))

    # Set num_fingers from graph
    graph_nf = getattr(g, "num_fingers", 5)
    tip_subsets = {
        2: ["robot0_ffdistal", "robot0_thdistal"],
        3: ["robot0_ffdistal", "robot0_mfdistal", "robot0_thdistal"],
        4: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_thdistal"],
        5: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"],
    }
    env_cfg.hand = {"name": "shadow", "num_fingers": graph_nf,
                    "fingertip_links": tip_subsets.get(graph_nf, tip_subsets[5])}

    # Disable DR
    env_cfg.reset_randomization = {
        "wrist_pos_jitter_std": 0.0,
        "wrist_rot_std_deg": 0.0,
        "align_palm_up": True,
    }

    try:
        import yaml
        with open(args.config) as f:
            cfg_file = yaml.safe_load(f) or {}
        from scripts.train_rl import apply_env_config, apply_dr_config
        apply_env_config(env_cfg, cfg_file.get("env", {}))
    except Exception:
        pass

    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    env_ids = torch.arange(args.num_envs, device=device, dtype=torch.long)

    # Initial reset to get env running (applies palm-up transform)
    env.reset()

    # Store palm-up wrist pose from initial reset (relative to env origin)
    palm_up_wrist_local = robot.data.root_pos_w[0].clone() - env.scene.env_origins[0]
    palm_up_wrist_quat = robot.data.root_quat_w[0].clone()

    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    grasps = g.grasp_set.grasps
    grasp_idx = 0

    # Print Isaac Sim joint info for debugging MJCF→USD mapping
    joint_names = robot.data.joint_names
    print(f"\n[GraspViz] Isaac USD joint order ({len(joint_names)} joints):")
    for i, name in enumerate(joint_names):
        q = float(robot.data.joint_pos[0, i].item())
        lo = float(robot.data.soft_joint_pos_limits[0, i, 0].item())
        hi = float(robot.data.soft_joint_pos_limits[0, i, 1].item())
        print(f"  [{i:2d}] {name:30s}  pos={q:.4f}  limits=[{lo:.3f}, {hi:.3f}]")

    # Print first grasp's MJCF→Isaac mapped joints
    if grasps[0].joint_angles is not None:
        ja = grasps[0].joint_angles
        print(f"\n[GraspViz] First grasp joint_angles ({len(ja)} DOF):")
        for i in range(len(ja)):
            name = joint_names[i] if i < len(joint_names) else "???"
            print(f"  [{i:2d}] {name:30s}  stored={ja[i]:.4f}")

    # Print actuator config
    print(f"\n[GraspViz] Actuator config:")
    for act_name, act in robot.actuators.items():
        print(f"  {act_name}: {type(act).__name__}")
        for attr in ["stiffness", "damping", "effort_limit", "velocity_limit"]:
            val = getattr(act, attr, None)
            if val is not None:
                if hasattr(val, 'mean'):
                    print(f"    {attr}: {val.mean().item():.2f} (tensor)")
                else:
                    print(f"    {attr}: {val}")

    # Print default joint position targets
    default_targets = robot.data.default_joint_pos[0]
    print(f"\n[GraspViz] Default joint position targets:")
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}] {name:30s}  default={default_targets[i].item():.4f}")

    # Pause physics so sim_app.update() only renders
    from omni.timeline import get_timeline_interface
    timeline = get_timeline_interface()
    timeline.pause()

    print(f"\n[GraspViz] Cycling through grasps every {args.interval}s")
    print(f"[GraspViz] Physics PAUSED — render only")
    print(f"[GraspViz] Press Ctrl+C to stop")

    last_switch = time.time()

    try:
        while sim_app.is_running():
            # Switch grasp periodically
            now = time.time()
            if now - last_switch >= args.interval:
                last_switch = now
                grasp = grasps[grasp_idx % len(grasps)]

                if grasp.joint_angles is not None:
                    # Set wrist to palm-up pose per env (offset by env_origins)
                    wrist_pos = palm_up_wrist_local.unsqueeze(0).expand(args.num_envs, -1).clone() \
                              + env.scene.env_origins[env_ids]
                    wrist_quat = palm_up_wrist_quat.unsqueeze(0).expand(args.num_envs, -1).clone()
                    set_robot_root_pose(env, env_ids, wrist_pos, wrist_quat)

                    # Set stored joints
                    joints_list = [grasp.joint_angles] * args.num_envs
                    set_robot_joints_direct(env, env_ids, joints_list)

                    # Move object away, step for FK, then place
                    temp_state = obj.data.default_root_state[env_ids].clone()
                    temp_state[:, :3] = env.scene.env_origins[env_ids] + torch.tensor([[0, 0, -10.0]], device=device)
                    temp_state[:, 7:] = 0.0
                    obj.write_root_state_to_sim(temp_state, env_ids=env_ids)
                    obj.update(0.0)
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)

                    # Place object at fingertip centroid
                    ft_pos_w = robot.data.body_pos_w[env_ids][:, ft_ids, :]
                    obj_pos_w = ft_pos_w.mean(dim=1)
                    obj_quat_w = robot.data.root_quat_w[env_ids].clone()

                    obj_state = obj.data.default_root_state[env_ids].clone()
                    obj_state[:, :3] = obj_pos_w
                    obj_state[:, 3:7] = obj_quat_w
                    obj_state[:, 7:] = 0.0
                    obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
                    obj.update(0.0)

                    # Re-pause physics after sim.step
                    timeline.pause()

                    # Print info
                    q = grasp.object_quat_hand
                    q_str = f"[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]" if q is not None else "None"
                    print(f"[GraspViz] Grasp {grasp_idx}/{len(grasps)}  "
                          f"quality={grasp.quality:.3f}  "
                          f"obj_quat_hand={q_str}  "
                          f"obj_pos={obj_pos_w[0].tolist()}")

                    for fi in range(len(ft_ids)):
                        print(f"  finger {fi}: {ft_pos_w[0, fi].tolist()}")

                grasp_idx += 1

            # Render only, no physics
            sim_app.update()

    except KeyboardInterrupt:
        print("[GraspViz] Done.")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
