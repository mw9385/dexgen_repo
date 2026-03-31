"""
Environment visualizer — zero / random actions.

Launches the AnyGrasp environment without a trained policy so you can inspect:
  - Reset behavior (hand + object placement)
  - Zero-action response (does the object fall? do joints hold?)
  - Random-action response

Usage:
    # Zero actions (default) — check reset quality
    /workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
        --grasp_graph data/grasp_graph.pkl --num_envs 4

    # Random actions
    /workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
        --grasp_graph data/grasp_graph.pkl --num_envs 4 --action_mode random

    # Hold reset pose for N steps before stepping
    /workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
        --grasp_graph data/grasp_graph.pkl --hold_steps 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser(description="Visualize AnyGrasp env with zero/random actions")
    p.add_argument("--grasp_graph", action="append", default=None,
                   help="Path to grasp_graph.pkl (repeat for multiple)")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=0,
                   help="Max steps to run (0 = run until window closed)")
    p.add_argument("--action_mode", type=str, default="zero",
                   choices=["zero", "random", "hold"],
                   help="zero: all-zero actions | random: uniform [-1,1] | hold: maintain initial grasp pose")
    p.add_argument("--hold_steps", type=int, default=0,
                   help="Send zero actions for this many steps before switching to action_mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"))
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.grasp_graph is None:
        args.grasp_graph = ["data/grasp_graph.pkl"]

    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    try:
        from isaaclab.envs import ManagerBasedRLEnv
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sim_app.close()
        sys.exit(1)

    from envs import AnyGraspEnvCfg, register_anygrasp_env
    from grasp_generation.graph_io import load_merged_graph
    from envs.anygrasp_env import _build_object_spawner
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

    try:
        import yaml
        with open(args.config) as f:
            cfg_file = yaml.safe_load(f) or {}
    except Exception:
        cfg_file = {}

    register_anygrasp_env()

    # Load grasp graph
    missing = [p for p in args.grasp_graph if not Path(p).exists()]
    if missing:
        print(f"ERROR: GraspGraph not found: {missing}")
        sim_app.close()
        sys.exit(1)

    merged_graph = load_merged_graph(args.grasp_graph)

    # Build env config
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = args.grasp_graph
    if getattr(args, "headless", False):
        env_cfg.viewer = None

    # Apply object pool from graph
    if isinstance(merged_graph, MultiObjectGraspGraph) and merged_graph.object_specs:
        specs = list(merged_graph.object_specs.values())
        env_cfg.scene.object = env_cfg.scene.object.replace(spawn=_build_object_spawner(specs))
        print(f"[Viz] Loaded {len(specs)} object spec(s) from grasp graph")

    # Apply num_fingers from graph
    graph_num_fingers = None
    if isinstance(merged_graph, MultiObjectGraspGraph) and merged_graph.graphs:
        first = next(iter(merged_graph.graphs.values()))
        graph_num_fingers = getattr(first, "num_fingers", None)
    if graph_num_fingers is not None:
        tip_subsets = {
            2: ["robot0_ffdistal", "robot0_thdistal"],
            3: ["robot0_ffdistal", "robot0_mfdistal", "robot0_thdistal"],
            4: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_thdistal"],
            5: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"],
        }
        env_cfg.hand = dict(getattr(env_cfg, "hand", {}) or {})
        env_cfg.hand["num_fingers"] = int(graph_num_fingers)
        env_cfg.hand["fingertip_links"] = tip_subsets.get(int(graph_num_fingers), tip_subsets[5])

    # Apply env/DR config from yaml
    from scripts.train_rl import apply_env_config, apply_dr_config
    apply_env_config(env_cfg, cfg_file.get("env", {}))
    apply_dr_config(env_cfg, cfg_file.get("domain_randomization", {}))

    env = ManagerBasedRLEnv(env_cfg)
    obs, _ = env.reset()

    rng = torch.Generator(device=env.unwrapped.device)
    rng.manual_seed(args.seed)

    action_dim = env.unwrapped.action_manager.action.shape[-1]
    print(f"[Viz] action_dim={action_dim}, action_mode={args.action_mode}, "
          f"hold_steps={args.hold_steps}")
    print(f"[Viz] Press Ctrl+C or close the window to stop")

    steps = 0
    try:
        while sim_app.is_running():
            with torch.inference_mode():
                if args.action_mode == "hold" or (steps < args.hold_steps):
                    actions = torch.zeros(args.num_envs, action_dim,
                                         device=env.unwrapped.device)
                elif args.action_mode == "zero":
                    actions = torch.zeros(args.num_envs, action_dim,
                                         device=env.unwrapped.device)
                else:
                    actions = torch.empty(args.num_envs, action_dim,
                                          device=env.unwrapped.device).uniform_(-1, 1,
                                          generator=rng)

                obs, reward, terminated, truncated, info = env.step(actions)

            steps += 1
            if steps % 100 == 0:
                print(f"[Viz] step={steps}  reward_mean={reward.mean():.4f}")
            if args.num_steps > 0 and steps >= args.num_steps:
                print(f"[Viz] Reached {args.num_steps} steps, stopping.")
                break
    except KeyboardInterrupt:
        print("[Viz] Interrupted.")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
