"""
Evaluate a trained RL policy checkpoint.

Runs the policy in the AnyGrasp environment and reports quantitative metrics:
  - Per-episode success rate (fingertip proximity to goal)
  - Mean fingertip tracking error
  - Rolling goal update count
  - Reward breakdown by term
  - Object drop / left-hand termination rates

Can run headless (default) for fast evaluation, or with viewer for visual inspection.

Usage:
    # Headless evaluation (fast, metrics only)
    /isaac-sim/python.sh scripts/evaluate_policy.py \
        --checkpoint logs/rl/.../model_30000.pt \
        --num_episodes 100

    # Visual inspection (opens viewer window)
    /isaac-sim/python.sh scripts/evaluate_policy.py \
        --checkpoint logs/rl/.../model_30000.pt \
        --num_episodes 20 --no-headless

    # Custom grasp graph
    /isaac-sim/python.sh scripts/evaluate_policy.py \
        --checkpoint logs/rl/.../model_30000.pt \
        --grasp_graph data/grasp_graph.pkl \
        --num_episodes 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Stage 1 RL checkpoint")
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to rl_games checkpoint (model_*.pt)",
    )
    p.add_argument(
        "--grasp_graph", action="append", default=None,
        help="Path to GraspGraph PKL. Repeat for multiple.",
    )
    p.add_argument("--num_envs", type=int, default=64,
                   help="Parallel environments")
    p.add_argument("--num_episodes", type=int, default=100,
                   help="Total episodes to evaluate")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic actions (default: True)")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
    )
    AppLauncher.add_app_launcher_args(p)

    # Default to headless
    raw_args = sys.argv[1:]
    if "--headless" not in raw_args and "--no-headless" not in raw_args:
        raw_args.append("--headless")

    args = p.parse_args(raw_args)
    if args.grasp_graph is None:
        args.grasp_graph = ["data/grasp_graph.pkl"]
    return args


def main():
    args = parse_args()

    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    try:
        import carb as _carb
        _cs = _carb.settings.get_settings()
        if not _cs.get("/persistent/isaac/asset_root/cloud"):
            _cs.set(
                "/persistent/isaac/asset_root/cloud",
                "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
                "/Assets/Isaac/5.0",
            )
    except Exception:
        pass

    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.common.player import BasePlayer
        from rl_games.torch_runner import Runner
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}")
        sim_app.close()
        sys.exit(1)

    from envs import AnyGraspEnvCfg, register_anygrasp_env
    from scripts.train_rl import (
        _IsaacLabVecEnv,
        _resolve_grasp_graph_arg,
        build_rl_games_config,
        load_config,
        apply_env_config,
        apply_dr_config,
    )
    from grasp_generation.graph_io import load_merged_graph

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sim_app.close()
        sys.exit(1)

    register_anygrasp_env()

    grasp_graph_paths = _resolve_grasp_graph_arg(args)
    for gp in grasp_graph_paths:
        if not Path(gp).exists():
            print(f"ERROR: GraspGraph not found at {gp}")
            sim_app.close()
            sys.exit(1)

    cfg_file = load_config(args.config)
    merged_graph = load_merged_graph(grasp_graph_paths)

    # ── Environment setup ──
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = grasp_graph_paths
    if getattr(args, "headless", False):
        env_cfg.viewer = None

    try:
        from grasp_generation.rrt_expansion import MultiObjectGraspGraph
        from envs.anygrasp_env import _build_object_spawner

        if isinstance(merged_graph, MultiObjectGraspGraph) and merged_graph.object_specs:
            specs = list(merged_graph.object_specs.values())
            env_cfg.scene.object = env_cfg.scene.object.replace(
                spawn=_build_object_spawner(specs)
            )

        graph_num_fingers = getattr(merged_graph, "num_fingers", None)
        if graph_num_fingers is None and isinstance(merged_graph, MultiObjectGraspGraph) and merged_graph.graphs:
            first_graph = next(iter(merged_graph.graphs.values()))
            graph_num_fingers = getattr(first_graph, "num_fingers", None)
        if graph_num_fingers is not None:
            graph_num_fingers = int(graph_num_fingers)
            tip_subsets = {
                2: ["robot0_ffdistal", "robot0_thdistal"],
                3: ["robot0_ffdistal", "robot0_mfdistal", "robot0_thdistal"],
                4: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_thdistal"],
                5: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"],
            }
            env_cfg.hand = dict(getattr(env_cfg, "hand", {}) or {})
            env_cfg.hand["num_fingers"] = graph_num_fingers
            env_cfg.hand["fingertip_links"] = tip_subsets.get(
                graph_num_fingers, tip_subsets[5][:graph_num_fingers],
            )
    except Exception as exc:
        print(f"[Eval] WARNING: Could not load graph metadata: {exc}")

    # Disable DR for clean evaluation
    env_dr = cfg_file.get("domain_randomization", {})
    env_dr_clean = {}
    for section in env_dr:
        if section == "obs_noise":
            env_dr_clean[section] = {k: 0.0 for k in env_dr[section]}
        elif section == "action_delay":
            env_dr_clean[section] = {"max_delay": 0}
        else:
            env_dr_clean[section] = env_dr[section]

    apply_env_config(env_cfg, cfg_file.get("env", {}))
    apply_dr_config(env_cfg, env_dr_clean)

    def create_env(**kwargs):
        return ManagerBasedRLEnv(env_cfg)

    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "RLGPU", "env_creator": create_env},
    )
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: _IsaacLabVecEnv(
            create_env(), num_actors
        ),
    )

    cfg = build_rl_games_config(args, cfg_file)
    cfg["params"]["load_checkpoint"] = True
    cfg["params"]["load_path"] = str(checkpoint_path)
    cfg["params"]["config"]["num_actors"] = args.num_envs

    # ── Load policy ──
    runner = Runner(IsaacAlgoObserver())
    runner.load(cfg)
    runner.reset()

    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint_path))
    agent.reset()

    env = agent.env
    obs = env.reset()
    if agent.is_rnn:
        agent.init_rnn()
    _ = agent.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)

    # ── Metric accumulators ──
    total_episodes = 0
    total_success = 0
    total_goal_updates = 0
    total_drops = 0
    total_left_hand = 0
    fingertip_errors = []
    episode_rewards = []
    step_count = 0

    print(f"\n{'='*60}")
    print(f"  Policy Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint:     {checkpoint_path}")
    print(f"  Grasp graph(s): {', '.join(grasp_graph_paths)}")
    print(f"  Num envs:       {args.num_envs}")
    print(f"  Target episodes:{args.num_episodes}")
    print(f"  Deterministic:  {args.deterministic}")
    print(f"  DR:             disabled (clean eval)")
    print(f"{'='*60}\n")

    t_start = time.time()

    while total_episodes < args.num_episodes and sim_app.is_running():
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=args.deterministic)
            obs, rew, dones, info = env.step(actions)

            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                done_mask = dones.bool() if isinstance(dones, torch.Tensor) else torch.as_tensor(dones, dtype=torch.bool)
                for state in agent.states:
                    state[:, done_mask, :] = 0.0

        step_count += 1

        # Track per-step info
        if isinstance(info, dict):
            total_goal_updates += info.get("rolling_goal_updates", 0)

            # Count completed episodes from done signals
            if isinstance(dones, torch.Tensor):
                n_done = int(dones.sum().item())
            else:
                n_done = int(np.sum(dones))

            if n_done > 0:
                total_episodes += n_done

                sr = info.get("success_ratio", 0.0)
                total_success += sr * n_done

                dr = info.get("drop_ratio", 0.0)
                total_drops += dr * n_done

                lr = info.get("left_hand_ratio", 0.0)
                total_left_hand += lr * n_done

            # Track fingertip error from observations
            inner_env = env.env if hasattr(env, "env") else env
            if hasattr(inner_env, "extras"):
                target_fp = inner_env.extras.get("target_fingertip_pos")
                current_fp = inner_env.extras.get("current_fingertip_pos")
                if target_fp is not None and current_fp is not None:
                    err = (target_fp - current_fp).norm(dim=-1).mean().item()
                    fingertip_errors.append(err)

        # Progress print
        if total_episodes > 0 and total_episodes % (args.num_envs * 5) < args.num_envs:
            elapsed = time.time() - t_start
            sr = total_success / max(total_episodes, 1)
            print(f"  [{total_episodes:>5d}/{args.num_episodes}]  "
                  f"success={sr:.1%}  "
                  f"goal_updates={total_goal_updates}  "
                  f"drops={total_drops:.0f}  "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t_start

    # ── Results ──
    success_rate = total_success / max(total_episodes, 1)
    drop_rate = total_drops / max(total_episodes, 1)
    left_hand_rate = total_left_hand / max(total_episodes, 1)
    mean_fp_err = np.mean(fingertip_errors) if fingertip_errors else float("nan")
    goals_per_ep = total_goal_updates / max(total_episodes, 1)

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Episodes evaluated:      {total_episodes}")
    print(f"  Total sim steps:         {step_count}")
    print(f"  Wall time:               {elapsed:.1f}s")
    print(f"  Steps/sec:               {step_count * args.num_envs / elapsed:.0f}")
    print(f"{'─'*60}")
    print(f"  Success rate:            {success_rate:.1%}")
    print(f"  Mean fingertip error:    {mean_fp_err * 1000:.1f} mm")
    print(f"  Rolling goal updates/ep: {goals_per_ep:.2f}")
    print(f"  Drop rate:               {drop_rate:.1%}")
    print(f"  Left-hand rate:          {left_hand_rate:.1%}")
    print(f"{'='*60}\n")

    try:
        env.env.close()
    except Exception:
        pass
    sim_app.close()


if __name__ == "__main__":
    main()
