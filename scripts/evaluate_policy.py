"""
Evaluate a trained RL policy checkpoint.

Runs the policy in the AnyGrasp environment and reports quantitative metrics:
  - Per-episode success rate (fingertip proximity to goal)
  - Mean fingertip tracking error
  - Rolling goal update count
  - Object drop / left-hand termination rates

Bypasses rl_games Runner/Player entirely — loads MLP weights directly
from checkpoint. No tensorboard logging, results printed to stdout.

Usage:
    # Headless evaluation (fast, metrics only)
    /isaac-sim/python.sh scripts/evaluate_policy.py \
        --checkpoint logs/rl/.../model_30000.pt \
        --num_episodes 100

    # Visual inspection (opens viewer window)
    /isaac-sim/python.sh scripts/evaluate_policy.py \
        --checkpoint logs/rl/.../model_30000.pt \
        --num_episodes 20 --no-headless
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
        help="Path to rl_games checkpoint (.pt / .pth)",
    )
    p.add_argument(
        "--grasp_graph", action="append", default=None,
        help="Path to GraspGraph PKL. Repeat for multiple.",
    )
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
    )
    AppLauncher.add_app_launcher_args(p)

    # Default to headless unless user explicitly passes --headless (which is a flag, not negatable)
    raw_args = sys.argv[1:]
    if "--headless" not in raw_args:
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

    from isaaclab.envs import ManagerBasedRLEnv
    from envs import AnyGraspEnvCfg, register_anygrasp_env
    from scripts.train_rl import (
        _resolve_grasp_graph_arg,
        load_config,
        apply_env_config,
        apply_dr_config,
    )
    from scripts.view_rl_checkpoint import PolicyMLP, _to_policy_obs
    from envs.mdp import events as mdp_events
    from envs.mdp import rewards as mdp_rewards
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

    env = ManagerBasedRLEnv(env_cfg)

    # ── Load policy directly from checkpoint ──
    device = env.device
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    policy = PolicyMLP(ckpt).to(device).eval()

    # ── Init ──
    obs, _ = env.reset()
    num_dof = env.action_manager.action.shape[-1]
    env.extras["last_action"] = torch.zeros(args.num_envs, num_dof, device=device)
    env.extras["current_action"] = torch.zeros(args.num_envs, num_dof, device=device)

    # ── Metric accumulators ──
    total_episodes = 0
    total_success = 0
    total_goal_updates = 0
    total_drops = 0
    total_left_hand = 0
    fingertip_errors = []
    step_count = 0

    print(f"\n{'='*60}")
    print(f"  Policy Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint:     {checkpoint_path}")
    print(f"  Grasp graph(s): {', '.join(grasp_graph_paths)}")
    print(f"  Num envs:       {args.num_envs}")
    print(f"  Target episodes:{args.num_episodes}")
    print(f"  Policy params:  {sum(p.numel() for p in policy.parameters())}")
    print(f"  DR:             disabled (clean eval)")
    print(f"{'='*60}\n")

    t_start = time.time()

    while total_episodes < args.num_episodes and sim_app.is_running():
        with torch.inference_mode():
            obs_tensor = _to_policy_obs(obs)
            actions = policy(obs_tensor).clamp(-1.0, 1.0)

            env.extras["last_action"] = env.extras["current_action"].clone()
            env.extras["current_action"] = actions.clone()

            obs, rew, terminated, truncated, info = env.step(actions)
            done = terminated | truncated

            n_updated = mdp_events.update_rolling_goal(env, success_threshold=0.02)
            total_goal_updates += n_updated

        step_count += 1

        # Count done episodes
        n_done = int(done.sum().item())
        if n_done > 0:
            total_episodes += n_done

            sr = float(mdp_rewards.grasp_success_reward(env).mean().item())
            total_success += sr * n_done

            # Termination stats
            term_manager = getattr(env, "termination_manager", None)
            if term_manager is not None:
                active = set(getattr(term_manager, "active_terms", []))
                if "object_drop" in active:
                    total_drops += float(term_manager.get_term("object_drop").float().mean().item()) * n_done
                if "object_left_hand" in active:
                    total_left_hand += float(term_manager.get_term("object_left_hand").float().mean().item()) * n_done

        # Track fingertip error
        target_fp = env.extras.get("target_fingertip_pos")
        current_fp = env.extras.get("current_fingertip_pos")
        if target_fp is not None and current_fp is not None:
            err = (target_fp - current_fp).norm(dim=-1).mean().item()
            fingertip_errors.append(err)

        # Progress
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

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
