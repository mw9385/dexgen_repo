"""
Stage 1 checkpoint viewer.

Loads a saved rl_games checkpoint and runs the DexGen AnyGrasp environment
in inference mode so the trained policy can be inspected visually.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser(description="View a saved Stage 1 RL checkpoint")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to rl_games checkpoint (model_*.pt)",
    )
    p.add_argument(
        "--grasp_graph",
        action="append",
        default=None,
        help="Path to GraspGraph from Stage 0. Repeat the flag or use comma-separated values to load multiple PKL files.",
    )
    p.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate")
    p.add_argument("--num_steps", type=int, default=0, help="Optional max number of sim steps (0 = run until window closed)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
        help="Path to YAML config",
    )
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
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
        print(f"ERROR: missing runtime dependency: {exc}")
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
    missing_graph_paths = [path for path in grasp_graph_paths if not Path(path).exists()]
    if missing_graph_paths:
        print(f"ERROR: GraspGraph not found at {missing_graph_paths}")
        sim_app.close()
        sys.exit(1)

    cfg_file = load_config(args.config)
    merged_graph = load_merged_graph(grasp_graph_paths)

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
            env_cfg.scene.object = env_cfg.scene.object.replace(spawn=_build_object_spawner(specs))
            print(f"[Viewer] Loaded {len(specs)} object spec(s) from grasp graph.")

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
                graph_num_fingers,
                tip_subsets[5][:graph_num_fingers],
            )
    except Exception as exc:
        print(f"[Viewer] WARNING: Could not load graph metadata: {exc}")

    apply_env_config(env_cfg, cfg_file.get("env", {}))
    apply_dr_config(env_cfg, cfg_file.get("domain_randomization", {}))

    def create_env(**kwargs):
        return ManagerBasedRLEnv(env_cfg)

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": create_env,
        },
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

    print(f"[Viewer] Checkpoint: {checkpoint_path}")
    print(f"[Viewer] Num envs: {args.num_envs}")
    print(f"[Viewer] Grasp graph(s): {', '.join(grasp_graph_paths)}")

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

    steps = 0
    while sim_app.is_running():
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                done_mask = dones.bool() if isinstance(dones, torch.Tensor) else torch.as_tensor(dones, dtype=torch.bool)
                for state in agent.states:
                    state[:, done_mask, :] = 0.0
        steps += 1
        if args.num_steps > 0 and steps >= args.num_steps:
            break

    try:
        env.env.close()
    except Exception:
        pass
    sim_app.close()


if __name__ == "__main__":
    main()
