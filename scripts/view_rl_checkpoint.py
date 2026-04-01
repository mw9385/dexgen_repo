"""
Stage 1 checkpoint viewer.

Loads a saved rl_games checkpoint and runs the DexGen AnyGrasp environment
in inference mode so the trained policy can be inspected visually.

Bypasses rl_games Runner/Player entirely to avoid observation_space shape
issues. Builds the MLP directly from checkpoint weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser(description="View a saved Stage 1 RL checkpoint")
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to rl_games checkpoint (.pt / .pth)",
    )
    p.add_argument(
        "--grasp_graph", action="append", default=None,
        help="Path to GraspGraph PKL. Repeat for multiple.",
    )
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=0,
                   help="Max sim steps (0 = run until window closed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
    )
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.grasp_graph is None:
        args.grasp_graph = ["data/grasp_graph.pkl"]
    return args


class PolicyMLP(nn.Module):
    """Reconstruct the actor MLP from rl_games checkpoint weights."""

    def __init__(self, checkpoint: dict):
        super().__init__()
        model_sd = checkpoint["model"]

        # Detect MLP layers: a2c_network.actor_mlp.0.weight, .2.weight, ...
        actor_keys = sorted(
            [k for k in model_sd if k.startswith("a2c_network.actor_mlp.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[2]),
        )

        layers = []
        for wk in actor_keys:
            bk = wk.replace(".weight", ".bias")
            w = model_sd[wk]
            b = model_sd[bk]
            linear = nn.Linear(w.shape[1], w.shape[0])
            linear.weight.data.copy_(w)
            linear.bias.data.copy_(b)
            layers.append(linear)
            layers.append(nn.ELU())

        # Output head (mu): a2c_network.mu.weight / .bias
        mu_w = model_sd["a2c_network.mu.weight"]
        mu_b = model_sd["a2c_network.mu.bias"]
        mu = nn.Linear(mu_w.shape[1], mu_w.shape[0])
        mu.weight.data.copy_(mu_w)
        mu.bias.data.copy_(mu_b)
        layers.append(mu)

        self.net = nn.Sequential(*layers)

        # Running mean/std for input normalization
        self.running_mean = checkpoint.get("running_mean_std", {}).get("running_mean", None)
        self.running_var = checkpoint.get("running_mean_std", {}).get("running_var", None)
        if self.running_mean is not None:
            self.running_mean = self.running_mean.float()
            self.running_var = self.running_var.float()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        if self.running_mean is not None:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
            x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return self.net(x)


def _to_policy_obs(obs):
    """Extract policy observation tensor from Isaac Lab obs dict."""
    if isinstance(obs, dict):
        return obs.get("policy", next(iter(obs.values())))
    return obs


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
        _IsaacLabVecEnv,
        _resolve_grasp_graph_arg,
        load_config,
        apply_env_config,
        apply_dr_config,
    )
    from envs.mdp import events as mdp_events
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

    # ── Build environment ──
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
                graph_num_fingers, tip_subsets[5][:graph_num_fingers],
            )
    except Exception as exc:
        print(f"[Viewer] WARNING: Could not load graph metadata: {exc}")

    apply_env_config(env_cfg, cfg_file.get("env", {}))
    apply_dr_config(env_cfg, cfg_file.get("domain_randomization", {}))

    env = ManagerBasedRLEnv(env_cfg)

    # ── Load policy directly from checkpoint ──
    device = env.device
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    policy = PolicyMLP(ckpt).to(device).eval()

    print(f"[Viewer] Checkpoint: {checkpoint_path}")
    print(f"[Viewer] Num envs: {args.num_envs}")
    print(f"[Viewer] Grasp graph(s): {', '.join(grasp_graph_paths)}")
    print(f"[Viewer] Policy: {sum(p.numel() for p in policy.parameters())} params")

    # ── Init extras for action tracking ──
    obs, _ = env.reset()
    num_dof = env.action_manager.action.shape[-1]
    env.extras["last_action"] = torch.zeros(args.num_envs, num_dof, device=device)
    env.extras["current_action"] = torch.zeros(args.num_envs, num_dof, device=device)

    steps = 0
    while sim_app.is_running():
        with torch.inference_mode():
            obs_tensor = _to_policy_obs(obs)
            actions = policy(obs_tensor)
            actions = actions.clamp(-1.0, 1.0)

            env.extras["last_action"] = env.extras["current_action"].clone()
            env.extras["current_action"] = actions.clone()

            obs, _, terminated, truncated, info = env.step(actions)
            mdp_events.update_rolling_goal(env, success_threshold=0.02)

        steps += 1
        if args.num_steps > 0 and steps >= args.num_steps:
            break

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
