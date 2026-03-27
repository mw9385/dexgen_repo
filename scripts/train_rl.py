"""
Stage 1 – RL Policy Training
==============================
Trains an AnyGrasp-to-AnyGrasp policy using PPO (via rl_games)
on the Isaac Lab environment.

The trained policy is used in Stage 2 to collect a dataset of
successful grasp transitions for DexGen controller training.

Usage:
    python scripts/train_rl.py \
        --grasp_graph data/grasp_graph.pkl \
        --num_envs 512 \
        --max_iterations 30000 \
        --headless

    # Resume from checkpoint
    python scripts/train_rl.py \
        --resume logs/rl/allegro_anygrasp/checkpoints/model_10000.pt
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 1: RL Training")
    p.add_argument("--grasp_graph", type=str, default="data/grasp_graph.pkl",
                   help="Path to GraspGraph from Stage 0")
    p.add_argument("--num_envs", type=int, default=512,
                   help="Number of parallel environments")
    p.add_argument("--max_iterations", type=int, default=30000,
                   help="Maximum PPO training iterations")
    p.add_argument("--headless", action="store_true", default=False,
                   help="Run without rendering")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--log_dir", type=str, default="logs/rl/allegro_anygrasp",
                   help="Training log / checkpoint directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
                   help="Path to YAML config (ppo / domain_randomization settings)")
    return p.parse_args()


def load_config(path: str) -> dict:
    """Load YAML config, return empty dict if file not found."""
    p = Path(path)
    if not p.exists():
        print(f"[WARNING] Config not found at {path}, using defaults.")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def apply_dr_config(env_cfg, dr_cfg: dict):
    # (기존 코드와 동일)
    if not dr_cfg:
        return

    events = env_cfg.events

    obj = dr_cfg.get("object_physics", {})
    if obj:
        p = events.randomize_object_physics.params
        if "mass_range"        in obj: p["mass_range"]        = tuple(obj["mass_range"])
        if "friction_range"    in obj: p["friction_range"]    = tuple(obj["friction_range"])
        if "restitution_range" in obj: p["restitution_range"] = tuple(obj["restitution_range"])

    rob = dr_cfg.get("robot_physics", {})
    if rob:
        p = events.randomize_robot_physics.params
        if "damping_range"  in rob: p["damping_range"]  = tuple(rob["damping_range"])
        if "armature_range" in rob: p["armature_range"] = tuple(rob["armature_range"])

    delay = dr_cfg.get("action_delay", {})
    if delay:
        p = events.randomize_action_delay.params
        if "max_delay" in delay: p["max_delay"] = int(delay["max_delay"])

    noise = dr_cfg.get("obs_noise", {})
    if noise:
        obs = env_cfg.observations
        if "joint_pos_std"     in noise:
            obs.policy.joint_pos.noise.std     = float(noise["joint_pos_std"])
            obs.critic.joint_pos.noise.std     = float(noise["joint_pos_std"])
        if "joint_vel_std"     in noise:
            obs.policy.joint_vel.noise.std     = float(noise["joint_vel_std"])
            obs.critic.joint_vel.noise.std     = float(noise["joint_vel_std"])
        if "fingertip_pos_std" in noise:
            obs.policy.fingertip_pos.noise.std = float(noise["fingertip_pos_std"])
            obs.critic.fingertip_pos.noise.std = float(noise["fingertip_pos_std"])


def build_rl_games_config(args) -> dict:
    # (기존 코드와 동일)
    return {
        "params": {
            "seed": args.seed,
            "algo": {
                "name": "a2c_continuous",
            },
            "model": {
                "name": "continuous_a2c_logstd",
            },
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": [512, 512, 256, 128],
                    "activation": "elu",
                    "d2rl": False,
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"},
                },
            },
            "load_checkpoint": args.resume is not None,
            "load_path": args.resume or "",
            "config": {
                "name": "DexGen-AnyGrasp-Allegro",
                "env_name": "rlgpu",
                "device": args.device,
                "device_name": args.device,
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "num_actors": args.num_envs,
                "reward_shaper": {
                    "scale_value": 0.01,
                },
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,               
                "learning_rate": 5e-4,
                "lr_schedule": "adaptive",
                "lr_threshold": 0.008,
                "score_to_win": 20000,
                "max_epochs": args.max_iterations,
                "save_best_after": 100,
                "save_frequency": 1000,
                "print_stats": True,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,             
                "num_steps_per_env": 8,    
                "mini_epochs": 5,
                "minibatch_size": 16384,
                "critic_coef": 4,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 0.0001,
                "log_dir": args.log_dir,
            },
        }
    }


def main():
    args = parse_args()

    # ── MUST set Isaac Sim cloud asset root BEFORE any isaaclab import ──────
    # isaaclab/utils/assets.py computes NUCLEUS_ASSET_ROOT_DIR at module-
    # import time via  carb.settings.get("/persistent/isaac/asset_root/cloud").
    # In a headless container this setting is unset (None), which makes every
    # asset USD path resolve to "None/Isaac/..." → FileNotFoundError.
    # Setting it here (before `import isaaclab`) ensures the first lazy import
    # of isaaclab.utils.assets picks up the correct S3 URL.
    try:
        import carb as _carb
        _cs = _carb.settings.get_settings()
        if not _cs.get("/persistent/isaac/asset_root/cloud"):
            _cs.set(
                "/persistent/isaac/asset_root/cloud",
                "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
                "/Assets/Isaac/5.0",
            )
            print("[train_rl] Set Isaac Sim cloud asset root to S3 (5.0).")
    except Exception as _e:
        print(f"[train_rl] WARNING: could not set carb asset root: {_e}")
    # ────────────────────────────────────────────────────────────────────────

    # Validate grasp graph exists
    if not Path(args.grasp_graph).exists():
        print(f"ERROR: GraspGraph not found at {args.grasp_graph}")
        print("Run Stage 0 first:")
        print("  python scripts/run_grasp_generation.py")
        sim_app.close()
        sys.exit(1)

    # Import Isaac Lab dependencies (런타임 초기화가 완료된 이후에 import)
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        print("ERROR: Isaac Lab not found. Run ./setup_isaaclab.sh first.")
        sim_app.close()
        sys.exit(1)

    try:
        import rl_games  # noqa: F401
    except ImportError:
        print("ERROR: rl_games not found. Run: pip install rl_games")
        sim_app.close()
        sys.exit(1)

    from isaaclab.envs import ManagerBasedRLEnv
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from envs import AnyGraspEnvCfg, register_anygrasp_env

    # Register environment
    register_anygrasp_env()

    # Load YAML config
    cfg_file = load_config(args.config)

    # Build environment config
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = args.grasp_graph
    if args.headless:
        env_cfg.viewer = None

    # Apply domain-randomization
    apply_dr_config(env_cfg, cfg_file.get("domain_randomization", {}))

    print(f"[Stage 1] Config:   {args.config}")
    print(f"[Stage 1] Task: DexGen-AnyGrasp-Allegro-v0")
    print(f"[Stage 1] Num envs: {args.num_envs}")
    print(f"[Stage 1] Max iterations: {args.max_iterations}")
    print(f"[Stage 1] Grasp graph: {args.grasp_graph}")
    print(f"[Stage 1] Log dir: {args.log_dir}")
    print("-" * 60)

    # rl_games env wrapper
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

    # Build and run rl_games trainer
    cfg = build_rl_games_config(args)
    runner = Runner(IsaacAlgoObserver())
    runner.load(cfg)
    runner.reset()
    runner.run({"train": True})

    print(f"\n=== Stage 1 Complete ===")
    print(f"Checkpoints saved to: {args.log_dir}")
    print(f"\nNext: collect dataset")
    print(f"  python scripts/collect_data.py --log_dir {args.log_dir}")
    
    # 런타임 종료
    sim_app.close()


def _to_rl_obs(obs):
    """
    Convert Isaac Lab observation output to rl_games format.

    Isaac Lab returns a dict keyed by obs-group name, e.g.
      {"policy": Tensor(N, 76), "critic": Tensor(N, 104)}

    rl_games expects:
      {"obs": Tensor(N, obs_dim)}          — actor observations
      (asymmetric-AC also uses "states", but we use a shared network here)
    """
    if isinstance(obs, dict):
        policy_obs = obs.get("policy", next(iter(obs.values())))
        return {"obs": policy_obs}
    return {"obs": obs}


class _IsaacLabVecEnv:
    """Thin wrapper to make Isaac Lab ManagerBasedRLEnv compatible with rl_games."""

    def __init__(self, env, num_envs: int):
        self.env = env
        self.num_envs = num_envs

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated
        return _to_rl_obs(obs), rew, done, info

    def reset(self):
        obs, _ = self.env.reset()
        return _to_rl_obs(obs)

    def get_number_of_agents(self):
        return self.num_envs

    def get_env_info(self):
        import gymnasium as gym

        action_space = self.env.action_space

        # Isaac Lab may return a Dict obs space (one Box per obs group).
        # rl_games needs a flat Box for the policy group.
        raw_obs_space = self.env.observation_space
        if isinstance(raw_obs_space, gym.spaces.Dict):
            obs_space = raw_obs_space.spaces.get(
                "policy",
                next(iter(raw_obs_space.spaces.values())),
            )
        else:
            obs_space = raw_obs_space

        return {
            "action_space": action_space,
            "observation_space": obs_space,
        }


if __name__ == "__main__":
    main()
