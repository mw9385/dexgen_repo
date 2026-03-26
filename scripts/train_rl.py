"""
Stage 1 – RL Policy Training
==============================
Trains an AnyGrasp-to-AnyGrasp policy using PPO (via rl_games)
on the Isaac Lab environment.

The trained policy is used in Stage 2 to collect a dataset of
successful grasp transitions for DexGen controller training.

Usage:
    python scripts/train_rl.py \\
        --grasp_graph data/grasp_graph.pkl \\
        --num_envs 512 \\
        --max_iterations 30000 \\
        --headless

    # Resume from checkpoint
    python scripts/train_rl.py \\
        --resume logs/rl/allegro_anygrasp/checkpoints/model_10000.pt
"""

import argparse
import os
import sys
from pathlib import Path

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
    return p.parse_args()


def build_rl_games_config(args) -> dict:
    """
    Build rl_games PPO config dict for the AnyGrasp-to-AnyGrasp task.
    Based on Isaac Lab's AllegroHand config with tuned hyperparameters.
    """
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
                # PPO hyperparameters (tuned for dexterous manipulation)
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
                "tau": 0.95,               # GAE lambda
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
                "e_clip": 0.2,             # PPO clip ratio
                "num_steps_per_env": 8,    # rollout horizon
                "mini_epochs": 5,
                "minibatch_size": 16384,
                "critic_coef": 4,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 0.0001,
                # Logging
                "log_dir": args.log_dir,
            },
        }
    }


def main():
    args = parse_args()

    # Validate grasp graph exists
    if not Path(args.grasp_graph).exists():
        print(f"ERROR: GraspGraph not found at {args.grasp_graph}")
        print("Run Stage 0 first:")
        print("  python scripts/run_grasp_generation.py")
        sys.exit(1)

    # Import Isaac Lab (must be installed)
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        print("ERROR: Isaac Lab not found. Run ./setup_isaaclab.sh first.")
        sys.exit(1)

    try:
        import rl_games  # noqa: F401
    except ImportError:
        print("ERROR: rl_games not found. Run: pip install rl_games")
        sys.exit(1)

    from isaaclab.envs import ManagerBasedRLEnv
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from envs import AnyGraspEnvCfg, register_anygrasp_env

    # Register environment
    register_anygrasp_env()

    # Build environment config
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = args.grasp_graph
    if args.headless:
        env_cfg.viewer = None

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


class _IsaacLabVecEnv:
    """Thin wrapper to make Isaac Lab env compatible with rl_games VecEnv API."""

    def __init__(self, env, num_envs: int):
        self.env = env
        self.num_envs = num_envs

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated
        return obs, rew, done, info

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def get_number_of_agents(self):
        return self.num_envs

    def get_env_info(self):
        return {
            "action_space": self.env.action_space,
            "observation_space": self.env.observation_space,
        }


if __name__ == "__main__":
    main()
