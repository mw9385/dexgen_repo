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
        --headless

    # Resume from checkpoint
    python scripts/train_rl.py \
        --resume logs/rl/allegro_anygrasp_v2/checkpoints/model_10000.pt
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher
from grasp_generation.graph_io import load_merged_graph, parse_graph_paths


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 1: RL Training")
    p.add_argument(
        "--grasp_graph",
        action="append",
        default=None,
        help="Path to GraspGraph from Stage 0. Repeat the flag or use comma-separated values to load multiple PKL files.",
    )
    p.add_argument("--num_envs", type=int, default=512,
                   help="Number of parallel environments")
    p.add_argument("--max_iterations", type=int, default=30000,
                   help="Maximum PPO training iterations")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--log_dir", type=str, default="logs/rl/allegro_anygrasp_v2",
                   help="Training log / checkpoint directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
                   help="Path to YAML config (ppo / domain_randomization settings)")

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.grasp_graph is None:
        args.grasp_graph = ["data/grasp_graph.pkl"]
    return args


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[WARNING] Config not found at {path}, using defaults.")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _resolve_grasp_graph_arg(args) -> list[str]:
    graph_paths = parse_graph_paths(args.grasp_graph)
    if not graph_paths:
        raise ValueError("At least one --grasp_graph path is required.")
    return graph_paths


def apply_dr_config(env_cfg, dr_cfg: dict):
    if not dr_cfg:
        return

    events = env_cfg.events

    obj = dr_cfg.get("object_physics", {})
    if obj and hasattr(events, "randomize_object_physics"):
        p = events.randomize_object_physics.params
        new_params = {}
        if "mass_range"        in obj: new_params["mass_range"]        = tuple(obj["mass_range"])
        if "friction_range"    in obj: new_params["friction_range"]    = tuple(obj["friction_range"])
        if "restitution_range" in obj: new_params["restitution_range"] = tuple(obj["restitution_range"])
        p.update(new_params)

    rob = dr_cfg.get("robot_physics", {})
    if rob and hasattr(events, "randomize_robot_physics"):
        p = events.randomize_robot_physics.params
        new_params = {}
        if "damping_range"  in rob: new_params["damping_range"]  = tuple(rob["damping_range"])
        if "armature_range" in rob: new_params["armature_range"] = tuple(rob["armature_range"])
        p.update(new_params)

    delay = dr_cfg.get("action_delay", {})
    if delay and hasattr(events, "randomize_action_delay"):
        p = events.randomize_action_delay.params
        new_params = {}
        if "max_delay" in delay: new_params["max_delay"] = int(delay["max_delay"])
        p.update(new_params)

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


def apply_env_config(env_cfg, env_cfg_dict: dict):
    if not env_cfg_dict:
        return

    if "episode_length_s" in env_cfg_dict:
        env_cfg.episode_length_s = float(env_cfg_dict["episode_length_s"])
    if "action_scale" in env_cfg_dict:
        env_cfg.action_scale = float(env_cfg_dict["action_scale"])
        env_cfg.actions.joint_pos.scale = env_cfg.action_scale
    if "decimation" in env_cfg_dict:
        env_cfg.decimation = int(env_cfg_dict["decimation"])
        env_cfg.sim.render_interval = env_cfg.decimation

    rewards_cfg = env_cfg_dict.get("rewards", {})
    if rewards_cfg:
        reward_terms = {
            "object_pose": "object_pose",
            "finger_joint_goal": "finger_joint_goal",
            "fingertip_tracking": "fingertip_tracking",
            "grasp_success": "grasp_success",
            "fingertip_velocity": "fingertip_velocity",
            "fingertip_contact": "fingertip_contact",
            "action_scale": "action_scale",
            "torque": "torque",
            "mechanical_work": "mechanical_work",
            "action_rate": "action_rate",
            "object_velocity": "object_velocity",
            "object_drop": "object_drop",
            "object_left_hand": "object_left_hand",
            "joint_limit": "joint_limit",
            "wrist_height": "wrist_height",
        }
        for cfg_name, term_name in reward_terms.items():
            if cfg_name in rewards_cfg and hasattr(env_cfg.rewards, term_name):
                getattr(env_cfg.rewards, term_name).weight = float(rewards_cfg[cfg_name])


def _resolve_valid_minibatch_size(batch_size: int, requested_minibatch: int, seq_length: int) -> int:
    """Return the largest valid minibatch not exceeding the requested size."""
    # rl_games requires batch_size % minibatch_size == 0.
    # When sequence training is enabled, minibatch_size should also align with seq_length.
    max_candidate = min(requested_minibatch, batch_size)
    alignment = max(seq_length, 1)

    minibatch_size = None
    for candidate in range(max_candidate, 0, -alignment):
        if candidate % alignment == 0 and batch_size % candidate == 0:
            minibatch_size = candidate
            break

    if minibatch_size is None:
        raise ValueError(
            f"Could not find a valid minibatch size for batch_size={batch_size}, "
            f"requested_minibatch={requested_minibatch}, seq_length={seq_length}."
        )

    return minibatch_size


def _resolve_ppo_sizes(args, cfg_file: dict) -> tuple[int, int, int]:
    """Resolve PPO rollout sizes so rl_games batch constraints always hold."""
    ppo_cfg = cfg_file.get("ppo", {})

    horizon_length = int(ppo_cfg.get("horizon_length", 16))
    seq_length = int(ppo_cfg.get("seq_length", 4))
    requested_minibatch = int(ppo_cfg.get("minibatch_size", 4096))

    batch_size = args.num_envs * horizon_length
    minibatch_size = _resolve_valid_minibatch_size(batch_size, requested_minibatch, seq_length)

    return horizon_length, seq_length, minibatch_size


def build_rl_games_config(args, cfg_file: dict) -> dict:
    """
    Build rl_games PPO config.

    Key fixes vs previous version:
      - normalize_input: True   (was False → reward signal was weak)
      - normalize_value: True   (was False → value function unstable)
      - reward_shaper scale_value: 0.1  (was 0.01 → reward too small)
      - minibatch_size: 4096    (rollout=8192 → 2 minibatches/epoch)
      - horizon_length: 16      (was 8 → more stable gradient estimates)
    """
    ppo_cfg = cfg_file.get("ppo", {})
    cv_cfg = cfg_file.get("central_value", {})

    horizon_length, seq_length, minibatch_size = _resolve_ppo_sizes(args, cfg_file)
    batch_size = args.num_envs * horizon_length
    cv_minibatch_size = _resolve_valid_minibatch_size(
        batch_size,
        int(cv_cfg.get("minibatch_size", minibatch_size)),
        seq_length,
    )

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
                        "sigma_init": {"name": "const_initializer", "val": -1.0},
                        "fixed_sigma": True,
                    }
                },
                # Paper (arXiv:2502.04307): [1024, 512, 512, 256, 256] for both
                # actor and critic.  Shadow Hand has more DOF than the original
                # LEAP Hand so matching the paper size is important.
                "mlp": {
                    "units": ppo_cfg.get("units", [1024, 512, 512, 256, 256]),
                    "activation": ppo_cfg.get("activation", "elu"),
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
                # [FIX] normalize_input/value: True — critical for stable learning
                "normalize_input": True,
                "normalize_value": True,
                "num_actors": args.num_envs,
                "reward_shaper": {
                    "scale_value": float(ppo_cfg.get("reward_shaper_scale", 0.1)),
                },
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": float(ppo_cfg.get("learning_rate", 5e-4)),
                "lr_schedule": ppo_cfg.get("lr_schedule", "adaptive"),
                "lr_threshold": float(ppo_cfg.get("lr_threshold", 0.008)),
                "score_to_win": 20000,
                "max_epochs": args.max_iterations,
                "save_best_after": 100,
                "save_frequency": 1000,
                "print_stats": True,
                "grad_norm": float(ppo_cfg.get("grad_norm", 1.0)),
                "entropy_coef": float(ppo_cfg.get("entropy_coef", 0.0)),
                "truncate_grads": True,
                "e_clip": float(ppo_cfg.get("e_clip", 0.2)),
                "kl_threshold": float(ppo_cfg.get("kl_threshold", 0.008)),
                "horizon_length": horizon_length,
                "num_steps_per_env": int(ppo_cfg.get("num_steps_per_env", horizon_length)),
                "mini_epochs": int(ppo_cfg.get("mini_epochs", 5)),
                "minibatch_size": minibatch_size,
                "critic_coef": float(ppo_cfg.get("critic_coef", 4)),
                "clip_value": True,
                "seq_length": seq_length,
                # bounds_loss was exploding (0→70) causing policy to saturate actions.
                # 0.0001 was too small to prevent action clamping. 0.005 adds real penalty.
                "bounds_loss_coef": float(ppo_cfg.get("bounds_loss_coef", 0.005)),
                "log_dir": args.log_dir,

                # Asymmetric Actor-Critic
                "use_central_value": True,
                "central_value_config": {
                    "minibatch_size": cv_minibatch_size,
                    "mini_epochs": int(cv_cfg.get("mini_epochs", 4)),
                    "learning_rate": float(cv_cfg.get("learning_rate", 5e-4)),
                    "lr_schedule": cv_cfg.get("lr_schedule", "adaptive"),
                    "lr_threshold": float(cv_cfg.get("lr_threshold", 0.008)),
                    "clip_value": True,
                    # [FIX] normalize_input: True for critic as well
                    "normalize_input": bool(cv_cfg.get("normalize_input", True)),
                    "truncate_grads": True,
                    "grad_norm": float(cv_cfg.get("grad_norm", 1.0)),
                    "network": {
                        "name": "actor_critic",
                        "central_value": True,
                        "mlp": {
                            "units": cv_cfg.get("units", [512, 512, 256]),
                            "activation": cv_cfg.get("activation", "elu"),
                            "d2rl": False,
                            "initializer": {"name": "default"},
                            "regularizer": {"name": "None"},
                        }
                    }
                }
            },
        }
    }


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
            print("[train_rl] Set Isaac Sim cloud asset root to S3 (5.0).")
    except Exception as _e:
        print(f"[train_rl] WARNING: could not set carb asset root: {_e}")

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

    register_anygrasp_env()

    grasp_graph_paths = _resolve_grasp_graph_arg(args)
    missing_graph_paths = [path for path in grasp_graph_paths if not Path(path).exists()]
    if missing_graph_paths:
        print(f"ERROR: GraspGraph not found at {missing_graph_paths}")
        print("Run Stage 0 first:")
        print("  python scripts/run_grasp_generation.py")
        sim_app.close()
        sys.exit(1)

    cfg_file = load_config(args.config)
    merged_graph = load_merged_graph(grasp_graph_paths)

    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = grasp_graph_paths
    if getattr(args, "headless", False):
        env_cfg.viewer = None

    # Load object specs from the grasp graph so the environment spawns the
    # same object pool that was used during Stage 0 grasp generation.
    try:
        _graph = merged_graph
        from grasp_generation.rrt_expansion import MultiObjectGraspGraph
        if isinstance(_graph, MultiObjectGraspGraph) and _graph.object_specs:
            _specs = list(_graph.object_specs.values())
            from envs.anygrasp_env import _build_object_spawner
            env_cfg.scene.object = env_cfg.scene.object.replace(
                spawn=_build_object_spawner(_specs)
            )
            print(f"[Stage 1] Loaded {len(_specs)} object spec(s) from grasp graph:")
            for _s in _specs:
                print(f"          - {_s.get('name','?')}  shape={_s.get('shape_type','?')}  size={_s.get('size',0):.3f}m")
        else:
            print("[Stage 1] Grasp graph has no MultiObjectGraspGraph specs; using default cube.")

        graph_num_fingers = getattr(_graph, "num_fingers", None)
        if graph_num_fingers is None and isinstance(_graph, MultiObjectGraspGraph) and _graph.graphs:
            first_graph = next(iter(_graph.graphs.values()))
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
            print(f"[Stage 1] Hand config overridden from grasp graph: num_fingers={graph_num_fingers}")
    except Exception as _e:
        print(f"[Stage 1] WARNING: Could not load object specs from graph: {_e}")

    apply_env_config(env_cfg, cfg_file.get("env", {}))
    apply_dr_config(env_cfg, cfg_file.get("domain_randomization", {}))

    print(f"[Stage 1] Config:   {args.config}")
    print(f"[Stage 1] Task: DexGen-AnyGrasp-Allegro-v0")
    print(f"[Stage 1] Num envs: {args.num_envs}")
    print(f"[Stage 1] Max iterations: {args.max_iterations}")
    print(f"[Stage 1] Grasp graph(s): {', '.join(grasp_graph_paths)}")
    print(f"[Stage 1] Log dir: {args.log_dir}")
    print(f"[Stage 1] Action scale: {env_cfg.action_scale}")
    print("-" * 60)

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

    class _CompactIsaacAlgoObserver(IsaacAlgoObserver):
        def after_print_stats(self, frame, epoch_num, total_time):
            if self.ep_infos:
                for key in self.ep_infos[0]:
                    info_tensor = torch.tensor([], device=self.algo.device)
                    for ep_info in self.ep_infos:
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        info_tensor = torch.cat((info_tensor, ep_info[key].to(self.algo.device)))
                    value = torch.mean(info_tensor)
                    self.writer.add_scalar("Episode/" + key, value, epoch_num)
                self.ep_infos.clear()

            for k, v in self.direct_info.items():
                self.writer.add_scalar(f"Performance/{k}", v, epoch_num)

            if self.mean_scores.current_size > 0:
                mean_scores = self.mean_scores.get_mean()
                self.writer.add_scalar("Performance/score", mean_scores, epoch_num)

    cfg = build_rl_games_config(args, cfg_file)
    ppo_runtime_cfg = cfg["params"]["config"]
    print(f"[Stage 1] Horizon length: {ppo_runtime_cfg['horizon_length']}")
    print(f"[Stage 1] Seq length: {ppo_runtime_cfg['seq_length']}")
    print(f"[Stage 1] Batch size: {args.num_envs * ppo_runtime_cfg['horizon_length']}")
    print(f"[Stage 1] Minibatch size: {ppo_runtime_cfg['minibatch_size']}")
    runner = Runner(_CompactIsaacAlgoObserver())
    runner.load(cfg)
    runner.reset()
    runner.run({"train": True})

    print(f"\n=== Stage 1 Complete ===")
    print(f"Checkpoints saved to: {args.log_dir}")
    print(f"\nNext: collect dataset")
    print(f"  python scripts/collect_data.py --log_dir {args.log_dir}")

    sim_app.close()


def _to_rl_obs(obs):
    """
    Convert Isaac Lab observation output to rl_games format.
    Adds a safety net to prevent ANY NaN/Inf from reaching the network.
    """
    if isinstance(obs, dict):
        res = {}
        p_tensor = obs.get("policy", next(iter(obs.values())))
        res["obs"] = torch.nan_to_num(p_tensor, nan=0.0, posinf=100.0, neginf=-100.0)

        if "critic" in obs:
            res["states"] = torch.nan_to_num(obs["critic"], nan=0.0, posinf=100.0, neginf=-100.0)

        return res

    return {"obs": torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)}


class _IsaacLabVecEnv:
    """Thin wrapper to make Isaac Lab ManagerBasedRLEnv compatible with rl_games."""

    def __init__(self, env, num_envs: int):
        self.env = env
        self.num_envs = num_envs

    def step(self, actions):
        # [FIX] Update action buffers BEFORE stepping the env.
        # This ensures:
        #   - last_action obs = previous step's action (not always 0)
        #   - action_rate_penalty can compute delta correctly
        extras = self.env.extras
        if "current_action" in extras:
            extras["last_action"] = extras["current_action"].clone()
        else:
            extras["last_action"] = actions.clone()
        extras["current_action"] = actions.clone()

        # Apply action delay DR (0-2 step lag set by randomize_action_delay).
        # Must be called here — the randomize event only sets the delay length;
        # the actual buffering and delayed output happens at every step.
        from envs.mdp.domain_rand import apply_action_delay
        delayed_actions = apply_action_delay(self.env, actions)

        obs, rew, terminated, truncated, info = self.env.step(delayed_actions)
        done = terminated | truncated
        from envs.mdp import events as mdp_events
        from envs.mdp import rewards as mdp_rewards

        # Rolling goal: when a grasp is achieved mid-episode, immediately
        # select a new nearby goal (kNN) so the policy keeps transitioning
        # rather than idling until reset.  Uses threshold=2 cm (looser than
        # the 1 cm sparse-reward threshold) so the new goal is set just
        # before the sparse bonus fires.
        n_updated = mdp_events.update_rolling_goal(self.env, success_threshold=0.02)

        info = dict(info) if isinstance(info, dict) else {}
        info["success_ratio"] = float(mdp_rewards.grasp_success_reward(self.env).mean().item())

        # Isaac Lab resets done envs inside env.step() before returning obs/info.
        # Read termination terms from the manager's cached per-step masks so these
        # ratios reflect the step that just ended, not the freshly reset state.
        term_manager = getattr(self.env, "termination_manager", None)
        if term_manager is not None:
            active_terms = set(getattr(term_manager, "active_terms", []))
            if "object_drop" in active_terms:
                drop_mask = term_manager.get_term("object_drop")
                info["drop_ratio"] = float(drop_mask.float().mean().item())
            else:
                info["drop_ratio"] = float(mdp_events.object_dropped(self.env).float().mean().item())

            if "object_left_hand" in active_terms:
                left_hand_mask = term_manager.get_term("object_left_hand")
                info["left_hand_ratio"] = float(left_hand_mask.float().mean().item())
            else:
                info["left_hand_ratio"] = float(mdp_events.object_left_hand(self.env).float().mean().item())
        else:
            info["drop_ratio"] = float(mdp_events.object_dropped(self.env).float().mean().item())
            info["left_hand_ratio"] = float(mdp_events.object_left_hand(self.env).float().mean().item())

        info["rolling_goal_updates"] = n_updated
        return _to_rl_obs(obs), rew, done, info

    def reset(self):
        obs, _ = self.env.reset()
        # Initialise action buffers on first reset
        num_dof = self.env.action_manager.action.shape[-1]
        if "last_action" not in self.env.extras:
            self.env.extras["last_action"] = torch.zeros(
                self.num_envs, num_dof, device=self.env.device
            )
        if "current_action" not in self.env.extras:
            self.env.extras["current_action"] = torch.zeros(
                self.num_envs, num_dof, device=self.env.device
            )
        return _to_rl_obs(obs)

    def get_number_of_agents(self):
        return self.num_envs

    def set_train_info(self, frame, self_agent):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass

    def get_env_info(self):
        import gymnasium
        import gym as old_gym
        import numpy as np

        raw_obs_space = self.env.observation_space
        if isinstance(raw_obs_space, (gymnasium.spaces.Dict, old_gym.spaces.Dict)):
            obs_space = raw_obs_space.spaces.get(
                "policy",
                next(iter(raw_obs_space.spaces.values())),
            )
            state_space = raw_obs_space.spaces.get("critic", None)
        else:
            obs_space = raw_obs_space
            state_space = None

        obs_shape = obs_space.shape
        if len(obs_shape) > 1:
            obs_shape = obs_shape[1:]

        raw_act_space = self.env.action_space
        act_shape = raw_act_space.shape
        if len(act_shape) > 1:
            act_shape = act_shape[1:]

        obs_space_old = old_gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        action_space_old = old_gym.spaces.Box(
            low=-1.0, high=1.0, shape=act_shape, dtype=np.float32
        )

        info = {
            "action_space": action_space_old,
            "observation_space": obs_space_old,
        }

        if state_space is not None:
            state_shape = state_space.shape
            if len(state_shape) > 1:
                state_shape = state_shape[1:]
            info["state_space"] = old_gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
            )

        return info


if __name__ == "__main__":
    main()
