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
        --resume logs/rl/sharpa_anygrasp_v1/checkpoints/model_10000.pt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher
from grasp_generation.graph_io import MultiObjectGraspGraph, load_merged_graph, parse_graph_paths


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
    p.add_argument("--max_iterations", type=int, default=10000,
                   help="Maximum PPO training iterations")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--log_dir", type=str, default=None,
                   help="Training log / checkpoint directory. "
                        "If omitted, a timestamped subdir is used so "
                        "previous runs are not overwritten.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
                   help="Path to YAML config (ppo / domain_randomization settings)")

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.grasp_graph is None:
        args.grasp_graph = ["data/sharpa_grasp_cube_050.npy"]
    cfg = load_config(args.config)
    if args.log_dir is None:
        if args.resume:
            # Reuse the run directory the checkpoint was saved in so that
            # tensorboard continues in the same experiment instead of
            # spawning a parallel one. Checkpoint layout is
            #   <run_dir>/nn/<file>.pth
            resume_path = Path(args.resume).resolve()
            args.log_dir = str(resume_path.parent.parent)
        else:
            base = cfg.get("logging", {}).get("log_dir", "logs/rl/sharpa_anygrasp_v1")
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            args.log_dir = str(Path(base) / stamp)
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
        # Our Sharpa obs uses a single `temporal` term that already bakes in
        # joint noise internally (joint_noise_scale in observations.py), so
        # there is no separate obs.policy.joint_pos to attach a noise model
        # to. Skip silently rather than emitting spurious warnings.
        obs = env_cfg.observations
        for group_name in ("policy", "critic"):
            group = getattr(obs, group_name, None)
            if group is None:
                continue
            if "joint_pos_std" in noise and hasattr(group, "joint_pos"):
                jp = getattr(group, "joint_pos")
                if getattr(jp, "noise", None) is not None:
                    jp.noise.std = float(noise["joint_pos_std"])
            if "joint_vel_std" in noise and hasattr(group, "joint_vel"):
                jv = getattr(group, "joint_vel")
                if getattr(jv, "noise", None) is not None:
                    jv.noise.std = float(noise["joint_vel_std"])


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
    if "training_curriculum" in env_cfg_dict:
        env_cfg.training_curriculum = dict(env_cfg_dict["training_curriculum"])
    elif "curriculum" in env_cfg_dict:
        env_cfg.training_curriculum = dict(env_cfg_dict["curriculum"])
    elif "gravity_curriculum" in env_cfg_dict:
        env_cfg.training_curriculum = dict(env_cfg_dict["gravity_curriculum"])

    rewards_cfg = env_cfg_dict.get("rewards", {})
    if rewards_cfg:
        if "orientation_delta_weight" in rewards_cfg and hasattr(env_cfg.rewards, "orientation_delta"):
            env_cfg.rewards.orientation_delta.weight = float(rewards_cfg["orientation_delta_weight"])
        if hasattr(env_cfg.rewards, "goal_bonus"):
            if "goal_bonus" in rewards_cfg:
                env_cfg.rewards.goal_bonus.params["bonus"] = float(rewards_cfg["goal_bonus"])
            if "goal_thresh" in rewards_cfg:
                env_cfg.rewards.goal_bonus.params["rot_thresh"] = float(rewards_cfg["goal_thresh"])
        if "drop_penalty" in rewards_cfg and hasattr(env_cfg.rewards, "drop"):
            env_cfg.rewards.drop.params["penalty"] = float(rewards_cfg["drop_penalty"])

    term_cfg = env_cfg_dict.get("terminations", {})
    if term_cfg and hasattr(env_cfg.terminations, "object_drop"):
        if "object_drop_max_dist" in term_cfg:
            env_cfg.terminations.object_drop.params["max_dist"] = float(
                term_cfg["object_drop_max_dist"]
            )


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


def _build_network_config(ppo_cfg: dict) -> dict:
    """Build rl_games network config, optionally with LSTM."""
    net = {
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
        "mlp": {
            "units": ppo_cfg.get("units", [1024, 512, 512, 256, 256]),
            "activation": ppo_cfg.get("activation", "elu"),
            "d2rl": False,
            "initializer": {"name": "default"},
            "regularizer": {"name": "None"},
        },
    }

    # Add LSTM if enabled
    if bool(ppo_cfg.get("use_rnn", False)):
        rnn_cfg = ppo_cfg.get("rnn", {})
        net["rnn"] = {
            "name": rnn_cfg.get("name", "lstm"),
            "units": int(rnn_cfg.get("units", 256)),
            "layers": int(rnn_cfg.get("layers", 1)),
            "before_mlp": bool(rnn_cfg.get("before_mlp", False)),
            "concat_input": bool(rnn_cfg.get("concat_input", True)),
        }

    return net


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
    log_dir = Path(args.log_dir)

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
            "network": _build_network_config(ppo_cfg),
            "load_checkpoint": args.resume is not None,
            "load_path": args.resume or "",
            "config": {
                "name": "DexGen-AnyGrasp-Sharpa",
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
                "bounds_loss_coef": float(ppo_cfg.get("bounds_loss_coef", 0.0001)),
                # rl_games writes to train_dir/full_experiment_name, not log_dir.
                "train_dir": str(log_dir.parent),
                "full_experiment_name": log_dir.name,

                # Symmetric Actor-Critic (actor sees full obs = critic obs)
                "use_central_value": False,
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
        print("Run Stage 0 first, for example:")
        print("  /workspace/IsaacLab/isaaclab.sh -p scripts/gen_grasp.py --shape cube --size 0.05 --num_grasps 1000 --headless")
        sim_app.close()
        sys.exit(1)

    cfg_file = load_config(args.config)
    merged_graph = load_merged_graph(grasp_graph_paths)

    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = grasp_graph_paths
    # Note: do NOT set env_cfg.viewer = None when headless. Isaac Lab's
    # ViewportCameraController is constructed unconditionally and reads
    # cfg.viewer.eye, which crashes if viewer is None. This also breaks
    # --livestream which needs a real viewport. The default ViewerCfg
    # works fine for headless runs (just hidden).

    # Load object specs from the grasp graph so the environment spawns the
    # same object pool that was used during Stage 0 grasp generation.
    try:
        _graph = merged_graph
        if isinstance(_graph, MultiObjectGraspGraph) and _graph.object_specs:
            _specs = list(_graph.object_specs.values())
            from envs.anygrasp_env import _build_object_spawner
            env_cfg.scene.object = env_cfg.scene.object.replace(
                spawn=_build_object_spawner(_specs)
            )
            # When the grasp graph has a single rigid asset, enable Isaac Lab
            # physics replication (clone once, reuse across envs). This must
            # be set AFTER __post_init__ because __post_init__ sees the full
            # default object pool and conservatively sets it to False.
            # Single-asset fast-path → minutes → seconds startup.
            env_cfg.scene.replicate_physics = (len(_specs) == 1)
            print(f"[Stage 1] Loaded {len(_specs)} object spec(s) from grasp graph:")
            for _s in _specs:
                print(f"          - {_s.get('name','?')}  shape={_s.get('shape_type','?')}  size={_s.get('size',0):.3f}m")
            print(f"[Stage 1] replicate_physics={env_cfg.scene.replicate_physics}")
        else:
            print("[Stage 1] Grasp graph has no MultiObjectGraspGraph specs; using default cube.")

        graph_num_fingers = getattr(_graph, "num_fingers", None)
        if graph_num_fingers is None and isinstance(_graph, MultiObjectGraspGraph) and _graph.graphs:
            first_graph = next(iter(_graph.graphs.values()))
            graph_num_fingers = getattr(first_graph, "num_fingers", None)
        if graph_num_fingers is not None:
            graph_num_fingers = int(graph_num_fingers)
            tip_subsets = {
                2: ["right_thumb_fingertip", "right_index_fingertip"],
                3: ["right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip"],
                4: ["right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip", "right_ring_fingertip"],
                5: ["right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip", "right_ring_fingertip", "right_pinky_fingertip"],
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

    try:
        apply_env_config(env_cfg, cfg_file.get("env", {}))
    except Exception as _e:
        print(f"[WARNING] apply_env_config failed: {_e}")
    try:
        apply_dr_config(env_cfg, cfg_file.get("domain_randomization", {}))
    except Exception as _e:
        print(f"[WARNING] apply_dr_config failed: {_e}")

    # ── Verify final config after YAML overrides ──
    try:
        _dr_obj = getattr(getattr(env_cfg.events, "randomize_object_physics", None), "params", {})
        _rw = env_cfg.rewards
        _term = env_cfg.terminations
        print("=" * 60)
        print("[CONFIG] ── Final values (after YAML override) ──")
        if _dr_obj:
            print(f"[CONFIG] mass_range:      {_dr_obj.get('mass_range')}")
            print(f"[CONFIG] friction_range:  {_dr_obj.get('friction_range')}")
            print(f"[CONFIG] restitution:     {_dr_obj.get('restitution_range')}")
        reward_parts = []
        for reward_name in ("object_orientation", "object_position", "goal_bonus", "drop", "work", "action", "torque"):
            reward_term = getattr(_rw, reward_name, None)
            if reward_term is not None and hasattr(reward_term, "weight"):
                reward_parts.append(f"{reward_name}={reward_term.weight}")
        if reward_parts:
            print(f"[CONFIG] rewards:  {'  '.join(reward_parts)}")
        if hasattr(_rw, "goal_bonus") and getattr(_rw.goal_bonus, "params", None):
            _gp = _rw.goal_bonus.params
            _gget = _gp.get if hasattr(_gp, "get") else (lambda k, d=None: _gp[k] if k in _gp else d)
            print(
                f"[CONFIG] goal_bonus: rot_thresh={_gget('rot_thresh')} rad  "
                f"pos_thresh={_gget('pos_thresh')} m  "
                f"bonus={_gget('bonus')}"
            )
        if hasattr(_term, "object_drop") and getattr(_term.object_drop, "params", None):
            _od = _term.object_drop.params
            _odget = _od.get if hasattr(_od, "get") else (lambda k, d=None: _od[k] if k in _od else d)
            print(f"[CONFIG] object_drop: max_dist={_odget('max_dist')} m (palm–object)")
        if hasattr(_term, "no_fingertip_contact"):
            print(f"[CONFIG] no_contact_patience: {_term.no_fingertip_contact.params.get('patience')}")
        print(f"[CONFIG] episode_length_s: {env_cfg.episode_length_s}")
        print("=" * 60)
    except Exception as _e:
        print(f"[WARNING] Config verification failed: {_e}")
        import traceback; traceback.print_exc()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1] Config:   {args.config}")
    print(f"[Stage 1] Task: DexGen-AnyGrasp-Sharpa-v0")
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
    _action_mode = cfg_file.get("env", {}).get("action_mode", "absolute")
    _delta_scale = float(cfg_file.get("env", {}).get("delta_scale", 1.0 / 24.0))
    _actions_ma = float(cfg_file.get("env", {}).get("actions_moving_average", 1.0))
    if _action_mode == "delta":
        print(f"[Stage 1] Action mode: delta (scale={_delta_scale}, moving_avg={_actions_ma})")
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: _IsaacLabVecEnv(
            create_env(), num_actors,
            action_mode=_action_mode, delta_scale=_delta_scale,
            actions_moving_average=_actions_ma,
        ),
    )

    _max_iters = args.max_iterations

    class _FilteredWriter:
        """Wraps a SummaryWriter to drop redundant rl_games metrics."""
        _KEEP_PREFIXES = ("Performance/", "Episode/", "losses/", "info/", "rewards/")

        def __init__(self, writer):
            self._writer = writer

        def add_scalar(self, tag, value, step, *args, **kwargs):
            if tag.startswith(self._KEEP_PREFIXES):
                self._writer.add_scalar(tag, value, step, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._writer, name)

    class _CompactIsaacAlgoObserver(IsaacAlgoObserver):
        _writer_wrapped = False

        def after_init(self, algo):
            # rl_games calls this AFTER the algo is built and AFTER the
            # checkpoint (if any) has been restored. This is the only
            # reliable place to read the restored epoch_num and seed the
            # curriculum before the first rollout runs.
            super().after_init(algo)
            # Wrap writer immediately so rl_games' own lowercase
            # "performance/" metrics are filtered from the first epoch.
            if not self._writer_wrapped and hasattr(self, "writer"):
                self.writer = _FilteredWriter(self.writer)
                if hasattr(algo, "writer"):
                    algo.writer = _FilteredWriter(algo.writer)
                self._writer_wrapped = True
            epoch = int(getattr(algo, "epoch_num", 0) or 0)
            frame = int(getattr(algo, "frame", 0) or 0)
            if epoch > 0 or frame > 0:
                print(f"[Stage 1] Resumed: epoch_num={epoch}  frame={frame}")
            try:
                from envs.mdp import events as mdp_events
                mdp_events.update_curriculum(
                    algo.vec_env.env, epoch, _max_iters,
                )
                print(f"[Stage 1] Curriculum applied for epoch {epoch}")
            except Exception as _e:
                print(f"[Stage 1] WARNING: could not apply curriculum in after_init: {_e}")

        def after_print_stats(self, frame, epoch_num, total_time):
            from envs.mdp import events as mdp_events
            # min_orn + gravity: see mdp_events.update_curriculum (warmup_ratio)
            mdp_events.update_curriculum(
                self.algo.vec_env.env, epoch_num, _max_iters,
            )
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

            # Use the same cumulative reset-reason counters that the print
            # statement in events.py uses (_reset_log_stats), so Performance/
            # metrics and console output show identical values.
            env = self.algo.vec_env.env
            _rg = getattr(env, "_last_rolling_goal_updates", 0)
            stats = env.extras.get("_reset_log_stats", {})
            total_resets = max(stats.get("total_resets", 0), 1)
            self.direct_info = {
                "success_ratio": _rg / max(env.num_envs, 1),
                "rolling_goal_updates": _rg,
                "drop_ratio": stats.get("total_drops", 0) / total_resets,
                "timeout_ratio": stats.get("total_timeouts", 0) / total_resets,
            }

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

    # rl_games Runner.run_train() restores a checkpoint by reading
    #   _restore(agent, args): if args.get('checkpoint'): agent.restore(...)
    # so we MUST pass the checkpoint path through runner.run(args), not
    # just via load_path / load_checkpoint in the yaml-style config (those
    # are ignored by current rl_games). Without this, --resume silently
    # started training from random weights.
    run_args = {"train": True}
    if args.resume:
        run_args["checkpoint"] = args.resume
        print(f"[Stage 1] Will restore checkpoint: {args.resume}")

    runner.run(run_args)

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

    def __init__(self, env, num_envs: int, action_mode: str = "absolute", delta_scale: float = 0.05,
                 actions_moving_average: float = 1.0):
        self.env = env
        self.num_envs = num_envs
        self._action_mode = action_mode
        self._delta_scale = delta_scale
        self._actions_moving_average = actions_moving_average
        self._joint_target = None  # initialised on first reset
        self._prev_actions = None  # for moving average

        # Build batch-free spaces so rl_games sees (obs_dim,) not (num_envs, obs_dim).
        # rl_games reads both env.observation_space AND get_env_info() —
        # everything must be a simple gym.spaces.Box with shape (dim,).
        import gym as old_gym
        import numpy as np

        def _strip_batch(shape):
            return shape[1:] if len(shape) > 1 else shape

        raw_obs_space = env.observation_space
        self._state_space = None
        # Use duck typing — Isaac Lab may use gymnasium.spaces.Dict or a subclass
        # that doesn't match isinstance checks against gym/gymnasium.
        if hasattr(raw_obs_space, "spaces"):
            obs_sp = raw_obs_space.spaces.get("policy", next(iter(raw_obs_space.spaces.values())))
            critic_sp = raw_obs_space.spaces.get("critic", None)
            if critic_sp is not None:
                self._state_space = old_gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=_strip_batch(critic_sp.shape), dtype=np.float32,
                )
        else:
            obs_sp = raw_obs_space

        raw_act_space = env.action_space

        self.observation_space = old_gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=_strip_batch(obs_sp.shape), dtype=np.float32,
        )
        self.action_space = old_gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=_strip_batch(raw_act_space.shape), dtype=np.float32,
        )

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

        # Delta action mode (IsaacGymEnvs ShadowHand style):
        #   smoothed = α * actions + (1-α) * prev_actions
        #   targets  = prev_targets + dof_speed_scale * dt * smoothed
        #   targets  = clamp(targets, joint_lower, joint_upper)
        if self._action_mode == "delta":
            if self._joint_target is None:
                self._joint_target = torch.zeros_like(actions)
            if self._prev_actions is None:
                self._prev_actions = torch.zeros_like(actions)
            alpha = self._actions_moving_average
            smoothed = alpha * actions + (1.0 - alpha) * self._prev_actions
            self._prev_actions = smoothed.clone()
            self._joint_target = (self._joint_target + self._delta_scale * smoothed).clamp(-1.0, 1.0)
            env_actions = self._joint_target
        else:
            # Absolute mode: apply EMA for smooth actions (OpenAI Dactyl style)
            if self._prev_actions is None:
                # Init to current joint positions so EMA starts from actual pose
                robot = self.env.scene["robot"]
                cur_q = robot.data.joint_pos
                soft_lower = robot.data.soft_joint_pos_limits[..., 0]
                soft_upper = robot.data.soft_joint_pos_limits[..., 1]
                mid = (soft_upper + soft_lower) * 0.5
                rng = (soft_upper - soft_lower) * 0.5
                rng = rng.clamp(min=1e-6)
                norm_q = ((cur_q - mid) / rng).clamp(-1.0, 1.0)
                if norm_q.shape[-1] > actions.shape[-1]:
                    norm_q = norm_q[:, -actions.shape[-1]:]
                self._prev_actions = norm_q
            alpha = self._actions_moving_average
            env_actions = alpha * actions + (1.0 - alpha) * self._prev_actions
            self._prev_actions = env_actions.clone()

        # Apply action delay DR (0-2 step lag set by randomize_action_delay).
        # Must be called here — the randomize event only sets the delay length;
        # the actual buffering and delayed output happens at every step.
        from envs.mdp.domain_rand import apply_action_delay
        delayed_actions = apply_action_delay(self.env, env_actions)

        obs, rew, terminated, truncated, info = self.env.step(delayed_actions)
        done = terminated | truncated

        # Debug: log per-step reward stats
        if hasattr(self, '_step_log_counter'):
            self._step_log_counter += 1
        else:
            self._step_log_counter = 0
        if self._step_log_counter % 200 == 0:
            print(f"  [REW] mean={rew.mean().item():.4f}  min={rew.min().item():.4f}  "
                  f"max={rew.max().item():.4f}  std={rew.std().item():.4f}")

        # Re-initialise action buffers for reset envs
        if done.any() and self._action_mode != "delta" and self._prev_actions is not None:
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            # Set prev_actions to current joint positions (normalized to [-1,1])
            # so EMA doesn't pull the hand toward 0 (mid-range) after reset.
            robot = self.env.scene["robot"]
            cur_q = robot.data.joint_pos[done_ids]
            soft_lower = robot.data.soft_joint_pos_limits[done_ids, :, 0]
            soft_upper = robot.data.soft_joint_pos_limits[done_ids, :, 1]
            mid = (soft_upper + soft_lower) * 0.5
            rng = (soft_upper - soft_lower) * 0.5
            rng = rng.clamp(min=1e-6)
            norm_q = ((cur_q - mid) / rng).clamp(-1.0, 1.0)
            num_dof = self._prev_actions.shape[-1]
            if norm_q.shape[-1] > num_dof:
                norm_q = norm_q[:, -num_dof:]
            self._prev_actions[done_ids] = norm_q

        if self._action_mode == "delta" and done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            robot = self.env.scene["robot"]
            cur_q = robot.data.joint_pos[done_ids]
            soft_lower = robot.data.soft_joint_pos_limits[done_ids, :, 0]
            soft_upper = robot.data.soft_joint_pos_limits[done_ids, :, 1]
            mid = (soft_upper + soft_lower) * 0.5
            rng = (soft_upper - soft_lower) * 0.5
            rng = rng.clamp(min=1e-6)
            norm_q = ((cur_q - mid) / rng).clamp(-1.0, 1.0)
            num_dof = self._joint_target.shape[-1]
            if norm_q.shape[-1] > num_dof:
                norm_q = norm_q[:, -num_dof:]
            self._joint_target[done_ids] = norm_q
            self._prev_actions[done_ids] = 0.0

        from envs.mdp import events as mdp_events
        from envs.mdp import rewards as mdp_rewards

        # Rolling goal: when orientation error < rot_threshold and object still
        # in hand, sample a new nearby grasp goal (kNN).
        n_updated = mdp_events.update_rolling_goal(self.env)
        # Cache on env so the observer can read it for Performance/ logging
        self.env._last_rolling_goal_updates = n_updated

        info = dict(info) if isinstance(info, dict) else {}
        info["success_ratio"] = n_updated / self.env.num_envs

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
        # Delta mode: initialise joint target from current joint positions
        if self._action_mode == "delta":
            robot = self.env.scene["robot"]
            cur_q = robot.data.joint_pos  # (N, num_joints)
            soft_lower = robot.data.soft_joint_pos_limits[..., 0]
            soft_upper = robot.data.soft_joint_pos_limits[..., 1]
            # Normalise current joints to [-1, 1] to match action space
            mid = (soft_upper + soft_lower) * 0.5
            rng = (soft_upper - soft_lower) * 0.5
            rng = rng.clamp(min=1e-6)
            self._joint_target = ((cur_q - mid) / rng).clamp(-1.0, 1.0)
            # Slice to action dim (exclude wrist if needed)
            if self._joint_target.shape[-1] > num_dof:
                self._joint_target = self._joint_target[:, -num_dof:]
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
        info = {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        if self._state_space is not None:
            info["state_space"] = self._state_space
        return info


if __name__ == "__main__":
    main()
