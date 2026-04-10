"""
Policy Evaluation
=================
Evaluates a trained AnyGrasp-to-AnyGrasp checkpoint (from `train_rl.py`)
on the Isaac Lab environment using rl_games' play loop.

Reports per-episode success metrics:
    - mean episode reward
    - mean episode length
    - drop / left-hand / no-contact termination ratios
    - rolling-goal updates per step (success rate)

Usage:
    python scripts/evaluate.py \
        --grasp_graph data/sharpa_grasp_cube_050.npy \
        --checkpoint  logs/rl/sharpa_anygrasp_v1/nn/DexGen-AnyGrasp-Sharpa.pth \
        --num_envs    16 \
        --num_episodes 50

    # Headless
    python scripts/evaluate.py \
        --grasp_graph data/sharpa_grasp_cube_050.npy \
        --checkpoint  logs/rl/sharpa_anygrasp_v1/nn/last.pth \
        --headless
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # for `import train_rl`

from isaaclab.app import AppLauncher
from grasp_generation.graph_io import MultiObjectGraspGraph, load_merged_graph, parse_graph_paths

# Reuse helpers from train_rl.py so behaviour matches 1:1.
from train_rl import (  # noqa: E402
    _IsaacLabVecEnv,
    _build_network_config,
    _to_rl_obs,
    apply_dr_config,
    apply_env_config,
    load_config,
)


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 1: Policy Evaluation")
    p.add_argument(
        "--grasp_graph",
        action="append",
        default=None,
        help="Grasp graph path(s). Repeat flag or use comma-separated values.",
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint file (.pth) produced by train_rl.py")
    p.add_argument("--num_envs", type=int, default=16,
                   help="Parallel environments to run during evaluation")
    p.add_argument("--num_episodes", type=int, default=50,
                   help="Total number of episodes to evaluate")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic actions (mean of policy distribution)")
    p.add_argument("--stochastic", dest="deterministic", action="store_false",
                   help="Sample stochastic actions from the policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "rl_training.yaml"),
                   help="Path to YAML config (env / ppo settings)")
    p.add_argument("--results_json", type=str, default=None,
                   help="Optional JSON output path for the evaluation summary")
    p.add_argument("--max_iterations", type=int, default=10000,
                   help="Total training iterations used at training time. "
                        "Required to reproduce curriculum interpolation "
                        "(gravity / min_orn are a function of epoch / "
                        "(warmup_ratio * max_iterations)). Must match the "
                        "value passed to train_rl.py.")
    p.add_argument("--curriculum_epoch", type=int, default=None,
                   help="Override curriculum epoch. Default: auto-detect "
                        "from the checkpoint's 'epoch' field.")

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.grasp_graph is None:
        args.grasp_graph = ["data/sharpa_grasp_cube_050.npy"]
    return args


def _resolve_grasp_graph_arg(args) -> list[str]:
    graph_paths = parse_graph_paths(args.grasp_graph)
    if not graph_paths:
        raise ValueError("At least one --grasp_graph path is required.")
    return graph_paths


def build_eval_rl_games_config(args, cfg_file: dict) -> dict:
    """Minimal rl_games config for play mode (no training bells & whistles)."""
    ppo_cfg = cfg_file.get("ppo", {})

    return {
        "params": {
            "seed": args.seed,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": _build_network_config(ppo_cfg),
            "load_checkpoint": True,
            "load_path": args.checkpoint,
            "config": {
                "name": "DexGen-AnyGrasp-Sharpa-Eval",
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
                    "scale_value": float(ppo_cfg.get("reward_shaper_scale", 0.1)),
                },
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": float(ppo_cfg.get("learning_rate", 5e-4)),
                "score_to_win": 20000,
                "max_epochs": 1,
                "save_best_after": 1_000_000,
                "save_frequency": 1_000_000,
                "print_stats": False,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,
                "horizon_length": int(ppo_cfg.get("horizon_length", 16)),
                "num_steps_per_env": int(ppo_cfg.get("horizon_length", 16)),
                "mini_epochs": 1,
                "minibatch_size": max(args.num_envs, 1),
                "critic_coef": 4,
                "clip_value": True,
                "seq_length": int(ppo_cfg.get("seq_length", 4)),
                "bounds_loss_coef": 0.0001,
                "use_central_value": False,
                # Force Player to use the registered vecenv (our _EvalVecEnv).
                # rl_games reads `use_vecenv` from the top-level config, not
                # from `config.player`.
                "use_vecenv": True,
                "player": {
                    "render": False,
                    "deterministic": bool(args.deterministic),
                    "games_num": int(args.num_episodes),
                    "print_stats": True,
                },
            },
        }
    }


def main():
    args = parse_args()

    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

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
    except Exception as _e:
        print(f"[evaluate] WARNING: could not set carb asset root: {_e}")

    try:
        import isaaclab  # noqa: F401
    except ImportError:
        print("ERROR: Isaac Lab not found.")
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
    from rl_games.torch_runner import Runner

    from envs import AnyGraspEnvCfg, register_anygrasp_env

    register_anygrasp_env()

    # --- Grasp graph ---------------------------------------------------
    grasp_graph_paths = _resolve_grasp_graph_arg(args)
    missing = [p for p in grasp_graph_paths if not Path(p).exists()]
    if missing:
        print(f"ERROR: GraspGraph not found at {missing}")
        sim_app.close()
        sys.exit(1)

    cfg_file = load_config(args.config)
    merged_graph = load_merged_graph(grasp_graph_paths)

    # --- Env config ----------------------------------------------------
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = grasp_graph_paths
    # Don't null out env_cfg.viewer when headless — Isaac Lab constructs
    # ViewportCameraController unconditionally and reads cfg.viewer.eye.
    # Also required for --livestream to work.

    try:
        if isinstance(merged_graph, MultiObjectGraspGraph) and merged_graph.object_specs:
            _specs = list(merged_graph.object_specs.values())
            from envs.anygrasp_env import _build_object_spawner
            env_cfg.scene.object = env_cfg.scene.object.replace(
                spawn=_build_object_spawner(_specs)
            )
            # Fast startup path: single-asset grasp graphs can use physics
            # replication. __post_init__ had to assume the default 6-item
            # pool, so we override after loading the graph.
            env_cfg.scene.replicate_physics = (len(_specs) == 1)
            print(f"[Evaluate] Loaded {len(_specs)} object spec(s) from grasp graph "
                  f"(replicate_physics={env_cfg.scene.replicate_physics}).")

        graph_num_fingers = getattr(merged_graph, "num_fingers", None)
        if graph_num_fingers is None and isinstance(merged_graph, MultiObjectGraspGraph) and merged_graph.graphs:
            first_graph = next(iter(merged_graph.graphs.values()))
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
    except Exception as _e:
        print(f"[Evaluate] WARNING: could not load object specs from graph: {_e}")

    try:
        apply_env_config(env_cfg, cfg_file.get("env", {}))
    except Exception as _e:
        print(f"[WARNING] apply_env_config failed: {_e}")
    try:
        apply_dr_config(env_cfg, cfg_file.get("domain_randomization", {}))
    except Exception as _e:
        print(f"[WARNING] apply_dr_config failed: {_e}")

    # Determine the curriculum epoch we want to evaluate at.
    #   1. --curriculum_epoch CLI override, if provided
    #   2. auto-detect from the checkpoint's 'epoch' key
    #   3. fall back to 0 (start of curriculum)
    # We intentionally do NOT disable env_cfg.gravity_curriculum — it stays
    # enabled so mdp_events.update_curriculum() can apply the matching
    # gravity later, once the sim is running. _EvalVecEnv.step() never
    # re-calls update_curriculum, so whatever state we apply stays fixed
    # for the whole eval run.
    ckpt_epoch = args.curriculum_epoch
    if ckpt_epoch is None:
        try:
            _ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ckpt_epoch = int(_ckpt.get("epoch", 0) or 0)
            del _ckpt
            print(f"[Evaluate] Auto-detected checkpoint epoch: {ckpt_epoch}")
        except Exception as _e:
            print(f"[Evaluate] WARNING: could not read epoch from checkpoint "
                  f"({_e}); defaulting to 0")
            ckpt_epoch = 0
    else:
        print(f"[Evaluate] Using override curriculum epoch: {ckpt_epoch}")

    print("=" * 60)
    print(f"[Evaluate] Task: DexGen-AnyGrasp-Sharpa-v0")
    print(f"[Evaluate] Checkpoint:   {args.checkpoint}")
    print(f"[Evaluate] Grasp graph:  {', '.join(grasp_graph_paths)}")
    print(f"[Evaluate] Num envs:     {args.num_envs}")
    print(f"[Evaluate] Num episodes: {args.num_episodes}")
    print(f"[Evaluate] Deterministic:{args.deterministic}")
    print("=" * 60)

    # --- Env registration ---------------------------------------------
    # We create the wrapped env ONCE and have env_creator return it directly.
    # That way rl_games' Player reads the wrapper's flat Box(obs_dim,) space
    # (instead of the raw ManagerBasedRLEnv Dict space) when it builds the
    # policy network via env_configurations.get_env_info().
    _action_mode = cfg_file.get("env", {}).get("action_mode", "absolute")
    _delta_scale = float(cfg_file.get("env", {}).get("delta_scale", 1.0 / 24.0))
    _actions_ma = float(cfg_file.get("env", {}).get("actions_moving_average", 1.0))

    raw_env = ManagerBasedRLEnv(env_cfg)
    wrapped_env = _EvalVecEnv(
        raw_env, args.num_envs,
        action_mode=_action_mode, delta_scale=_delta_scale,
        actions_moving_average=_actions_ma,
    )

    def create_env(**kwargs):
        return wrapped_env

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": create_env,
        },
    )
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: wrapped_env,
    )

    # --- Runner / play loop -------------------------------------------
    cfg = build_eval_rl_games_config(args, cfg_file)

    runner = Runner()
    runner.load(cfg)
    runner.reset()

    # rl_games "play" path instantiates a Player that loads the checkpoint.
    player = runner.create_player()
    player.restore(args.checkpoint)

    # Our wrapper already emits batched obs (num_envs, obs_dim). Tell the
    # Player not to add an extra batch dim (otherwise a2c_network.forward
    # calls .flatten(1) on (1, num_envs, obs_dim) → mismatched matmul).
    player.has_batch_dimension = True
    # Ensure Player uses our wrapped env (not a fresh raw ManagerBasedRLEnv).
    player.env = wrapped_env

    # Apply the curriculum state that matches the checkpoint's epoch.
    # Mirrors what _CompactIsaacAlgoObserver.after_init() does during
    # training, so the policy is evaluated in the exact same env it was
    # learning in at that checkpoint.
    from envs.mdp import events as mdp_events
    try:
        mdp_events.update_curriculum(raw_env, ckpt_epoch, args.max_iterations)
        applied_min_orn = getattr(merged_graph, "_curriculum_min_orn", None)
        applied_g = None
        try:
            import isaaclab.sim as sim_utils
            g_vec = sim_utils.SimulationContext.instance().physics_sim_view.get_gravity()
            applied_g = float((g_vec[0] ** 2 + g_vec[1] ** 2 + g_vec[2] ** 2) ** 0.5)
        except Exception:
            pass
        print(f"[Evaluate] Curriculum applied for epoch "
              f"{ckpt_epoch}/{args.max_iterations}"
              + (f"  gravity≈{applied_g:.3f}" if applied_g is not None else "")
              + (f"  min_orn={applied_min_orn:.3f}rad" if applied_min_orn is not None else ""))
    except Exception as _e:
        print(f"[Evaluate] WARNING: curriculum update failed: {_e}")

    vec_env = wrapped_env
    _run_eval_loop(
        player, vec_env,
        num_episodes=args.num_episodes,
        device=args.device,
        results_json=args.results_json,
        checkpoint_path=str(Path(args.checkpoint).resolve()),
    )

    sim_app.close()


# ---------------------------------------------------------------------------
# Evaluation-only vector env wrapper
# ---------------------------------------------------------------------------

class _EvalVecEnv(_IsaacLabVecEnv):
    """Same wrapper used in train_rl.py, but without curriculum updates."""

    def step(self, actions):
        # Identical to _IsaacLabVecEnv.step, but without touching curriculum.
        extras = self.env.extras
        if "current_action" in extras:
            extras["last_action"] = extras["current_action"].clone()
        else:
            extras["last_action"] = actions.clone()
        extras["current_action"] = actions.clone()

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
            env_actions = actions

        from envs.mdp.domain_rand import apply_action_delay
        delayed_actions = apply_action_delay(self.env, env_actions)

        obs, rew, terminated, truncated, info = self.env.step(delayed_actions)
        done = terminated | truncated

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

        n_updated = mdp_events.update_rolling_goal(self.env)
        self.env._last_rolling_goal_updates = n_updated

        info = dict(info) if isinstance(info, dict) else {}
        info["rolling_goal_updates"] = n_updated

        term_manager = getattr(self.env, "termination_manager", None)
        drop_ratio = 0.0
        left_hand_ratio = 0.0
        no_contact_ratio = 0.0
        if term_manager is not None:
            active = set(getattr(term_manager, "active_terms", []))
            if "object_drop" in active:
                drop_ratio = float(term_manager.get_term("object_drop").float().mean().item())
            if "object_left_hand" in active:
                left_hand_ratio = float(term_manager.get_term("object_left_hand").float().mean().item())
            if "no_fingertip_contact" in active:
                no_contact_ratio = float(term_manager.get_term("no_fingertip_contact").float().mean().item())

        info["drop_ratio"] = drop_ratio
        info["left_hand_ratio"] = left_hand_ratio
        info["no_contact_ratio"] = no_contact_ratio

        return _to_rl_obs(obs), rew, done, info


# ---------------------------------------------------------------------------
# Metric-collecting play loop
# ---------------------------------------------------------------------------

def _compute_orn_error(env) -> torch.Tensor:
    """Per-env object orientation error (rad) against the current target."""
    target_quat = env.extras.get("target_object_quat_hand")
    if target_quat is None:
        return torch.full((env.num_envs,), float("inf"), device=env.device)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    root_quat = robot.data.root_quat_w

    def _qc(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def _qm(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    cur_quat = _qm(_qc(root_quat), obj.data.root_quat_w)
    dot = (cur_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _run_eval_loop(player, vec_env, num_episodes: int, device: str, results_json: str | None,
                   checkpoint_path: str = "", goal_threshold: float = 0.4):
    """Manually drive the rl_games Player so we can collect per-episode metrics."""
    num_envs = vec_env.num_envs
    raw_env = vec_env.env  # inner ManagerBasedRLEnv

    # Episode accumulators (one entry per env slot, reset on done)
    ep_reward = torch.zeros(num_envs, device=device)
    ep_length = torch.zeros(num_envs, device=device, dtype=torch.long)
    ep_goal_hits = torch.zeros(num_envs, device=device, dtype=torch.long)
    ep_reached_goal = torch.zeros(num_envs, device=device, dtype=torch.bool)
    # Track per-env goal state across steps so a sustained "at goal" segment
    # only counts as one success (rising-edge detection).
    was_at_goal = torch.zeros(num_envs, device=device, dtype=torch.bool)

    finished_rewards: list[float] = []
    finished_lengths: list[int] = []
    finished_goal_hits: list[int] = []
    finished_reached: list[bool] = []
    drop_events = 0
    left_hand_events = 0
    no_contact_events = 0
    total_steps = 0

    obs_dict = vec_env.reset()
    is_rnn = getattr(player, "is_rnn", False)
    if is_rnn and hasattr(player, "init_rnn"):
        player.init_rnn()

    print("\n" + "=" * 60)
    print(f"[Evaluate] Rolling goal threshold: {goal_threshold:.2f} rad "
          f"(~{goal_threshold * 180 / 3.14159:.1f}°)")
    print("=" * 60)

    while len(finished_rewards) < num_episodes:
        obs_tensor = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
        with torch.no_grad():
            action = player.get_action(obs_tensor, is_deterministic=player.is_deterministic)

        obs_dict, rewards, dones, info = vec_env.step(action)

        ep_reward += rewards.to(device)
        ep_length += 1
        total_steps += 1

        # --- Goal success detection (rising edge of rot_dist < threshold) ---
        orn_err = _compute_orn_error(raw_env)
        at_goal = orn_err < goal_threshold
        new_goal_hits = at_goal & ~was_at_goal   # rising edge this step
        was_at_goal = at_goal

        if new_goal_hits.any():
            hit_ids = new_goal_hits.nonzero(as_tuple=False).squeeze(-1).tolist()
            for eid in hit_ids:
                err_deg = float(orn_err[eid].item()) * 180 / 3.14159
                print(f"  [✓ GOAL REACHED] step={total_steps}  env={eid}  "
                      f"err={err_deg:.1f}°  ep_len={int(ep_length[eid].item())}")
            ep_goal_hits[new_goal_hits] += 1
            ep_reached_goal[new_goal_hits] = True

        drop_events += float(info.get("drop_ratio", 0.0)) * num_envs
        left_hand_events += float(info.get("left_hand_ratio", 0.0)) * num_envs
        no_contact_events += float(info.get("no_contact_ratio", 0.0)) * num_envs

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
            for i in done_ids:
                finished_rewards.append(float(ep_reward[i].item()))
                finished_lengths.append(int(ep_length[i].item()))
                finished_goal_hits.append(int(ep_goal_hits[i].item()))
                finished_reached.append(bool(ep_reached_goal[i].item()))

                ep_idx = len(finished_rewards)
                marker = "SUCCESS" if ep_reached_goal[i].item() else "fail   "
                print(f"  [EP {ep_idx:4d}/{num_episodes}] env={i}  {marker}  "
                      f"reward={ep_reward[i].item():7.2f}  "
                      f"len={int(ep_length[i].item()):4d}  "
                      f"goal_hits={int(ep_goal_hits[i].item())}")

                ep_reward[i] = 0.0
                ep_length[i] = 0
                ep_goal_hits[i] = 0
                ep_reached_goal[i] = False
                was_at_goal[i] = False

                if len(finished_rewards) >= num_episodes:
                    break

            if is_rnn and hasattr(player, "reset"):
                player.reset()

    # Trim to exactly num_episodes (last batch may overshoot).
    finished_rewards = finished_rewards[:num_episodes]
    finished_lengths = finished_lengths[:num_episodes]
    finished_goal_hits = finished_goal_hits[:num_episodes]
    finished_reached = finished_reached[:num_episodes]

    rewards_np = np.array(finished_rewards, dtype=np.float64)
    lengths_np = np.array(finished_lengths, dtype=np.float64)
    hits_np = np.array(finished_goal_hits, dtype=np.float64)
    reached_np = np.array(finished_reached, dtype=bool)

    n_success = int(reached_np.sum())
    n_total = int(len(reached_np))

    summary = {
        "checkpoint": checkpoint_path,
        "goal_threshold_rad": float(goal_threshold),
        "num_episodes": n_total,
        "num_success": n_success,
        "success_rate": float(n_success / max(n_total, 1)),
        "mean_goal_hits_per_episode": float(hits_np.mean()) if len(hits_np) else 0.0,
        "total_goal_hits": int(hits_np.sum()),
        "mean_reward": float(rewards_np.mean()) if len(rewards_np) else 0.0,
        "std_reward": float(rewards_np.std()) if len(rewards_np) else 0.0,
        "mean_length": float(lengths_np.mean()) if len(lengths_np) else 0.0,
        "drop_events": float(drop_events),
        "left_hand_events": float(left_hand_events),
        "no_contact_events": float(no_contact_events),
        "total_steps": int(total_steps),
        "drop_rate_per_step": float(drop_events / max(total_steps * num_envs, 1)),
        "left_hand_rate_per_step": float(left_hand_events / max(total_steps * num_envs, 1)),
        "no_contact_rate_per_step": float(no_contact_events / max(total_steps * num_envs, 1)),
    }

    print("\n" + "=" * 60)
    print("[Evaluate] Results")
    print("=" * 60)
    print(f"  SUCCESS RATE:  {n_success}/{n_total}  ({summary['success_rate'] * 100:.1f}%)")
    print("-" * 60)
    for k, v in summary.items():
        if k in ("success_rate", "num_success", "num_episodes"):
            continue  # already printed above
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")
    print("=" * 60)

    if results_json:
        out_path = Path(results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Evaluate] Saved results to {out_path}")


if __name__ == "__main__":
    main()
