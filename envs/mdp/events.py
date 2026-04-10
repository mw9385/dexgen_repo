"""
Event (reset / randomisation) functions for the AnyGrasp-to-AnyGrasp env.

Reset logic per episode (DexterityGen §3.2):
  1. Sample start grasp + nearby goal grasp from the GraspGraph
  2. Set hand to START grasp configuration:
     a. Solved graph: set stored joint_angles + place object from hand-relative pose
     b. Unsolved graph: place object fixed, compute wrist, adaptive joints + IK
  3. Apply palm-up transform + wrist pose diversity (position jitter + tilt noise)
  4. Compute GOAL object pose as delta(start→goal) applied to actual sim state
     → start ≠ goal from step 0; policy must reorient toward goal immediately

Rolling goal: when orientation error < rot_thresh and object_dropped is false,
  pick a new nearby goal via kNN. ``rot_thresh`` matches ``rewards.goal_bonus``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np

from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: used in reset logic
from grasp_generation.graph_io import load_merged_graph, parse_graph_paths

from .math_utils import (
    quat_multiply,
    quat_conjugate,
    add_rotation_noise,
    local_to_world_points,
    world_to_local_points,
)
from .sim_utils import (
    set_robot_joints_direct,
    expand_grasp_joint_vector,
    set_robot_root_pose,
    get_palm_body_id_from_env,
    set_adaptive_joint_pose,
    apply_palm_up_transform,
    refine_hand_to_start_grasp,
    joint_positions_to_normalized_action,
    get_fingertip_body_ids_from_env,
)

# ---------------------------------------------------------------------------
# Termination predicates
# ---------------------------------------------------------------------------

def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def object_drop_max_dist_for_env(env, default: float = 0.08) -> float:
    """``max_dist`` from ``terminations.object_drop`` if set, else ``default``."""
    term = getattr(getattr(env.cfg, "terminations", None), "object_drop", None)
    if term is None:
        return default
    p = getattr(term, "params", None)
    if not p:
        return default
    get = p.get if hasattr(p, "get") else (lambda k, d=None: p[k] if k in p else d)
    v = get("max_dist", default)
    return float(v)


def object_dropped(env, max_dist: float | None = None) -> torch.Tensor:
    """Object dropped = palm–object distance exceeds ``max_dist`` (metres).

    If ``max_dist`` is None, uses :func:`object_drop_max_dist_for_env`.
    """
    if max_dist is None:
        max_dist = object_drop_max_dist_for_env(env)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    palm_body_id = get_palm_body_id_from_env(robot, env)
    palm_pos_w = robot.data.body_pos_w[:, palm_body_id, :]
    dist = torch.norm(obj.data.root_pos_w - palm_pos_w, dim=-1)
    return dist > max_dist

# ---------------------------------------------------------------------------
# Main reset event
# ---------------------------------------------------------------------------

def reset_to_random_grasp(
    env,
    env_ids: torch.Tensor,
):
    """
    Reset the environment from the grasp graph while preserving a stable
    initial object/hand pose. The start grasp must come directly from the
    Stage 0 grasp_set, and only the goal grasp is derived via nearest-neighbor
    search around that sampled start.
    """
    graph = _load_grasp_graph(env)
    if graph is None:
        _reset_to_default_pose(env, env_ids)
        return

    n = len(env_ids)
    # env_num_fingers is the FIXED finger count the env was built with.
    # All grasp data is padded/truncated to this count so tensors are uniform.
    env_num_fingers = int(
        (getattr(env.cfg, "hand", None) or {}).get("num_fingers", 4)
    )
    rng = _get_reset_rng(env)

    # ------------------------------------------------------------------
    # 1. Sample start grasps from grasp_set and goals via nearest neighbor
    #    Batch by object name to avoid per-env Python overhead.
    # ------------------------------------------------------------------
    sampled_obj_names = [None] * n
    forced_object_name = _resolve_scene_graph_object_name(env, graph, env_num_fingers)
    env_graph_names = _detect_env_graph_names(env, graph, env_num_fingers)

    # Resolve per-env object names
    for i in range(n):
        env_id = int(env_ids[i].item())
        if env_graph_names is not None:
            sampled_obj_names[i] = env_graph_names[env_id] or forced_object_name
        else:
            sampled_obj_names[i] = forced_object_name

    # Batch sample: group envs by object name, sample all at once per group
    start_idx_list = [0] * n
    goal_idx_list = [0] * n

    from grasp_generation.graph_io import MultiObjectGraspGraph
    # Group env indices by object name
    name_to_envs: dict = {}
    for i in range(n):
        name_to_envs.setdefault(sampled_obj_names[i], []).append(i)

    for obj_name, env_indices in name_to_envs.items():
        if isinstance(graph, MultiObjectGraspGraph):
            g = graph.graphs.get(obj_name)
            if g is None:
                g = next(iter(graph.graphs.values()))
        else:
            g = graph
        N_grasps = len(g)
        if N_grasps == 0:
            continue

        batch_size = len(env_indices)
        starts = rng.integers(0, N_grasps, size=batch_size)
        cur_min_orn = getattr(graph, "_curriculum_min_orn", 0.10)
        # Batch goal selection — single matrix op instead of per-env loop
        goals = _batch_sample_nearby_goals(
            g, starts, rng, min_orn=cur_min_orn, num_fingers=env_num_fingers,
        )
        for j, local_i in enumerate(env_indices):
            start_idx_list[local_i] = int(starts[j])
            goal_idx_list[local_i] = int(goals[j])

    # Extract grasp data in batch (one pass per object name group)
    start_joints_list = [None] * n
    start_object_pos_hand_list = [None] * n
    start_object_quat_hand_list = [None] * n
    start_object_pose_frame_list = [None] * n
    goal_object_pos_hand_list = [None] * n
    goal_object_quat_hand_list = [None] * n
    goal_object_pose_frame_list = [None] * n

    for obj_name, env_indices in name_to_envs.items():
        if isinstance(graph, MultiObjectGraspGraph):
            g = graph.graphs.get(obj_name)
            if g is None:
                g = next(iter(graph.graphs.values()))
        else:
            g = graph

        # Build cached arrays once per sub-graph (numpy, fast indexing)
        if not hasattr(g, "_cached_joints"):
            grasps = g.grasp_set.grasps
            g._cached_joints = np.stack([
                gr.joint_angles if gr.joint_angles is not None
                else np.zeros(22, dtype=np.float32) for gr in grasps
            ])
            g._cached_obj_pos = np.stack([
                gr.object_pos_hand if gr.object_pos_hand is not None
                else np.zeros(3, dtype=np.float32) for gr in grasps
            ])
            g._cached_obj_quat = np.stack([
                gr.object_quat_hand if gr.object_quat_hand is not None
                else np.array([1, 0, 0, 0], dtype=np.float32) for gr in grasps
            ])

        si_arr = np.array([start_idx_list[i] for i in env_indices])
        gi_arr = np.array([goal_idx_list[i] for i in env_indices])

        s_joints = g._cached_joints[si_arr]     # (B, 22)
        s_pos    = g._cached_obj_pos[si_arr]     # (B, 3)
        s_quat   = g._cached_obj_quat[si_arr]    # (B, 4)
        g_pos    = g._cached_obj_pos[gi_arr]     # (B, 3)
        g_quat   = g._cached_obj_quat[gi_arr]    # (B, 4)

        for j, local_i in enumerate(env_indices):
            start_joints_list[local_i] = s_joints[j]
            start_object_pos_hand_list[local_i] = s_pos[j]
            start_object_quat_hand_list[local_i] = s_quat[j]
            start_object_pose_frame_list[local_i] = "hand_root"
            goal_object_pos_hand_list[local_i] = g_pos[j]
            goal_object_quat_hand_list[local_i] = g_quat[j]
            goal_object_pose_frame_list[local_i] = "hand_root"

    if "start_grasp_idx" not in env.extras:
        env.extras["start_grasp_idx"] = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    if "goal_grasp_idx" not in env.extras:
        env.extras["goal_grasp_idx"] = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )

    # Store per-env object name for rolling goal re-selection
    if "env_obj_names" not in env.extras:
        env.extras["env_obj_names"] = [None] * env.num_envs
    for i, env_id in enumerate(env_ids.tolist()):
        env.extras["env_obj_names"][env_id] = sampled_obj_names[i]
    env.extras["start_grasp_idx"][env_ids] = torch.tensor(
        start_idx_list, device=env.device, dtype=torch.long
    )
    env.extras["goal_grasp_idx"][env_ids] = torch.tensor(
        goal_idx_list, device=env.device, dtype=torch.long
    )

    # Store goal object pose in hand frame for object_pose_reward
    if "target_object_pos_hand" not in env.extras:
        env.extras["target_object_pos_hand"] = torch.zeros(
            env.num_envs, 3, device=env.device
        )
    if "target_object_quat_hand" not in env.extras:
        env.extras["target_object_quat_hand"] = torch.zeros(
            env.num_envs, 4, device=env.device
        )
        env.extras["target_object_quat_hand"][:, 0] = 1.0  # identity quat

    for i, (gp, gq) in enumerate(zip(goal_object_pos_hand_list, goal_object_quat_hand_list)):
        if gp is not None:
            env.extras["target_object_pos_hand"][env_ids[i]] = torch.tensor(
                gp, device=env.device, dtype=torch.float32
            )
        if gq is not None:
            env.extras["target_object_quat_hand"][env_ids[i]] = torch.tensor(
                gq, device=env.device, dtype=torch.float32
            )

    # ------------------------------------------------------------------
    # 2. Reset directly to the stored start grasp.
    #    No additional pose randomization or reset-time pose reshaping.
    # ------------------------------------------------------------------
    robot = env.scene["robot"]
    obj = env.scene["object"]
    has_stored_reset = (
        all(j is not None for j in start_joints_list)
        and all(p is not None for p in start_object_pos_hand_list)
        and all(q is not None for q in start_object_quat_hand_list)
        and all(f == "hand_root" for f in start_object_pose_frame_list)
    )

    if not has_stored_reset:
        raise RuntimeError(
            "Reset grasp data must contain stored joint angles and hand-frame object poses. "
            "Regenerate the .npy grasp data with scripts/gen_grasp.py."
        )

    wrist_pos = (
        robot.data.default_root_state[env_ids, :3].clone()
        + env.scene.env_origins[env_ids]
    )
    wrist_quat = robot.data.default_root_state[env_ids, 3:7].clone()
    set_robot_root_pose(env, env_ids, wrist_pos, wrist_quat)

    set_robot_joints_direct(env, env_ids, start_joints_list)
    robot.update(0.0)

    start_pos_hand_t = torch.tensor(
        np.stack(start_object_pos_hand_list), device=env.device, dtype=torch.float32
    )
    start_quat_hand_t = torch.tensor(
        np.stack(start_object_quat_hand_list), device=env.device, dtype=torch.float32
    )
    goal_pos_hand_t = torch.tensor(
        np.stack(goal_object_pos_hand_list), device=env.device, dtype=torch.float32
    )
    goal_quat_hand_t = torch.tensor(
        np.stack(goal_object_quat_hand_list), device=env.device, dtype=torch.float32
    )

    obj_pos_w = wrist_pos + quat_apply(wrist_quat, start_pos_hand_t)
    obj_quat_w = quat_multiply(wrist_quat, start_quat_hand_t)
    obj_quat_w = obj_quat_w / (torch.norm(obj_quat_w, dim=-1, keepdim=True) + 1e-8)

    obj_root_state = obj.data.default_root_state[env_ids].clone()
    obj_root_state[:, :3] = obj_pos_w
    obj_root_state[:, 3:7] = obj_quat_w
    obj_root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(obj_root_state, env_ids=env_ids)
    obj.update(0.0)

    # Step 7: Goal object pose in hand frame comes directly from the sampled goal grasp.
    robot.update(0.0)
    obj.update(0.0)
    env.extras["target_object_pos_hand"][env_ids] = goal_pos_hand_t
    env.extras["target_object_quat_hand"][env_ids] = goal_quat_hand_t

    # ------------------------------------------------------------------
    # Log start→goal distances (position & orientation) at reset
    # ------------------------------------------------------------------
    _log_goal_distances(env, env_ids)

    # ------------------------------------------------------------------
    # 8. Initialise action buffers
    # ------------------------------------------------------------------
    current_q = robot.data.joint_pos[env_ids]   # (n, num_dof=24)
    current_action_full = joint_positions_to_normalized_action(robot, env_ids, current_q)
    action_dim = _get_action_dim(env, current_q.shape[-1])
    current_action = current_action_full[:, (current_q.shape[-1] - action_dim):]
    if "last_action" not in env.extras:
        env.extras["last_action"] = torch.zeros(
            env.num_envs, action_dim, device=env.device
        )
    if "current_action" not in env.extras:
        env.extras["current_action"] = torch.zeros(
            env.num_envs, action_dim, device=env.device
        )
    env.extras["last_action"][env_ids] = current_action
    env.extras["current_action"][env_ids] = current_action

    # Reset delta reward prev-error buffers
    if "_prev_orn_error" in env.extras:
        from .rewards import _get_orn_error
        env.extras["_prev_orn_error"][env_ids] = _get_orn_error(env)[env_ids]

# ---------------------------------------------------------------------------
# Nearest-neighbor goal selection
# ---------------------------------------------------------------------------

def _batch_sample_nearby_goals(
    graph, start_indices: np.ndarray, rng: np.random.Generator,
    top_k: int = 5, min_orn: float = 0.50,
    max_pos: float = 0.05,
    num_fingers: int = 5,
) -> np.ndarray:
    """Fully vectorised goal sampling — no Python per-env loop.

    Returns: np.ndarray of shape (B,) with goal indices.
    """
    N = len(graph.grasp_set.grasps)
    B = len(start_indices)
    if N <= 1:
        return start_indices.copy()

    all_quats = _get_cached_quats(graph)   # (N, 4) or None
    all_pos = _get_cached_positions(graph)  # (N, 3) or None

    if all_quats is None and all_pos is None:
        return rng.integers(0, N, size=B)

    # Score matrix (B, N) — lower = closer. Invalid entries get inf.
    score = np.zeros((B, N), dtype=np.float32)

    if all_quats is not None:
        start_q = all_quats[start_indices]           # (B, 4)
        dots = np.abs(start_q @ all_quats.T)         # (B, N)
        np.clip(dots, 0.0, 1.0, out=dots)
        orn_dists = 2.0 * np.arccos(dots)            # (B, N)
        # Filter: must be >= min_orn
        score += (orn_dists / max(min_orn, 1e-6)).astype(np.float32)
        too_close = orn_dists < min_orn
        score[too_close] = np.inf

    if all_pos is not None:
        start_p = all_pos[start_indices]              # (B, 3)
        # Chunked position distance to avoid (B, N, 3) memory blow-up
        pos_dists = np.empty((B, N), dtype=np.float32)
        chunk = 512
        for i in range(0, B, chunk):
            end = min(i + chunk, B)
            diff = all_pos[np.newaxis, :, :] - start_p[i:end, np.newaxis, :]
            pos_dists[i:end] = np.linalg.norm(diff, axis=-1)
        # Filter: must be <= max_pos
        score += (pos_dists / max(max_pos, 1e-6)).astype(np.float32)
        too_far = pos_dists > max_pos
        score[too_far] = np.inf

    # Mask self
    score[np.arange(B), start_indices] = np.inf

    # Fallback: if entire row is inf, allow any non-self index
    all_inf = np.all(np.isinf(score), axis=1)
    if all_inf.any():
        # For these rows, set all non-self to 0 (equal chance)
        fallback_score = np.zeros((int(all_inf.sum()), N), dtype=np.float32)
        fb_starts = start_indices[all_inf]
        fallback_score[np.arange(len(fb_starts)), fb_starts] = np.inf
        score[all_inf] = fallback_score

    # Batch top-k via argpartition
    k = min(top_k, N - 1)
    # argpartition along axis=1: first k elements are the k smallest
    partitioned = np.argpartition(score, k - 1, axis=1)[:, :k]  # (B, k)

    # Random select one from top-k per row
    choices = rng.integers(0, k, size=B)
    goals = partitioned[np.arange(B), choices]

    return goals


def _sample_nearby_goal_index(
    graph, start_idx: int, rng: np.random.Generator,
    top_k: int = 5, min_orn: float = 0.50,
    max_pos: float = 0.05,
    num_fingers: int = 5,
) -> int:
    """
    Sample a nearby goal grasp using K-nearest neighbors in pose space.

    Args:
        min_orn: minimum orientation distance (rad) to avoid trivial goals.
        max_pos: maximum hand-frame position delta (m) for "nearby" goals.
    """
    grasps = graph.grasp_set.grasps
    N = len(grasps)
    if N <= 1:
        return start_idx

    start_quat = getattr(grasps[start_idx], "object_quat_hand", None)
    if start_quat is None:
        idx = int(rng.integers(0, N))
        return idx if idx != start_idx else (start_idx + 1) % N

    # Build cached pose arrays
    all_quats = _get_cached_quats(graph)
    all_pos = _get_cached_positions(graph)
    if all_quats is None and all_pos is None:
        idx = int(rng.integers(0, N))
        return idx if idx != start_idx else (start_idx + 1) % N

    if all_quats is not None:
        sq = all_quats[start_idx]
        dots = np.abs(np.dot(all_quats, sq))
        np.clip(dots, 0.0, 1.0, out=dots)
        orn_dists = 2.0 * np.arccos(dots)
        orn_dists[start_idx] = np.inf
    else:
        orn_dists = np.full(N, np.inf, dtype=np.float64)
        orn_dists[start_idx] = np.inf

    if all_pos is not None:
        sp = all_pos[start_idx]
        pos_dists = np.linalg.norm(all_pos - sp, axis=-1)
        pos_dists[start_idx] = np.inf
    else:
        pos_dists = np.full(N, np.inf, dtype=np.float64)
        pos_dists[start_idx] = np.inf

    valid_mask = np.ones(N, dtype=bool)
    valid_mask[start_idx] = False
    if all_quats is not None:
        valid_mask &= np.isfinite(orn_dists)
        valid_mask &= orn_dists >= min_orn
    if all_pos is not None:
        valid_mask &= np.isfinite(pos_dists)
        valid_mask &= pos_dists <= max_pos

    valid_idx = np.where(valid_mask)[0]

    # Progressive fallback to preserve KNN behavior when the nearby set is empty.
    if len(valid_idx) == 0 and all_pos is not None and all_quats is not None:
        valid_idx = np.where(
            (np.arange(N) != start_idx)
            & np.isfinite(pos_dists)
            & np.isfinite(orn_dists)
            & (orn_dists >= min_orn)
        )[0]
    if len(valid_idx) == 0 and all_pos is not None:
        valid_idx = np.where((np.arange(N) != start_idx) & np.isfinite(pos_dists))[0]
    if len(valid_idx) == 0 and all_quats is not None:
        valid_idx = np.where(
            (np.arange(N) != start_idx)
            & np.isfinite(orn_dists)
            & (orn_dists >= min_orn)
        )[0]
    if len(valid_idx) == 0 and all_quats is not None:
        valid_idx = np.where((np.arange(N) != start_idx) & np.isfinite(orn_dists))[0]
    if len(valid_idx) == 0:
        valid_idx = np.where(np.arange(N) != start_idx)[0]
    if len(valid_idx) == 0:
        return start_idx

    # KNN in pose space: smaller position/orientation deltas are closer.
    pose_score = np.zeros(len(valid_idx), dtype=np.float64)
    if all_pos is not None:
        pose_score += pos_dists[valid_idx] / max(max_pos, 1e-6)
    if all_quats is not None:
        pose_score += orn_dists[valid_idx] / max(min_orn, 1e-6)

    k = min(top_k, len(valid_idx))
    top_k_local = np.argpartition(pose_score, k - 1)[:k]
    top_k_indices = valid_idx[top_k_local]
    return int(rng.choice(top_k_indices))


def _get_cached_quats(graph) -> Optional[np.ndarray]:
    """Cache normalized quaternion array on the graph object."""
    if hasattr(graph, "_cached_quats"):
        return graph._cached_quats
    grasps = graph.grasp_set.grasps
    quats = []
    for g in grasps:
        q = getattr(g, "object_quat_hand", None)
        if q is None:
            graph._cached_quats = None
            return None
        quats.append(np.array(q, dtype=np.float64))
    arr = np.stack(quats)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    arr = arr / (norms + 1e-8)
    graph._cached_quats = arr
    return arr

def _get_cached_positions(graph) -> Optional[np.ndarray]:
    """Cache hand-frame object positions on the graph object."""
    if hasattr(graph, "_cached_positions"):
        return graph._cached_positions
    grasps = graph.grasp_set.grasps
    pos = []
    for g in grasps:
        p = getattr(g, "object_pos_hand", None)
        if p is None:
            graph._cached_positions = None
            return None
        pos.append(np.array(p, dtype=np.float64))
    graph._cached_positions = np.stack(pos)
    return graph._cached_positions

# ---------------------------------------------------------------------------
# Curriculum: gradually increase goal difficulty
# ---------------------------------------------------------------------------

def update_curriculum(env, epoch: int, total_epochs: int = 10000):
    """
    Update gravity and goal difficulty over training.

    - Gravity: ramp from start_gravity to end_gravity
    - Goal difficulty (min_orn): ramp from min_orn_start to min_orn_end

    Both ramp linearly over warmup_ratio fraction of total_epochs.
    Call once per epoch from the training loop.

    Gravity updates run whenever ``training_curriculum.enabled`` is true.
    Orientation ``_curriculum_min_orn`` is stored on the merged grasp graph only
    when a graph is loaded (no-op if missing).
    """
    cur_cfg = dict((getattr(env.cfg, "training_curriculum", None) or {})
                   or (getattr(env.cfg, "gravity_curriculum", None) or {}))
    warmup_ratio = float(cur_cfg.get("warmup_ratio", 0.10))
    warmup_epochs = int(total_epochs * warmup_ratio)
    t = min(epoch / max(warmup_epochs, 1), 1.0)

    graph = _load_grasp_graph(env)
    if graph is not None:
        # Orientation curriculum: increase min goal distance over warmup (kNN goals)
        min_orn_start = float(cur_cfg.get("min_orn_start", 0.10))
        min_orn_end = float(cur_cfg.get("min_orn_end", 0.50))
        graph._curriculum_min_orn = min_orn_start + t * (min_orn_end - min_orn_start)

    # Gravity curriculum: ramp from near-zero to full gravity
    if cur_cfg.get("enabled", False):
        gravity_start = float(cur_cfg.get("start_gravity", 0.05))
        gravity_end = float(cur_cfg.get("end_gravity", 9.81))
        gravity_warmup_epochs = int(total_epochs * warmup_ratio)
        gravity_t = min(epoch / max(gravity_warmup_epochs, 1), 1.0)
        gravity_mag = gravity_start + gravity_t * (gravity_end - gravity_start)
        try:
            import carb
            import isaaclab.sim as sim_utils

            sim_utils.SimulationContext.instance().physics_sim_view.set_gravity(
                carb.Float3(0.0, 0.0, -gravity_mag)
            )
            env._curriculum_gravity = gravity_mag
        except Exception as exc:
            if not getattr(env, "_curriculum_gravity_warned", False):
                print(f"[WARNING] Gravity curriculum update failed: {exc}")
                env._curriculum_gravity_warned = True

# ---------------------------------------------------------------------------
# Rolling goal: update goal when current goal is achieved mid-episode
# ---------------------------------------------------------------------------

def _goal_bonus_params_get(env, key: str, default):
    """Read a scalar from ``env.cfg.rewards.goal_bonus.params`` (OmegaConf-safe)."""
    rw = getattr(env.cfg, "rewards", None)
    if rw is None:
        return default
    gb = getattr(rw, "goal_bonus", None)
    if gb is None:
        return default
    p = getattr(gb, "params", None)
    if p is None:
        return default
    get = p.get if hasattr(p, "get") else (lambda k, d=None: p[k] if k in p else d)
    v = get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def goal_rot_thresh_from_env(env, default: float = 0.4) -> float:
    """``rot_thresh`` from ``env.cfg.rewards.goal_bonus`` (matches ``goal_bonus`` reward)."""
    return _goal_bonus_params_get(env, "rot_thresh", default)


def update_rolling_goal(
    env,
    rot_threshold: float | None = None,
) -> int:
    """
    Called every control step from train_rl / evaluate after env.step().

    For each env where orientation / position error are below the same
    thresholds as ``goal_bonus`` and ``object_dropped`` is false, samples a new
    nearby goal from the grasp graph and updates ``target_object_*``.

    If ``rot_threshold`` is omitted, uses ``goal_rot_thresh_from_env`` (same as
    ``rewards.goal_bonus.params['rot_thresh']``). Position uses
    ``rewards.goal_bonus.params['pos_thresh']`` (default 0.05 m).

    Returns:
        Number of envs whose goal was updated this step.
    """
    if rot_threshold is None:
        rot_threshold = goal_rot_thresh_from_env(env)
    pos_threshold = _goal_bonus_params_get(env, "pos_thresh", 0.05)

    _zero_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    graph = _load_grasp_graph(env)
    if graph is None:
        env.extras["_rolling_goal_success_mask"] = _zero_mask
        return 0

    goal_idx_buf  = env.extras.get("goal_grasp_idx")
    obj_names     = env.extras.get("env_obj_names")
    target_pos    = env.extras.get("target_object_pos_hand")
    target_quat   = env.extras.get("target_object_quat_hand")
    if goal_idx_buf is None or target_pos is None or target_quat is None:
        env.extras["_rolling_goal_success_mask"] = _zero_mask
        return 0

    # Same error metrics as rewards.goal_bonus (single source of truth).
    from . import rewards as mdp_rewards

    orn_err = mdp_rewards._get_orn_error(env)
    pos_err = mdp_rewards._get_pos_error(env)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    def _qc(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    def _qm(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

    dropped = object_dropped(env)
    success_mask = (orn_err < rot_threshold) & (pos_err < pos_threshold) & ~dropped

    # Debug: log first few goal hits to diagnose false positives
    if success_mask.any():
        _sids = success_mask.nonzero(as_tuple=False).squeeze(-1)[:5]
        for _sid in _sids.tolist():
            print(f"  [GOAL HIT DEBUG] env={_sid}  orn_err={orn_err[_sid].item():.4f}rad  "
                  f"pos_err={pos_err[_sid].item():.4f}m  dropped={dropped[_sid].item()}  "
                  f"step={env.episode_length_buf[_sid].item()}")

    # Store per-env success mask so evaluate.py can read it after step().
    # Must be stored BEFORE the target is updated below.
    env.extras["_rolling_goal_success_mask"] = success_mask.clone()

    success_ids = success_mask.nonzero(as_tuple=False).squeeze(-1)
    if success_ids.numel() == 0:
        return 0

    rng = _get_reset_rng(env)
    env_num_fingers = int(
        (getattr(env.cfg, "hand", None) or {}).get("num_fingers", 4)
    )

    from grasp_generation.graph_io import MultiObjectGraspGraph

    updated = 0
    for env_id in success_ids.tolist():
        cur_goal_idx = int(goal_idx_buf[env_id].item())

        # Pick the sub-graph for this env
        obj_name = (obj_names[env_id] if obj_names else None)
        if isinstance(graph, MultiObjectGraspGraph):
            g = graph.graphs.get(obj_name) if obj_name else None
            if g is None:
                # fallback: pick any graph
                g = next(iter(graph.graphs.values()))
        else:
            g = graph

        # kNN from current goal → new goal (use curriculum min_orn)
        cur_min_orn = getattr(graph, "_curriculum_min_orn", 0.10)
        new_goal_idx = _sample_nearby_goal_index(
            g, cur_goal_idx, rng, min_orn=cur_min_orn, num_fingers=env_num_fingers,
        )
        new_goal_grasp = g.grasp_set[new_goal_idx]

        # Rebase target: delta between old goal (now start) and new goal
        # stored object poses, applied to current actual sim pose.
        old_goal_grasp = g.grasp_set[cur_goal_idx]
        old_goal_obj_pos  = getattr(old_goal_grasp, "object_pos_hand",  None)
        old_goal_obj_quat = getattr(old_goal_grasp, "object_quat_hand", None)
        new_goal_obj_pos  = getattr(new_goal_grasp, "object_pos_hand",  None)
        new_goal_obj_quat = getattr(new_goal_grasp, "object_quat_hand", None)

        # Current actual object pose in hand frame
        _rp = robot.data.root_pos_w[env_id]
        _rq = robot.data.root_quat_w[env_id]
        _op = obj.data.root_pos_w[env_id]
        _oq = obj.data.root_quat_w[env_id]
        _actual_pos = quat_apply_inverse(_rq.unsqueeze(0), (_op - _rp).unsqueeze(0))[0]
        _actual_quat = _qm(_qc(_rq.unsqueeze(0)), _oq.unsqueeze(0))[0]

        if new_goal_obj_pos is not None and old_goal_obj_pos is not None \
                and "target_object_pos_hand" in env.extras:
            delta_pos = torch.tensor(new_goal_obj_pos, device=env.device, dtype=torch.float32) \
                      - torch.tensor(old_goal_obj_pos, device=env.device, dtype=torch.float32)
            env.extras["target_object_pos_hand"][env_id] = _actual_pos + delta_pos
        if new_goal_obj_quat is not None and old_goal_obj_quat is not None \
                and "target_object_quat_hand" in env.extras:
            _sq = torch.tensor(old_goal_obj_quat, device=env.device, dtype=torch.float32).unsqueeze(0)
            _gq = torch.tensor(new_goal_obj_quat, device=env.device, dtype=torch.float32).unsqueeze(0)
            delta_quat = _qm(_qc(_sq), _gq)[0]
            target_quat = _qm(_actual_quat.unsqueeze(0), delta_quat.unsqueeze(0))[0]
            target_quat = target_quat / (torch.norm(target_quat) + 1e-8)
            env.extras["target_object_quat_hand"][env_id] = target_quat

        # Update indices: old goal becomes new start
        env.extras["goal_grasp_idx"][env_id]  = new_goal_idx
        env.extras["start_grasp_idx"][env_id] = cur_goal_idx
        if obj_names:
            obj_names[env_id] = obj_name  # unchanged but keep in sync

        updated += 1

    # Track cumulative goal hits for the per-reset training log.
    env.extras["_reset_log_goal_hits_window"] = (
        int(env.extras.get("_reset_log_goal_hits_window", 0)) + updated
    )
    env.extras["_reset_log_goal_hits_total"] = (
        int(env.extras.get("_reset_log_goal_hits_total", 0)) + updated
    )

    return updated

def _resolve_scene_graph_object_name(
    env, graph, env_num_fingers: int = 4
) -> Optional[str]:
    """
    Choose the graph object that best matches the currently configured scene
    object AND the env's finger count.

    When Stage 0 generates f=2/3/4 graphs with names like "cube_0.06_f2",
    this ensures Stage 1 only samples from graphs whose num_fingers == env
    finger count.
    """
    from grasp_generation.graph_io import MultiObjectGraspGraph

    if not isinstance(graph, MultiObjectGraspGraph):
        return None

    object_cfg = getattr(getattr(env.cfg, "scene", None), "object", None)
    spawn_cfg = getattr(object_cfg, "spawn", None)
    if spawn_cfg is None:
        return None

    shape_type = None
    size = None

    cfg_name = type(spawn_cfg).__name__.lower()
    if "cuboid" in cfg_name:
        shape_type = "cube"
        dims = getattr(spawn_cfg, "size", None)
        if dims is not None:
            size = float(max(dims))
    elif "sphere" in cfg_name:
        shape_type = "sphere"
        radius = getattr(spawn_cfg, "radius", None)
        if radius is not None:
            size = float(radius) * 2.0
    elif "cylinder" in cfg_name:
        shape_type = "cylinder"
        height = getattr(spawn_cfg, "height", None)
        radius = getattr(spawn_cfg, "radius", None)
        if height is not None:
            size = float(height)
        elif radius is not None:
            size = float(radius) * 2.0

    if shape_type is None:
        # No shape match possible — fall back to matching only by num_fingers
        best_name = None
        for name, g in graph.graphs.items():
            if g.num_fingers == env_num_fingers:
                best_name = name
                break
        return best_name

    # Primary: match shape_type + num_fingers; rank by size closeness
    best_name  = None
    best_score = float("inf")
    for name, spec in graph.object_specs.items():
        if spec.get("shape_type") != shape_type:
            continue
        # Filter by num_fingers: only consider graphs matching env count
        g = graph.graphs.get(name)
        if g is not None and g.num_fingers != env_num_fingers:
            continue

        spec_size = spec.get("size")
        if spec_size is None or size is None:
            return name

        score = abs(float(spec_size) - size)
        if score < best_score:
            best_score = score
            best_name = name

    # Fallback: if no matching num_fingers graph found, ignore finger filter
    if best_name is None:
        for name, spec in graph.object_specs.items():
            if spec.get("shape_type") != shape_type:
                continue
            spec_size = spec.get("size")
            if spec_size is None or size is None:
                return name
            score = abs(float(spec_size) - size)
            if score < best_score:
                best_score = score
                best_name = name

    return best_name

def _detect_shape_type_from_prim(prim, depth: int = 0) -> Optional[str]:
    """
    Recursively inspect a USD prim (up to 4 levels) to detect cube/sphere/cylinder.

    Isaac Lab primitive spawners set the prim type directly on the object prim
    (e.g. typeName "Cube", "Sphere", "Cylinder") or on a direct child when the
    root prim is an Xform.
    """
    if prim is None or not prim.IsValid():
        return None
    type_lower = prim.GetTypeName().lower()
    if "cube" in type_lower:
        return "cube"
    if "sphere" in type_lower:
        return "sphere"
    if "cylinder" in type_lower:
        return "cylinder"
    if depth < 4:
        for child in prim.GetChildren():
            result = _detect_shape_type_from_prim(child, depth + 1)
            if result:
                return result
    return None

def _detect_env_graph_names(
    env, graph, env_num_fingers: int = 4
) -> Optional[list]:
    """
    Return a per-env list of MultiObjectGraspGraph object names by inspecting
    the USD stage to determine which shape was actually spawned in each env.

    Result is cached in ``env.extras["_env_graph_names"]`` after the first call
    so USD inspection only happens once per training run.

    Returns None if ``graph`` is not a MultiObjectGraspGraph.
    """
    from grasp_generation.graph_io import MultiObjectGraspGraph

    if not isinstance(graph, MultiObjectGraspGraph):
        return None

    # Return cached result if already detected
    if "_env_graph_names" in env.extras:
        return env.extras["_env_graph_names"]

    # Build shape_type → list-of-graph-names for the correct finger count
    names_by_shape: dict = {}
    for name, spec in graph.object_specs.items():
        g = graph.graphs.get(name)
        if g is None:
            continue
        if g.num_fingers != env_num_fingers:
            continue
        shape = spec.get("shape_type", "cube")
        names_by_shape.setdefault(shape, []).append(name)

    # Default fallback: first graph entry matching env_num_fingers
    default_name: Optional[str] = None
    for name, g in graph.graphs.items():
        if g.num_fingers == env_num_fingers:
            default_name = name
            break

    env_names: list = [default_name] * env.num_envs

    try:
        import omni.usd  # only available inside Isaac Sim
        stage = omni.usd.get_context().get_stage()

        for env_i in range(env.num_envs):
            obj_path = f"/World/envs/env_{env_i}/Object"
            prim = stage.GetPrimAtPath(obj_path)
            shape = _detect_shape_type_from_prim(prim)
            if shape and shape in names_by_shape:
                env_names[env_i] = names_by_shape[shape][0]
            # else: keep default_name

        print(
            f"[reset] Per-env object types detected via USD stage "
            f"({len(set(env_names))} distinct objects across {env.num_envs} envs)"
        )
    except Exception as e:
        print(f"[WARNING] _detect_env_graph_names: USD inspection failed ({e}); "
              f"all envs will use '{default_name}'")

    env.extras["_env_graph_names"] = env_names
    return env_names

# ---------------------------------------------------------------------------
# Robot joint setting — direct (when joint angles are stored in grasp)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# New reset helpers: object-fixed → wrist → IK → palm-up
# ---------------------------------------------------------------------------

def _get_action_dim(env, num_dof: int) -> int:
    """Return the action-space dimensionality (wrist joints excluded)."""
    try:
        return env.action_manager.action.shape[-1]
    except (AttributeError, RuntimeError):
        # Sharpa: all 22 DOF are action space. Shadow: exclude 2 wrist.
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        if hand_cfg.get("name") == "shadow" and num_dof == 24:
            return num_dof - 2
        return num_dof

def _reset_to_default_pose(env, env_ids: torch.Tensor):
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    num_dof = joint_pos.shape[-1]
    action_full = joint_positions_to_normalized_action(robot, env_ids, joint_pos)
    action_dim = _get_action_dim(env, num_dof)
    default_action = action_full[:, (num_dof - action_dim):]  # drop wrist
    env.extras["last_action"] = torch.zeros(env.num_envs, action_dim, device=env.device)
    env.extras["current_action"] = torch.zeros(env.num_envs, action_dim, device=env.device)
    env.extras["last_action"][env_ids] = default_action
    env.extras["current_action"][env_ids] = default_action

# ---------------------------------------------------------------------------
# Cached grasp graph loader
# ---------------------------------------------------------------------------

_GRASP_GRAPH_CACHE: dict = {}
_RESET_RNG_CACHE: dict = {}

def _get_reset_rng(env) -> np.random.Generator:
    """Cache one RNG per environment instance for reproducible reset sampling."""
    key = id(env)
    if key not in _RESET_RNG_CACHE:
        seed = getattr(env.cfg, "seed", None)
        _RESET_RNG_CACHE[key] = np.random.default_rng(seed)
    return _RESET_RNG_CACHE[key]

def _load_grasp_graph(env):
    """Load and cache the MultiObjectGraspGraph (or GraspGraph) from cfg."""
    graph_paths = parse_graph_paths(getattr(env.cfg, "grasp_graph_path", None))
    if not graph_paths:
        return None
    cache_key = tuple(graph_paths)
    if cache_key in _GRASP_GRAPH_CACHE:
        return _GRASP_GRAPH_CACHE[cache_key]

    for graph_path in graph_paths:
        if not Path(graph_path).exists():
            print(f"[events] Warning: GraspGraph not found at {graph_path}. "
                  f"Run scripts/run_grasp_generation.py first.")
            return None

    graph = load_merged_graph(graph_paths)
    _GRASP_GRAPH_CACHE[cache_key] = graph
    graph.summary()
    return graph


# ---------------------------------------------------------------------------
# Goal distance logging
# ---------------------------------------------------------------------------

_GOAL_LOG_COUNT = 0

def _log_goal_distances(env, env_ids: torch.Tensor):
    """Log per-reset metrics: start→goal distances, termination (drop/timeout),
    and cumulative rolling-goal hit counts."""
    global _GOAL_LOG_COUNT
    _GOAL_LOG_COUNT += 1

    target_pos = env.extras.get("target_object_pos_hand")
    target_quat = env.extras.get("target_object_quat_hand")
    if target_pos is None or target_quat is None:
        return

    n = len(env_ids)
    robot = env.scene["robot"]
    obj = env.scene["object"]

    env_ids_list = env_ids.tolist()
    eids_t = env_ids.to(dtype=torch.long)
    cur_rp = robot.data.root_pos_w[eids_t]
    cur_rq = robot.data.root_quat_w[eids_t]
    cur_op = obj.data.root_pos_w[eids_t]
    cur_oq = obj.data.root_quat_w[eids_t]
    cur_pos = quat_apply_inverse(cur_rq, cur_op - cur_rp)
    cur_quat = quat_multiply(quat_conjugate(cur_rq), cur_oq)

    pos_err = torch.norm(cur_pos - target_pos[eids_t], dim=-1)
    dots = (cur_quat * target_quat[eids_t]).sum(dim=-1).abs().clamp(0.0, 1.0)
    orn_err = 2.0 * torch.acos(dots)
    pos_dists = pos_err.cpu().tolist()
    orn_dists = orn_err.cpu().tolist()

    # --- Termination reason breakdown for the envs being reset ---
    tm = getattr(env, "termination_manager", None)
    n_drop = n_timeout = 0
    if tm is not None:
        active = set(getattr(tm, "active_terms", []))
        try:
            if "object_drop" in active:
                n_drop = int(tm.get_term("object_drop")[eids_t].sum().item())
        except Exception:
            pass
        n_timeout = max(n - n_drop, 0)

    # --- Cumulative counters across training ---
    stats = env.extras.setdefault("_reset_log_stats", {
        "total_resets": 0,
        "total_drops": 0,
        "total_timeouts": 0,
    })
    stats["total_resets"] += n
    stats["total_drops"] += n_drop
    stats["total_timeouts"] += n_timeout

    goal_window = int(env.extras.pop("_reset_log_goal_hits_window", 0))
    goal_total = int(env.extras.get("_reset_log_goal_hits_total", 0))

    drop_rate = stats["total_drops"] / max(stats["total_resets"], 1)

    print(
        f"[Reset #{_GOAL_LOG_COUNT}] ({n} envs) "
        f"pos={np.mean(pos_dists):.4f}m [{np.min(pos_dists):.4f}-{np.max(pos_dists):.4f}] "
        f"orn={np.mean(orn_dists):.2f}rad [{np.min(orn_dists):.2f}-{np.max(orn_dists):.2f}]  "
        f"term: drop={n_drop} timeout={n_timeout}  "
        f"goal_hits(win/total)={goal_window}/{goal_total}  "
        f"cum_drop_rate={drop_rate:.3f}"
    )


_world_to_local_points = world_to_local_points
