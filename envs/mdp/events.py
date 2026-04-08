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

Rolling goal: when current goal is achieved (pos < 2cm, rot < 0.1rad),
  a new nearby goal is selected via kNN from the grasp graph.
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
    pad_fingertip_positions,
    set_robot_joints_direct,
    expand_grasp_joint_vector,
    sample_wrist_pose_world,
    set_robot_root_pose,
    align_wrist_palm_down,
    align_wrist_palm_up,
    get_local_palm_normal,
    get_palm_body_id_from_env,
    place_object_fixed,
    compute_wrist_from_fingertips,
    set_adaptive_joint_pose,
    apply_palm_up_transform,
    place_object_in_hand,
    refine_hand_to_start_grasp,
    joint_positions_to_normalized_action,
    get_fingertip_body_ids_from_env,
)

# ---------------------------------------------------------------------------
# Termination predicates
# ---------------------------------------------------------------------------

def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length

def object_dropped(env, min_height: float = 0.2) -> torch.Tensor:
    obj = env.scene["object"]
    return obj.data.root_pos_w[:, 2] < min_height

def object_left_hand(env, max_dist: float = 0.20) -> torch.Tensor:
    """Terminate when object is too far from palm center."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    palm_body_id = get_palm_body_id_from_env(robot, env)
    palm_pos_w = robot.data.body_pos_w[:, palm_body_id, :]
    dist = torch.norm(obj.data.root_pos_w - palm_pos_w, dim=-1)
    return dist > max_dist

def no_fingertip_contact(env, patience: int = 30) -> torch.Tensor:
    """Terminate when no fingertip touches the object for `patience` consecutive steps.

    Uses a per-env counter stored in env.extras["_no_contact_steps"].
    patience=30 at 30Hz control = 1 second grace period.
    """
    from .observations import fingertip_contact_binary

    contact = fingertip_contact_binary(env)          # (N, num_fingers)
    any_contact = contact.sum(dim=-1) > 0             # (N,)

    # Lazily initialise counter
    buf = env.extras.get("_no_contact_steps")
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env.extras["_no_contact_steps"] = buf

    # Reset counter where contact exists, increment where it doesn't
    buf[any_contact] = 0
    buf[~any_contact] += 1

    return buf >= patience

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
        # Batch random start indices
        starts = rng.integers(0, N_grasps, size=batch_size)
        cur_min_orn = getattr(graph, "_curriculum_min_orn", 0.15)
        # Batch goal selection
        for j, local_i in enumerate(env_indices):
            start_idx_list[local_i] = int(starts[j])
            goal_idx_list[local_i] = _sample_nearby_goal_index(
                g, int(starts[j]), rng, min_orn=cur_min_orn, num_fingers=env_num_fingers,
            )

    # Extract grasp data in batch (one pass per object name group)
    start_fps_list = [None] * n
    goal_fps_list = [None] * n
    start_joints_list = [None] * n
    start_object_pos_hand_list = [None] * n
    start_object_quat_hand_list = [None] * n
    start_object_pose_frame_list = [None] * n
    goal_joints_list = [None] * n
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

        # Cache padded fingertip positions for this sub-graph
        if not hasattr(g, "_cached_padded_fps"):
            g._cached_padded_fps = np.stack([
                pad_fingertip_positions(grasp.fingertip_positions.copy(), env_num_fingers)
                for grasp in g.grasp_set.grasps
            ])  # (N, F, 3)

        cached_fps = g._cached_padded_fps
        grasps = g.grasp_set.grasps

        for local_i in env_indices:
            si = start_idx_list[local_i]
            gi = goal_idx_list[local_i]
            start_fps_list[local_i] = cached_fps[si]
            goal_fps_list[local_i] = cached_fps[gi]
            sg = grasps[si]
            gg = grasps[gi]
            start_joints_list[local_i] = getattr(sg, "joint_angles", None)
            start_object_pos_hand_list[local_i] = getattr(sg, "object_pos_hand", None)
            start_object_quat_hand_list[local_i] = getattr(sg, "object_quat_hand", None)
            start_object_pose_frame_list[local_i] = getattr(sg, "object_pose_frame", None)
            goal_joints_list[local_i] = getattr(gg, "joint_angles", None)
            goal_object_pos_hand_list[local_i] = getattr(gg, "object_pos_hand", None)
            goal_object_quat_hand_list[local_i] = getattr(gg, "object_quat_hand", None)
            goal_object_pose_frame_list[local_i] = getattr(gg, "object_pose_frame", None)

    start_fps = torch.tensor(
        np.stack(start_fps_list), device=env.device, dtype=torch.float32
    )   # (n, num_fingers, 3)
    goal_fps = torch.tensor(
        np.stack(goal_fps_list), device=env.device, dtype=torch.float32
    )   # (n, num_fingers, 3)

    fp_dim = env_num_fingers * 3

    # Store goal fingertip positions (object frame)
    if "target_fingertip_pos" not in env.extras:
        env.extras["target_fingertip_pos"] = torch.zeros(
            env.num_envs, fp_dim, device=env.device
        )
    env.extras["target_fingertip_pos"][env_ids] = goal_fps.reshape(n, fp_dim)

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

    # Reset no-contact termination counter
    if "_no_contact_steps" in env.extras:
        env.extras["_no_contact_steps"][env_ids] = 0

    # Reset delta reward prev-error buffers
    if "_prev_orn_error" in env.extras:
        from .rewards import _get_orn_error
        env.extras["_prev_orn_error"][env_ids] = _get_orn_error(env)[env_ids]
    if "_prev_pos_error" in env.extras:
        from .rewards import _get_pos_error
        env.extras["_prev_pos_error"][env_ids] = _get_pos_error(env)[env_ids]

# ---------------------------------------------------------------------------
# Start grasp sampling + nearest-neighbor goal selection
# ---------------------------------------------------------------------------

def _sample_start_and_nn_goal(
    graph,
    rng: np.random.Generator,
    object_name: Optional[str] = None,
    env_num_fingers: int = 4,
):
    """
    Sample a start grasp directly from the Stage 0 grasp_set and choose the
    nearest-neighbor grasp in fingertip-position space as the goal.

    env_num_fingers: the env's fixed finger count. Grasp fingertip_positions
    are padded / truncated to match so all tensors have uniform shape.
    """
    from grasp_generation.graph_io import MultiObjectGraspGraph

    if isinstance(graph, MultiObjectGraspGraph):
        obj_name = object_name or graph.sample_object(rng)
        g = graph.graphs[obj_name]
    else:
        obj_name = graph.object_name
        g = graph

    N = len(g)
    if N == 0:
        raise RuntimeError(
            f"Grasp graph for '{obj_name}' has 0 grasps. "
            f"Run scripts/run_grasp_generation.py to generate grasps first."
        )
    if N < 2:
        grasp = g.grasp_set[0]
        _pos = getattr(grasp, "object_pos_hand", None)
        _quat = getattr(grasp, "object_quat_hand", None)
        _ja = getattr(grasp, "joint_angles", None)
        _fps = pad_fingertip_positions(grasp.fingertip_positions.copy(), env_num_fingers)
        return (
            obj_name, _fps, _fps,
            _ja,
            _pos.copy() if _pos is not None else None,
            _quat.copy() if _quat is not None else None,
            getattr(grasp, "object_pose_frame", None),
            _ja,
            _pos.copy() if _pos is not None else None,
            _quat.copy() if _quat is not None else None,
            getattr(grasp, "object_pose_frame", None),
            0, 0,
        )

    start_idx = int(rng.integers(0, N))
    start_grasp = g.grasp_set[start_idx]
    cur_min_orn = getattr(graph, "_curriculum_min_orn", 0.15)
    goal_idx = _sample_nearby_goal_index(
        g, start_idx, rng, min_orn=cur_min_orn, num_fingers=env_num_fingers,
    )
    goal_grasp = g.grasp_set[goal_idx]

    start_fps = pad_fingertip_positions(start_grasp.fingertip_positions.copy(), env_num_fingers)
    goal_fps = pad_fingertip_positions(goal_grasp.fingertip_positions.copy(), env_num_fingers)

    start_joints = getattr(start_grasp, "joint_angles", None)
    start_object_pos_hand = getattr(start_grasp, "object_pos_hand", None)
    start_object_quat_hand = getattr(start_grasp, "object_quat_hand", None)
    start_object_pose_frame = getattr(start_grasp, "object_pose_frame", None)
    goal_joints = getattr(goal_grasp, "joint_angles", None)
    goal_object_pos_hand = getattr(goal_grasp, "object_pos_hand", None)
    goal_object_quat_hand = getattr(goal_grasp, "object_quat_hand", None)
    goal_object_pose_frame = getattr(goal_grasp, "object_pose_frame", None)

    return (
        obj_name,
        start_fps,
        goal_fps,
        start_joints,
        start_object_pos_hand.copy() if start_object_pos_hand is not None else None,
        start_object_quat_hand.copy() if start_object_quat_hand is not None else None,
        start_object_pose_frame,
        goal_joints,
        goal_object_pos_hand.copy() if goal_object_pos_hand is not None else None,
        goal_object_quat_hand.copy() if goal_object_quat_hand is not None else None,
        goal_object_pose_frame,
        int(start_idx),
        int(goal_idx),
    )

def _sample_nearby_goal_index(
    graph, start_idx: int, rng: np.random.Generator,
    top_k: int = 5, min_orn: float = 0.15,
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
    Linearly increase min_orn from 0.05 to 0.5 rad over the first 30%
    of training. Stored on the graph object so reset picks it up.

    Call once per epoch from the training loop.
    """
    graph = _load_grasp_graph(env)
    if graph is None:
        return
    warmup_epochs = int(total_epochs * 0.3)
    t = min(epoch / max(warmup_epochs, 1), 1.0)
    min_orn_start = 0.15
    min_orn_end = 0.50
    graph._curriculum_min_orn = min_orn_start + t * (min_orn_end - min_orn_start)

    gravity_cfg = dict((getattr(env.cfg, "gravity_curriculum", None) or {}))
    if gravity_cfg.get("enabled", False):
        warmup_ratio = float(gravity_cfg.get("warmup_ratio", 0.30))
        gravity_start = float(gravity_cfg.get("start_gravity", 0.05))
        gravity_end = float(gravity_cfg.get("end_gravity", 9.81))
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

def update_rolling_goal(
    env,
    rot_threshold: float = 0.1,
) -> int:
    """
    Called every step. For each env where the orientation is within
    threshold of the target, select a new nearby goal via kNN.

    Success = object orientation error < rot_threshold (0.1 rad ~5.7°).

    Returns:
        Number of envs whose goal was updated this step.
    """
    from isaaclab.utils.math import quat_apply_inverse

    graph = _load_grasp_graph(env)
    if graph is None:
        return 0

    goal_idx_buf  = env.extras.get("goal_grasp_idx")
    obj_names     = env.extras.get("env_obj_names")
    target_pos    = env.extras.get("target_object_pos_hand")
    target_quat   = env.extras.get("target_object_quat_hand")
    if goal_idx_buf is None or target_pos is None or target_quat is None:
        return 0

    # Object pose in hand frame
    robot = env.scene["robot"]
    obj = env.scene["object"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    cur_pos = quat_apply_inverse(root_quat, obj.data.root_pos_w - root_pos)

    def _qc(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    def _qm(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

    cur_quat = _qm(_qc(root_quat), obj.data.root_quat_w)

    pos_err = torch.norm(cur_pos - target_pos, dim=-1)
    dot = (cur_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    orn_err = 2.0 * torch.acos(dot)

    success_mask = (orn_err < rot_threshold)

    success_mask = success_mask & ~object_dropped(env)
    success_mask = success_mask & ~object_left_hand(env)

    success_ids = success_mask.nonzero(as_tuple=False).squeeze(-1)
    if success_ids.numel() == 0:
        return 0

    rng = _get_reset_rng(env)
    env_num_fingers = int(
        (getattr(env.cfg, "hand", None) or {}).get("num_fingers", 4)
    )
    fp_dim = env_num_fingers * 3

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

        # kNN from current goal → new goal
        new_goal_idx = _sample_nearby_goal_index(g, cur_goal_idx, rng, num_fingers=env_num_fingers)
        new_goal_grasp = g.grasp_set[new_goal_idx]

        # Update goal fingertip positions
        new_fps = pad_fingertip_positions(
            new_goal_grasp.fingertip_positions.copy(), env_num_fingers
        )
        env.extras["target_fingertip_pos"][env_id] = torch.tensor(
            new_fps.reshape(fp_dim), device=env.device, dtype=torch.float32
        )

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
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    num_fingers = hand_cfg.get("num_fingers", 4)
    num_dof = joint_pos.shape[-1]
    env.extras["target_fingertip_pos"] = torch.zeros(
        env.num_envs, num_fingers * 3, device=env.device
    )
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
    """Log start→goal pos/orn distances at reset. Prints every reset."""
    global _GOAL_LOG_COUNT
    _GOAL_LOG_COUNT += 1

    target_pos = env.extras.get("target_object_pos_hand")
    target_quat = env.extras.get("target_object_quat_hand")
    if target_pos is None or target_quat is None:
        return

    n = len(env_ids)
    robot = env.scene["robot"]
    obj = env.scene["object"]

    pos_dists = []
    orn_dists = []
    for i, eid in enumerate(env_ids.tolist()):
        rp = robot.data.root_pos_w[eid]
        rq = robot.data.root_quat_w[eid]
        op = obj.data.root_pos_w[eid]
        oq = obj.data.root_quat_w[eid]
        cur_pos = quat_apply_inverse(rq.unsqueeze(0), (op - rp).unsqueeze(0))[0]
        cur_quat = quat_multiply(quat_conjugate(rq.unsqueeze(0)), oq.unsqueeze(0))[0]

        pos_d = float(torch.norm(cur_pos - target_pos[eid]).item())
        pos_dists.append(pos_d)

        dot = float(torch.abs(torch.dot(cur_quat, target_quat[eid])).clamp(0.0, 1.0).item())
        orn_d = 2.0 * float(torch.acos(torch.tensor(dot)).item())
        orn_dists.append(orn_d)

    at_goal = sum(1 for o in orn_dists if o < 0.1)
    print(f"[Goal] Reset #{_GOAL_LOG_COUNT} ({n} envs)  "
          f"pos: {np.mean(pos_dists):.4f}m [{np.min(pos_dists):.4f}-{np.max(pos_dists):.4f}]  "
          f"orn: {np.mean(orn_dists):.2f}rad [{np.min(orn_dists):.2f}-{np.max(orn_dists):.2f}]  "
          f"at_goal: {at_goal}/{n}")


# ---------------------------------------------------------------------------
# Backwards-compatibility re-exports (for scripts that import from events.py)
# ---------------------------------------------------------------------------
_pad_fingertip_positions = pad_fingertip_positions
_set_robot_joints_direct = set_robot_joints_direct
_expand_grasp_joint_vector = expand_grasp_joint_vector
_sample_wrist_pose_world = sample_wrist_pose_world
_set_robot_root_pose = set_robot_root_pose
_align_wrist_palm_up = align_wrist_palm_up
_align_wrist_palm_down = align_wrist_palm_down
_get_local_palm_normal = get_local_palm_normal
_get_palm_body_id_from_env = get_palm_body_id_from_env
_place_object_fixed = place_object_fixed
_compute_wrist_from_fingertips = compute_wrist_from_fingertips
_set_adaptive_joint_pose = set_adaptive_joint_pose
_apply_palm_up_transform = apply_palm_up_transform
_place_object_in_hand = place_object_in_hand
_refine_hand_to_start_grasp = refine_hand_to_start_grasp
_joint_positions_to_normalized_action = joint_positions_to_normalized_action
_get_fingertip_body_ids_from_env = get_fingertip_body_ids_from_env
_quat_multiply = quat_multiply
_quat_conjugate = quat_conjugate
_local_to_world_points = local_to_world_points
_world_to_local_points = world_to_local_points
