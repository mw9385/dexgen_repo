"""
Event (reset / randomisation) functions for the AnyGrasp-to-AnyGrasp env.

Reset logic per episode:
  1. Sample a random start grasp from the GraspGraph grasp_set
  2. Find the nearest-neighbor grasp as the goal
  3. Keep the object close to its canonical pose used by grasp generation
  4. Place the wrist near a nominal grasping pose with only small jitter
  5. Set hand joint positions using stored joint angles (or IK fallback)
  6. Store goal fingertip positions in env.extras
"""

from __future__ import annotations

import math
import os
import pickle
from itertools import combinations
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_matrix
from grasp_generation.graph_io import load_merged_graph, parse_graph_paths
from .observations import _get_fingertip_contact_forces_world


# ---------------------------------------------------------------------------
# Termination predicates
# ---------------------------------------------------------------------------

def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def object_dropped(env, min_height: float = 0.2) -> torch.Tensor:
    obj = env.scene["object"]
    return obj.data.root_pos_w[:, 2] < min_height


def object_left_hand(env, max_dist: float = 0.20) -> torch.Tensor:
    """
    True when the object has escaped the palm support region.

    We treat two cases as failure:
      1. the object is too far from the palm and fingertips no longer touch it,
      2. the object moved behind the palm plane (wrist-side cheat).
    """
    escaped, _ = _object_escape_mask(env, max_dist=max_dist)
    return escaped


def _log_reset_reasons(env, env_ids: torch.Tensor, max_dist: float = 0.20) -> None:
    return


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

    _log_reset_reasons(env, env_ids)

    n = len(env_ids)
    # env_num_fingers is the FIXED finger count the env was built with.
    # All grasp data is padded/truncated to this count so tensors are uniform.
    env_num_fingers = int(
        (getattr(env.cfg, "hand", None) or {}).get("num_fingers", 4)
    )
    rng = _get_reset_rng(env)

    # ------------------------------------------------------------------
    # 1. Sample start grasps from grasp_set and goals via nearest neighbor
    # ------------------------------------------------------------------
    start_fps_list, goal_fps_list = [], []
    start_joints_list = []
    start_object_pos_hand_list, start_object_quat_hand_list = [], []
    start_object_pose_frame_list = []
    start_idx_list, goal_idx_list = [], []

    sampled_obj_names = []
    forced_object_name = _resolve_scene_graph_object_name(env, graph, env_num_fingers)
    # Per-env object name: detected from the actual spawned USD prim so that
    # envs with sphere objects use sphere grasps, cube envs use cube grasps, etc.
    env_graph_names = _detect_env_graph_names(env, graph, env_num_fingers)

    goal_joints_list = []
    goal_object_pos_hand_list, goal_object_quat_hand_list = [], []
    goal_object_pose_frame_list = []

    for i in range(n):
        env_id = int(env_ids[i].item())
        # Use per-env object name from USD detection, fall back to scene-cfg match
        if env_graph_names is not None:
            per_env_name = env_graph_names[env_id] or forced_object_name
        else:
            per_env_name = forced_object_name
        (
            obj_name, start_fp, goal_fp,
            start_joints, start_object_pos_hand, start_object_quat_hand, start_object_pose_frame,
            goal_joints, goal_object_pos_hand, goal_object_quat_hand, goal_object_pose_frame,
            start_idx, goal_idx
        ) = _sample_start_and_nn_goal(
            graph, rng,
            object_name=per_env_name,
            env_num_fingers=env_num_fingers,
        )
        sampled_obj_names.append(obj_name)
        start_fps_list.append(start_fp)
        goal_fps_list.append(goal_fp)
        start_joints_list.append(start_joints)
        start_object_pos_hand_list.append(start_object_pos_hand)
        start_object_quat_hand_list.append(start_object_quat_hand)
        start_object_pose_frame_list.append(start_object_pose_frame)
        goal_joints_list.append(goal_joints)
        goal_object_pos_hand_list.append(goal_object_pos_hand)
        goal_object_quat_hand_list.append(goal_object_quat_hand)
        goal_object_pose_frame_list.append(goal_object_pose_frame)
        start_idx_list.append(start_idx)
        goal_idx_list.append(goal_idx)

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
    env.extras["start_grasp_idx"][env_ids] = torch.tensor(
        start_idx_list, device=env.device, dtype=torch.long
    )
    env.extras["goal_grasp_idx"][env_ids] = torch.tensor(
        goal_idx_list, device=env.device, dtype=torch.long
    )

    # Store goal joint angles for finger_joint_goal_reward
    robot = env.scene["robot"]
    num_dof = robot.data.default_joint_pos.shape[-1]
    if "target_joint_angles" not in env.extras:
        env.extras["target_joint_angles"] = robot.data.default_joint_pos[
            :env.num_envs
        ].clone()
    for i, gj in enumerate(goal_joints_list):
        if gj is not None:
            env.extras["target_joint_angles"][env_ids[i]] = _expand_grasp_joint_vector(
                torch.tensor(gj, device=env.device, dtype=torch.float32),
                num_dof,
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
    # 2. Set object / robot initial state.
    #    For exact resets, keep the object at its canonical spawned pose and
    #    move the hand around it using the stored object pose in the hand
    #    frame.  This avoids the visual artifact where the object appears to
    #    spawn far from the hand and then gets teleported into place.
    # ------------------------------------------------------------------
    has_exact_pose = any(
        pos is not None and quat is not None
        for pos, quat in zip(start_object_pos_hand_list, start_object_quat_hand_list)
    )
    has_joints = any(j is not None for j in start_joints_list)
    exact_reset_mode = os.getenv("DEXGEN_EXACT_RESET_MODE", "object_fixed").strip().lower()

    if has_exact_pose and exact_reset_mode == "hand_fixed":
        wrist_pos_w, wrist_quat_w = _sample_wrist_pose_world(
            env, env_ids, apply_noise=False
        )
        _set_robot_root_pose(env, env_ids, wrist_pos_w, wrist_quat_w)
        _randomise_object_pose(env, env_ids)
    elif has_exact_pose:
        _randomise_object_pose(env, env_ids)
    else:
        _randomise_object_pose(env, env_ids)
        _randomise_wrist_pose(env, env_ids)

    if has_joints:
        _set_robot_joints_direct(env, env_ids, start_joints_list)
    else:
        _set_robot_to_fingertip_config(env, env_ids, start_fps)

    if has_exact_pose and exact_reset_mode == "hand_fixed":
        robot = env.scene["robot"]
        obj = env.scene["object"]
        robot.update(0.0)
        obj.update(0.0)
        if all(frame == "root" for frame in start_object_pose_frame_list if frame is not None):
            _set_object_pose_from_grasp(
                env, env_ids, start_object_pos_hand_list, start_object_quat_hand_list
            )
        else:
            _set_object_pose_from_hand_frame_grasp(
                env, env_ids, start_object_pos_hand_list, start_object_quat_hand_list
            )
    elif has_exact_pose:
        robot = env.scene["robot"]
        obj = env.scene["object"]
        robot.update(0.0)
        obj.update(0.0)
        if all(frame == "root" for frame in start_object_pose_frame_list if frame is not None):
            _set_robot_root_pose_from_root_frame_grasp_pose(
                env, env_ids, start_object_pos_hand_list, start_object_quat_hand_list
            )
        else:
            _set_robot_root_pose_from_grasp_pose(
                env, env_ids, start_object_pos_hand_list, start_object_quat_hand_list
            )

    # ------------------------------------------------------------------
    # 3. Place object.
    #    ALWAYS use _place_object_in_hand (fingertip-matching) rather than
    #    the stored object_pos_hand. This avoids hand-root frame mismatches
    #    between Stage-0 data and the Isaac Lab articulation root.
    #    After writing root + joints we must update body transforms first.
    # ------------------------------------------------------------------
    robot = env.scene["robot"]
    robot.update(0.0)

    placement_debug = None
    if has_exact_pose:
        # Keep the object fixed and refine the hand so the sampled Stage-0
        # fingertip contacts are matched without teleporting the object.
        _refine_hand_to_start_grasp(env, env_ids, start_fps)
        robot.update(0.0)
        obj = env.scene["object"]
        obj.update(0.0)
        solve_mean_err, solve_max_err = _measure_grasp_contact_error(env, env_ids, start_fps)
    else:
        solve_mean_err, solve_max_err, placement_debug = _place_object_in_hand(env, env_ids, start_fps)
        # Exact Stage-0 tuples still suffer from hand-model / joint-order /
        # simulator discrepancies. Refine the hand against the sampled grasp
        # and then place the object again using the refined fingertips.
        _refine_hand_to_start_grasp(env, env_ids, start_fps)
        robot.update(0.0)
        solve_mean_err, solve_max_err, placement_debug = _place_object_in_hand(env, env_ids, start_fps)
        # One final measurement from the actual sim state after re-placement.
        solve_mean_err, solve_max_err = _measure_grasp_contact_error(env, env_ids, start_fps)

    # ------------------------------------------------------------------
    # 7. Compute target_object_pos/quat_hand from the ACTUAL sim state.
    #    Some stored object poses are defined in a palm/body-local frame,
    #    while the reward uses the articulation root frame. To avoid that
    #    mismatch, ALWAYS compute from the current sim state. Since the
    #    object was just placed via _place_object_in_hand, the current sim
    #    pose is consistent with the articulation root frame.
    # ------------------------------------------------------------------
    robot_for_goal = env.scene["robot"]
    obj_for_goal   = env.scene["object"]
    obj_for_goal.update(0.0)
    for i in range(n):
        rp_w = robot_for_goal.data.root_pos_w[env_ids[i]]   # (3,)
        rq_w = robot_for_goal.data.root_quat_w[env_ids[i]]  # (4,)
        op_w = obj_for_goal.data.root_pos_w[env_ids[i]]     # (3,)
        oq_w = obj_for_goal.data.root_quat_w[env_ids[i]]    # (4,)
        rel  = op_w - rp_w
        cur_obj_pos_hand = quat_apply_inverse(rq_w.unsqueeze(0), rel.unsqueeze(0))[0]
        env.extras["target_object_pos_hand"][env_ids[i]] = cur_obj_pos_hand.clone()
        cur_obj_quat_hand = _quat_multiply(
            _quat_conjugate(rq_w.unsqueeze(0)), oq_w.unsqueeze(0)
        )[0]
        env.extras["target_object_quat_hand"][env_ids[i]] = cur_obj_quat_hand.clone()

    # ------------------------------------------------------------------
    # 8. Initialise action buffers (fixes last_action always being 0)
    # ------------------------------------------------------------------
    robot = env.scene["robot"]
    current_q = robot.data.joint_pos[env_ids]  # (n, num_dof)
    current_action = _joint_positions_to_normalized_action(robot, env_ids, current_q)
    if "last_action" not in env.extras:
        env.extras["last_action"] = torch.zeros(
            env.num_envs, current_q.shape[-1], device=env.device
        )
    if "current_action" not in env.extras:
        env.extras["current_action"] = torch.zeros(
            env.num_envs, current_q.shape[-1], device=env.device
        )
    # Reset action buffers to current joint positions for reset envs
    env.extras["last_action"][env_ids] = current_action
    env.extras["current_action"][env_ids] = current_action

# ---------------------------------------------------------------------------
# Start grasp sampling + nearest-neighbor goal selection
# ---------------------------------------------------------------------------

def _pad_fingertip_positions(fps: np.ndarray, target_nf: int) -> np.ndarray:
    """
    Pad or truncate (num_fingers, 3) array to (target_nf, 3).

    Extra fingers (added by padding) are set to the last valid finger's
    position so they have a neutral target and contribute no gradient.
    """
    nf = fps.shape[0]
    if nf == target_nf:
        return fps
    if nf > target_nf:
        return fps[:target_nf]
    # pad by repeating last row
    pad = np.tile(fps[-1:], (target_nf - nf, 1))
    return np.concatenate([fps, pad], axis=0).astype(np.float32)


def _sample_start_and_nn_goal(
    graph,
    rng: np.random.Generator,
    object_name: Optional[str] = None,
    env_num_fingers: int = 4,
):
    """
    Sample a start grasp directly from the Stage 0 grasp_set and choose the
    nearest-neighbor grasp in fingertip-position space as the goal.

    env_num_fingers: the env's fixed finger count.  Grasp fingertip_positions
    are padded / truncated to match so all tensors have uniform shape.
    """
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph, GraspGraph

    if isinstance(graph, MultiObjectGraspGraph):
        obj_name = object_name or graph.sample_object(rng)
        g = graph.graphs[obj_name]
    else:
        obj_name = graph.object_name
        g = graph

    N = len(g)
    if N < 2:
        grasp = g.grasp_set[0]
        _pos  = getattr(grasp, "object_pos_hand",  None)
        _quat = getattr(grasp, "object_quat_hand", None)
        _ja   = getattr(grasp, "joint_angles",      None)
        _fps  = _pad_fingertip_positions(grasp.fingertip_positions.copy(), env_num_fingers)
        return (
            obj_name, _fps, _fps,
            _ja,
            _pos.copy()  if _pos  is not None else None,
            _quat.copy() if _quat is not None else None,
            getattr(grasp, "object_pose_frame", None),
            _ja,
            _pos.copy()  if _pos  is not None else None,
            _quat.copy() if _quat is not None else None,
            getattr(grasp, "object_pose_frame", None),
            0, 0,
        )

    start_idx = int(rng.integers(0, N))
    start_grasp = g.grasp_set[start_idx]
    goal_idx = _sample_nearby_goal_index(g, start_idx, rng)
    goal_grasp = g.grasp_set[goal_idx]

    start_fps  = _pad_fingertip_positions(start_grasp.fingertip_positions.copy(), env_num_fingers)
    goal_fps   = _pad_fingertip_positions(goal_grasp.fingertip_positions.copy(),  env_num_fingers)

    start_joints           = getattr(start_grasp, "joint_angles",     None)
    start_object_pos_hand  = getattr(start_grasp, "object_pos_hand",  None)
    start_object_quat_hand = getattr(start_grasp, "object_quat_hand", None)
    start_object_pose_frame = getattr(start_grasp, "object_pose_frame", None)
    goal_joints            = getattr(goal_grasp,  "joint_angles",     None)
    goal_object_pos_hand   = getattr(goal_grasp,  "object_pos_hand",  None)
    goal_object_quat_hand  = getattr(goal_grasp,  "object_quat_hand", None)
    goal_object_pose_frame = getattr(goal_grasp, "object_pose_frame", None)

    return (
        obj_name,
        start_fps,
        goal_fps,
        start_joints,
        start_object_pos_hand.copy()  if start_object_pos_hand  is not None else None,
        start_object_quat_hand.copy() if start_object_quat_hand is not None else None,
        start_object_pose_frame,
        goal_joints,
        goal_object_pos_hand.copy()   if goal_object_pos_hand   is not None else None,
        goal_object_quat_hand.copy()  if goal_object_quat_hand  is not None else None,
        goal_object_pose_frame,
        int(start_idx),
        int(goal_idx),
    )


def _sample_nearby_goal_index(graph, start_idx: int, rng: np.random.Generator) -> int:
    grasps = graph.grasp_set.grasps
    N = len(grasps)
    if N <= 1:
        return start_idx

    max_candidates = min(N, 512)
    candidate_indices = np.arange(N)
    if N > max_candidates:
        keep = rng.choice(candidate_indices[candidate_indices != start_idx], size=max_candidates - 1, replace=False)
        candidate_indices = np.concatenate([np.array([start_idx]), np.sort(keep)])

    start_grasp = grasps[start_idx]
    dists = np.full((N,), np.inf, dtype=np.float32)
    for idx in candidate_indices:
        if idx == start_idx:
            continue
        dists[idx] = _grasp_state_distance(start_grasp, grasps[int(idx)])

    goal_idx = int(np.argmin(dists))
    if not np.isfinite(dists[goal_idx]):
        all_fps = graph.grasp_set.as_array()
        start_flat = all_fps[start_idx]
        fallback = np.linalg.norm(all_fps - start_flat, axis=-1)
        fallback[start_idx] = np.inf
        goal_idx = int(np.argmin(fallback))
    return goal_idx


def _grasp_state_distance(grasp_a, grasp_b) -> float:
    if (
        getattr(grasp_a, "joint_angles", None) is not None and
        getattr(grasp_b, "joint_angles", None) is not None and
        getattr(grasp_a, "object_pos_hand", None) is not None and
        getattr(grasp_b, "object_pos_hand", None) is not None and
        getattr(grasp_a, "object_quat_hand", None) is not None and
        getattr(grasp_b, "object_quat_hand", None) is not None
    ):
        q_a = np.asarray(grasp_a.joint_angles, dtype=np.float32)
        q_b = np.asarray(grasp_b.joint_angles, dtype=np.float32)
        pos_a = np.asarray(grasp_a.object_pos_hand, dtype=np.float32)
        pos_b = np.asarray(grasp_b.object_pos_hand, dtype=np.float32)
        quat_a = np.asarray(grasp_a.object_quat_hand, dtype=np.float32)
        quat_b = np.asarray(grasp_b.object_quat_hand, dtype=np.float32)

        joint_dist = float(np.linalg.norm(q_a - q_b) / max(len(q_a), 1))
        pos_dist = float(np.linalg.norm(pos_a - pos_b))
        quat_dot = float(np.clip(abs(np.dot(quat_a, quat_b)), 0.0, 1.0))
        rot_dist = float(2.0 * np.arccos(quat_dot))
        return pos_dist + 0.05 * rot_dist + 0.02 * joint_dist

    fps_a = np.asarray(grasp_a.fingertip_positions, dtype=np.float32).reshape(-1)
    fps_b = np.asarray(grasp_b.fingertip_positions, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(fps_a - fps_b))


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
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

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
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

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

def _set_robot_joints_direct(
    env,
    env_ids: torch.Tensor,
    joints_list: list,
):
    """
    Set robot joints directly from stored joint angles in the Grasp object.
    This is the correct approach when grasp generation also solves IK.
    """
    robot = env.scene["robot"]
    n = len(env_ids)
    num_dof = robot.data.default_joint_pos.shape[-1]

    joint_pos = robot.data.default_joint_pos[env_ids].clone()

    for i, joints in enumerate(joints_list):
        if joints is not None:
            joint_pos[i] = _expand_grasp_joint_vector(
                torch.tensor(joints, device=env.device, dtype=torch.float32),
                num_dof,
            )

    robot.write_joint_state_to_sim(
        joint_pos,
        torch.zeros_like(joint_pos),
        env_ids=env_ids,
    )
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def _expand_grasp_joint_vector(
    joint_vec: torch.Tensor,
    target_num_dof: int,
) -> torch.Tensor:
    """
    Expand/pad a stored grasp joint vector to the articulation DOF count.

    Some stored Shadow grasps use 22 finger joints, while Isaac Lab's Shadow
    USD articulation exposes 24 joints with 2 wrist DOFs leading the vector.
    """
    if joint_vec.shape[0] == target_num_dof:
        return joint_vec
    if joint_vec.shape[0] == target_num_dof - 2:
        expanded = torch.zeros(target_num_dof, device=joint_vec.device, dtype=joint_vec.dtype)
        expanded[2:] = joint_vec
        return expanded

    expanded = torch.zeros(target_num_dof, device=joint_vec.device, dtype=joint_vec.dtype)
    n_copy = min(target_num_dof, joint_vec.shape[0])
    expanded[:n_copy] = joint_vec[:n_copy]
    return expanded


# ---------------------------------------------------------------------------
# Wrist randomization
# ---------------------------------------------------------------------------

def _randomise_wrist_pose(env, env_ids: torch.Tensor):
    """
    Place the wrist near a nominal grasping pose with only small jitter.

    The previous reset sampled the wrist on a wide hemisphere around the
    object. That made the reset largely unrelated to the sampled grasp graph
    node and frequently placed the hand too far away to learn a meaningful
    start grasp.
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    n = len(env_ids)
    cfg = getattr(env.cfg, "reset_randomization", {}) or {}

    default_robot_root = robot.data.default_root_state[env_ids, :7].clone()
    default_obj_root = obj.data.default_root_state[env_ids, :7].clone()

    wrist_pos = obj.data.root_pos_w[env_ids].clone()
    nominal_offset = default_robot_root[:, :3] - default_obj_root[:, :3]
    wrist_pos += nominal_offset

    pos_jitter_std = float(cfg.get("wrist_pos_jitter_std", 0.005))
    if pos_jitter_std > 0.0:
        wrist_pos += torch.randn(n, 3, device=env.device) * pos_jitter_std

    wrist_quat = default_robot_root[:, 3:7]
    if bool(cfg.get("align_palm_up", False)):
        wrist_quat = _align_wrist_palm_up(env, env_ids, wrist_quat)
    rot_std = math.radians(float(cfg.get("wrist_rot_std_deg", 5.0)))
    if rot_std > 0.0:
        wrist_quat = _add_rotation_noise(wrist_quat, rot_std, env.device, n)

    _set_robot_root_pose(env, env_ids, wrist_pos, wrist_quat)


def _sample_wrist_pose_world(
    env,
    env_ids: torch.Tensor,
    apply_noise: bool = True,
):
    robot = env.scene["robot"]
    n = len(env_ids)
    cfg = getattr(env.cfg, "reset_randomization", {}) or {}

    # default_root_state is in LOCAL env frame.  Add env origin to get world frame.
    root_local = robot.data.default_root_state[env_ids, :7].clone()
    wrist_pos = root_local[:, :3] + env.scene.env_origins[env_ids]
    wrist_quat = root_local[:, 3:7]

    pos_jitter_std = float(cfg.get("wrist_pos_jitter_std", 0.005))
    if apply_noise and pos_jitter_std > 0.0:
        wrist_pos = wrist_pos + torch.randn(n, 3, device=env.device) * pos_jitter_std

    if apply_noise and bool(cfg.get("align_palm_up", False)):
        wrist_quat = _align_wrist_palm_up(env, env_ids, wrist_quat)

    rot_std = math.radians(float(cfg.get("wrist_rot_std_deg", 5.0)))
    if apply_noise and rot_std > 0.0:
        wrist_quat = _add_rotation_noise(wrist_quat, rot_std, env.device, n)

    return wrist_pos, wrist_quat


def _set_robot_root_pose(
    env,
    env_ids: torch.Tensor,
    wrist_pos_w: torch.Tensor,
    wrist_quat_w: torch.Tensor,
):
    robot = env.scene["robot"]
    root_pose = robot.data.default_root_state[env_ids, :7].clone()
    root_pose[:, :3] = wrist_pos_w
    root_pose[:, 3:7] = wrist_quat_w / (torch.norm(wrist_quat_w, dim=-1, keepdim=True) + 1e-8)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)


def _set_object_pose_from_grasp(
    env,
    env_ids: torch.Tensor,
    object_pos_hand_list: list,
    object_quat_hand_list: list,
):
    robot = env.scene["robot"]
    obj = env.scene["object"]

    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]
    root_state = obj.data.default_root_state[env_ids].clone()
    # default_root_state is LOCAL → convert base positions to world for the fallback
    root_state[:, :3] += env.scene.env_origins[env_ids]

    for i, (pos_hand, quat_hand) in enumerate(zip(object_pos_hand_list, object_quat_hand_list)):
        if pos_hand is None or quat_hand is None:
            continue

        pos_hand_t = torch.tensor(pos_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        quat_hand_t = torch.tensor(quat_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        world_pos = _local_to_world_points(pos_hand_t.unsqueeze(1), robot_pos[i:i+1], robot_quat[i:i+1]).squeeze(1)[0]
        world_quat = _quat_multiply(robot_quat[i:i+1], quat_hand_t)[0]
        root_state[i, :3] = world_pos
        root_state[i, 3:7] = world_quat / (torch.norm(world_quat) + 1e-8)

    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def _set_object_pose_from_hand_frame_grasp(
    env,
    env_ids: torch.Tensor,
    object_pos_hand_list: list,
    object_quat_hand_list: list,
):
    robot = env.scene["robot"]
    obj = env.scene["object"]
    palm_body_id = _get_palm_body_id_from_env(robot, env)

    palm_pos_w = robot.data.body_pos_w[env_ids][:, palm_body_id, :].clone()
    palm_quat_w = robot.data.body_quat_w[env_ids][:, palm_body_id, :].clone()
    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    for i, (pos_hand, quat_hand) in enumerate(zip(object_pos_hand_list, object_quat_hand_list)):
        if pos_hand is None or quat_hand is None:
            continue

        pos_hand_t = torch.tensor(pos_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        quat_hand_t = torch.tensor(quat_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        world_pos = _local_to_world_points(
            pos_hand_t.unsqueeze(1),
            palm_pos_w[i : i + 1],
            palm_quat_w[i : i + 1],
        ).squeeze(1)[0]
        world_quat = _quat_multiply(palm_quat_w[i : i + 1], quat_hand_t)[0]
        root_state[i, :3] = world_pos
        root_state[i, 3:7] = world_quat / (torch.norm(world_quat) + 1e-8)

    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def _set_robot_root_pose_from_grasp_pose(
    env,
    env_ids: torch.Tensor,
    object_pos_hand_list: list,
    object_quat_hand_list: list,
):
    robot = env.scene["robot"]
    obj = env.scene["object"]
    palm_body_id = _get_palm_body_id_from_env(robot, env)

    root_pos_w = robot.data.root_pos_w[env_ids].clone()
    root_quat_w = robot.data.root_quat_w[env_ids].clone()
    palm_pos_w = robot.data.body_pos_w[env_ids][:, palm_body_id, :].clone()
    palm_quat_w = robot.data.body_quat_w[env_ids][:, palm_body_id, :].clone()

    palm_pos_root = quat_apply_inverse(root_quat_w, palm_pos_w - root_pos_w)
    palm_quat_root = _quat_multiply(_quat_conjugate(root_quat_w), palm_quat_w)
    palm_quat_root = palm_quat_root / (torch.norm(palm_quat_root, dim=-1, keepdim=True) + 1e-8)

    obj_pos_w = obj.data.root_pos_w[env_ids].clone()
    obj_quat_w = obj.data.root_quat_w[env_ids].clone()
    target_root_pos_w = root_pos_w.clone()
    target_root_quat_w = root_quat_w.clone()

    for i, (obj_pos_hand, obj_quat_hand) in enumerate(zip(object_pos_hand_list, object_quat_hand_list)):
        if obj_pos_hand is None or obj_quat_hand is None:
            continue

        obj_pos_hand_t = torch.tensor(obj_pos_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        obj_quat_hand_t = torch.tensor(obj_quat_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        palm_quat_target = _quat_multiply(obj_quat_w[i : i + 1], _quat_conjugate(obj_quat_hand_t))
        palm_quat_target = palm_quat_target / (torch.norm(palm_quat_target, dim=-1, keepdim=True) + 1e-8)
        root_quat_target = _quat_multiply(palm_quat_target, _quat_conjugate(palm_quat_root[i : i + 1]))
        root_quat_target = root_quat_target / (torch.norm(root_quat_target, dim=-1, keepdim=True) + 1e-8)

        palm_offset_world = quat_apply(root_quat_target, palm_pos_root[i : i + 1])[0]
        obj_offset_world = quat_apply(palm_quat_target, obj_pos_hand_t)[0]
        target_root_pos_w[i] = obj_pos_w[i] - obj_offset_world - palm_offset_world
        target_root_quat_w[i] = root_quat_target[0]

    _set_robot_root_pose(env, env_ids, target_root_pos_w, target_root_quat_w)


def _set_robot_root_pose_from_root_frame_grasp_pose(
    env,
    env_ids: torch.Tensor,
    object_pos_hand_list: list,
    object_quat_hand_list: list,
):
    robot = env.scene["robot"]
    obj = env.scene["object"]

    obj_pos_w = obj.data.root_pos_w[env_ids].clone()
    obj_quat_w = obj.data.root_quat_w[env_ids].clone()
    target_root_pos_w = robot.data.root_pos_w[env_ids].clone()
    target_root_quat_w = robot.data.root_quat_w[env_ids].clone()

    for i, (obj_pos_hand, obj_quat_hand) in enumerate(zip(object_pos_hand_list, object_quat_hand_list)):
        if obj_pos_hand is None or obj_quat_hand is None:
            continue

        obj_pos_hand_t = torch.tensor(obj_pos_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        obj_quat_hand_t = torch.tensor(obj_quat_hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        root_quat_target = _quat_multiply(obj_quat_w[i : i + 1], _quat_conjugate(obj_quat_hand_t))
        root_quat_target = root_quat_target / (torch.norm(root_quat_target, dim=-1, keepdim=True) + 1e-8)
        root_offset_world = quat_apply(root_quat_target, obj_pos_hand_t)[0]
        target_root_pos_w[i] = obj_pos_w[i] - root_offset_world
        target_root_quat_w[i] = root_quat_target[0]

    _set_robot_root_pose(env, env_ids, target_root_pos_w, target_root_quat_w)


def _align_wrist_palm_up(env, env_ids: torch.Tensor, wrist_quat: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    palm_normal_local = _get_local_palm_normal(robot, env).unsqueeze(0).expand(len(env_ids), 3)
    palm_normal_world = quat_apply(wrist_quat, palm_normal_local)
    target_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(palm_normal_world)
    correction = _quat_from_two_vectors(palm_normal_world, target_up)
    return _quat_multiply(correction, wrist_quat)


def _get_local_palm_normal(robot, env) -> torch.Tensor:
    key = id(robot)
    if key not in _PALM_NORMAL_CACHE:
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        if hand_cfg.get("name") == "shadow":
            base_names = ["robot0_ffknuckle", "robot0_rfknuckle", "robot0_thbase"]
        else:
            base_names = ["index_link_0", "ring_link_0", "thumb_link_0"]
        body_ids = [robot.find_bodies(name)[0][0] for name in base_names]
        pts_world = robot.data.body_pos_w[0:1, body_ids, :]
        root_pos = robot.data.root_pos_w[0:1]
        root_quat = robot.data.root_quat_w[0:1]
        pts_local = _world_to_local_points(pts_world, root_pos, root_quat)[0]
        v1 = pts_local[0] - pts_local[2]
        v2 = pts_local[1] - pts_local[2]
        normal = torch.cross(v1, v2, dim=-1)
        normal = normal / (torch.norm(normal) + 1e-8)
        _PALM_NORMAL_CACHE[key] = normal
    return _PALM_NORMAL_CACHE[key]


def _get_palm_body_id_from_env(robot, env) -> int:
    key = ("palm_body", id(robot))
    cached = _FT_IDS_CACHE.get(key)
    if cached is not None:
        return cached

    hand_cfg = getattr(env.cfg, "hand", None) or {}
    if hand_cfg.get("name") == "shadow":
        candidate_names = ["robot0_palm", "robot0:palm", "palm"]
    else:
        candidate_names = ["palm_link", "base_link", "palm"]

    for name in candidate_names:
        try:
            body_ids = robot.find_bodies(name)[0]
        except Exception:
            continue
        if len(body_ids) > 0:
            _FT_IDS_CACHE[key] = int(body_ids[0])
            return _FT_IDS_CACHE[key]

    raise RuntimeError(
        f"Could not resolve palm body for hand={hand_cfg.get('name', 'unknown')}; "
        f"tried {candidate_names}"
    )


def _object_escape_mask(
    env,
    max_dist: float = 0.20,
    max_behind_offset: float = 0.015,
    contact_force_thresh: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    robot = env.scene["robot"]
    obj = env.scene["object"]

    palm_body_id = _get_palm_body_id_from_env(robot, env)
    palm_pos_w = robot.data.body_pos_w[:, palm_body_id, :]
    rel_obj_w = obj.data.root_pos_w - palm_pos_w
    dist = torch.norm(rel_obj_w, dim=-1)

    palm_normal_root = _get_local_palm_normal(robot, env).unsqueeze(0).expand(env.num_envs, 3)
    palm_normal_world = quat_apply(robot.data.root_quat_w, palm_normal_root)
    palm_normal_world = palm_normal_world / (torch.norm(palm_normal_world, dim=-1, keepdim=True) + 1e-8)

    normal_offset = (rel_obj_w * palm_normal_world).sum(dim=-1)
    lateral_vec = rel_obj_w - normal_offset.unsqueeze(-1) * palm_normal_world
    lateral_offset = torch.norm(lateral_vec, dim=-1)

    contact_forces = _get_fingertip_contact_forces_world(env)
    has_contact = (torch.norm(contact_forces, dim=-1) > contact_force_thresh).any(dim=-1)

    too_far = dist > max_dist
    behind_palm = normal_offset < -max_behind_offset

    escaped = behind_palm | (too_far & (~has_contact))
    return escaped, {
        "dist": dist,
        "normal_offset": normal_offset,
        "lateral_offset": lateral_offset,
        "has_contact": has_contact,
        "too_far": too_far,
        "behind_palm": behind_palm,
    }


def _quat_from_two_vectors(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    v_from = v_from / (torch.norm(v_from, dim=-1, keepdim=True) + 1e-8)
    v_to = v_to / (torch.norm(v_to, dim=-1, keepdim=True) + 1e-8)
    cross = torch.cross(v_from, v_to, dim=-1)
    dot = (v_from * v_to).sum(dim=-1, keepdim=True)
    quat = torch.cat([1.0 + dot, cross], dim=-1)

    opposite = dot.squeeze(-1) < -0.9999
    if opposite.any():
        ortho = torch.zeros_like(v_from[opposite])
        use_x = torch.abs(v_from[opposite, 0]) < 0.9
        ortho[use_x, 0] = 1.0
        ortho[~use_x, 1] = 1.0
        axis = torch.cross(v_from[opposite], ortho, dim=-1)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        quat[opposite] = torch.cat([torch.zeros((axis.shape[0], 1), device=axis.device), axis], dim=-1)

    return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)


def _place_object_in_hand(
    env,
    env_ids: torch.Tensor,
    fingertip_positions_obj: torch.Tensor,   # (n, F, 3) in object frame
):
    """
    Place the object so the sampled grasp-set fingertip positions line up
    with the current fingertip world positions of the hand.

    This makes reset start from the Stage 0 grasp itself instead of from an
    unrelated hand/object arrangement.
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]

    ft_ids = _get_fingertip_body_ids_from_env(robot, env)
    ft_world = robot.data.body_pos_w[env_ids][:, ft_ids, :].clone()  # (n, F, 3)

    full_pos_w, full_quat_w = _solve_rigid_alignment(
        points_src=fingertip_positions_obj,
        points_dst=ft_world,
    )
    full_reconstructed_world = _local_to_world_points(
        fingertip_positions_obj, full_pos_w, full_quat_w
    )
    full_solve_err = torch.norm(full_reconstructed_world - ft_world, dim=-1)

    pos_w, quat_w = _solve_object_pose_from_contacts(
        points_obj=fingertip_positions_obj,
        points_world=ft_world,
    )
    reconstructed_world = _local_to_world_points(fingertip_positions_obj, pos_w, quat_w)
    solve_err = torch.norm(reconstructed_world - ft_world, dim=-1)

    pose_pos_delta = torch.norm(pos_w - full_pos_w, dim=-1)
    pose_quat_dot = (quat_w * full_quat_w).sum(dim=-1).abs().clamp(0.0, 1.0)
    pose_rot_delta = 2.0 * torch.arccos(pose_quat_dot)

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = pos_w
    root_state[:, 3:7] = quat_w
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)
    placement_debug = {
        "chosen_mean_err": solve_err.mean(dim=-1),
        "chosen_max_err": solve_err.max(dim=-1).values,
        "full_mean_err": full_solve_err.mean(dim=-1),
        "full_max_err": full_solve_err.max(dim=-1).values,
        "pose_pos_delta": pose_pos_delta,
        "pose_rot_delta": pose_rot_delta,
    }
    return solve_err.mean(dim=-1), solve_err.max(dim=-1).values, placement_debug


def _measure_grasp_contact_error(
    env,
    env_ids: torch.Tensor,
    fingertip_positions_obj: torch.Tensor,
):
    robot = env.scene["robot"]
    obj = env.scene["object"]
    ft_ids = _get_fingertip_body_ids_from_env(robot, env)

    ft_world = robot.data.body_pos_w[env_ids][:, ft_ids, :].clone()
    obj_pos = obj.data.root_pos_w[env_ids].clone()
    obj_quat = obj.data.root_quat_w[env_ids].clone()
    ft_obj = _world_to_local_points(ft_world, obj_pos, obj_quat)
    err = torch.norm(ft_obj - fingertip_positions_obj, dim=-1)
    return err.mean(dim=-1), err.max(dim=-1).values


def _refine_hand_to_start_grasp(
    env,
    env_ids: torch.Tensor,
    start_fps_obj: torch.Tensor,   # (n, F, 3)
):
    """
    Refine the hand joint state with a small multi-fingertip differential IK
    pass so the reset hand actually matches the sampled Stage 0 grasp.
    """
    cfg = getattr(env.cfg, "reset_refinement", {}) or {}
    if not bool(cfg.get("enabled", True)):
        return

    iterations = int(cfg.get("iterations", 3))
    if iterations <= 0:
        return

    robot = env.scene["robot"]
    obj = env.scene["object"]
    n = len(env_ids)

    ft_ids = _get_fingertip_body_ids_from_env(robot, env)
    jac_body_ids = [body_id - 1 if robot.is_fixed_base else body_id for body_id in ft_ids]
    joint_count = robot.data.joint_pos.shape[-1]
    joint_ids = torch.arange(joint_count, device=env.device, dtype=torch.long)

    step_gain = float(cfg.get("step_gain", 0.6))
    damping = float(cfg.get("damping", 0.05))
    max_delta = float(cfg.get("max_delta", 0.2))
    pos_threshold = float(cfg.get("pos_threshold", 0.005))
    sim_dt = env.sim.get_physics_dt()

    joint_lower = robot.data.soft_joint_pos_limits[env_ids, :, 0]
    joint_upper = robot.data.soft_joint_pos_limits[env_ids, :, 1]
    target_obj_state = obj.data.root_state_w[env_ids].clone()

    for _ in range(iterations):
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(sim_dt)
        obj.update(sim_dt)

        # Keep the object fixed at the intended reset pose during refinement.
        obj.write_root_state_to_sim(target_obj_state, env_ids=env_ids)
        obj.update(0.0)

        target_world = _local_to_world_points(
            start_fps_obj,
            target_obj_state[:, :3],
            target_obj_state[:, 3:7],
        )
        current_world = robot.data.body_pos_w[env_ids][:, ft_ids, :]
        pos_error = target_world - current_world
        mean_err = torch.norm(pos_error, dim=-1).mean(dim=-1)
        if torch.all(mean_err <= pos_threshold):
            break

        jacobian = robot.root_physx_view.get_jacobians()[env_ids][:, jac_body_ids, :3, :][:, :, :, joint_ids]
        jacobian = jacobian.reshape(n, -1, joint_count)
        error_vec = pos_error.reshape(n, -1, 1)

        jt = jacobian.transpose(1, 2)
        lhs = torch.bmm(jacobian, jt)
        eye = torch.eye(lhs.shape[-1], device=env.device, dtype=lhs.dtype).unsqueeze(0).expand_as(lhs)
        system = lhs + (damping**2) * eye
        try:
            solved = torch.linalg.solve(system, error_vec)
        except RuntimeError:
            # cuSOLVER can fail intermittently on these small reset-time batched systems.
            # Fall back to CPU solve so reset remains robust instead of aborting the episode.
            solved = torch.linalg.solve(system.cpu(), error_vec.cpu()).to(env.device)
        delta = torch.bmm(jt, solved).squeeze(-1)
        delta = (step_gain * delta).clamp(-max_delta, max_delta)

        joint_pos = robot.data.joint_pos[env_ids].clone()
        joint_pos = torch.clamp(joint_pos + delta, joint_lower, joint_upper)
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)

def _look_at_quat(direction: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) that rotates +Z to point along `direction`."""
    N = direction.shape[0]
    device = direction.device

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3)
    axis = torch.cross(z_axis, direction, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    parallel = axis_norm.squeeze(-1) < 1e-6
    axis = axis / (axis_norm + 1e-8)

    cos_a = (z_axis * direction).sum(dim=-1)
    sin_half = torch.sqrt(((1.0 - cos_a) / 2.0).clamp(0, 1))
    cos_half = torch.sqrt(((1.0 + cos_a) / 2.0).clamp(0, 1))

    quat = torch.stack([
        cos_half,
        axis[:, 0] * sin_half,
        axis[:, 1] * sin_half,
        axis[:, 2] * sin_half,
    ], dim=-1)

    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(N, 4)
    quat = torch.where(parallel.unsqueeze(-1), identity, quat)
    return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)


def _add_rotation_noise(quat, std_rad, device, n):
    """
    Apply yaw-only (Z-axis) Gaussian rotation noise.

    Stage 0 (grasp generation) jitters the wrist with Gaussian yaw around Z so
    grasps only cover that distribution.  Stage 1 must use the SAME distribution
    (yaw-only, Gaussian std = wrist_rot_std_deg) to avoid out-of-distribution
    resets that Stage 0 data never covered.
    """
    angle = torch.randn(n, device=device) * std_rad   # Gaussian yaw angle
    half  = angle * 0.5
    zero  = torch.zeros(n, device=device)
    # Pure yaw quaternion (w, x, y, z) = (cos(θ/2), 0, 0, sin(θ/2))
    noise_quat = torch.stack(
        [torch.cos(half), zero, zero, torch.sin(half)], dim=-1
    )
    return _quat_multiply(quat, noise_quat)


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _clamp_translation(delta: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return delta
    norm = torch.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    return delta * scale


def _scaled_delta_quat(
    quat: torch.Tensor,
    gain: float,
    max_angle: float,
) -> torch.Tensor:
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    quat = torch.where(quat[:, :1] < 0.0, -quat, quat)

    half_angle = torch.acos(torch.clamp(quat[:, 0], -1.0, 1.0))
    angle = 2.0 * half_angle
    scaled_angle = torch.clamp(angle * gain, max=max_angle)

    axis = quat[:, 1:].clone()
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    valid = axis_norm.squeeze(-1) > 1e-6
    if valid.any():
        axis[valid] = axis[valid] / axis_norm[valid]
    if (~valid).any():
        axis[~valid] = torch.tensor([0.0, 0.0, 1.0], device=quat.device, dtype=quat.dtype)

    half = 0.5 * scaled_angle
    sin_half = torch.sin(half).unsqueeze(-1)
    cos_half = torch.cos(half).unsqueeze(-1)
    return torch.cat([cos_half, axis * sin_half], dim=-1)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    q0 = q0 / (torch.norm(q0, dim=-1, keepdim=True) + 1e-8)
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    alpha = alpha.unsqueeze(-1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    small = sin_theta.abs() < 1e-6
    w0 = torch.sin((1.0 - alpha) * theta) / (sin_theta + 1e-8)
    w1 = torch.sin(alpha * theta) / (sin_theta + 1e-8)
    out = w0 * q0 + w1 * q1
    out = torch.where(small, (1.0 - alpha) * q0 + alpha * q1, out)
    return out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8)


def _solve_object_pose_from_contacts(
    points_obj: torch.Tensor,    # (n, F, 3)
    points_world: torch.Tensor,  # (n, F, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the rigid transform that maps object-frame grasp contacts to the
    current fingertip world positions.

    Evaluate both the full least-squares rigid fit and all 3-point subsets,
    then keep the transform with the smallest worst-case fingertip error.
    This is more stable than using only the 3-point subsets or only the full
    fit when the stored joint state does not realize every contact exactly.
    """
    n, num_points, _ = points_obj.shape
    if num_points <= 3:
        pos, quat = _solve_rigid_alignment(points_obj, points_world)
        return pos, quat

    def _candidate_error(pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed = _local_to_world_points(points_obj, pos, quat)
        err = torch.norm(reconstructed - points_world, dim=-1)
        return err.max(dim=-1).values, err.mean(dim=-1)

    candidate_solutions: list[tuple[torch.Tensor, torch.Tensor]] = [
        _solve_rigid_alignment(points_obj, points_world)
    ]
    candidate_subsets = list(combinations(range(num_points), 3))
    for subset in candidate_subsets:
        idx = torch.tensor(subset, device=points_obj.device, dtype=torch.long)
        candidate_solutions.append(
            _solve_rigid_alignment(
                points_obj.index_select(1, idx),
                points_world.index_select(1, idx),
            )
        )

    best_pos = None
    best_quat = None
    best_max_err = torch.full((n,), float("inf"), device=points_obj.device)
    best_mean_err = torch.full((n,), float("inf"), device=points_obj.device)

    for pos, quat in candidate_solutions:
        max_err, mean_err = _candidate_error(pos, quat)
        if best_pos is None:
            best_pos = pos
            best_quat = quat
            best_max_err = max_err
            best_mean_err = mean_err
            continue

        improved = (max_err < best_max_err - 1e-6) | (
            torch.isclose(max_err, best_max_err, atol=1e-6) & (mean_err < best_mean_err)
        )
        best_max_err = torch.where(improved, max_err, best_max_err)
        best_mean_err = torch.where(improved, mean_err, best_mean_err)
        best_pos = torch.where(improved.unsqueeze(-1), pos, best_pos)
        best_quat = torch.where(improved.unsqueeze(-1), quat, best_quat)

    return best_pos, best_quat


def _solve_rigid_alignment(
    points_src: torch.Tensor,    # (n, K, 3)
    points_dst: torch.Tensor,    # (n, K, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    src_centroid = points_src.mean(dim=1, keepdim=True)
    dst_centroid = points_dst.mean(dim=1, keepdim=True)

    src_centered = points_src - src_centroid
    dst_centered = points_dst - dst_centroid

    cov = torch.matmul(src_centered.transpose(1, 2), dst_centered)
    u, _, vh = torch.linalg.svd(cov)
    rot = torch.matmul(vh.transpose(1, 2), u.transpose(1, 2))

    det = torch.det(rot)
    reflection = det < 0.0
    if reflection.any():
        vh_reflect = vh.clone()
        vh_reflect[reflection, -1, :] *= -1.0
        rot = torch.matmul(vh_reflect.transpose(1, 2), u.transpose(1, 2))

    pos = dst_centroid.squeeze(1) - torch.matmul(
        rot, src_centroid.squeeze(1).unsqueeze(-1)
    ).squeeze(-1)
    quat = quat_from_matrix(rot)
    return pos, quat


def _world_to_local_points(
    points_world: torch.Tensor,   # (n, F, 3)
    frame_pos: torch.Tensor,      # (n, 3)
    frame_quat: torch.Tensor,     # (n, 4)
) -> torch.Tensor:
    points_rel = points_world - frame_pos.unsqueeze(1)
    return quat_apply_inverse(
        frame_quat.unsqueeze(1).expand(-1, points_world.shape[1], -1).reshape(-1, 4),
        points_rel.reshape(-1, 3),
    ).reshape_as(points_world)


def _local_to_world_points(
    points_local: torch.Tensor,   # (n, F, 3)
    frame_pos: torch.Tensor,      # (n, 3)
    frame_quat: torch.Tensor,     # (n, 4)
) -> torch.Tensor:
    rotated = quat_apply(
        frame_quat.unsqueeze(1).expand(-1, points_local.shape[1], -1).reshape(-1, 4),
        points_local.reshape(-1, 3),
    ).reshape_as(points_local)
    return rotated + frame_pos.unsqueeze(1)


# ---------------------------------------------------------------------------
# IK fallback: set robot joints to approximate fingertip config
# ---------------------------------------------------------------------------

def _set_robot_to_fingertip_config(
    env,
    env_ids: torch.Tensor,
    fingertip_positions: torch.Tensor,   # (n, num_fingers, 3) object frame
):
    """
    Approximate IK fallback when joint angles are not stored in the grasp.

    Maps fingertip distance from object centroid to joint opening angle.
    This is a heuristic — for accurate results, store joint angles during
    grasp generation (see grasp_sampler.py Grasp.joint_angles field).

    Mapping:
      dist from object center → how "open" the hand is
      [0.02 m, 0.12 m] → joint scale [0.1, 0.9]
    """
    robot = env.scene["robot"]
    n     = len(env_ids)

    hand_cfg       = getattr(env.cfg, "hand", None) or {}
    num_fingers    = hand_cfg.get("num_fingers",    fingertip_positions.shape[1])
    dof_per_finger = hand_cfg.get("dof_per_finger", 4)

    default_q = robot.data.default_joint_pos[env_ids].clone()

    # Distance from object centroid (origin in object frame) to each fingertip
    ft_dist_from_obj = torch.norm(fingertip_positions, dim=-1)   # (n, F)

    # Map [0.02, 0.12] → scale [0.1, 0.9]
    scale = ((ft_dist_from_obj - 0.02) / 0.10).clamp(0.0, 1.0) * 0.8 + 0.1

    for f in range(num_fingers):
        s = scale[:, f:f+1]
        jrange = slice(f * dof_per_finger, f * dof_per_finger + dof_per_finger)
        q_low  = robot.data.soft_joint_pos_limits[env_ids, jrange, 0]
        q_high = robot.data.soft_joint_pos_limits[env_ids, jrange, 1]
        default_q[:, jrange] = q_low + s * (q_high - q_low)

    robot.write_joint_state_to_sim(
        default_q,
        torch.zeros_like(default_q),
        env_ids=env_ids,
    )
    robot.set_joint_position_target(default_q, env_ids=env_ids)


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
    default_action = _joint_positions_to_normalized_action(robot, env_ids, joint_pos)
    env.extras["last_action"] = torch.zeros(env.num_envs, num_dof, device=env.device)
    env.extras["current_action"] = torch.zeros(env.num_envs, num_dof, device=env.device)
    env.extras["last_action"][env_ids] = default_action
    env.extras["current_action"][env_ids] = default_action


def _randomise_object_pose(env, env_ids: torch.Tensor):
    """
    Keep the object close to its canonical pose used by grasp generation.

    Large object drop/randomisation at reset breaks the correspondence
    between the sampled grasp-set contact locations and the actual object
    state seen by the policy.
    """
    obj = env.scene["object"]
    n = len(env_ids)
    cfg = getattr(env.cfg, "reset_randomization", {}) or {}

    default_state = obj.data.default_root_state[env_ids].clone()
    # default_root_state is in LOCAL env frame → add env origin to get world frame.
    default_pos = default_state[:, :3].clone() + env.scene.env_origins[env_ids]
    pos_std = float(cfg.get("object_pos_jitter_std", 0.0))
    if pos_std > 0.0:
        default_pos += torch.randn(n, 3, device=env.device) * pos_std

    quat = default_state[:, 3:7].clone()
    rot_std = math.radians(float(cfg.get("object_rot_jitter_deg", 0.0)))
    if rot_std > 0.0:
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        quat = _quat_multiply(quat, _axis_angle_to_quat(axis, torch.randn(n, device=env.device) * rot_std))

    new_state = default_state
    new_state[:, :3] = default_pos
    new_state[:, 3:7] = quat
    new_state[:, 7:]  = 0.0

    obj.write_root_state_to_sim(new_state, env_ids=env_ids)


def _axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half  = angle / 2.0
    sin_h = torch.sin(half).unsqueeze(-1)
    cos_h = torch.cos(half).unsqueeze(-1)
    return torch.cat([cos_h, axis * sin_h], dim=-1)


# ---------------------------------------------------------------------------
# Cached grasp graph loader
# ---------------------------------------------------------------------------

_GRASP_GRAPH_CACHE: dict = {}
_RESET_RNG_CACHE: dict = {}
_FT_IDS_CACHE: dict = {}
_PALM_NORMAL_CACHE: dict = {}


def _get_reset_rng(env) -> np.random.Generator:
    """Cache one RNG per environment instance for reproducible reset sampling."""
    key = id(env)
    if key not in _RESET_RNG_CACHE:
        seed = getattr(env.cfg, "seed", None)
        _RESET_RNG_CACHE[key] = np.random.default_rng(seed)
    return _RESET_RNG_CACHE[key]


def _joint_positions_to_normalized_action(
    robot,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
) -> torch.Tensor:
    joint_lower = robot.data.soft_joint_pos_limits[env_ids, :, 0]
    joint_upper = robot.data.soft_joint_pos_limits[env_ids, :, 1]
    action = 2.0 * (joint_pos - joint_lower) / (joint_upper - joint_lower + 1e-6) - 1.0
    return action.clamp(-1.0, 1.0)


def _get_fingertip_body_ids_from_env(robot, env) -> list[int]:
    key = id(robot)
    if key not in _FT_IDS_CACHE:
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        tip_names = hand_cfg.get(
            "fingertip_links",
            ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"],
        )
        _FT_IDS_CACHE[key] = [robot.find_bodies(name)[0][0] for name in tip_names]
    return _FT_IDS_CACHE[key]


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
