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
import pickle
from pathlib import Path
from typing import Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Termination predicates
# ---------------------------------------------------------------------------

def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def object_dropped(env, min_height: float = 0.2) -> torch.Tensor:
    obj = env.scene["object"]
    return obj.data.root_pos_w[:, 2] < min_height


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
    num_fingers = getattr(graph, "num_fingers", 4)
    rng = _get_reset_rng(env)

    # ------------------------------------------------------------------
    # 1. Sample start grasps from grasp_set and goals via nearest neighbor
    # ------------------------------------------------------------------
    start_fps_list, goal_fps_list = [], []
    start_joints_list = []
    start_idx_list, goal_idx_list = [], []

    for _ in range(n):
        _, start_fp, goal_fp, start_joints, start_idx, goal_idx = _sample_start_and_nn_goal(
            graph, rng
        )
        start_fps_list.append(start_fp)
        goal_fps_list.append(goal_fp)
        start_joints_list.append(start_joints)
        start_idx_list.append(start_idx)
        goal_idx_list.append(goal_idx)

    start_fps = torch.tensor(
        np.stack(start_fps_list), device=env.device, dtype=torch.float32
    )   # (n, num_fingers, 3)
    goal_fps = torch.tensor(
        np.stack(goal_fps_list), device=env.device, dtype=torch.float32
    )   # (n, num_fingers, 3)

    fp_dim = num_fingers * 3

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

    # ------------------------------------------------------------------
    # 2. Keep the object near the canonical grasp-generation pose.
    # ------------------------------------------------------------------
    _randomise_object_pose(env, env_ids)

    # ------------------------------------------------------------------
    # 3. Keep the wrist close to a nominal grasping pose.
    # ------------------------------------------------------------------
    _randomise_wrist_pose(env, env_ids)

    # ------------------------------------------------------------------
    # 4. Set robot joints
    #    Priority: stored joint angles > IK approximation
    # ------------------------------------------------------------------
    has_joints = any(j is not None for j in start_joints_list)
    if has_joints:
        _set_robot_joints_direct(env, env_ids, start_joints_list)
    else:
        _set_robot_to_fingertip_config(env, env_ids, start_fps)

    # ------------------------------------------------------------------
    # 5. Snap the object into the sampled start grasp.
    # ------------------------------------------------------------------
    _place_object_in_hand(env, env_ids, start_fps)

    # ------------------------------------------------------------------
    # 6. Initialise action buffers (fixes last_action always being 0)
    # ------------------------------------------------------------------
    robot = env.scene["robot"]
    current_q = robot.data.joint_pos[env_ids]  # (n, num_dof)
    if "last_action" not in env.extras:
        env.extras["last_action"] = torch.zeros(
            env.num_envs, current_q.shape[-1], device=env.device
        )
    if "current_action" not in env.extras:
        env.extras["current_action"] = torch.zeros(
            env.num_envs, current_q.shape[-1], device=env.device
        )
    # Reset action buffers to current joint positions for reset envs
    env.extras["last_action"][env_ids] = current_q
    env.extras["current_action"][env_ids] = current_q

    _log_reset_debug(env, env_ids, start_fps)


# ---------------------------------------------------------------------------
# Start grasp sampling + nearest-neighbor goal selection
# ---------------------------------------------------------------------------

def _sample_start_and_nn_goal(
    graph,
    rng: np.random.Generator,
):
    """
    Sample a start grasp directly from the Stage 0 grasp_set and choose the
    nearest-neighbor grasp in fingertip-position space as the goal.
    """
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph, GraspGraph

    if isinstance(graph, MultiObjectGraspGraph):
        obj_name = graph.sample_object(rng)
        g = graph.graphs[obj_name]
    else:
        obj_name = graph.object_name
        g = graph

    N = len(g)
    if N < 2:
        grasp = g.grasp_set[0]
        return (
            obj_name,
            grasp.fingertip_positions.copy(),
            grasp.fingertip_positions.copy(),
            getattr(grasp, "joint_angles", None),
            0,
            0,
        )

    all_fps = g.grasp_set.as_array()
    start_idx = int(rng.integers(0, N))
    start_flat = all_fps[start_idx]

    diffs = all_fps - start_flat
    dists = np.linalg.norm(diffs, axis=-1)
    dists[start_idx] = np.inf
    goal_idx = int(np.argmin(dists))

    start_grasp = g.grasp_set[start_idx]
    goal_grasp = g.grasp_set[goal_idx]

    start_joints = getattr(start_grasp, "joint_angles", None)

    return (
        obj_name,
        start_grasp.fingertip_positions.copy(),
        goal_grasp.fingertip_positions.copy(),
        start_joints,
        int(start_idx),
        int(goal_idx),
    )


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
            j = torch.tensor(joints, device=env.device, dtype=torch.float32)
            if j.shape[0] == num_dof:
                joint_pos[i] = j

    robot.write_joint_state_to_sim(
        joint_pos,
        torch.zeros_like(joint_pos),
        env_ids=env_ids,
    )


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
    rot_std = math.radians(float(cfg.get("wrist_rot_std_deg", 5.0)))
    if rot_std > 0.0:
        wrist_quat = _add_rotation_noise(wrist_quat, rot_std, env.device, n)

    robot.write_root_pose_to_sim(
        torch.cat([wrist_pos, wrist_quat], dim=-1),
        env_ids=env_ids,
    )


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

    pos_w, quat_w = _solve_object_pose_from_contacts(
        points_obj=fingertip_positions_obj,
        points_world=ft_world,
    )

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = pos_w
    root_state[:, 3:7] = quat_w
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def _log_reset_debug(
    env,
    env_ids: torch.Tensor,
    start_fps_obj: torch.Tensor,   # (n, F, 3)
):
    """
    Print high-signal reset diagnostics for the first few resets so we can see
    whether the object is actually being placed into the sampled start grasp.
    """
    debug_cfg = getattr(env.cfg, "reset_debug", {}) or {}
    if not bool(debug_cfg.get("enabled", True)):
        return

    limit = int(debug_cfg.get("max_prints", 8))
    counter = int(env.extras.get("_reset_debug_counter", 0))
    if counter >= limit:
        return

    robot = env.scene["robot"]
    obj = env.scene["object"]
    ft_ids = _get_fingertip_body_ids_from_env(robot, env)

    ft_world = robot.data.body_pos_w[env_ids][:, ft_ids, :].clone()
    obj_pos = obj.data.root_pos_w[env_ids].clone()
    obj_quat = obj.data.root_quat_w[env_ids].clone()
    ft_obj = _world_to_local_points(ft_world, obj_pos, obj_quat)

    err = torch.norm(ft_obj - start_fps_obj, dim=-1)
    mean_err = err.mean(dim=-1)
    max_err = err.max(dim=-1).values

    start_idx = env.extras.get("start_grasp_idx")
    goal_idx = env.extras.get("goal_grasp_idx")

    num_to_print = min(len(env_ids), max(limit - counter, 0))
    for local_i in range(num_to_print):
        env_i = int(env_ids[local_i].item())
        start_i = int(start_idx[env_i].item()) if start_idx is not None else -1
        goal_i = int(goal_idx[env_i].item()) if goal_idx is not None else -1
        pos = obj_pos[local_i].detach().cpu().tolist()
        quat = obj_quat[local_i].detach().cpu().tolist()
        print(
            "[reset-debug] "
            f"env={env_i} start={start_i} goal={goal_i} "
            f"obj_pos={[round(v, 4) for v in pos]} "
            f"obj_quat={[round(v, 4) for v in quat]} "
            f"mean_contact_err={mean_err[local_i].item():.5f} "
            f"max_contact_err={max_err[local_i].item():.5f}"
        )

    env.extras["_reset_debug_counter"] = counter + num_to_print


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
    axis  = torch.randn(n, 3, device=device)
    axis  = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    angle = torch.randn(n, device=device) * std_rad
    noise_quat = _axis_angle_to_quat(axis, angle)
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


def _quat_from_rotmat(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (..., 3, 3) to quaternions (w, x, y, z)."""
    batch = rot.shape[0]
    quat = torch.zeros(batch, 4, device=rot.device, dtype=rot.dtype)

    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    positive = trace > 0.0

    if positive.any():
        t = torch.sqrt(trace[positive] + 1.0) * 2.0
        quat[positive, 0] = 0.25 * t
        quat[positive, 1] = (rot[positive, 2, 1] - rot[positive, 1, 2]) / t
        quat[positive, 2] = (rot[positive, 0, 2] - rot[positive, 2, 0]) / t
        quat[positive, 3] = (rot[positive, 1, 0] - rot[positive, 0, 1]) / t

    mask = ~positive
    if mask.any():
        r = rot[mask]
        diag = torch.stack([r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]], dim=-1)
        idx = torch.argmax(diag, dim=-1)

        for axis in range(3):
            axis_mask = mask.clone()
            axis_mask[mask] = idx == axis
            if not axis_mask.any():
                continue
            rr = rot[axis_mask]
            if axis == 0:
                t = torch.sqrt(1.0 + rr[:, 0, 0] - rr[:, 1, 1] - rr[:, 2, 2]) * 2.0
                quat[axis_mask, 0] = (rr[:, 2, 1] - rr[:, 1, 2]) / t
                quat[axis_mask, 1] = 0.25 * t
                quat[axis_mask, 2] = (rr[:, 0, 1] + rr[:, 1, 0]) / t
                quat[axis_mask, 3] = (rr[:, 0, 2] + rr[:, 2, 0]) / t
            elif axis == 1:
                t = torch.sqrt(1.0 + rr[:, 1, 1] - rr[:, 0, 0] - rr[:, 2, 2]) * 2.0
                quat[axis_mask, 0] = (rr[:, 0, 2] - rr[:, 2, 0]) / t
                quat[axis_mask, 1] = (rr[:, 0, 1] + rr[:, 1, 0]) / t
                quat[axis_mask, 2] = 0.25 * t
                quat[axis_mask, 3] = (rr[:, 1, 2] + rr[:, 2, 1]) / t
            else:
                t = torch.sqrt(1.0 + rr[:, 2, 2] - rr[:, 0, 0] - rr[:, 1, 1]) * 2.0
                quat[axis_mask, 0] = (rr[:, 1, 0] - rr[:, 0, 1]) / t
                quat[axis_mask, 1] = (rr[:, 0, 2] + rr[:, 2, 0]) / t
                quat[axis_mask, 2] = (rr[:, 1, 2] + rr[:, 2, 1]) / t
                quat[axis_mask, 3] = 0.25 * t

    return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)


def _solve_object_pose_from_contacts(
    points_obj: torch.Tensor,    # (n, F, 3)
    points_world: torch.Tensor,  # (n, F, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the rigid transform that maps object-frame grasp contacts to the
    current fingertip world positions using a batched Kabsch alignment.
    """
    obj_centroid = points_obj.mean(dim=1, keepdim=True)
    world_centroid = points_world.mean(dim=1, keepdim=True)

    obj_centered = points_obj - obj_centroid
    world_centered = points_world - world_centroid

    cov = torch.matmul(obj_centered.transpose(1, 2), world_centered)
    u, _, vh = torch.linalg.svd(cov)
    rot = torch.matmul(vh.transpose(1, 2), u.transpose(1, 2))

    det = torch.det(rot)
    reflection = det < 0.0
    if reflection.any():
        vh_reflect = vh.clone()
        vh_reflect[reflection, -1, :] *= -1.0
        rot = torch.matmul(vh_reflect.transpose(1, 2), u.transpose(1, 2))

    pos = world_centroid.squeeze(1) - torch.matmul(
        rot, obj_centroid.squeeze(1).unsqueeze(-1)
    ).squeeze(-1)
    quat = _quat_from_rotmat(rot)
    return pos, quat


def _world_to_local_points(
    points_world: torch.Tensor,   # (n, F, 3)
    frame_pos: torch.Tensor,      # (n, 3)
    frame_quat: torch.Tensor,     # (n, 4)
) -> torch.Tensor:
    points_rel = points_world - frame_pos.unsqueeze(1)
    quat_inv = torch.cat([frame_quat[:, :1], -frame_quat[:, 1:]], dim=-1)
    return _quat_rotate_points(quat_inv, points_rel)


def _quat_rotate_points(
    quat: torch.Tensor,    # (n, 4)
    points: torch.Tensor,  # (n, F, 3)
) -> torch.Tensor:
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    w = quat[:, 0:1].unsqueeze(-1)
    xyz = quat[:, 1:].unsqueeze(1)
    t = 2.0 * torch.cross(xyz.expand_as(points), points, dim=-1)
    return points + w * t + torch.cross(xyz.expand_as(t), t, dim=-1)


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


def _reset_to_default_pose(env, env_ids: torch.Tensor):
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    num_fingers = hand_cfg.get("num_fingers", 4)
    num_dof = joint_pos.shape[-1]
    env.extras["target_fingertip_pos"] = torch.zeros(
        env.num_envs, num_fingers * 3, device=env.device
    )
    env.extras["last_action"] = torch.zeros(
        env.num_envs, num_dof, device=env.device
    )
    env.extras["current_action"] = torch.zeros(
        env.num_envs, num_dof, device=env.device
    )


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
    default_pos = default_state[:, :3].clone()
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


def _get_reset_rng(env) -> np.random.Generator:
    """Cache one RNG per environment instance for reproducible reset sampling."""
    key = id(env)
    if key not in _RESET_RNG_CACHE:
        seed = getattr(env.cfg, "seed", None)
        _RESET_RNG_CACHE[key] = np.random.default_rng(seed)
    return _RESET_RNG_CACHE[key]


def _get_fingertip_body_ids_from_env(robot, env) -> list[int]:
    key = id(robot)
    if key not in _FT_IDS_CACHE:
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        tip_names = hand_cfg.get(
            "fingertip_links",
            ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"],
        )
        _FT_IDS_CACHE[key] = [robot.find_bodies(name)[0][0] for name in tip_names]
    return _FT_IDS_CACHE[key]


def _load_grasp_graph(env):
    """Load and cache the MultiObjectGraspGraph (or GraspGraph) from cfg."""
    path = getattr(env.cfg, "grasp_graph_path", None)
    if path is None:
        return None
    if path in _GRASP_GRAPH_CACHE:
        return _GRASP_GRAPH_CACHE[path]

    p = Path(path)
    if not p.exists():
        print(f"[events] Warning: GraspGraph not found at {p}. "
              f"Run scripts/run_grasp_generation.py first.")
        return None

    with open(p, "rb") as f:
        graph = pickle.load(f)

    from grasp_generation.rrt_expansion import GraspGraph, MultiObjectGraspGraph
    if isinstance(graph, GraspGraph):
        multi = MultiObjectGraspGraph()
        multi.add(graph, {"name": graph.object_name, "shape_type": "cube",
                          "size": 0.06, "mass": 0.1, "color": (0.8, 0.2, 0.2)})
        graph = multi
        print(f"[events] Wrapped single GraspGraph into MultiObjectGraspGraph")

    _GRASP_GRAPH_CACHE[str(path)] = graph
    graph.summary()
    return graph
