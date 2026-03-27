"""
Event (reset / randomisation) functions for the AnyGrasp-to-AnyGrasp env.

Reset logic per episode:
  1. Sample a random start grasp from GraspGraph
  2. Find the NEAREST NEIGHBOR grasp as the goal (paper §3.2)
  3. Randomise object pose (small noise around default)
  4. Set hand joint positions using stored joint angles (or IK fallback)
  5. Store goal fingertip positions in env.extras

Key fix: Nearest Neighbor goal selection
  - Paper: "the goal grasp is the nearest grasp in the GraspGraph
    to the current start grasp position"
  - Implementation: compute L2 distance in fingertip-position space
    between start and all other grasps, pick the closest one
    (excluding self). Optionally sample from top-K nearest for diversity.
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
    Full episode reset implementing DexterityGen §3.2:

      1. Sample random start grasp g_start from GraspGraph
      2. Find nearest neighbor g_goal using L2 distance in fingertip space
         (Nearest Neighbor — paper's key design choice for curriculum)
      3. Randomise object pose
      4. Set robot joints to approximate g_start
      5. Store g_goal fingertip positions as target
    """
    graph = _load_grasp_graph(env)
    if graph is None:
        _reset_to_default_pose(env, env_ids)
        return

    n = len(env_ids)
    num_fingers = getattr(graph, "num_fingers", 4)

    # ------------------------------------------------------------------
    # 1 & 2. Sample start grasp + find nearest neighbor goal
    # ------------------------------------------------------------------
    rng = np.random.default_rng()
    start_fps_list, goal_fps_list = [], []
    start_joints_list = []

    for _ in range(n):
        obj_name, start_fp, goal_fp, start_joints = _sample_start_and_nn_goal(
            graph, rng
        )
        start_fps_list.append(start_fp)
        goal_fps_list.append(goal_fp)
        start_joints_list.append(start_joints)

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

    # ------------------------------------------------------------------
    # 3. Randomise object pose first (wrist placed relative to it)
    # ------------------------------------------------------------------
    _randomise_object_pose(env, env_ids)

    # ------------------------------------------------------------------
    # 4. Randomise wrist position
    # ------------------------------------------------------------------
    _randomise_wrist_pose(env, env_ids)

    # ------------------------------------------------------------------
    # 5. Set robot joints
    #    Priority: stored joint angles > IK approximation
    # ------------------------------------------------------------------
    has_joints = any(j is not None for j in start_joints_list)
    if has_joints:
        _set_robot_joints_direct(env, env_ids, start_joints_list)
    else:
        _set_robot_to_fingertip_config(env, env_ids, start_fps)

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


# ---------------------------------------------------------------------------
# Nearest Neighbor goal selection  (CORE FIX)
# ---------------------------------------------------------------------------

def _sample_start_and_nn_goal(
    graph,
    rng: np.random.Generator,
    top_k: int = 5,
):
    """
    Sample a start grasp and find its nearest neighbor as the goal.

    Algorithm (DexterityGen §3.2):
      1. Uniformly sample a start grasp index from the GraspGraph
      2. Compute L2 distance in flattened fingertip-position space
         between start and ALL other grasps in the same object's graph
      3. Select the nearest neighbor (or sample from top-K for diversity)

    Args:
        graph:  MultiObjectGraspGraph or GraspGraph
        rng:    numpy random generator
        top_k:  sample goal from top-K nearest neighbors (diversity)

    Returns:
        obj_name:     str
        start_fp:     (num_fingers, 3) ndarray  — start fingertip positions
        goal_fp:      (num_fingers, 3) ndarray  — goal fingertip positions
        start_joints: (num_dof,) ndarray or None — joint angles if stored
    """
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph, GraspGraph

    # Handle both MultiObjectGraspGraph and single GraspGraph
    if isinstance(graph, MultiObjectGraspGraph):
        obj_name = graph.sample_object(rng)
        g = graph.graphs[obj_name]
    else:
        obj_name = graph.object_name
        g = graph

    N = len(g)
    if N < 2:
        # Degenerate: only one grasp, use it for both start and goal
        grasp = g.grasp_set[0]
        return (
            obj_name,
            grasp.fingertip_positions.copy(),
            grasp.fingertip_positions.copy(),
            getattr(grasp, "joint_angles", None),
        )

    # All fingertip positions as flat array: (N, num_fingers*3)
    all_fps = g.grasp_set.as_array()   # (N, F*3)

    # 1. Sample start index uniformly
    start_idx = int(rng.integers(0, N))
    start_flat = all_fps[start_idx]    # (F*3,)

    # 2. Compute L2 distances to all other grasps
    diffs = all_fps - start_flat       # (N, F*3)
    dists = np.linalg.norm(diffs, axis=-1)  # (N,)
    dists[start_idx] = np.inf          # exclude self

    # 3. Select goal from top-K nearest neighbors
    k = min(top_k, N - 1)
    top_k_indices = np.argpartition(dists, k)[:k]  # indices of k smallest
    goal_idx = int(rng.choice(top_k_indices))

    start_grasp = g.grasp_set[start_idx]
    goal_grasp  = g.grasp_set[goal_idx]

    start_joints = getattr(start_grasp, "joint_angles", None)

    return (
        obj_name,
        start_grasp.fingertip_positions.copy(),
        goal_grasp.fingertip_positions.copy(),
        start_joints,
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
    Randomise the Allegro Hand wrist (robot base) position and orientation.

    Strategy:
      - Sample a point on a hemisphere of radius [r_min, r_max] above the object
      - The wrist z-axis points roughly toward the object centre
      - Add small random rotation noise around that pointing direction
    """
    robot = env.scene["robot"]
    obj   = env.scene["object"]
    n     = len(env_ids)
    cfg   = getattr(env.cfg, "wrist_randomization", {})

    r_min   = cfg.get("pos_radius_min", 0.12)
    r_max   = cfg.get("pos_radius_max", 0.22)
    h_min   = cfg.get("pos_height_min", 0.08)
    h_max   = cfg.get("pos_height_max", 0.20)
    rot_std = math.radians(cfg.get("rot_std_deg", 15.0))

    obj_pos = obj.data.root_pos_w[env_ids]   # (n, 3)

    radius = torch.empty(n, device=env.device).uniform_(r_min, r_max)
    angle  = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
    x_off  = radius * torch.cos(angle)
    y_off  = radius * torch.sin(angle)
    z_off  = torch.empty(n, device=env.device).uniform_(h_min, h_max)

    wrist_pos = obj_pos + torch.stack([x_off, y_off, z_off], dim=-1)  # (n, 3)

    to_obj = obj_pos - wrist_pos
    to_obj = to_obj / (torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8)

    wrist_quat = _look_at_quat(to_obj)
    wrist_quat = _add_rotation_noise(wrist_quat, rot_std, env.device, n)

    robot.write_root_pose_to_sim(
        torch.cat([wrist_pos, wrist_quat], dim=-1),
        env_ids=env_ids,
    )


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
    """Small random perturbation to object position and orientation."""
    obj = env.scene["object"]
    n   = len(env_ids)

    default_pos  = obj.data.default_root_state[env_ids, :3].clone()
    pos_noise    = torch.randn(n, 3, device=env.device) * 0.015
    default_pos += pos_noise

    angle = torch.randn(n, device=env.device) * 0.10
    axis  = torch.randn(n, 3, device=env.device)
    axis  = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    quat  = _axis_angle_to_quat(axis, angle)

    new_state         = obj.data.default_root_state[env_ids].clone()
    new_state[:, :3]  = default_pos
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
