"""
Event (reset / randomisation) functions for the AnyGrasp-to-AnyGrasp env.

Reset logic per episode:
  1. Sample a random object name from MultiObjectGraspGraph
  2. Sample a random edge (g_start, g_goal) for that object
  3. Randomise Allegro Hand wrist position (hemisphere above object)
  4. Set hand joint positions to approximate g_start
  5. Store g_goal as target in env.extras
  6. Randomise object pose (small noise around default)
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Optional

import torch


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
    env_ids: Optional[torch.Tensor] = None,
):
    """
    Full episode reset:
      1. Sample random object + grasp edge from MultiObjectGraspGraph
      2. Randomise wrist pose (position + orientation)
      3. Set robot joints to approximate start grasp
      4. Store goal fingertip positions
      5. Randomise object pose
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    graph = _load_grasp_graph(env)
    if graph is None:
        _reset_to_default_pose(env, env_ids)
        return

    n = len(env_ids)
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Sample random object + (start, goal) grasp pair per env
    # ------------------------------------------------------------------
    rng = np.random.default_rng()
    start_fps_list, goal_fps_list = [], []

    for _ in range(n):
        obj_name, (si, gi) = graph.sample_edge(rng=rng)
        start_fps_list.append(graph.get_grasp(obj_name, si).fingertip_positions)
        goal_fps_list.append(graph.get_grasp(obj_name, gi).fingertip_positions)

    start_fps = torch.tensor(
        np.stack(start_fps_list), device=env.device, dtype=torch.float32
    )   # (n, 4, 3)
    goal_fps = torch.tensor(
        np.stack(goal_fps_list), device=env.device, dtype=torch.float32
    )   # (n, 4, 3)

    # Store goal
    if "target_fingertip_pos" not in env.extras:
        env.extras["target_fingertip_pos"] = torch.zeros(
            env.num_envs, 12, device=env.device
        )
    env.extras["target_fingertip_pos"][env_ids] = goal_fps.reshape(n, 12)

    # ------------------------------------------------------------------
    # 2. Randomise object pose first (wrist will be placed relative to it)
    # ------------------------------------------------------------------
    _randomise_object_pose(env, env_ids)

    # ------------------------------------------------------------------
    # 3. Randomise wrist position
    # ------------------------------------------------------------------
    _randomise_wrist_pose(env, env_ids)

    # ------------------------------------------------------------------
    # 4. Set robot joints to approximate start grasp
    # ------------------------------------------------------------------
    _set_robot_to_fingertip_config(env, env_ids, start_fps)


# ---------------------------------------------------------------------------
# Wrist randomization  (NEW)
# ---------------------------------------------------------------------------

def _randomise_wrist_pose(env, env_ids: torch.Tensor):
    """
    Randomise the Allegro Hand wrist (robot base) position and orientation.

    Strategy:
      - Sample a point on a hemisphere of radius [r_min, r_max] above the object
      - The wrist z-axis points roughly toward the object centre
      - Add small random rotation noise around that pointing direction

    Config is read from env.cfg.wrist_randomization (set in AnyGraspEnvCfg).
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

    # Sample horizontal offset in a ring of radius [r_min, r_max]
    radius = torch.empty(n, device=env.device).uniform_(r_min, r_max)
    angle  = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
    x_off  = radius * torch.cos(angle)
    y_off  = radius * torch.sin(angle)

    # Sample height above object
    z_off = torch.empty(n, device=env.device).uniform_(h_min, h_max)

    wrist_pos = obj_pos + torch.stack([x_off, y_off, z_off], dim=-1)  # (n, 3)

    # Orientation: wrist z-axis points from wrist toward object (palm-down)
    to_obj = obj_pos - wrist_pos                              # (n, 3)
    to_obj = to_obj / (torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8)

    # Build quaternion that aligns +z → to_obj direction, then add noise
    wrist_quat = _look_at_quat(to_obj)                        # (n, 4)
    wrist_quat = _add_rotation_noise(wrist_quat, rot_std, env.device, n)

    # Write root pose to sim
    # Isaac Lab API: write_root_pose_to_sim(pos, quat, env_ids)
    robot.write_root_pose_to_sim(
        torch.cat([wrist_pos, wrist_quat], dim=-1),
        env_ids=env_ids,
    )


def _look_at_quat(direction: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion (w, x, y, z) that rotates the canonical +Z axis
    to point along `direction`.

    direction: (N, 3) unit vectors
    Returns:   (N, 4) quaternions
    """
    N = direction.shape[0]
    device = direction.device

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3)

    # Rotation axis = z_axis × direction
    axis = torch.cross(z_axis, direction, dim=-1)           # (N, 3)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # cos(angle) = dot(z_axis, direction)
    cos_a = (z_axis * direction).sum(dim=-1)                # (N,)

    # Handle near-parallel case (angle ≈ 0 or π)
    parallel = axis_norm.squeeze(-1) < 1e-6
    axis = axis / (axis_norm + 1e-8)

    sin_half = torch.sqrt(((1.0 - cos_a) / 2.0).clamp(0, 1))
    cos_half = torch.sqrt(((1.0 + cos_a) / 2.0).clamp(0, 1))

    quat = torch.stack([
        cos_half,
        axis[:, 0] * sin_half,
        axis[:, 1] * sin_half,
        axis[:, 2] * sin_half,
    ], dim=-1)                                               # (N, 4)

    # For exactly-parallel cases, use identity quaternion
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(N, 4)
    quat = torch.where(parallel.unsqueeze(-1), identity, quat)

    return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)


def _add_rotation_noise(
    quat: torch.Tensor,      # (N, 4) w, x, y, z
    std_rad: float,
    device: torch.device,
    n: int,
) -> torch.Tensor:
    """Perturb quaternions by a small random rotation (axis-angle noise)."""
    axis  = torch.randn(n, 3, device=device)
    axis  = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    angle = torch.randn(n, device=device) * std_rad
    noise_quat = _axis_angle_to_quat(axis, angle)
    return _quat_multiply(quat, noise_quat)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternion tensors (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ---------------------------------------------------------------------------
# Existing helpers (updated)
# ---------------------------------------------------------------------------

def _reset_to_default_pose(env, env_ids: torch.Tensor):
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.extras["target_fingertip_pos"] = torch.zeros(
        env.num_envs, 12, device=env.device
    )


def _set_robot_to_fingertip_config(
    env,
    env_ids: torch.Tensor,
    fingertip_positions: torch.Tensor,   # (n, 4, 3) object frame
):
    """Approximate IK: scale joint angles to reach fingertip targets."""
    robot = env.scene["robot"]
    obj   = env.scene["object"]
    n     = len(env_ids)

    default_q = robot.data.default_joint_pos[env_ids].clone()   # (n, 16)
    obj_pos   = obj.data.root_pos_w[env_ids]                    # (n, 3)
    ft_world  = fingertip_positions + obj_pos.unsqueeze(1)      # (n, 4, 3)
    palm_pos  = robot.data.body_pos_w[env_ids, 0, :]            # (n, 3) base link
    ft_dist   = torch.norm(ft_world - palm_pos.unsqueeze(1), dim=-1)  # (n, 4)

    # Map distance [0.05, 0.18] → joint scale [0.2, 1.0]
    scale = ((ft_dist - 0.05) / 0.13).clamp(0, 1) * 0.8 + 0.2   # (n, 4)

    for f in range(4):
        s = scale[:, f:f+1]
        jrange = slice(f * 4, f * 4 + 4)
        q_upper = robot.data.soft_joint_pos_limits[env_ids, jrange, 1]
        default_q[:, jrange] = q_upper * s

    robot.write_joint_state_to_sim(
        default_q,
        torch.zeros_like(default_q),
        env_ids=env_ids,
    )


def _randomise_object_pose(env, env_ids: torch.Tensor):
    """Small random perturbation to object position and orientation."""
    obj = env.scene["object"]
    n   = len(env_ids)

    default_pos  = obj.data.default_root_state[env_ids, :3].clone()
    pos_noise    = torch.randn(n, 3, device=env.device) * 0.015  # ±1.5 cm
    default_pos += pos_noise

    angle = torch.randn(n, device=env.device) * 0.10             # ~6° std
    axis  = torch.randn(n, 3, device=env.device)
    axis  = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    quat  = _axis_angle_to_quat(axis, angle)

    new_state         = obj.data.default_root_state[env_ids].clone()
    new_state[:, :3]  = default_pos
    new_state[:, 3:7] = quat
    new_state[:, 7:]  = 0.0

    obj.write_root_state_to_sim(new_state, env_ids=env_ids)


def _axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half    = angle / 2.0
    sin_h   = torch.sin(half).unsqueeze(-1)
    cos_h   = torch.cos(half).unsqueeze(-1)
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

    # Wrap legacy single-object GraspGraph in a MultiObjectGraspGraph
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
