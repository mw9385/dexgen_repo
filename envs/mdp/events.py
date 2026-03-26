"""
Event (reset / randomisation) functions for the AnyGrasp-to-AnyGrasp env.

Key reset logic (DexterityGen §3.2):
  1. Sample a random edge (g_start, g_goal) from the GraspGraph
  2. Place the Allegro Hand at g_start (set joint targets to match fingertip pos)
  3. Store g_goal as the target in env.extras
  4. Randomise object pose slightly (domain randomisation)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Termination predicates
# ---------------------------------------------------------------------------

def time_out(env) -> torch.Tensor:
    """Episode time-out (handled automatically by Isaac Lab)."""
    return env.episode_length_buf >= env.max_episode_length


def object_dropped(env, min_height: float = 0.3) -> torch.Tensor:
    """True when object z position falls below threshold."""
    obj = env.scene["object"]
    height = obj.data.root_pos_w[:, 2]
    return height < min_height


# ---------------------------------------------------------------------------
# Reset events
# ---------------------------------------------------------------------------

def reset_to_random_grasp(
    env,
    env_ids: Optional[torch.Tensor] = None,
):
    """
    Reset selected environments to a random (start, goal) grasp pair
    from the GraspGraph.

    Steps:
      1. Load GraspGraph (cached after first call)
      2. Sample random edge per env
      3. Set robot joint positions to approximate g_start
      4. Set target fingertip positions to g_goal
      5. Randomise object pose (small noise)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    graph = _load_grasp_graph(env)
    if graph is None:
        # No grasp graph available → reset to zero pose
        _reset_to_default_pose(env, env_ids)
        return

    n = len(env_ids)
    edge_indices = torch.randint(0, len(graph.edges), (n,))

    # Gather start and goal fingertip positions
    start_fps = []
    goal_fps = []
    for idx in edge_indices:
        edge = graph.edges[int(idx)]
        start_fps.append(graph.grasp_set[edge[0]].fingertip_positions)
        goal_fps.append(graph.grasp_set[edge[1]].fingertip_positions)

    import numpy as np
    start_fps = torch.tensor(np.stack(start_fps), device=env.device)  # (n, 4, 3)
    goal_fps = torch.tensor(np.stack(goal_fps), device=env.device)    # (n, 4, 3)

    # Store goal in extras (used by observation and reward terms)
    if "target_fingertip_pos" not in env.extras:
        env.extras["target_fingertip_pos"] = torch.zeros(
            env.num_envs, 12, device=env.device
        )
    env.extras["target_fingertip_pos"][env_ids] = goal_fps.reshape(n, 12)

    # Reset robot to start grasp via joint position IK approximation
    _set_robot_to_fingertip_config(env, env_ids, start_fps)

    # Randomise object pose slightly
    _randomise_object_pose(env, env_ids)


def _reset_to_default_pose(env, env_ids: torch.Tensor):
    """Fallback: reset to neutral hand pose."""
    robot = env.scene["robot"]
    n = len(env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.extras["target_fingertip_pos"] = torch.zeros(
        env.num_envs, 12, device=env.device
    )


def _set_robot_to_fingertip_config(
    env,
    env_ids: torch.Tensor,
    fingertip_positions: torch.Tensor,  # (n, 4, 3) in object frame
):
    """
    Approximate IK: set robot joint positions to roughly achieve
    the given fingertip positions.

    For now uses default pose + a heuristic scaling.
    Full FK/IK can be replaced with a proper solver (e.g. pytorch_kinematics).
    """
    robot = env.scene["robot"]
    default_q = robot.data.default_joint_pos[env_ids].clone()  # (n, 16)

    # Heuristic: scale finger joints proportionally to fingertip target distance
    # Each finger has 4 joints; roughly: close fingers → increase joint angles
    obj = env.scene["object"]
    obj_pos = obj.data.root_pos_w[env_ids]   # (n, 3)

    # Convert fingertip pos from object frame to world frame (rough)
    ft_world = fingertip_positions + obj_pos.unsqueeze(1)   # (n, 4, 3)

    # Simple heuristic: fingers closer to palm → more open joints
    palm_pos = robot.data.body_pos_w[env_ids, 0, :]        # (n, 3) base link
    ft_dist = torch.norm(ft_world - palm_pos.unsqueeze(1), dim=-1)  # (n, 4)

    # Map distance [0.05, 0.15] → joint angle scale [0.3, 1.0]
    scale = ((ft_dist - 0.05) / 0.10).clamp(0, 1) * 0.7 + 0.3  # (n, 4)

    # Apply scale to each finger's 4 joints
    for f in range(4):
        s = scale[:, f:f+1]           # (n, 1)
        joint_range = slice(f * 4, f * 4 + 4)
        q_f = robot.data.soft_joint_pos_limits[env_ids, joint_range, 1]  # upper limit
        default_q[:, joint_range] = q_f * s

    joint_vel = torch.zeros_like(default_q)
    robot.write_joint_state_to_sim(default_q, joint_vel, env_ids=env_ids)


def _randomise_object_pose(env, env_ids: torch.Tensor):
    """Apply small random perturbation to the object pose."""
    obj = env.scene["object"]
    n = len(env_ids)

    # Default position with small noise
    default_pos = obj.data.default_root_state[env_ids, :3].clone()
    pos_noise = torch.randn(n, 3, device=env.device) * 0.01    # ±1 cm
    default_pos += pos_noise

    # Small random rotation (≤5°)
    angle = torch.randn(n, device=env.device) * 0.087           # ~5° std
    axis = torch.randn(n, 3, device=env.device)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    quat = _axis_angle_to_quat(axis, angle)

    new_state = obj.data.default_root_state[env_ids].clone()
    new_state[:, :3] = default_pos
    new_state[:, 3:7] = quat
    new_state[:, 7:] = 0.0   # zero velocity

    obj.write_root_state_to_sim(new_state, env_ids=env_ids)


def _axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to quaternion (w, x, y, z)."""
    half = angle / 2.0
    sin_h = torch.sin(half).unsqueeze(-1)
    cos_h = torch.cos(half).unsqueeze(-1)
    return torch.cat([cos_h, axis * sin_h], dim=-1)


# ---------------------------------------------------------------------------
# Cached grasp graph loader
# ---------------------------------------------------------------------------

_GRASP_GRAPH_CACHE: dict = {}


def _load_grasp_graph(env):
    """Load and cache the GraspGraph from env.cfg.grasp_graph_path."""
    path = getattr(env.cfg, "grasp_graph_path", None)
    if path is None:
        return None
    if path in _GRASP_GRAPH_CACHE:
        return _GRASP_GRAPH_CACHE[path]

    path = Path(path)
    if not path.exists():
        print(f"[events] Warning: GraspGraph not found at {path}. "
              f"Run scripts/run_grasp_generation.py first.")
        return None

    with open(path, "rb") as f:
        graph = pickle.load(f)
    _GRASP_GRAPH_CACHE[str(path)] = graph
    print(f"[events] Loaded GraspGraph: {len(graph)} nodes, {graph.num_edges} edges")
    return graph
