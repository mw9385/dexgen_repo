"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

All functions follow the Isaac Lab RewardTerm signature:
    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  (num_envs,)

Reward design follows DexterityGen §3.2.
"""

from __future__ import annotations

import torch

from .observations import (
    fingertip_positions_in_object_frame,
    target_fingertip_positions,
    _get_fingertip_body_ids,
)


def fingertip_tracking_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """
    Exponential reward for each fingertip approaching its target.

    r = (1/4) Σ_i exp(-α * ||p_i - p*_i||)

    where p_i = current fingertip position, p*_i = target fingertip position
    (both in object frame).

    alpha: sharpness of exponential – higher = reward concentrated near goal
    Returns: (num_envs,)
    """
    current = fingertip_positions_in_object_frame(env).reshape(-1, 4, 3)
    target = target_fingertip_positions(env).reshape(-1, 4, 3)

    dist = torch.norm(current - target, dim=-1)   # (N, 4)
    reward = torch.exp(-alpha * dist).mean(dim=-1) # (N,)
    return reward


def grasp_success_reward(env, threshold: float = 0.01) -> torch.Tensor:
    """
    Binary success bonus: +1 when ALL fingertips are within `threshold` metres
    of their targets.

    Returns: (num_envs,) float (0 or 1)
    """
    current = fingertip_positions_in_object_frame(env).reshape(-1, 4, 3)
    target = target_fingertip_positions(env).reshape(-1, 4, 3)

    dist = torch.norm(current - target, dim=-1)   # (N, 4)
    success = (dist < threshold).all(dim=-1).float()
    return success


def action_smoothness_penalty(env) -> torch.Tensor:
    """
    L2 penalty on action differences between consecutive steps.
    Encourages smooth, jerk-free motion.

    Returns: (num_envs,)
    """
    last_act = env.extras.get("last_action")
    current_act = env.extras.get("current_action")
    if last_act is None or current_act is None:
        return torch.zeros(env.num_envs, device=env.device)

    diff = torch.norm(current_act - last_act, dim=-1)
    return diff


def object_drop_penalty(env, min_height: float = 0.3) -> torch.Tensor:
    """
    Large penalty when the object falls below `min_height`.

    Returns: (num_envs,) float (0 or 1)
    """
    obj = env.scene["object"]
    height = obj.data.root_pos_w[:, 2]   # z position
    dropped = (height < min_height).float()
    return dropped


def joint_limit_penalty(env) -> torch.Tensor:
    """
    Soft penalty for approaching joint limits (within 5% of range).

    Returns: (num_envs,)
    """
    robot = env.scene["robot"]
    q = robot.data.joint_pos
    q_low = robot.data.soft_joint_pos_limits[..., 0]
    q_high = robot.data.soft_joint_pos_limits[..., 1]
    q_range = (q_high - q_low).clamp(min=1e-6)

    # Normalise to [0, 1]
    q_norm = (q - q_low) / q_range

    # Penalty increases near 0 or 1 (at limits)
    penalty = torch.relu(0.05 - q_norm) + torch.relu(q_norm - 0.95)
    return penalty.sum(dim=-1)


def fingertip_contact_reward(env) -> torch.Tensor:
    """
    Bonus for maintaining contact between fingertips and the object.
    Uses contact sensor data if available.

    Returns: (num_envs,)
    """
    contact_sensor = env.scene.sensors.get("contact_sensor")
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Contact force magnitude per fingertip
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]  # (N, K, 3)
    force_mag = torch.norm(forces, dim=-1)                           # (N, K)
    in_contact = (force_mag > 0.1).float()
    return in_contact.mean(dim=-1)
