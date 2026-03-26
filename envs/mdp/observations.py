"""
Observation functions for the AnyGrasp-to-AnyGrasp environment.

All functions follow the Isaac Lab ObservationTerm signature:
    func(env: ManagerBasedRLEnv) -> torch.Tensor

Tensors are (num_envs, dim).
"""

from __future__ import annotations

import torch


def joint_positions_normalized(env) -> torch.Tensor:
    """
    Allegro Hand joint positions, normalized to [-1, 1] using joint limits.

    Returns: (num_envs, 16)
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos                     # (N, 16)
    q_low = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    q_norm = 2.0 * (q - q_low) / (q_high - q_low + 1e-6) - 1.0
    return q_norm.clamp(-1.0, 1.0)


def joint_velocities_normalized(env) -> torch.Tensor:
    """
    Allegro Hand joint velocities, clipped and normalized.

    Returns: (num_envs, 16)
    """
    asset = env.scene["robot"]
    dq = asset.data.joint_vel                    # (N, 16)
    return (dq / 5.0).clamp(-1.0, 1.0)          # assume max vel ~5 rad/s


def fingertip_positions_in_object_frame(env) -> torch.Tensor:
    """
    Current fingertip positions expressed in the object's local frame.

    The object frame centres the observation, making the policy
    object-pose invariant (key insight from DexterityGen §3.2).

    Returns: (num_envs, 12)  [index, middle, ring, thumb] × [x, y, z]
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Fingertip body indices (Allegro: bodies 4, 8, 12, 16 are fingertip links)
    fingertip_ids = _get_fingertip_body_ids(robot)

    # World-frame fingertip positions: (N, 4, 3)
    ft_world = robot.data.body_pos_w[:, fingertip_ids, :]

    # Object pose
    obj_pos = obj.data.root_pos_w        # (N, 3)
    obj_quat = obj.data.root_quat_w      # (N, 4)  w, x, y, z

    # Transform to object frame
    ft_obj = _transform_points_to_local_frame(ft_world, obj_pos, obj_quat)
    return ft_obj.reshape(env.num_envs, -1)      # (N, 12)


def target_fingertip_positions(env) -> torch.Tensor:
    """
    Goal fingertip positions in the object frame.

    Set by the reset event and stored in env.extras["target_fingertip_pos"].

    Returns: (num_envs, 12)
    """
    target = env.extras.get("target_fingertip_pos")
    if target is None:
        return torch.zeros(env.num_envs, 12, device=env.device)
    return target


def object_linear_velocity(env) -> torch.Tensor:
    """Object linear velocity (world frame). Returns: (num_envs, 3)"""
    obj = env.scene["object"]
    return obj.data.root_lin_vel_w


def object_angular_velocity(env) -> torch.Tensor:
    """Object angular velocity (world frame). Returns: (num_envs, 3)"""
    obj = env.scene["object"]
    return obj.data.root_ang_vel_w


def last_action(env) -> torch.Tensor:
    """Last action taken by the policy. Returns: (num_envs, 16)"""
    action = env.extras.get("last_action")
    if action is None:
        return torch.zeros(env.num_envs, 16, device=env.device)
    return action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_fingertip_body_ids(robot) -> list:
    """
    Return body indices for the 4 Allegro fingertips.
    Allegro body names: link_3.0_tip, link_7.0_tip, link_11.0_tip, link_15.0_tip
    """
    tip_names = [
        "link_3.0_tip",   # index
        "link_7.0_tip",   # middle
        "link_11.0_tip",  # ring
        "link_15.0_tip",  # thumb
    ]
    ids = [robot.find_bodies(name)[0][0] for name in tip_names]
    return ids


def _transform_points_to_local_frame(
    points_world: torch.Tensor,   # (N, K, 3)
    frame_pos: torch.Tensor,      # (N, 3)
    frame_quat: torch.Tensor,     # (N, 4)  w, x, y, z
) -> torch.Tensor:
    """Transform world-frame points into the local frame defined by pos/quat."""
    # Translate
    p = points_world - frame_pos.unsqueeze(1)        # (N, K, 3)

    # Rotate by inverse quaternion
    q_inv = _quat_conjugate(frame_quat)              # (N, 4)
    p_local = _quat_rotate_batch(q_inv, p)           # (N, K, 3)
    return p_local


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (= inverse for unit quaternion). q = (w, x, y, z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_rotate_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors v by quaternion q.

    q: (N, 4)  w, x, y, z
    v: (N, K, 3)
    Returns: (N, K, 3)
    """
    w = q[:, 0:1, None]    # (N, 1, 1) – broadcast over K and 3
    xyz = q[:, 1:]         # (N, 3)

    # Expand q for batch × K operation
    xyz_b = xyz.unsqueeze(1)       # (N, 1, 3)
    t = 2.0 * torch.cross(xyz_b.expand_as(v), v, dim=-1)   # (N, K, 3)
    return v + w * t + torch.cross(xyz_b.expand_as(t), t, dim=-1)
