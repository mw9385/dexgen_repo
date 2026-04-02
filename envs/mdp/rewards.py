"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

Follows the DexterityGen (arXiv:2502.04307) reward structure:

  r = r_goal + r_style + r_reg

  GOAL (positive, [0, 1]):
    fingertip_tracking_reward    mean_i exp(-alpha * ||tip_i - goal_i||)

  STYLE (negative, [-1, 0]):
    fingertip_velocity_penalty   -tanh(scale * mean_speed)

  REGULARIZATION (negative, [-1, 0]):
    action_scale_penalty         -tanh(scale * ||action||)
    applied_torque_penalty       -tanh(scale * mean(|torque|))
    mechanical_work_penalty      -tanh(scale * mean(|torque * vel|))

All functions output values in [-1, 1].
Relative importance is controlled solely by the weight in the RewTerm config.
"""

from __future__ import annotations
import torch

from .observations import (
    fingertip_positions_in_object_frame,
    target_fingertip_positions,
    _get_fingertip_body_ids,
    _get_num_fingers,
    _get_fingertip_contact_forces_world,
)

# ═══════════════════════════════════════════════════════════
# 1. GOAL REWARD  →  [0, 1]
# ═══════════════════════════════════════════════════════════

def fingertip_tracking_reward(
    env, alpha: float = 20.0, contact_force_thresh: float = 0.5,
) -> torch.Tensor:
    """
    mean_i exp(-alpha * ||tip_i - goal_i||)

    Each finger independently incentivised to reach its goal.
    When NO fingertip has contact with the object, reward is forced to 0
    so the policy cannot earn reward by letting the object drift away.

    Returns: (N,) in [0, 1]
    """
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target  = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist    = torch.norm(current - target, dim=-1)
    reward  = torch.exp(-alpha * dist).mean(dim=-1)

    # Zero reward when all fingertips have lost contact with the object
    forces = _get_fingertip_contact_forces_world(env)  # (N, F, 3)
    has_any_contact = (torch.norm(forces, dim=-1) > contact_force_thresh).any(dim=-1)
    reward = reward * has_any_contact.float()

    return reward


def object_pose_reward(env, pos_alpha: float = 10.0, rot_alpha: float = 5.0) -> torch.Tensor:
    """
    Reward for matching the target object pose in the hand frame.

      r = 0.5 * exp(-pos_alpha * ||pos_err||) + 0.5 * exp(-rot_alpha * rot_err)

    pos_err: Euclidean distance between current and target object position (hand frame)
    rot_err: angular difference between current and target object orientation

    Returns: (N,) in (0, 1]
    """
    from isaaclab.utils.math import quat_apply_inverse

    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Current object pose in hand (root) frame
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    cur_obj_pos_hand = quat_apply_inverse(root_quat, obj_pos_w - root_pos)

    # Target object pose in hand frame
    target_pos = env.extras.get("target_object_pos_hand")
    target_quat = env.extras.get("target_object_quat_hand")

    if target_pos is None or target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Position error
    pos_err = torch.norm(cur_obj_pos_hand - target_pos, dim=-1)
    pos_reward = torch.exp(-pos_alpha * pos_err)

    # Rotation error (angular distance)
    cur_obj_quat_hand = _quat_multiply(_quat_conjugate(root_quat), obj_quat_w)
    dot = (cur_obj_quat_hand * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    rot_err = 2.0 * torch.acos(dot)
    rot_reward = torch.exp(-rot_alpha * rot_err)

    return 0.5 * pos_reward + 0.5 * rot_reward


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion (w,x,y,z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ═══════════════════════════════════════════════════════════
# 2. STYLE REWARD  →  [-1, 0]
# ═══════════════════════════════════════════════════════════

def fingertip_velocity_penalty(env, scale: float = 5.0) -> torch.Tensor:
    """
    -tanh(scale * mean_fingertip_speed)

    Penalizes fast fingertip motion. Varying this weight across
    training runs generates diverse manipulation styles (fast/slow)
    for dataset collection (DexterityGen §3.2).

    Returns: (N,) in [-1, 0]
    """
    robot   = env.scene["robot"]
    nf      = _get_num_fingers(env)
    tip_ids = _get_fingertip_body_ids(robot, env)[:nf]
    tip_vels = robot.data.body_lin_vel_w[:, tip_ids, :]
    mean_speed = torch.norm(tip_vels, dim=-1).mean(dim=-1)
    return -torch.tanh(scale * mean_speed)


# ═══════════════════════════════════════════════════════════
# 3. REGULARIZATION  →  [-1, 0]
# ═══════════════════════════════════════════════════════════

def action_scale_penalty(env, scale: float = 0.5) -> torch.Tensor:
    """
    -tanh(scale * ||action||)
    Returns: (N,) in [-1, 0]
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.tanh(scale * torch.norm(current_act, dim=-1))


def applied_torque_penalty(env, scale: float = 0.1) -> torch.Tensor:
    """
    -tanh(scale * mean(|torque|))
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    return -torch.tanh(scale * torques.abs().mean(dim=-1))


def mechanical_work_penalty(env, scale: float = 0.5) -> torch.Tensor:
    """
    -tanh(scale * mean(|torque * velocity|))
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    velocities = robot.data.joint_vel
    power = (torques * velocities).abs().mean(dim=-1)
    return -torch.tanh(scale * power)
