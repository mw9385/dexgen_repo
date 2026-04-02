"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

Follows the DexterityGen (arXiv:2502.04307) reward structure:

  r = w_goal * r_goal + w_style * r_style + w_reg * r_reg

  GOAL (Eq. 5-7):
    r_goal = exp(-α_pos * ||p_obj - p_target||² - α_orn * d(R, R_target))
           - α_hand * ||q - q_target||²
           + α_bonus * 1(goal achieved)

  STYLE (Eq. 9):
    r_style = -α_i * ||ẋ_tip||

  REGULARIZATION (Eq. 8):
    r_reg = -α_work * |q̇ᵀ||τ| - α_action * ||a||² - α_tau * ||τ||²
"""

from __future__ import annotations
import torch

from .observations import (
    _get_fingertip_body_ids,
    _get_num_fingers,
)


# ═══════════════════════════════════════════════════════════
# 1. GOAL REWARD  (Eq. 5-7)
# ═══════════════════════════════════════════════════════════

def goal_reward(
    env,
    alpha_pos: float = 20.0,
    alpha_orn: float = 10.0,
    alpha_hand: float = 1.0,
    alpha_bonus: float = 5.0,
    bonus_threshold: float = 0.05,
) -> torch.Tensor:
    """
    DexterityGen Eq. 5-7:
      exp(-α_pos * ||p_obj - p_target||² - α_orn * d(R, R_target))
      - α_hand * ||q - q_target||²
      + α_bonus * 1(goal achieved)
    """
    from isaaclab.utils.math import quat_apply_inverse

    robot = env.scene["robot"]
    obj = env.scene["object"]

    # --- Object pose term (Eq. 5) ---
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    # Current object pose in hand frame
    cur_obj_pos_hand = quat_apply_inverse(root_quat, obj_pos_w - root_pos)
    cur_obj_quat_hand = _quat_multiply(_quat_conjugate(root_quat), obj_quat_w)

    target_pos = env.extras.get("target_object_pos_hand")
    target_quat = env.extras.get("target_object_quat_hand")

    if target_pos is None or target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)

    pos_err_sq = ((cur_obj_pos_hand - target_pos) ** 2).sum(dim=-1)
    dot = (cur_obj_quat_hand * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    orn_err = 2.0 * torch.acos(dot)  # angular distance in radians

    r_pose = torch.exp(-alpha_pos * pos_err_sq - alpha_orn * orn_err)

    # --- Joint tracking term (Eq. 6) ---
    target_joints = env.extras.get("target_joint_angles")
    if target_joints is not None:
        cur_joints = robot.data.joint_pos
        joint_err_sq = ((cur_joints - target_joints) ** 2).sum(dim=-1)
        r_joint = -alpha_hand * joint_err_sq
    else:
        r_joint = torch.zeros(env.num_envs, device=env.device)

    # --- Goal bonus (Eq. 7) ---
    pos_err = torch.sqrt(pos_err_sq + 1e-8)
    goal_achieved = (pos_err < bonus_threshold) & (orn_err < 0.3)  # ~17 deg
    r_bonus = alpha_bonus * goal_achieved.float()

    return r_pose + r_joint + r_bonus


# ═══════════════════════════════════════════════════════════
# 2. STYLE REWARD  (Eq. 9)  →  [-inf, 0]
# ═══════════════════════════════════════════════════════════

def fingertip_velocity_penalty(env, scale: float = 1.0) -> torch.Tensor:
    """
    DexterityGen Eq. 9: -α_i * ||ẋ_tip||
    """
    robot = env.scene["robot"]
    nf = _get_num_fingers(env)
    tip_ids = _get_fingertip_body_ids(robot, env)[:nf]
    tip_vels = robot.data.body_lin_vel_w[:, tip_ids, :]
    speed = torch.norm(tip_vels, dim=-1).sum(dim=-1)
    return -scale * speed


# ═══════════════════════════════════════════════════════════
# 3. REGULARIZATION  (Eq. 8)
# ═══════════════════════════════════════════════════════════

def work_penalty(env, scale: float = 1.0) -> torch.Tensor:
    """
    DexterityGen Eq. 8: -α_work * |q̇ᵀ| * |τ|
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    velocities = robot.data.joint_vel
    work = (torques.abs() * velocities.abs()).sum(dim=-1)
    return -scale * work


def action_penalty(env, scale: float = 1.0) -> torch.Tensor:
    """
    DexterityGen Eq. 8: -α_action * ||a||²
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -scale * (current_act ** 2).sum(dim=-1)


def torque_penalty(env, scale: float = 1.0) -> torch.Tensor:
    """
    DexterityGen Eq. 8: -α_tau * ||τ||²
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    return -scale * (torques ** 2).sum(dim=-1)


# ═══════════════════════════════════════════════════════════
# Quaternion helpers
# ═══════════════════════════════════════════════════════════

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
