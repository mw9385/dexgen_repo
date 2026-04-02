"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

DexterityGen (arXiv:2502.04307) reward structure (Eq. 4-9):
  r = w_goal * r_goal + w_style * r_style + w_reg * r_reg

All individual reward functions are normalized to [-1, 1] or [0, 1].
Relative importance is controlled solely by the weight in the RewTerm config.
"""

from __future__ import annotations
import torch

from .observations import (
    _get_fingertip_body_ids,
    _get_num_fingers,
)

# ═══════════════════════════════════════════════════════════
# Quaternion helpers
# ═══════════════════════════════════════════════════════════

def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def _obj_pose_in_hand_frame(env):
    """Return (pos_hand, quat_hand) of the object in robot root frame."""
    from isaaclab.utils.math import quat_apply_inverse
    robot = env.scene["robot"]
    obj = env.scene["object"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    pos_hand = quat_apply_inverse(root_quat, obj.data.root_pos_w - root_pos)
    quat_hand = _quat_multiply(_quat_conjugate(root_quat), obj.data.root_quat_w)
    return pos_hand, quat_hand


# ═══════════════════════════════════════════════════════════
# 1. GOAL REWARDS  (Eq. 5-7) — each [0, 1]
# ═══════════════════════════════════════════════════════════

def object_position_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """
    Eq. 5 (position part): exp(-α * ||p_obj - p_target||²)
    Returns: (N,) in (0, 1]
    """
    cur_pos, _ = _obj_pose_in_hand_frame(env)
    target_pos = env.extras.get("target_object_pos_hand")
    if target_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    pos_err_sq = ((cur_pos - target_pos) ** 2).sum(dim=-1)
    return torch.exp(-alpha * pos_err_sq)


def object_orientation_reward(env, alpha: float = 10.0) -> torch.Tensor:
    """
    Eq. 5 (orientation part): exp(-α * d(R_obj, R_target))
    Returns: (N,) in (0, 1]
    """
    _, cur_quat = _obj_pose_in_hand_frame(env)
    target_quat = env.extras.get("target_object_quat_hand")
    if target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)
    dot = (cur_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    orn_err = 2.0 * torch.acos(dot)
    return torch.exp(-alpha * orn_err)


def joint_tracking_reward(env, alpha: float = 2.0) -> torch.Tensor:
    """
    Eq. 6: -α_hand * ||q - q_target||², normalized via tanh.
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    target_joints = env.extras.get("target_joint_angles")
    if target_joints is None:
        return torch.zeros(env.num_envs, device=env.device)
    cur_joints = robot.data.joint_pos
    joint_err = torch.norm(cur_joints - target_joints, dim=-1)
    return -torch.tanh(alpha * joint_err)


def goal_bonus(env, pos_thresh: float = 0.02, rot_thresh: float = 0.1) -> torch.Tensor:
    """
    Eq. 7: 1(goal achieved). pos < 2cm, rot < 0.1 rad (~5.7°).
    Returns: (N,) in {0, 1}
    """
    cur_pos, cur_quat = _obj_pose_in_hand_frame(env)
    target_pos = env.extras.get("target_object_pos_hand")
    target_quat = env.extras.get("target_object_quat_hand")
    if target_pos is None or target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)
    pos_err = torch.norm(cur_pos - target_pos, dim=-1)
    dot = (cur_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    orn_err = 2.0 * torch.acos(dot)
    return ((pos_err < pos_thresh) & (orn_err < rot_thresh)).float()


# ═══════════════════════════════════════════════════════════
# 2. STYLE REWARD  (Eq. 9)  →  [-1, 0]
# ═══════════════════════════════════════════════════════════

def fingertip_velocity_penalty(env, alpha: float = 1.0) -> torch.Tensor:
    """
    Eq. 9: -α * ||ẋ_tip||, normalized via tanh.
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    nf = _get_num_fingers(env)
    tip_ids = _get_fingertip_body_ids(robot, env)[:nf]
    tip_vels = robot.data.body_lin_vel_w[:, tip_ids, :]
    speed = torch.norm(tip_vels, dim=-1).mean(dim=-1)
    return -torch.tanh(alpha * speed)


# ═══════════════════════════════════════════════════════════
# 3. REGULARIZATION  (Eq. 8)  →  [-1, 0]
# ═══════════════════════════════════════════════════════════

def work_penalty(env, alpha: float = 0.01) -> torch.Tensor:
    """
    Eq. 8: -α_work * |q̇ᵀ| * |τ|, normalized via tanh.
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    velocities = robot.data.joint_vel
    work = (torques.abs() * velocities.abs()).sum(dim=-1)
    return -torch.tanh(alpha * work)


def action_penalty(env, alpha: float = 0.5) -> torch.Tensor:
    """
    Eq. 8: -α_action * ||a||², normalized via tanh.
    Returns: (N,) in [-1, 0]
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    act_sq = (current_act ** 2).sum(dim=-1)
    return -torch.tanh(alpha * act_sq)


def torque_penalty(env, alpha: float = 0.005) -> torch.Tensor:
    """
    Eq. 8: -α_tau * ||τ||², normalized via tanh.
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    torque_sq = (torques ** 2).sum(dim=-1)
    return -torch.tanh(alpha * torque_sq)


# ═══════════════════════════════════════════════════════════
# 4. TERMINATION PENALTY  →  [-1, 0]
# ═══════════════════════════════════════════════════════════

def termination_penalty(env) -> torch.Tensor:
    """
    Penalty when object is dropped or leaves hand.
    Scaled by remaining episode fraction so early drops cost more.
    Returns: (N,) in [-1, 0]
    """
    from . import events as mdp_events
    dropped = mdp_events.object_dropped(env, min_height=0.2)
    left = mdp_events.object_left_hand(env, max_dist=0.20)
    terminated = dropped | left
    remaining = (env.max_episode_length - env.episode_length_buf).float() / env.max_episode_length
    return -terminated.float() * remaining
