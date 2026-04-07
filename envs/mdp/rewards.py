"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

All reward functions return values in [0, 1].
Goal rewards are normalized so that the initial error gives ~0 reward
and reaching the goal gives ~1. This prevents free reward from standing still.
"""

from __future__ import annotations
import torch

from .math_utils import quat_conjugate as _quat_conjugate
from .math_utils import quat_multiply as _quat_multiply


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
# 1. GOAL REWARDS — [0, 1], normalized so initial state ≈ 0
# ═══════════════════════════════════════════════════════════

def object_position_reward(env, alpha: float = 40.0, max_err: float = 0.02) -> torch.Tensor:
    """
    Normalized position reward. Returns 0 at max_err, 1 at zero error.
    reward = clamp((1 - pos_err / max_err), 0, 1)
    Returns: (N,) in [0, 1]
    """
    cur_pos, _ = _obj_pose_in_hand_frame(env)
    target_pos = env.extras.get("target_object_pos_hand")
    if target_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    pos_err = torch.norm(cur_pos - target_pos, dim=-1)
    return (1.0 - pos_err / max_err).clamp(0.0, 1.0)


def object_orientation_reward(env, alpha: float = 10.0, max_err: float = 0.5) -> torch.Tensor:
    """
    Normalized orientation reward. Returns 0 at max_err, 1 at zero error.
    reward = clamp((1 - orn_err / max_err), 0, 1)
    Returns: (N,) in [0, 1]
    """
    _, cur_quat = _obj_pose_in_hand_frame(env)
    target_quat = env.extras.get("target_object_quat_hand")
    if target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)
    dot = (cur_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    orn_err = 2.0 * torch.acos(dot)
    return (1.0 - orn_err / max_err).clamp(0.0, 1.0)


def goal_bonus(env, pos_thresh: float = 0.02, rot_thresh: float = 0.1) -> torch.Tensor:
    """
    Binary reward: 1 if goal achieved (pos < 2cm AND rot < 0.1rad).
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
# 2. REGULARIZATION — [-1, 0]
# ═══════════════════════════════════════════════════════════

def work_penalty(env, alpha: float = 0.01) -> torch.Tensor:
    """
    -tanh(α * |torque * vel|). Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    velocities = robot.data.joint_vel
    work = (torques.abs() * velocities.abs()).sum(dim=-1)
    return -torch.tanh(alpha * work)


def action_penalty(env, alpha: float = 0.5) -> torch.Tensor:
    """
    -tanh(α * ||a||²). Returns: (N,) in [-1, 0]
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    act_sq = (current_act ** 2).sum(dim=-1)
    return -torch.tanh(alpha * act_sq)


def torque_penalty(env, alpha: float = 0.005) -> torch.Tensor:
    """
    -tanh(α * ||τ||²). Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    torque_sq = (torques ** 2).sum(dim=-1)
    return -torch.tanh(alpha * torque_sq)
