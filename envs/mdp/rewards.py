"""
Reward functions for in-hand object reorientation (DeXtreme-aligned).

DeXtreme (Handa et al., ICRA 2023):
  r_t = dist_rew + rot_rew + action_penalty + action_delta_penalty
        + velocity_penalty + reach_goal_bonus

  - No explicit drop penalty (episode termination = opportunity cost of +250)
  - No contact gating
  - No position threshold for goal success (rotation only)
"""

from __future__ import annotations
import torch

from .math_utils import quat_conjugate as _quat_conjugate
from .math_utils import quat_multiply as _quat_multiply


def _obj_pose_in_hand_frame(env):
    from isaaclab.utils.math import quat_apply_inverse
    robot = env.scene["robot"]
    obj = env.scene["object"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    pos_hand = quat_apply_inverse(root_quat, obj.data.root_pos_w - root_pos)
    quat_hand = _quat_multiply(_quat_conjugate(root_quat), obj.data.root_quat_w)
    return pos_hand, quat_hand


def _rotation_distance(q1, q2):
    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _get_orn_error(env):
    _, cur_quat = _obj_pose_in_hand_frame(env)
    target_quat = env.extras.get("target_object_quat_hand")
    if target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)
    return _rotation_distance(cur_quat, target_quat)


def _get_pos_error(env):
    cur_pos, _ = _obj_pose_in_hand_frame(env)
    target_pos = env.extras.get("target_object_pos_hand")
    if target_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.norm(cur_pos - target_pos, dim=-1)


# ── Distance reward: goal_dist × weight (dense, negative) ──

def distance_reward(env) -> torch.Tensor:
    """Penalises L2 distance between object and target position. (DeXtreme dist_rew)"""
    return _get_pos_error(env)


# ── Rotation reward: 1/(|rot_dist| + eps) (dense, positive) ──

def rotation_reward(env, rot_eps: float = 0.1) -> torch.Tensor:
    """Inverse rotation distance — reward increases as object aligns with goal.
    (DeXtreme rot_rew)"""
    rot_dist = _get_orn_error(env)
    return 1.0 / (rot_dist.abs() + rot_eps)


# ── Action penalty: -scale × Σ(actions²) ──

def action_penalty(env) -> torch.Tensor:
    """L2 penalty on action magnitude. (DeXtreme action_penalty)"""
    action = env.extras.get("current_action")
    if action is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (action ** 2).sum(dim=-1)


# ── Action delta penalty: -scale × Σ(Δactions²) ──

def action_delta_penalty(env) -> torch.Tensor:
    """L2 penalty on action change between consecutive steps. (DeXtreme action_delta_penalty)"""
    cur = env.extras.get("current_action")
    prev = env.extras.get("last_action")
    if cur is None or prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    return ((cur - prev) ** 2).sum(dim=-1)


# ── Velocity penalty: -0.05 × Σ((dof_vel / 4.0)²) ──

def velocity_penalty(env) -> torch.Tensor:
    """Joint velocity penalty. (DeXtreme velocity_penalty)"""
    hand = env.scene["robot"]
    vel_normalised = hand.data.joint_vel / 4.0  # max_vel(5) - tolerance(1)
    return (vel_normalised ** 2).sum(dim=-1)


# ── +250 goal bonus (rotation only, DeXtreme-style) ──

def goal_bonus(env, rot_thresh: float = 0.4,
               bonus: float = 250.0) -> torch.Tensor:
    """Sparse bonus when rotation error is below threshold.
    No position threshold (DeXtreme-aligned)."""
    rot_dist = _get_orn_error(env)
    reached = rot_dist < rot_thresh
    return torch.where(reached, bonus, torch.zeros_like(rot_dist))
