"""
Reward functions for in-hand object reorientation.

Based on OpenAI "Learning Dexterous In-Hand Manipulation" (2018):
  r_t = d_t - d_{t+1}                     (rotation distance reduction)
  + reach_goal_bonus  when rot_dist < success_tolerance
  + fall_penalty      when object drops

Also includes IsaacGymEnvs ShadowHand reward components:
  rot_rew   = 1/(|rot_dist| + eps) * scale    (inverse rotation distance)
  dist_rew  = -pos_dist * scale                (position distance penalty)
  action_penalty = -sum(a²) * scale            (action regularization)
"""

from __future__ import annotations
import torch

from .math_utils import quat_conjugate as _quat_conjugate
from .math_utils import quat_multiply as _quat_multiply


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

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


def _rotation_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between two quaternions. Returns: (N,) in [0, pi]."""
    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _get_orn_error(env) -> torch.Tensor:
    """Current orientation error (rad). Returns: (N,)"""
    _, cur_quat = _obj_pose_in_hand_frame(env)
    target_quat = env.extras.get("target_object_quat_hand")
    if target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)
    return _rotation_distance(cur_quat, target_quat)


def _get_pos_error(env) -> torch.Tensor:
    """Current position error (m). Returns: (N,)"""
    cur_pos, _ = _obj_pose_in_hand_frame(env)
    target_pos = env.extras.get("target_object_pos_hand")
    if target_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.norm(cur_pos - target_pos, dim=-1)


# ---------------------------------------------------------------------------
# OpenAI-style rewards
# ---------------------------------------------------------------------------

def orientation_delta_reward(env) -> torch.Tensor:
    """
    OpenAI: r_t = d_t - d_{t+1}
    Positive when rotation error decreases.
    """
    cur_err = _get_orn_error(env)
    prev_err = env.extras.get("_prev_orn_error")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err
    env.extras["_prev_orn_error"] = cur_err.clone()
    return delta


def rotation_distance_reward(
    env, rot_eps: float = 0.1, scale: float = 1.0,
) -> torch.Tensor:
    """
    IsaacGymEnvs: rot_rew = 1/(|rot_dist| + eps) * scale
    Higher reward when closer to goal orientation.
    """
    rot_dist = _get_orn_error(env)
    return (1.0 / (rot_dist.abs() + rot_eps)) * scale


def position_distance_reward(env, scale: float = 1.0) -> torch.Tensor:
    """
    IsaacGymEnvs: dist_rew = -goal_dist * scale
    Penalizes object being far from target position.
    """
    return -_get_pos_error(env) * scale


def goal_bonus(
    env, rot_thresh: float = 0.4, bonus: float = 5.0,
) -> torch.Tensor:
    """
    OpenAI: +5 when rotation goal achieved (rot_dist < 0.4 rad).
    Returns: (N,) — bonus or 0.
    """
    rot_dist = _get_orn_error(env)
    return torch.where(rot_dist < rot_thresh, bonus, torch.zeros_like(rot_dist))


def drop_penalty(
    env, min_height: float = 0.2, max_dist: float = 0.20,
    penalty: float = -20.0,
) -> torch.Tensor:
    """
    OpenAI: -20 when object is dropped.
    Returns: (N,) — penalty or 0.
    """
    from . import events as mdp_events
    dropped = mdp_events.object_dropped(env, min_height=min_height)
    left = mdp_events.object_left_hand(env, max_dist=max_dist)
    failed = dropped | left
    return torch.where(failed, penalty, torch.zeros(env.num_envs, device=env.device))


def action_penalty(env, scale: float = 1.0) -> torch.Tensor:
    """
    IsaacGymEnvs: -sum(a²) * scale
    Penalizes large actions for smooth control.
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -(current_act ** 2).sum(dim=-1) * scale


# ---------------------------------------------------------------------------
# Position delta (optional, for curriculum)
# ---------------------------------------------------------------------------

def position_delta_reward(env) -> torch.Tensor:
    """(prev_pos_err - cur_pos_err): positive when getting closer."""
    cur_err = _get_pos_error(env)
    prev_err = env.extras.get("_prev_pos_error")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err
    env.extras["_prev_pos_error"] = cur_err.clone()
    return delta
