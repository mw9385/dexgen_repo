"""
Reward functions for in-hand reorientation (OpenAI style).

Core idea: reward = (d_t - d_{t+1}) — reward the reduction in error.
  - Object moves closer to goal → positive reward
  - Object stays still → zero reward
  - Object moves away from goal → negative reward

Plus sparse bonuses/penalties for goal achievement and object drop.
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


def _get_orn_error(env) -> torch.Tensor:
    """Compute current orientation error (rad). Returns: (N,)"""
    _, cur_quat = _obj_pose_in_hand_frame(env)
    target_quat = env.extras.get("target_object_quat_hand")
    if target_quat is None:
        return torch.zeros(env.num_envs, device=env.device)
    dot = (cur_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _get_pos_error(env) -> torch.Tensor:
    """Compute current position error (m). Returns: (N,)"""
    cur_pos, _ = _obj_pose_in_hand_frame(env)
    target_pos = env.extras.get("target_object_pos_hand")
    if target_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.norm(cur_pos - target_pos, dim=-1)


# ═══════════════════════════════════════════════════════════
# 1. GOAL REWARDS — delta (d_t - d_{t+1})
# ═══════════════════════════════════════════════════════════

def orientation_delta_reward(env) -> torch.Tensor:
    """
    (prev_orn_err - cur_orn_err): positive when getting closer.
    Returns: (N,) — unbounded, typically in [-0.1, 0.1] per step.
    """
    cur_err = _get_orn_error(env)
    prev_err = env.extras.get("_prev_orn_error")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err  # positive = improvement
    env.extras["_prev_orn_error"] = cur_err.clone()
    return delta


def position_delta_reward(env) -> torch.Tensor:
    """
    (prev_pos_err - cur_pos_err): positive when getting closer.
    Returns: (N,) — unbounded, typically small.
    """
    cur_err = _get_pos_error(env)
    prev_err = env.extras.get("_prev_pos_error")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err
    env.extras["_prev_pos_error"] = cur_err.clone()
    return delta


def goal_bonus(env, pos_thresh: float = 0.02, rot_thresh: float = 0.1) -> torch.Tensor:
    """
    Sparse bonus: 1 if goal achieved (pos < 2cm AND rot < 0.1rad).
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
# 2. PENALTIES
# ═══════════════════════════════════════════════════════════

def drop_penalty(env, min_height: float = 0.2, max_dist: float = 0.20) -> torch.Tensor:
    """
    -1 when object is dropped or leaves hand. 0 otherwise.
    Returns: (N,) in {-1, 0}
    """
    from . import events as mdp_events
    dropped = mdp_events.object_dropped(env, min_height=min_height)
    left = mdp_events.object_left_hand(env, max_dist=max_dist)
    return -(dropped | left).float()


def action_penalty(env) -> torch.Tensor:
    """
    -||a||². Penalizes large actions.
    Returns: (N,) in (-inf, 0]
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -(current_act ** 2).sum(dim=-1)
