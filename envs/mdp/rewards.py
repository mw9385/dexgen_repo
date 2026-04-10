"""
Reward functions for in-hand object reorientation.

OpenAI "Learning Dexterous In-Hand Manipulation" (2018):
  r_t = d_t - d_{t+1}   (rotation distance reduction)
  + 5   when rot_dist < 0.4 rad  (goal achieved)
  - 20  when object_dropped (object too far from palm; same predicate as termination)
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


# ── r_t = d_t - d_{t+1} ──

def orientation_delta_reward(env) -> torch.Tensor:
    """Positive when rotation error decreases."""
    cur_err = _get_orn_error(env)
    prev_err = env.extras.get("_prev_orn_error")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err
    env.extras["_prev_orn_error"] = cur_err.clone()
    return delta


# ── +5 goal bonus ──

def goal_bonus(env, rot_thresh: float = 0.4, bonus: float = 5.0) -> torch.Tensor:
    """Sparse bonus when goal achieved."""
    rot_dist = _get_orn_error(env)
    return torch.where(rot_dist < rot_thresh, bonus, torch.zeros_like(rot_dist))


# ── -20 drop penalty ──

def drop_penalty(env, penalty: float = -20.0) -> torch.Tensor:
    """Sparse penalty when ``object_dropped`` (palm–object distance exceeds threshold)."""
    from . import events as mdp_events
    dropped = mdp_events.object_dropped(env)
    return torch.where(dropped, penalty, torch.zeros(env.num_envs, device=env.device))
