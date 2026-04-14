"""
Reward functions for in-hand object reorientation (DeXtreme exact form).

DeXtreme (Handa et al., ICRA 2023) — AllegroHandDextremeManualDR:
  dist_rew         = goal_dist × dist_reward_scale    (-10.0)
  rot_rew          = 1.0 / (|rot_dist| + rot_eps) × rot_reward_scale  (+1.0)
  action_penalty   = Σ(a²) × action_penalty_scale     (-0.0001)
  action_delta     = Σ(Δa²) × action_delta_scale      (-0.01)
  velocity_penalty = Σ((v/4)²) × velocity_coef        (-0.05)
  reach_goal_bonus = +250  when rot_dist < success_tolerance
  fall_penalty     = 0.0
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


# ── DeXtreme dist_rew: pos_err (multiply by -10 weight) ──

def distance_reward(env) -> torch.Tensor:
    return _get_pos_error(env)


# ── DeXtreme rot_rew: 1/(|rot_err| + rot_eps) ──

def rotation_reward(env, rot_eps: float = 0.1) -> torch.Tensor:
    return 1.0 / (_get_orn_error(env).abs() + rot_eps)


# ── DeXtreme action_penalty: Σ(actions²) ──

def action_penalty(env) -> torch.Tensor:
    action = env.extras.get("current_action")
    if action is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (action ** 2).sum(dim=-1)


# ── DeXtreme action_delta_penalty: Σ((cur-prev)²) ──

def action_delta_penalty(env) -> torch.Tensor:
    cur = env.extras.get("current_action")
    prev = env.extras.get("last_action")
    if cur is None or prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    return ((cur - prev) ** 2).sum(dim=-1)


# ── DeXtreme velocity_penalty: Σ((v/(vmax-vtol))²) ──

def velocity_penalty(env) -> torch.Tensor:
    hand = env.scene["robot"]
    # DeXtreme uses max_velocity=5, vel_tolerance=1 → divisor = 4
    vel_normalised = hand.data.joint_vel / 4.0
    return (vel_normalised ** 2).sum(dim=-1)


# ── DeXtreme reach_goal_bonus: +250 when rot_dist < success_tolerance ──

def goal_bonus(env, rot_thresh: float = 0.4, bonus: float = 250.0) -> torch.Tensor:
    rot_dist = _get_orn_error(env)
    reached = rot_dist < rot_thresh
    return torch.where(reached, bonus, torch.zeros_like(rot_dist))
