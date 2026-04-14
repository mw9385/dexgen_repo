"""
Reward functions for in-hand object reorientation (delta-based primary).

Primary signal (DexGen-style delta):
  r_t = w_rot × (prev_rot_err - cur_rot_err)
      + w_pos × (prev_pos_err - cur_pos_err)
      + r_finger
      + r_style + r_reg + goal_bonus

Delta form ensures "do nothing" gives 0 reward (no free baseline).
Only progress toward goal gives positive reward.
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


# ── Rotation delta reward: Δrot_err × weight (do-nothing → 0) ──

def rotation_reward(env) -> torch.Tensor:
    """(prev_rot_err - cur_rot_err) — positive when reducing orientation error.
    Zero when stationary. Negative when moving away from goal.
    """
    cur_err = _get_orn_error(env)
    prev_err = env.extras.get("_prev_rot_err")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err
    env.extras["_prev_rot_err"] = cur_err.clone()
    return delta


# ── Distance delta reward: Δpos_err × weight (do-nothing → 0) ──

def distance_reward(env) -> torch.Tensor:
    """(prev_pos_err - cur_pos_err) — positive when reducing position error."""
    cur_err = _get_pos_error(env)
    prev_err = env.extras.get("_prev_pos_err")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err
    env.extras["_prev_pos_err"] = cur_err.clone()
    return delta


# ── Action penalty: -scale × Σ(actions²) ──
# NOTE: action here is raw policy output (pre-EMA, pre-clip).
# EMA smoothing happens in the wrapper before env.step().

def action_penalty(env) -> torch.Tensor:
    """L2 penalty on raw policy output magnitude."""
    action = env.extras.get("current_action")
    if action is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (action ** 2).sum(dim=-1)


# ── Action delta penalty: -scale × Σ(Δactions²) ──

def action_delta_penalty(env) -> torch.Tensor:
    """L2 penalty on action change between consecutive steps."""
    cur = env.extras.get("current_action")
    prev = env.extras.get("last_action")
    if cur is None or prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    return ((cur - prev) ** 2).sum(dim=-1)


# ── Velocity penalty ──

def velocity_penalty(env) -> torch.Tensor:
    """Joint velocity penalty."""
    hand = env.scene["robot"]
    vel_normalised = hand.data.joint_vel / 4.0
    return (vel_normalised ** 2).sum(dim=-1)


# ── Finger joint matching: (prev_err - cur_err) × weight (delta form) ──

def finger_match_reward(env) -> torch.Tensor:
    """Δ joint-pos distance from current to target grasp.
    Positive when hand shape approaches goal grasp."""
    target_q = env.extras.get("target_joint_pos")
    if target_q is None:
        return torch.zeros(env.num_envs, device=env.device)
    cur_q = env.scene["robot"].data.joint_pos
    n = min(cur_q.shape[-1], target_q.shape[-1])
    cur_err = torch.norm(cur_q[:, :n] - target_q[:, :n], dim=-1)
    prev_err = env.extras.get("_prev_finger_err")
    if prev_err is None:
        prev_err = cur_err.clone()
    delta = prev_err - cur_err
    env.extras["_prev_finger_err"] = cur_err.clone()
    return delta


# ── Style: fingertip velocity penalty ──

def fingertip_velocity_penalty(env) -> torch.Tensor:
    """Σ ||v_tip||² for all fingertips."""
    from .sim_utils import get_fingertip_body_ids_from_env
    robot = env.scene["robot"]
    try:
        ft_ids = get_fingertip_body_ids_from_env(robot, env)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)
    v_tip = robot.data.body_lin_vel_w[:, ft_ids, :]
    return (v_tip ** 2).sum(dim=(-2, -1))


# ── Regularization: torque penalty ──

def torque_penalty(env) -> torch.Tensor:
    """Σ τ²."""
    robot = env.scene["robot"]
    tau = robot.data.applied_torque
    return (tau ** 2).sum(dim=-1)


# ── Regularization: work penalty ──

def work_penalty(env) -> torch.Tensor:
    """Σ |τ × ω|."""
    robot = env.scene["robot"]
    tau = robot.data.applied_torque
    omega = robot.data.joint_vel
    return (tau * omega).abs().sum(dim=-1)


# ── Sparse goal bonus (rotation + position) ──

def goal_bonus(env, rot_thresh: float = 0.4, pos_thresh: float = 0.05,
               bonus: float = 250.0) -> torch.Tensor:
    """Sparse bonus when BOTH rotation and position errors are below threshold."""
    rot_dist = _get_orn_error(env)
    pos_dist = _get_pos_error(env)
    reached = (rot_dist < rot_thresh) & (pos_dist < pos_thresh)
    return torch.where(reached, bonus, torch.zeros_like(rot_dist))
