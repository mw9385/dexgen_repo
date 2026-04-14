"""
Reward functions for in-hand object reorientation (DexterityGen-aligned).

DexterityGen (Yin et al., 2025) — AnyGrasp-to-AnyGrasp primitive:
  r_t = r_goal + r_style + r_reg

  r_goal:  rotation, distance, finger_match, goal_bonus
  r_style: fingertip_velocity penalty (manipulation diversity)
  r_reg:   action, action_delta, torque, work, velocity penalties
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


# ── Distance reward: -pos_err × weight (DeXtreme linear negative form) ──

def distance_reward(env) -> torch.Tensor:
    """L2 distance between object and target position.
    DeXtreme form: multiplied by negative weight → penalises distance.
    At pos_err=0 the reward is 0 (no free baseline)."""
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


# ── Finger joint matching (DexGen): 1/(||q_cur - q_target|| + eps) ──

def finger_match_reward(env, finger_eps: float = 0.5) -> torch.Tensor:
    """Inverse joint-pos distance from current to target grasp.
    DexGen r_goal component — encourages adopting the goal hand shape."""
    target_q = env.extras.get("target_joint_pos")
    if target_q is None:
        return torch.zeros(env.num_envs, device=env.device)
    cur_q = env.scene["robot"].data.joint_pos
    if cur_q.shape[-1] != target_q.shape[-1]:
        cur_q = cur_q[:, -target_q.shape[-1]:]
    err = torch.norm(cur_q - target_q, dim=-1)
    return 1.0 / (err + finger_eps)


# ── Style: fingertip velocity penalty (DexGen r_style) ──

def fingertip_velocity_penalty(env) -> torch.Tensor:
    """Σ ||v_tip||² for all fingertips. Negative weight encourages slower / diverse motion."""
    from .sim_utils import get_fingertip_body_ids_from_env
    robot = env.scene["robot"]
    try:
        ft_ids = get_fingertip_body_ids_from_env(robot, env)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)
    v_tip = robot.data.body_lin_vel_w[:, ft_ids, :]   # (N, F, 3)
    return (v_tip ** 2).sum(dim=(-2, -1))


# ── Regularization: torque penalty (DexGen r_reg) ──

def torque_penalty(env) -> torch.Tensor:
    """Σ τ² — penalises large applied torques."""
    robot = env.scene["robot"]
    tau = robot.data.applied_torque
    return (tau ** 2).sum(dim=-1)


# ── Regularization: work penalty (DexGen r_reg) ──

def work_penalty(env) -> torch.Tensor:
    """Σ |τ × ω| — penalises mechanical work (energy use)."""
    robot = env.scene["robot"]
    tau = robot.data.applied_torque
    omega = robot.data.joint_vel
    return (tau * omega).abs().sum(dim=-1)


# ── +250 goal bonus (rotation + position) ──

def goal_bonus(env, rot_thresh: float = 0.4, pos_thresh: float = 0.05,
               bonus: float = 250.0) -> torch.Tensor:
    """Sparse bonus when BOTH rotation and position errors are below threshold."""
    rot_dist = _get_orn_error(env)
    pos_dist = _get_pos_error(env)
    reached = (rot_dist < rot_thresh) & (pos_dist < pos_thresh)
    return torch.where(reached, bonus, torch.zeros_like(rot_dist))
