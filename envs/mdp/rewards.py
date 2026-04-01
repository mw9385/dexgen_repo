"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

All reward functions output values in [-1, 1] (or [0, 1] for positive rewards).
Relative importance is controlled solely by the weight in the RewTerm config.

  GOAL-RELATED REWARDS (positive, [0, 1]):
    object_pose_goal_reward      exp(-alpha * error)
    finger_joint_goal_reward     exp(-alpha * error)
    fingertip_tracking_reward    exp(-alpha * error) averaged over fingers
    grasp_success_reward         fraction of fingertips at goal

  STYLE REWARD (negative, [-1, 0]):
    fingertip_velocity_penalty   -tanh(velocity)

  REGULARIZATION (negative, [-1, 0]):
    action_scale_penalty         -tanh(action_norm)
    applied_torque_penalty       -tanh(torque_norm)
    mechanical_work_penalty      -tanh(power)
    action_rate_penalty          -tanh(jerk)

  SAFETY (negative, {-1, 0} or [-1, 0]):
    fingertip_contact_reward     [0, 1] fraction in contact
    object_velocity_penalty      -tanh(excess_velocity)
    object_drop_penalty          {0, -1} binary
    object_left_hand_penalty     {0, -1} binary
    joint_limit_penalty          [-1, 0] near-limit fraction
"""

from __future__ import annotations
import torch

from .observations import (
    fingertip_positions_in_object_frame,
    target_fingertip_positions,
    _get_fingertip_body_ids,
    _get_num_fingers,
    _get_fingertip_contact_forces_world,
)

# ═══════════════════════════════════════════════════════════
# 1. GOAL-RELATED REWARDS  →  [0, 1]
# ═══════════════════════════════════════════════════════════

def object_pose_goal_reward(
    env,
    alpha_pos: float = 10.0,
    alpha_orn: float = 5.0,
) -> torch.Tensor:
    """
    exp(-alpha_pos * pos_err - alpha_orn * rot_err)
    Returns: (N,) in (0, 1]
    """
    from isaaclab.utils.math import quat_apply_inverse

    goal_pos  = env.extras.get("target_object_pos_hand")
    goal_quat = env.extras.get("target_object_quat_hand")
    if goal_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj   = env.scene["object"]

    rel_pos      = obj.data.root_pos_w - robot.data.root_pos_w
    cur_pos_hand = quat_apply_inverse(robot.data.root_quat_w, rel_pos)
    d_pos = torch.norm(cur_pos_hand - goal_pos, dim=-1)

    if goal_quat is not None:
        qw = robot.data.root_quat_w
        qo = obj.data.root_quat_w
        qw_inv = torch.cat([qw[:, :1], -qw[:, 1:]], dim=-1)
        def _qmul(a, b):
            aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            return torch.stack([
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            ], dim=-1)
        cur_quat_hand = _qmul(qw_inv, qo)
        quat_dot = torch.sum(cur_quat_hand * goal_quat, dim=-1).abs().clamp(0.0, 1.0)
        d_rot = 1.0 - quat_dot
        return torch.exp(-alpha_pos * d_pos - alpha_orn * d_rot)

    return torch.exp(-alpha_pos * d_pos)


def finger_joint_goal_reward(env, alpha_hand: float = 2.0) -> torch.Tensor:
    """
    exp(-alpha * ||q - q*|| / sqrt(num_dof))
    Returns: (N,) in (0, 1]
    """
    goal_joints = env.extras.get("target_joint_angles")
    if goal_joints is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot    = env.scene["robot"]
    q        = robot.data.joint_pos
    num_dof  = q.shape[-1]
    dq_norm  = torch.norm(q - goal_joints, dim=-1) / (num_dof ** 0.5)
    return torch.exp(-alpha_hand * dq_norm)


def fingertip_tracking_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """
    mean_i exp(-alpha * ||tip_i - goal_i||)
    Returns: (N,) in (0, 1]
    """
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target  = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist    = torch.norm(current - target, dim=-1)
    return torch.exp(-alpha * dist).mean(dim=-1)


def grasp_success_reward(
    env,
    threshold: float = 0.02,
    min_fraction: float = 1.0,
) -> torch.Tensor:
    """
    Soft grasp-success: fraction of fingertips within threshold.
    Gated by min_fraction (1.0 = ALL fingertips must be within threshold).
    Returns: (N,) in [0, 1]
    """
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target  = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist    = torch.norm(current - target, dim=-1)

    in_threshold = (dist < threshold).float()
    fraction_in  = in_threshold.mean(dim=-1)

    gate = (fraction_in >= min_fraction).float()
    return gate * fraction_in


# ═══════════════════════════════════════════════════════════
# 2. STYLE REWARD  →  [-1, 0]
# ═══════════════════════════════════════════════════════════

def fingertip_velocity_penalty(env, scale: float = 5.0) -> torch.Tensor:
    """
    -tanh(scale * mean_fingertip_speed)
    Returns: (N,) in [-1, 0]
    """
    robot   = env.scene["robot"]
    nf      = _get_num_fingers(env)
    tip_ids = _get_fingertip_body_ids(robot, env)[:nf]
    tip_vels = robot.data.body_lin_vel_w[:, tip_ids, :]
    mean_speed = torch.norm(tip_vels, dim=-1).mean(dim=-1)
    return -torch.tanh(scale * mean_speed)


# ═══════════════════════════════════════════════════════════
# 3. REGULARIZATION  →  [-1, 0]
# ═══════════════════════════════════════════════════════════

def action_scale_penalty(env, scale: float = 0.5) -> torch.Tensor:
    """
    -tanh(scale * ||action||)
    Returns: (N,) in [-1, 0]
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.tanh(scale * torch.norm(current_act, dim=-1))


def applied_torque_penalty(env, scale: float = 0.1) -> torch.Tensor:
    """
    -tanh(scale * mean(|torque|))
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    return -torch.tanh(scale * torques.abs().mean(dim=-1))


def mechanical_work_penalty(env, scale: float = 0.5) -> torch.Tensor:
    """
    -tanh(scale * mean(|torque * velocity|))
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    velocities = robot.data.joint_vel
    power = (torques * velocities).abs().mean(dim=-1)
    return -torch.tanh(scale * power)


def action_rate_penalty(env, scale: float = 1.0) -> torch.Tensor:
    """
    -tanh(scale * ||a_t - a_{t-1}||)
    Returns: (N,) in [-1, 0]
    """
    last_act = env.extras.get("last_action")
    current_act = env.extras.get("current_action")
    if last_act is None or current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.tanh(scale * torch.norm(current_act - last_act, dim=-1))


# ═══════════════════════════════════════════════════════════
# 4. SAFETY  →  [-1, 0] or {-1, 0}
# ═══════════════════════════════════════════════════════════

def fingertip_contact_reward(env) -> torch.Tensor:
    """Fraction of fingertips in contact. Returns: (N,) in [0, 1]"""
    forces = _get_fingertip_contact_forces_world(env)
    in_contact = (torch.norm(forces, dim=-1) > 0.5).float()
    return in_contact.mean(dim=-1)


def object_velocity_penalty(env, scale: float = 5.0) -> torch.Tensor:
    """
    -tanh(scale * object_speed)
    Returns: (N,) in [-1, 0]
    """
    obj = env.scene["object"]
    speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    return -torch.tanh(scale * speed)


def object_drop_penalty(env, min_height: float = 0.2) -> torch.Tensor:
    """Binary drop penalty. Returns: (N,) in {-1, 0}"""
    return -(env.scene["object"].data.root_pos_w[:, 2] < min_height).float()


def joint_limit_penalty(env) -> torch.Tensor:
    """
    Fraction of joints near limits (5% margin), negated.
    Returns: (N,) in [-1, 0]
    """
    robot = env.scene["robot"]
    q = robot.data.joint_pos
    q_low = robot.data.soft_joint_pos_limits[..., 0]
    q_high = robot.data.soft_joint_pos_limits[..., 1]
    q_range = (q_high - q_low).clamp(min=1e-6)
    q_norm = (q - q_low) / q_range
    near_limit = (q_norm < 0.05) | (q_norm > 0.95)
    return -near_limit.float().mean(dim=-1)


def object_left_hand_penalty(env, max_dist: float = 0.20) -> torch.Tensor:
    """
    Binary penalty when object escapes hand.
    Returns: (N,) in {-1, 0}
    """
    from .events import _object_escape_mask

    escaped, _ = _object_escape_mask(env, max_dist=max_dist)
    return -escaped.float()
