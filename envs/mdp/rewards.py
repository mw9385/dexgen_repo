"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

All functions follow the Isaac Lab RewardTerm signature:
  FULL REWARD BREAKDOWN  (aligned with DexGen paper §3.2)

  Goal-related rewards (encouraging goal-directed behavior):
    object_pose_reward        +15.0   exp distance: object → target pose in hand
    finger_joint_goal_reward   +8.0   exp distance: joints → goal joint angles
    fingertip_tracking         +5.0   exp distance to goal per fingertip (auxiliary)
    grasp_success             +50.0   binary bonus: all tips within ε

  Style reward (DexGen paper: fingertip velocity penalty):
    fingertip_velocity        -0.5    penalise fast fingertip motion

  Safety / contact rewards:
    fingertip_contact          +2.0   maintain contact during transition

  Regularization (DexGen paper: action scale, torque, work):
    torque_penalty            -0.002  penalise large joint torques
    mechanical_work_penalty   -0.001  penalise mechanical work (τ·q̇)
    action_rate               -0.01   penalise large action changes (jerk)

  Stability penalties:
    object_velocity           -0.5    penalise spinning / flinging object
    object_drop             -200.0    heavy penalty on drop
    object_left_hand         -100.0   penalty when object escapes from hand
    joint_limit               -0.1    soft penalty near joint limits
    wrist_height              -1.0    prevent wrist from hitting table

  Reward scale discussion:
    - object_pose_reward exp(-20*dist): at dist=5 cm → 0.37, at 0 → 1.0
    - finger_joint_goal_reward exp(-5*||Δq||): at dist=0.2 rad → 0.37
    - grasp_success fires only when ALL 4 tips are within 1 cm
    - fingertip_velocity: relu(||v_tip|| - 0.1 m/s) per tip, summed
    - torque_penalty: sum(||τ_i||) / num_joints
    - mechanical_work_penalty: sum(|τ_i · q̇_i|) / num_joints
func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  (num_envs,)

# =======================================================================
DexGen 논문 기반 3축 보상 구조

1. GOAL-RELATED REWARDS
   object_pose_goal_reward     [NEW]  object position + orientation
   finger_joint_goal_reward    [NEW]  joint-space goal
   fingertip_tracking_reward          fingertip position (기존)
   grasp_success_reward               all tips within threshold (기존)
2. STYLE REWARD
   fingertip_velocity_penalty  [NEW]  fingertip velocity 최소화
3. REGULARIZATION
   action_scale_penalty        [NEW]  action magnitude
   applied_torque_penalty      [NEW]  torque 최소화
   mechanical_work_penalty     [NEW]  에너지 소비 최소화
   action_rate_penalty                jerk 최소화 (기존)
4. SAFETY (논문 외)
   fingertip_contact_reward           접촉 유지 (기존)
   object_velocity_penalty            물체 속도 제한 (기존)
   object_drop_penalty                낙하 페널티 (기존)
   joint_limit_penalty                관절 한계 (기존)
   wrist_height_penalty               테이블 충돌 방지 (기존)
# =======================================================================
"""

from __future__ import annotations
import torch

from .observations import (
    fingertip_positions_in_object_frame,
    target_fingertip_positions,
    _get_fingertip_body_ids,
    _get_num_fingers,
    _sensor_force_vectors,
)

# ═══════════════════════════════════════════════════════════
# 1. GOAL-RELATED REWARDS
# ═══════════════════════════════════════════════════════════

def object_pose_goal_reward(env, pos_scale: float = 10.0, rot_scale: float = 5.0) -> torch.Tensor:
    """
    [NEW] Object pose가 목표 grasp의 object pose에 가까운지.
    DexGen goal-related reward의 핵심.

    Position: exp(-pos_scale * ||p - p*||)
    Rotation: exp(-rot_scale * (1 - |<q, q*>|))

    Returns: (N,) in [0, 2]

    Requires: env.goal_object_pos (N,3), env.goal_object_quat (N,4)
    """
    obj = env.scene["object"]
    current_pos = obj.data.root_pos_w
    current_quat = obj.data.root_quat_w
    goal_pos = env.goal_object_pos
    goal_quat = env.goal_object_quat

    d_pos = torch.norm(current_pos - goal_pos, dim=-1)
    quat_dot = torch.sum(current_quat * goal_quat, dim=-1).abs()
    d_rot = 1.0 - quat_dot.clamp(0.0, 1.0)

    return torch.exp(-pos_scale * d_pos) + torch.exp(-rot_scale * d_rot)


def finger_joint_goal_reward(env, scale: float = 5.0) -> torch.Tensor:
    """
    [NEW] Joint positions가 목표 grasp의 joint config에 가까운지.
    Fingertip position만으로는 joint ambiguity가 남음.

    reward = exp(-scale * ||q - q*||)

    Returns: (N,) in [0, 1]

    Requires: env.goal_joint_positions (N, 16)
    """
    robot = env.scene["robot"]
    current_joints = robot.data.joint_pos
    goal_joints = env.goal_joint_positions

    d_joint = torch.norm(current_joints - goal_joints, dim=-1)
    return torch.exp(-scale * d_joint)


def fingertip_tracking_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """exp(-alpha * dist) per fingertip, averaged. Returns: (N,)"""
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist = torch.norm(current - target, dim=-1)
    return torch.exp(-alpha * dist).mean(dim=-1)


def grasp_success_reward(env, threshold: float = 0.01) -> torch.Tensor:
    """+1 when ALL fingertips within threshold. Returns: (N,) binary"""
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist = torch.norm(current - target, dim=-1)
    return (dist < threshold).all(dim=-1).float()

# ═══════════════════════════════════════════════════════════
# 2. STYLE REWARD
# ═══════════════════════════════════════════════════════════

def fingertip_velocity_penalty(env) -> torch.Tensor:
    """
    [NEW] Fingertip velocity penalty (논문 style reward).
    기존 object_velocity_penalty(safety)와 별개.

    penalty = sum_i ||v_tip_i||^2

    Returns: (N,) >= 0
    """
    robot = env.scene["robot"]
    tip_ids = _get_fingertip_body_ids(env)
    tip_vels = robot.data.body_lin_vel_w[:, tip_ids, :]
    return (tip_vels ** 2).sum(dim=-1).sum(dim=-1)

# ═══════════════════════════════════════════════════════════
# 3. REGULARIZATION
# ═══════════════════════════════════════════════════════════

def action_scale_penalty(env) -> torch.Tensor:
    """
    [NEW] Action magnitude penalty (논문 regularization).
    action_rate(jerk)와 다름: 이건 action 자체 크기.

    penalty = ||a||^2

    Returns: (N,) >= 0
    """
    current_act = env.extras.get("current_action")
    if current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (current_act ** 2).sum(dim=-1)


def applied_torque_penalty(env) -> torch.Tensor:
    """
    [NEW] Applied torque penalty (논문 regularization).

    penalty = ||tau||^2

    Returns: (N,) >= 0
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    return (torques ** 2).sum(dim=-1)


def mechanical_work_penalty(env) -> torch.Tensor:
    """
    [NEW] Mechanical work (power) penalty (논문 regularization).

    penalty = sum_i |tau_i * qdot_i|

    Returns: (N,) >= 0
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    velocities = robot.data.joint_vel
    return (torques * velocities).abs().sum(dim=-1)


def action_rate_penalty(env) -> torch.Tensor:
    """Jerk penalty: ||a_t - a_{t-1}||. Returns: (N,) >= 0"""
    last_act = env.extras.get("last_action")
    current_act = env.extras.get("current_action")
    if last_act is None or current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.norm(current_act - last_act, dim=-1)

# ═══════════════════════════════════════════════════════════
# 4. SAFETY (논문 외)
# ═══════════════════════════════════════════════════════════

def fingertip_contact_reward(env) -> torch.Tensor:
    """Fraction of fingertips in contact. Returns: (N,) in [0,1]"""
    sensor = env.scene.sensors.get("fingertip_contact_sensor")
    if sensor is None:
        return torch.zeros(env.num_envs, device=env.device)
    nf = _get_num_fingers(env)
    forces = _sensor_force_vectors(sensor)[:, :nf, :]
    in_contact = (torch.norm(forces, dim=-1) > 0.5).float()
    return in_contact.mean(dim=-1)


def object_velocity_penalty(env, lin_thresh: float = 0.1, ang_thresh: float = 1.0) -> torch.Tensor:
    """Safety: penalise object fling. Returns: (N,) >= 0"""
    obj = env.scene["object"]
    v_lin = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    return torch.relu(v_lin - lin_thresh) + 0.1 * torch.relu(v_ang - ang_thresh)


def object_drop_penalty(env, min_height: float = 0.2) -> torch.Tensor:
    """Binary drop penalty. Returns: (N,) 0 or 1"""
    return (env.scene["object"].data.root_pos_w[:, 2] < min_height).float()


def joint_limit_penalty(env) -> torch.Tensor:
    """Soft penalty near joint limits (5% margin). Returns: (N,) >= 0"""
    robot = env.scene["robot"]
    q = robot.data.joint_pos
    q_low = robot.data.soft_joint_pos_limits[..., 0]
    q_high = robot.data.soft_joint_pos_limits[..., 1]
    q_range = (q_high - q_low).clamp(min=1e-6)
    q_norm = (q - q_low) / q_range
    return (torch.relu(0.05 - q_norm) + torch.relu(q_norm - 0.95)).sum(dim=-1)


def wrist_height_penalty(env, min_height: float = 0.1) -> torch.Tensor:
    """Prevent wrist-table collision. Returns: (N,) >= 0"""
    return torch.relu(min_height - env.scene["robot"].data.root_pos_w[:, 2])


def object_left_hand_penalty(env, max_dist: float = 0.25) -> torch.Tensor:
    """
    Binary penalty when object escapes from the hand (any direction).

    Catches upward flings, sideways slips, and downward drops — unlike
    object_drop_penalty which only fires when the object hits the floor.
    Used together with the object_left_hand DoneTerm for immediate episode
    termination + large negative reward.

    Returns: (N,)  0.0 or 1.0
    """
    robot = env.scene["robot"]
    obj   = env.scene["object"]
    dist  = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    return (dist > max_dist).float()
