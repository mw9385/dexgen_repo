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
    _get_fingertip_contact_forces_world,
)

# ═══════════════════════════════════════════════════════════
# 1. GOAL-RELATED REWARDS
# ═══════════════════════════════════════════════════════════

def object_pose_goal_reward(
    env,
    alpha_pos: float = 10.0,
    alpha_orn: float = 5.0,
) -> torch.Tensor:
    """
    DexterityGen paper eq.(5): single combined exponential goal reward.

        r_goal_pose = exp(-alpha_pos * ||p_obj - p*_obj|| - alpha_orn * d(R, R*))

    where d(R, R*) = 1 - |q · q*| (geodesic distance proxy in [0, 1]).

    Returns: (N,) in (0, 1]

    env.extras["target_object_pos_hand"]:  (N, 3) goal object pos in wrist frame
    env.extras["target_object_quat_hand"]: (N, 4) goal object quat in wrist frame
    (set by reset_to_random_grasp from the GOAL grasp node, not the start pose)
    """
    from isaaclab.utils.math import quat_apply_inverse

    goal_pos  = env.extras.get("target_object_pos_hand")
    goal_quat = env.extras.get("target_object_quat_hand")
    if goal_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj   = env.scene["object"]

    # Current object position expressed in wrist (hand root) frame
    rel_pos      = obj.data.root_pos_w - robot.data.root_pos_w          # (N, 3)
    cur_pos_hand = quat_apply_inverse(robot.data.root_quat_w, rel_pos)  # (N, 3)
    d_pos = torch.norm(cur_pos_hand - goal_pos, dim=-1)                 # (N,)

    if goal_quat is not None:
        # Current object quat in hand frame: q_hand = q_wrist_inv * q_obj
        qw = robot.data.root_quat_w    # (N, 4) w,x,y,z
        qo = obj.data.root_quat_w      # (N, 4)
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
        cur_quat_hand = _qmul(qw_inv, qo)                               # (N, 4)
        quat_dot = torch.sum(cur_quat_hand * goal_quat, dim=-1).abs().clamp(0.0, 1.0)
        d_rot = 1.0 - quat_dot                                          # (N,) in [0,1]
        # Paper eq.(5): single combined exponent
        return torch.exp(-alpha_pos * d_pos - alpha_orn * d_rot)

    return torch.exp(-alpha_pos * d_pos)


def finger_joint_goal_reward(env, alpha_hand: float = 2.0) -> torch.Tensor:
    """
    Positive exponential reward for joint-space proximity to the goal grasp.

        exp(-alpha_hand * ||q - q*||_2 / sqrt(num_dof))

    Normalised by sqrt(num_dof) so the effective distance is per-DOF
    independent of the number of joints (Shadow Hand: 24 DOF).

    Typical ranges (Shadow Hand, 24 DOF):
      ||q - q*|| ~ 0.0  → exp(0)      = 1.0   (at perfect goal)
      ||q - q*|| ~ 1.0  → exp(-0.41)  = 0.67  (~0.2 rad/joint avg)
      ||q - q*|| ~ 3.0  → exp(-1.22)  = 0.29  (~0.6 rad/joint avg, far)

    We deviate slightly from the paper's negative-linear formulation to get
    a positive shaping signal that makes it profitable to move toward goal.
    The negative-linear form makes "staying still" preferable to taking risks.

    Returns: (N,) in (0, 1]
    """
    goal_joints = env.extras.get("target_joint_angles")
    if goal_joints is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot    = env.scene["robot"]
    q        = robot.data.joint_pos     # (N, num_dof)
    num_dof  = q.shape[-1]
    dq_norm  = torch.norm(q - goal_joints, dim=-1) / (num_dof ** 0.5)  # (N,)
    return torch.exp(-alpha_hand * dq_norm)


def fingertip_tracking_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """
    Exponential fingertip-tracking reward, averaged over fingertips.

        reward = mean_i  exp(-alpha * ||tip_i - goal_i||)

    Output ∈ (0, 1] per finger. Rises smoothly toward 1.0 as each fingertip
    approaches its goal position (stored in env.extras["target_fingertip_pos"]
    from the grasp graph, in object frame).

    With alpha=20.0:
      dist=0.00m (at goal) → exp(0)    = 1.00
      dist=0.05m (5 cm)    → exp(-1.0) = 0.37
      dist=0.10m (10 cm)   → exp(-2.0) = 0.14

    Returns: (N,) in (0, 1]
    """
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target  = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist    = torch.norm(current - target, dim=-1)              # (N, F)
    return torch.exp(-alpha * dist).mean(dim=-1)                # (N,)


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

    penalty = mean_i ||v_tip_i||^2   (mean over fingers, not sum)

    Returns: (N,) >= 0
    """
    robot   = env.scene["robot"]
    nf      = _get_num_fingers(env)
    tip_ids = _get_fingertip_body_ids(robot, env)[:nf]
    tip_vels = robot.data.body_lin_vel_w[:, tip_ids, :]    # (N, F, 3)
    return (tip_vels ** 2).sum(dim=-1).mean(dim=-1)         # (N,)

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

    penalty = mean(tau_i^2)  — mean over joints so the value is independent
    of the number of DOF (Shadow Hand: 24 joints would otherwise inflate the
    sum-version by 24x compared to a 4-finger hand).

    Returns: (N,) >= 0
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    return (torques ** 2).mean(dim=-1)


def mechanical_work_penalty(env) -> torch.Tensor:
    """
    [NEW] Mechanical work (power) penalty (논문 regularization).

    penalty = mean_i |tau_i * qdot_i|  — mean over joints for DOF-independence

    Returns: (N,) >= 0
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    velocities = robot.data.joint_vel
    return (torques * velocities).abs().mean(dim=-1)


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
    forces = _get_fingertip_contact_forces_world(env)
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


def object_left_hand_penalty(env, max_dist: float = 0.20) -> torch.Tensor:
    """
    Binary penalty when object escapes from the hand (any direction).

    Catches upward flings, sideways slips, and downward drops — unlike
    object_drop_penalty which only fires when the object hits the floor.
    Used together with the object_left_hand DoneTerm for immediate episode
    termination + large negative reward.

    Returns: (N,)  0.0 or 1.0
    """
    from .events import _object_escape_mask

    escaped, _ = _object_escape_mask(env, max_dist=max_dist)
    return escaped.float()
