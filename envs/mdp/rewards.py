"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

All functions follow the Isaac Lab RewardTerm signature:
    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  (num_envs,)

=======================================================================
  FULL REWARD BREAKDOWN  (aligned with DexGen paper §3.2)
=======================================================================

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
=======================================================================
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


# ---------------------------------------------------------------------------
# Goal-related rewards (DexGen paper §3.2 — goal reward)
# ---------------------------------------------------------------------------

def object_pose_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """
    Dense reward for object reaching its target pose in the hand (wrist) frame.

    DexGen goal reward component: target object pose.
    The goal grasp defines where the object should be relative to the hand;
    this reward provides a continuous gradient toward that configuration.

    target_object_pos_hand is set by reset_to_random_grasp from the goal
    grasp's object_pos_hand field (stored during Stage 0 refinement).

    Returns: (N,)  ∈ [0, 1]
    """
    target_pos = env.extras.get("target_object_pos_hand")
    if target_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj   = env.scene["object"]

    # Current object position in hand (wrist root) frame
    from isaaclab.utils.math import quat_apply_inverse
    obj_pos_w    = obj.data.root_pos_w          # (N, 3)
    robot_pos_w  = robot.data.root_pos_w        # (N, 3)
    robot_quat_w = robot.data.root_quat_w       # (N, 4)
    rel_pos      = obj_pos_w - robot_pos_w      # (N, 3)
    obj_pos_hand = quat_apply_inverse(robot_quat_w, rel_pos)  # (N, 3)

    dist = torch.norm(obj_pos_hand - target_pos, dim=-1)      # (N,)
    return torch.exp(-alpha * dist)


def finger_joint_goal_reward(env, alpha: float = 5.0) -> torch.Tensor:
    """
    Dense reward for joint positions reaching the goal grasp configuration.

    DexGen goal reward component: finger joint positions.
    Penalises L2 distance between current joint angles and the goal grasp's
    stored joint_angles in a normalised joint space.

    Returns: (N,)  ∈ [0, 1]
    """
    target_joints = env.extras.get("target_joint_angles")
    if target_joints is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot  = env.scene["robot"]
    q      = robot.data.joint_pos               # (N, D)
    q_low  = robot.data.soft_joint_pos_limits[..., 0]   # (N, D)
    q_high = robot.data.soft_joint_pos_limits[..., 1]   # (N, D)
    q_range = (q_high - q_low).clamp(min=1e-6)

    # Normalise to [0, 1] so all joints contribute equally regardless of range
    q_norm      = (q - q_low) / q_range
    target_norm = (target_joints - q_low) / q_range

    dist = torch.norm(q_norm - target_norm, dim=-1)   # (N,) L2 in normalised space
    return torch.exp(-alpha * dist)


def fingertip_tracking_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """
    Dense tracking reward: exp(-α * ||p_i - p*_i||) per fingertip, averaged.

    Auxiliary goal-related reward providing per-fingertip gradient signal.
    α = 20 gives ~0.82 at 1 cm distance, ~0.37 at 5 cm.

    Returns: (N,)
    """
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target  = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist    = torch.norm(current - target, dim=-1)           # (N, F)
    return torch.exp(-alpha * dist).mean(dim=-1)             # (N,)


def grasp_success_reward(env, threshold: float = 0.01) -> torch.Tensor:
    """
    Sparse success bonus: +1 when ALL fingertips are within `threshold`
    of their targets simultaneously.

    threshold = 0.01 m (1 cm) by default.
    Returns: (N,) binary float
    """
    nf = _get_num_fingers(env)
    current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
    target  = target_fingertip_positions(env).reshape(-1, nf, 3)
    dist    = torch.norm(current - target, dim=-1)           # (N, F)
    return (dist < threshold).all(dim=-1).float()            # (N,)


# ---------------------------------------------------------------------------
# Style reward (DexGen paper: fingertip velocity penalty)
# ---------------------------------------------------------------------------

def fingertip_velocity_penalty(env, vel_thresh: float = 0.1) -> torch.Tensor:
    """
    DexGen style reward: penalise fast fingertip motion.

    Encourages smooth, controlled manipulation rather than jerky transitions.
    penalty = sum_i relu(||v_tip_i|| - vel_thresh) / num_fingers

    Returns: (N,)  ≥ 0
    """
    robot  = env.scene["robot"]
    nf     = _get_num_fingers(env)
    ft_ids = _get_fingertip_body_ids(robot, env)[:nf]

    # body_vel_w: (N, num_bodies, 3)
    ft_vel_w   = robot.data.body_lin_vel_w[:, ft_ids, :]    # (N, F, 3)
    ft_speed   = torch.norm(ft_vel_w, dim=-1)                # (N, F)
    penalty    = torch.relu(ft_speed - vel_thresh).mean(dim=-1)  # (N,)
    return penalty


# ---------------------------------------------------------------------------
# Contact reward
# ---------------------------------------------------------------------------

def fingertip_contact_reward(env) -> torch.Tensor:
    """
    Reward for maintaining contact between fingertips and the object
    during the grasp transition.

    Rationale: the hand should keep touching the object while moving
    to the goal — pure kinematic approaches that lose and regain contact
    are physically unstable.

    Uses ContactSensor if available, otherwise returns 0.

    Returns: (N,)  ∈ [0, 1]  (fraction of tips in contact)
    """
    sensor = env.scene.sensors.get("fingertip_contact_sensor")
    if sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    nf = _get_num_fingers(env)
    forces = _sensor_force_vectors(sensor)
    forces = forces[:, :nf, :]
    force_mag = torch.norm(forces, dim=-1)                      # (N, 4)
    in_contact = (force_mag > 0.5).float()                      # 0.5 N threshold
    return in_contact.mean(dim=-1)                              # (N,)


# ---------------------------------------------------------------------------
# Regularization penalties (DexGen paper: action scale, torque, work)
# ---------------------------------------------------------------------------

def torque_penalty(env) -> torch.Tensor:
    """
    DexGen regularization: penalise large joint torques.

    High torques indicate the robot is fighting against itself or the
    object, which leads to unstable grasps and unrealistic simulation.

    Returns: (N,)  ≥ 0  (mean |τ| across joints)
    """
    robot  = env.scene["robot"]
    # applied_torque shape: (N, D)
    torques = robot.data.applied_torque              # (N, D)
    return torch.abs(torques).mean(dim=-1)           # (N,)


def mechanical_work_penalty(env) -> torch.Tensor:
    """
    DexGen regularization: penalise mechanical work |τ · q̇|.

    Minimising work encourages energy-efficient manipulation strategies
    consistent with the DexGen paper's regularization term.

    Returns: (N,)  ≥ 0  (mean |τ·q̇| across joints)
    """
    robot  = env.scene["robot"]
    torques = robot.data.applied_torque              # (N, D)
    q_dot   = robot.data.joint_vel                   # (N, D)
    work    = torch.abs(torques * q_dot)             # (N, D)
    return work.mean(dim=-1)                         # (N,)


def action_rate_penalty(env) -> torch.Tensor:
    """
    Penalise large changes between consecutive actions (jerk penalty).

    Encourages smooth, continuous motions rather than jerky behaviour.
    Uses L2 norm of action delta normalised by action dimension.

    Returns: (N,)  ≥ 0
    """
    last_act    = env.extras.get("last_action")
    current_act = env.extras.get("current_action")
    if last_act is None or current_act is None:
        return torch.zeros(env.num_envs, device=env.device)
    delta = current_act - last_act                   # (N, 16)
    return torch.norm(delta, dim=-1)                 # (N,)


# ---------------------------------------------------------------------------
# Stability / safety penalties
# ---------------------------------------------------------------------------

def object_left_hand_penalty(env, max_dist: float = 0.25) -> torch.Tensor:
    """
    Binary penalty when the object has escaped from the hand.

    Fires for ANY escape direction (upward fling, sideways slip, downward
    drop), unlike the height-only object_drop_penalty.

    max_dist: 0.25 m — object can be 25 cm from wrist before this fires.
    Returns: (N,)  0 or 1
    """
    robot = env.scene["robot"]
    obj   = env.scene["object"]
    dist  = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    return (dist > max_dist).float()


def object_velocity_penalty(env, lin_thresh: float = 0.1,
                             ang_thresh: float = 1.0) -> torch.Tensor:
    """
    Penalise excessive object linear and angular velocity.

    Rationale: the object should move slowly and stably during the
    grasp transition.  High velocity indicates the hand is flinging
    the object rather than manipulating it.

      penalty = relu(||v_lin|| - lin_thresh) + 0.1 * relu(||ω|| - ang_thresh)

    lin_thresh: 0.1 m/s — object shouldn't be thrown
    ang_thresh: 1.0 rad/s — object shouldn't spin fast

    Returns: (N,)  ≥ 0
    """
    obj     = env.scene["object"]
    v_lin   = torch.norm(obj.data.root_lin_vel_w, dim=-1)    # (N,)
    v_ang   = torch.norm(obj.data.root_ang_vel_w, dim=-1)    # (N,)
    penalty = torch.relu(v_lin - lin_thresh) + 0.1 * torch.relu(v_ang - ang_thresh)
    return penalty


def object_drop_penalty(env, min_height: float = 0.2) -> torch.Tensor:
    """
    Binary penalty when the object falls below `min_height`.

    This is a large one-time penalty that also terminates the episode.
    Combined with the episode termination in AnyGraspTerminationsCfg,
    this strongly discourages dropping.

    Returns: (N,)  0 or 1
    """
    obj    = env.scene["object"]
    height = obj.data.root_pos_w[:, 2]
    return (height < min_height).float()


def joint_limit_penalty(env) -> torch.Tensor:
    """
    Soft penalty that increases as joints approach their limits.

    Activates within 5% of the joint range at either end.
    Prevents the robot from hitting hard stops, which causes
    unrealistic simulation behaviour.

    Returns: (N,)  ≥ 0
    """
    robot   = env.scene["robot"]
    q       = robot.data.joint_pos
    q_low   = robot.data.soft_joint_pos_limits[..., 0]
    q_high  = robot.data.soft_joint_pos_limits[..., 1]
    q_range = (q_high - q_low).clamp(min=1e-6)
    q_norm  = (q - q_low) / q_range                           # [0, 1]

    # Penalty ramps up within 5% of either limit
    penalty = torch.relu(0.05 - q_norm) + torch.relu(q_norm - 0.95)
    return penalty.sum(dim=-1)                                 # (N,)


def wrist_height_penalty(env, min_height: float = 0.1) -> torch.Tensor:
    """
    Penalty if the wrist (robot base) drops below `min_height`.

    Prevents the randomised wrist position from colliding with the table.
    Activates as a smooth ramp: penalty = relu(min_height - wrist_z).

    Returns: (N,)  ≥ 0
    """
    robot      = env.scene["robot"]
    wrist_z    = robot.data.root_pos_w[:, 2]                  # (N,)
    return torch.relu(min_height - wrist_z)
