"""
Reward functions for the AnyGrasp-to-AnyGrasp environment.

All functions follow the Isaac Lab RewardTerm signature:
    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  (num_envs,)

=======================================================================
  FULL REWARD BREAKDOWN
=======================================================================

  Positive (encouraging goal-directed behavior):
    fingertip_tracking      +10.0   exp distance to goal per fingertip
    grasp_success           +50.0   binary bonus: all tips within ε
    fingertip_contact        +2.0   maintain contact during transition

  Negative (penalising bad behavior):
    action_rate             -0.01   penalise large action changes (jerk)
    object_velocity         -0.5    penalise spinning / flinging object
    object_drop           -200.0    heavy penalty on drop
    joint_limit             -0.1    soft penalty near joint limits
    wrist_height            -1.0    prevent wrist from hitting table

  Reward scale discussion:
    - fingertip_tracking uses exp(-20*dist): at dist=5 cm → 0.37,
      at dist=1 cm → 0.82, at dist=0 → 1.0
    - grasp_success fires only when ALL 4 tips are within 1 cm
    - object_velocity penalises ||v|| > 0.1 m/s (smooth manipulation)
    - wrist_height fires below 0.1 m to prevent table collisions
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
# Positive rewards
# ---------------------------------------------------------------------------

def fingertip_tracking_reward(env, alpha: float = 20.0) -> torch.Tensor:
    """
    Dense tracking reward: exp(-α * ||p_i - p*_i||) per fingertip, averaged.

    Provides a continuous gradient signal throughout the episode.
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
# Negative rewards (penalties)
# ---------------------------------------------------------------------------

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
