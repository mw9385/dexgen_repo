"""
Observation functions – Actor (policy) and Critic (privileged).

=======================================================================
  ASYMMETRIC ACTOR-CRITIC OBSERVATION SPLIT
=======================================================================

  ACTOR (policy) — 76 dims — available on a real robot at deployment
  ┌─────────────────────────────────────────────────────────────────┐
  │ joint_pos_normalized       16   encoder, normalised to [-1, 1] │
  │ joint_vel_normalized       16   encoder derivative             │
  │ fingertip_pos_obj_frame    12   FK → object-centric frame      │
  │ target_fingertip_pos       12   goal from GraspGraph           │
  │ fingertip_contact_binary    4   tactile: 1 if tip in contact   │
  │ last_action                16   previous joint targets         │
  └─────────────────────────────────────────────────────────────────┘
  Total: 76

  CRITIC (privileged) — 104 dims — simulation-only, training only
  ┌─────────────────────────────────────────────────────────────────┐
  │ [All actor obs]            76                                   │
  │ object_pos_world            3   true 3-D position              │
  │ object_quat_world           4   true orientation               │
  │ object_lin_vel              3   true linear velocity           │
  │ object_ang_vel              3   true angular velocity          │
  │ fingertip_contact_forces   12   full 3-D force per fingertip   │
  │ dr_params                   3   mass / obj_friction / damping  │
  └─────────────────────────────────────────────────────────────────┘
  Total: 76 + 3 + 4 + 3 + 3 + 12 + 3 = 104

=======================================================================
  Tactile note:
    The standard Allegro Hand has no built-in tactile sensors.
    We approximate tactile feedback using Isaac Lab ContactSensorCfg
    attached to each of the 4 fingertip links.  This gives:
      - Binary contact indicator  →  actor obs (4 dims)
      - Full 3-D contact force    →  critic obs (12 dims)
    At sim-to-real transfer, the binary contact can be replaced with
    a real BioTac / Hall-effect tactile signal.
=======================================================================
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Actor (policy) observations
# ---------------------------------------------------------------------------

def joint_positions_normalized(env) -> torch.Tensor:
    """
    Joint positions normalised to [-1, 1] using soft joint limits.
    Returns: (N, 16)
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    q_low  = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    q_norm = 2.0 * (q - q_low) / (q_high - q_low + 1e-6) - 1.0
    return q_norm.clamp(-1.0, 1.0)


def joint_velocities_normalized(env) -> torch.Tensor:
    """
    Joint velocities clipped and normalised by max velocity (~5 rad/s).
    Returns: (N, 16)
    """
    asset = env.scene["robot"]
    return (asset.data.joint_vel / 5.0).clamp(-1.0, 1.0)


def fingertip_positions_in_object_frame(env) -> torch.Tensor:
    """
    Current fingertip positions in the object's local frame.

    Object-centric representation makes the policy invariant to
    absolute object pose (key insight from DexterityGen §3.2).

    Returns: (N, 12)  order: [index, middle, ring, thumb] × xyz
    """
    robot = env.scene["robot"]
    obj   = env.scene["object"]

    ft_ids   = _get_fingertip_body_ids(robot)
    ft_world = robot.data.body_pos_w[:, ft_ids, :]      # (N, 4, 3)

    obj_pos  = obj.data.root_pos_w                       # (N, 3)
    obj_quat = obj.data.root_quat_w                      # (N, 4) w,x,y,z

    ft_obj = _transform_points_to_local_frame(ft_world, obj_pos, obj_quat)
    return ft_obj.reshape(env.num_envs, -1)              # (N, 12)


def target_fingertip_positions(env) -> torch.Tensor:
    """
    Goal fingertip positions in object frame (set by reset event).
    Returns: (N, 12)
    """
    target = env.extras.get("target_fingertip_pos")
    if target is None:
        return torch.zeros(env.num_envs, 12, device=env.device)
    return target


def fingertip_contact_binary(env) -> torch.Tensor:
    """
    Binary contact indicator per fingertip (1 = in contact with object).

    Source: ContactSensor on each fingertip link.
    Threshold: force magnitude > 0.5 N → contact.

    At sim-to-real, replace with actual tactile sensor signal.

    Returns: (N, 4)
    """
    sensor = env.scene.sensors.get("fingertip_contact_sensor")
    if sensor is None:
        return torch.zeros(env.num_envs, 4, device=env.device)

    # net_forces_w_history: (N, num_bodies, history, 3)
    forces    = sensor.data.net_forces_w_history[:, :, 0, :]   # (N, 4, 3)
    force_mag = torch.norm(forces, dim=-1)                       # (N, 4)
    return (force_mag > 0.5).float()


def last_action(env) -> torch.Tensor:
    """Previous joint position targets. Returns: (N, 16)"""
    action = env.extras.get("last_action")
    if action is None:
        return torch.zeros(env.num_envs, 16, device=env.device)
    return action


# ---------------------------------------------------------------------------
# Critic (privileged) observations
# ---------------------------------------------------------------------------

def object_position_world(env) -> torch.Tensor:
    """True object position in world frame. Returns: (N, 3)"""
    return env.scene["object"].data.root_pos_w


def object_orientation_world(env) -> torch.Tensor:
    """True object quaternion (w, x, y, z). Returns: (N, 4)"""
    return env.scene["object"].data.root_quat_w


def object_linear_velocity(env) -> torch.Tensor:
    """True object linear velocity. Returns: (N, 3)"""
    return env.scene["object"].data.root_lin_vel_w


def object_angular_velocity(env) -> torch.Tensor:
    """True object angular velocity. Returns: (N, 3)"""
    return env.scene["object"].data.root_ang_vel_w


def fingertip_contact_forces(env) -> torch.Tensor:
    """
    Full 3-D contact force vector at each fingertip (simulation only).
    Normalised by 10 N for scale invariance.

    This is privileged info — not available on most real robots.
    Added to critic obs to help value function learn contact dynamics.

    Returns: (N, 12)  [fx, fy, fz] × 4 tips
    """
    sensor = env.scene.sensors.get("fingertip_contact_sensor")
    if sensor is None:
        return torch.zeros(env.num_envs, 12, device=env.device)

    forces = sensor.data.net_forces_w_history[:, :, 0, :]   # (N, 4, 3)
    return (forces / 10.0).clamp(-3.0, 3.0).reshape(env.num_envs, -1)


def domain_randomization_params(env) -> torch.Tensor:
    """
    Current DR parameter values — gives the critic knowledge of the
    simulated physics, so it can better estimate returns under
    varying dynamics.

    Parameters (stored in env.extras["dr_params"] by domain_rand.py):
      [0] object_mass        normalised: mass / 0.15
      [1] object_friction    normalised: friction / 0.75
      [2] joint_damping_mean normalised: damping / 0.1

    Returns: (N, 3)
    """
    params = env.extras.get("dr_params")
    if params is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Cached fingertip body ID lookup
_FT_IDS_CACHE: dict = {}

def _get_fingertip_body_ids(robot) -> list:
    """
    Body indices for the 4 Allegro fingertip links.
      index  → link_3.0_tip
      middle → link_7.0_tip
      ring   → link_11.0_tip
      thumb  → link_15.0_tip
    """
    key = id(robot)
    if key not in _FT_IDS_CACHE:
        tip_names = [
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_15.0_tip",
        ]
        _FT_IDS_CACHE[key] = [robot.find_bodies(n)[0][0] for n in tip_names]
    return _FT_IDS_CACHE[key]


def _transform_points_to_local_frame(
    points_world: torch.Tensor,   # (N, K, 3)
    frame_pos:    torch.Tensor,   # (N, 3)
    frame_quat:   torch.Tensor,   # (N, 4) w,x,y,z
) -> torch.Tensor:
    p     = points_world - frame_pos.unsqueeze(1)
    q_inv = _quat_conjugate(frame_quat)
    return _quat_rotate_batch(q_inv, p)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_rotate_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: (N,4), v: (N,K,3) → (N,K,3)"""
    w     = q[:, 0:1, None]
    xyz_b = q[:, 1:].unsqueeze(1)
    t     = 2.0 * torch.cross(xyz_b.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz_b.expand_as(t), t, dim=-1)
