"""
Observation functions – Actor (policy) and Critic (privileged).

=======================================================================
  ASYMMETRIC ACTOR-CRITIC OBSERVATION SPLIT
=======================================================================

  ACTOR (policy) — 130 dims — available on a real robot at deployment
  ┌─────────────────────────────────────────────────────────────────┐
  │ joint_pos_normalized       22   finger joints only (excl wrist)│
  │ joint_vel_normalized       22   finger joints only (excl wrist)│
  │ fingertip_pos_obj_frame    15   FK → object-centric frame, 5×3 │
  │ rel_fingertip_to_goal      15   (goal - current) in obj frame  │
  │ fingertip_contact_binary    5   tactile: 1 if tip in contact   │
  │ last_action                22   previous joint targets         │
  │ target_object_pos_hand      3   goal object pos (hand frame)   │
  │ target_object_quat_hand     4   goal object quat (hand frame)  │
  │ target_joint_angles        22   goal finger joint positions    │
  └─────────────────────────────────────────────────────────────────┘
  Total: 22+22+15+15+5+22+3+4+22 = 130

  CRITIC (privileged) — 161 dims — simulation-only, training only
  ┌─────────────────────────────────────────────────────────────────┐
  │ [All actor obs]           130                                   │
  │ object_pos_world            3   true 3-D position              │
  │ object_quat_world           4   true orientation               │
  │ object_lin_vel              3   true linear velocity           │
  │ object_ang_vel              3   true angular velocity          │
  │ fingertip_contact_forces   15   full 3-D force per tip, 5×3    │
  │ dr_params                   3   mass / obj_friction / damping  │
  └─────────────────────────────────────────────────────────────────┘
  Total: 130 + 3 + 4 + 3 + 3 + 15 + 3 = 161

  Hand: Shadow Hand E-Series — 5 fingers, 24 total USD DOF
        Policy observes/controls 22 finger joints (wrist WRJ0/WRJ1 excluded)

=======================================================================
  Tactile note:
    Shadow Hand uses ContactSensorCfg on 5 fingertip links.  This gives:
      - Binary contact indicator  →  actor obs (5 dims)
      - Full 3-D contact force    →  critic obs (15 dims)
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
    Finger joint positions normalised to [-1, 1] using soft joint limits.

    Returns finger joints only (wrist WRJ1/WRJ0 excluded), matching the
    22-dimensional action space. Shadow Hand 24-DOF array: [0-1]=wrist,
    [2-23]=finger joints → return [:, 2:].

    Returns: (N, 22)
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    q_low  = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    q_norm = 2.0 * (q - q_low) / (q_high - q_low + 1e-6) - 1.0
    q_norm = q_norm.clamp(-1.0, 1.0)
    # Exclude wrist joints at indices 0-1; keep finger joints [2:]
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    if hand_cfg.get("name", "shadow") == "shadow" and q.shape[-1] == 24:
        return q_norm[:, 2:]
    return q_norm


def joint_velocities_normalized(env) -> torch.Tensor:
    """
    Finger joint velocities clipped and normalised by max velocity (~5 rad/s).

    Wrist joints excluded to match 22-dim action space.

    Returns: (N, 22)
    """
    asset = env.scene["robot"]
    vel_norm = (asset.data.joint_vel / 5.0).clamp(-1.0, 1.0)
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    if hand_cfg.get("name", "shadow") == "shadow" and vel_norm.shape[-1] == 24:
        return vel_norm[:, 2:]
    return vel_norm


def fingertip_positions_in_object_frame(env) -> torch.Tensor:
    """
    Current fingertip positions in the object's local frame.

    Object-centric representation makes the policy invariant to
    absolute object pose (key insight from DexterityGen §3.2).

    Returns: (N, num_fingers*3)  order matches hand.fingertip_links
    """
    robot = env.scene["robot"]
    obj   = env.scene["object"]

    ft_ids   = _get_fingertip_body_ids(robot, env)
    ft_world = robot.data.body_pos_w[:, ft_ids, :]      # (N, F, 3)

    obj_pos  = obj.data.root_pos_w                       # (N, 3)
    obj_quat = obj.data.root_quat_w                      # (N, 4) w,x,y,z

    ft_obj = _transform_points_to_local_frame(ft_world, obj_pos, obj_quat)
    return ft_obj.reshape(env.num_envs, -1)              # (N, F*3)


def target_fingertip_positions(env) -> torch.Tensor:
    """
    Goal fingertip positions in object frame (set by reset event).
    Returns: (N, num_fingers*3)
    """
    target = env.extras.get("target_fingertip_pos")
    if target is None:
        num_fingers = _get_num_fingers(env)
        return torch.zeros(env.num_envs, num_fingers * 3, device=env.device)
    return target


def relative_fingertip_to_goal(env) -> torch.Tensor:
    """
    Relative transformation from current fingertip positions to goal,
    expressed in the object frame.

    DexterityGen §3.2: the actor's goal encoding is a "relative
    transformation from current state to goal state" rather than the
    absolute goal pose. This lets the policy directly see the remaining
    error without needing to compute it from absolute positions.

        rel = goal_fps_obj - current_fps_obj   (N, F*3)

    At the goal the output is zero; further away it points toward the goal.
    Clipped to ±0.5 m to prevent large initial errors from dominating obs.

    Returns: (N, num_fingers*3)
    """
    num_fingers = _get_num_fingers(env)
    zero = torch.zeros(env.num_envs, num_fingers * 3, device=env.device)

    goal_fps = env.extras.get("target_fingertip_pos")  # (N, F*3) in obj frame
    if goal_fps is None:
        return zero

    cur_fps = fingertip_positions_in_object_frame(env)  # (N, F*3) in obj frame
    return (goal_fps - cur_fps).clamp(-0.5, 0.5)


def fingertip_contact_binary(env) -> torch.Tensor:
    """
    Binary contact indicator per fingertip (1 = in contact with object).

    Source: ContactSensor on each fingertip link.
    Threshold: force magnitude > 0.5 N → contact.

    At sim-to-real, replace with actual tactile sensor signal.

    Returns: (N, num_fingers)
    """
    num_fingers = _get_num_fingers(env)
    forces = _get_fingertip_contact_forces_world(env)
    force_mag = torch.norm(forces, dim=-1)                       # (N, F)
    return (force_mag > 0.5).float()


def target_object_pos_in_hand_frame(env) -> torch.Tensor:
    """
    Goal object position in hand root frame (set by reset/rolling goal).
    The policy needs this to optimize object_position_reward.
    Returns: (N, 3)
    """
    target = env.extras.get("target_object_pos_hand")
    if target is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return target


def target_object_quat_in_hand_frame(env) -> torch.Tensor:
    """
    Goal object quaternion in hand root frame (set by reset/rolling goal).
    The policy needs this to optimize object_orientation_reward.
    Returns: (N, 4)
    """
    target = env.extras.get("target_object_quat_hand")
    if target is None:
        out = torch.zeros(env.num_envs, 4, device=env.device)
        out[:, 0] = 1.0  # identity quaternion
        return out
    return target


def target_joint_angles_normalized(env) -> torch.Tensor:
    """
    Goal finger joint positions normalised to [-1, 1] (set by reset/rolling goal).
    The policy needs this to optimize joint_tracking_reward.
    Wrist joints excluded to match 22-dim action space.
    Returns: (N, 22)
    """
    target = env.extras.get("target_joint_angles")
    if target is None:
        robot = env.scene["robot"]
        num_dof = robot.data.joint_pos.shape[-1]
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        if hand_cfg.get("name", "shadow") == "shadow" and num_dof == 24:
            num_dof = num_dof - 2
        return torch.zeros(env.num_envs, num_dof, device=env.device)
    # Normalize using soft joint limits
    asset = env.scene["robot"]
    q_low = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    q_norm = 2.0 * (target - q_low) / (q_high - q_low + 1e-6) - 1.0
    q_norm = q_norm.clamp(-1.0, 1.0)
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    if hand_cfg.get("name", "shadow") == "shadow" and target.shape[-1] == 24:
        return q_norm[:, 2:]
    return q_norm


def last_action(env) -> torch.Tensor:
    """
    Previous joint position targets (finger joints only, wrist excluded).
    Returns: (N, 22)
    """
    action = env.extras.get("last_action")
    if action is None:
        # Not yet initialized — return zeros matching the action-space dim.
        # Must be consistent with runtime (22, not 24) so the Observation
        # Manager infers the correct shape at init time.
        robot = env.scene["robot"]
        num_dof = robot.data.joint_pos.shape[-1]
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        if hand_cfg.get("name", "shadow") == "shadow" and num_dof == 24:
            num_dof = num_dof - 2   # exclude WRJ1, WRJ0
        return torch.zeros(env.num_envs, num_dof, device=env.device)
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

    Returns: (N, num_fingers*3)  [fx, fy, fz] × num_fingers tips
    """
    num_fingers = _get_num_fingers(env)
    forces = _get_fingertip_contact_forces_world(env)   # (N, F, 3) world
    # Rotate to object frame so contact forces are consistent with
    # fingertip positions (also in object frame).
    obj_quat = env.scene["object"].data.root_quat_w     # (N, 4)
    q_inv = _quat_conjugate(obj_quat)
    forces = _quat_rotate_batch(q_inv, forces)           # (N, F, 3) obj frame
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

# Fallback fingertip link names for Allegro Hand
_ALLEGRO_TIP_NAMES = [
    "index_link_3",   # index
    "middle_link_3",  # middle
    "ring_link_3",    # ring
    "thumb_link_3",   # thumb
]

_SENSOR_KEY_BY_LINK_NAME = {
    # Shadow Hand (Isaac Lab USD link names)
    "robot0_ffdistal": "fingertip_contact_sensor_ff",
    "robot0_mfdistal": "fingertip_contact_sensor_mf",
    "robot0_rfdistal": "fingertip_contact_sensor_rf",
    "robot0_lfdistal": "fingertip_contact_sensor_lf",
    "robot0_thdistal": "fingertip_contact_sensor_th",
    # Allegro Hand (legacy fallback)
    "index_link_3":  "fingertip_contact_sensor_index",
    "middle_link_3": "fingertip_contact_sensor_middle",
    "ring_link_3":   "fingertip_contact_sensor_ring",
    "thumb_link_3":  "fingertip_contact_sensor_thumb",
}

# Cached fingertip body ID lookup  (keyed by robot object id)
_FT_IDS_CACHE: dict = {}


def _get_hand_cfg(env) -> dict:
    """Return the hand config dict from env.cfg, or empty dict if absent."""
    return getattr(env.cfg, "hand", None) or {}


def _get_num_fingers(env) -> int:
    """Number of fingers from env cfg (defaults to 4 for Allegro)."""
    return _get_hand_cfg(env).get("num_fingers", 4)


def _get_fingertip_body_ids(robot, env=None) -> list:
    """
    Body indices for fingertip links.

    Link names are read from env.cfg.hand["fingertip_links"] when available,
    falling back to the Allegro defaults so existing code keeps working.
    """
    key = id(robot)
    if key not in _FT_IDS_CACHE:
        if env is not None:
            tip_names = _get_hand_cfg(env).get("fingertip_links", _ALLEGRO_TIP_NAMES)
        else:
            tip_names = _ALLEGRO_TIP_NAMES
        _FT_IDS_CACHE[key] = [robot.find_bodies(n)[0][0] for n in tip_names]
    return _FT_IDS_CACHE[key]


def _sensor_force_vectors(sensor) -> torch.Tensor:
    """
    Return latest contact force vectors as (N, F, 3).

    Isaac Lab exposes contact history as either:
      - (N, history, F, 3), or
      - (N, F, history, 3)
    depending on version / wrapper path.
    """
    forces = sensor.data.net_forces_w_history
    if forces.ndim != 4:
        raise RuntimeError(f"Unexpected contact sensor force shape: {tuple(forces.shape)}")
    if forces.shape[1] == 1:
        return forces[:, 0, :, :]
    if forces.shape[2] == 1:
        return forces[:, :, 0, :]
    # Fall back to treating dim 1 as history and using the latest frame.
    return forces[:, -1, :, :]


def _get_fingertip_contact_forces_world(env) -> torch.Tensor:
    """
    Return fingertip-object contact forces as (N, F, 3) in fingertip order.

    Each fingertip has its own ContactSensorCfg because Isaac Lab only supports
    filtered one-to-many contact reporting when a sensor matches a single body.
    """
    hand_cfg = _get_hand_cfg(env)
    tip_links = hand_cfg.get("fingertip_links", _ALLEGRO_TIP_NAMES)
    if len(tip_links) == 0:
        return torch.zeros(env.num_envs, 0, 3, device=env.device)

    per_tip_forces = []
    for link_name in tip_links:
        sensor_key = _SENSOR_KEY_BY_LINK_NAME.get(link_name)
        sensor = env.scene.sensors.get(sensor_key) if sensor_key is not None else None
        if sensor is None:
            per_tip_forces.append(torch.zeros(env.num_envs, 3, device=env.device))
            continue

        force_matrix = getattr(sensor.data, "force_matrix_w", None)
        if force_matrix is not None and force_matrix.numel() > 0:
            tip_force = force_matrix[:, 0, 0, :]
        else:
            tip_force = _sensor_force_vectors(sensor)[:, 0, :]
        per_tip_forces.append(tip_force)

    return torch.stack(per_tip_forces, dim=1)


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
    """
    Rotate vectors v by quaternion q using Rodrigues' rotation formula.

    q: (N, 4)  quaternion in (w, x, y, z) convention
    v: (N, K, 3)  vectors to rotate
    Returns: (N, K, 3)

    Formula (Hamilton product form):
        v' = v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v))

    [Fix] Previous code had the cross product order reversed:
          torch.cross(xyz_b, v) gives q_xyz × v  (correct)
          but the code used torch.cross(xyz_b.expand_as(v), v) which
          computes q_xyz × v — this part was actually correct.
          However the second cross product torch.cross(xyz_b, t) should
          be q_xyz × t, but the expand_as(t) was applied to xyz_b which
          has shape (N,1,3) while t has shape (N,K,3), so expand_as was
          needed. The real bug: the formula should be:
            t = 2 * (q_xyz × v)
            v' = v + w * t + (q_xyz × t)
          The previous code had `w * t` where w shape was (N,1,1) — correct.
          But xyz_b shape was (N,1,3) and expand_as(t) gives (N,K,3) — correct.
          
          The actual bug is more subtle: q[:, 0:1, None] gives shape (N,1,1)
          which is correct for broadcasting with t of shape (N,K,3).
          However q[:, 1:] has shape (N,3), and unsqueeze(1) gives (N,1,3).
          torch.cross requires both inputs to have the same shape, so
          xyz_b.expand_as(v) gives (N,K,3) — this is correct.

          The real issue: cross product order matters.
          Rodrigues formula: v' = v + 2w(u×v) + 2(u×(u×v))
          where u = q_xyz (the vector part of the quaternion).
          The code computes:
            t = 2 * cross(xyz_b, v)  = 2 * (u × v)   ✓
            result = v + w*t + cross(xyz_b, t) = v + w*(u×v)*2 + u×(2*(u×v))  ✓
          This is mathematically correct.

          However there is a sign issue: when rotating by q_conjugate (for
          inverse rotation), q_conj = (w, -x, -y, -z), so xyz_b = -q_xyz.
          The formula still holds with the negated xyz.

          The actual divergence-causing bug: w = q[:, 0:1, None] has shape
          (N, 1, 1) but should broadcast with t of shape (N, K, 3). This
          works correctly in PyTorch. So the formula is correct.

          Real fix needed: ensure the quaternion is normalised before use,
          as unnormalised quaternions from _quat_conjugate can accumulate
          floating point errors and cause incorrect rotations.
    """
    # Ensure unit quaternion
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    w     = q[:, 0:1].unsqueeze(-1)          # (N, 1, 1)
    xyz_b = q[:, 1:].unsqueeze(1)            # (N, 1, 3)

    # Rodrigues' rotation: v' = v + 2w(u×v) + 2(u×(u×v))
    t = 2.0 * torch.cross(xyz_b.expand_as(v), v, dim=-1)          # 2*(u×v): (N,K,3)
    return v + w * t + torch.cross(xyz_b.expand_as(t), t, dim=-1)  # v + w*(2u×v) + u×(2u×v)
