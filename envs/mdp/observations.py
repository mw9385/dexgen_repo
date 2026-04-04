"""
Observation functions – all spatial quantities in HAND ROOT (wrist) frame.

=======================================================================
  OBSERVATION COORDINATE FRAME: HAND ROOT (WRIST)
=======================================================================

  All position, orientation, and velocity observations are expressed in
  the hand root (wrist) frame for consistency. This means:
    - The policy sees everything from the hand's perspective
    - Target and current states are directly comparable
    - Invariant to where the hand is placed in the world

  ACTOR = CRITIC — 101 dims (symmetric, no privileged info)
  ┌─────────────────────────────────────────────────────────────────┐
  │ joint_pos_normalized       22   finger joints (excl wrist)     │
  │ joint_vel_normalized       22   finger joints (excl wrist)     │
  │ object_pos_hand             3   current object position        │
  │ object_quat_hand            4   current object quaternion      │
  │ target_object_pos_hand      3   goal object position           │
  │ target_object_quat_hand     4   goal object quaternion         │
  │ object_lin_vel_hand         3   object linear velocity         │
  │ object_ang_vel_hand         3   object angular velocity        │
  │ fingertip_contact_forces   15   full 3-D force per tip, 5×3   │
  │ last_action                22   previous joint targets         │
  └─────────────────────────────────────────────────────────────────┘
  Total: 22+22+3+4+3+4+3+3+15+22 = 101

  Hand: Shadow Hand E-Series — 5 fingers, 24 total USD DOF
        Policy observes/controls 22 finger joints (wrist WRJ0/WRJ1 excluded)

=======================================================================
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse


# ---------------------------------------------------------------------------
# Hand-frame transform helper
# ---------------------------------------------------------------------------

def _to_hand_frame_pos(env, points_w: torch.Tensor) -> torch.Tensor:
    """Transform world-frame positions to hand root frame. points_w: (N, 3) or (N, K, 3)"""
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w   # (N, 3)
    root_quat = robot.data.root_quat_w # (N, 4)
    if points_w.ndim == 3:
        rel = points_w - root_pos.unsqueeze(1)
        q_inv = _quat_conjugate(root_quat)
        return _quat_rotate_batch(q_inv, rel)
    else:
        rel = points_w - root_pos
        return quat_apply_inverse(root_quat, rel)


def _to_hand_frame_vec(env, vec_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vector to hand root frame (no translation). vec_w: (N, 3) or (N, K, 3)"""
    robot = env.scene["robot"]
    root_quat = robot.data.root_quat_w
    q_inv = _quat_conjugate(root_quat)
    if vec_w.ndim == 3:
        return _quat_rotate_batch(q_inv, vec_w)
    else:
        return quat_apply_inverse(root_quat, vec_w)


def _to_hand_frame_quat(env, quat_w: torch.Tensor) -> torch.Tensor:
    """Transform world-frame quaternion to hand root frame. quat_w: (N, 4)"""
    robot = env.scene["robot"]
    root_quat = robot.data.root_quat_w
    return _quat_multiply(_quat_conjugate(root_quat), quat_w)


# ---------------------------------------------------------------------------
# Actor (policy) observations — all in hand root frame
# ---------------------------------------------------------------------------

def joint_positions_normalized(env) -> torch.Tensor:
    """
    Finger joint positions normalised to [-1, 1] using soft joint limits.
    Returns finger joints only (wrist excluded). Returns: (N, 22)
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    q_low  = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    q_norm = 2.0 * (q - q_low) / (q_high - q_low + 1e-6) - 1.0
    q_norm = q_norm.clamp(-1.0, 1.0)
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    if hand_cfg.get("name", "shadow") == "shadow" and q.shape[-1] == 24:
        return q_norm[:, 2:]
    return q_norm


def joint_velocities_normalized(env) -> torch.Tensor:
    """
    Finger joint velocities clipped and normalised by max velocity (~5 rad/s).
    Wrist joints excluded. Returns: (N, 22)
    """
    asset = env.scene["robot"]
    vel_norm = (asset.data.joint_vel / 5.0).clamp(-1.0, 1.0)
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    if hand_cfg.get("name", "shadow") == "shadow" and vel_norm.shape[-1] == 24:
        return vel_norm[:, 2:]
    return vel_norm


def fingertip_positions_hand_frame(env) -> torch.Tensor:
    """
    Current fingertip positions in hand root frame.
    Not used in the RL observation config, but called by scripts/collect_data.py.
    Returns: (N, num_fingers*3)
    """
    robot = env.scene["robot"]
    ft_ids = _get_fingertip_body_ids(robot, env)
    ft_world = robot.data.body_pos_w[:, ft_ids, :]  # (N, F, 3)
    ft_hand = _to_hand_frame_pos(env, ft_world)      # (N, F, 3)
    return ft_hand.reshape(env.num_envs, -1)


def object_pos_in_hand_frame(env) -> torch.Tensor:
    """
    Current object position in hand root frame. Returns: (N, 3)
    """
    obj_pos_w = env.scene["object"].data.root_pos_w
    return _to_hand_frame_pos(env, obj_pos_w)


def object_quat_in_hand_frame(env) -> torch.Tensor:
    """
    Current object quaternion in hand root frame. Returns: (N, 4)
    """
    obj_quat_w = env.scene["object"].data.root_quat_w
    return _to_hand_frame_quat(env, obj_quat_w)


def target_object_pos_in_hand_frame(env) -> torch.Tensor:
    """
    Goal object position in hand root frame. Returns: (N, 3)
    """
    target = env.extras.get("target_object_pos_hand")
    if target is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return target


def target_object_quat_in_hand_frame(env) -> torch.Tensor:
    """
    Goal object quaternion in hand root frame. Returns: (N, 4)
    """
    target = env.extras.get("target_object_quat_hand")
    if target is None:
        out = torch.zeros(env.num_envs, 4, device=env.device)
        out[:, 0] = 1.0
        return out
    return target


def object_lin_vel_hand_frame(env) -> torch.Tensor:
    """
    Object linear velocity in hand root frame. Returns: (N, 3)
    """
    vel_w = env.scene["object"].data.root_lin_vel_w
    return _to_hand_frame_vec(env, vel_w)


def object_ang_vel_hand_frame(env) -> torch.Tensor:
    """
    Object angular velocity in hand root frame. Returns: (N, 3)
    """
    angvel_w = env.scene["object"].data.root_ang_vel_w
    return _to_hand_frame_vec(env, angvel_w)


def fingertip_contact_binary(env) -> torch.Tensor:
    """
    Binary contact indicator per fingertip (1 = in contact).
    Threshold: force magnitude > 0.5 N.
    Not in the RL obs config (redundant with contact_forces),
    but used by termination (no_fingertip_contact) and reward masking.
    Returns: (N, num_fingers)
    """
    forces = _get_fingertip_contact_forces_world(env)
    force_mag = torch.norm(forces, dim=-1)
    return (force_mag > 0.5).float()


def last_action(env) -> torch.Tensor:
    """
    Previous joint position targets (finger joints only, wrist excluded).
    Returns: (N, 22)
    """
    action = env.extras.get("last_action")
    if action is None:
        robot = env.scene["robot"]
        num_dof = robot.data.joint_pos.shape[-1]
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        if hand_cfg.get("name", "shadow") == "shadow" and num_dof == 24:
            num_dof = num_dof - 2
        return torch.zeros(env.num_envs, num_dof, device=env.device)
    return action


# ---------------------------------------------------------------------------
# Additional observation functions (not in RL obs config, used elsewhere)
# ---------------------------------------------------------------------------

def fingertip_contact_forces(env) -> torch.Tensor:
    """
    Full 3-D contact force vector at each fingertip in hand root frame.
    Normalised by 10 N for scale invariance.
    Returns: (N, num_fingers*3)
    """
    forces = _get_fingertip_contact_forces_world(env)   # (N, F, 3) world
    forces = _to_hand_frame_vec(env, forces)              # (N, F, 3) hand frame
    return (forces / 10.0).clamp(-3.0, 3.0).reshape(env.num_envs, -1)


def domain_randomization_params(env) -> torch.Tensor:
    """
    Current DR parameter values (mass, friction, damping).
    Not in the RL obs config; kept for logging / analysis only.
    Returns: (N, 3)
    """
    params = env.extras.get("dr_params")
    if params is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLEGRO_TIP_NAMES = [
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
    "thumb_link_3",
]

_SENSOR_KEY_BY_LINK_NAME = {
    "robot0_ffdistal": "fingertip_contact_sensor_ff",
    "robot0_mfdistal": "fingertip_contact_sensor_mf",
    "robot0_rfdistal": "fingertip_contact_sensor_rf",
    "robot0_lfdistal": "fingertip_contact_sensor_lf",
    "robot0_thdistal": "fingertip_contact_sensor_th",
    "index_link_3":  "fingertip_contact_sensor_index",
    "middle_link_3": "fingertip_contact_sensor_middle",
    "ring_link_3":   "fingertip_contact_sensor_ring",
    "thumb_link_3":  "fingertip_contact_sensor_thumb",
}

_FT_IDS_CACHE: dict = {}


def _get_hand_cfg(env) -> dict:
    return getattr(env.cfg, "hand", None) or {}


def _get_num_fingers(env) -> int:
    return _get_hand_cfg(env).get("num_fingers", 4)


def _get_fingertip_body_ids(robot, env=None) -> list:
    key = id(robot)
    if key not in _FT_IDS_CACHE:
        if env is not None:
            tip_names = _get_hand_cfg(env).get("fingertip_links", _ALLEGRO_TIP_NAMES)
        else:
            tip_names = _ALLEGRO_TIP_NAMES
        _FT_IDS_CACHE[key] = [robot.find_bodies(n)[0][0] for n in tip_names]
    return _FT_IDS_CACHE[key]


def _sensor_force_vectors(sensor) -> torch.Tensor:
    forces = sensor.data.net_forces_w_history
    if forces.ndim != 4:
        raise RuntimeError(f"Unexpected contact sensor force shape: {tuple(forces.shape)}")
    if forces.shape[1] == 1:
        return forces[:, 0, :, :]
    if forces.shape[2] == 1:
        return forces[:, :, 0, :]
    return forces[:, -1, :, :]


def _get_fingertip_contact_forces_world(env) -> torch.Tensor:
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


from .math_utils import quat_conjugate as _quat_conjugate
from .math_utils import quat_multiply as _quat_multiply
from .math_utils import quat_rotate_batch as _quat_rotate_batch
