"""
Observation functions – all spatial quantities in HAND ROOT frame.

  ACTOR = CRITIC — 101 dims (symmetric, no privileged info)
  ┌─────────────────────────────────────────────────────────────────┐
  │ joint_pos_normalized       22   all joints (Sharpa Hand 22-DOF)│
  │ joint_vel_normalized       22   all joints                     │
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

  Hand: Sharpa Wave Hand — 5 fingers, 22 DOF (all actuated)
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse


# ---------------------------------------------------------------------------
# Hand-frame transform helper
# ---------------------------------------------------------------------------

def _to_hand_frame_pos(env, points_w: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    if points_w.ndim == 3:
        rel = points_w - root_pos.unsqueeze(1)
        q_inv = _quat_conjugate(root_quat)
        return _quat_rotate_batch(q_inv, rel)
    else:
        rel = points_w - root_pos
        return quat_apply_inverse(root_quat, rel)


def _to_hand_frame_vec(env, vec_w: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    root_quat = robot.data.root_quat_w
    q_inv = _quat_conjugate(root_quat)
    if vec_w.ndim == 3:
        return _quat_rotate_batch(q_inv, vec_w)
    else:
        return quat_apply_inverse(root_quat, vec_w)


def _to_hand_frame_quat(env, quat_w: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    root_quat = robot.data.root_quat_w
    return _quat_multiply(_quat_conjugate(root_quat), quat_w)


# ---------------------------------------------------------------------------
# Actor (policy) observations — all in hand root frame
# ---------------------------------------------------------------------------

def joint_positions_normalized(env) -> torch.Tensor:
    """Joint positions normalised to [-1, 1]. Returns all 22 DOF."""
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    q_low = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    q_norm = 2.0 * (q - q_low) / (q_high - q_low + 1e-6) - 1.0
    return q_norm.clamp(-1.0, 1.0)


def joint_velocities_normalized(env) -> torch.Tensor:
    """Joint velocities normalised by 5 rad/s. Returns all 22 DOF."""
    asset = env.scene["robot"]
    return (asset.data.joint_vel / 5.0).clamp(-1.0, 1.0)


def fingertip_positions_hand_frame(env) -> torch.Tensor:
    """Fingertip positions in hand root frame. Returns: (N, num_fingers*3)"""
    robot = env.scene["robot"]
    ft_ids = _get_fingertip_body_ids(robot, env)
    ft_world = robot.data.body_pos_w[:, ft_ids, :]
    ft_hand = _to_hand_frame_pos(env, ft_world)
    return ft_hand.reshape(env.num_envs, -1)


def object_pos_in_hand_frame(env) -> torch.Tensor:
    obj_pos_w = env.scene["object"].data.root_pos_w
    return _to_hand_frame_pos(env, obj_pos_w)


def object_quat_in_hand_frame(env) -> torch.Tensor:
    obj_quat_w = env.scene["object"].data.root_quat_w
    return _to_hand_frame_quat(env, obj_quat_w)


def target_object_pos_in_hand_frame(env) -> torch.Tensor:
    target = env.extras.get("target_object_pos_hand")
    if target is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return target


def target_object_quat_in_hand_frame(env) -> torch.Tensor:
    target = env.extras.get("target_object_quat_hand")
    if target is None:
        out = torch.zeros(env.num_envs, 4, device=env.device)
        out[:, 0] = 1.0
        return out
    return target


def object_lin_vel_hand_frame(env) -> torch.Tensor:
    vel_w = env.scene["object"].data.root_lin_vel_w
    return _to_hand_frame_vec(env, vel_w)


def object_ang_vel_hand_frame(env) -> torch.Tensor:
    angvel_w = env.scene["object"].data.root_ang_vel_w
    return _to_hand_frame_vec(env, angvel_w)


def fingertip_contact_binary(env) -> torch.Tensor:
    """Binary contact per fingertip (force > 0.5N). Returns: (N, num_fingers)"""
    forces = _get_fingertip_contact_forces_world(env)
    return (torch.norm(forces, dim=-1) > 0.5).float()


def last_action(env) -> torch.Tensor:
    """Previous joint targets. Returns: (N, 22)"""
    action = env.extras.get("last_action")
    if action is None:
        num_dof = env.scene["robot"].data.joint_pos.shape[-1]
        return torch.zeros(env.num_envs, num_dof, device=env.device)
    return action


def fingertip_contact_forces(env) -> torch.Tensor:
    """3-D contact force per fingertip in hand frame, normalised. Returns: (N, 15)"""
    forces = _get_fingertip_contact_forces_world(env)
    forces = _to_hand_frame_vec(env, forces)
    return (forces / 10.0).clamp(-3.0, 3.0).reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sharpa Hand fingertip link → sensor attribute mapping
_SENSOR_KEY_BY_LINK_NAME = {
    "right_thumb_fingertip": "fingertip_contact_sensor_thumb",
    "right_index_fingertip": "fingertip_contact_sensor_index",
    "right_middle_fingertip": "fingertip_contact_sensor_middle",
    "right_ring_fingertip": "fingertip_contact_sensor_ring",
    "right_pinky_fingertip": "fingertip_contact_sensor_pinky",
}

_DEFAULT_TIP_NAMES = [
    "right_thumb_fingertip",
    "right_index_fingertip",
    "right_middle_fingertip",
    "right_ring_fingertip",
    "right_pinky_fingertip",
]

_FT_IDS_CACHE: dict = {}


def _get_hand_cfg(env) -> dict:
    return getattr(env.cfg, "hand", None) or {}


def _get_fingertip_body_ids(robot, env=None) -> list:
    key = id(robot)
    if key not in _FT_IDS_CACHE:
        if env is not None:
            tip_names = _get_hand_cfg(env).get("fingertip_links", _DEFAULT_TIP_NAMES)
        else:
            tip_names = _DEFAULT_TIP_NAMES
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
    tip_links = hand_cfg.get("fingertip_links", _DEFAULT_TIP_NAMES)

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
