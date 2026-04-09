"""
Observation functions — Sharpa Wave Hand.

Observation structure (sharpa-rl-lab 기반 + DexGen target 유지):

  Per-step block (64 dims, sharpa와 동일):
    joint_pos_normalized    22   (unscale to [-1, 1])
    joint_targets           22   (current position targets)
    tactile_forces           5   (smoothed contact force magnitude)
    contact_positions       15   (5 fingers × 3, in tactile frame)

  3-step temporal stacking: 64 × 3 = 192 dims

  + Target info (non-temporal, appended once):
    target_object_pos_hand   3
    target_object_quat_hand  4

  Total: 192 + 7 = 199 dims
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse

from .math_utils import quat_conjugate as _quat_conjugate
from .math_utils import quat_multiply as _quat_multiply


# ---------------------------------------------------------------------------
# Sharpa tactile sensing (from sharpa_wave_env.py compute_observations)
# ---------------------------------------------------------------------------

def _compute_sharpa_tactile(env) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute smoothed tactile forces + contact positions (sharpa-rl-lab style).

    Returns:
        sensed_contacts: (N, 5) smoothed force magnitudes
        contact_pos: (N, 15) contact positions (5×3), zeros if disabled
    """
    hand_cfg = _get_hand_cfg(env)
    contact_smooth = float(hand_cfg.get("contact_smooth", 0.5))

    # Net contact forces history from elastomer sensors
    sensor_keys = [
        "fingertip_contact_sensor_thumb",
        "fingertip_contact_sensor_index",
        "fingertip_contact_sensor_middle",
        "fingertip_contact_sensor_ring",
        "fingertip_contact_sensor_pinky",
    ]

    force_history = []
    for key in sensor_keys:
        sensor = env.scene.sensors.get(key)
        if sensor is None:
            force_history.append(
                torch.zeros(env.num_envs, 3, 1, device=env.device)
            )
            continue
        # net_forces_w_history: (N, history_len, num_bodies, 3)
        h = sensor.data.net_forces_w_history
        if h.ndim == 4:
            # (N, history, bodies, 3) → take body 0
            force_history.append(h[:, :, 0, :])  # (N, history, 3)
        else:
            force_history.append(
                torch.zeros(env.num_envs, 3, 3, device=env.device)
            )

    # Stack: (N, history, 5, 3) → norm → (N, history, 5)
    stacked = torch.stack(force_history, dim=2)  # (N, H, 5, 3)
    norm_forces = torch.norm(stacked, dim=-1)     # (N, H, 5)

    # Smoothing: latest * smooth + prev * (1-smooth)
    if norm_forces.shape[1] >= 2:
        smooth = norm_forces[:, 0, :] * contact_smooth + norm_forces[:, 1, :] * (1 - contact_smooth)
    else:
        smooth = norm_forces[:, 0, :]

    # Latency simulation
    if "last_contacts" not in env.extras:
        env.extras["last_contacts"] = torch.zeros(
            env.num_envs, 5, device=env.device,
        )

    latency = 0.005
    latency_mask = torch.where(
        torch.rand_like(env.extras["last_contacts"]) < latency, 1.0, 0.0,
    )
    env.extras["last_contacts"] = (
        env.extras["last_contacts"] * latency_mask
        + smooth * (1 - latency_mask)
    )
    sensed_contacts = env.extras["last_contacts"].clone()

    # Contact positions (zeros for now — enable_contact_pos=False by default)
    contact_pos = torch.zeros(env.num_envs, 15, device=env.device)

    return sensed_contacts, contact_pos


# ---------------------------------------------------------------------------
# Normalized joint positions (sharpa style: unscale to [-1, 1])
# ---------------------------------------------------------------------------

def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower + 1e-6)


def joint_positions_normalized(env) -> torch.Tensor:
    """Joint positions normalised to [-1, 1]. All 22 DOF."""
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    q_low = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    return _unscale(q, q_low, q_high).clamp(-1.0, 1.0)


def joint_velocities_normalized(env) -> torch.Tensor:
    """Joint velocities normalised by 5 rad/s. All 22 DOF."""
    asset = env.scene["robot"]
    return (asset.data.joint_vel / 5.0).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# 3-step temporal stacking observation (sharpa-rl-lab style)
# ---------------------------------------------------------------------------

def sharpa_observation_temporal(env) -> torch.Tensor:
    """
    Sharpa-style 64-dim block × 3 temporal steps = 192 dims.

    Block: joint_pos(22) + joint_targets(22) + tactile(5) + contact_pos(15)
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    q_low = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]

    # Add joint noise (sharpa default 0.02)
    hand_cfg = _get_hand_cfg(env)
    noise_scale = float(hand_cfg.get("joint_noise_scale", 0.02))
    noise = (torch.rand_like(q) * 2.0 - 1.0) * noise_scale
    q_noisy = _unscale(q + noise, q_low, q_high).clamp(-1.0, 1.0)

    # Current targets
    cur_targets = env.extras.get("current_action")
    if cur_targets is None:
        cur_targets = asset.data.joint_pos.clone()

    # Tactile
    sensed_contacts, contact_pos = _compute_sharpa_tactile(env)

    # Build current 64-dim block
    cur_block = torch.cat([
        q_noisy,                       # 22
        cur_targets,                   # 22
        sensed_contacts,               # 5
        contact_pos,                   # 15
    ], dim=-1)  # (N, 64)

    # Temporal buffer (sliding window of 80 frames, use last 3)
    buf_key = "_obs_temporal_buf"
    if buf_key not in env.extras:
        env.extras[buf_key] = cur_block.unsqueeze(1).repeat(1, 80, 1)

    buf = env.extras[buf_key]
    # Shift and append
    buf[:, :-1] = buf[:, 1:].clone()
    buf[:, -1] = cur_block

    # Reset buffer for newly reset envs
    reset_key = "_obs_at_reset"
    if reset_key in env.extras:
        reset_ids = env.extras[reset_key].nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            buf[reset_ids] = cur_block[reset_ids].unsqueeze(1).expand(-1, 80, -1)
            env.extras[reset_key][reset_ids] = 0

    # Last 3 frames → flatten
    obs_temporal = buf[:, -3:].reshape(env.num_envs, -1)  # (N, 192)
    return obs_temporal


# ---------------------------------------------------------------------------
# Target object info (DexGen — non-temporal, appended once)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Standalone observation functions (used by env config ObsTerm)
# ---------------------------------------------------------------------------

def _to_hand_frame_pos(env, points_w):
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    rel = points_w - root_pos
    return quat_apply_inverse(root_quat, rel)


def _to_hand_frame_vec(env, vec_w):
    robot = env.scene["robot"]
    root_quat = robot.data.root_quat_w
    return quat_apply_inverse(root_quat, vec_w)


def object_pos_in_hand_frame(env) -> torch.Tensor:
    return _to_hand_frame_pos(env, env.scene["object"].data.root_pos_w)


def object_quat_in_hand_frame(env) -> torch.Tensor:
    robot = env.scene["robot"]
    obj_quat = env.scene["object"].data.root_quat_w
    return _quat_multiply(_quat_conjugate(robot.data.root_quat_w), obj_quat)


def object_lin_vel_hand_frame(env) -> torch.Tensor:
    return _to_hand_frame_vec(env, env.scene["object"].data.root_lin_vel_w)


def object_ang_vel_hand_frame(env) -> torch.Tensor:
    return _to_hand_frame_vec(env, env.scene["object"].data.root_ang_vel_w)


def fingertip_contact_binary(env) -> torch.Tensor:
    """Binary contact per fingertip (force > 0.5N). Returns: (N, 5)"""
    sensed, _ = _compute_sharpa_tactile(env)
    return (sensed > 0.5).float()


def fingertip_contact_forces(env) -> torch.Tensor:
    """3-D contact force per fingertip in hand frame, normalised. Returns: (N, 15)"""
    sensor_keys = [
        "fingertip_contact_sensor_thumb",
        "fingertip_contact_sensor_index",
        "fingertip_contact_sensor_middle",
        "fingertip_contact_sensor_ring",
        "fingertip_contact_sensor_pinky",
    ]
    per_tip = []
    for key in sensor_keys:
        sensor = env.scene.sensors.get(key)
        if sensor is None:
            per_tip.append(torch.zeros(env.num_envs, 3, device=env.device))
            continue
        fm = getattr(sensor.data, "force_matrix_w", None)
        if fm is not None and fm.numel() > 0:
            per_tip.append(fm[:, 0, 0, :])
        else:
            h = sensor.data.net_forces_w_history
            per_tip.append(h[:, 0, 0, :] if h.ndim == 4 else h[:, -1, :])

    forces = torch.stack(per_tip, dim=1)  # (N, 5, 3)
    forces = _to_hand_frame_vec_batch(env, forces)
    return (forces / 10.0).clamp(-3.0, 3.0).reshape(env.num_envs, -1)


def last_action(env) -> torch.Tensor:
    action = env.extras.get("last_action")
    if action is None:
        return torch.zeros(env.num_envs, 22, device=env.device)
    return action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hand_cfg(env) -> dict:
    return getattr(env.cfg, "hand", None) or {}


def _to_hand_frame_vec_batch(env, vecs_w):
    """Rotate (N, K, 3) vectors to hand frame."""
    robot = env.scene["robot"]
    q_inv = _quat_conjugate(robot.data.root_quat_w)
    # Expand quat for batch: (N, 1, 4) → (N, K, 4)
    q_exp = q_inv.unsqueeze(1).expand(-1, vecs_w.shape[1], -1)
    # Rotate each vector
    from isaaclab.utils.math import quat_apply
    return quat_apply(q_exp.reshape(-1, 4), vecs_w.reshape(-1, 3)).reshape_as(vecs_w)
