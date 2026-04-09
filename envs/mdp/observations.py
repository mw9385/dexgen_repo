"""
Observation functions — Sharpa Wave Hand.

Tactile sensing and temporal stacking copied verbatim from sharpa-rl-lab
(sharpa_wave_env.py compute_observations).

  Per-step block (64 dims):
    joint_pos_normalized    22   (unscale to [-1, 1] + noise)
    joint_targets           22   (current position targets)
    tactile_forces           5   (smoothed contact force magnitude)
    contact_positions       15   (5 fingers × 3, in tactile frame)

  3-step temporal stacking: 64 × 3 = 192 dims

  + Target info (DexGen, non-temporal):
    target_object_pos_hand   3
    target_object_quat_hand  4

  Total: 192 + 7 = 199 dims
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_apply

from .math_utils import quat_conjugate as _quat_conjugate
from .math_utils import quat_multiply as _quat_multiply


# ---------------------------------------------------------------------------
# Helpers from sharpa-rl-lab (verbatim)
# ---------------------------------------------------------------------------

def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def _transform_between_frames(p_A, q_A, q_B):
    """Transform point from frame A to frame B (rotation only)."""
    p_world = quat_apply(q_A, p_A)
    p_B = quat_apply(quat_inv(q_B), p_world)
    return p_B


# ---------------------------------------------------------------------------
# Init helper — must be called once from env setup
# ---------------------------------------------------------------------------

def init_sharpa_obs_buffers(env):
    """
    Initialise observation buffers. Call once after env creation.
    Sets up the same buffers as sharpa_wave_env.__init__.
    """
    N = env.num_envs
    device = env.device
    hand = env.scene["robot"]

    env.extras["_obs_lag_history"] = torch.zeros(
        N, 80, 64, device=device, dtype=torch.float,
    )
    env.extras["_at_reset_buf"] = torch.ones(N, device=device, dtype=torch.long)
    env.extras["_last_contacts"] = torch.zeros(N, 5, device=device, dtype=torch.float)

    # Elastomer body IDs for contact position transform
    elastomer_names = [
        "right_thumb_elastomer", "right_index_elastomer",
        "right_middle_elastomer", "right_ring_elastomer",
        "right_pinky_elastomer",
    ]
    env.extras["_elastomer_ids"] = [
        hand.body_names.index(n) for n in elastomer_names
    ]

    # Contact sensor list (indices 0-4 = elastomer, matching sharpa _contact_body_ids)
    env.extras["_contact_body_ids"] = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    hand_cfg = _get_hand_cfg(env)
    disable_ids = hand_cfg.get("disable_tactile_ids", [])
    env.extras["_contact_body_ids_disable"] = torch.tensor(disable_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Sharpa tactile + temporal observation (verbatim from compute_observations)
# ---------------------------------------------------------------------------

def sharpa_observation_temporal(env) -> torch.Tensor:
    """
    Sharpa-style 64-dim × 3 temporal = 192 dims.
    Copied from sharpa_wave_env.py compute_observations.
    """
    hand = env.scene["robot"]
    device = env.device
    N = env.num_envs

    hand_cfg = _get_hand_cfg(env)

    # Lazy init
    if "_obs_lag_history" not in env.extras:
        init_sharpa_obs_buffers(env)

    obs_buf_lag_history = env.extras["_obs_lag_history"]
    at_reset_buf = env.extras["_at_reset_buf"]
    last_contacts = env.extras["_last_contacts"]
    elastomer_ids = env.extras["_elastomer_ids"]
    contact_body_ids = env.extras["_contact_body_ids"]
    contact_body_ids_disable = env.extras["_contact_body_ids_disable"]

    # Config values
    contact_smooth = float(hand_cfg.get("contact_smooth", 0.5))
    binary_contact = bool(hand_cfg.get("binary_contact", False))
    contact_threshold = float(hand_cfg.get("contact_threshold", 0.05))
    contact_latency = float(hand_cfg.get("contact_latency", 0.005))
    contact_sensor_noise = float(hand_cfg.get("contact_sensor_noise", 0.01))
    enable_contact_pos = bool(hand_cfg.get("enable_contact_pos", False))
    enable_tactile = bool(hand_cfg.get("enable_tactile", True))
    joint_noise_scale = float(hand_cfg.get("joint_noise_scale", 0.02))

    hand_dof_pos = hand.data.joint_pos
    hand_dof_lower = hand.data.soft_joint_pos_limits[..., 0]
    hand_dof_upper = hand.data.soft_joint_pos_limits[..., 1]

    # ── Contact forces (sharpa verbatim) ──────────────────────
    # Tactile frame for contact position transform
    tactile_frame_pose = hand.data.body_link_state_w[:, elastomer_ids, :7]
    tactile_frame_pos = tactile_frame_pose[..., :3]
    tactile_frame_quat = tactile_frame_pose[..., 3:7]
    world_quat = torch.zeros_like(tactile_frame_quat)
    world_quat[..., 0] = 1.0

    # Collect net forces history from elastomer contact sensors (indices 0-4)
    sensor_list = [env.scene.sensors.get(f"contact_{i}") for i in contact_body_ids]
    net_contact_forces_history = torch.cat(
        [s.data.net_forces_w_history[:, :, 0, :].unsqueeze(2) for s in sensor_list if s is not None],
        dim=2,
    )  # (N, H, 5, 3)
    norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)  # (N, H, 5)

    # Smoothing
    smooth_contact_forces = (
        norm_contact_forces_history[:, 0, :] * contact_smooth
        + norm_contact_forces_history[:, 1, :] * (1 - contact_smooth)
    )
    if len(contact_body_ids_disable) > 0:
        smooth_contact_forces[:, contact_body_ids_disable] = 0.0

    # Binary or continuous contact
    if binary_contact:
        binary_contacts = torch.where(
            smooth_contact_forces > contact_threshold, 1.0, 0.0,
        )
        latency_samples = torch.rand_like(last_contacts)
        latency = torch.where(latency_samples < contact_latency, 1.0, 0.0)
        last_contacts[:] = last_contacts * latency + binary_contacts * (1 - latency)
        mask = torch.rand_like(last_contacts)
        mask = torch.where(mask < contact_sensor_noise, 0.0, 1.0)
        sensed_contacts = torch.where(
            last_contacts > 0.1, mask * last_contacts, last_contacts,
        )
    else:
        latency_samples = torch.rand_like(last_contacts)
        latency = torch.where(latency_samples < contact_latency, 1.0, 0.0)
        last_contacts[:] = last_contacts * latency + smooth_contact_forces * (1 - latency)
        sensed_contacts = last_contacts.clone()

    env.extras["_last_contacts"] = last_contacts

    # ── Contact positions (sharpa verbatim) ───────────────────
    not_contact_mask = sensed_contacts < 1.0e-6
    if len(contact_body_ids_disable) > 0:
        not_contact_mask[:, contact_body_ids_disable] = True
    contact_mask = ~not_contact_mask

    contact_pos = torch.cat(
        [s.data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for s in sensor_list if s is not None],
        dim=1,
    )  # (N, 5, 3)
    contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
    contact_pos[contact_mask, :] = _transform_between_frames(
        contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :],
        world_quat[contact_mask, :],
        tactile_frame_quat[contact_mask, :],
    )
    contact_pos[not_contact_mask, :] = 0.0
    contact_pos = contact_pos.reshape(N, -1)  # (N, 15)

    if not enable_contact_pos:
        contact_pos[:] = 0.0
    if not enable_tactile:
        contact_pos[:] = 0.0
        sensed_contacts[:] = 0.0

    # ── Sliding window observation (sharpa verbatim) ──────────
    prev_obs_buf = obs_buf_lag_history[:, 1:].clone()

    joint_noise_matrix = (
        torch.rand(hand_dof_pos.shape, device=device) * 2.0 - 1.0
    ) * joint_noise_scale

    cur_obs_buf = _unscale(
        joint_noise_matrix + hand_dof_pos,
        hand_dof_lower, hand_dof_upper,
    ).clone().unsqueeze(1)  # (N, 1, 22)

    cur_targets = env.extras.get("current_action")
    if cur_targets is None:
        cur_targets = hand_dof_pos.clone()
    cur_tar_buf = cur_targets[:, None]  # (N, 1, 22)

    cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # (N, 1, 44)
    cur_obs_buf = torch.cat([
        cur_obs_buf,
        sensed_contacts.clone().unsqueeze(1),   # (N, 1, 5)
        contact_pos.clone().unsqueeze(1),        # (N, 1, 15)
    ], dim=-1)  # (N, 1, 64)

    obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

    # Refill initialized buffers (sharpa verbatim)
    at_reset_env_ids = at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(at_reset_env_ids) > 0:
        obs_buf_lag_history[at_reset_env_ids, :, 0:22] = _unscale(
            hand_dof_pos[at_reset_env_ids],
            hand_dof_lower[at_reset_env_ids],
            hand_dof_upper[at_reset_env_ids],
        ).clone().unsqueeze(1)
        obs_buf_lag_history[at_reset_env_ids, :, 22:44] = hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        obs_buf_lag_history[at_reset_env_ids, :, 44:49] = sensed_contacts[at_reset_env_ids].unsqueeze(1)
        obs_buf_lag_history[at_reset_env_ids, :, 49:64] = contact_pos[at_reset_env_ids].unsqueeze(1)
        at_reset_buf[at_reset_env_ids] = 0

    env.extras["_obs_lag_history"] = obs_buf_lag_history
    env.extras["_at_reset_buf"] = at_reset_buf

    obs_buf = obs_buf_lag_history[:, -3:].reshape(N, -1).clone()  # (N, 192)
    return obs_buf


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
# Standalone functions (used by other modules)
# ---------------------------------------------------------------------------

def joint_positions_normalized(env) -> torch.Tensor:
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    q_low = asset.data.soft_joint_pos_limits[..., 0]
    q_high = asset.data.soft_joint_pos_limits[..., 1]
    return _unscale(q, q_low, q_high).clamp(-1.0, 1.0)


def joint_velocities_normalized(env) -> torch.Tensor:
    asset = env.scene["robot"]
    return (asset.data.joint_vel / 5.0).clamp(-1.0, 1.0)


def fingertip_contact_binary(env) -> torch.Tensor:
    """Binary contact per fingertip (force > 0.5N). Returns: (N, 5)"""
    if "_last_contacts" not in env.extras:
        init_sharpa_obs_buffers(env)
    return (env.extras["_last_contacts"] > 0.5).float()


def last_action(env) -> torch.Tensor:
    action = env.extras.get("last_action")
    if action is None:
        return torch.zeros(env.num_envs, 22, device=env.device)
    return action


def object_pos_in_hand_frame(env) -> torch.Tensor:
    robot = env.scene["robot"]
    obj_pos = env.scene["object"].data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, obj_pos - robot.data.root_pos_w)


def object_quat_in_hand_frame(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return _quat_multiply(_quat_conjugate(robot.data.root_quat_w),
                           env.scene["object"].data.root_quat_w)


def object_lin_vel_hand_frame(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w,
                               env.scene["object"].data.root_lin_vel_w)


def object_ang_vel_hand_frame(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w,
                               env.scene["object"].data.root_ang_vel_w)


def fingertip_contact_forces(env) -> torch.Tensor:
    """3-D contact force per fingertip in hand frame. Returns: (N, 15)"""
    sensor_list = [env.scene.sensors.get(f"contact_{i}") for i in range(5)]
    per_tip = []
    for s in sensor_list:
        if s is None:
            per_tip.append(torch.zeros(env.num_envs, 3, device=env.device))
            continue
        fm = getattr(s.data, "force_matrix_w", None)
        if fm is not None and fm.numel() > 0:
            per_tip.append(fm[:, 0, 0, :])
        else:
            per_tip.append(s.data.net_forces_w_history[:, 0, 0, :])
    forces = torch.stack(per_tip, dim=1)
    robot = env.scene["robot"]
    q_inv = _quat_conjugate(robot.data.root_quat_w)
    q_exp = q_inv.unsqueeze(1).expand(-1, 5, -1)
    forces_hand = quat_apply(q_exp.reshape(-1, 4), forces.reshape(-1, 3)).reshape(env.num_envs, 5, 3)
    return (forces_hand / 10.0).clamp(-3.0, 3.0).reshape(env.num_envs, -1)


def domain_randomization_params(env) -> torch.Tensor:
    params = env.extras.get("dr_params")
    if params is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hand_cfg(env) -> dict:
    return getattr(env.cfg, "hand", None) or {}
