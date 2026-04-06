"""
Isaac Sim robot/object manipulation utilities.

Shared infrastructure for RL reset (events.py) and graph validation scripts.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import numpy as np

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .math_utils import (
    quat_multiply,
    quat_conjugate,
    quat_from_two_vectors,
    add_rotation_noise,
    add_tilt_noise,
    local_to_world_points,
    world_to_local_points,
    solve_rigid_alignment,
    solve_object_pose_from_contacts,
)


# Module-level caches
_FT_IDS_CACHE: dict = {}
_PALM_NORMAL_CACHE: dict = {}

def pad_fingertip_positions(fps: np.ndarray, target_nf: int) -> np.ndarray:
    """
    Pad or truncate (num_fingers, 3) array to (target_nf, 3).

    Extra fingers (added by padding) are set to the last valid finger's
    position so they have a neutral target and contribute no gradient.
    """
    nf = fps.shape[0]
    if nf == target_nf:
        return fps
    if nf > target_nf:
        return fps[:target_nf]
    # pad by repeating last row
    pad = np.tile(fps[-1:], (target_nf - nf, 1))
    return np.concatenate([fps, pad], axis=0).astype(np.float32)


def set_robot_joints_direct(
    env,
    env_ids: torch.Tensor,
    joints_list: list,
):
    """
    Set robot joints directly from stored joint angles in the Grasp object.
    This is the correct approach when grasp generation also solves IK.
    """
    robot = env.scene["robot"]
    n = len(env_ids)
    num_dof = robot.data.default_joint_pos.shape[-1]

    joint_pos = robot.data.default_joint_pos[env_ids].clone()

    for i, joints in enumerate(joints_list):
        if joints is not None:
            joint_pos[i] = expand_grasp_joint_vector(
                torch.tensor(joints, device=env.device, dtype=torch.float32),
                num_dof,
            )

    robot.write_joint_state_to_sim(
        joint_pos,
        torch.zeros_like(joint_pos),
        env_ids=env_ids,
    )
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def expand_grasp_joint_vector(
    joint_vec: torch.Tensor,
    target_num_dof: int,
) -> torch.Tensor:
    """
    Expand/pad a stored grasp joint vector to the full 24-DOF articulation count.

    Shadow Hand USD (24 DOF):
      [0-1]:   WRJ1, WRJ0           (wrist)
      [2-5]:   FFJ4(passive), FFJ3, FFJ2, FFJ1
      [6-9]:   MFJ4(passive), MFJ3, MFJ2, MFJ1
      [10-13]: RFJ4(passive), RFJ3, RFJ2, RFJ1
      [14-18]: LFJ5(passive), LFJ4, LFJ3, LFJ2, LFJ1
      [19-23]: THJ4, THJ3, THJ2, THJ1, THJ0

    Supported input sizes:
      24 → pass-through (already in Isaac Sim format)
      22 → DexGraspNet MJCF format (shadow_hand_wrist_free.xml):
             [0-3]:   FFJ3, FFJ2, FFJ1, FFJ0(coupled)
             [4-7]:   MFJ3, MFJ2, MFJ1, MFJ0(coupled)
             [8-11]:  RFJ3, RFJ2, RFJ1, RFJ0(coupled)
             [12-16]: LFJ4, LFJ3, LFJ2, LFJ1, LFJ0(coupled)
             [17-21]: THJ4, THJ3, THJ2, THJ1, THJ0
           MJCF has J0 (coupled DIP) not in Isaac → skip.
           Isaac has J4/J5 (passive spread) not in MJCF → zero.
    """
    if joint_vec.shape[0] == target_num_dof:
        return joint_vec

    expanded = torch.zeros(target_num_dof, device=joint_vec.device, dtype=joint_vec.dtype)

    if joint_vec.shape[0] == target_num_dof - 2:  # 22-DOF MJCF format
        # FF: in[0:3]=(FFJ3,2,1) → out[3:6], skip in[3]=FFJ0
        expanded[3:6] = joint_vec[0:3]
        # MF: in[4:7]=(MFJ3,2,1) → out[7:10], skip in[7]=MFJ0
        expanded[7:10] = joint_vec[4:7]
        # RF: in[8:11]=(RFJ3,2,1) → out[11:14], skip in[11]=RFJ0
        expanded[11:14] = joint_vec[8:11]
        # LF: in[12:16]=(LFJ4,3,2,1) → out[15:19], skip in[16]=LFJ0
        expanded[15:19] = joint_vec[12:16]
        # TH: in[17:22]=(THJ4,3,2,1,0) → out[19:24] (direct match)
        expanded[19:24] = joint_vec[17:22]
        return expanded

    # Fallback: zero-pad or truncate
    n_copy = min(target_num_dof, joint_vec.shape[0])
    expanded[:n_copy] = joint_vec[:n_copy]
    return expanded


def sample_wrist_pose_world(
    env,
    env_ids: torch.Tensor,
    apply_noise: bool = True,
):
    robot = env.scene["robot"]
    n = len(env_ids)
    cfg = getattr(env.cfg, "reset_randomization", {}) or {}

    # default_root_state is in LOCAL env frame.  Add env origin to get world frame.
    root_local = robot.data.default_root_state[env_ids, :7].clone()
    wrist_pos = root_local[:, :3] + env.scene.env_origins[env_ids]
    wrist_quat = root_local[:, 3:7]

    pos_jitter_std = float(cfg.get("wrist_pos_jitter_std", 0.005))
    if apply_noise and pos_jitter_std > 0.0:
        wrist_pos = wrist_pos + torch.randn(n, 3, device=env.device) * pos_jitter_std

    if apply_noise and bool(cfg.get("align_palm_up", False)):
        wrist_quat = align_wrist_palm_down(env, env_ids, wrist_quat)

    rot_std = math.radians(float(cfg.get("wrist_rot_std_deg", 5.0)))
    if apply_noise and rot_std > 0.0:
        wrist_quat = add_rotation_noise(wrist_quat, rot_std, env.device, n)

    return wrist_pos, wrist_quat


def set_robot_root_pose(
    env,
    env_ids: torch.Tensor,
    wrist_pos_w: torch.Tensor,
    wrist_quat_w: torch.Tensor,
):
    robot = env.scene["robot"]
    root_pose = robot.data.default_root_state[env_ids, :7].clone()
    root_pose[:, :3] = wrist_pos_w
    root_pose[:, 3:7] = wrist_quat_w / (torch.norm(wrist_quat_w, dim=-1, keepdim=True) + 1e-8)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)


def align_wrist_palm_down(env, env_ids: torch.Tensor, wrist_quat: torch.Tensor) -> torch.Tensor:
    """Align palm normal to world -Z (downward). Hand grasps from above."""
    robot = env.scene["robot"]
    palm_normal_local = get_local_palm_normal(robot, env).unsqueeze(0).expand(len(env_ids), 3)
    palm_normal_world = quat_apply(wrist_quat, palm_normal_local)
    target_down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(palm_normal_world)
    correction = quat_from_two_vectors(palm_normal_world, target_down)
    return quat_multiply(correction, wrist_quat)


def align_wrist_palm_up(env, env_ids: torch.Tensor, wrist_quat: torch.Tensor) -> torch.Tensor:
    """Align palm normal to world +Z (upward). Object rests on palm."""
    robot = env.scene["robot"]
    palm_normal_local = get_local_palm_normal(robot, env).unsqueeze(0).expand(len(env_ids), 3)
    palm_normal_world = quat_apply(wrist_quat, palm_normal_local)
    target_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(palm_normal_world)
    correction = quat_from_two_vectors(palm_normal_world, target_up)
    return quat_multiply(correction, wrist_quat)


def get_local_palm_normal(robot, env) -> torch.Tensor:
    key = id(robot)
    if key not in _PALM_NORMAL_CACHE:
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        if hand_cfg.get("name") == "shadow":
            base_names = ["robot0_ffknuckle", "robot0_rfknuckle", "robot0_thbase"]
        else:
            base_names = ["index_link_0", "ring_link_0", "thumb_link_0"]
        body_ids = [robot.find_bodies(name)[0][0] for name in base_names]
        pts_world = robot.data.body_pos_w[0:1, body_ids, :]
        root_pos = robot.data.root_pos_w[0:1]
        root_quat = robot.data.root_quat_w[0:1]
        pts_local = world_to_local_points(pts_world, root_pos, root_quat)[0]
        v1 = pts_local[0] - pts_local[2]
        v2 = pts_local[1] - pts_local[2]
        normal = torch.cross(v1, v2, dim=-1)
        normal = normal / (torch.norm(normal) + 1e-8)
        _PALM_NORMAL_CACHE[key] = normal
    return _PALM_NORMAL_CACHE[key]


def get_palm_body_id_from_env(robot, env) -> int:
    key = ("palm_body", id(robot))
    cached = _FT_IDS_CACHE.get(key)
    if cached is not None:
        return cached

    hand_cfg = getattr(env.cfg, "hand", None) or {}
    if hand_cfg.get("name") == "shadow":
        candidate_names = ["robot0_palm", "robot0:palm", "palm"]
    else:
        candidate_names = ["palm_link", "base_link", "palm"]

    for name in candidate_names:
        try:
            body_ids = robot.find_bodies(name)[0]
        except Exception:
            continue
        if len(body_ids) > 0:
            _FT_IDS_CACHE[key] = int(body_ids[0])
            return _FT_IDS_CACHE[key]

    raise RuntimeError(
        f"Could not resolve palm body for hand={hand_cfg.get('name', 'unknown')}; "
        f"tried {candidate_names}"
    )


def place_object_fixed(
    env,
    env_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Place the object above the palm center.

    Reads the actual palm body position from the simulator and places
    the object slightly above it. No heuristic offsets.

    Returns (obj_pos_w, obj_quat_w) for later frame transforms.
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Get actual palm position from sim
    palm_body_id = get_palm_body_id_from_env(robot, env)
    palm_pos_w = robot.data.body_pos_w[env_ids, palm_body_id, :].clone()  # (n, 3)

    # Place object slightly above palm center (~3cm)
    obj_pos = palm_pos_w.clone()
    obj_pos[:, 2] += 0.03

    obj_quat = torch.zeros(len(env_ids), 4, device=env.device)
    obj_quat[:, 0] = 1.0    # identity quaternion

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = obj_pos
    root_state[:, 3:7] = obj_quat
    root_state[:, 7:] = 0.0   # zero velocity
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)

    return obj_pos, obj_quat


def compute_wrist_from_fingertips(
    env,
    env_ids: torch.Tensor,
    target_world: torch.Tensor,   # (n, F, 3)  — unused, kept for API compat
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the default wrist pose. No heuristic computation.

    The hand is already in a reasonable default pose (palm up).
    IK refinement will adjust finger joints to reach the targets.
    """
    robot = env.scene["robot"]

    wrist_pos = (
        robot.data.default_root_state[env_ids, :3].clone()
        + env.scene.env_origins[env_ids]
    )
    wrist_quat = robot.data.default_root_state[env_ids, 3:7].clone()

    return wrist_pos, wrist_quat


def set_adaptive_joint_pose(
    env,
    env_ids: torch.Tensor,
    object_size: float,
):
    """
    Set hand joints to joint-limit midpoint as IK initial configuration.

    This gives the IK solver maximum freedom to move in any direction.
    Wrist joints are kept at zero (fixed during episode).
    """
    robot = env.scene["robot"]

    q_low = robot.data.soft_joint_pos_limits[env_ids, :, 0]
    q_high = robot.data.soft_joint_pos_limits[env_ids, :, 1]
    joint_pos = (q_low + q_high) / 2.0

    # Wrist joints stay at zero
    joint_pos[:, :2] = 0.0

    robot.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids,
    )
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def apply_palm_up_transform(
    env,
    env_ids: torch.Tensor,
    goal_world: torch.Tensor,   # (n, F, 3) goal fingertips in world frame
) -> torch.Tensor:
    """
    Rotate the entire hand+object system so the palm faces upward (+Z).

    After differential IK, the hand may be in an arbitrary orientation.
    This function:
      1. Computes the correction rotation (palm normal → +Z)
      2. Rotates wrist root pose around the fingertip centroid
      3. Rotates the object with the same rotation
      4. Returns the transformed goal fingertip world positions

    Joint angles are NOT changed — only the root pose is rotated,
    so FK automatically places fingertips in the correct rotated positions.
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    n = len(env_ids)

    # Current wrist state
    wrist_pos = robot.data.root_pos_w[env_ids].clone()
    wrist_quat = robot.data.root_quat_w[env_ids].clone()

    # Current palm normal in world frame
    palm_normal_local = get_local_palm_normal(robot, env).unsqueeze(0).expand(n, 3)
    palm_normal_world = quat_apply(wrist_quat, palm_normal_local)

    # Correction: rotate palm normal to point +Z
    target_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3)
    correction = quat_from_two_vectors(palm_normal_world, target_up)

    # Pivot = fingertip centroid (current actual positions from FK)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    ft_world = robot.data.body_pos_w[env_ids][:, ft_ids, :]
    pivot = ft_world.mean(dim=1)   # (n, 3)

    # Rotate wrist around pivot
    new_wrist_quat = quat_multiply(correction, wrist_quat)
    new_wrist_quat = new_wrist_quat / (torch.norm(new_wrist_quat, dim=-1, keepdim=True) + 1e-8)
    wrist_rel = wrist_pos - pivot
    new_wrist_pos = quat_apply(correction, wrist_rel) + pivot

    # Rotate object around same pivot
    obj_pos = obj.data.root_pos_w[env_ids].clone()
    obj_quat = obj.data.root_quat_w[env_ids].clone()
    obj_rel = obj_pos - pivot
    new_obj_pos = quat_apply(correction, obj_rel) + pivot
    new_obj_quat = quat_multiply(correction, obj_quat)
    new_obj_quat = new_obj_quat / (torch.norm(new_obj_quat, dim=-1, keepdim=True) + 1e-8)

    # Rotate goal fingertip world positions around same pivot
    goal_rel = goal_world - pivot.unsqueeze(1)   # (n, F, 3)
    correction_expanded = correction.unsqueeze(1).expand(-1, goal_world.shape[1], -1)
    new_goal_world = quat_apply(
        correction_expanded.reshape(-1, 4), goal_rel.reshape(-1, 3)
    ).reshape_as(goal_world) + pivot.unsqueeze(1)

    # Write to sim
    set_robot_root_pose(env, env_ids, new_wrist_pos, new_wrist_quat)

    obj_root_state = obj.data.default_root_state[env_ids].clone()
    obj_root_state[:, :3] = new_obj_pos
    obj_root_state[:, 3:7] = new_obj_quat
    obj_root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(obj_root_state, env_ids=env_ids)

    # FK update with new root pose (joint angles unchanged)
    robot.update(0.0)
    obj.update(0.0)

    return new_goal_world


def place_object_in_hand(
    env,
    env_ids: torch.Tensor,
    fingertip_positions_obj: torch.Tensor,   # (n, F, 3) in object frame
):
    """
    Place the object so the sampled grasp-set fingertip positions line up
    with the current fingertip world positions of the hand.

    This makes reset start from the Stage 0 grasp itself instead of from an
    unrelated hand/object arrangement.
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]

    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    ft_world = robot.data.body_pos_w[env_ids][:, ft_ids, :].clone()  # (n, F, 3)

    full_pos_w, full_quat_w = solve_rigid_alignment(
        points_src=fingertip_positions_obj,
        points_dst=ft_world,
    )
    full_reconstructed_world = local_to_world_points(
        fingertip_positions_obj, full_pos_w, full_quat_w
    )
    full_solve_err = torch.norm(full_reconstructed_world - ft_world, dim=-1)

    pos_w, quat_w = solve_object_pose_from_contacts(
        points_obj=fingertip_positions_obj,
        points_world=ft_world,
    )
    reconstructed_world = local_to_world_points(fingertip_positions_obj, pos_w, quat_w)
    solve_err = torch.norm(reconstructed_world - ft_world, dim=-1)

    pose_pos_delta = torch.norm(pos_w - full_pos_w, dim=-1)
    pose_quat_dot = (quat_w * full_quat_w).sum(dim=-1).abs().clamp(0.0, 1.0)
    pose_rot_delta = 2.0 * torch.arccos(pose_quat_dot)

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = pos_w
    root_state[:, 3:7] = quat_w
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)
    placement_debug = {
        "chosen_mean_err": solve_err.mean(dim=-1),
        "chosen_max_err": solve_err.max(dim=-1).values,
        "full_mean_err": full_solve_err.mean(dim=-1),
        "full_max_err": full_solve_err.max(dim=-1).values,
        "pose_pos_delta": pose_pos_delta,
        "pose_rot_delta": pose_rot_delta,
    }
    return solve_err.mean(dim=-1), solve_err.max(dim=-1).values, placement_debug


def refine_hand_to_start_grasp(
    env,
    env_ids: torch.Tensor,
    start_fps_obj: torch.Tensor,   # (n, F, 3)
):
    """
    Per-finger differential IK with null-space projection.

    Each finger is solved independently as a 4-5 DOF serial chain targeting
    a 3D fingertip position. This avoids interference between fingers that
    occurs when stacking all fingertip Jacobians into one system.

    Null-space projection (Liegeois 1977) pushes joints toward the midpoint
    of their limits, preventing reverse-joint and extreme poses.

    SVD-based fallback (lstsq) handles singular/near-singular Jacobians
    robustly when cuSOLVER fails.

    References:
      - Maciejewski & Klein (1988), SVD for robotics
      - Siciliano & Khatib, Springer Handbook of Robotics (null-space)
    """
    cfg = getattr(env.cfg, "reset_refinement", {}) or {}
    if not bool(cfg.get("enabled", True)):
        return

    iterations = int(cfg.get("iterations", 200))
    if iterations <= 0:
        return

    robot = env.scene["robot"]
    obj = env.scene["object"]
    n = len(env_ids)
    device = env.device

    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    jac_body_ids = [
        body_id - 1 if robot.is_fixed_base else body_id
        for body_id in ft_ids
    ]
    joint_count = robot.data.joint_pos.shape[-1]

    step_gain = float(cfg.get("step_gain", 0.8))
    damping = float(cfg.get("damping", 0.05))
    max_delta = float(cfg.get("max_delta", 0.2))
    pos_threshold = float(cfg.get("pos_threshold", 0.005))
    null_space_gain = float(cfg.get("null_space_gain", 0.1))

    # Floating-base Jacobian offset: PhysX prepends 6 columns for base DOFs
    jac_col_offset = 0 if robot.is_fixed_base else 6

    joint_lower = robot.data.soft_joint_pos_limits[env_ids, :, 0]
    joint_upper = robot.data.soft_joint_pos_limits[env_ids, :, 1]
    q_rest = (joint_lower + joint_upper) / 2.0

    target_obj_state = obj.data.root_state_w[env_ids].clone()

    # Shadow Hand 24-DOF USD finger→joint mapping.
    # Wrist joints [0,1] are excluded (not modified by IK).
    _FINGER_JOINT_IDS = [
        [2, 3, 4, 5],          # FF: FFJ4(passive), FFJ3, FFJ2, FFJ1
        [6, 7, 8, 9],          # MF: MFJ4(passive), MFJ3, MFJ2, MFJ1
        [10, 11, 12, 13],      # RF: RFJ4(passive), RFJ3, RFJ2, RFJ1
        [14, 15, 16, 17, 18],  # LF: LFJ5(passive), LFJ4, LFJ3, LFJ2, LFJ1
        [19, 20, 21, 22, 23],  # TH: THJ4, THJ3, THJ2, THJ1, THJ0
    ]
    num_fingers = min(len(ft_ids), len(_FINGER_JOINT_IDS))

    _debug_ik = bool(os.environ.get("DEBUG_IK", ""))

    for _iter in range(iterations):
        # Kinematics-only update (no physics step)
        robot.write_data_to_sim()
        robot.update(0.0)
        obj.update(0.0)

        # Keep object fixed during IK
        obj.write_root_state_to_sim(target_obj_state, env_ids=env_ids)
        obj.update(0.0)

        # Compute fingertip targets in world frame
        target_world = local_to_world_points(
            start_fps_obj,
            target_obj_state[:, :3],
            target_obj_state[:, 3:7],
        )
        current_world = robot.data.body_pos_w[env_ids][:, ft_ids, :]
        pos_error = target_world - current_world   # (n, F, 3)
        per_finger_err = torch.norm(pos_error, dim=-1)  # (n, F)
        mean_err = per_finger_err.mean(dim=-1)
        if _debug_ik and _iter % 10 == 0:
            print(f"    [IK iter {_iter:3d}] mean_err={float(mean_err[0]):.4f}m  "
                  f"per_finger={[f'{float(e):.4f}' for e in per_finger_err[0]]}")
        if torch.all(mean_err <= pos_threshold):
            if _debug_ik:
                print(f"    [IK] Converged at iter {_iter}")
            break

        # Full Jacobian from PhysX: (n, num_bodies, 6, num_joints+base_dofs)
        full_jac = robot.root_physx_view.get_jacobians()[env_ids]
        if _debug_ik and _iter == 0:
            print(f"    [IK] Jacobian shape={tuple(full_jac.shape)}, "
                  f"jac_col_offset={jac_col_offset}")

        # Accumulate per-joint deltas from all fingers
        joint_pos = robot.data.joint_pos[env_ids].clone()
        delta_all = torch.zeros_like(joint_pos)

        for fi in range(num_fingers):
            finger_joint_ids = _FINGER_JOINT_IDS[fi]
            fj = torch.tensor(finger_joint_ids, device=device, dtype=torch.long)
            fj_jac = fj + jac_col_offset  # Jacobian column indices (offset for floating base)
            ndof = len(finger_joint_ids)

            # Extract sub-Jacobian for this finger: (n, 3, ndof)
            # fj_jac = column indices in Jacobian (accounts for base DOFs)
            # fj = joint indices in joint_pos (no offset)
            sub_jac = full_jac[:, jac_body_ids[fi], :3, :][:, :, fj_jac]  # (n, 3, ndof)

            # Fingertip error for this finger: (n, 3, 1)
            err = pos_error[:, fi, :].unsqueeze(-1)  # (n, 3, 1)

            # Damped least-squares: J^T (J J^T + λ²I)^{-1} e
            jt = sub_jac.transpose(1, 2)                    # (n, ndof, 3)
            jjt = torch.bmm(sub_jac, jt)                    # (n, 3, 3)
            eye3 = torch.eye(3, device=device, dtype=jjt.dtype).unsqueeze(0).expand(n, -1, -1)
            system = jjt + (damping ** 2) * eye3             # (n, 3, 3)

            try:
                solved = torch.linalg.solve(system, err)     # (n, 3, 1)
                # Also compute J_star for null-space projection
                system_inv_J = torch.linalg.solve(system, sub_jac)  # (n, 3, ndof)
                J_star = system_inv_J.transpose(1, 2)        # (n, ndof, 3)
            except RuntimeError:
                # SVD-based robust fallback on CPU
                solved = torch.linalg.lstsq(
                    system.cpu(), err.cpu()
                ).solution.to(device)
                system_inv_J = torch.linalg.lstsq(
                    system.cpu(), sub_jac.cpu()
                ).solution.to(device)
                J_star = system_inv_J.transpose(1, 2)

            delta_primary = torch.bmm(jt, solved).squeeze(-1)  # (n, ndof)

            # Null-space projection: (I - J⁺J) * k * (q_rest - q)
            eye_ndof = torch.eye(ndof, device=device, dtype=jt.dtype).unsqueeze(0).expand(n, -1, -1)
            null_proj = eye_ndof - torch.bmm(J_star, sub_jac)  # (n, ndof, ndof)

            q_finger = joint_pos[:, fj]                        # (n, ndof)
            q_rest_finger = q_rest[:, fj]                      # (n, ndof)
            grad_rest = null_space_gain * (q_rest_finger - q_finger)
            delta_null = torch.bmm(null_proj, grad_rest.unsqueeze(-1)).squeeze(-1)

            # Combined delta for this finger
            delta_finger = delta_primary + delta_null
            delta_finger = (step_gain * delta_finger).clamp(-max_delta, max_delta)

            # Scatter into full joint delta
            delta_all[:, fj] += delta_finger

        # Apply accumulated delta
        joint_pos = torch.clamp(joint_pos + delta_all, joint_lower, joint_upper)
        robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids,
        )
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def joint_positions_to_normalized_action(
    robot,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
) -> torch.Tensor:
    joint_lower = robot.data.soft_joint_pos_limits[env_ids, :, 0]
    joint_upper = robot.data.soft_joint_pos_limits[env_ids, :, 1]
    action = 2.0 * (joint_pos - joint_lower) / (joint_upper - joint_lower + 1e-6) - 1.0
    return action.clamp(-1.0, 1.0)


def get_fingertip_body_ids_from_env(robot, env) -> list[int]:
    key = id(robot)
    if key not in _FT_IDS_CACHE:
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        tip_names = hand_cfg.get(
            "fingertip_links",
            ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"],
        )
        _FT_IDS_CACHE[key] = [robot.find_bodies(name)[0][0] for name in tip_names]
    return _FT_IDS_CACHE[key]

