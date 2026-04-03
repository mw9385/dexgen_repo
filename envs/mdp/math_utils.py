"""
Shared torch math utilities for quaternion operations and coordinate transforms.

All quaternion functions use the (w, x, y, z) convention.
No Isaac Sim dependency — only torch and isaaclab.utils.math.
"""
from __future__ import annotations

from itertools import combinations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_matrix


# ---------------------------------------------------------------------------
# Quaternion arithmetic
# ---------------------------------------------------------------------------

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (w,x,y,z). Supports batched input."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate: (w,x,y,z) → (w,-x,-y,-z)."""
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def quat_from_two_vectors(
    v_from: torch.Tensor, v_to: torch.Tensor,
) -> torch.Tensor:
    """Quaternion that rotates unit vector *v_from* to *v_to*. Both (N,3)."""
    v_from = v_from / (torch.norm(v_from, dim=-1, keepdim=True) + 1e-8)
    v_to = v_to / (torch.norm(v_to, dim=-1, keepdim=True) + 1e-8)
    cross = torch.cross(v_from, v_to, dim=-1)
    dot = (v_from * v_to).sum(dim=-1, keepdim=True)
    quat = torch.cat([1.0 + dot, cross], dim=-1)

    opposite = dot.squeeze(-1) < -0.9999
    if opposite.any():
        ortho = torch.zeros_like(v_from[opposite])
        use_x = torch.abs(v_from[opposite, 0]) < 0.9
        ortho[use_x, 0] = 1.0
        ortho[~use_x, 1] = 1.0
        axis = torch.cross(v_from[opposite], ortho, dim=-1)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        quat[opposite] = torch.cat(
            [torch.zeros((axis.shape[0], 1), device=axis.device), axis], dim=-1,
        )

    return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)


def quat_rotate_batch(
    q: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    """Rotate vectors *v* by quaternion *q*. q:(N,4), v:(N,K,3) → (N,K,3)."""
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    w = q[:, 0:1].unsqueeze(-1)
    xyz_b = q[:, 1:].unsqueeze(1)
    t = 2.0 * torch.cross(xyz_b.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz_b.expand_as(t), t, dim=-1)


# ---------------------------------------------------------------------------
# Rotation noise
# ---------------------------------------------------------------------------

def add_tilt_noise(
    quat: torch.Tensor, std_rad: float, device: torch.device, n: int,
) -> torch.Tensor:
    """Apply random tilt around X and Y axes (Gaussian)."""
    angle_x = torch.randn(n, device=device) * std_rad
    angle_y = torch.randn(n, device=device) * std_rad
    hx = angle_x * 0.5
    hy = angle_y * 0.5
    zero = torch.zeros(n, device=device)
    qx = torch.stack([torch.cos(hx), torch.sin(hx), zero, zero], dim=-1)
    qy = torch.stack([torch.cos(hy), zero, torch.sin(hy), zero], dim=-1)
    tilt = quat_multiply(qx, qy)
    return quat_multiply(tilt, quat)


def add_rotation_noise(
    quat: torch.Tensor, std_rad: float, device: torch.device, n: int,
) -> torch.Tensor:
    """Apply random 3-axis rotation noise (uniform axis, Gaussian angle)."""
    axis = torch.randn(n, 3, device=device)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    angle = torch.randn(n, device=device) * std_rad
    half = angle * 0.5
    sin_half = torch.sin(half).unsqueeze(-1)
    cos_half = torch.cos(half).unsqueeze(-1)
    noise_quat = torch.cat([cos_half, axis * sin_half], dim=-1)
    return quat_multiply(noise_quat, quat)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def local_to_world_points(
    points_local: torch.Tensor,   # (n, F, 3)
    frame_pos: torch.Tensor,      # (n, 3)
    frame_quat: torch.Tensor,     # (n, 4)
) -> torch.Tensor:
    """Transform points from local frame to world frame."""
    rotated = quat_apply(
        frame_quat.unsqueeze(1).expand(-1, points_local.shape[1], -1).reshape(-1, 4),
        points_local.reshape(-1, 3),
    ).reshape_as(points_local)
    return rotated + frame_pos.unsqueeze(1)


def world_to_local_points(
    points_world: torch.Tensor,   # (n, F, 3)
    frame_pos: torch.Tensor,      # (n, 3)
    frame_quat: torch.Tensor,     # (n, 4)
) -> torch.Tensor:
    """Transform points from world frame to local frame."""
    points_rel = points_world - frame_pos.unsqueeze(1)
    return quat_apply_inverse(
        frame_quat.unsqueeze(1).expand(-1, points_world.shape[1], -1).reshape(-1, 4),
        points_rel.reshape(-1, 3),
    ).reshape_as(points_world)


# ---------------------------------------------------------------------------
# Rigid body alignment
# ---------------------------------------------------------------------------

def solve_rigid_alignment(
    points_src: torch.Tensor,    # (n, K, 3)
    points_dst: torch.Tensor,    # (n, K, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD-based rigid alignment. Returns (translation, quaternion)."""
    src_centroid = points_src.mean(dim=1, keepdim=True)
    dst_centroid = points_dst.mean(dim=1, keepdim=True)

    src_centered = points_src - src_centroid
    dst_centered = points_dst - dst_centroid

    cov = torch.matmul(src_centered.transpose(1, 2), dst_centered)
    u, _, vh = torch.linalg.svd(cov)
    rot = torch.matmul(vh.transpose(1, 2), u.transpose(1, 2))

    det = torch.det(rot)
    reflection = det < 0.0
    if reflection.any():
        vh_reflect = vh.clone()
        vh_reflect[reflection, -1, :] *= -1.0
        rot = torch.matmul(vh_reflect.transpose(1, 2), u.transpose(1, 2))

    pos = dst_centroid.squeeze(1) - torch.matmul(
        rot, src_centroid.squeeze(1).unsqueeze(-1)
    ).squeeze(-1)
    quat = quat_from_matrix(rot)
    return pos, quat


def solve_object_pose_from_contacts(
    points_obj: torch.Tensor,    # (n, F, 3)
    points_world: torch.Tensor,  # (n, F, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Best-of-subset rigid alignment from object-frame contacts to world
    fingertip positions. Returns (translation, quaternion).
    """
    n, num_points, _ = points_obj.shape
    if num_points <= 3:
        return solve_rigid_alignment(points_obj, points_world)

    def _candidate_error(
        pos: torch.Tensor, q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed = local_to_world_points(points_obj, pos, q)
        err = torch.norm(reconstructed - points_world, dim=-1)
        return err.max(dim=-1).values, err.mean(dim=-1)

    candidate_solutions: list[tuple[torch.Tensor, torch.Tensor]] = [
        solve_rigid_alignment(points_obj, points_world)
    ]
    for subset in combinations(range(num_points), 3):
        idx = torch.tensor(subset, device=points_obj.device, dtype=torch.long)
        candidate_solutions.append(
            solve_rigid_alignment(
                points_obj.index_select(1, idx),
                points_world.index_select(1, idx),
            )
        )

    best_pos = None
    best_quat = None
    best_max_err = torch.full((n,), float("inf"), device=points_obj.device)
    best_mean_err = torch.full((n,), float("inf"), device=points_obj.device)

    for pos, q in candidate_solutions:
        max_err, mean_err = _candidate_error(pos, q)
        if best_pos is None:
            best_pos, best_quat = pos, q
            best_max_err, best_mean_err = max_err, mean_err
            continue

        improved = (max_err < best_max_err - 1e-6) | (
            torch.isclose(max_err, best_max_err, atol=1e-6)
            & (mean_err < best_mean_err)
        )
        best_max_err = torch.where(improved, max_err, best_max_err)
        best_mean_err = torch.where(improved, mean_err, best_mean_err)
        best_pos = torch.where(improved.unsqueeze(-1), pos, best_pos)
        best_quat = torch.where(improved.unsqueeze(-1), q, best_quat)

    return best_pos, best_quat
