"""
Shared numpy math utilities for grasp generation.

All quaternion functions use the (w, x, y, z) convention.
No Isaac Sim dependency — only numpy.
"""
from __future__ import annotations

import numpy as np

from .grasp_sampler import Grasp


# ---------------------------------------------------------------------------
# Quaternion arithmetic (numpy)
# ---------------------------------------------------------------------------

def quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w,x,y,z). Returns normalized."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    out = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)
    return out / (np.linalg.norm(out) + 1e-8)


def quat_slerp_np(
    q0: np.ndarray, q1: np.ndarray, alpha: float,
) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-8)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / (sin_theta_0 + 1e-8)
    s1 = sin_theta / (sin_theta_0 + 1e-8)
    out = s0 * q0 + s1 * q1
    return out / (np.linalg.norm(out) + 1e-8)


def sample_quat_noise(
    rng: np.random.Generator, std: float = 0.25,
) -> np.ndarray:
    """Sample a small random rotation quaternion (Gaussian angle, random axis)."""
    axis = rng.normal(size=3).astype(np.float32)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = float(rng.normal(0.0, std))
    half = angle * 0.5
    quat = np.array(
        [np.cos(half), *(axis * np.sin(half))], dtype=np.float32,
    )
    return quat / (np.linalg.norm(quat) + 1e-8)


# ---------------------------------------------------------------------------
# Grasp distance metric
# ---------------------------------------------------------------------------

def grasp_distance(ga: Grasp, gb: Grasp) -> float:
    """
    Mean fingertip Euclidean distance + optional object pose distance.

    Used for RRT nearest-neighbor queries and graph edge construction.
    """
    tip_delta = ga.fingertip_positions - gb.fingertip_positions
    dist = float(np.linalg.norm(tip_delta, axis=-1).mean())

    if (getattr(ga, "object_pos_hand", None) is not None
            and getattr(gb, "object_pos_hand", None) is not None):
        dist += 0.25 * float(np.linalg.norm(
            np.asarray(ga.object_pos_hand) - np.asarray(gb.object_pos_hand)
        ))

    if (getattr(ga, "object_quat_hand", None) is not None
            and getattr(gb, "object_quat_hand", None) is not None):
        qa = np.asarray(ga.object_quat_hand, dtype=np.float32)
        qb = np.asarray(gb.object_quat_hand, dtype=np.float32)
        dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
        dist += 0.01 * float(2.0 * np.arccos(dot))

    return dist
