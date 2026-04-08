"""
Net Force Optimization (NFO)
============================
Evaluates grasp quality using wrench-space ε-metric.

Paper reference (DexterityGen §3.1, Appendix A):
  For each contact point i with position p_i and outward normal n_i:
    - Friction cone: linearized with F friction directions
    - Contact wrench w_i^f = [f_i; p_i × f_i]  (force + torque, 6-dim)
    - Grasp wrench matrix W = [w_1^1 ... w_1^F | ... | w_K^F]  (6 × KF)
    - NFO score = max ε  s.t.  ε·ball ⊂ conv(columns of W)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .grasp_sampler import Grasp, GraspSet


# ---------------------------------------------------------------------------
# Friction Cone Linearisation
# ---------------------------------------------------------------------------

def friction_cone_edges(
    normal: np.ndarray,
    mu: float = 0.5,
    num_edges: int = 8,
) -> np.ndarray:
    """
    Linearise the friction cone around a contact normal.

    Returns (num_edges, 3) force directions on the cone boundary.
    """
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Orthonormal tangent frame
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    else:
        t1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    t1 /= np.linalg.norm(t1) + 1e-8
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2) + 1e-8

    angles = np.linspace(0, 2 * np.pi, num_edges, endpoint=False)
    edges = (-normal[None, :]
             + mu * (np.cos(angles[:, None]) * t1 + np.sin(angles[:, None]) * t2))
    norms = np.linalg.norm(edges, axis=-1, keepdims=True)
    edges /= norms + 1e-8
    return edges.astype(np.float32)


# ---------------------------------------------------------------------------
# Wrench Matrix
# ---------------------------------------------------------------------------

def contact_wrench_matrix(
    positions: np.ndarray,
    normals: np.ndarray,
    mu: float = 0.5,
    num_edges: int = 8,
) -> np.ndarray:
    """
    Build the grasp wrench matrix W for K contacts.

    W = (6, K * num_edges) where each column is [force; torque].
    """
    K = len(positions)
    cols = []
    for i in range(K):
        edges = friction_cone_edges(normals[i], mu, num_edges)
        for e in edges:
            force = e
            torque = np.cross(positions[i], e)
            cols.append(np.concatenate([force, torque]))
    return np.stack(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# LP ε-metric
# ---------------------------------------------------------------------------

def grasp_quality_lp(W: np.ndarray) -> float:
    """
    Compute the ε-metric via LP.

    Checks 12 cardinal wrench directions (±1 along each of 6 axes).
    Returns min ε across all directions (0 = not force-closure).
    """
    n_cols = W.shape[1]
    min_eps = float("inf")

    for sign in [1, -1]:
        for axis in range(6):
            d = np.zeros(6, dtype=np.float64)
            d[axis] = float(sign)

            A_eq = np.zeros((7, n_cols + 1), dtype=np.float64)
            A_eq[:6, :n_cols] = W.astype(np.float64)
            A_eq[:6, n_cols] = -d
            A_eq[6, :n_cols] = 1.0
            b_eq = np.zeros(7, dtype=np.float64)
            b_eq[6] = 1.0

            c = np.zeros(n_cols + 1, dtype=np.float64)
            c[-1] = -1.0

            bounds = [(0.0, None)] * n_cols + [(None, None)]

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success and res.fun < 0:
                min_eps = min(min_eps, -res.fun)
            else:
                return 0.0

    return float(min_eps) if min_eps < float("inf") else 0.0


# ---------------------------------------------------------------------------
# NFO Evaluator
# ---------------------------------------------------------------------------

class NetForceOptimizer:
    """
    Evaluates grasp quality via wrench-space ε-metric (LP).

    Args:
        mu:          friction coefficient
        num_edges:   friction cone linearisation edges per contact
        min_quality: minimum acceptable ε-metric score
    """

    def __init__(
        self,
        mu: float = 0.5,
        num_edges: int = 8,
        min_quality: float = 0.03,
    ):
        self.mu = mu
        self.num_edges = num_edges
        self.min_quality = min_quality

    def evaluate(self, grasp: Grasp) -> float:
        """Compute NFO ε-metric for a single grasp."""
        K = len(grasp.fingertip_positions)
        if K < 2:
            return 0.0
        W = contact_wrench_matrix(
            grasp.fingertip_positions,
            grasp.contact_normals,
            mu=self.mu,
            num_edges=self.num_edges,
        )
        return grasp_quality_lp(W)

    def evaluate_set(self, grasp_set: GraspSet, verbose: bool = True) -> GraspSet:
        """Score all grasps and filter by min_quality."""
        scored = []
        for i, grasp in enumerate(grasp_set.grasps):
            q = self.evaluate(grasp)
            grasp.quality = q
            if q >= self.min_quality:
                scored.append(grasp)
            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(grasp_set)}] good so far: {len(scored)}")

        result = GraspSet(grasps=scored, object_name=grasp_set.object_name)
        if verbose:
            print(f"[NFO] Kept {len(result)}/{len(grasp_set)} "
                  f"(quality >= {self.min_quality})")
        return result
