"""
Stage 0 – Net Force Optimization (NFO)
======================================
Evaluates and optimizes grasp quality using the wrench-space analysis.

Paper reference (DexterityGen §3.1, Appendix A):
  A grasp is force-closure if the convex hull of the contact wrenches
  contains the origin. The *NFO score* measures how far the origin is
  from the boundary of this hull – larger is more robust.

  For each contact point i with position p_i and outward normal n_i:
    - Friction cone: linearized with F friction directions
    - Contact wrench w_i^f = [f_i; p_i × f_i]  (force + torque, 6-dim)
    - Grasp wrench matrix W = [w_1^1 ... w_1^F | ... | w_K^F]  (6 × KF)
    - NFO score = max ε  s.t.  ε·ball ⊂ conv(columns of W)
                = min ||G q||  over unit-norm q in feasible set
                ≈ solved via LP / QP

  We use a simplified L1-norm version (fast, good enough for filtering):
    score = min_{q≥0, Σq=1}  max_direction ||W q||_1
  and approximate with iterative projection.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from typing import List, Optional

from .grasp_sampler import Grasp, GraspSet


# ---------------------------------------------------------------------------
# Contact Wrench Computation
# ---------------------------------------------------------------------------

def friction_cone_edges(
    normal: np.ndarray,
    mu: float = 0.5,
    num_edges: int = 8,
) -> np.ndarray:
    """
    Linearise the friction cone around a contact normal.

    Args:
        normal: (3,) outward surface normal (unit vector)
        mu: friction coefficient
        num_edges: number of linearisation edges

    Returns:
        edges: (num_edges, 3) force directions (unit vectors along cone edges)
    """
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Build tangent frame
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, [1, 0, 0])
    else:
        t1 = np.cross(normal, [0, 1, 0])
    t1 /= np.linalg.norm(t1) + 1e-8
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2) + 1e-8

    angles = np.linspace(0, 2 * np.pi, num_edges, endpoint=False)
    edges = (-normal[None, :]                   # reaction = push inward
             + mu * (np.cos(angles[:, None]) * t1 + np.sin(angles[:, None]) * t2))
    edges /= np.linalg.norm(edges, axis=-1, keepdims=True) + 1e-8
    return edges.astype(np.float32)


def contact_wrench_matrix(
    positions: np.ndarray,
    normals: np.ndarray,
    mu: float = 0.5,
    num_edges: int = 8,
) -> np.ndarray:
    """
    Build the grasp wrench matrix W for K contacts.

    Args:
        positions: (K, 3) contact point positions (object frame)
        normals:   (K, 3) outward surface normals
        mu:        friction coefficient
        num_edges: friction cone linearisation edges per contact

    Returns:
        W: (6, K * num_edges) grasp wrench matrix
    """
    K = len(positions)
    cols = []
    for i in range(K):
        edges = friction_cone_edges(normals[i], mu, num_edges)  # (E, 3)
        for e in edges:
            force = e                            # (3,)
            torque = np.cross(positions[i], e)   # (3,)
            cols.append(np.concatenate([force, torque]))
    W = np.stack(cols, axis=1).astype(np.float32)   # (6, K*E)
    return W


# ---------------------------------------------------------------------------
# Grasp Quality Metrics
# ---------------------------------------------------------------------------

def grasp_quality_lp(W: np.ndarray) -> float:
    """
    Compute the ε-metric (largest ball inscribed in Grasp Wrench Space)
    via Linear Programming.

    Formulation:
        max  ε
        s.t. W q = ε * d    for all unit wrench directions d
             q ≥ 0

    We approximate by checking 12 cardinal wrench directions
    (±1 along each of 6 wrench axes) and taking the minimum ε.

    Returns:
        quality: float in [0, ∞), 0 means not force-closure
    """
    n_cols = W.shape[1]
    min_eps = float("inf")

    # 12 canonical wrench directions (± unit vectors)
    for sign in [1, -1]:
        for axis in range(6):
            d = np.zeros(6, dtype=np.float64)
            d[axis] = float(sign)

            # LP: max  ε  s.t.  W q = ε d,  q ≥ 0, Σq = 1
            # Rewrite: min -ε
            # Variables: x = [q (n_cols), ε (1)]
            # Eq constraint: W q - ε d = 0  →  [W | -d] [q; ε] = 0
            # Sum constraint: 1^T q = 1
            # Bounds: q ≥ 0, ε free

            A_eq = np.zeros((7, n_cols + 1), dtype=np.float64)
            A_eq[:6, :n_cols] = W.astype(np.float64)
            A_eq[:6, n_cols] = -d
            A_eq[6, :n_cols] = 1.0
            b_eq = np.zeros(7, dtype=np.float64)
            b_eq[6] = 1.0

            c = np.zeros(n_cols + 1, dtype=np.float64)
            c[-1] = -1.0   # minimise -ε

            bounds = [(0.0, None)] * n_cols + [(None, None)]

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success and res.fun < 0:
                eps = -res.fun
                min_eps = min(min_eps, eps)
            else:
                return 0.0   # infeasible → not force-closure

    return float(min_eps) if min_eps < float("inf") else 0.0


def grasp_quality_fast(W: np.ndarray) -> float:
    """
    Fast approximate quality metric using the minimum singular value of W.

    This is O(n) vs O(n^3) for the LP, useful for quick filtering.
    A non-zero minimum singular value is necessary (not sufficient) for
    force closure.
    """
    try:
        sv = np.linalg.svd(W, compute_uv=False)
        return float(sv.min())
    except np.linalg.LinAlgError:
        return 0.0


# ---------------------------------------------------------------------------
# Net Force Optimizer
# ---------------------------------------------------------------------------

class NetForceOptimizer:
    """
    Evaluates grasp quality for each grasp in a GraspSet.

    Per DexterityGen, quality = NFO score = ε-metric of wrench space.
    Grasps below min_quality are discarded.

    Args:
        mu:           friction coefficient (default 0.5)
        num_edges:    friction cone edges (default 8)
        min_quality:  minimum acceptable quality (default 0.01)
        fast_mode:    use fast SVD approximation instead of LP (default False)
    """

    def __init__(
        self,
        mu: float = 0.5,
        num_edges: int = 8,
        min_quality: float = 0.01,
        fast_mode: bool = False,
    ):
        self.mu = mu
        self.num_edges = num_edges
        self.min_quality = min_quality
        self.fast_mode = fast_mode

    def evaluate(self, grasp: Grasp) -> float:
        """Compute the NFO quality score for a single grasp."""
        W = contact_wrench_matrix(
            grasp.fingertip_positions,
            grasp.contact_normals,
            mu=self.mu,
            num_edges=self.num_edges,
        )
        if self.fast_mode:
            return grasp_quality_fast(W)
        else:
            return grasp_quality_lp(W)

    def evaluate_set(self, grasp_set: GraspSet, verbose: bool = True) -> GraspSet:
        """
        Score all grasps and filter out low-quality ones.

        Returns a new GraspSet with quality scores filled in.
        """
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
            print(f"[NFO] Kept {len(result)}/{len(grasp_set)} grasps "
                  f"(quality ≥ {self.min_quality})")
        return result

    def optimize_single(
        self,
        grasp: Grasp,
        n_perturbations: int = 20,
        step_size: float = 0.003,
    ) -> Grasp:
        """
        Locally optimise a grasp by perturbing fingertip positions
        on the mesh surface to maximise NFO score.

        This is a simple hill-climbing refinement step.
        """
        best_grasp = grasp
        best_quality = self.evaluate(grasp)

        rng = np.random.default_rng(0)
        for _ in range(n_perturbations):
            noise = rng.normal(0, step_size, grasp.fingertip_positions.shape)
            new_pos = grasp.fingertip_positions + noise.astype(np.float32)
            candidate = Grasp(
                fingertip_positions=new_pos,
                contact_normals=grasp.contact_normals,
                quality=0.0,
                object_name=grasp.object_name,
            )
            q = self.evaluate(candidate)
            if q > best_quality:
                best_quality = q
                candidate.quality = q
                best_grasp = candidate

        return best_grasp
