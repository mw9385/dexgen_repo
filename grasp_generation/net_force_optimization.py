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
# 2-finger pinch quality (opposition-based)
# ---------------------------------------------------------------------------

def pinch_quality(positions: np.ndarray, normals: np.ndarray) -> float:
    """
    Quality metric for 2-contact (pinch) grasps.

    Full 6-DOF force closure is generally unachievable with only 2 contacts
    (the LP wrench-space analysis always returns 0).  Instead we use an
    opposition-alignment score:

        quality = max(0, -n0 · n1)

    where n0, n1 are the unit outward normals.  A perfectly antipodal pair
    (normals pointing in exactly opposite directions) scores 1.0; contacts
    on the same side score 0.

    A secondary weight adds a mild bonus for contacts that are spread apart
    (spread_fraction = dist / max_possible_dist, capped at 1).
    """
    n0 = normals[0] / (np.linalg.norm(normals[0]) + 1e-8)
    n1 = normals[1] / (np.linalg.norm(normals[1]) + 1e-8)
    opposition = float(-np.dot(n0, n1))      # +1 = perfect antipodal
    opposition = max(0.0, opposition)

    # Mild spread bonus (contact distance normalised by max object dimension)
    dist = float(np.linalg.norm(positions[0] - positions[1]))
    obj_span = float(np.linalg.norm(positions.mean(axis=0))) * 2.0 + 1e-3
    spread = min(1.0, dist / obj_span)

    return float(opposition * 0.8 + spread * 0.2)


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

    [Fix] The contact reaction force acts INWARD (opposite to the outward normal).
          The friction cone is centred on -normal (inward direction).
          Each edge = -normal + mu * tangential_component, then normalised.
          Previously the code was correct in using -normal as the cone axis,
          but the normalisation was applied AFTER adding the tangential part,
          which is correct. However the resulting force vectors were not
          guaranteed to be unit vectors pointing into the friction cone.

          More importantly: the contact wrench matrix W must have columns
          that represent feasible contact forces. The force at contact i
          must lie INSIDE the friction cone, i.e.:
            f = alpha * (-n_i + mu * t)  where alpha > 0
          The wrench is [f; p × f] (6-dim).

          The previous code was correct in structure but the normalisation
          changes the cone geometry — normalised edges no longer span the
          same convex set as the original cone. We keep normalisation for
          numerical stability but document this approximation.
    """
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Build orthonormal tangent frame via Gram-Schmidt
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    else:
        t1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    t1 /= np.linalg.norm(t1) + 1e-8
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2) + 1e-8

    angles = np.linspace(0, 2 * np.pi, num_edges, endpoint=False)
    # Contact reaction force: inward normal + friction tangential component
    # -normal: reaction pushes INTO the object (Newton's 3rd law)
    # mu * tangential: friction allows lateral force up to mu * normal_force
    edges = (-normal[None, :]
             + mu * (np.cos(angles[:, None]) * t1 + np.sin(angles[:, None]) * t2))
    # Normalise for numerical stability (approximation of cone boundary)
    norms = np.linalg.norm(edges, axis=-1, keepdims=True)
    edges /= norms + 1e-8
    return edges.astype(np.float32)


def contact_wrench_matrix(
    positions: np.ndarray,
    normals: np.ndarray,
    mu: float = 0.5,
    num_edges: int = 8,
    characteristic_length: Optional[float] = None,
) -> np.ndarray:
    """
    Build the grasp wrench matrix W for K contacts.

    Args:
        positions:             (K, 3) contact point positions (object frame)
        normals:               (K, 3) outward surface normals
        mu:                    friction coefficient
        num_edges:             friction cone linearisation edges per contact
        characteristic_length: object size used to normalise torque components
                               so the ε-metric is size-independent.
                               If None, estimated from max contact distance.

    Returns:
        W: (6, K * num_edges) grasp wrench matrix

    Note on torque normalisation
    ----------------------------
    The ε-metric (largest inscribed ball in wrench space) scales with the
    torque components of the wrenches, which are O(||p|| * ||f||).  For
    small objects ||p|| is small → small torque → small ε → fails quality
    threshold even for geometrically valid grasps.  Dividing the torque
    rows by a characteristic length L (≈ object radius) makes the metric
    dimensionless and size-independent, matching robotics convention.
    """
    K = len(positions)
    if characteristic_length is None:
        # Estimate: mean contact-to-centroid distance, at least 1 cm
        centroid = positions.mean(axis=0)
        characteristic_length = float(
            max(np.linalg.norm(positions - centroid, axis=-1).mean(), 0.01)
        )

    cols = []
    for i in range(K):
        edges = friction_cone_edges(normals[i], mu, num_edges)  # (E, 3)
        for e in edges:
            force = e                                                      # (3,)
            torque = np.cross(positions[i], e) / characteristic_length    # (3,) normalised
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
        """
        Compute the quality score for a single grasp.

        For 2-contact grasps the LP wrench-space analysis always returns 0
        (two contacts cannot achieve 6-DOF force closure in 3-D space).
        We use an opposition-based pinch quality instead.

        For 3+ contact grasps the standard NFO ε-metric is used with
        torque normalisation so the score is size-independent.
        """
        K = len(grasp.fingertip_positions)
        if K <= 2:
            return pinch_quality(grasp.fingertip_positions, grasp.contact_normals)

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
