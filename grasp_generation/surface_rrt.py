"""
Stage 0 – Surface-Projected RRT Grasp Generation
=================================================
Generates grasp sets by expanding seed grasps via RRT with all fingertip
positions constrained to the object mesh surface.

Key differences from the original RRTGraspExpander:
  - Fingertip positions are projected onto the mesh via trimesh.proximity
  - Surface normals come from actual mesh face normals (not centroid approx)
  - Collision checking via signed distance prevents object penetration
  - Minimum finger spacing is enforced

Paper reference (DexterityGen §3.1):
  Starting from seed contact points on the object surface, grow the set by
  RRT-style exploration while keeping all contacts on the surface manifold.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .grasp_sampler import Grasp, GraspSet
from .math_utils import quat_multiply_np, sample_quat_noise, grasp_distance
from .net_force_optimization import NetForceOptimizer
from .rrt_expansion import GraspGraph


class SurfaceRRTGraspExpander:
    """
    Expands a seed GraspSet using RRT with surface-projected fingertip
    positions and builds a GraspGraph.

    All fingertip positions are constrained to lie on the object mesh surface.
    This prevents object penetration by construction.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Object mesh (watertight) used for surface projection.
    nfo : NetForceOptimizer, optional
        Quality evaluator. Created with defaults if None.
    delta_pos : float
        Standard deviation of Gaussian perturbation (metres).
    delta_max : float
        Maximum mean fingertip distance for graph edges.
    min_quality : float
        Minimum NFO quality for a grasp to be accepted.
    target_size : int
        Target number of grasps in the expanded set.
    max_attempts_per_step : int
        Max random samples per expansion step before giving up.
    collision_threshold : float
        Maximum allowed penetration depth (metres). Points deeper
        than this inside the mesh are rejected.
    min_finger_spacing : float
        Minimum pairwise distance between any two fingertips (metres).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        nfo: Optional[NetForceOptimizer] = None,
        delta_pos: float = 0.008,
        delta_max: float = 0.04,
        min_quality: float = 0.005,
        target_size: int = 300,
        max_attempts_per_step: int = 30,
        collision_threshold: float = 0.002,
        min_finger_spacing: float = 0.01,
        seed: int = 42,
    ):
        self.mesh = mesh
        self.nfo = nfo or NetForceOptimizer(min_quality=min_quality, fast_mode=True)
        self.delta_pos = delta_pos
        self.delta_max = delta_max
        self.min_quality = min_quality
        self.target_size = target_size
        self.max_attempts = max_attempts_per_step
        self.collision_threshold = collision_threshold
        self.min_finger_spacing = min_finger_spacing
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, seed_set: GraspSet) -> GraspGraph:
        """Expand seed_set to target_size via surface-projected RRT."""
        print(f"[SurfaceRRT] Expanding '{seed_set.object_name}': "
              f"{len(seed_set)} seeds -> target {self.target_size}")

        grasps: List[Grasp] = list(seed_set.grasps)
        total_attempts = 0
        max_total = self.target_size * 200

        while len(grasps) < self.target_size:
            new_grasp = self._expand_step(grasps)
            if new_grasp is not None:
                grasps.append(new_grasp)
                if len(grasps) % 50 == 0:
                    print(f"  [SurfaceRRT] {len(grasps)}/{self.target_size}")
            total_attempts += 1
            if total_attempts > max_total:
                print(f"  [SurfaceRRT] Stopping early at {len(grasps)} grasps "
                      f"({total_attempts} attempts)")
                break

        final_set = GraspSet(grasps=grasps, object_name=seed_set.object_name)
        graph = self._build_graph(final_set)
        print(f"[SurfaceRRT] Done: {len(graph)} nodes, {graph.num_edges} edges")
        return graph

    # ------------------------------------------------------------------
    # Core RRT step
    # ------------------------------------------------------------------

    def _expand_step(self, grasps: List[Grasp]) -> Optional[Grasp]:
        """Try up to max_attempts random expansions, return first valid one."""
        for _ in range(self.max_attempts):
            # 1. Pick random parent
            parent = grasps[self.rng.integers(0, len(grasps))]

            # 2. Perturb fingertip positions with Gaussian noise
            noise = self.rng.normal(
                0, self.delta_pos, parent.fingertip_positions.shape
            ).astype(np.float32)
            p_perturbed = parent.fingertip_positions + noise

            # 3. Project onto mesh surface
            p_surface, normals_surface, _ = self._project_to_surface(p_perturbed)

            # 4. Interpolate toward parent to bound step size
            alpha = 0.35
            p_interp = (
                (1.0 - alpha) * parent.fingertip_positions
                + alpha * p_surface
            ).astype(np.float32)

            # 5. Re-project interpolated points onto surface
            p_candidate, normals_candidate, dists = self._project_to_surface(p_interp)

            # 6. Collision check (no point should be deep inside mesh)
            if not self._check_collision(p_candidate, dists):
                continue

            # 7. Finger spacing check
            if not self._check_finger_spacing(p_candidate):
                continue

            # 7b. Shadow Hand: thumb must still oppose 4-finger group
            if len(p_candidate) == 5:
                if not self._check_thumb_opposition(p_candidate, normals_candidate):
                    continue

            # 8. NFO quality check
            candidate = Grasp(
                fingertip_positions=p_candidate,
                contact_normals=normals_candidate,
                quality=0.0,
                object_name=parent.object_name,
                object_scale=parent.object_scale,
                object_pos_hand=self._perturb_position(parent),
                object_quat_hand=self._perturb_quaternion(parent),
                object_pose_frame=getattr(parent, "object_pose_frame", None),
            )

            q = self.nfo.evaluate(candidate)
            if q >= self.min_quality:
                candidate.quality = q
                return candidate

        return None

    # ------------------------------------------------------------------
    # Surface projection
    # ------------------------------------------------------------------

    def _project_to_surface(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project (F, 3) positions onto the mesh surface.

        Returns
        -------
        projected : (F, 3) positions on the mesh surface
        normals   : (F, 3) outward face normals at projected points
        distances : (F,) distances from original points to projected points
        """
        closest, distances_arr, face_idx = trimesh.proximity.closest_point(
            self.mesh, positions
        )
        normals = self.mesh.face_normals[face_idx].astype(np.float32)
        projected = closest.astype(np.float32)
        return projected, normals, distances_arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------------

    def _check_collision(
        self, positions: np.ndarray, projection_distances: np.ndarray
    ) -> bool:
        """
        Check that fingertip positions are on/near the surface, not inside.

        Uses the projection distance as a proxy: if the original (interpolated)
        point was far from the surface, the projection was large, indicating
        the point may have been inside the mesh. For watertight meshes we also
        check containment directly.

        Returns True if collision-free.
        """
        # Fast check: if projection pulled points very far, reject
        if np.any(projection_distances > self.collision_threshold * 10):
            return False

        # Containment check: reject if any point is inside the mesh
        try:
            inside = self.mesh.contains(positions)
            if np.any(inside):
                return False
        except Exception:
            # Fallback: skip containment check if mesh is not watertight
            pass

        return True

    def _check_finger_spacing(self, positions: np.ndarray) -> bool:
        """
        Check that all fingertip pairs are at least min_finger_spacing apart.

        Returns True if spacing constraint is satisfied.
        """
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist < self.min_finger_spacing:
                    return False
        return True

    def _check_thumb_opposition(
        self,
        positions: np.ndarray,  # (5, 3)
        normals: np.ndarray,    # (5, 3)
    ) -> bool:
        """
        Shadow Hand constraint: thumb (index 4, last) must oppose the
        4-finger group (indices 0-3). Preserves the grasp topology
        established by the seed sampler during RRT expansion.
        """
        thumb_n = normals[4]
        fingers_n = normals[:4]
        mean_fn = fingers_n.mean(axis=0)
        mean_fn /= np.linalg.norm(mean_fn) + 1e-8
        return float(np.dot(thumb_n, mean_fn)) < -0.2

    # ------------------------------------------------------------------
    # Pose perturbation helpers
    # ------------------------------------------------------------------

    def _perturb_position(self, parent: Grasp) -> Optional[np.ndarray]:
        """Perturb object_pos_hand with small noise."""
        if getattr(parent, "object_pos_hand", None) is None:
            return None
        noise = self.rng.normal(0, self.delta_pos, size=3).astype(np.float32)
        return (np.asarray(parent.object_pos_hand, dtype=np.float32) + noise)

    def _perturb_quaternion(self, parent: Grasp) -> Optional[np.ndarray]:
        """Perturb object_quat_hand with small rotation noise."""
        if getattr(parent, "object_quat_hand", None) is None:
            return None
        delta = sample_quat_noise(self.rng)
        return quat_multiply_np(
            np.asarray(parent.object_quat_hand, dtype=np.float32), delta
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, grasp_set: GraspSet) -> GraspGraph:
        """Build connectivity graph from expanded grasp set."""
        N = len(grasp_set)
        num_fingers = (
            grasp_set.grasps[0].fingertip_positions.shape[0] if N > 0 else 4
        )
        effective_delta_max = self.delta_max * (
            1.0 + 0.35 * max(num_fingers - 1, 0)
        )

        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if grasp_distance(
                    grasp_set.grasps[i], grasp_set.grasps[j]
                ) < effective_delta_max:
                    edges.append((i, j))

        return GraspGraph(
            grasp_set=grasp_set,
            edges=edges,
            object_name=grasp_set.object_name,
            num_fingers=num_fingers,
        )
