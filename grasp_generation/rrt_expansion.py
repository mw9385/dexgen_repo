"""
Stage 0 – RRT-based Grasp Set Expansion
=========================================
Expands a seed grasp set and builds a GraspGraph (or MultiObjectGraspGraph
when multiple objects are used).

Paper reference (DexterityGen §3.1):
  Starting from an initial set of M seed grasps, we grow the set by
  repeatedly:
    1. Sampling a random grasp from the current set (random node)
    2. Applying a small random perturbation to get a new candidate
    3. Checking quality (NFO score ≥ threshold)
    4. Checking connectivity (reachable from existing grasp via
       continuous fingertip path on the object surface)
    5. Adding to the set if valid

  This produces a *connected* grasp graph G = (V, E) where:
    V = grasp configurations
    E = (g_i, g_j) if mean fingertip distance < δ_max

  The graph is the input to Stage 1 RL training (each episode
  picks a random edge as start→goal).
"""

from __future__ import annotations

import pickle
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grasp_sampler import Grasp, GraspSet
from .math_utils import grasp_distance, quat_multiply_np, quat_slerp_np, sample_quat_noise
from .net_force_optimization import NetForceOptimizer


# ---------------------------------------------------------------------------
# GraspGraph  (single object)
# ---------------------------------------------------------------------------

@dataclass
class GraspGraph:
    """
    Connected graph of grasps for a single object.

    Nodes: grasp configurations (GraspSet)
    Edges: (i, j) pairs reachable by continuous fingertip motion

    num_fingers is stored explicitly so downstream code (env, obs, events)
    can adapt without inspecting grasp data shapes.
    """
    grasp_set: GraspSet
    edges: List[Tuple[int, int]] = field(default_factory=list)
    object_name: str = ""
    num_fingers: int = 4

    def __len__(self):
        return len(self.grasp_set)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_neighbors(self, node_idx: int) -> List[int]:
        neighbors = []
        for i, j in self.edges:
            if i == node_idx:
                neighbors.append(j)
            elif j == node_idx:
                neighbors.append(i)
        return neighbors

    def sample_edge(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, len(self.edges))
        return self.edges[idx]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[GraspGraph] Saved {len(self)} nodes, {self.num_edges} edges → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GraspGraph":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        print(f"[GraspGraph] Loaded {len(obj)} nodes, {obj.num_edges} edges ← {path}")
        return obj


# ---------------------------------------------------------------------------
# MultiObjectGraspGraph  (NEW — one graph per object)
# ---------------------------------------------------------------------------

@dataclass
class MultiObjectGraspGraph:
    """
    Collection of per-object GraspGraphs used when training with
    a randomised object pool.

    At each RL episode reset:
      1. Sample a random object name from self.graphs
      2. Sample a random edge from graphs[object_name]
      → This gives (start_grasp, goal_grasp) + which object to spawn

    Attributes:
        graphs:       object_name → GraspGraph
        object_specs: object_name → dict with shape/size/mass/color
                      (passed to Isaac Lab scene for spawning)
    """
    graphs: Dict[str, GraspGraph] = field(default_factory=dict)
    object_specs: Dict[str, dict] = field(default_factory=dict)

    def __len__(self):
        return sum(len(g) for g in self.graphs.values())

    @property
    def num_objects(self) -> int:
        return len(self.graphs)

    @property
    def object_names(self) -> List[str]:
        return list(self.graphs.keys())

    @property
    def num_fingers(self) -> int:
        """Number of contact points per grasp (consistent across all objects)."""
        if not self.graphs:
            return 4
        return next(iter(self.graphs.values())).num_fingers

    def add(self, graph: GraspGraph, spec: dict):
        """Add a per-object GraspGraph with its Isaac Lab spawn spec."""
        self.graphs[graph.object_name] = graph
        self.object_specs[graph.object_name] = spec

    def sample_object(
        self, rng: Optional[np.random.Generator] = None
    ) -> str:
        """Uniformly sample an object name."""
        if rng is None:
            rng = np.random.default_rng()
        names = self.object_names
        return names[rng.integers(0, len(names))]

    def sample_edge(
        self,
        object_name: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Sample a random (object_name, (start_idx, goal_idx)) pair.
        If object_name is None, also samples the object randomly.
        """
        if rng is None:
            rng = np.random.default_rng()
        if object_name is None:
            object_name = self.sample_object(rng)
        edge = self.graphs[object_name].sample_edge(rng)
        return object_name, edge

    def get_grasp(self, object_name: str, grasp_idx: int) -> Grasp:
        return self.graphs[object_name].grasp_set[grasp_idx]

    def summary(self):
        print(f"[MultiObjectGraspGraph] {self.num_objects} objects:")
        for name, g in self.graphs.items():
            print(f"  {name}: {len(g)} nodes, {g.num_edges} edges")

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        total = sum(len(g) for g in self.graphs.values())
        print(f"[MultiObjectGraspGraph] Saved {self.num_objects} objects, "
              f"{total} total grasps → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MultiObjectGraspGraph":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        print(f"[MultiObjectGraspGraph] Loaded {obj.num_objects} objects ← {path}")
        return obj


# ---------------------------------------------------------------------------
# Standalone graph builder for direct grasp lists.
# ---------------------------------------------------------------------------

def build_graph_from_grasps(
    grasps: List[Grasp],
    object_name: str = "",
    delta_max: float = 0.04,
    num_fingers: int = 4,
) -> GraspGraph:
    """
    Build a connectivity graph from a flat list of grasps.

    This is used when grasps are produced directly without RRT expansion.
    Edges connect grasp pairs whose
    mean fingertip distance is below ``delta_max``.

    Parameters
    ----------
    grasps : list[Grasp]
    object_name : str
    delta_max : float
        Maximum mean fingertip distance for an edge.  Scaled by
        ``1 + 0.35 * (num_fingers - 1)`` to match RRT convention.
    num_fingers : int

    Returns
    -------
    GraspGraph
    """
    grasp_set = GraspSet(grasps=list(grasps), object_name=object_name)
    N = len(grasp_set)
    if N == 0:
        return GraspGraph(grasp_set=grasp_set, object_name=object_name,
                          num_fingers=num_fingers)

    # Same effective delta_max as RRTGraspExpander._build_graph
    effective_delta_max = delta_max * (1.0 + 0.35 * max(num_fingers - 1, 0))

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if grasp_distance(grasps[i], grasps[j]) < effective_delta_max:
                edges.append((i, j))

    graph = GraspGraph(
        grasp_set=grasp_set,
        edges=edges,
        object_name=object_name,
        num_fingers=num_fingers,
    )
    print(f"[build_graph_from_grasps] {N} nodes, {len(edges)} edges "
          f"(delta_max={effective_delta_max:.4f}m)")
    return graph


# ---------------------------------------------------------------------------
# RRT Expander
# ---------------------------------------------------------------------------

class RRTGraspExpander:
    """
    Expands a seed GraspSet using RRT-style exploration and builds a GraspGraph.
    """

    def __init__(
        self,
        nfo: Optional[NetForceOptimizer] = None,
        delta_pos: float = 0.008,
        delta_max: float = 0.04,
        min_quality: float = 0.005,
        target_size: int = 500,
        max_attempts_per_step: int = 20,
        manifold_contact_count: Optional[int] = None,
        seed: int = 42,
    ):
        self.nfo = nfo or NetForceOptimizer(min_quality=min_quality, fast_mode=True)
        self.delta_pos = delta_pos
        self.delta_max = delta_max
        self.min_quality = min_quality
        self.target_size = target_size
        self.max_attempts = max_attempts_per_step
        self.manifold_contact_count = manifold_contact_count
        self.rng = np.random.default_rng(seed)

    def expand(self, seed_set: GraspSet) -> GraspGraph:
        """Expand seed_set to target_size and build a GraspGraph."""
        print(f"[RRT] Expanding '{seed_set.object_name}': "
              f"{len(seed_set)} seeds → target {self.target_size}")

        grasps: List[Grasp] = list(seed_set.grasps)
        total_attempts = 0

        while len(grasps) < self.target_size:
            new_grasp = self._expand_step(grasps)
            if new_grasp is not None:
                grasps.append(new_grasp)
                if len(grasps) % 50 == 0:
                    print(f"  [RRT] {len(grasps)}/{self.target_size}")
            total_attempts += 1
            if total_attempts > self.target_size * 200:
                print(f"  [RRT] Stopping early at {len(grasps)} grasps")
                break

        final_set = GraspSet(grasps=grasps, object_name=seed_set.object_name)
        graph = self._build_graph(final_set)
        print(f"[RRT] Done: {len(graph)} nodes, {graph.num_edges} edges")
        return graph

    def _expand_step(self, grasps: List[Grasp]) -> Optional[Grasp]:
        for _ in range(self.max_attempts):
            random_state = self._sample_random_state(grasps)
            nearest = self._nearest_neighbor(random_state, grasps)
            candidate = self._projected_step(nearest, random_state)
            if candidate is None:
                continue

            q = self.nfo.evaluate(candidate)
            if q >= self.min_quality:
                candidate.quality = q
                return candidate
        return None

    def _sample_random_state(self, grasps: List[Grasp]) -> Grasp:
        src = grasps[self.rng.integers(0, len(grasps))]
        pos_noise = self.rng.normal(0, self.delta_pos, src.fingertip_positions.shape).astype(np.float32)
        new_pos = src.fingertip_positions + pos_noise
        new_normals = self._recompute_normals_from_positions(new_pos)

        object_pos_hand = None
        if getattr(src, "object_pos_hand", None) is not None:
            object_pos_hand = (
                np.asarray(src.object_pos_hand, dtype=np.float32)
                + self.rng.normal(0, self.delta_pos, size=3).astype(np.float32)
            )

        object_quat_hand = None
        if getattr(src, "object_quat_hand", None) is not None:
            delta_quat = sample_quat_noise(self.rng)
            object_quat_hand = quat_multiply_np(np.asarray(src.object_quat_hand, dtype=np.float32), delta_quat)

        return Grasp(
            fingertip_positions=new_pos,
            contact_normals=new_normals,
            quality=0.0,
            object_name=src.object_name,
            object_scale=src.object_scale,
            object_pos_hand=object_pos_hand,
            object_quat_hand=object_quat_hand,
            object_pose_frame=getattr(src, "object_pose_frame", None),
        )

    def _nearest_neighbor(self, target: Grasp, grasps: List[Grasp]) -> Grasp:
        dists = [grasp_distance(g, target) for g in grasps]
        return grasps[int(np.argmin(dists))]

    def _interpolate_state(self, src: Grasp, dst: Grasp) -> Grasp:
        alpha = 0.35
        new_pos = (1.0 - alpha) * src.fingertip_positions + alpha * dst.fingertip_positions
        new_pos = new_pos.astype(np.float32)
        new_normals = self._recompute_normals_from_positions(new_pos)

        object_pos_hand = None
        if getattr(src, "object_pos_hand", None) is not None and getattr(dst, "object_pos_hand", None) is not None:
            object_pos_hand = (
                (1.0 - alpha) * np.asarray(src.object_pos_hand, dtype=np.float32)
                + alpha * np.asarray(dst.object_pos_hand, dtype=np.float32)
            ).astype(np.float32)

        object_quat_hand = None
        if getattr(src, "object_quat_hand", None) is not None and getattr(dst, "object_quat_hand", None) is not None:
            object_quat_hand = quat_slerp_np(
                np.asarray(src.object_quat_hand, dtype=np.float32),
                np.asarray(dst.object_quat_hand, dtype=np.float32),
                alpha,
            ).astype(np.float32)

        return Grasp(
            fingertip_positions=new_pos,
            contact_normals=new_normals,
            quality=0.0,
            object_name=src.object_name,
            object_scale=src.object_scale,
            object_pos_hand=object_pos_hand,
            object_quat_hand=object_quat_hand,
            object_pose_frame=getattr(src, "object_pose_frame", None),
        )

    def _projected_step(self, xnode: Grasp, xsample: Grasp) -> Optional[Grasp]:
        num_fingers = xnode.fingertip_positions.shape[0]
        if self.manifold_contact_count is None:
            keep_count = num_fingers
        else:
            keep_count = min(max(1, self.manifold_contact_count), num_fingers)
        best = None
        best_dist = float("inf")

        for contact_set in combinations(range(num_fingers), keep_count):
            candidate = self._project_to_contact_manifold(xnode, xsample, contact_set)
            q = self.nfo.evaluate(candidate)
            if q < self.min_quality:
                continue
            candidate.quality = q
            dist = grasp_distance(candidate, xsample)
            if dist < best_dist:
                best = candidate
                best_dist = dist

        return best

    def _project_to_contact_manifold(
        self,
        xnode: Grasp,
        xsample: Grasp,
        contact_set: tuple[int, ...],
    ) -> Grasp:
        alpha = 0.35
        new_pos = np.array(xnode.fingertip_positions, copy=True)
        for finger_idx in range(new_pos.shape[0]):
            if finger_idx in contact_set:
                continue
            new_pos[finger_idx] = (
                (1.0 - alpha) * xnode.fingertip_positions[finger_idx]
                + alpha * xsample.fingertip_positions[finger_idx]
            )
        new_pos = new_pos.astype(np.float32)
        new_normals = self._recompute_normals_from_positions(new_pos)

        object_pos_hand = None
        if getattr(xnode, "object_pos_hand", None) is not None and getattr(xsample, "object_pos_hand", None) is not None:
            object_pos_hand = (
                (1.0 - alpha) * np.asarray(xnode.object_pos_hand, dtype=np.float32)
                + alpha * np.asarray(xsample.object_pos_hand, dtype=np.float32)
            ).astype(np.float32)

        object_quat_hand = None
        if getattr(xnode, "object_quat_hand", None) is not None and getattr(xsample, "object_quat_hand", None) is not None:
            object_quat_hand = quat_slerp_np(
                np.asarray(xnode.object_quat_hand, dtype=np.float32),
                np.asarray(xsample.object_quat_hand, dtype=np.float32),
                alpha,
            ).astype(np.float32)

        return Grasp(
            fingertip_positions=new_pos,
            contact_normals=new_normals,
            quality=0.0,
            object_name=xnode.object_name,
            object_scale=xnode.object_scale,
            object_pos_hand=object_pos_hand,
            object_quat_hand=object_quat_hand,
            object_pose_frame=getattr(xnode, "object_pose_frame", None),
        )

    def _recompute_normals_from_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Approximate outward surface normals from fingertip positions.

        For convex objects, the outward normal at a surface point is
        approximately the direction from the object centroid to that point.
        This is exact for spheres and a good approximation for cubes/cylinders.

        Args:
            positions: (F, 3) fingertip positions in object frame

        Returns:
            normals: (F, 3) unit outward normals
        """
        # Object centroid is at origin in object frame
        centroid = positions.mean(axis=0)  # approximate object center
        normals = positions - centroid     # outward direction
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        return (normals / (norms + 1e-8)).astype(np.float32)

    def _perturb_normals(self, normals: np.ndarray) -> np.ndarray:
        """Legacy method kept for compatibility. Use _recompute_normals_from_positions instead."""
        noise = self.rng.normal(0, 0.05, normals.shape).astype(np.float32)
        new_normals = normals + noise
        norms = np.linalg.norm(new_normals, axis=-1, keepdims=True)
        return new_normals / (norms + 1e-8)

    def _build_graph(self, grasp_set: GraspSet) -> GraspGraph:
        N = len(grasp_set)
        # Infer num_fingers from actual grasp data (robust to any hand)
        num_fingers = grasp_set.grasps[0].fingertip_positions.shape[0] if N > 0 else 4
        effective_delta_max = self.delta_max * (1.0 + 0.35 * max(num_fingers - 1, 0))
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if grasp_distance(grasp_set.grasps[i], grasp_set.grasps[j]) < effective_delta_max:
                    edges.append((i, j))
        return GraspGraph(
            grasp_set=grasp_set,
            edges=edges,
            object_name=grasp_set.object_name,
            num_fingers=num_fingers,
        )
