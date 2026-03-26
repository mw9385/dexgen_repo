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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grasp_sampler import Grasp, GraspSet
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
    """
    grasp_set: GraspSet
    edges: List[Tuple[int, int]] = field(default_factory=list)
    object_name: str = ""

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
        seed: int = 42,
    ):
        self.nfo = nfo or NetForceOptimizer(min_quality=min_quality, fast_mode=True)
        self.delta_pos = delta_pos
        self.delta_max = delta_max
        self.min_quality = min_quality
        self.target_size = target_size
        self.max_attempts = max_attempts_per_step
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
            src = grasps[self.rng.integers(0, len(grasps))]
            noise = self.rng.normal(0, self.delta_pos,
                                    src.fingertip_positions.shape).astype(np.float32)
            candidate = Grasp(
                fingertip_positions=src.fingertip_positions + noise,
                contact_normals=self._perturb_normals(src.contact_normals),
                quality=0.0,
                object_name=src.object_name,
                object_scale=src.object_scale,
            )
            q = self.nfo.evaluate(candidate)
            if q >= self.min_quality:
                candidate.quality = q
                return candidate
        return None

    def _perturb_normals(self, normals: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0, 0.05, normals.shape).astype(np.float32)
        new_normals = normals + noise
        norms = np.linalg.norm(new_normals, axis=-1, keepdims=True)
        return new_normals / (norms + 1e-8)

    def _build_graph(self, grasp_set: GraspSet) -> GraspGraph:
        positions = grasp_set.as_array()   # (N, 12)
        N = len(grasp_set)
        edges = []
        for i in range(N):
            diffs = positions[i + 1:] - positions[i]
            dists = np.linalg.norm(diffs.reshape(-1, 4, 3), axis=-1).mean(axis=-1)
            for k in np.where(dists < self.delta_max)[0]:
                edges.append((i, i + 1 + k))
        return GraspGraph(
            grasp_set=grasp_set,
            edges=edges,
            object_name=grasp_set.object_name,
        )
