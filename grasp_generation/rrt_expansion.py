"""
Stage 0 – RRT-based Grasp Set Expansion
=========================================
Expands a seed grasp set by exploring the space of valid grasps
using an RRT (Rapidly-exploring Random Tree) variant.

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
    E = (g_i, g_j) if ||g_i - g_j||_2 < δ_max  (fingertip distance)

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
# Grasp Graph (used as RL task definition in Stage 1)
# ---------------------------------------------------------------------------

@dataclass
class GraspGraph:
    """
    Connected graph of grasps.

    Nodes: grasp configurations (GraspSet)
    Edges: (i, j) pairs reachable by continuous fingertip motion
    """
    grasp_set: GraspSet
    edges: List[Tuple[int, int]] = field(default_factory=list)

    def __len__(self):
        return len(self.grasp_set)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_neighbors(self, node_idx: int) -> List[int]:
        """Return indices of nodes connected to node_idx."""
        neighbors = []
        for i, j in self.edges:
            if i == node_idx:
                neighbors.append(j)
            elif j == node_idx:
                neighbors.append(i)
        return neighbors

    def sample_edge(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
        """Sample a random edge (start_idx, goal_idx)."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, len(self.edges))
        return self.edges[idx]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[GraspGraph] Saved {len(self)} nodes, {self.num_edges} edges to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GraspGraph":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        print(f"[GraspGraph] Loaded {len(obj)} nodes, {obj.num_edges} edges from {path}")
        return obj


# ---------------------------------------------------------------------------
# RRT Expander
# ---------------------------------------------------------------------------

class RRTGraspExpander:
    """
    Expands a seed GraspSet using RRT-style exploration.

    Key parameters (tuned for Allegro Hand, 6 cm cube):
        delta_pos:    max fingertip displacement per expansion step (m)
        delta_max:    max fingertip distance to add an edge (m)
        min_quality:  NFO quality threshold for new grasps
        target_size:  desired final number of grasps
    """

    def __init__(
        self,
        nfo: Optional[NetForceOptimizer] = None,
        delta_pos: float = 0.008,      # 8 mm per step
        delta_max: float = 0.04,       # 4 cm → edge threshold
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, seed_set: GraspSet) -> GraspGraph:
        """
        Expand seed_set to target_size grasps and build a grasp graph.

        Returns a GraspGraph with nodes = grasps and edges = reachable pairs.
        """
        print(f"[RRT] Expanding {len(seed_set)} seed grasps → target {self.target_size}")

        grasps: List[Grasp] = list(seed_set.grasps)

        # Expand
        total_attempts = 0
        while len(grasps) < self.target_size:
            new_grasp = self._expand_step(grasps)
            if new_grasp is not None:
                grasps.append(new_grasp)
                if len(grasps) % 50 == 0:
                    print(f"  [RRT] {len(grasps)}/{self.target_size} grasps")
            total_attempts += 1
            if total_attempts > self.target_size * 200:
                print(f"  [RRT] Warning: stopping early at {len(grasps)} grasps "
                      f"(too many failed attempts)")
                break

        # Build graph
        final_set = GraspSet(grasps=grasps, object_name=seed_set.object_name)
        graph = self._build_graph(final_set)
        print(f"[RRT] Final: {len(graph)} nodes, {graph.num_edges} edges")
        return graph

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _expand_step(self, grasps: List[Grasp]) -> Optional[Grasp]:
        """
        Try to grow the grasp tree by one node:
          1. Pick a random existing grasp (nearest to a random sample)
          2. Perturb it slightly
          3. Check quality
        """
        for _ in range(self.max_attempts):
            # Sample random target in fingertip space
            src = grasps[self.rng.integers(0, len(grasps))]
            noise = self.rng.normal(0, self.delta_pos,
                                    src.fingertip_positions.shape).astype(np.float32)
            new_pos = src.fingertip_positions + noise
            new_normals = self._perturb_normals(src.contact_normals)

            candidate = Grasp(
                fingertip_positions=new_pos,
                contact_normals=new_normals,
                quality=0.0,
                object_name=src.object_name,
            )
            q = self.nfo.evaluate(candidate)
            if q >= self.min_quality:
                candidate.quality = q
                return candidate
        return None

    def _perturb_normals(self, normals: np.ndarray) -> np.ndarray:
        """Slightly perturb surface normals (keep approximately unit)."""
        noise = self.rng.normal(0, 0.05, normals.shape).astype(np.float32)
        new_normals = normals + noise
        norms = np.linalg.norm(new_normals, axis=-1, keepdims=True)
        return new_normals / (norms + 1e-8)

    def _build_graph(self, grasp_set: GraspSet) -> GraspGraph:
        """
        Build edges between grasps that are within delta_max fingertip distance.
        Edge weight = mean fingertip displacement between the two grasps.
        """
        positions = grasp_set.as_array()   # (N, 12)
        N = len(grasp_set)
        edges = []

        for i in range(N):
            # Vectorised distance to all j > i
            diffs = positions[i + 1:] - positions[i]          # (N-i-1, 12)
            dists = np.linalg.norm(diffs.reshape(-1, 4, 3), axis=-1).mean(axis=-1)
            close = np.where(dists < self.delta_max)[0]
            for k in close:
                j = i + 1 + k
                edges.append((i, j))

        return GraspGraph(grasp_set=grasp_set, edges=edges)
