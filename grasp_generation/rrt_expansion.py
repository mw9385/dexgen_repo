"""
Grasp Graph data structures.

GraspGraph: per-object graph of grasp configurations.
MultiObjectGraspGraph: collection of per-object graphs for object pool training.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grasp_sampler import Grasp, GraspSet


# ---------------------------------------------------------------------------
# GraspGraph (single object)
# ---------------------------------------------------------------------------

@dataclass
class GraspGraph:
    """
    Connected graph of grasps for a single object.

    Nodes: grasp configurations (GraspSet)
    Edges: (i, j) pairs reachable by continuous motion
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
# MultiObjectGraspGraph (object pool)
# ---------------------------------------------------------------------------

@dataclass
class MultiObjectGraspGraph:
    """
    Collection of per-object GraspGraphs for object pool training.

    At each RL episode reset:
      1. Sample a random object name from self.graphs
      2. Sample a random edge from graphs[object_name]
      → (start_grasp, goal_grasp) + which object to spawn
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
        if not self.graphs:
            return 4
        return next(iter(self.graphs.values())).num_fingers

    def add(self, graph: GraspGraph, spec: dict):
        self.graphs[graph.object_name] = graph
        self.object_specs[graph.object_name] = spec

    def sample_object(self, rng: Optional[np.random.Generator] = None) -> str:
        if rng is None:
            rng = np.random.default_rng()
        names = self.object_names
        return names[rng.integers(0, len(names))]

    def sample_edge(
        self,
        object_name: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[str, Tuple[int, int]]:
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
