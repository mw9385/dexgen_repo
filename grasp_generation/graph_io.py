"""
Grasp data structures + IO for .npy grasp caches.

Supports:
  - .npy (sharpa-style): (N, 29) = 22 joints + 3 obj_pos + 4 obj_quat
  - .pkl (legacy GraspGraph): pickle format

Cube .npy files are automatically augmented with 24 symmetry rotations
at load time, expanding orientation coverage from ~60° to full SO(3).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures (simplified from former grasp_sampler.py + rrt_expansion.py)
# ---------------------------------------------------------------------------

@dataclass
class Grasp:
    """A single grasp state."""
    fingertip_positions: np.ndarray  # (F, 3)
    contact_normals: np.ndarray      # (F, 3)
    quality: float = 0.0
    object_name: str = ""
    object_scale: float = 1.0
    joint_angles: Optional[np.ndarray] = None  # (22,) for Sharpa
    object_pos_hand: Optional[np.ndarray] = None  # (3,)
    object_quat_hand: Optional[np.ndarray] = None  # (4,) w,x,y,z
    object_pose_frame: Optional[str] = None


@dataclass
class GraspSet:
    grasps: List[Grasp] = field(default_factory=list)
    object_name: str = ""
    def __len__(self): return len(self.grasps)
    def __getitem__(self, idx): return self.grasps[idx]


@dataclass
class GraspGraph:
    grasp_set: GraspSet
    edges: List[Tuple[int, int]] = field(default_factory=list)
    object_name: str = ""
    num_fingers: int = 5

    def __len__(self): return len(self.grasp_set)

    @property
    def num_edges(self): return len(self.edges)

    def get_neighbors(self, node_idx: int) -> List[int]:
        neighbors = []
        for i, j in self.edges:
            if i == node_idx: neighbors.append(j)
            elif j == node_idx: neighbors.append(i)
        return neighbors

    def sample_edge(self, rng=None):
        if rng is None: rng = np.random.default_rng()
        return self.edges[rng.integers(0, len(self.edges))]


@dataclass
class MultiObjectGraspGraph:
    graphs: Dict[str, GraspGraph] = field(default_factory=dict)
    object_specs: Dict[str, dict] = field(default_factory=dict)

    def __len__(self): return sum(len(g) for g in self.graphs.values())

    @property
    def num_objects(self): return len(self.graphs)

    @property
    def object_names(self): return list(self.graphs.keys())

    @property
    def num_fingers(self):
        if not self.graphs: return 5
        return next(iter(self.graphs.values())).num_fingers

    def add(self, graph: GraspGraph, spec: dict):
        self.graphs[graph.object_name] = graph
        self.object_specs[graph.object_name] = spec

    def sample_object(self, rng=None):
        if rng is None: rng = np.random.default_rng()
        names = self.object_names
        return names[rng.integers(0, len(names))]

    def get_grasp(self, object_name, grasp_idx):
        return self.graphs[object_name].grasp_set[grasp_idx]

    def summary(self):
        print(f"[MultiObjectGraspGraph] {self.num_objects} objects:")
        for name, g in self.graphs.items():
            print(f"  {name}: {len(g)} grasps, {g.num_edges} edges")


# ---------------------------------------------------------------------------
# Cube symmetry augmentation (24 rotations)
# ---------------------------------------------------------------------------

def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-8)
    h = angle * 0.5
    return np.array([np.cos(h), *(a * np.sin(h))], dtype=np.float64)


def _cube_symmetry_quats() -> np.ndarray:
    """24 quaternions of the cube rotation group. Returns (24, 4)."""
    quats = [np.array([1, 0, 0, 0], dtype=np.float64)]
    # Face rotations: 90°, 180°, 270° around x, y, z
    for ax in [[1,0,0], [0,1,0], [0,0,1]]:
        for ang in [np.pi/2, np.pi, 3*np.pi/2]:
            quats.append(_axis_angle_quat(ax, ang))
    # Edge rotations: 180° around face diagonals
    for ax in [[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]]:
        quats.append(_axis_angle_quat(ax, np.pi))
    # Vertex rotations: 120°, 240° around body diagonals
    for ax in [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]]:
        for ang in [2*np.pi/3, 4*np.pi/3]:
            quats.append(_axis_angle_quat(ax, ang))
    arr = np.stack(quats)
    return arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8)


def _augment_cube_symmetry(data: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply a random cube symmetry rotation to each grasp.

    Input: (N, 29). Returns (N, 29) — same count, wider orientation spread.
    Each grasp gets one of the 24 cube symmetry rotations chosen at random.
    Joint angles and object position are unchanged; only object_quat_hand
    is rotated. Valid because a cube is invariant under these rotations.
    """
    sym_quats = _cube_symmetry_quats()  # (24, 4)
    N = data.shape[0]
    rng = np.random.default_rng(seed)

    out = data.copy()
    # Pick a random symmetry rotation for each grasp
    indices = rng.integers(0, 24, size=N)
    chosen = sym_quats[indices]  # (N, 4)

    obj_q = out[:, 25:29].astype(np.float64)
    new_q = _qmul(obj_q, chosen)
    new_q = new_q / (np.linalg.norm(new_q, axis=-1, keepdims=True) + 1e-8)
    out[:, 25:29] = new_q.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# .npy loader
# ---------------------------------------------------------------------------

def load_npy_as_graph(path: str | Path) -> MultiObjectGraspGraph:
    """
    Load sharpa-style .npy grasp cache → MultiObjectGraspGraph.

    Input: (N, 29) = [22 joint_pos | 3 obj_pos | 4 obj_quat]
    Filename convention: sharpa_grasp_{shape}_{size_mm}.npy

    Cube files are automatically augmented with 24 symmetry rotations
    (N → N*24) to expand orientation coverage to the full rotation space.
    """
    path = Path(path)
    data = np.load(str(path))
    N_orig = data.shape[0]

    # Parse shape/size from filename
    stem = path.stem
    parts = stem.split("_")
    shape_type = "cube"
    size = 0.05
    for i, p in enumerate(parts):
        if p in ("cube", "sphere", "cylinder"):
            shape_type = p
            if i + 1 < len(parts):
                try:
                    size = int(parts[i + 1]) / 1000.0
                except ValueError:
                    pass

    # Auto-augment cubes with random symmetry rotations
    if shape_type == "cube":
        data = _augment_cube_symmetry(data)
        print(f"[graph_io] Cube symmetry augmentation applied to {N_orig} grasps")

    N = data.shape[0]
    obj_name = f"{shape_type}_{int(size * 1000):03d}_f5"

    grasps = []
    for i in range(N):
        row = data[i]
        grasps.append(Grasp(
            fingertip_positions=np.zeros((5, 3), dtype=np.float32),
            contact_normals=np.zeros((5, 3), dtype=np.float32),
            quality=1.0,
            object_name=obj_name,
            object_scale=size,
            joint_angles=row[:22].astype(np.float32),
            object_pos_hand=row[22:25].astype(np.float32),
            object_quat_hand=row[25:29].astype(np.float32),
            object_pose_frame="hand_root",
        ))

    # NOTE: goal sampling (_sample_nearby_goal_index) computes nearest
    # neighbours on-the-fly from cached quaternion / position arrays — it
    # does NOT read the `edges` list. Keep edges empty.
    grasp_set = GraspSet(grasps=grasps, object_name=obj_name)
    graph = GraspGraph(
        grasp_set=grasp_set, edges=[],
        object_name=obj_name, num_fingers=5,
    )

    multi = MultiObjectGraspGraph()
    multi.add(graph, {
        "name": obj_name, "shape_type": shape_type,
        "size": size, "num_fingers": 5,
    })

    print(f"[graph_io] Loaded .npy: {N} grasps → {obj_name}")
    return multi


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------

def parse_graph_paths(paths) -> list[str]:
    if paths is None: return []
    if isinstance(paths, (str, Path)): raw_items = [paths]
    else: raw_items = list(paths)
    resolved = []
    for item in raw_items:
        if item is None: continue
        for part in str(item).split(","):
            part = part.strip()
            if part: resolved.append(part)
    return resolved


def load_graph(path: str | Path):
    path = Path(path)
    if path.suffix == ".npy":
        return load_npy_as_graph(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_multi_object_graph(graph, source_path=None) -> MultiObjectGraspGraph:
    if isinstance(graph, MultiObjectGraspGraph):
        return graph
    if isinstance(graph, GraspGraph):
        multi = MultiObjectGraspGraph()
        name = graph.object_name or Path(source_path or "graph").stem
        multi.add(graph, {"name": name, "shape_type": "cube", "size": 0.05})
        return multi
    raise TypeError(f"Unsupported grasp graph type: {type(graph)!r}")


def load_merged_graph(paths):
    graph_paths = parse_graph_paths(paths)
    if not graph_paths: return None
    loaded = [(p, load_graph(p)) for p in graph_paths]
    if len(loaded) == 1:
        return ensure_multi_object_graph(loaded[0][1], source_path=loaded[0][0])
    merged = MultiObjectGraspGraph()
    for source_path, graph in loaded:
        multi = ensure_multi_object_graph(graph, source_path=source_path)
        for name, subgraph in multi.graphs.items():
            merged.graphs[name] = subgraph
            if name in multi.object_specs:
                merged.object_specs[name] = dict(multi.object_specs[name])
    return merged
