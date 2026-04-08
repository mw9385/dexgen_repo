from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from .rrt_expansion import GraspGraph, MultiObjectGraspGraph
from .grasp_sampler import Grasp, GraspSet


def load_npy_as_graph(path: str | Path) -> MultiObjectGraspGraph:
    """
    Load a sharpa-style .npy grasp cache and convert to MultiObjectGraspGraph.

    Input: (N, 29) array = [22 joint_pos | 3 obj_pos | 4 obj_quat]
    Output: MultiObjectGraspGraph with edges based on quaternion distance.

    Filename convention: sharpa_grasp_{shape}_{size_mm}.npy
    """
    path = Path(path)
    data = np.load(str(path))  # (N, 29)
    N = data.shape[0]

    # Parse shape/size from filename (e.g., sharpa_grasp_cube_050.npy)
    stem = path.stem  # e.g., "sharpa_grasp_cube_050"
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

    obj_name = f"{shape_type}_{int(size * 1000):03d}_f5"

    grasps = []
    for i in range(N):
        row = data[i]
        joint_angles = row[:22].astype(np.float32)
        obj_pos = row[22:25].astype(np.float32)
        obj_quat = row[25:29].astype(np.float32)

        grasps.append(Grasp(
            fingertip_positions=np.zeros((5, 3), dtype=np.float32),  # not used for solved graph
            contact_normals=np.zeros((5, 3), dtype=np.float32),
            quality=1.0,
            object_name=obj_name,
            object_scale=size,
            joint_angles=joint_angles,
            object_pos_hand=obj_pos,
            object_quat_hand=obj_quat,
            object_pose_frame="hand_root",
        ))

    # Build edges: quaternion distance between all pairs
    quats = data[:, 25:29].astype(np.float64)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            dot = abs(np.dot(quats[i], quats[j]))
            orn_dist = 2.0 * np.arccos(min(dot, 1.0))
            if orn_dist < 1.0:  # ~57 degrees — reachable transition
                edges.append((i, j))

    grasp_set = GraspSet(grasps=grasps, object_name=obj_name)
    graph = GraspGraph(
        grasp_set=grasp_set, edges=edges,
        object_name=obj_name, num_fingers=5,
    )

    multi = MultiObjectGraspGraph()
    multi.add(graph, {
        "name": obj_name, "shape_type": shape_type,
        "size": size, "num_fingers": 5,
    })

    print(f"[graph_io] Loaded .npy: {N} grasps, {len(edges)} edges "
          f"→ {obj_name}")
    return multi


def parse_graph_paths(paths: str | Path | Iterable[str | Path] | None) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        raw_items = [paths]
    else:
        raw_items = list(paths)

    resolved: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        for part in str(item).split(","):
            part = part.strip()
            if part:
                resolved.append(part)
    return resolved


def load_graph(path: str | Path):
    path = Path(path)
    if path.suffix == ".npy":
        return load_npy_as_graph(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_multi_object_graph(graph, source_path: str | Path | None = None) -> MultiObjectGraspGraph:
    if isinstance(graph, MultiObjectGraspGraph):
        return graph
    if isinstance(graph, GraspGraph):
        multi = MultiObjectGraspGraph()
        graph_name = graph.object_name or Path(source_path or "graph").stem
        multi.add(
            graph,
            {
                "name": graph_name,
                "shape_type": "cube",
                "size": 0.06,
                "mass": 0.1,
                "color": (0.8, 0.2, 0.2),
            },
        )
        return multi
    raise TypeError(f"Unsupported grasp graph type: {type(graph)!r}")


def merge_graphs(graphs: Iterable[tuple[str | Path | None, object]]) -> MultiObjectGraspGraph:
    merged = MultiObjectGraspGraph()
    for source_path, graph in graphs:
        multi = ensure_multi_object_graph(graph, source_path=source_path)
        for name, subgraph in multi.graphs.items():
            if name in merged.graphs:
                raise ValueError(
                    f"Duplicate object graph name '{name}' while merging grasp graphs. "
                    "Rename the conflicting object entries or pass non-overlapping PKL files."
                )
            merged.graphs[name] = subgraph
            if name in multi.object_specs:
                merged.object_specs[name] = dict(multi.object_specs[name])
    return merged


def load_merged_graph(paths: str | Path | Iterable[str | Path] | None):
    graph_paths = parse_graph_paths(paths)
    if not graph_paths:
        return None
    loaded = [(path, load_graph(path)) for path in graph_paths]
    if len(loaded) == 1:
        return ensure_multi_object_graph(loaded[0][1], source_path=loaded[0][0])
    return merge_graphs(loaded)
