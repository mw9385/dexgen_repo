from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from .rrt_expansion import GraspGraph, MultiObjectGraspGraph


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
    with open(Path(path), "rb") as f:
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
