"""Backward-compatible exports for legacy grasp graph pickle files.

Older saved graphs were pickled with classes under ``grasp_generation.rrt_expansion``.
The current codebase moved those dataclasses into ``grasp_generation.graph_io``.
Keeping this shim allows existing ``.pkl`` assets to unpickle without conversion.
"""

from .graph_io import Grasp, GraspGraph, GraspSet, MultiObjectGraspGraph

__all__ = ["Grasp", "GraspSet", "GraspGraph", "MultiObjectGraspGraph"]
