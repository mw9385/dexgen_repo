"""
Mesh Export for DexGraspNet
===========================
Exports ObjectSpec meshes to the directory structure expected by
DexGraspNet's ObjectModel:

    {output_root}/{object_code}/coacd/decomposed.obj

For our parametric primitives (cube, sphere, cylinder) the mesh is
already convex, so CoACD decomposition is unnecessary — we just
export the trimesh directly as an .obj file.

The mesh is normalised to fit within a unit sphere (DexGraspNet convention)
and the scale factor is returned so it can be applied at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh


def export_mesh_for_dexgraspnet(
    mesh: trimesh.Trimesh,
    object_code: str,
    output_root: str | Path,
) -> Tuple[Path, float]:
    """
    Export a trimesh mesh to DexGraspNet's expected directory layout.

    DexGraspNet normalises meshes to fit inside a unit sphere and applies
    a runtime scale factor.  We follow the same convention:

      1. Centre the mesh at the origin.
      2. Compute the scale = max vertex distance from origin.
      3. Save the *unit-normalised* mesh as ``decomposed.obj``.
      4. Return (obj_path, scale) so the caller can pass ``scale`` to
         DexGraspNet's ObjectModel.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Source mesh (already in metres, centred near origin).
    object_code : str
        Unique name for the object (e.g. ``"cube_065"``).
    output_root : path-like
        Root directory.  Files are written to
        ``{output_root}/{object_code}/coacd/decomposed.obj``.

    Returns
    -------
    obj_path : Path
        Absolute path to the exported ``.obj`` file.
    scale : float
        The normalisation scale (max vertex radius in metres).
        Pass this to DexGraspNet's ``object_scale_tensor``.
    """
    output_root = Path(output_root)
    coacd_dir = output_root / object_code / "coacd"
    coacd_dir.mkdir(parents=True, exist_ok=True)

    # Centre the mesh
    centroid = mesh.centroid.copy()
    verts = mesh.vertices - centroid

    # Normalise to unit sphere
    scale = float(np.max(np.linalg.norm(verts, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    unit_verts = verts / scale

    unit_mesh = trimesh.Trimesh(vertices=unit_verts, faces=mesh.faces.copy())
    obj_path = coacd_dir / "decomposed.obj"
    unit_mesh.export(str(obj_path), file_type="obj")

    return obj_path, scale


def export_object_pool(
    object_specs: list,
    output_root: str | Path,
) -> List[Tuple[str, Path, float]]:
    """
    Export an entire ObjectPool to DexGraspNet mesh format.

    Parameters
    ----------
    object_specs : list[ObjectSpec]
        From ``ObjectPool.objects``.
    output_root : path-like
        Root directory for all exported meshes.

    Returns
    -------
    results : list[(object_code, obj_path, scale)]
    """
    output_root = Path(output_root)
    results = []
    for spec in object_specs:
        obj_path, scale = export_mesh_for_dexgraspnet(
            spec.mesh, spec.name, output_root,
        )
        results.append((spec.name, obj_path, scale))
        print(f"[mesh_export] {spec.name}: scale={scale:.4f}m → {obj_path}")
    return results
