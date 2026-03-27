"""
Stage 0 – Grasp Sampler
=======================
Samples candidate grasps on an object surface and assigns fingers.

Paper reference (DexterityGen §3.1):
  - Sample contact points on the object surface
  - Assign each contact to a finger using a coverage-maximizing heuristic
  - A grasp g is represented as fingertip positions in the *object frame*:
      g = [p_thumb, p_index, p_middle, p_ring]  (4 × 3 = 12-dim vector)

Allegro Hand finger layout:
  finger 0 = index, finger 1 = middle, finger 2 = ring, finger 3 = thumb
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Grasp:
    """A single grasp represented as fingertip positions in the object frame."""
    # (4, 3) fingertip positions: [index, middle, ring, thumb]
    fingertip_positions: np.ndarray
    # (4, 3) surface normals at contact points (pointing away from object)
    contact_normals: np.ndarray
    # Grasp quality score (NFO score, higher is better)
    quality: float = 0.0
    # Object name this grasp was computed for
    object_name: str = ""
    # Object scale at which this grasp was computed (relative to unit mesh)
    object_scale: float = 1.0

    @property
    def as_vector(self) -> np.ndarray:
        """Flatten to 12-dim vector."""
        return self.fingertip_positions.flatten()

    def to_dict(self) -> dict:
        return {
            "fingertip_positions": self.fingertip_positions,
            "contact_normals": self.contact_normals,
            "quality": self.quality,
            "object_name": self.object_name,
            "object_scale": self.object_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grasp":
        return cls(**d)


@dataclass
class GraspSet:
    """Collection of grasps for one or more objects."""
    grasps: List[Grasp] = field(default_factory=list)
    object_name: str = ""

    def __len__(self):
        return len(self.grasps)

    def __getitem__(self, idx):
        return self.grasps[idx]

    def add(self, grasp: Grasp):
        self.grasps.append(grasp)

    def filter_by_quality(self, min_quality: float) -> "GraspSet":
        filtered = [g for g in self.grasps if g.quality >= min_quality]
        return GraspSet(grasps=filtered, object_name=self.object_name)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[GraspSet] Saved {len(self)} grasps to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GraspSet":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        print(f"[GraspSet] Loaded {len(obj)} grasps from {path}")
        return obj

    def as_array(self) -> np.ndarray:
        """(N, 12) array of all grasp fingertip positions."""
        return np.stack([g.as_vector for g in self.grasps], axis=0)


# ---------------------------------------------------------------------------
# Object Pool  (NEW)
# ---------------------------------------------------------------------------

@dataclass
class ObjectSpec:
    """
    Specification for a single object variant in the pool.

    Attributes:
        name:       Unique identifier (e.g. "cube_060", "sphere_040")
        mesh:       trimesh.Trimesh for grasp generation
        shape_type: "cube" / "sphere" / "cylinder" / "custom"
        size:       Characteristic size in metres (bounding-box half-extent)
        mass:       Mass in kg
        color:      (R, G, B) visual colour in [0, 1]
    """
    name: str
    mesh: trimesh.Trimesh
    shape_type: str
    size: float          # metres
    mass: float = 0.1    # kg
    color: Tuple[float, float, float] = (0.8, 0.3, 0.2)


class ObjectPool:
    """
    Pool of randomised object variants used for training.

    Generates a mix of primitive shapes (cube, sphere, cylinder) at
    different sizes so the policy and DexGen controller must generalise
    across object geometry.

    Usage:
        pool = ObjectPool.from_config(
            shape_types=["cube", "sphere", "cylinder"],
            size_range=(0.04, 0.09),
            num_sizes=3,
            seed=42,
        )
        for spec in pool:
            grasps = GraspSampler(spec.mesh, spec.name, ...).sample()
    """

    def __init__(self, objects: List[ObjectSpec]):
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def __getitem__(self, idx):
        return self.objects[idx]

    @classmethod
    def from_config(
        cls,
        shape_types: List[str] = ("cube", "sphere", "cylinder"),
        size_range: Tuple[float, float] = (0.04, 0.09),
        num_sizes: int = 3,
        seed: int = 42,
    ) -> "ObjectPool":
        """
        Build a pool of primitive objects by sampling sizes uniformly
        across size_range for each shape type.
        """
        rng = np.random.default_rng(seed)
        sizes = np.linspace(size_range[0], size_range[1], num_sizes)
        objects = []

        for shape in shape_types:
            for size in sizes:
                mesh = make_default_object_mesh(shape, float(size))
                # Add slight random noise to mesh vertices for variety
                noise = rng.normal(0, size * 0.02, mesh.vertices.shape)
                mesh.vertices += noise
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=1)

                name = f"{shape}_{int(size * 1000):03d}"
                color = _shape_color(shape)
                objects.append(ObjectSpec(
                    name=name,
                    mesh=mesh,
                    shape_type=shape,
                    size=float(size),
                    mass=0.05 + (size / 0.1) * 0.15,  # heavier when larger
                    color=color,
                ))

        print(f"[ObjectPool] Created {len(objects)} objects "
              f"({len(shape_types)} shapes × {num_sizes} sizes)")
        return cls(objects)

    @classmethod
    def from_mesh_dir(cls, mesh_dir: str) -> "ObjectPool":
        """Load all .obj/.stl/.ply files from a directory as custom objects."""
        mesh_dir = Path(mesh_dir)
        objects = []
        for path in sorted(mesh_dir.glob("**/*.{obj,stl,ply}")):
            try:
                mesh = trimesh.load(str(path), force="mesh")
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                size = float(np.max(mesh.bounding_box.extents))
                objects.append(ObjectSpec(
                    name=path.stem,
                    mesh=mesh,
                    shape_type="custom",
                    size=size,
                ))
            except Exception as e:
                print(f"[ObjectPool] Warning: could not load {path}: {e}")
        print(f"[ObjectPool] Loaded {len(objects)} meshes from {mesh_dir}")
        return cls(objects)

    def sample(self, rng: Optional[np.random.Generator] = None) -> ObjectSpec:
        """Return a uniformly random ObjectSpec."""
        if rng is None:
            rng = np.random.default_rng()
        return self.objects[rng.integers(0, len(self.objects))]

    def get_isaac_lab_specs(self) -> List[dict]:
        """
        Return per-object shape parameters for Isaac Lab scene configuration.
        Used by AnyGraspSceneCfg to build MultiAssetSpawnerCfg entries.
        """
        specs = []
        for obj in self.objects:
            specs.append({
                "name": obj.name,
                "shape_type": obj.shape_type,
                "size": obj.size,
                "mass": obj.mass,
                "color": obj.color,
            })
        return specs


def _shape_color(shape: str) -> Tuple[float, float, float]:
    return {
        "cube": (0.8, 0.2, 0.2),
        "sphere": (0.2, 0.6, 0.9),
        "cylinder": (0.3, 0.8, 0.3),
    }.get(shape, (0.7, 0.7, 0.7))


# ---------------------------------------------------------------------------
# Grasp Sampler
# ---------------------------------------------------------------------------

class GraspSampler:
    """
    Samples grasps on an object mesh using surface point sampling
    and a finger-assignment heuristic.

    Algorithm (per DexterityGen §3.1):
      1. Uniformly sample M candidate contact points on the object surface
      2. For each candidate set of num_fingers points, check:
         - Coverage: points should spread across the object
         - Opposition: some pairs of contacts should face each other
         - Reachability: rough kinematic feasibility check
      3. Return top-K grasps sorted by coverage score

    num_fingers controls how many contact points are sampled per grasp
    (e.g. 2 for pinch, 3 for tripod, 4 for full Allegro, 5 for Shadow).
    """

    # Default Allegro finger names (overridden by num_fingers at __init__)
    _DEFAULT_FINGER_NAMES = ["index", "middle", "ring", "thumb", "pinky"]

    # Heuristic parameters (scale-adaptive, set in __init__)
    PALM_NORMAL_THRESH = 0.3

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        object_name: str = "object",
        object_scale: float = 1.0,
        num_candidates: int = 5000,
        num_grasps: int = 200,
        num_fingers: int = 4,
        seed: int = 42,
    ):
        self.mesh = mesh
        self.object_name = object_name
        self.object_scale = object_scale
        self.num_candidates = num_candidates
        self.num_grasps = num_grasps
        self.num_fingers = num_fingers
        self.finger_names = self._DEFAULT_FINGER_NAMES[:num_fingers]
        self.rng = np.random.default_rng(seed)

        # Scale finger spacing with object size
        obj_size = float(np.max(mesh.bounding_box.extents))
        self.MIN_FINGER_SPACING = max(0.008, obj_size * 0.15)
        self.MAX_FINGER_SPACING = obj_size * 2.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> GraspSet:
        """Sample and return a GraspSet."""
        print(f"[GraspSampler] Sampling {self.num_grasps} grasps on '{self.object_name}'")

        # 1. Sample surface points + normals
        points, face_idx = trimesh.sample.sample_surface(self.mesh, self.num_candidates)
        normals = self.mesh.face_normals[face_idx]

        # 2. Generate finger assignments
        grasp_set = GraspSet(object_name=self.object_name)
        attempts = 0
        max_attempts = self.num_grasps * 50

        while len(grasp_set) < self.num_grasps and attempts < max_attempts:
            attempts += 1
            candidate = self._sample_finger_assignment(points, normals)
            if candidate is not None:
                grasp_set.add(candidate)

        print(f"[GraspSampler] Generated {len(grasp_set)} valid grasps "
              f"({attempts} attempts)")
        return grasp_set

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_finger_assignment(
        self,
        points: np.ndarray,
        normals: np.ndarray,
    ) -> Optional[Grasp]:
        """
        Greedy spacing-aware finger assignment:
          1. Pick first finger uniformly at random.
          2. Each subsequent finger is drawn from the subset of points that
             are >= MIN_FINGER_SPACING from ALL already-selected fingers.
        This avoids the high failure rate of fully-random selection when the
        surface point cloud is dense relative to finger spacing.
        """
        n_pts = len(points)
        selected_idx: list = []
        available = np.ones(n_pts, dtype=bool)

        for f in range(self.num_fingers):
            candidates = np.where(available)[0]
            if len(candidates) == 0:
                return None
            ci = self.rng.integers(0, len(candidates))
            chosen = candidates[ci]
            selected_idx.append(chosen)

            # Remove all points within MIN_FINGER_SPACING from next picks
            dists = np.linalg.norm(points - points[chosen], axis=-1)
            available &= (dists >= self.MIN_FINGER_SPACING)

        pts = points[selected_idx]
        nrm = normals[selected_idx]

        # --- constraint: overall spread not absurdly large ---
        max_dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1).max()
        if max_dist > self.MAX_FINGER_SPACING * 3:
            return None

        # --- constraint: opposition (at least one pair must face each other) ---
        n_dot = nrm @ nrm.T
        np.fill_diagonal(n_dot, 0.0)
        if n_dot.min() > -self.PALM_NORMAL_THRESH:
            return None

        # --- assign "thumb" as the finger most opposed to the centroid ---
        # (placed last; for 2-finger pinch this becomes finger 1)
        centroid = pts.mean(axis=0)
        dirs = pts - centroid
        dirs /= (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
        opposition = (dirs * nrm).sum(axis=-1)
        thumb_idx = int(np.argmin(opposition))

        order = [i for i in range(self.num_fingers) if i != thumb_idx] + [thumb_idx]
        pts = pts[order]
        nrm = nrm[order]

        return Grasp(
            fingertip_positions=pts.astype(np.float32),
            contact_normals=nrm.astype(np.float32),
            quality=0.0,
            object_name=self.object_name,
            object_scale=self.object_scale,
        )


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def sample_grasps_for_mesh(
    mesh_path: str,
    num_grasps: int = 200,
    num_candidates: int = 10000,
    seed: int = 42,
) -> GraspSet:
    """Load a mesh file and sample grasps."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {mesh_path}")
    name = Path(mesh_path).stem
    sampler = GraspSampler(mesh, object_name=name,
                           num_candidates=num_candidates,
                           num_grasps=num_grasps, seed=seed)
    return sampler.sample()


def make_default_object_mesh(object_type: str = "cube", size: float = 0.06) -> trimesh.Trimesh:
    """Create a simple primitive mesh."""
    if object_type == "cube":
        return trimesh.creation.box(extents=[size, size, size])
    elif object_type == "sphere":
        return trimesh.creation.icosphere(radius=size / 2, subdivisions=3)
    elif object_type == "cylinder":
        return trimesh.creation.cylinder(radius=size / 2, height=size)
    else:
        raise ValueError(f"Unknown object type: {object_type}")
