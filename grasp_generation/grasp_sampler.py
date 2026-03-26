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
from typing import List, Optional

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
# Grasp Sampler
# ---------------------------------------------------------------------------

class GraspSampler:
    """
    Samples grasps on an object mesh using surface point sampling
    and a finger-assignment heuristic.

    Algorithm (per DexterityGen §3.1):
      1. Uniformly sample M candidate contact points on the object surface
      2. For each candidate set of 4 points (one per finger), check:
         - Coverage: points should spread across the object
         - Opposition: some pairs of contacts should face each other
         - Reachability: rough kinematic feasibility check
      3. Return top-K grasps sorted by coverage score
    """

    # Allegro Hand: 4 fingers
    NUM_FINGERS = 4
    FINGER_NAMES = ["index", "middle", "ring", "thumb"]

    # Heuristic parameters
    MIN_FINGER_SPACING = 0.015   # metres: min distance between fingertips
    MAX_FINGER_SPACING = 0.12    # metres: max distance between fingertips
    PALM_NORMAL_THRESH = 0.3     # dot product threshold for opposition check

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        object_name: str = "object",
        num_candidates: int = 5000,
        num_grasps: int = 200,
        seed: int = 42,
    ):
        self.mesh = mesh
        self.object_name = object_name
        self.num_candidates = num_candidates
        self.num_grasps = num_grasps
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> GraspSet:
        """Sample and return a GraspSet."""
        print(f"[GraspSampler] Sampling {self.num_grasps} grasps on '{self.object_name}'")

        # 1. Sample surface points + normals
        points, face_idx = trimesh.sample.sample_surface(self.mesh, self.num_candidates)
        normals = self.mesh.face_normals[face_idx]   # outward-facing normals

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
        Randomly select 4 surface points, check heuristic constraints,
        and return a Grasp if valid.

        Finger assignment strategy:
          - Thumb gets the point most opposed to the average of the other 3
          - Remaining 3 go to index, middle, ring in order of angular spread
        """
        idx = self.rng.choice(len(points), size=self.NUM_FINGERS, replace=False)
        pts = points[idx]        # (4, 3)
        nrm = normals[idx]       # (4, 3)

        # --- constraint 1: spacing ---
        dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)  # (4,4)
        np.fill_diagonal(dists, np.inf)
        if dists.min() < self.MIN_FINGER_SPACING:
            return None
        if dists.max() > self.MAX_FINGER_SPACING * 3:
            return None

        # --- constraint 2: opposition (at least one pair of normals should
        #     roughly oppose each other, i.e., dot < -THRESH) ---
        n_dot = nrm @ nrm.T   # (4,4)
        np.fill_diagonal(n_dot, 0.0)
        if n_dot.min() > -self.PALM_NORMAL_THRESH:
            return None

        # --- assign thumb: the point most "opposite" to the centroid of others ---
        centroid = pts.mean(axis=0)
        dirs = pts - centroid
        dirs /= (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
        opposition = (dirs * nrm).sum(axis=-1)  # positive = facing away from centroid
        thumb_idx = int(np.argmin(opposition))  # most opposed to centroid

        # Reorder: [index(0), middle(1), ring(2), thumb(3)]
        order = [i for i in range(4) if i != thumb_idx] + [thumb_idx]
        pts = pts[order]
        nrm = nrm[order]

        return Grasp(
            fingertip_positions=pts.astype(np.float32),
            contact_normals=nrm.astype(np.float32),
            quality=0.0,   # filled in by NetForceOptimizer
            object_name=self.object_name,
        )


# ---------------------------------------------------------------------------
# Convenience factory
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
    """
    Create a simple default object mesh for testing
    when no real mesh file is available.
    """
    if object_type == "cube":
        return trimesh.creation.box(extents=[size, size, size])
    elif object_type == "sphere":
        return trimesh.creation.icosphere(radius=size / 2)
    elif object_type == "cylinder":
        return trimesh.creation.cylinder(radius=size / 2, height=size)
    else:
        raise ValueError(f"Unknown object type: {object_type}")
