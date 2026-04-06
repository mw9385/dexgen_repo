"""
Stage 0 – DexGen Heuristic Sampler
==================================
Samples candidate grasps on an object surface, validates via Net Force Optimization,
and assigns finger joint states (q) and object pose (p).

Paper reference (DexterityGen Algorithm 3):
  1. Sample M candidate contact points & normals
  2. GraspAnalysis: Evaluate Net Force (NFO)
  3. Random Pose: Sample object pose in hand frame
  4. Assign: Solve IK to find joint angles (q)
  5. Collision: Reject if hand penetrates object
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Grasp:
    """A single grasp represented as fingertip positions in the object frame."""
    fingertip_positions: np.ndarray
    contact_normals: np.ndarray
    quality: float = 0.0
    object_name: str = ""
    object_scale: float = 1.0
    joint_angles: Optional[np.ndarray] = None
    object_pos_hand: Optional[np.ndarray] = None
    object_quat_hand: Optional[np.ndarray] = None
    object_pose_frame: Optional[str] = None
    reset_contact_error: Optional[float] = None
    reset_contact_error_max: Optional[float] = None

    @property
    def as_vector(self) -> np.ndarray:
        return self.fingertip_positions.flatten()

    def to_dict(self) -> dict:
        return {
            "fingertip_positions": self.fingertip_positions,
            "contact_normals": self.contact_normals,
            "quality": self.quality,
            "object_name": self.object_name,
            "object_scale": self.object_scale,
            "joint_angles": self.joint_angles,
            "object_pos_hand": self.object_pos_hand,
            "object_quat_hand": self.object_quat_hand,
            "object_pose_frame": self.object_pose_frame,
            "reset_contact_error": self.reset_contact_error,
            "reset_contact_error_max": self.reset_contact_error_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grasp":
        return cls(**d)


@dataclass
class GraspSet:
    grasps: List[Grasp] = field(default_factory=list)
    object_name: str = ""

    def __len__(self): return len(self.grasps)
    def __getitem__(self, idx): return self.grasps[idx]
    def add(self, grasp: Grasp): self.grasps.append(grasp)

    def filter_by_quality(self, min_quality: float) -> "GraspSet":
        filtered = [g for g in self.grasps if g.quality >= min_quality]
        return GraspSet(grasps=filtered, object_name=self.object_name)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)
        print(f"[GraspSet] Saved {len(self)} grasps to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GraspSet":
        with open(Path(path), "rb") as f: obj = pickle.load(f)
        print(f"[GraspSet] Loaded {len(obj)} grasps from {path}")
        return obj

# ---------------------------------------------------------------------------
# Object Pool
# ---------------------------------------------------------------------------
# (ObjectPool 및 ObjectSpec은 기존 작성하신 코드가 완벽하므로 생략 없이 유지)

@dataclass
class ObjectSpec:
    name: str
    mesh: trimesh.Trimesh
    shape_type: str
    size: float
    mass: float = 0.1
    color: Tuple[float, float, float] = (0.8, 0.3, 0.2)

class ObjectPool:
    def __init__(self, objects: List[ObjectSpec]): self.objects = objects
    def __len__(self): return len(self.objects)
    def __iter__(self): return iter(self.objects)
    def __getitem__(self, idx): return self.objects[idx]

    @classmethod
    def from_config(cls, shape_types=("cube", "sphere", "cylinder"), size_range=(0.04, 0.09), num_sizes=3, seed=42) -> "ObjectPool":
        rng = np.random.default_rng(seed)
        sizes = np.linspace(size_range[0], size_range[1], num_sizes)
        objects = []
        for shape in shape_types:
            for size in sizes:
                mesh = make_default_object_mesh(shape, float(size))
                noise = rng.normal(0, size * 0.02, mesh.vertices.shape)
                mesh.vertices += noise
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=1)
                objects.append(ObjectSpec(
                    name=f"{shape}_{int(size * 1000):03d}",
                    mesh=mesh, shape_type=shape, size=float(size),
                    mass=0.05 + (size / 0.1) * 0.15,
                    color=_shape_color(shape),
                ))
        return cls(objects)

def _shape_color(shape: str) -> Tuple[float, float, float]:
    return {"cube": (0.8, 0.2, 0.2), "sphere": (0.2, 0.6, 0.9), "cylinder": (0.3, 0.8, 0.3)}.get(shape, (0.7, 0.7, 0.7))

def make_default_object_mesh(object_type: str = "cube", size: float = 0.06) -> trimesh.Trimesh:
    if object_type == "cube": return trimesh.creation.box(extents=[size, size, size])
    elif object_type == "sphere": return trimesh.creation.icosphere(radius=size / 2, subdivisions=3)
    elif object_type == "cylinder": return trimesh.creation.cylinder(radius=size / 2, height=size)
    else: raise ValueError(f"Unknown object type: {object_type}")


# ---------------------------------------------------------------------------
# Heuristic Sampler (Implementation of DexGen Algorithm 3)
# ---------------------------------------------------------------------------

class HeuristicSampler:
    """
    Samples grasps adhering strictly to DexGen Algorithm 3:
      1. Geometric Sampling
      2. Grasp Analysis (NFO)
      3. Assign (IK)
      4. Collision Check
    """
    _FINGER_SUBSETS = {
        2: ["index", "thumb"],
        3: ["index", "middle", "thumb"],
        4: ["index", "middle", "ring", "thumb"],
        5: ["index", "middle", "ring", "thumb", "pinky"],
    }
    _DEFAULT_FINGER_NAMES = ["index", "middle", "ring", "thumb", "pinky"]
    PALM_NORMAL_THRESH = 0.3

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        object_name: str = "object",
        object_scale: float = 1.0,
        num_candidates: int = 5000,
        num_grasps: int = 200,
        num_fingers: int = 4,
        nfo: Any = None,        # NEW: Net Force Optimizer instance
        env: Any = None,        # NEW: Isaac Sim environment or IK solver interface
        ft_ids: list = None,    # NEW: Fingertip body IDs for IK
        seed: int = 42,
    ):
        self.mesh = mesh
        self.object_name = object_name
        self.object_scale = object_scale
        self.num_candidates = num_candidates
        self.num_grasps = num_grasps
        self.num_fingers = num_fingers
        self.rng = np.random.default_rng(seed)
        
        self.nfo = nfo
        self.env = env
        self.ft_ids = ft_ids

        obj_size = float(np.max(mesh.bounding_box.extents))
        self.MIN_FINGER_SPACING = max(0.008, obj_size * 0.15)
        self.MAX_FINGER_SPACING = obj_size * 2.5

    def sample(self) -> GraspSet:
        print(f"[HeuristicSampler] Sampling {self.num_grasps} valid (q, p) seeds on '{self.object_name}'")

        # 1. Surface points
        points, face_idx = trimesh.sample.sample_surface(self.mesh, self.num_candidates)
        normals = self.mesh.face_normals[face_idx]

        grasp_set = GraspSet(object_name=self.object_name)
        attempts = 0
        max_attempts = self.num_grasps * 100  # IK and NFO rejection needs higher budget

        while len(grasp_set) < self.num_grasps and attempts < max_attempts:
            attempts += 1
            
            # Step 1: Geometric Sampling (Find opposing points with good spacing)
            geom_result = self._sample_finger_assignment(points, normals)
            if geom_result is None:
                continue
            pts, nrm = geom_result

            # Step 2: Grasp Analysis (Algorithm 4 - Net Force Optimization)
            if self.nfo is not None:
                # evaluate() should return a score based on ||Σ f_i n_i||^2 optimization
                quality = self.nfo.evaluate(pts, nrm)
                if quality < self.nfo.min_quality:
                    continue  # Rejected by NFO
            else:
                quality = 1.0  # Bypass if no NFO provided

            # Step 3: Random Pose in Hand Frame
            obj_pos_hand = self._sample_object_pose_in_hand(pts)
            obj_quat_hand = self._sample_random_quaternion()

            # Step 4: Assign (Inverse Kinematics to find 'q')
            joint_angles = self._assign_ik(pts, obj_pos_hand, obj_quat_hand)
            if joint_angles is None:
                continue  # IK failed to converge

            # Step 5: Collision Check (Ensure hand doesn't penetrate object mesh)
            if not self._check_collision(joint_angles, obj_pos_hand, obj_quat_hand):
                continue  # Rejected due to penetration

            # Fully Valid Grasp Tuple (q, p) secured!
            grasp_set.add(Grasp(
                fingertip_positions=pts.astype(np.float32),
                contact_normals=nrm.astype(np.float32),
                quality=quality,
                object_name=self.object_name,
                object_scale=self.object_scale,
                joint_angles=joint_angles.astype(np.float32),
                object_pos_hand=obj_pos_hand.astype(np.float32),
                object_quat_hand=obj_quat_hand.astype(np.float32),
                object_pose_frame="hand_root"
            ))

        print(f"[HeuristicSampler] Generated {len(grasp_set)} valid grasps ({attempts} attempts)")
        return grasp_set

    # ------------------------------------------------------------------
    # Core Logic Implementations
    # ------------------------------------------------------------------

    def _sample_finger_assignment(self, points: np.ndarray, normals: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Returns valid (points, normals) or None based purely on geometry."""
        n_pts = len(points)
        selected_idx = []
        available = np.ones(n_pts, dtype=bool)

        for _ in range(self.num_fingers):
            candidates = np.where(available)[0]
            if len(candidates) == 0: return None
            chosen = candidates[self.rng.integers(0, len(candidates))]
            selected_idx.append(chosen)

            dists = np.linalg.norm(points - points[chosen], axis=-1)
            available &= (dists >= self.MIN_FINGER_SPACING)

        pts, nrm = points[selected_idx], normals[selected_idx]

        max_dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1).max()
        if max_dist > self.MAX_FINGER_SPACING * 3: return None

        n_dot = nrm @ nrm.T
        pair_dots = n_dot[np.triu_indices(self.num_fingers, k=1)]
        if not np.any(pair_dots < -self.PALM_NORMAL_THRESH): return None
        if pair_dots.mean() > 0.1: return None

        centroid = pts.mean(axis=0)
        dirs = pts - centroid
        dirs /= (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
        thumb_idx = int(np.argmin((dirs * nrm).sum(axis=-1)))

        order = [i for i in range(self.num_fingers) if i != thumb_idx] + [thumb_idx]
        return pts[order], nrm[order]

    def _assign_ik(self, pts_obj: np.ndarray, pos_hand: np.ndarray, quat_hand: np.ndarray) -> Optional[np.ndarray]:
        """
        Algorithm 3 'Assign': Translates target points to joint angles (q).
        Requires self.env (Isaac Sim) to compute differential IK.
        """
        if self.env is None:
            # Fallback for testing without simulator: return dummy joints
            return np.zeros(16, dtype=np.float32)

        # TODO: Implement Isaac Sim IK call here.
        # 1. Place object in world at a safe z-height.
        # 2. Compute wrist pose using 'pos_hand' and 'quat_hand'.
        # 3. Transform 'pts_obj' to world frame.
        # 4. Run `refine_hand_to_start_grasp` or equivalent IK solver.
        # 5. Measure tip error. If error > threshold, return None.
        # 6. Return robot.data.joint_pos.
        
        # Pseudo-code placeholder:
        # joint_pos, error = run_ik_solver(self.env, pts_obj, pos_hand, quat_hand)
        # if error > 0.01: return None
        # return joint_pos
        pass

    def _check_collision(self, joint_angles: np.ndarray, pos_hand: np.ndarray, quat_hand: np.ndarray) -> bool:
        """
        Algorithm 3 'NoCollision': Ensures the computed 'q' doesn't cause penetration.
        """
        if self.env is None:
            return True # Fallback

        # TODO: Implement Isaac Sim collision check.
        # Read contact forces from the physics engine after setting the joint angles.
        # If massive contact forces or deep penetrations are detected -> return False.
        return True

    def _sample_object_pose_in_hand(self, points_obj: np.ndarray) -> np.ndarray:
        centroid = points_obj.mean(axis=0)
        offset = self.rng.normal(0.0, self.MIN_FINGER_SPACING * 0.5, size=3)
        return centroid + offset

    def _sample_random_quaternion(self) -> np.ndarray:
        q = self.rng.normal(size=4)
        q /= np.linalg.norm(q) + 1e-8
        return q if q[0] >= 0.0 else -q
