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
import torch
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


# ---------------------------------------------------------------------------
# Algorithm 5: GraspRRTExpander — joint-space RRT expansion
# ---------------------------------------------------------------------------

class GraspRRTExpander:
    """
    DexGen Algorithm 5: RRT expansion in joint space.

    Perturbs parent grasp joints, validates via FK + surface proximity +
    NFO + penetration check. Builds a connectivity graph.
    """

    def __init__(
        self,
        env,
        mesh: trimesh.Trimesh,
        nfo,
        ft_ids: list,
        rrt_steps: int = 300,
        joint_noise_std: float = 0.1,
        contact_threshold: float = 0.015,
        penetration_margin: float = 0.008,
        delta_max: float = 0.5,
        max_attempts_per_step: int = 50,
        seed: int = 42,
    ):
        self.env = env
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.env_ids = torch.tensor([0], device=self.device, dtype=torch.long)

        self.mesh = mesh
        self.nfo = nfo
        self.ft_ids = ft_ids
        self.rrt_steps = rrt_steps
        self.joint_noise_std = joint_noise_std
        self.contact_threshold = contact_threshold
        self.penetration_margin = penetration_margin
        self.delta_max = delta_max
        self.max_attempts_per_step = max_attempts_per_step
        self.rng = np.random.default_rng(seed)

        self.finger_body_ids = self._resolve_finger_body_ids()
        self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()

    def _resolve_finger_body_ids(self) -> list:
        _LINKS = [
            "robot0_ffknuckle", "robot0_ffproximal", "robot0_ffmiddle", "robot0_ffdistal",
            "robot0_mfknuckle", "robot0_mfproximal", "robot0_mfmiddle", "robot0_mfdistal",
            "robot0_rfknuckle", "robot0_rfproximal", "robot0_rfmiddle", "robot0_rfdistal",
            "robot0_lfmetacarpal", "robot0_lfknuckle", "robot0_lfproximal",
            "robot0_lfmiddle", "robot0_lfdistal",
            "robot0_thbase", "robot0_thproximal", "robot0_thhub",
            "robot0_thmiddle", "robot0_thdistal",
        ]
        ids = []
        for name in _LINKS:
            try:
                found = self.robot.find_bodies(name)[0]
                if len(found) > 0:
                    ids.append(int(found[0]))
            except Exception:
                pass
        if not ids:
            num_bodies = self.robot.data.body_pos_w.shape[1]
            ids = list(range(2, num_bodies))
        return ids

    def expand(self, seed_set: GraspSet) -> "GraspGraph":
        """Expand seeds via joint-space RRT."""
        from .rrt_expansion import GraspGraph

        grasps = list(seed_set.grasps)
        obj_pos_w = self._get_object_pos(grasps[0])
        total_attempts = 0
        max_total = self.rrt_steps * self.max_attempts_per_step * 2

        while len(grasps) < len(seed_set) + self.rrt_steps:
            new = self._expand_step(grasps, obj_pos_w)
            if new is not None:
                grasps.append(new)
                if len(grasps) % 50 == 0:
                    print(f"    [RRT] {len(grasps)} grasps")
            total_attempts += 1
            if total_attempts >= max_total:
                print(f"    [RRT] Stopping at {len(grasps)} grasps ({total_attempts} attempts)")
                break

        edges = self._build_edges(grasps)
        return GraspGraph(
            grasp_set=GraspSet(grasps=grasps, object_name=seed_set.object_name),
            edges=edges,
            object_name=seed_set.object_name,
            num_fingers=len(self.ft_ids),
        )

    def _get_object_pos(self, grasp: Grasp) -> torch.Tensor:
        """Reconstruct world-frame object position from grasp."""
        if grasp.joint_angles is not None:
            q = torch.tensor(grasp.joint_angles, device=self.device, dtype=torch.float32)
            self.robot.write_joint_state_to_sim(
                q.unsqueeze(0), torch.zeros(1, len(q), device=self.device),
                env_ids=self.env_ids,
            )
            self.robot.update(0.0)
        ft = self.robot.data.body_pos_w[0, self.ft_ids, :]
        return ft.mean(dim=0)

    def _expand_step(self, grasps: list, obj_pos_w: torch.Tensor) -> Optional[Grasp]:
        for _ in range(self.max_attempts_per_step):
            parent = grasps[self.rng.integers(0, len(grasps))]
            q_parent = torch.tensor(
                parent.joint_angles, device=self.device, dtype=torch.float32,
            )
            noise = torch.randn_like(q_parent) * self.joint_noise_std
            noise[:2] = 0.0
            q_new = torch.clamp(q_parent + noise, self.q_low, self.q_high)
            grasp = self._evaluate(q_new, obj_pos_w)
            if grasp is not None:
                return grasp
        return None

    def _evaluate(self, q: torch.Tensor, obj_pos_w: torch.Tensor) -> Optional[Grasp]:
        """FK + penetration + surface proximity + NFO."""
        q_2d = q.unsqueeze(0)
        self.robot.write_joint_state_to_sim(
            q_2d, torch.zeros_like(q_2d), env_ids=self.env_ids,
        )
        self.robot.set_joint_position_target(q_2d, env_ids=self.env_ids)
        self.robot.update(0.0)

        # Penetration check
        fl_world = self.robot.data.body_pos_w[0, self.finger_body_ids, :]
        fl_obj = (fl_world - obj_pos_w).cpu().numpy()
        fl_closest, fl_dists, fl_face = trimesh.proximity.closest_point(self.mesh, fl_obj)
        to_pt = fl_obj - fl_closest
        fl_normals = self.mesh.face_normals[fl_face]
        sign = np.sum(to_pt * fl_normals, axis=-1)
        penetration = np.where(sign < 0, fl_dists, 0.0)
        if np.any(penetration > self.penetration_margin):
            return None

        # Surface proximity
        ft_world = self.robot.data.body_pos_w[0, self.ft_ids, :].clone()
        ft_obj = (ft_world - obj_pos_w).cpu().numpy()
        closest, dists, face_idx = trimesh.proximity.closest_point(self.mesh, ft_obj)
        if np.any(dists > self.contact_threshold):
            return None

        # NFO
        normals = self.mesh.face_normals[face_idx].astype(np.float32)
        test_grasp = Grasp(
            fingertip_positions=closest.astype(np.float32),
            contact_normals=normals,
        )
        quality = self.nfo.evaluate(test_grasp)
        if quality < self.nfo.min_quality:
            return None

        # Per-grasp object pose
        from isaaclab.utils.math import quat_apply_inverse
        from envs.mdp.math_utils import quat_multiply, quat_conjugate

        grasp_centroid = ft_obj.mean(axis=0)
        fp_local = (closest - grasp_centroid).astype(np.float32)
        grasp_obj_pos_w = ft_world.mean(dim=0)

        rp = self.robot.data.root_pos_w[0]
        rq = self.robot.data.root_quat_w[0]
        rel = grasp_obj_pos_w - rp
        pos_hand = quat_apply_inverse(rq.unsqueeze(0), rel.unsqueeze(0))[0]
        obj_quat_w = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32)
        quat_hand = quat_multiply(quat_conjugate(rq.unsqueeze(0)), obj_quat_w)[0]

        return Grasp(
            fingertip_positions=fp_local,
            contact_normals=normals,
            quality=quality,
            joint_angles=q.cpu().numpy().copy(),
            object_pos_hand=pos_hand.cpu().numpy().copy(),
            object_quat_hand=quat_hand.cpu().numpy().copy(),
            object_pose_frame="hand_root",
        )

    def _build_edges(self, grasps: list) -> list:
        n = len(grasps)
        if n < 2:
            return []
        all_q = np.stack([g.joint_angles for g in grasps])
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.linalg.norm(all_q[i] - all_q[j])) < self.delta_max:
                    edges.append((i, j))
        return edges
