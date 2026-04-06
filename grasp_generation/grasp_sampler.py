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

    def as_array(self) -> np.ndarray:
        """Stack all fingertip positions into (N, F*3) array."""
        return np.stack([g.fingertip_positions.flatten() for g in self.grasps])

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
# Shared helpers for FK-based grasp validation
# ---------------------------------------------------------------------------

_SHADOW_FINGER_LINKS = [
    "robot0_ffknuckle", "robot0_ffproximal", "robot0_ffmiddle", "robot0_ffdistal",
    "robot0_mfknuckle", "robot0_mfproximal", "robot0_mfmiddle", "robot0_mfdistal",
    "robot0_rfknuckle", "robot0_rfproximal", "robot0_rfmiddle", "robot0_rfdistal",
    "robot0_lfmetacarpal", "robot0_lfknuckle", "robot0_lfproximal",
    "robot0_lfmiddle", "robot0_lfdistal",
    "robot0_thbase", "robot0_thproximal", "robot0_thhub",
    "robot0_thmiddle", "robot0_thdistal",
]


def _resolve_finger_body_ids(robot) -> list:
    """Get all finger link body IDs for penetration check."""
    ids = []
    for name in _SHADOW_FINGER_LINKS:
        try:
            found = robot.find_bodies(name)[0]
            if len(found) > 0:
                ids.append(int(found[0]))
        except Exception:
            pass
    if not ids:
        num_bodies = robot.data.body_pos_w.shape[1]
        ids = list(range(2, num_bodies))
    return ids


def _compute_obj_pose_hand(robot, obj_pos_w, device):
    """Compute object pose in hand root frame. Returns (pos_np, quat_np)."""
    from isaaclab.utils.math import quat_apply_inverse
    from envs.mdp.math_utils import quat_multiply, quat_conjugate

    rp = robot.data.root_pos_w[0]
    rq = robot.data.root_quat_w[0]
    rel = obj_pos_w - rp
    pos_hand = quat_apply_inverse(rq.unsqueeze(0), rel.unsqueeze(0))[0]
    obj_quat_w = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32)
    quat_hand = quat_multiply(quat_conjugate(rq.unsqueeze(0)), obj_quat_w)[0]
    return pos_hand.cpu().numpy().copy(), quat_hand.cpu().numpy().copy()


# ---------------------------------------------------------------------------
# Heuristic Sampler — FK-based seed generation
# ---------------------------------------------------------------------------

class HeuristicSampler:
    """
    FK-based seed generation for Shadow Hand.

    Samples joint configurations centered on joint-limit midpoint with
    moderate noise. Validates via FK + partial contact check + NFO.

    Key design choices for Shadow Hand 5-finger:
      - Pre-shape centered sampling (midpoint ± noise_std) instead of
        full-range random. This keeps fingers in a natural grasp envelope.
      - Partial contact: requires min_contact_fingers (default 3) out of
        num_fingers to be within contact_threshold. Remaining fingers can
        be further away and get refined during RRT expansion.
      - Relaxed contact_threshold (default 3cm) for seeds — expansion
        will tighten contact as grasps evolve.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        object_name: str = "object",
        object_scale: float = 1.0,
        num_candidates: int = 10000,
        num_grasps: int = 50,
        num_fingers: int = 5,
        nfo: Any = None,
        env: Any = None,
        ft_ids: list = None,
        noise_std: float = 0.3,
        contact_threshold: float = 0.03,
        min_contact_fingers: int = 3,
        penetration_margin: float = 0.008,
        seed: int = 42,
    ):
        self.mesh = mesh
        self.object_name = object_name
        self.object_scale = object_scale
        self.num_candidates = num_candidates
        self.num_grasps = num_grasps
        self.num_fingers = num_fingers
        self.nfo = nfo
        self.env = env
        self.ft_ids = ft_ids
        self.noise_std = noise_std
        self.contact_threshold = contact_threshold
        self.min_contact_fingers = min_contact_fingers
        self.penetration_margin = penetration_margin
        self.rng = np.random.default_rng(seed)

        if env is not None:
            self.robot = env.scene["robot"]
            self.obj = env.scene["object"]
            self.device = env.device
            self.env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
            self.num_dof = self.robot.data.joint_pos.shape[-1]
            self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
            self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()
            self.q_mid = (self.q_low + self.q_high) / 2.0
            self.q_mid[:2] = 0.0  # wrist fixed
            self.q_range = self.q_high - self.q_low
            self.finger_body_ids = _resolve_finger_body_ids(self.robot)

    def sample(self) -> GraspSet:
        """Generate seeds via pre-shape centered joint sampling + FK validation."""
        print(f"[HeuristicSampler] Sampling {self.num_grasps} seeds on '{self.object_name}' "
              f"(noise_std={self.noise_std}, contact_thresh={self.contact_threshold}m, "
              f"min_contact={self.min_contact_fingers}/{self.num_fingers})")

        if self.env is None:
            print("[HeuristicSampler] WARNING: no env, returning empty set")
            return GraspSet(object_name=self.object_name)

        obj_pos_w = self._setup_object()
        grasp_set = GraspSet(object_name=self.object_name)

        for attempt in range(self.num_candidates):
            if len(grasp_set) >= self.num_grasps:
                break

            q = self._sample_preshape()
            grasp = self._evaluate(q, obj_pos_w)
            if grasp is not None:
                grasp_set.add(grasp)
                if len(grasp_set) % 10 == 0:
                    print(f"    seeds: {len(grasp_set)} (attempt {attempt + 1})")

        print(f"[HeuristicSampler] {len(grasp_set)} seeds from {min(attempt+1, self.num_candidates)} attempts")
        return grasp_set

    def _setup_object(self) -> torch.Tensor:
        """Place object at fingertip centroid of midpoint pose."""
        self._set_joints(self.q_mid)
        ft_world = self.robot.data.body_pos_w[self.env_ids][:, self.ft_ids, :]
        obj_pos = ft_world.mean(dim=1)  # (1, 3)

        obj_state = self.obj.data.default_root_state[self.env_ids].clone()
        obj_state[:, :3] = obj_pos
        obj_state[:, 3:7] = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32)
        obj_state[:, 7:] = 0.0
        self.obj.write_root_state_to_sim(obj_state, env_ids=self.env_ids)
        self.obj.update(0.0)
        print(f"    Object at {obj_pos[0].tolist()}")
        return obj_pos[0]  # (3,)

    def _set_joints(self, q: torch.Tensor):
        q_2d = q.unsqueeze(0) if q.ndim == 1 else q
        self.robot.write_joint_state_to_sim(
            q_2d, torch.zeros_like(q_2d), env_ids=self.env_ids,
        )
        self.robot.set_joint_position_target(q_2d, env_ids=self.env_ids)
        self.robot.update(0.0)

    def _sample_preshape(self) -> torch.Tensor:
        """Sample joints centered on midpoint with moderate noise."""
        noise = torch.randn(self.num_dof, device=self.device) * self.noise_std
        noise[:2] = 0.0  # wrist fixed
        q = self.q_mid + noise * self.q_range
        return torch.clamp(q, self.q_low, self.q_high)

    def _evaluate(self, q: torch.Tensor, obj_pos_w: torch.Tensor) -> Optional[Grasp]:
        """FK + penetration check + partial contact + NFO."""
        self._set_joints(q)

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

        # Partial contact check: at least min_contact_fingers within threshold
        ft_world = self.robot.data.body_pos_w[0, self.ft_ids, :].clone()
        ft_obj = (ft_world - obj_pos_w).cpu().numpy()
        closest, dists, face_idx = trimesh.proximity.closest_point(self.mesh, ft_obj)
        in_contact = dists <= self.contact_threshold
        if int(in_contact.sum()) < self.min_contact_fingers:
            return None

        # NFO on contacting fingers only
        normals = self.mesh.face_normals[face_idx].astype(np.float32)
        contact_pts = closest[in_contact].astype(np.float32)
        contact_nrm = normals[in_contact]
        if self.nfo is not None and len(contact_pts) >= 2:
            quality = self.nfo.evaluate(Grasp(
                fingertip_positions=contact_pts,
                contact_normals=contact_nrm,
            ))
            if quality < self.nfo.min_quality:
                return None
        else:
            quality = 0.0

        # Build grasp with per-grasp object pose
        grasp_obj_pos_w = ft_world.mean(dim=0)
        pos_hand, quat_hand = _compute_obj_pose_hand(self.robot, grasp_obj_pos_w, self.device)

        grasp_centroid = ft_obj.mean(axis=0)
        fp_local = (closest - grasp_centroid).astype(np.float32)

        return Grasp(
            fingertip_positions=fp_local,
            contact_normals=normals,
            quality=quality,
            object_name=self.object_name,
            object_scale=self.object_scale,
            joint_angles=q.cpu().numpy().copy(),
            object_pos_hand=pos_hand,
            object_quat_hand=quat_hand,
            object_pose_frame="hand_root",
        )


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
        contact_threshold: float = 0.02,
        min_contact_fingers: int = 4,
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
        self.min_contact_fingers = min_contact_fingers
        self.penetration_margin = penetration_margin
        self.delta_max = delta_max
        self.max_attempts_per_step = max_attempts_per_step
        self.rng = np.random.default_rng(seed)

        self.finger_body_ids = _resolve_finger_body_ids(self.robot)
        self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()

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

        # Partial contact check
        ft_world = self.robot.data.body_pos_w[0, self.ft_ids, :].clone()
        ft_obj = (ft_world - obj_pos_w).cpu().numpy()
        closest, dists, face_idx = trimesh.proximity.closest_point(self.mesh, ft_obj)
        in_contact = dists <= self.contact_threshold
        if int(in_contact.sum()) < self.min_contact_fingers:
            return None

        # NFO on contacting fingers
        normals = self.mesh.face_normals[face_idx].astype(np.float32)
        contact_pts = closest[in_contact].astype(np.float32)
        contact_nrm = normals[in_contact]
        if self.nfo is not None and len(contact_pts) >= 2:
            quality = self.nfo.evaluate(Grasp(
                fingertip_positions=contact_pts,
                contact_normals=contact_nrm,
            ))
            if quality < self.nfo.min_quality:
                return None
        else:
            quality = 0.0

        # Per-grasp object pose
        grasp_obj_pos_w = ft_world.mean(dim=0)
        pos_hand, quat_hand = _compute_obj_pose_hand(self.robot, grasp_obj_pos_w, self.device)

        grasp_centroid = ft_obj.mean(axis=0)
        fp_local = (closest - grasp_centroid).astype(np.float32)

        return Grasp(
            fingertip_positions=fp_local,
            contact_normals=normals,
            quality=quality,
            joint_angles=q.cpu().numpy().copy(),
            object_pos_hand=pos_hand,
            object_quat_hand=quat_hand,
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
