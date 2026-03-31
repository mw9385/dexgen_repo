"""
Hand Model for Grasp Optimization
===================================
Wraps the DexGraspNet HandModel (MJCF-based FK via pytorch_kinematics)
for use in our grasp generation pipeline.

Two modes:
  1. Full mode: uses DexGraspNet's HandModel with MJCF, mesh collision,
     contact candidates, and penetration keypoints (requires pytorch_kinematics).
  2. Lite mode: simplified DH-parameter FK when DexGraspNet assets are unavailable.

The full mode is recommended for production grasp generation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup for DexGraspNet imports
# ---------------------------------------------------------------------------

_DEXGRASPNET_ROOT = Path(__file__).parent.parent / "third_party" / "DexGraspNet"
_DEXGRASPNET_GRASP = _DEXGRASPNET_ROOT / "grasp_generation"
_DEXGRASPNET_THIRDPARTY = _DEXGRASPNET_ROOT / "thirdparty"
_PK_PATH = _DEXGRASPNET_THIRDPARTY / "pytorch_kinematics"

# Asset paths
MJCF_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "shadow_hand_wrist_free.xml")
MESH_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "meshes")
CONTACT_POINTS_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "contact_points.json")
PENETRATION_POINTS_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "penetration_points.json")


def _ensure_dexgraspnet_imports():
    """Add DexGraspNet paths to sys.path if needed."""
    for p in [str(_DEXGRASPNET_GRASP), str(_PK_PATH)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _has_dexgraspnet_assets() -> bool:
    """Check if DexGraspNet assets are available."""
    return (
        os.path.isfile(MJCF_PATH)
        and os.path.isdir(MESH_PATH)
        and os.path.isfile(CONTACT_POINTS_PATH)
        and os.path.isfile(PENETRATION_POINTS_PATH)
    )


# ---------------------------------------------------------------------------
# Rotation utilities (from DexGraspNet rot6d.py)
# ---------------------------------------------------------------------------

def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    return v / v_mag


def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)


def robust_compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix (robust version)."""
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw)
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    return torch.cat((x, y, z), 2)


def rotation_matrix_to_ortho6d(R: torch.Tensor) -> torch.Tensor:
    """Extract 6D rotation from (B, 3, 3) rotation matrix."""
    return R.transpose(1, 2)[:, :2].reshape(-1, 6)


def random_rotation_6d(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample random rotations in 6D representation."""
    M = torch.randn(batch_size, 3, 3, device=device)
    Q, _ = torch.linalg.qr(M)
    det = torch.det(Q)
    Q[:, :, 0] *= det.unsqueeze(-1).sign()
    return rotation_matrix_to_ortho6d(Q)


# ---------------------------------------------------------------------------
# Full DexGraspNet Hand Model (MJCF + pytorch_kinematics)
# ---------------------------------------------------------------------------

class DexGraspNetHandModel:
    """
    Wrapper around DexGraspNet's HandModel.

    Provides the same interface used by the optimizer:
      - set_parameters(hand_pose, contact_point_indices)
      - contact_points: (B, n_contact, 3)
      - cal_distance(x): signed distance from points to hand
      - self_penetration(): self-collision energy
    """

    def __init__(self, device: str = "cuda", n_surface_points: int = 0):
        _ensure_dexgraspnet_imports()

        # Import DexGraspNet's HandModel
        from utils.hand_model import HandModel as _DGNHandModel

        self._model = _DGNHandModel(
            mjcf_path=MJCF_PATH,
            mesh_path=MESH_PATH,
            contact_points_path=CONTACT_POINTS_PATH,
            penetration_points_path=PENETRATION_POINTS_PATH,
            n_surface_points=n_surface_points,
            device=device,
        )
        self.device = device
        self.n_dofs = self._model.n_dofs
        self.n_contact_candidates = self._model.n_contact_candidates
        self.joints_lower = self._model.joints_lower
        self.joints_upper = self._model.joints_upper

    @property
    def hand_pose(self):
        return self._model.hand_pose

    @hand_pose.setter
    def hand_pose(self, value):
        self._model.hand_pose = value

    @property
    def contact_point_indices(self):
        return self._model.contact_point_indices

    @contact_point_indices.setter
    def contact_point_indices(self, value):
        self._model.contact_point_indices = value

    @property
    def contact_points(self):
        return self._model.contact_points

    @contact_points.setter
    def contact_points(self, value):
        self._model.contact_points = value

    @property
    def global_translation(self):
        return self._model.global_translation

    @global_translation.setter
    def global_translation(self, value):
        self._model.global_translation = value

    @property
    def global_rotation(self):
        return self._model.global_rotation

    @global_rotation.setter
    def global_rotation(self, value):
        self._model.global_rotation = value

    @property
    def current_status(self):
        return self._model.current_status

    @current_status.setter
    def current_status(self, value):
        self._model.current_status = value

    @property
    def chain(self):
        return self._model.chain

    def set_parameters(self, hand_pose, contact_point_indices=None):
        self._model.set_parameters(hand_pose, contact_point_indices)

    def cal_distance(self, x):
        return self._model.cal_distance(x)

    def self_penetration(self):
        return self._model.self_penetration()


# ---------------------------------------------------------------------------
# Primitive Object Model (replaces DexGraspNet's ObjectModel for our shapes)
# ---------------------------------------------------------------------------

class PrimitiveObjectModel:
    """
    Object model for primitive shapes (cube, sphere, cylinder).

    Provides the same interface as DexGraspNet's ObjectModel:
      - cal_distance(x): signed distance from contact points to object
      - surface_points_tensor: sampled surface points for penetration checking
    """

    def __init__(self, mesh, shape_type: str, size: float,
                 num_samples: int = 2000, device: str = "cuda"):
        import trimesh
        self.device = device
        self.shape_type = shape_type
        self.size = size
        self.mesh = mesh

        # Sample surface points for penetration energy
        points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        self.surface_points_tensor = torch.tensor(
            points, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, num_samples, 3)

        # Object scale (always 1.0 for our pipeline — scaling is baked into mesh)
        self.object_scale_tensor = torch.ones(1, 1, dtype=torch.float32, device=device)

    def cal_distance(self, x):
        """
        Compute signed distance from points to object surface.

        Convention matches DexGraspNet: positive = inside, negative = outside.

        Args:
            x: (B, N, 3) points

        Returns:
            distance: (B, N) signed distances
            normals: (B, N, 3) contact normals
        """
        B, N, _ = x.shape

        if self.shape_type == "sphere":
            return self._sdf_sphere(x)
        elif self.shape_type == "cube":
            return self._sdf_box(x)
        elif self.shape_type == "cylinder":
            return self._sdf_cylinder(x)
        else:
            return self._sdf_sphere(x)  # fallback

    def _sdf_sphere(self, x):
        """SDF for sphere centered at origin. Positive = inside."""
        r = self.size / 2.0
        dist_from_center = torch.norm(x, dim=-1)
        sdf = r - dist_from_center  # positive inside
        normals = x / (dist_from_center.unsqueeze(-1) + 1e-8)
        return sdf, normals

    def _sdf_box(self, x):
        """SDF for axis-aligned box centered at origin. Positive = inside."""
        h = self.size / 2.0
        q = torch.abs(x) - h
        outside_dist = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        inside_dist = torch.clamp(q.max(dim=-1).values, max=0.0)
        sdf = -(outside_dist + inside_dist)  # flip sign: positive = inside
        # Normal = gradient of SDF
        normals = self._finite_diff_normals(x, self._box_sdf_scalar)
        return sdf, normals

    def _sdf_cylinder(self, x):
        """SDF for Z-aligned cylinder centered at origin. Positive = inside."""
        r = self.size / 2.0
        half_h = self.size / 2.0
        xy = x[..., :2]
        z = x[..., 2]
        d_radial = torch.norm(xy, dim=-1) - r
        d_height = torch.abs(z) - half_h
        outside_dist = torch.norm(
            torch.stack([torch.clamp(d_radial, min=0.0),
                         torch.clamp(d_height, min=0.0)], dim=-1), dim=-1)
        inside_dist = torch.clamp(torch.max(d_radial, d_height), max=0.0)
        sdf = -(outside_dist + inside_dist)  # positive = inside
        normals = self._finite_diff_normals(x, self._cylinder_sdf_scalar)
        return sdf, normals

    def _box_sdf_scalar(self, x):
        h = self.size / 2.0
        q = torch.abs(x) - h
        outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        return -(outside + inside)

    def _cylinder_sdf_scalar(self, x):
        r = self.size / 2.0
        half_h = self.size / 2.0
        xy = x[..., :2]
        z = x[..., 2]
        d_r = torch.norm(xy, dim=-1) - r
        d_h = torch.abs(z) - half_h
        outside = torch.norm(torch.stack([torch.clamp(d_r, min=0.0),
                                           torch.clamp(d_h, min=0.0)], dim=-1), dim=-1)
        inside = torch.clamp(torch.max(d_r, d_h), max=0.0)
        return -(outside + inside)

    def _finite_diff_normals(self, x, sdf_fn, eps=0.001):
        dx = torch.zeros_like(x); dx[..., 0] = eps
        dy = torch.zeros_like(x); dy[..., 1] = eps
        dz = torch.zeros_like(x); dz[..., 2] = eps
        gx = sdf_fn(x + dx) - sdf_fn(x - dx)
        gy = sdf_fn(x + dy) - sdf_fn(x - dy)
        gz = sdf_fn(x + dz) - sdf_fn(x - dz)
        grad = torch.stack([gx, gy, gz], dim=-1)
        return grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_hand_model(hand_name: str = "shadow", device: str = "cuda",
                     **kwargs) -> DexGraspNetHandModel:
    """Build DexGraspNet hand model (MJCF-based FK)."""
    if not _has_dexgraspnet_assets():
        raise RuntimeError(
            f"DexGraspNet assets not found at {_DEXGRASPNET_GRASP}/mjcf/. "
            "Run: git submodule update --init third_party/DexGraspNet"
        )
    return DexGraspNetHandModel(device=device, **kwargs)


def build_object_model(mesh, shape_type: str, size: float,
                       device: str = "cuda", **kwargs) -> PrimitiveObjectModel:
    """Build object model for primitive shapes."""
    return PrimitiveObjectModel(mesh, shape_type, size, device=device, **kwargs)
