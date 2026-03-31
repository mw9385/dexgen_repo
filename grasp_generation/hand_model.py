"""
Hand Model for Grasp Optimization (torchsdf/pytorch3d-free)
=============================================================
Reimplements DexGraspNet's HandModel using only pytorch_kinematics + trimesh.
No torchsdf or pytorch3d dependency — collision shapes use analytical SDFs
(capsule + box) which covers all Shadow Hand links.

Architecture:
  - FK: pytorch_kinematics (MJCF parsing, batched FK)
  - Contact candidates: loaded from DexGraspNet's contact_points.json
  - Penetration keypoints: loaded from DexGraspNet's penetration_points.json
  - Hand SDF (cal_distance): analytical capsule SDF for fingers,
    analytical box SDF for palm/metacarpal links
  - Self-penetration: pairwise keypoint distance check
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh as tm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_DEXGRASPNET_ROOT = Path(__file__).parent.parent / "third_party" / "DexGraspNet"
_DEXGRASPNET_GRASP = _DEXGRASPNET_ROOT / "grasp_generation"
_PK_PATH = _DEXGRASPNET_ROOT / "thirdparty" / "pytorch_kinematics"

MJCF_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "shadow_hand_wrist_free.xml")
MESH_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "meshes")
CONTACT_POINTS_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "contact_points.json")
PENETRATION_POINTS_PATH = str(_DEXGRASPNET_GRASP / "mjcf" / "penetration_points.json")


def _ensure_pk_import():
    """Add pytorch_kinematics to sys.path."""
    p = str(_PK_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)


def _has_assets() -> bool:
    return os.path.isfile(MJCF_PATH) and os.path.isfile(CONTACT_POINTS_PATH)


# ---------------------------------------------------------------------------
# Rotation utilities (from DexGraspNet rot6d.py, no external deps)
# ---------------------------------------------------------------------------

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    return v / v_mag


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)


def robust_compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw)
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))
    return torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)


def rotation_matrix_to_ortho6d(R):
    return R.transpose(1, 2)[:, :2].reshape(-1, 6)


def random_rotation_6d(batch_size, device):
    M = torch.randn(batch_size, 3, 3, device=device)
    Q, _ = torch.linalg.qr(M)
    det = torch.det(Q)
    Q[:, :, 0] *= det.unsqueeze(-1).sign()
    return rotation_matrix_to_ortho6d(Q)


# ---------------------------------------------------------------------------
# Skip links for collision (same as DexGraspNet)
# ---------------------------------------------------------------------------

SKIP_LINKS = frozenset([
    'robot0:forearm', 'robot0:wrist_child',
    'robot0:ffknuckle_child', 'robot0:mfknuckle_child',
    'robot0:rfknuckle_child', 'robot0:lfknuckle_child',
    'robot0:thbase_child', 'robot0:thhub_child',
])

# Links that use box geometry (not capsule) in DexGraspNet
BOX_LINKS = frozenset([
    'robot0:palm', 'robot0:palm_child', 'robot0:lfmetacarpal_child',
])


# ---------------------------------------------------------------------------
# Hand Model (torchsdf-free reimplementation)
# ---------------------------------------------------------------------------

class DexGraspNetHandModel:
    """
    Shadow Hand model for grasp optimization.

    Uses pytorch_kinematics for FK, analytical SDFs for collision.
    Drop-in replacement for DexGraspNet's HandModel without
    torchsdf/pytorch3d dependencies.
    """

    def __init__(self, device='cuda'):
        _ensure_pk_import()
        import pytorch_kinematics as pk

        self.device = device

        # Load kinematic chain from MJCF
        # MJCF parser resolves mesh paths relative to CWD, so we must
        # chdir to the grasp_generation directory where meshes live.
        original_cwd = os.getcwd()
        try:
            os.chdir(str(_DEXGRASPNET_GRASP))
            self.chain = pk.build_chain_from_mjcf(
                open(MJCF_PATH).read()
            ).to(dtype=torch.float, device=device)
        finally:
            os.chdir(original_cwd)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # Load contact + penetration points
        contact_points = json.load(open(CONTACT_POINTS_PATH, 'r'))
        penetration_points = json.load(open(PENETRATION_POINTS_PATH, 'r'))

        # Build mesh/collision info per link
        self.mesh = {}
        self._build_mesh_recurse(self.chain._root, contact_points, penetration_points)

        # Joint limits
        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []
        self._set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        # Indexing for contact candidates
        self.link_name_to_link_index = dict(
            zip([n for n in self.mesh], range(len(self.mesh)))
        )

        self.contact_candidates = [
            self.mesh[n]['contact_candidates'] for n in self.mesh
        ]
        self.global_index_to_link_index = sum(
            [[i] * len(cc) for i, cc in enumerate(self.contact_candidates)], []
        )
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=device
        )
        self.n_contact_candidates = self.contact_candidates.shape[0]

        # Indexing for penetration keypoints
        self.penetration_keypoints = [
            self.mesh[n]['penetration_keypoints'] for n in self.mesh
        ]
        self.global_index_to_link_index_penetration = sum(
            [[i] * len(pk_pts) for i, pk_pts in enumerate(self.penetration_keypoints)], []
        )
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(
            self.global_index_to_link_index_penetration, dtype=torch.long, device=device
        )
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # State (set by set_parameters)
        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

        # Precompute collision parameters (cached tensors on GPU)
        self._precompute_collision_params()
        self._precompute_penetration_params()

        print(f"[HandModel] Shadow Hand: {self.n_dofs} DOFs, "
              f"{self.n_contact_candidates} contact candidates, "
              f"{self.n_keypoints} penetration keypoints, "
              f"{self._n_col_links} collision links")

    def _build_mesh_recurse(self, body, contact_points, penetration_points):
        """Build collision geometry per link (no torchsdf needed)."""
        if len(body.link.visuals) > 0:
            link_name = body.link.name
            # Store geom parameters for analytical SDF
            vis = body.link.visuals[0]
            geom_info = {'geom_type': vis.geom_type}

            if vis.geom_type == "capsule":
                geom_info['radius'] = float(vis.geom_param[0])
                geom_info['height'] = float(vis.geom_param[1]) * 2
            elif vis.geom_type == "box":
                geom_info['half_extents'] = vis.geom_param.detach().cpu().numpy().astype(np.float32)
            elif vis.geom_type == "mesh":
                # For mesh links (palm etc.), load and compute bounding box for approx SDF
                mesh_file = os.path.join(MESH_PATH, vis.geom_param[0].split(":")[1] + ".obj")
                if os.path.isfile(mesh_file):
                    link_mesh = tm.load_mesh(mesh_file, process=False)
                    scale = vis.geom_param[1] if vis.geom_param[1] is not None else [1, 1, 1]
                    verts = link_mesh.vertices * np.array(scale, dtype=np.float32)
                    # Use bounding box as approximate collision shape
                    half_ext = (verts.max(axis=0) - verts.min(axis=0)) / 2.0
                    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
                    geom_info['geom_type'] = 'box'
                    geom_info['half_extents'] = half_ext.astype(np.float32)
                    geom_info['center'] = center.astype(np.float32)

            # Contact candidates and penetration keypoints
            cc = torch.tensor(
                contact_points.get(link_name, [[0, 0, 0]]),
                dtype=torch.float32, device=self.device,
            ).reshape(-1, 3)
            pk_pts = torch.tensor(
                penetration_points.get(link_name, [[0, 0, 0]]),
                dtype=torch.float32, device=self.device,
            ).reshape(-1, 3)

            self.mesh[link_name] = {
                'contact_candidates': cc,
                'penetration_keypoints': pk_pts,
                'geom_info': geom_info,
            }

        for child in body.children:
            self._build_mesh_recurse(child, contact_points, penetration_points)

    def _set_joint_range_recurse(self, body):
        if body.joint.joint_type != "fixed":
            self.joints_names.append(body.joint.name)
            self.joints_lower.append(body.joint.range[0])
            self.joints_upper.append(body.joint.range[1])
        for child in body.children:
            self._set_joint_range_recurse(child)

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set hand configuration.

        Parameters
        ----------
        hand_pose: (B, 3+6+n_dofs) tensor
            [translation(3) | rot6d(6) | joint_angles(n_dofs)]
        contact_point_indices: (B, n_contact) LongTensor
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9]
        )
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])

        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape

            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]

            # Vectorized: gather transforms for each contact point via index
            all_matrices = torch.stack([
                self.current_status[name].get_matrix()
                for name in self._all_link_names
            ], dim=1)  # (B, num_all_links, 4, 4)

            idx = link_indices.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, n_contact, 4, 4
            )  # (B, n_contact, 4, 4)
            transforms = torch.gather(all_matrices, 1, idx)  # (B, n_contact, 4, 4)

            self.contact_points = torch.cat([
                self.contact_points,
                torch.ones(batch_size, n_contact, 1,
                           dtype=torch.float, device=self.device),
            ], dim=2)
            self.contact_points = (
                transforms @ self.contact_points.unsqueeze(3)
            )[:, :, :3, 0]
            self.contact_points = (
                self.contact_points @ self.global_rotation.transpose(1, 2)
                + self.global_translation.unsqueeze(1)
            )

    def _precompute_collision_params(self):
        """
        Cache collision geometry as batched tensors on GPU.

        Instead of iterating 17 links in Python, we pack all parameters
        into tensors of shape (L, ...) and compute all SDFs in one
        vectorized operation. This maximizes GPU utilization.
        """
        self._col_link_names = []
        capsule_heights = []
        capsule_radii = []
        box_half_extents = []
        box_centers = []
        is_capsule = []

        for link_name in self.mesh:
            if link_name in SKIP_LINKS:
                continue
            geom = self.mesh[link_name]['geom_info']
            self._col_link_names.append(link_name)

            if geom['geom_type'] == 'capsule':
                is_capsule.append(True)
                capsule_heights.append(geom['height'])
                capsule_radii.append(geom['radius'])
                box_half_extents.append([0.01, 0.01, 0.01])  # dummy
                box_centers.append([0.0, 0.0, 0.0])
            elif geom['geom_type'] == 'box':
                is_capsule.append(False)
                capsule_heights.append(0.01)  # dummy
                capsule_radii.append(0.01)
                box_half_extents.append(geom['half_extents'].tolist())
                box_centers.append(
                    geom.get('center', np.zeros(3)).tolist()
                    if isinstance(geom.get('center'), np.ndarray)
                    else geom.get('center', [0, 0, 0])
                )

        L = len(self._col_link_names)
        self._n_col_links = L
        # (L,) bool mask: True=capsule, False=box
        self._is_capsule = torch.tensor(is_capsule, dtype=torch.bool, device=self.device)
        # Capsule params: (L,)
        self._cap_heights = torch.tensor(capsule_heights, dtype=torch.float, device=self.device)
        self._cap_radii = torch.tensor(capsule_radii, dtype=torch.float, device=self.device)
        # Box params: (L, 3)
        self._box_he = torch.tensor(box_half_extents, dtype=torch.float, device=self.device)
        self._box_center = torch.tensor(box_centers, dtype=torch.float, device=self.device)

    def cal_distance(self, x):
        """
        Signed distance from points to hand surface (vectorized, no Python loop).
        Positive = inside hand, negative = outside.

        All L collision links are processed in a single batched operation.
        Shape flow: x (B, N, 3) → x_all_local (B, L, N, 3) → dis (B, L, N) → max over L
        """
        B, N, _ = x.shape
        L = self._n_col_links
        device = self.device

        # Transform to hand frame: (B, N, 3)
        x_hand = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation

        # Gather all link transforms: (B, L, 4, 4)
        matrices = torch.stack([
            self.current_status[name].get_matrix()
            for name in self._col_link_names
        ], dim=1)  # (B, L, 4, 4)

        # Transform x to each link's local frame in one shot
        # x_hand: (B, 1, N, 3), translations: (B, L, 1, 3), rotations: (B, L, 3, 3)
        link_pos = matrices[:, :, :3, 3].unsqueeze(2)    # (B, L, 1, 3)
        link_rot = matrices[:, :, :3, :3]                 # (B, L, 3, 3)
        x_expanded = x_hand.unsqueeze(1).expand(B, L, N, 3)  # (B, L, N, 3)
        x_local = (x_expanded - link_pos) @ link_rot      # (B, L, N, 3)

        # Flatten for SDF computation: (B*L*N, 3)
        x_flat = x_local.reshape(B * L * N, 3)

        # Compute capsule SDF for ALL links (we'll mask later)
        # capsule: nearest point on z-axis segment [0, height]
        heights = self._cap_heights.view(1, L, 1).expand(B, L, N).reshape(B * L * N)
        radii = self._cap_radii.view(1, L, 1).expand(B, L, N).reshape(B * L * N)
        z_clamped = torch.clamp(x_flat[:, 2], min=0.0, max=heights)
        dx = x_flat[:, 0]
        dy = x_flat[:, 1]
        dz = x_flat[:, 2] - z_clamped
        dist_to_axis = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
        capsule_sdf = radii - dist_to_axis  # positive = inside

        # Compute box SDF for ALL links
        centers = self._box_center.view(1, L, 1, 3).expand(B, L, N, 3).reshape(B * L * N, 3)
        half_exts = self._box_he.view(1, L, 1, 3).expand(B, L, N, 3).reshape(B * L * N, 3)
        p = x_flat - centers
        q = torch.abs(p) - half_exts
        outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        box_sdf = -(outside + inside)  # positive = inside

        # Select capsule or box based on link type
        is_cap = self._is_capsule.view(1, L, 1).expand(B, L, N).reshape(B * L * N)
        dis_flat = torch.where(is_cap, capsule_sdf, box_sdf)

        # Reshape and take max over links: (B, L, N) → (B, N)
        dis = dis_flat.reshape(B, L, N)
        return dis.max(dim=1).values

    def _precompute_penetration_params(self):
        """Precompute the link index → all-links-list index mapping for penetration."""
        # Build a mapping: for each keypoint, which index in _all_link_names?
        all_link_names = list(self.mesh.keys())
        self._all_link_names = all_link_names
        self._pen_link_indices = self.global_index_to_link_index_penetration  # (K,)
        # Map link_name_to_link_index → index in all_link_names (for gathering)
        self._pen_link_gather_idx = self._pen_link_indices  # already correct

    def self_penetration(self):
        """Self-penetration energy via pairwise keypoint distances (vectorized)."""
        batch_size = self.global_translation.shape[0]
        K = self.n_keypoints

        # Gather all link transforms: (B, num_all_links, 4, 4)
        all_matrices = torch.stack([
            self.current_status[name].get_matrix()
            for name in self._all_link_names
        ], dim=1)  # (B, num_all_links, 4, 4)

        # Select transform for each keypoint: (B, K, 4, 4)
        idx = self._pen_link_gather_idx.view(1, K, 1, 1).expand(batch_size, K, 4, 4)
        transforms = torch.gather(all_matrices, 1, idx)  # (B, K, 4, 4)

        # Transform keypoints
        points = self.penetration_keypoints.unsqueeze(0).expand(batch_size, K, 3)
        points_h = torch.cat([
            points,
            torch.ones(batch_size, K, 1, dtype=torch.float, device=self.device),
        ], dim=2)  # (B, K, 4)
        points = (transforms @ points_h.unsqueeze(3))[:, :, :3, 0]  # (B, K, 3)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

        # Pairwise distance
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1, 2))


# ---------------------------------------------------------------------------
# Primitive Object Model (analytical SDF, no torchsdf)
# ---------------------------------------------------------------------------

class PrimitiveObjectModel:
    """Object model for cube/sphere/cylinder with analytical SDF."""

    def __init__(self, mesh, shape_type, size, num_samples=500, device='cuda'):
        self.device = device
        self.shape_type = shape_type
        self.size = size
        self.mesh = mesh

        points, _ = tm.sample.sample_surface(mesh, num_samples)
        self.surface_points_tensor = torch.tensor(
            points, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        self.object_scale_tensor = torch.ones(1, 1, dtype=torch.float32, device=device)

    def cal_distance(self, x):
        """
        Signed distance: positive = inside object, negative = outside.
        Returns (distance, normals).
        """
        if self.shape_type == "sphere":
            return self._sdf_sphere(x)
        elif self.shape_type == "cube":
            return self._sdf_box(x)
        elif self.shape_type == "cylinder":
            return self._sdf_cylinder(x)
        return self._sdf_sphere(x)

    def _sdf_sphere(self, x):
        r = self.size / 2.0
        d = torch.norm(x, dim=-1)
        sdf = r - d
        normals = x / (d.unsqueeze(-1) + 1e-8)
        return sdf, normals

    def _sdf_box(self, x):
        h = self.size / 2.0
        q = torch.abs(x) - h
        outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        sdf = -(outside + inside)
        normals = self._fd_normals(x, lambda p: self._box_scalar(p, h))
        return sdf, normals

    def _sdf_cylinder(self, x):
        r, hh = self.size / 2.0, self.size / 2.0
        xy = x[..., :2]
        z = x[..., 2]
        dr = torch.norm(xy, dim=-1) - r
        dh = torch.abs(z) - hh
        outside = torch.norm(torch.stack([
            torch.clamp(dr, min=0.0), torch.clamp(dh, min=0.0)
        ], dim=-1), dim=-1)
        inside = torch.clamp(torch.max(dr, dh), max=0.0)
        sdf = -(outside + inside)
        normals = self._fd_normals(x, lambda p: self._cyl_scalar(p, r, hh))
        return sdf, normals

    @staticmethod
    def _box_scalar(p, h):
        q = torch.abs(p) - h
        return -(torch.norm(torch.clamp(q, min=0.0), dim=-1) +
                 torch.clamp(q.max(dim=-1).values, max=0.0))

    @staticmethod
    def _cyl_scalar(p, r, hh):
        dr = torch.norm(p[..., :2], dim=-1) - r
        dh = torch.abs(p[..., 2]) - hh
        outside = torch.norm(torch.stack([
            torch.clamp(dr, min=0.0), torch.clamp(dh, min=0.0)
        ], dim=-1), dim=-1)
        inside = torch.clamp(torch.max(dr, dh), max=0.0)
        return -(outside + inside)

    def _fd_normals(self, x, sdf_fn, eps=0.001):
        dx = torch.zeros_like(x); dx[..., 0] = eps
        dy = torch.zeros_like(x); dy[..., 1] = eps
        dz = torch.zeros_like(x); dz[..., 2] = eps
        g = torch.stack([
            sdf_fn(x + dx) - sdf_fn(x - dx),
            sdf_fn(x + dy) - sdf_fn(x - dy),
            sdf_fn(x + dz) - sdf_fn(x - dz),
        ], dim=-1)
        return g / (torch.norm(g, dim=-1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_hand_model(hand_name='shadow', device='cuda', **kwargs):
    if not _has_assets():
        raise RuntimeError(
            f"DexGraspNet assets not found. Run: git submodule update --init third_party/DexGraspNet"
        )
    return DexGraspNetHandModel(device=device)


def build_object_model(mesh, shape_type, size, device='cuda', **kwargs):
    return PrimitiveObjectModel(mesh, shape_type, size, device=device, **kwargs)
