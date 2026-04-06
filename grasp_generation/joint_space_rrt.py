"""
Joint-Space RRT Grasp Generation for Shadow Hand.

All grasps are generated in joint space via FK, guaranteeing kinematic
feasibility. No IK step is needed — every grasp in the output graph
has valid joint_angles, fingertip_positions, and object_pose.

Algorithm:
  1. Fix object position relative to the hand (fingertip centroid of default pose)
  2. Sample random joint configurations within soft limits
  3. FK → fingertip world positions → transform to object frame
  4. Check surface proximity via trimesh closest_point
  5. Compute contact normals, evaluate NFO quality
  6. RRT expansion: perturb parent joints → same validation
  7. Build graph with edges based on joint-space distance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh

from .grasp_sampler import Grasp, GraspSet
from .rrt_expansion import GraspGraph
from .net_force_optimization import NetForceOptimizer


@dataclass
class JointSpaceRRTConfig:
    """All parameters for joint-space RRT generation."""
    target_size: int = 300
    num_seed_attempts: int = 5000
    max_expand_attempts_per_step: int = 50
    max_total_expand_attempts: int = 100000

    # Contact validation
    contact_threshold: float = 0.015   # max fingertip-to-surface distance (m)
    penetration_margin: float = 0.008  # max allowed penetration depth for finger links (m)

    # RRT expansion
    joint_noise_std: float = 0.1       # std dev for joint perturbation (rad)

    # Graph edges
    delta_max: float = 0.5             # max joint-space L2 distance for edges (rad)

    # Quality
    min_quality: float = 0.03


class JointSpaceRRTGenerator:
    """
    Generate kinematically feasible grasps by exploring joint space directly.

    Requires an Isaac Sim environment with robot + object for FK computation.
    """

    def __init__(
        self,
        env,
        mesh: trimesh.Trimesh,
        nfo: NetForceOptimizer,
        ft_ids: List[int],
        cfg: JointSpaceRRTConfig,
        object_name: str = "",
        object_size: float = 0.06,
        rng: Optional[np.random.Generator] = None,
    ):
        self.env = env
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.env_ids = torch.tensor([0], device=self.device, dtype=torch.long)

        self.mesh = mesh
        self.nfo = nfo
        self.ft_ids = ft_ids
        self.cfg = cfg
        self.object_name = object_name
        self.object_size = object_size
        self.rng = rng or np.random.default_rng()

        self.num_fingers = len(ft_ids)
        self.num_dof = self.robot.data.joint_pos.shape[-1]

        # Joint limits
        self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()

        # All finger link body IDs for penetration check (excluding root/palm)
        self.finger_body_ids = self._resolve_finger_body_ids()

        # Object world pose (set once in setup)
        self.obj_pos_w: Optional[torch.Tensor] = None
        self.obj_quat_w: Optional[torch.Tensor] = None

    def generate(self) -> GraspGraph:
        """Run full joint-space RRT and return a GraspGraph."""
        self._setup_object_pose()

        # Phase 1: Seed generation
        print(f"  [JointRRT] Generating seeds (max {self.cfg.num_seed_attempts} attempts)...")
        seeds = self._generate_seeds()
        print(f"  [JointRRT] {len(seeds)} seeds found")

        if len(seeds) == 0:
            print("  [JointRRT] WARNING: no valid seeds found")
            return GraspGraph(
                grasp_set=GraspSet([]),
                edges=[],
                object_name=self.object_name,
                num_fingers=self.num_fingers,
            )

        # Phase 2: RRT expansion
        print(f"  [JointRRT] Expanding to {self.cfg.target_size} grasps...")
        grasps = list(seeds)
        total_attempts = 0

        while len(grasps) < self.cfg.target_size:
            new_grasp = self._expand_step(grasps)
            if new_grasp is not None:
                grasps.append(new_grasp)
                if len(grasps) % 50 == 0:
                    print(f"  [JointRRT] {len(grasps)}/{self.cfg.target_size} grasps")

            total_attempts += 1
            if total_attempts >= self.cfg.max_total_expand_attempts:
                print(f"  [JointRRT] Stopping early at {len(grasps)} grasps "
                      f"({total_attempts} attempts)")
                break

        # Phase 3: Build graph edges
        edges = self._build_edges(grasps)
        print(f"  [JointRRT] {len(grasps)} grasps, {len(edges)} edges")

        grasp_set = GraspSet(grasps=grasps, object_name=self.object_name)
        return GraspGraph(
            grasp_set=grasp_set,
            edges=edges,
            object_name=self.object_name,
            num_fingers=self.num_fingers,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_finger_body_ids(self) -> List[int]:
        """Get body IDs of all finger links (proximal through distal) for penetration check."""
        # Shadow Hand finger link naming convention
        _SHADOW_FINGER_LINKS = [
            # FF
            "robot0_ffknuckle", "robot0_ffproximal", "robot0_ffmiddle", "robot0_ffdistal",
            # MF
            "robot0_mfknuckle", "robot0_mfproximal", "robot0_mfmiddle", "robot0_mfdistal",
            # RF
            "robot0_rfknuckle", "robot0_rfproximal", "robot0_rfmiddle", "robot0_rfdistal",
            # LF
            "robot0_lfmetacarpal", "robot0_lfknuckle", "robot0_lfproximal",
            "robot0_lfmiddle", "robot0_lfdistal",
            # TH
            "robot0_thbase", "robot0_thproximal", "robot0_thhub",
            "robot0_thmiddle", "robot0_thdistal",
        ]

        body_ids = []
        for name in _SHADOW_FINGER_LINKS:
            try:
                ids = self.robot.find_bodies(name)[0]
                if len(ids) > 0:
                    body_ids.append(int(ids[0]))
            except Exception:
                pass

        if not body_ids:
            # Fallback: use all bodies except first 2 (root + palm)
            num_bodies = self.robot.data.body_pos_w.shape[1]
            body_ids = list(range(2, num_bodies))

        print(f"  [JointRRT] Finger link body IDs for penetration check: "
              f"{len(body_ids)} bodies")
        return body_ids

    def _setup_object_pose(self):
        """Place object at fingertip centroid of joint-midpoint pose."""
        # Set joints to midpoint
        q_mid = (self.q_low + self.q_high) / 2.0
        q_mid[:2] = 0.0  # wrist fixed
        self._set_joints(q_mid)

        # Read fingertip positions after FK
        ft_world = self.robot.data.body_pos_w[self.env_ids][:, self.ft_ids, :]
        centroid = ft_world.mean(dim=1)  # (1, 3)

        self.obj_pos_w = centroid.clone()
        self.obj_quat_w = torch.zeros(1, 4, device=self.device)
        self.obj_quat_w[:, 0] = 1.0  # identity

        # Write object to sim
        obj_state = self.obj.data.default_root_state[self.env_ids].clone()
        obj_state[:, :3] = self.obj_pos_w
        obj_state[:, 3:7] = self.obj_quat_w
        obj_state[:, 7:] = 0.0
        self.obj.write_root_state_to_sim(obj_state, env_ids=self.env_ids)
        self.obj.update(0.0)

        print(f"  [JointRRT] Object placed at {self.obj_pos_w[0].tolist()}")

    # ------------------------------------------------------------------
    # FK + Validation
    # ------------------------------------------------------------------

    def _set_joints(self, q: torch.Tensor):
        """Set joint positions and run FK."""
        q_2d = q.unsqueeze(0) if q.ndim == 1 else q
        self.robot.write_joint_state_to_sim(
            q_2d, torch.zeros_like(q_2d), env_ids=self.env_ids,
        )
        self.robot.set_joint_position_target(q_2d, env_ids=self.env_ids)
        self.robot.update(0.0)

    def _get_fingertip_world(self) -> torch.Tensor:
        """Get current fingertip positions in world frame. Returns (F, 3)."""
        return self.robot.data.body_pos_w[0, self.ft_ids, :].clone()

    def _world_to_object_frame(self, pts_w: torch.Tensor) -> np.ndarray:
        """Transform world positions (F, 3) to object frame. Returns (F, 3) numpy."""
        # Object has identity quaternion, so just subtract position
        pts_obj = pts_w - self.obj_pos_w[0]
        return pts_obj.cpu().numpy()

    def _evaluate_joint_config(self, q: torch.Tensor) -> Optional[Grasp]:
        """
        Set joints, FK, check penetration + surface proximity + quality.
        Returns Grasp if valid, None otherwise.
        """
        self._set_joints(q)

        # Penetration check: reject if any finger link penetrates the object
        # beyond the allowed margin. Uses signed distance via surface normal.
        finger_link_world = self.robot.data.body_pos_w[0, self.finger_body_ids, :]
        finger_link_obj = self._world_to_object_frame(finger_link_world)
        fl_closest, fl_dists, fl_face_idx = trimesh.proximity.closest_point(
            self.mesh, finger_link_obj,
        )
        # Signed penetration: negative dot(to_point, normal) means inside
        to_point = finger_link_obj - fl_closest
        fl_normals = self.mesh.face_normals[fl_face_idx]
        sign = np.sum(to_point * fl_normals, axis=-1)
        # Penetration depth: only where sign < 0 (inside mesh)
        penetration = np.where(sign < 0, fl_dists, 0.0)
        if np.any(penetration > self.cfg.penetration_margin):
            return None

        # Surface proximity check (fingertips only)
        ft_world = self._get_fingertip_world()  # (F, 3)
        ft_obj = self._world_to_object_frame(ft_world)  # (F, 3) numpy

        closest, dists, face_idx = trimesh.proximity.closest_point(
            self.mesh, ft_obj,
        )

        if np.any(dists > self.cfg.contact_threshold):
            return None

        # Contact normals from mesh faces
        normals = self.mesh.face_normals[face_idx].astype(np.float32)

        # NFO quality check
        grasp = Grasp(
            fingertip_positions=closest.astype(np.float32),
            contact_normals=normals,
            quality=0.0,
            object_name=self.object_name,
            object_scale=self.object_size,
        )
        quality = self.nfo.evaluate(grasp)
        if quality < self.cfg.min_quality:
            return None

        # Store all fields needed for RL
        grasp.quality = quality
        grasp.joint_angles = q.cpu().numpy().copy()

        # Object pose in hand frame
        rp = self.robot.data.root_pos_w[0]
        rq = self.robot.data.root_quat_w[0]
        op = self.obj_pos_w[0]

        from isaaclab.utils.math import quat_apply_inverse
        from envs.mdp.math_utils import quat_multiply, quat_conjugate

        rel = op - rp
        obj_pos_hand = quat_apply_inverse(rq.unsqueeze(0), rel.unsqueeze(0))[0]
        obj_quat_hand = quat_multiply(
            quat_conjugate(rq.unsqueeze(0)),
            self.obj_quat_w,
        )[0]

        grasp.object_pos_hand = obj_pos_hand.cpu().numpy().copy()
        grasp.object_quat_hand = obj_quat_hand.cpu().numpy().copy()
        grasp.object_pose_frame = "hand_root"

        return grasp

    # ------------------------------------------------------------------
    # Seed generation
    # ------------------------------------------------------------------

    def _random_joint_config(self) -> torch.Tensor:
        """Sample a random joint configuration within soft limits."""
        alpha = torch.rand(self.num_dof, device=self.device)
        q = self.q_low + alpha * (self.q_high - self.q_low)
        q[:2] = 0.0  # wrist fixed
        return q

    def _generate_seeds(self) -> List[Grasp]:
        seeds = []
        for i in range(self.cfg.num_seed_attempts):
            q = self._random_joint_config()
            grasp = self._evaluate_joint_config(q)
            if grasp is not None:
                seeds.append(grasp)
                if len(seeds) % 10 == 0:
                    print(f"    seeds: {len(seeds)} (attempt {i+1})")
        return seeds

    # ------------------------------------------------------------------
    # RRT expansion
    # ------------------------------------------------------------------

    def _expand_step(self, grasps: List[Grasp]) -> Optional[Grasp]:
        """Perturb a random parent's joint config and validate."""
        for _ in range(self.cfg.max_expand_attempts_per_step):
            parent = grasps[self.rng.integers(0, len(grasps))]
            q_parent = torch.tensor(
                parent.joint_angles, device=self.device, dtype=torch.float32,
            )

            # Perturb
            noise = torch.randn_like(q_parent) * self.cfg.joint_noise_std
            noise[:2] = 0.0  # wrist fixed
            q_new = torch.clamp(q_parent + noise, self.q_low, self.q_high)

            grasp = self._evaluate_joint_config(q_new)
            if grasp is not None:
                return grasp

        return None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_edges(self, grasps: List[Grasp]) -> List[Tuple[int, int]]:
        """Build edges between grasps within delta_max joint-space distance."""
        n = len(grasps)
        if n < 2:
            return []

        # Stack all joint angles
        all_joints = np.stack([g.joint_angles for g in grasps])  # (N, num_dof)

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(all_joints[i] - all_joints[j]))
                if dist < self.cfg.delta_max:
                    edges.append((i, j))

        return edges
