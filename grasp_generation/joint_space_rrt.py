from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh

from isaaclab.utils.math import quat_apply_inverse

from envs.mdp.math_utils import quat_conjugate, quat_multiply
from envs.mdp.sim_utils import get_palm_body_id_from_env
from .grasp_sampler import Grasp, GraspSet
from .net_force_optimization import NetForceOptimizer
from .rrt_expansion import GraspGraph


@dataclass
class JointSpaceRRTConfig:
    """Parameters for Shadow-Hand-friendly joint-space RRT generation."""

    target_size: int = 300

    # Seed search
    num_seed_attempts: int = 6000
    max_seed_keep: int = 48
    min_seed_joint_dist: float = 0.20

    # Validation
    contact_threshold: float = 0.018
    seed_contact_threshold: float = 0.030
    min_quality: float = 0.03
    seed_min_quality: float = 0.010
    min_contacts_seed: int = 3
    min_contacts_expand: int = 4

    # Expansion
    max_expand_attempts_per_step: int = 60
    max_total_expand_attempts: int = 150000
    joint_noise_std: float = 0.08
    thumb_noise_scale: float = 1.25
    local_finger_move_prob: float = 0.75

    # Graph edges
    delta_max: float = 0.55

    # Penetration / object placement
    use_penetration_check: bool = True
    penetration_margin: float = 0.004
    object_clearance_scale: float = 0.55
    object_clearance_bias: float = 0.010

    # Preshape library
    preshape_noise_std: float = 0.18


class JointSpaceRRTGenerator:
    """
    Generate Shadow Hand grasps directly in joint space.

    Main idea:
    - Keep the object fixed relative to the hand during generation.
    - Generate seeds near grasp-like preshapes instead of full-range random joints.
    - Use relaxed validation for seed discovery, then stricter validation for RRT expansion.
    - Every accepted node stores joint_angles + fingertip_positions + object pose.
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
        self.object_size = float(object_size)
        self.rng = rng or np.random.default_rng()

        self.num_fingers = len(ft_ids)
        self.num_dof = self.robot.data.joint_pos.shape[-1]

        self.q_low = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.q_high = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()

        self.palm_body_id = get_palm_body_id_from_env(self.robot, self.env)
        self.finger_body_ids = self._resolve_finger_body_ids()
        self.active_link_body_ids = self._resolve_active_link_body_ids()

        self.obj_pos_w: Optional[torch.Tensor] = None
        self.obj_quat_w: Optional[torch.Tensor] = None

        self.finger_joint_groups = self._build_finger_joint_groups()
        self.preshape_library = self._build_preshape_library()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> GraspGraph:
        self._setup_object_pose()

        print(f"  [JointRRT] Finger link body IDs for penetration check: {len(self.active_link_body_ids)} bodies")
        print(f"  [JointRRT] Object placed at {self.obj_pos_w[0].tolist()}")
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

        print(f"  [JointRRT] Expanding to {self.cfg.target_size} grasps...")
        grasps = list(seeds)
        total_attempts = 0

        while len(grasps) < self.cfg.target_size:
            new_grasp = self._expand_step(grasps)
            if new_grasp is not None and not self._is_duplicate(grasps, new_grasp.joint_angles, self.cfg.min_seed_joint_dist * 0.75):
                grasps.append(new_grasp)
                if len(grasps) % 50 == 0:
                    print(f"  [JointRRT] {len(grasps)}/{self.cfg.target_size} grasps")

            total_attempts += 1
            if total_attempts >= self.cfg.max_total_expand_attempts:
                print(f"  [JointRRT] Stopping early at {len(grasps)} grasps ({total_attempts} attempts)")
                break

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
        names = [
            "robot0_ffknuckle", "robot0_ffproximal", "robot0_ffmiddle", "robot0_ffdistal",
            "robot0_mfknuckle", "robot0_mfproximal", "robot0_mfmiddle", "robot0_mfdistal",
            "robot0_rfknuckle", "robot0_rfproximal", "robot0_rfmiddle", "robot0_rfdistal",
            "robot0_lfmetacarpal", "robot0_lfknuckle", "robot0_lfproximal", "robot0_lfmiddle", "robot0_lfdistal",
            "robot0_thbase", "robot0_thproximal", "robot0_thhub", "robot0_thmiddle", "robot0_thdistal",
        ]
        body_ids = []
        for name in names:
            try:
                ids = self.robot.find_bodies(name)[0]
                if len(ids) > 0:
                    body_ids.append(int(ids[0]))
            except Exception:
                pass

        if not body_ids:
            num_bodies = self.robot.data.body_pos_w.shape[1]
            body_ids = list(range(2, num_bodies))
        return body_ids

    def _resolve_active_link_body_ids(self) -> List[int]:
        """
        Slightly relaxed penetration set:
        use mainly middle/distal links instead of every knuckle.
        This helps seed generation a lot for Shadow Hand.
        """
        names = [
            "robot0_ffmiddle", "robot0_ffdistal",
            "robot0_mfmiddle", "robot0_mfdistal",
            "robot0_rfmiddle", "robot0_rfdistal",
            "robot0_lfmiddle", "robot0_lfdistal",
            "robot0_thmiddle", "robot0_thdistal",
        ]
        body_ids = []
        for name in names:
            try:
                ids = self.robot.find_bodies(name)[0]
                if len(ids) > 0:
                    body_ids.append(int(ids[0]))
            except Exception:
                pass
        return body_ids if body_ids else self.finger_body_ids

    def _build_finger_joint_groups(self) -> List[List[int]]:
        # Isaac Shadow DOF layout (24):
        # [0-1]=wrist
        # [2-5]=FF, [6-9]=MF, [10-13]=RF, [14-18]=LF, [19-23]=TH
        return [
            [2, 3, 4, 5],         # FF
            [6, 7, 8, 9],         # MF
            [10, 11, 12, 13],     # RF
            [14, 15, 16, 17, 18], # LF
            [19, 20, 21, 22, 23], # TH
        ]

    def _build_preshape_library(self) -> List[torch.Tensor]:
        """
        Hand-crafted Shadow Hand preshapes.
        Purpose is only to make seed generation non-degenerate.
        """
        presets = []

        def clamp(q: torch.Tensor) -> torch.Tensor:
            q = torch.max(torch.min(q, self.q_high), self.q_low)
            q[:2] = 0.0
            return q

        # Midpoint base
        q_mid = (self.q_low + self.q_high) / 2.0
        q_mid[:2] = 0.0

        # Preshape 1: gentle enclosure
        q1 = q_mid.clone()
        q1[[3, 4, 5]] = torch.tensor([0.55, 0.75, 0.55], device=self.device)
        q1[[7, 8, 9]] = torch.tensor([0.55, 0.75, 0.55], device=self.device)
        q1[[11, 12, 13]] = torch.tensor([0.55, 0.75, 0.55], device=self.device)
        q1[[15, 16, 17, 18]] = torch.tensor([0.15, 0.45, 0.75, 0.55], device=self.device)
        q1[[19, 20, 21, 22, 23]] = torch.tensor([0.30, 0.55, 0.45, 0.35, 0.25], device=self.device)
        presets.append(clamp(q1))

        # Preshape 2: more closed power grasp
        q2 = q_mid.clone()
        q2[[3, 4, 5]] = torch.tensor([0.80, 1.05, 0.75], device=self.device)
        q2[[7, 8, 9]] = torch.tensor([0.80, 1.05, 0.75], device=self.device)
        q2[[11, 12, 13]] = torch.tensor([0.80, 1.05, 0.75], device=self.device)
        q2[[15, 16, 17, 18]] = torch.tensor([0.20, 0.65, 1.00, 0.70], device=self.device)
        q2[[19, 20, 21, 22, 23]] = torch.tensor([0.45, 0.80, 0.65, 0.55, 0.35], device=self.device)
        presets.append(clamp(q2))

        # Preshape 3: thumb-opposed enclosure
        q3 = q_mid.clone()
        q3[[3, 4, 5]] = torch.tensor([0.65, 0.95, 0.65], device=self.device)
        q3[[7, 8, 9]] = torch.tensor([0.65, 0.95, 0.65], device=self.device)
        q3[[11, 12, 13]] = torch.tensor([0.65, 0.95, 0.65], device=self.device)
        q3[[15, 16, 17, 18]] = torch.tensor([0.12, 0.55, 0.90, 0.62], device=self.device)
        q3[[19, 20, 21, 22, 23]] = torch.tensor([0.65, 1.00, 0.75, 0.65, 0.55], device=self.device)
        presets.append(clamp(q3))

        return presets

    def _setup_object_pose(self):
        """
        Place object above the palm using a size-aware offset.
        This is much more stable for Shadow Hand than fingertip-centroid placement.
        """
        q_seed = self.preshape_library[0].clone()
        self._set_joints(q_seed)

        palm_pos = self.robot.data.body_pos_w[0:1, self.palm_body_id, :].clone()
        obj_pos = palm_pos.clone()
        obj_pos[:, 2] += self.cfg.object_clearance_scale * self.object_size + self.cfg.object_clearance_bias

        obj_quat = torch.zeros(1, 4, device=self.device)
        obj_quat[:, 0] = 1.0

        obj_state = self.obj.data.default_root_state[self.env_ids].clone()
        obj_state[:, :3] = obj_pos
        obj_state[:, 3:7] = obj_quat
        obj_state[:, 7:] = 0.0
        self.obj.write_root_state_to_sim(obj_state, env_ids=self.env_ids)
        self.obj.update(0.0)

        self.obj_pos_w = obj_pos
        self.obj_quat_w = obj_quat

    # ------------------------------------------------------------------
    # FK / validation
    # ------------------------------------------------------------------

    def _set_joints(self, q: torch.Tensor):
        q_2d = q.unsqueeze(0) if q.ndim == 1 else q
        self.robot.write_joint_state_to_sim(
            q_2d,
            torch.zeros_like(q_2d),
            env_ids=self.env_ids,
        )
        self.robot.set_joint_position_target(q_2d, env_ids=self.env_ids)
        self.robot.update(0.0)

    def _get_fingertip_world(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[0, self.ft_ids, :].clone()

    def _world_to_object_frame(self, pts_w: torch.Tensor) -> np.ndarray:
        pts_obj = pts_w - self.obj_pos_w[0]
        return pts_obj.detach().cpu().numpy()

    def _count_active_contacts(self, dists: np.ndarray, thresh: float) -> int:
        return int(np.sum(dists <= thresh))

    def _penetration_violation(self) -> bool:
        if not self.cfg.use_penetration_check:
            return False
        if len(self.active_link_body_ids) == 0:
            return False

        link_world = self.robot.data.body_pos_w[0, self.active_link_body_ids, :]
        link_obj = self._world_to_object_frame(link_world)

        try:
            signed = trimesh.proximity.signed_distance(self.mesh, link_obj)
            # trimesh signed_distance: inside points are usually positive.
            if np.any(signed > self.cfg.penetration_margin):
                return True
            return False
        except Exception:
            try:
                inside = self.mesh.contains(link_obj)
                return bool(np.any(inside))
            except Exception:
                return False

    def _evaluate_joint_config(
        self,
        q: torch.Tensor,
        contact_threshold: float,
        min_quality: float,
        min_contacts: int,
    ) -> Optional[Grasp]:
        self._set_joints(q)

        if self._penetration_violation():
            return None

        ft_world = self._get_fingertip_world()
        ft_obj = self._world_to_object_frame(ft_world)

        try:
            closest, dists, face_idx = trimesh.proximity.closest_point(self.mesh, ft_obj)
        except Exception:
            return None

        active_contacts = self._count_active_contacts(dists, contact_threshold)
        if active_contacts < min_contacts:
            return None

        normals = self.mesh.face_normals[face_idx].astype(np.float32)

        grasp = Grasp(
            fingertip_positions=closest.astype(np.float32),
            contact_normals=normals,
            quality=0.0,
            object_name=self.object_name,
            object_scale=self.object_size,
        )

        try:
            quality = float(self.nfo.evaluate(grasp))
        except Exception:
            return None

        if quality < min_quality:
            return None

        rp = self.robot.data.root_pos_w[0]
        rq = self.robot.data.root_quat_w[0]
        op = self.obj_pos_w[0]

        rel = op - rp
        obj_pos_hand = quat_apply_inverse(rq.unsqueeze(0), rel.unsqueeze(0))[0]
        obj_quat_hand = quat_multiply(
            quat_conjugate(rq.unsqueeze(0)),
            self.obj_quat_w,
        )[0]

        grasp.quality = quality
        grasp.joint_angles = q.detach().cpu().numpy().copy()
        grasp.object_pos_hand = obj_pos_hand.detach().cpu().numpy().copy()
        grasp.object_quat_hand = obj_quat_hand.detach().cpu().numpy().copy()
        grasp.object_pose_frame = "hand_root"

        return grasp

    # ------------------------------------------------------------------
    # Seed generation
    # ------------------------------------------------------------------

    def _sample_from_preshape(self) -> torch.Tensor:
        base = self.preshape_library[self.rng.integers(0, len(self.preshape_library))].clone()
        noise = torch.randn(self.num_dof, device=self.device) * self.cfg.preshape_noise_std
        noise[:2] = 0.0

        # Thumb gets slightly more exploration
        thumb_idx = self.finger_joint_groups[-1]
        noise[thumb_idx] *= self.cfg.thumb_noise_scale

        q = base + noise
        q = torch.max(torch.min(q, self.q_high), self.q_low)
        q[:2] = 0.0
        return q

    def _is_duplicate(self, grasps: List[Grasp], q_new: np.ndarray, min_dist: float) -> bool:
        if len(grasps) == 0:
            return False
        for g in grasps:
            if g.joint_angles is None:
                continue
            d = float(np.linalg.norm(np.asarray(g.joint_angles) - q_new))
            if d < min_dist:
                return True
        return False

    def _generate_seeds(self) -> List[Grasp]:
        seeds: List[Grasp] = []

        for i in range(self.cfg.num_seed_attempts):
            q = self._sample_from_preshape()
            grasp = self._evaluate_joint_config(
                q=q,
                contact_threshold=self.cfg.seed_contact_threshold,
                min_quality=self.cfg.seed_min_quality,
                min_contacts=self.cfg.min_contacts_seed,
            )
            if grasp is None:
                continue

            if self._is_duplicate(seeds, grasp.joint_angles, self.cfg.min_seed_joint_dist):
                continue

            seeds.append(grasp)
            seeds.sort(key=lambda g: float(g.quality), reverse=True)

            if len(seeds) > self.cfg.max_seed_keep:
                seeds = seeds[: self.cfg.max_seed_keep]

            if len(seeds) % 10 == 0:
                print(f"    seeds: {len(seeds)} (attempt {i + 1})")

        return seeds

    # ------------------------------------------------------------------
    # RRT expansion
    # ------------------------------------------------------------------

    def _perturb_parent_q(self, q_parent: torch.Tensor) -> torch.Tensor:
        q_new = q_parent.clone()

        if self.rng.random() < self.cfg.local_finger_move_prob:
            # Move one or two finger groups only
            num_groups = 1 if self.rng.random() < 0.8 else 2
            chosen = self.rng.choice(len(self.finger_joint_groups), size=num_groups, replace=False)
            for gi in np.atleast_1d(chosen):
                idx = self.finger_joint_groups[int(gi)]
                scale = self.cfg.joint_noise_std * (self.cfg.thumb_noise_scale if int(gi) == 4 else 1.0)
                q_new[idx] += torch.randn(len(idx), device=self.device) * scale
        else:
            # Global small move
            noise = torch.randn_like(q_new) * (self.cfg.joint_noise_std * 0.6)
            noise[:2] = 0.0
            q_new = q_new + noise

        q_new = torch.max(torch.min(q_new, self.q_high), self.q_low)
        q_new[:2] = 0.0
        return q_new

    def _sample_parent(self, grasps: List[Grasp]) -> Grasp:
        qualities = np.array([max(float(g.quality), 1e-6) for g in grasps], dtype=np.float64)
        probs = qualities / qualities.sum()
        idx = int(self.rng.choice(len(grasps), p=probs))
        return grasps[idx]

    def _expand_step(self, grasps: List[Grasp]) -> Optional[Grasp]:
        for _ in range(self.cfg.max_expand_attempts_per_step):
            parent = self._sample_parent(grasps)
            q_parent = torch.tensor(parent.joint_angles, device=self.device, dtype=torch.float32)
            q_new = self._perturb_parent_q(q_parent)

            grasp = self._evaluate_joint_config(
                q=q_new,
                contact_threshold=self.cfg.contact_threshold,
                min_quality=self.cfg.min_quality,
                min_contacts=self.cfg.min_contacts_expand,
            )
            if grasp is not None:
                return grasp
        return None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_edges(self, grasps: List[Grasp]) -> List[Tuple[int, int]]:
        n = len(grasps)
        if n < 2:
            return []

        all_joints = np.stack([np.asarray(g.joint_angles) for g in grasps], axis=0)
        edges: List[Tuple[int, int]] = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(all_joints[i] - all_joints[j]))
                if dist < self.cfg.delta_max:
                    edges.append((i, j))

        return edges
