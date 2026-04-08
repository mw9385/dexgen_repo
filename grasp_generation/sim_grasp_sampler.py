"""
DexterityGen Algorithm 3 — Grasp Set Generation
=================================================
Paper reference (Appendix, Algorithm 3):
  1. P, n ← sampleSurface(M, N_pts)
  2. quality ← graspAnalysis(P, n)          [Algorithm 4: NFO]
  3. pose ← randomPose(hand_frame)
  4. q ← assign(IK)
  5. collision check → reject if penetrating
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh

from .grasp_sampler import Grasp, GraspSet, _resolve_finger_body_ids, _compute_obj_pose_hand
from .net_force_optimization import NetForceOptimizer


def make_primitive_mesh(shape: str, size: float) -> trimesh.Trimesh:
    if shape == "cube":
        return trimesh.creation.box(extents=[size, size, size])
    elif shape == "sphere":
        return trimesh.creation.icosphere(radius=size / 2.0, subdivisions=3)
    elif shape == "cylinder":
        return trimesh.creation.cylinder(radius=size / 2.0, height=size)
    else:
        raise ValueError(f"Unknown shape: {shape}")


# ---------------------------------------------------------------------------
# Step 1: sampleSurface(M, N_pts)
# ---------------------------------------------------------------------------

def sample_surface(mesh: trimesh.Trimesh, n_pts: int, rng: np.random.Generator):
    """Sample n_pts contact points + normals on mesh surface."""
    points, face_idx = trimesh.sample.sample_surface(mesh, n_pts)
    normals = mesh.face_normals[face_idx]
    return points.astype(np.float32), normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: graspAnalysis(P, n) — Algorithm 4 (NFO)
# ---------------------------------------------------------------------------

def grasp_analysis(points: np.ndarray, normals: np.ndarray, nfo: NetForceOptimizer) -> float:
    """Evaluate NFO ε-metric quality for a set of contact points."""
    return nfo.evaluate(Grasp(fingertip_positions=points, contact_normals=normals))


# ---------------------------------------------------------------------------
# Step 3: randomPose — sample object pose in hand frame
# ---------------------------------------------------------------------------

def sample_random_pose(
    base_pos: np.ndarray,
    rng: np.random.Generator,
    pos_std: float = 0.02,
    rot_std: float = 0.3,
):
    """
    Sample a random object pose near base_pos.

    Returns (position, rotation_matrix) where:
      - position: base_pos + Gaussian offset (±pos_std)
      - rotation: small random rotation (±rot_std radians)
    """
    from scipy.spatial.transform import Rotation as R_scipy

    pos_offset = rng.normal(0, pos_std, size=3).astype(np.float32)
    pos = base_pos + pos_offset

    angle = rng.normal(0, rot_std)
    axis = rng.normal(0, 1, size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    rot = R_scipy.from_rotvec(angle * axis).as_matrix().astype(np.float32)

    return pos, rot


# ---------------------------------------------------------------------------
# Step 4: assign(IK) — solve joint angles
# ---------------------------------------------------------------------------

def solve_ik(
    env,
    contact_points_obj: np.ndarray,  # (n_pts, 3) in object frame
    obj_pos: np.ndarray,             # (3,) world
    obj_rot: np.ndarray,             # (3, 3) rotation matrix
    palmup_pos, palmup_quat,
    object_size: float,
    ft_ids: list,
    ik_iterations: int = 30,
    ik_threshold: float = 0.03,
    render: bool = False,
) -> Optional[np.ndarray]:
    """
    Solve IK to place fingertips at contact points.

    Returns joint angles (24-DOF) or None if IK fails.
    """
    from envs.mdp.sim_utils import (
        set_robot_root_pose,
        set_adaptive_joint_pose,
        refine_hand_to_start_grasp,
    )

    robot = env.scene["robot"]
    obj_actor = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    # Transform contact points to world frame
    contact_world = (obj_rot @ contact_points_obj.T).T + obj_pos
    fp_tensor = torch.tensor(
        contact_world, device=device, dtype=torch.float32,
    ).unsqueeze(0)  # (1, n_pts, 3)

    # Palm-up wrist + midpoint joints as IK start
    set_robot_root_pose(env, env_ids, palmup_pos, palmup_quat)
    set_adaptive_joint_pose(env, env_ids, object_size)

    q_mid = robot.data.joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)

    # Place object at pose
    from scipy.spatial.transform import Rotation as R_scipy
    obj_quat_scipy = R_scipy.from_matrix(obj_rot).as_quat()  # (x,y,z,w)
    obj_quat_t = torch.tensor(
        [obj_quat_scipy[3], obj_quat_scipy[0],
         obj_quat_scipy[1], obj_quat_scipy[2]],
        device=device, dtype=torch.float32,
    ).unsqueeze(0)
    obj_pos_t = torch.tensor(obj_pos, device=device, dtype=torch.float32).unsqueeze(0)

    obj_state = obj_actor.data.default_root_state[env_ids].clone()
    obj_state[:, :3] = obj_pos_t
    obj_state[:, 3:7] = obj_quat_t
    obj_state[:, 7:] = 0.0
    obj_actor.write_root_state_to_sim(obj_state, env_ids=env_ids)
    obj_actor.update(0.0)

    # FK steps to stabilise
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)

    # IK solve
    ik_cfg = getattr(env.cfg, "reset_refinement", None) or {}
    if not isinstance(ik_cfg, dict):
        ik_cfg = {}
    ik_cfg["iterations"] = ik_iterations
    env.cfg.reset_refinement = ik_cfg

    refine_hand_to_start_grasp(env, env_ids, fp_tensor)
    robot.update(0.0)

    # Check convergence
    ft_after = robot.data.body_pos_w[0, ft_ids, :]
    ft_err = torch.norm(ft_after - fp_tensor[0], dim=-1)
    if float(ft_err.mean()) > ik_threshold:
        return None

    return robot.data.joint_pos[0].cpu().numpy().copy()


# ---------------------------------------------------------------------------
# Step 5: collision check
# ---------------------------------------------------------------------------

def check_collision(
    env,
    finger_body_ids: list,
    obj_pos: np.ndarray,
    obj_rot: np.ndarray,
    mesh: trimesh.Trimesh,
    margin: float = 0.001,
) -> bool:
    """
    Check if finger links penetrate the object mesh.
    Returns True if collision exists (should reject).
    """
    robot = env.scene["robot"]
    fl_world = robot.data.body_pos_w[0, finger_body_ids, :].cpu().numpy()

    # Transform to object frame
    rot_inv = obj_rot.T
    fl_obj = (rot_inv @ (fl_world - obj_pos).T).T

    closest, dists, face_idx = trimesh.proximity.closest_point(mesh, fl_obj)
    to_pt = fl_obj - closest
    normals = mesh.face_normals[face_idx]
    sign = np.sum(to_pt * normals, axis=-1)
    penetration = np.where(sign < 0, dists, 0.0)

    return bool(np.any(penetration > margin))


# ---------------------------------------------------------------------------
# Algorithm 3: Full pipeline
# ---------------------------------------------------------------------------

def generate_grasp_set(
    env,
    mesh: trimesh.Trimesh,
    object_name: str,
    object_size: float,
    n_pts: int = 5,
    num_grasps: int = 300,
    max_candidates: int = 100000,
    num_pose_samples: int = 10,
    nfo_min_quality: float = 0.03,
    collision_margin: float = 0.001,
    ik_iterations: int = 30,
    render: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> GraspSet:
    """
    DexterityGen Algorithm 3.

    Args:
        n_pts: number of contact points per grasp (3, 4, or 5).
    """
    from envs.mdp.sim_utils import (
        get_fingertip_body_ids_from_env,
        set_robot_root_pose,
        set_adaptive_joint_pose,
        get_local_palm_normal,
    )
    from envs.mdp.math_utils import quat_from_two_vectors, quat_multiply
    from isaaclab.utils.math import quat_apply

    robot = env.scene["robot"]
    device = env.device
    rng = np.random.default_rng(seed)

    ft_ids = get_fingertip_body_ids_from_env(robot, env)[:n_pts]
    finger_body_ids = _resolve_finger_body_ids(robot)

    nfo = NetForceOptimizer(mu=0.5, num_edges=8, min_quality=nfo_min_quality)

    # ── Compute palm-up wrist pose ───────────────────────────────
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    root_state = robot.data.default_root_state[env_ids, :7].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])
    set_adaptive_joint_pose(env, env_ids, object_size)

    temp = env.scene["object"].data.default_root_state[env_ids].clone()
    temp[:, :3] = env.scene.env_origins[env_ids] + torch.tensor([[0, 0, -10.0]], device=device)
    temp[:, 7:] = 0.0
    env.scene["object"].write_root_state_to_sim(temp, env_ids=env_ids)
    env.scene["object"].update(0.0)

    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)

    wrist_pos = robot.data.root_pos_w[env_ids].clone()
    wrist_quat = robot.data.root_quat_w[env_ids].clone()
    palm_n = get_local_palm_normal(robot, env).unsqueeze(0)
    palm_w = quat_apply(wrist_quat, palm_n)
    correction = quat_from_two_vectors(palm_w, torch.tensor([[0, 0, 1.0]], device=device))
    ft_w = robot.data.body_pos_w[env_ids][:, ft_ids, :]
    pivot = ft_w.mean(dim=1)
    palmup_quat = quat_multiply(correction, wrist_quat)
    palmup_quat = palmup_quat / (torch.norm(palmup_quat, dim=-1, keepdim=True) + 1e-8)
    palmup_pos = quat_apply(correction, wrist_pos - pivot) + pivot

    # Get base object position (fingertip centroid of palm-up midpoint)
    set_robot_root_pose(env, env_ids, palmup_pos, palmup_quat)
    q_mid = robot.data.joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    base_obj_pos = robot.data.body_pos_w[0, ft_ids, :].mean(dim=0).cpu().numpy()

    if verbose:
        print(f"  [Algorithm 3] n_pts={n_pts}, target={num_grasps}, "
              f"max_candidates={max_candidates}")
        print(f"    base_obj_pos={base_obj_pos.tolist()}")

    # ── Main loop ────────────────────────────────────────────────
    grasps = []
    stats = {"nfo_fail": 0, "ik_fail": 0, "collision_fail": 0}

    for attempt in range(max_candidates):
        if len(grasps) >= num_grasps:
            break

        # Step 1: sampleSurface
        P, n = sample_surface(mesh, n_pts, rng)

        # Step 2: graspAnalysis (NFO)
        quality = grasp_analysis(P, n, nfo)
        if quality < nfo_min_quality:
            stats["nfo_fail"] += 1
            continue

        # Step 3-5: try multiple random poses → IK → collision
        found = False
        for _ in range(num_pose_samples):
            obj_pos, obj_rot = sample_random_pose(base_obj_pos, rng)

            # Step 4: IK
            q = solve_ik(
                env, P, obj_pos, obj_rot,
                palmup_pos, palmup_quat,
                object_size, ft_ids,
                ik_iterations=ik_iterations,
                render=render,
            )
            if q is None:
                continue

            # Step 5: collision check
            if check_collision(env, finger_body_ids, obj_pos, obj_rot,
                               mesh, margin=collision_margin):
                stats["collision_fail"] += 1
                continue

            # Success — build Grasp
            obj_quat_w = env.scene["object"].data.root_quat_w[0]
            obj_pos_t = torch.tensor(obj_pos, device=device, dtype=torch.float32)
            pos_hand, quat_hand = _compute_obj_pose_hand(
                robot, obj_pos_t, device, obj_quat_w=obj_quat_w,
            )

            # Contact points in rotated object frame
            fp_rotated = (obj_rot @ P.T).T.astype(np.float32)

            grasps.append(Grasp(
                fingertip_positions=fp_rotated,
                contact_normals=(obj_rot @ n.T).T.astype(np.float32),
                quality=quality,
                object_name=object_name,
                object_scale=object_size,
                joint_angles=q,
                object_pos_hand=pos_hand,
                object_quat_hand=quat_hand,
                object_pose_frame="hand_root",
            ))
            found = True
            break

        if not found:
            stats["ik_fail"] += 1

        if verbose and (attempt + 1) % 200 == 0:
            print(f"    [{attempt+1}] grasps={len(grasps)} "
                  f"nfo_fail={stats['nfo_fail']} "
                  f"ik_fail={stats['ik_fail']} "
                  f"collision={stats['collision_fail']}")

    if verbose:
        print(f"  [Algorithm 3] {len(grasps)} grasps from {attempt+1} attempts")
        print(f"    stats: {stats}")

    return GraspSet(grasps=grasps, object_name=object_name)


# ---------------------------------------------------------------------------
# Physics Validator (Step 6)
# ---------------------------------------------------------------------------

class SimGraspValidator:
    """Validates grasps via PhysX physics settle."""

    def __init__(self, env, settle_steps=40, vel_threshold=0.01,
                 min_height=0.15, max_drift=0.05, render=False):
        self.env = env
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.num_envs = env.num_envs
        self.settle_steps = settle_steps
        self.vel_threshold = vel_threshold
        self.min_height = min_height
        self.max_drift = max_drift
        self.render = render

        from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
        self.ft_ids = get_fingertip_body_ids_from_env(self.robot, env)
        self.all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long,
        )

    def validate(self, grasps: List[Grasp], verbose=True) -> List[Grasp]:
        from envs.mdp.sim_utils import set_robot_joints_direct, set_robot_root_pose

        env, robot, obj = self.env, self.robot, self.obj
        valid = []
        rejects = {"velocity": 0, "height": 0, "drift": 0}
        grasps = [g for g in grasps if g.joint_angles is not None]

        if verbose:
            print(f"\n  [Validator] {len(grasps)} grasps, "
                  f"settle={self.settle_steps}")

        for bs_start in range(0, len(grasps), self.num_envs):
            batch = grasps[bs_start:bs_start + self.num_envs]
            bs = len(batch)
            eids = self.all_env_ids[:bs]

            root = robot.data.default_root_state[eids, :7].clone()
            root[:, :3] += env.scene.env_origins[eids]
            set_robot_root_pose(env, eids, root[:, :3], root[:, 3:7])
            set_robot_joints_direct(env, eids, [g.joint_angles for g in batch])

            q_batch = torch.stack([
                torch.tensor(g.joint_angles, device=self.device, dtype=torch.float32)
                for g in batch
            ])

            # Object away → FK → re-write → FK
            tmp = obj.data.default_root_state[eids].clone()
            tmp[:, :3] = env.scene.env_origins[eids] + torch.tensor([[0, 0, -10.0]], device=self.device)
            tmp[:, 7:] = 0.0
            obj.write_root_state_to_sim(tmp, env_ids=eids)
            obj.update(0.0)
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)
            robot.write_joint_state_to_sim(q_batch, torch.zeros_like(q_batch), env_ids=eids)
            env.sim.step(render=self.render)
            env.scene.update(dt=env.physics_dt)

            # Place object at centroid
            ft_pos = robot.data.body_pos_w[eids][:, self.ft_ids, :]
            obj_pos = ft_pos.mean(dim=1)
            ostate = obj.data.default_root_state[eids].clone()
            ostate[:, :3] = obj_pos
            ostate[:, 3:7] = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(bs, -1)
            ostate[:, 7:] = 0.0
            obj.write_root_state_to_sim(ostate, env_ids=eids)
            obj.update(0.0)

            # Hold
            for _ in range(self.settle_steps):
                robot.write_joint_state_to_sim(q_batch, torch.zeros_like(q_batch), env_ids=eids)
                env.sim.step(render=self.render)
                env.scene.update(dt=env.physics_dt)

            # Check
            speed = torch.norm(obj.data.root_lin_vel_w[eids], dim=-1)
            obj_z = obj.data.root_pos_w[eids, 2]
            ft_a = robot.data.body_pos_w[eids][:, self.ft_ids, :]
            drift = torch.norm(obj.data.root_pos_w[eids, :3] - ft_a.mean(dim=1), dim=-1)

            for j in range(bs):
                if speed[j] >= self.vel_threshold:
                    rejects["velocity"] += 1
                elif obj_z[j] < self.min_height:
                    rejects["height"] += 1
                elif drift[j] > self.max_drift:
                    rejects["drift"] += 1
                else:
                    valid.append(batch[j])

            if verbose:
                print(f"    [{min(bs_start+bs, len(grasps))}/{len(grasps)}] "
                      f"passed: {len(valid)}")

        if verbose:
            print(f"  [Validator] {len(valid)}/{len(grasps)} passed. "
                  f"Rejects: {rejects}")
        return valid
