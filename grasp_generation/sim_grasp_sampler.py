"""
DexterityGen Algorithm 3 — Grasp Set Generation
=================================================
Paper reference (Appendix, Algorithm 3 & 4):
  1. P, n ← sampleSurface(M, N_pts) [Heuristic: Opposing faces]
  2. quality ← graspAnalysis(P, n)  [Algorithm 4: NFO via NNLS]
  3. pose ← randomPose(hand_frame)
  4. q ← assign(IK)
  5. collision check → reject if penetrating
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh
from scipy.optimize import nnls

from .grasp_sampler import Grasp, GraspSet, _resolve_finger_body_ids, _compute_obj_pose_hand


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
    """
    휴리스틱 샘플링: 무작위 샘플링의 낮은 NFO 통과율을 보완하기 위해 
    첫 번째 점을 뽑고, 나머지 점들은 그와 마주보는 면(Opposing faces)에서 샘플링합니다.
    """
    # 1. 첫 번째 점 무작위 샘플링
    pt1, face_idx1 = trimesh.sample.sample_surface(mesh, 1)
    n1 = mesh.face_normals[face_idx1[0]]
    
    if n_pts == 1:
        return pt1.astype(np.float32), n1.reshape(1, 3).astype(np.float32)

    # 2. 마주보는 방향의 면(Face) 필터링 (내적이 -0.5 이하)
    dots = np.dot(mesh.face_normals, n1)
    opposing_face_indices = np.where(dots < -0.5)[0]
    
    if len(opposing_face_indices) == 0:
        opposing_face_indices = np.arange(len(mesh.faces))
        
    # 3. 면적 기반 확률 분포 계산 (Numpy Float 정밀도 에러 수정)
    face_areas = mesh.area_faces[opposing_face_indices]
    area_sum = face_areas.sum()
    
    if area_sum <= 0.0:
        face_probs = np.ones(len(opposing_face_indices)) / len(opposing_face_indices)
    else:
        face_probs = face_areas / area_sum
        
    # 부동소수점 오차로 인해 합이 1.0에서 미세하게 벗어나는 것을 방지하는 강제 정규화
    face_probs = face_probs / face_probs.sum()
    
    # 4. 나머지 점들 샘플링
    chosen_faces = rng.choice(opposing_face_indices, size=n_pts - 1, p=face_probs)
    
    r1 = rng.random(n_pts - 1)
    r2 = rng.random(n_pts - 1)
    r1_sqrt = np.sqrt(r1)
    
    u = 1.0 - r1_sqrt
    v = r1_sqrt * (1.0 - r2)
    w = r1_sqrt * r2
    
    vertices = mesh.vertices[mesh.faces[chosen_faces]]
    pts_rest = u[:, None] * vertices[:, 0] + v[:, None] * vertices[:, 1] + w[:, None] * vertices[:, 2]
    n_rest = mesh.face_normals[chosen_faces]
    
    # 5. 병합
    points = np.vstack([pt1, pts_rest]).astype(np.float32)
    normals = np.vstack([n1.reshape(1, 3), n_rest]).astype(np.float32)
    
    return points, normals



# ---------------------------------------------------------------------------
# Step 2: graspAnalysis(P, n) — Algorithm 4 (NFO)
# ---------------------------------------------------------------------------

def grasp_analysis(P: np.ndarray, n: np.ndarray, mu: float = 0.5, num_edges: int = 8) -> float:
    """
    논문의 Net Force Optimization (Algorithm 4) 수식을 NNLS로 엄밀하게 풉니다.
    마찰 원추(Friction Cone)를 고려한 잔차의 제곱(residual squared)을 반환합니다.
    값이 0.0에 가까울수록 안정적인 파지(Force Closure)입니다.
    """
    N = len(P)
    V = []
    
    for i in range(N):
        normal = n[i] / (np.linalg.norm(n[i]) + 1e-8)
        if abs(normal[0]) > 0.9:
            t1 = np.array([0.0, 1.0, 0.0])
        else:
            t1 = np.array([1.0, 0.0, 0.0])
            
        t1 -= np.dot(t1, normal) * normal
        t1 /= (np.linalg.norm(t1) + 1e-8)
        t2 = np.cross(normal, t1)
        
        for k in range(num_edges):
            theta = 2.0 * np.pi * k / num_edges
            edge = normal + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
            V.append(edge / (np.linalg.norm(edge) + 1e-8))
            
    V = np.column_stack(V)
    num_total_edges = V.shape[1]
    
    min_residual_sq = float('inf')
    
    # Subproblem 분할 (f_j = 1 고정)
    for j in range(num_total_edges):
        b = -V[:, j]
        A = np.delete(V, j, axis=1)
        
        # NNLS: min ||A*x - b||^2 s.t. x >= 0
        _, residual = nnls(A, b)
        
        res_sq = residual ** 2
        if res_sq < min_residual_sq:
            min_residual_sq = res_sq
            
    return float(min_residual_sq)


# ---------------------------------------------------------------------------
# Step 3: randomPose — sample object pose in hand frame
# ---------------------------------------------------------------------------

def sample_random_pose(
    base_pos: np.ndarray,
    rng: np.random.Generator,
    pos_std: float = 0.02,
    rot_std: float = 0.3,
):
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
    contact_points_obj: np.ndarray,  # (n_pts, 3) Object Local Frame
    obj_pos: np.ndarray,             
    obj_rot: np.ndarray,             
    palmup_pos, palmup_quat,
    object_size: float,
    ft_ids_all: list,                # [수정] 2~4개가 아닌 전체 5개 ID를 받음
    ik_iterations: int = 30,
    ik_threshold: float = 0.03,
    render: bool = False,
) -> Optional[np.ndarray]:
    from envs.mdp.sim_utils import (
        set_robot_root_pose,
        set_adaptive_joint_pose,
        refine_hand_to_start_grasp,
    )
    from envs.mdp.math_utils import world_to_local_points

    robot = env.scene["robot"]
    obj_actor = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    # 1. 로봇 셋업
    set_robot_root_pose(env, env_ids, palmup_pos, palmup_quat)
    set_adaptive_joint_pose(env, env_ids, object_size)

    q_mid = robot.data.joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)

    # 2. 물체 셋업
    from scipy.spatial.transform import Rotation as R_scipy
    obj_quat_scipy = R_scipy.from_matrix(obj_rot).as_quat()
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

    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)

    # 3. [핵심 버그 픽스] Target 텐서 구축 (Object Frame 유지 및 Padding)
    n_active = contact_points_obj.shape[0]
    
    # 잉여 손가락이 움직이지 않도록 현재 World 좌표를 구함
    current_ft_world = robot.data.body_pos_w[env_ids][:, ft_ids_all, :].clone()
    # refine_hand_to_start_grasp가 기대하는 Object Frame으로 변환
    current_ft_obj = world_to_local_points(current_ft_world, obj_pos_t, obj_quat_t)
    
    # 사용할 2~4개 손가락만 NFO에서 찾은 점으로 덮어씀 (Double-transform 방지)
    fp_tensor_obj = current_ft_obj.clone()
    fp_tensor_obj[0, :n_active, :] = torch.tensor(
        contact_points_obj, device=device, dtype=torch.float32
    )

    # 4. IK 풀이
    ik_cfg = getattr(env.cfg, "reset_refinement", None) or {}
    if not isinstance(ik_cfg, dict):
        ik_cfg = {}
    ik_cfg["iterations"] = ik_iterations
    env.cfg.reset_refinement = ik_cfg

    refine_hand_to_start_grasp(env, env_ids, fp_tensor_obj)
    robot.update(0.0)

    # 5. 수렴 확인 (실제 파지에 사용하는 손가락만 검사)
    target_world_active = (obj_rot @ contact_points_obj.T).T + obj_pos
    target_world_active_t = torch.tensor(
        target_world_active, device=device, dtype=torch.float32
    ).unsqueeze(0)
    
    ft_after = robot.data.body_pos_w[0, ft_ids_all[:n_active], :]
    ft_err = torch.norm(ft_after - target_world_active_t[0], dim=-1)
    
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
    robot = env.scene["robot"]
    fl_world = robot.data.body_pos_w[0, finger_body_ids, :].cpu().numpy()

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
    num_grasps: int = 300,
    max_candidates: int = 100000,
    nfo_min_quality: float = 0.03,
    collision_margin: float = 0.001,
    ik_iterations: int = 30,
    render: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> GraspSet:
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

    ft_ids_all = get_fingertip_body_ids_from_env(robot, env)
    finger_body_ids = _resolve_finger_body_ids(robot)

    if verbose:
        print(f"    [Setup] 시뮬레이터 초기화 및 Palm-up 자세 계산 시작...")

    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    root_state = robot.data.default_root_state[env_ids, :7].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])
    set_adaptive_joint_pose(env, env_ids, object_size)

    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)

    wrist_pos = robot.data.root_pos_w[env_ids].clone()
    wrist_quat = robot.data.root_quat_w[env_ids].clone()
    palm_n = get_local_palm_normal(robot, env).unsqueeze(0)
    palm_w = quat_apply(wrist_quat, palm_n)
    correction = quat_from_two_vectors(palm_w, torch.tensor([[0, 0, 1.0]], device=device))
    
    ft_w = robot.data.body_pos_w[env_ids][:, ft_ids_all[:4], :] 
    pivot = ft_w.mean(dim=1)
    palmup_quat = quat_multiply(correction, wrist_quat)
    palmup_quat = palmup_quat / (torch.norm(palmup_quat, dim=-1, keepdim=True) + 1e-8)
    palmup_pos = quat_apply(correction, wrist_pos - pivot) + pivot

    set_robot_root_pose(env, env_ids, palmup_pos, palmup_quat)
    q_mid = robot.data.joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    
    base_obj_pos = robot.data.body_pos_w[0, ft_ids_all[:4], :].mean(dim=0).cpu().numpy()

    grasps = []
    stats = {"nfo_fail": 0, "ik_fail": 0, "collision_fail": 0}

    if verbose:
        print(f"    [Setup] Initialization. Enter the loop (max_candidates: {max_candidates})")

    for attempt in range(max_candidates):
        if len(grasps) >= num_grasps:
            break

        current_n_pts = int(rng.choice([2, 3, 4]))

        P, n = sample_surface(mesh, current_n_pts, rng)
        
        # 내부 NFO 함수 사용
        quality = grasp_analysis(P, n, mu=0.5, num_edges=8)
        
        # 부등호 수정: 잔차가 기준치보다 크면 기각
        if quality > nfo_min_quality:
            stats["nfo_fail"] += 1
        else:
            obj_pos, obj_rot = sample_random_pose(base_obj_pos, rng)

            q = solve_ik(
                env, P, obj_pos, obj_rot,
                palmup_pos, palmup_quat,
                object_size, ft_ids_all,
                ik_iterations=ik_iterations,
                render=render,
            )
            
            if q is None:
                stats["ik_fail"] += 1
            else:
                if check_collision(env, finger_body_ids, obj_pos, obj_rot, mesh, margin=collision_margin):
                    stats["collision_fail"] += 1
                else:
                    obj_quat_w = env.scene["object"].data.root_quat_w[0]
                    obj_pos_t = torch.tensor(obj_pos, device=device, dtype=torch.float32)
                    pos_hand, quat_hand = _compute_obj_pose_hand(robot, obj_pos_t, device, obj_quat_w=obj_quat_w)

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

        if verbose and (attempt < 10 or (attempt + 1) % 10 == 0):
            progress_attempt = (attempt + 1) / max_candidates * 100
            progress_target = len(grasps) / num_grasps * 100 if num_grasps > 0 else 0
            print(f"    [Search: {attempt+1}/{max_candidates} ({progress_attempt:.2f}%)] "
                  f"Reaching Goal: {len(grasps)}/{num_grasps} ({progress_target:.1f}%) | "
                  f"Failure -> NFO: {stats['nfo_fail']}, IK: {stats['ik_fail']}, Collision: {stats['collision_fail']}")

    if verbose:
        print(f"    [Phase 1 Terminates] Total number of searches: {attempt+1}, Number of final grasping sets: {len(grasps)}")

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

            ft_pos = robot.data.body_pos_w[eids][:, self.ft_ids, :]
            obj_pos = ft_pos.mean(dim=1)
            ostate = obj.data.default_root_state[eids].clone()
            ostate[:, :3] = obj_pos
            ostate[:, 3:7] = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(bs, -1)
            ostate[:, 7:] = 0.0
            obj.write_root_state_to_sim(ostate, env_ids=eids)
            obj.update(0.0)

            for _ in range(self.settle_steps):
                robot.write_joint_state_to_sim(q_batch, torch.zeros_like(q_batch), env_ids=eids)
                env.sim.step(render=self.render)
                env.scene.update(dt=env.physics_dt)

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
                if len(grasps) > 0:
                    validation_progress = min(bs_start+bs, len(grasps)) / len(grasps) * 100
                    print(f"    Validation progress: {validation_progress:.1f}%")

        if verbose:
            print(f"  [Validator] {len(valid)}/{len(grasps)} passed. "
                  f"Rejects: {rejects}")
        return valid
