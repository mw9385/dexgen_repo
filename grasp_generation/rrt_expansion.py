"""
Grasp Graph data structures + RRT Expand algorithm.

GraspGraph: per-object graph of grasp configurations.
MultiObjectGraspGraph: collection of per-object graphs for object pool training.
rrt_expand: RRT-based graph expansion from seed grasps.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .grasp_sampler import Grasp, GraspSet
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R_scipy, Slerp
from typing import Tuple, Optional

from .grasp_sampler import Grasp, GraspSet, _resolve_finger_body_ids, _compute_obj_pose_hand
from .sim_grasp_sampler import grasp_analysis, solve_ik, check_collision


# ---------------------------------------------------------------------------
# GraspGraph (single object)
# ---------------------------------------------------------------------------

@dataclass
class GraspGraph:
    """
    Connected graph of grasps for a single object.

    Nodes: grasp configurations (GraspSet)
    Edges: (i, j) pairs reachable by continuous motion
    """
    grasp_set: GraspSet
    edges: List[Tuple[int, int]] = field(default_factory=list)
    object_name: str = ""
    num_fingers: int = 4

    def __len__(self):
        return len(self.grasp_set)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_neighbors(self, node_idx: int) -> List[int]:
        neighbors = []
        for i, j in self.edges:
            if i == node_idx:
                neighbors.append(j)
            elif j == node_idx:
                neighbors.append(i)
        return neighbors

    def sample_edge(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, len(self.edges))
        return self.edges[idx]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[GraspGraph] Saved {len(self)} nodes, {self.num_edges} edges → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GraspGraph":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        print(f"[GraspGraph] Loaded {len(obj)} nodes, {obj.num_edges} edges ← {path}")
        return obj


# ---------------------------------------------------------------------------
# MultiObjectGraspGraph (object pool)
# ---------------------------------------------------------------------------

@dataclass
class MultiObjectGraspGraph:
    """
    Collection of per-object GraspGraphs for object pool training.

    At each RL episode reset:
      1. Sample a random object name from self.graphs
      2. Sample a random edge from graphs[object_name]
      → (start_grasp, goal_grasp) + which object to spawn
    """
    graphs: Dict[str, GraspGraph] = field(default_factory=dict)
    object_specs: Dict[str, dict] = field(default_factory=dict)

    def __len__(self):
        return sum(len(g) for g in self.graphs.values())

    @property
    def num_objects(self) -> int:
        return len(self.graphs)

    @property
    def object_names(self) -> List[str]:
        return list(self.graphs.keys())

    @property
    def num_fingers(self) -> int:
        if not self.graphs:
            return 4
        return next(iter(self.graphs.values())).num_fingers

    def add(self, graph: GraspGraph, spec: dict):
        self.graphs[graph.object_name] = graph
        self.object_specs[graph.object_name] = spec

    def sample_object(self, rng: Optional[np.random.Generator] = None) -> str:
        if rng is None:
            rng = np.random.default_rng()
        names = self.object_names
        return names[rng.integers(0, len(names))]

    def sample_edge(
        self,
        object_name: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[str, Tuple[int, int]]:
        if rng is None:
            rng = np.random.default_rng()
        if object_name is None:
            object_name = self.sample_object(rng)
        edge = self.graphs[object_name].sample_edge(rng)
        return object_name, edge

    def get_grasp(self, object_name: str, grasp_idx: int) -> Grasp:
        return self.graphs[object_name].grasp_set[grasp_idx]

    def summary(self):
        print(f"[MultiObjectGraspGraph] {self.num_objects} objects:")
        for name, g in self.graphs.items():
            print(f"  {name}: {len(g)} nodes, {g.num_edges} edges")

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        total = sum(len(g) for g in self.graphs.values())
        print(f"[MultiObjectGraspGraph] Saved {self.num_objects} objects, "
              f"{total} total grasps → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MultiObjectGraspGraph":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        print(f"[MultiObjectGraspGraph] Loaded {obj.num_objects} objects ← {path}")
        return obj


# ---------------------------------------------------------------------------
# RRT Expand Algorithm
# ---------------------------------------------------------------------------

def random_sample_qp(
    nodes: list[Grasp],
    rng: np.random.Generator,
    q_noise_std: float = 0.5,
    p_pos_noise_std: float = 0.05,
    p_rot_noise_std: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [span_3](start_span)Algorithm 5, Line 2: (q, p) <- RandomSample()[span_3](end_span)
    기존 탐색된 노드 중 하나를 무작위로 선택한 후 Perturbation을 가하여
    새로운 Target Configuration (q, p)를 샘플링합니다.
    """
    rand_idx = rng.integers(0, len(nodes))
    g_base = nodes[rand_idx]
    
    # 1. q (finger configuration) perturbation
    q_rand = g_base.joint_angles + rng.normal(0, q_noise_std, size=g_base.joint_angles.shape)
    
    # 2. p (object pose) perturbation
    p_pos_rand = g_base.object_pos_hand + rng.normal(0, p_pos_noise_std, size=3).astype(np.float32)
    
    base_rot = R_scipy.from_quat(g_base.object_quat_hand[[1, 2, 3, 0]])
    angle = rng.normal(0, p_rot_noise_std)
    axis = rng.normal(0, 1, size=3)
    axis /= (np.linalg.norm(axis) + 1e-8)
    rot_noise = R_scipy.from_rotvec(angle * axis)
    
    p_rot_rand = (rot_noise * base_rot).as_matrix().astype(np.float32)
    
    return q_rand, p_pos_rand, p_rot_rand

def interpolate_qp(
    q1: np.ndarray, p_pos1: np.ndarray, p_rot1: np.ndarray, 
    q2: np.ndarray, p_pos2: np.ndarray, p_rot2: np.ndarray, 
    step_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [span_4](start_span)Algorithm 5, Line 4: (q, p) <- Interpolate((q, p), (q*, p*))[span_4](end_span)
    (q1, p1)에서 (q2, p2) 방향으로 step_size 만큼 보간합니다.
    """
    # Joint (C-space) 및 Position 보간 (선형 보간)
    q_new = q1 + step_size * (q2 - q1)
    p_pos_new = p_pos1 + step_size * (p_pos2 - p_pos1)
    
    # Rotation 보간 (구면 선형 보간, Slerp)
    rots = R_scipy.from_matrix([p_rot1, p_rot2])
    slerp = Slerp([0, 1], rots)
    p_rot_new = slerp([step_size])[0].as_matrix().astype(np.float32)
    
    return q_new, p_pos_new, p_rot_new


def fix_contact_and_collision(
    env, q_interp: np.ndarray, p_pos: np.ndarray, p_rot: np.ndarray, 
    mesh: trimesh.Trimesh, nfo, nfo_min_quality: float,
    palmup_pos: np.ndarray, palmup_quat: torch.Tensor, 
    object_size: float, ft_ids: list, finger_body_ids: list, 
    collision_margin: float, render: bool
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    [span_5](start_span)Algorithm 5, Line 5: (q, p) <- FixContactAndCollision(q, p, M)[span_5](end_span)
    보간된 상태가 물리적 정합성을 갖도록 메쉬 표면 투영 및 IK, 충돌 검사를 통해 보정합니다.
    """
    robot = env.scene["robot"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    # 1. 보간된 q로 로봇 FK 수행하여 Fingertip의 World 좌표 확보
    q_tensor = torch.tensor(q_interp, device=device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_state_to_sim(q_tensor, torch.zeros_like(q_tensor), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    
    fl_world = robot.data.body_pos_w[0, ft_ids, :].cpu().numpy()

    # 2. 메쉬 표면으로 투영 (Project to surface)
    rot_inv = p_rot.T
    fl_obj = (rot_inv @ (fl_world - p_pos).T).T
    closest, _, face_idx = trimesh.proximity.closest_point(mesh, fl_obj)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    
    # 3. NFO 품질 평가
    quality = grasp_analysis(closest, normals, nfo)
    if quality < nfo_min_quality:
        return None, None, None, quality
    
    # 4. 투영된 점(표면)을 목표로 IK를 풀어 관절 각도(q) 보정
    q_fixed = solve_ik(
        env, closest, p_pos, p_rot,
        palmup_pos, palmup_quat, object_size, ft_ids,
        ik_iterations=15, render=render  # 보간된 값이므로 적은 iteration으로도 수렴 가능
    )
    if q_fixed is None:
        return None, None, None, quality
        
    # 5. 충돌 검사 (Collision check)
    if check_collision(env, finger_body_ids, p_pos, p_rot, mesh, margin=collision_margin):
        return None, None, None, quality
        
    return q_fixed, closest, normals, float(quality)


def rrt_expand(
    seed_grasps: GraspSet,
    mesh: trimesh.Trimesh,
    nfo,
    env,
    target_size: int = 300,
    delta_step: float = 0.1,  
    min_quality: float = 0.03,
    max_attempts_per_step: int = 30,
    collision_threshold: float = 0.002,
    object_size: float = 0.06,
    render: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> "GraspGraph":
    from envs.mdp.sim_utils import (
        get_fingertip_body_ids_from_env,
        set_robot_root_pose,
        set_adaptive_joint_pose,
        get_local_palm_normal
    )
    from envs.mdp.math_utils import quat_from_two_vectors, quat_multiply
    from isaaclab.utils.math import quat_apply
    import torch

    robot = env.scene["robot"]
    device = env.device
    rng = np.random.default_rng(seed)
    
    # 전체 핑거팁 ID 확보 (동적 할당을 위해 전부 가져옴)
    ft_ids_all = get_fingertip_body_ids_from_env(robot, env)
    finger_body_ids = _resolve_finger_body_ids(robot)

    # ── Compute palm-up wrist pose ──
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    root_state = robot.data.default_root_state[env_ids, :7].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])
    set_adaptive_joint_pose(env, env_ids, object_size)
    
    tmp = env.scene["object"].data.default_root_state[env_ids].clone()
    tmp[:, :3] = env.scene.env_origins[env_ids] + torch.tensor([[0, 0, -10.0]], device=device)
    tmp[:, 7:] = 0.0
    env.scene["object"].write_root_state_to_sim(tmp, env_ids=env_ids)
    env.scene["object"].update(0.0)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)

    wp = robot.data.root_pos_w[env_ids].clone()
    wq = robot.data.root_quat_w[env_ids].clone()
    pn = get_local_palm_normal(robot, env).unsqueeze(0)
    pw = quat_apply(wq, pn)
    corr = quat_from_two_vectors(pw, torch.tensor([[0, 0, 1.0]], device=device))
    
    # Pivot 계산 시 최대 길이(4개) 기준 임시 적용
    ft_w = robot.data.body_pos_w[env_ids][:, ft_ids_all[:4], :]
    piv = ft_w.mean(dim=1)
    palmup_quat = quat_multiply(corr, wq)
    palmup_quat = palmup_quat / (torch.norm(palmup_quat, dim=-1, keepdim=True) + 1e-8)
    palmup_pos = quat_apply(corr, wp - piv) + piv

    nodes = list(seed_grasps.grasps)
    edges = []
    stats = {"nfo_fail": 0, "ik_fail": 0, "collision_fail": 0}

    # 탐색용 C-space 배열 사전 스택 (반복문 외곽으로 이동)
    graph_q = np.stack([g.joint_angles for g in nodes])

    stall = 0
    max_stall = 300

    if verbose:
        print(f"  [RRT Expand] seeds={len(nodes)}, target={target_size}")

    while len(nodes) < target_size and stall < max_stall:
        added = False
        
        for _ in range(max_attempts_per_step):
            # Algorithm 5, Line 2: RandomSample
            q_rand, p_pos_rand, p_rot_rand = random_sample_qp(nodes, rng)

            # Algorithm 5, Line 3: NearestNeighbor (미리 계산된 graph_q 사용)
            dists = np.linalg.norm(graph_q - q_rand, axis=-1)
            near_idx = int(np.argmin(dists))
            
            g_near = nodes[near_idx]
            q_near = g_near.joint_angles
            p_pos_near = g_near.object_pos_hand
            p_rot_near = R_scipy.from_quat(g_near.object_quat_hand[[1, 2, 3, 0]]).as_matrix().astype(np.float32)

            # [핵심 수정] g_near의 핑거팁 개수 파악 후 동적 할당
            current_n_pts = len(g_near.fingertip_positions)
            current_ft_ids = ft_ids_all[:current_n_pts]

            # Algorithm 5, Line 4: Interpolate
            q_new, p_pos_new, p_rot_new = interpolate_qp(
                q_near, p_pos_near, p_rot_near,
                q_rand, p_pos_rand, p_rot_rand, 
                step_size=delta_step
            )

            # Algorithm 5, Line 5: FixContactAndCollision (동적 current_ft_ids 전달)
            q_fixed, fp_fixed, normals_fixed, quality = fix_contact_and_collision(
                env, q_new, p_pos_new, p_rot_new, mesh, nfo, min_quality,
                palmup_pos, palmup_quat, object_size, current_ft_ids, finger_body_ids, 
                collision_threshold, render
            )

            if q_fixed is None:
                stats["collision_fail"] += 1 
                continue

            # Algorithm 5, Line 6: S = S U {(q, p)}
            obj_quat_w = env.scene["object"].data.root_quat_w[0]
            obj_pos_t = torch.tensor(p_pos_new, device=device, dtype=torch.float32)
            pos_hand, quat_hand = _compute_obj_pose_hand(robot, obj_pos_t, device, obj_quat_w=obj_quat_w)

            new_grasp = Grasp(
                fingertip_positions=fp_fixed,
                contact_normals=normals_fixed,
                quality=quality,
                joint_angles=q_fixed,
                object_pos_hand=pos_hand,
                object_quat_hand=quat_hand,
                object_pose_frame="hand_root",
            )
            
            nodes.append(new_grasp)
            edges.append((near_idx, len(nodes) - 1))
            
            # 새 노드 배열 업데이트
            graph_q = np.vstack([graph_q, q_fixed[np.newaxis]])
            added = True
            break

        stall = 0 if added else stall + 1

        if verbose and len(nodes) % 10 == 0:
            print(f"    [RRT] {len(nodes)}/{target_size} nodes, {len(edges)} edges")

    if verbose:
        print(f"  [RRT] Final: {len(nodes)} nodes, {len(edges)} edges | stats: {stats}")

    from .rrt_expansion import GraspGraph
    return GraspGraph(
        grasp_set=GraspSet(grasps=nodes, object_name=seed_grasps.object_name),
        edges=edges,
        object_name=seed_grasps.object_name,
        num_fingers=4, # 가변적이므로 의미 없는 속성이 될 수 있으나, 호환성을 위해 유지
    )
