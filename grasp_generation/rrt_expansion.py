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

def nearest_neighbor(
    graph_fps: np.ndarray,  # (N, n_pts*3) flattened fingertip positions
    target_fp: np.ndarray,  # (n_pts*3,) flattened
) -> int:
    """Find nearest node in graph by fingertip L2 distance."""
    dists = np.linalg.norm(graph_fps - target_fp, axis=-1)
    return int(np.argmin(dists))


def steer(
    fp_near: np.ndarray,  # (n_pts, 3)
    fp_rand: np.ndarray,  # (n_pts, 3)
    delta: float,
    mesh: trimesh.Trimesh,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move from fp_near toward fp_rand by delta, then project to surface.

    Returns (new_points, new_normals) on the mesh surface.
    """
    direction = fp_rand - fp_near
    dist = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / (dist + 1e-8)
    step = np.minimum(delta, dist)
    fp_new = fp_near + direction * step

    # Project to mesh surface
    closest, _, face_idx = trimesh.proximity.closest_point(mesh, fp_new)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    return closest.astype(np.float32), normals


def rrt_expand(
    seed_grasps: GraspSet,
    mesh: trimesh.Trimesh,
    nfo,
    env,
    n_pts: int = 5,
    target_size: int = 300,
    delta_pos: float = 0.008,
    delta_max: float = 0.04,
    min_quality: float = 0.03,
    max_attempts_per_step: int = 30,
    collision_threshold: float = 0.002,
    object_size: float = 0.06,
    render: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> GraspGraph:
    """
    RRT-based grasp graph expansion from seed grasps.

    Algorithm:
      1. q_rand ← sampleSurface(mesh, n_pts)
      2. q_near ← nearestNeighbor(graph, q_rand)
      3. q_new ← steer(q_near, q_rand, delta_pos) → project to surface
      4. NFO quality check
      5. IK + collision check
      6. Edge distance check (< delta_max)
      7. Add node + edge to graph
    """
    from .sim_grasp_sampler import (
        sample_surface,
        grasp_analysis,
        solve_ik,
        check_collision,
    )
    from .grasp_sampler import _resolve_finger_body_ids, _compute_obj_pose_hand
    from envs.mdp.sim_utils import (
        get_fingertip_body_ids_from_env,
        set_robot_root_pose,
        set_adaptive_joint_pose,
        get_local_palm_normal,
    )
    from envs.mdp.math_utils import quat_from_two_vectors, quat_multiply
    from isaaclab.utils.math import quat_apply
    import torch

    robot = env.scene["robot"]
    device = env.device
    rng = np.random.default_rng(seed)

    ft_ids = get_fingertip_body_ids_from_env(robot, env)[:n_pts]
    finger_body_ids = _resolve_finger_body_ids(robot)

    # Compute palm-up pose
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
    ft_w = robot.data.body_pos_w[env_ids][:, ft_ids, :]
    piv = ft_w.mean(dim=1)
    palmup_quat = quat_multiply(corr, wq)
    palmup_quat = palmup_quat / (torch.norm(palmup_quat, dim=-1, keepdim=True) + 1e-8)
    palmup_pos = quat_apply(corr, wp - piv) + piv

    # Base object position
    set_robot_root_pose(env, env_ids, palmup_pos, palmup_quat)
    q_mid = robot.data.joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    robot.write_joint_state_to_sim(q_mid, torch.zeros_like(q_mid), env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)
    base_obj_pos = robot.data.body_pos_w[0, ft_ids, :].mean(dim=0).cpu().numpy()

    # Init graph with seeds
    nodes = list(seed_grasps.grasps)
    edges = []

    # Build fps array for nearest neighbor
    graph_fps = np.stack([g.fingertip_positions.flatten() for g in nodes])

    stall = 0
    max_stall = 300
    stats = {"nfo_fail": 0, "ik_fail": 0, "collision_fail": 0, "dist_fail": 0}

    if verbose:
        print(f"  [RRT] seeds={len(nodes)}, target={target_size}, "
              f"delta_pos={delta_pos}")

    while len(nodes) < target_size and stall < max_stall:
        added = False

        for _ in range(max_attempts_per_step):
            # 1. q_rand: random surface grasp
            P_rand, n_rand = sample_surface(mesh, n_pts, rng)

            # 2. q_near: nearest neighbor
            near_idx = nearest_neighbor(graph_fps, P_rand.flatten())
            fp_near = nodes[near_idx].fingertip_positions

            # 3. steer: move toward q_rand, project to surface
            P_new, n_new = steer(fp_near, P_rand, delta_pos, mesh)

            # 4. NFO quality
            quality = grasp_analysis(P_new, n_new, nfo)
            if quality < min_quality:
                stats["nfo_fail"] += 1
                continue

            # 5. IK + collision
            obj_pos = base_obj_pos.copy()
            obj_rot = np.eye(3, dtype=np.float32)

            q = solve_ik(
                env, P_new, obj_pos, obj_rot,
                palmup_pos, palmup_quat,
                object_size, ft_ids,
                ik_iterations=30, render=render,
            )
            if q is None:
                stats["ik_fail"] += 1
                continue

            if check_collision(env, finger_body_ids, obj_pos, obj_rot,
                               mesh, margin=collision_threshold):
                stats["collision_fail"] += 1
                continue

            # 6. Edge distance check
            dist = float(np.linalg.norm(P_new.flatten() - fp_near.flatten()))
            if dist > delta_max * n_pts:
                stats["dist_fail"] += 1
                continue

            # 7. Add to graph
            obj_quat_w = env.scene["object"].data.root_quat_w[0]
            obj_pos_t = torch.tensor(obj_pos, device=device, dtype=torch.float32)
            pos_hand, quat_hand = _compute_obj_pose_hand(
                robot, obj_pos_t, device, obj_quat_w=obj_quat_w,
            )

            new_grasp = Grasp(
                fingertip_positions=P_new,
                contact_normals=n_new,
                quality=quality,
                joint_angles=q,
                object_pos_hand=pos_hand,
                object_quat_hand=quat_hand,
                object_pose_frame="hand_root",
            )
            nodes.append(new_grasp)
            edges.append((near_idx, len(nodes) - 1))
            graph_fps = np.vstack([graph_fps, P_new.flatten()[np.newaxis]])
            added = True
            break

        stall = 0 if added else stall + 1

        if verbose and len(nodes) % 20 == 0:
            print(f"    [RRT] {len(nodes)}/{target_size} nodes, "
                  f"{len(edges)} edges")

    if verbose:
        print(f"  [RRT] Final: {len(nodes)} nodes, {len(edges)} edges")
        print(f"    stats: {stats}")

    return GraspGraph(
        grasp_set=GraspSet(grasps=nodes, object_name=seed_grasps.object_name),
        edges=edges,
        object_name=seed_grasps.object_name,
        num_fingers=n_pts,
    )
