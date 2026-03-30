from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from .grasp_sampler import GraspSet
from .rrt_expansion import build_graph_from_grasps


def refine_multi_object_graph_with_isaac(
    graph,
    batch_envs: int = 16,
    keep_top_k: Optional[int] = None,
):
    from isaaclab.envs import ManagerBasedRLEnv

    from envs import AnyGraspEnvCfg, register_anygrasp_env
    from envs.mdp import events as mdp_events

    register_anygrasp_env()

    for object_name, subgraph in list(graph.graphs.items()):
        spec = dict(graph.object_specs.get(object_name, {}))
        refined_graph = _refine_single_object_graph(
            subgraph=subgraph,
            object_spec=spec,
            batch_envs=batch_envs,
            env_factory=ManagerBasedRLEnv,
            env_cfg_factory=AnyGraspEnvCfg,
            mdp_events=mdp_events,
            keep_top_k=keep_top_k,
        )
        refined_graph.object_name = object_name
        graph.graphs[object_name] = refined_graph
        graph.object_specs[object_name] = spec
    return graph


def _refine_single_object_graph(
    subgraph,
    object_spec: dict,
    batch_envs: int,
    env_factory,
    env_cfg_factory,
    mdp_events,
    keep_top_k: Optional[int] = None,
):
    num_fingers = int(getattr(subgraph, "num_fingers", 4))
    env_cfg = env_cfg_factory()
    env_cfg.scene.num_envs = min(int(batch_envs), max(1, len(subgraph.grasp_set.grasps)))
    env_cfg.object_pool_specs = [object_spec]
    env_cfg.reset_randomization = {
        "object_pos_jitter_std": 0.0,
        "object_rot_jitter_deg": 0.0,
        "wrist_pos_jitter_std": 0.0,
        "wrist_rot_std_deg": 0.0,
        "align_palm_up": False,
    }
    env_cfg.hand = dict(getattr(env_cfg, "hand", {}) or {})
    env_cfg.hand["num_fingers"] = num_fingers
    env_cfg.hand["fingertip_links"] = {
        2: ["robot0_ffdistal", "robot0_thdistal"],
        3: ["robot0_ffdistal", "robot0_mfdistal", "robot0_thdistal"],
        4: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_thdistal"],
        5: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"],
    }.get(num_fingers, ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"][:num_fingers])
    env_cfg.viewer = None

    env = env_factory(env_cfg)
    try:
        device = env.device
        all_grasps = subgraph.grasp_set.grasps
        for chunk_start in range(0, len(all_grasps), env_cfg.scene.num_envs):
            chunk = all_grasps[chunk_start : chunk_start + env_cfg.scene.num_envs]
            env_ids = torch.arange(len(chunk), device=device, dtype=torch.long)
            _refine_chunk(env, env_ids, chunk, mdp_events)

        if keep_top_k is not None and len(all_grasps) > keep_top_k:
            all_grasps = sorted(
                all_grasps,
                key=lambda grasp: (
                    float("inf") if grasp.reset_contact_error is None else float(grasp.reset_contact_error)
                ),
            )[: int(keep_top_k)]

        # Use the same 0.15 multiplier as run_grasp_generation.py so that graph
        # edges after refinement are consistent with the pre-refinement graph.
        refined = build_graph_from_grasps(
            all_grasps,
            object_name=subgraph.object_name,
            delta_max=float(object_spec.get("size", 0.06)) * 0.15,
            num_fingers=num_fingers,
        )
        refined.grasp_set = GraspSet(grasps=all_grasps, object_name=subgraph.object_name)
        return refined
    finally:
        env.close()


def _refine_chunk(env, env_ids: torch.Tensor, grasps: list, mdp_events):
    robot = env.scene["robot"]
    obj = env.scene["object"]

    fingertip_targets = []
    joint_list = []
    for grasp in grasps:
        fingertip_targets.append(grasp.fingertip_positions.copy())
        joint_list.append(getattr(grasp, "joint_angles", None))

    start_fps = torch.tensor(np.stack(fingertip_targets), device=env.device, dtype=torch.float32)

    mdp_events._reset_to_default_pose(env, env_ids)
    mdp_events._randomise_object_pose(env, env_ids)
    mdp_events._randomise_wrist_pose(env, env_ids)

    if any(j is not None for j in joint_list):
        mdp_events._set_robot_joints_direct(env, env_ids, joint_list)
    else:
        mdp_events._set_robot_to_fingertip_config(env, env_ids, start_fps)

    robot.update(0.0)
    mdp_events._place_object_in_hand(env, env_ids, start_fps)
    mdp_events._refine_hand_to_start_grasp(env, env_ids, start_fps)
    robot.update(0.0)
    mdp_events._place_object_in_hand(env, env_ids, start_fps)
    final_mean, final_max = mdp_events._measure_grasp_contact_error(env, env_ids, start_fps)
    robot.update(0.0)
    obj.update(0.0)

    root_pos = robot.data.root_pos_w[env_ids].clone()
    root_quat = robot.data.root_quat_w[env_ids].clone()
    obj_pos = obj.data.root_pos_w[env_ids].clone()
    obj_quat = obj.data.root_quat_w[env_ids].clone()
    rel_pos = torch.zeros(len(env_ids), 3, device=env.device)
    rel_quat = torch.zeros(len(env_ids), 4, device=env.device)
    rel_quat[:, 0] = 1.0

    for i in range(len(env_ids)):
        rel = obj_pos[i] - root_pos[i]
        rel_pos[i] = mdp_events.quat_apply_inverse(root_quat[i].unsqueeze(0), rel.unsqueeze(0))[0]
        rel_quat[i] = mdp_events._quat_multiply(
            mdp_events._quat_conjugate(root_quat[i].unsqueeze(0)),
            obj_quat[i].unsqueeze(0),
        )[0]

    current_q = robot.data.joint_pos[env_ids].detach().cpu().numpy()

    for i, grasp in enumerate(grasps):
        grasp.joint_angles = current_q[i].astype(np.float32)
        grasp.object_pos_hand = rel_pos[i].detach().cpu().numpy().astype(np.float32)
        grasp.object_quat_hand = rel_quat[i].detach().cpu().numpy().astype(np.float32)
        grasp.object_pose_frame = "root"
        grasp.reset_contact_error = float(final_mean[i].item())
        grasp.reset_contact_error_max = float(final_max[i].item())
