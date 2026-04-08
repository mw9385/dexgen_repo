"""
Visualize Phase 2 IK results BEFORE physics validation.

Loads IK-solved grasps one by one, sets joints + places object,
and renders without physics so you can visually inspect the grasp
configuration in the Isaac Sim GUI.

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/visualize_ik_grasps.py \
        --shapes cube --size_min 0.05 --size_max 0.05 \
        --num_surface_grasps 200 --interval 3
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Visualize IK-solved grasps")
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--size_min", type=float, default=0.05)
    p.add_argument("--size_max", type=float, default=0.05)
    p.add_argument("--num_sizes", type=int, default=1)
    p.add_argument("--num_surface_grasps", type=int, default=200)
    p.add_argument("--nfo_min_quality", type=float, default=0.03)
    p.add_argument("--interval", type=float, default=3.0,
                   help="Seconds to hold each grasp for inspection")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    # Isaac Sim (NOT headless — we need the GUI)
    p.add_argument("--headless", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    import trimesh
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply

    # Reuse env building from the generation script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_sim_grasp_generation import _build_env_cfg

    from grasp_generation.sim_grasp_sampler import (
        SurfaceGraspSampler,
        assign_ik_and_check_collision,
        make_primitive_mesh,
    )
    from grasp_generation.net_force_optimization import NetForceOptimizer
    from envs.mdp.sim_utils import (
        set_robot_root_pose,
        set_robot_joints_direct,
        get_fingertip_body_ids_from_env,
        get_local_palm_normal,
    )
    from envs.mdp.math_utils import quat_from_two_vectors, quat_multiply

    shape = args.shapes[0]
    size = float(args.size_min)

    # Create env
    env_cfg = _build_env_cfg(shape, size, args.num_envs)
    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)

    env.sim.step(render=True)
    env.scene.update(dt=env.physics_dt)

    mesh = make_primitive_mesh(shape, size)
    nfo = NetForceOptimizer(mu=0.5, num_edges=8, min_quality=args.nfo_min_quality)

    # ── Phase 1: Surface sampling ────────────────────────────────
    print(f"\n  Phase 1: Surface sampling...")
    sampler = SurfaceGraspSampler(
        mesh=mesh, nfo=nfo, num_fingers=5,
        min_quality=args.nfo_min_quality,
        seed=args.seed,
    )
    surface_grasps = sampler.sample(
        num_candidates=50000,
        num_grasps=args.num_surface_grasps,
    )
    print(f"  → {len(surface_grasps)} surface grasps")

    # ── Phase 2: IK ──────────────────────────────────────────────
    print(f"\n  Phase 2: IK...")
    ik_grasps = assign_ik_and_check_collision(
        env=env, grasps=surface_grasps, mesh=mesh,
        object_size=size, render=True, verbose=True,
    )
    print(f"  → {len(ik_grasps)} IK-solved grasps")

    if len(ik_grasps) == 0:
        print("  No grasps to visualize!")
        env.close()
        sim_app.close()
        return

    # ── Compute palm-up pose ─────────────────────────────────────
    root_state = robot.data.default_root_state[env_ids, :7].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])

    q_mid = (robot.data.soft_joint_pos_limits[0, :, 0]
             + robot.data.soft_joint_pos_limits[0, :, 1]) / 2.0
    q_mid[:2] = 0.0
    q_mid_2d = q_mid.unsqueeze(0)
    robot.write_joint_state_to_sim(q_mid_2d, torch.zeros_like(q_mid_2d), env_ids=env_ids)

    temp = obj.data.default_root_state[env_ids].clone()
    temp[:, :3] = env.scene.env_origins[env_ids] + torch.tensor([[0, 0, -10.0]], device=device)
    temp[:, 7:] = 0.0
    obj.write_root_state_to_sim(temp, env_ids=env_ids)
    obj.update(0.0)

    env.sim.step(render=True)
    env.scene.update(dt=env.physics_dt)

    wrist_pos = robot.data.root_pos_w[env_ids].clone()
    wrist_quat = robot.data.root_quat_w[env_ids].clone()
    palm_n = get_local_palm_normal(robot, env).unsqueeze(0)
    palm_w = quat_apply(wrist_quat, palm_n)
    correction = quat_from_two_vectors(
        palm_w, torch.tensor([[0, 0, 1.0]], device=device),
    )
    ft_w = robot.data.body_pos_w[env_ids][:, ft_ids, :]
    pivot = ft_w.mean(dim=1)
    palmup_quat = quat_multiply(correction, wrist_quat)
    palmup_quat = palmup_quat / (torch.norm(palmup_quat, dim=-1, keepdim=True) + 1e-8)
    palmup_pos = quat_apply(correction, wrist_pos - pivot) + pivot

    # ── Visualize each grasp ─────────────────────────────────────
    print(f"\n  Visualizing {len(ik_grasps)} grasps "
          f"({args.interval}s each). Ctrl+C to stop.\n")

    grasp_idx = 0
    try:
        while sim_app.is_running() and grasp_idx < len(ik_grasps):
            grasp = ik_grasps[grasp_idx]

            # Palm-up wrist
            set_robot_root_pose(env, env_ids, palmup_pos, palmup_quat)

            # Set joints
            q = torch.tensor(
                grasp.joint_angles, device=device, dtype=torch.float32,
            ).unsqueeze(0)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
            robot.set_joint_position_target(q, env_ids=env_ids)

            # Move object away, step for FK
            temp[:, :3] = env.scene.env_origins[env_ids] + torch.tensor([[0, 0, -10.0]], device=device)
            obj.write_root_state_to_sim(temp, env_ids=env_ids)
            obj.update(0.0)
            env.sim.step(render=True)
            env.scene.update(dt=env.physics_dt)

            # Re-write joints (fix drift)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
            env.sim.step(render=True)
            env.scene.update(dt=env.physics_dt)

            # Place object at fingertip centroid
            ft_pos = robot.data.body_pos_w[env_ids][:, ft_ids, :]
            obj_pos = ft_pos.mean(dim=1)

            obj_state = obj.data.default_root_state[env_ids].clone()
            obj_state[:, :3] = obj_pos
            obj_state[:, 3:7] = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32)
            obj_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
            obj.update(0.0)

            # Print info
            ft_obj = (ft_pos[0] - obj_pos[0]).cpu().numpy()
            closest, dists, _ = trimesh.proximity.closest_point(mesh, ft_obj)
            print(f"  Grasp {grasp_idx}/{len(ik_grasps)} "
                  f"(quality={grasp.quality:.3f})")
            for fi in range(len(ft_ids)):
                print(f"    ft[{fi}]: dist={dists[fi]*1000:.1f}mm "
                      f"pos={ft_pos[0, fi].tolist()}")
            print(f"    obj_center={obj_pos[0].tolist()}")
            print(f"    (holding for {args.interval}s — inspect in GUI)")

            # Hold and render for interval seconds (NO PHYSICS — static view)
            t0 = time.time()
            while time.time() - t0 < args.interval and sim_app.is_running():
                # Render only, no physics step — object stays in place
                sim_app.update()

            grasp_idx += 1

    except KeyboardInterrupt:
        print("\nStopped.")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
