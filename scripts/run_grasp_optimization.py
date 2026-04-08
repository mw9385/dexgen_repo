"""
DexGraspNet-style Grasp Optimization for Shadow Hand.

Runs Simulated Annealing optimization on GPU to find stable grasps,
then optionally validates in Isaac Sim physics.

Usage:
    # Optimization only (fast, no Isaac Sim needed)
    python scripts/run_grasp_optimization.py \
        --shapes cube --size_min 0.08 --size_max 0.08 --num_grasps 300

    # With Isaac Sim physics validation
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_optimization.py \
        --shapes cube --size_min 0.08 --size_max 0.08 --num_grasps 300 \
        --validate_physics
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="DexGraspNet Grasp Optimization")
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--size_min", type=float, default=0.06)
    p.add_argument("--size_max", type=float, default=0.08)
    p.add_argument("--num_sizes", type=int, default=1)
    p.add_argument("--num_grasps", type=int, default=300)
    p.add_argument("--output", type=str, default="data/grasp_graph.pkl")

    # Optimization params
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--n_iter", type=int, default=6000)
    p.add_argument("--n_contact", type=int, default=5)

    # Energy weights
    p.add_argument("--w_dis", type=float, default=100.0)
    p.add_argument("--w_pen", type=float, default=100.0)
    p.add_argument("--w_spen", type=float, default=10.0)
    p.add_argument("--w_joints", type=float, default=1.0)

    # Threshold filtering
    p.add_argument("--thres_fc", type=float, default=0.3)
    p.add_argument("--thres_dis", type=float, default=0.005)
    p.add_argument("--thres_pen", type=float, default=0.001)

    # Isaac Sim validation
    p.add_argument("--validate_physics", action="store_true", default=False)

    # Graph construction
    p.add_argument("--delta_max", type=float, default=0.04)

    # Isaac Sim args (only used with --validate_physics)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)

    return p.parse_args()


def _make_mesh(shape: str, size: float):
    import trimesh
    if shape == "cube":
        return trimesh.creation.box(extents=[size, size, size])
    elif shape == "sphere":
        return trimesh.creation.icosphere(radius=size / 2.0)
    elif shape == "cylinder":
        return trimesh.creation.cylinder(radius=size / 2.0, height=size)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def _build_graph(grasps, object_name, num_fingers, delta_max):
    """Build GraspGraph from flat grasp list using fingertip distance."""
    from grasp_generation.grasp_sampler import GraspSet
    from grasp_generation.rrt_expansion import GraspGraph

    grasp_set = GraspSet(grasps=list(grasps), object_name=object_name)
    n = len(grasp_set)
    if n == 0:
        return GraspGraph(grasp_set=grasp_set, object_name=object_name,
                          num_fingers=num_fingers)

    # Edges: fingertip-space distance
    effective_delta = delta_max * (1.0 + 0.35 * max(num_fingers - 1, 0))
    edges = []
    all_fps = np.stack([g.fingertip_positions.flatten() for g in grasps])
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(all_fps[i] - all_fps[j]))
            mean_dist = dist / max(num_fingers, 1)
            if mean_dist < effective_delta:
                edges.append((i, j))

    return GraspGraph(
        grasp_set=grasp_set, edges=edges,
        object_name=object_name, num_fingers=num_fingers,
    )


def _expand_joints_22_to_24(joint_angles_22):
    """Convert MJCF 22-DOF to Isaac Sim USD 24-DOF.

    MJCF order (per-finger): FFJ3210, MFJ3210, RFJ3210, LFJ43210, THJ43210
    Isaac USD order (per-level): WRJ10, {FFJ3,MFJ3,RFJ3,LFJ4,THJ4},
        {FFJ2,MFJ2,RFJ2,LFJ3,THJ3}, {FFJ1,MFJ1,RFJ1,LFJ2,THJ2},
        {FFJ0,MFJ0,RFJ0,LFJ1,THJ1}, {LFJ0,THJ0}
    """
    import numpy as np
    out = np.zeros(24, dtype=np.float32)
    j = joint_angles_22
    # WRJ1, WRJ0 = 0 (not in MJCF 22-DOF)
    # MJCF: FF=[0]J3 [1]J2 [2]J1 [3]J0
    out[2]  = j[0]   # FFJ3
    out[7]  = j[1]   # FFJ2
    out[12] = j[2]   # FFJ1
    out[17] = j[3]   # FFJ0
    # MJCF: MF=[4]J3 [5]J2 [6]J1 [7]J0
    out[3]  = j[4]   # MFJ3
    out[8]  = j[5]   # MFJ2
    out[13] = j[6]   # MFJ1
    out[18] = j[7]   # MFJ0
    # MJCF: RF=[8]J3 [9]J2 [10]J1 [11]J0
    out[4]  = j[8]   # RFJ3
    out[9]  = j[9]   # RFJ2
    out[14] = j[10]  # RFJ1
    out[19] = j[11]  # RFJ0
    # MJCF: LF=[12]J4 [13]J3 [14]J2 [15]J1 [16]J0
    out[5]  = j[12]  # LFJ4
    out[10] = j[13]  # LFJ3
    out[15] = j[14]  # LFJ2
    out[20] = j[15]  # LFJ1
    out[22] = j[16]  # LFJ0
    # MJCF: TH=[17]J4 [18]J3 [19]J2 [20]J1 [21]J0
    out[6]  = j[17]  # THJ4
    out[11] = j[18]  # THJ3
    out[16] = j[19]  # THJ2
    out[21] = j[20]  # THJ1
    out[23] = j[21]  # THJ0
    return out


def main():
    args = parse_args()

    # Isaac Sim launch only if physics validation requested
    sim_app = None
    if args.validate_physics:
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(args)
        sim_app = app_launcher.app

    import torch
    import trimesh
    from grasp_generation.grasp_optimization import GraspOptimizer
    from grasp_generation.hand_model import build_hand_model
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

    num_fingers = 5  # Shadow Hand
    sizes = np.linspace(args.size_min, args.size_max, args.num_sizes)
    multi_graph = MultiObjectGraspGraph(graphs={}, object_specs={})

    # Build hand model once (reused across all objects)
    hand_model = build_hand_model(hand_name="shadow", device=args.device)

    for shape in args.shapes:
        for size in sizes:
            size = float(round(size, 3))
            obj_name = f"{shape}_{int(size * 1000):03d}_f{num_fingers}"

            print(f"\n{'='*60}")
            print(f"  Optimizing: {obj_name} (shape={shape}, size={size}m)")
            print(f"{'='*60}")

            mesh = _make_mesh(shape, size)

            optimizer = GraspOptimizer(
                hand_model=hand_model,
                mesh=mesh,
                shape_type=shape,
                size=size,
                batch_size=args.batch_size,
                n_iter=args.n_iter,
                n_contact=args.n_contact,
                w_dis=args.w_dis,
                w_pen=args.w_pen,
                w_spen=args.w_spen,
                w_joints=args.w_joints,
                thres_fc=args.thres_fc,
                thres_dis=args.thres_dis,
                thres_pen=args.thres_pen,
                device=args.device,
            )

            # Incremental save callback: convert + save after optimization
            out_path = Path(args.output)

            def _save_incremental(grasps_so_far):
                converted = []
                for g in grasps_so_far:
                    if g.joint_angles is not None and len(g.joint_angles) == 22:
                        g.joint_angles = _expand_joints_22_to_24(g.joint_angles)
                    g.object_name = obj_name
                    converted.append(g)
                _graph = _build_graph(converted, obj_name, num_fingers, args.delta_max)
                _mg = MultiObjectGraspGraph(graphs={}, object_specs={})
                _mg.graphs[obj_name] = _graph
                _mg.object_specs[obj_name] = {
                    "name": obj_name, "shape_type": shape,
                    "size": size, "num_fingers": num_fingers,
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    pickle.dump(_mg, f)
                print(f"  [Saved] {len(converted)} grasps → {out_path}")

            grasp_set = optimizer.optimize(
                num_grasps=args.num_grasps,
                save_callback=_save_incremental,
            )
            grasps = list(grasp_set.grasps)

            if len(grasps) == 0:
                print(f"  WARNING: 0 grasps for {obj_name}")
                continue

            # Convert MJCF 22-DOF → Isaac 24-DOF (final pass for any unconverted)
            for g in grasps:
                if g.joint_angles is not None and len(g.joint_angles) == 22:
                    g.joint_angles = _expand_joints_22_to_24(g.joint_angles)
                g.object_name = obj_name

            # Isaac Sim physics validation (optional)
            if args.validate_physics and sim_app is not None:
                print(f"  Running Isaac Sim physics validation...")
                grasps = _validate_physics(args, grasps, obj_name)
                print(f"  {len(grasps)} grasps survived physics validation")

            # Build graph
            graph = _build_graph(grasps, obj_name, num_fingers, args.delta_max)

            multi_graph.graphs[obj_name] = graph
            multi_graph.object_specs[obj_name] = {
                "name": obj_name,
                "shape_type": shape,
                "size": size,
                "num_fingers": num_fingers,
            }

    # Summary
    print(f"\n{'='*60}")
    print("  GENERATION SUMMARY")
    print(f"{'='*60}")
    for name, g in multi_graph.graphs.items():
        n = len(g.grasp_set)
        e = len(g.edges)
        print(f"  {name:30s}  {n:4d} grasps  {e:4d} edges")
    print(f"{'='*60}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(multi_graph, f)
    print(f"\nSaved to: {out_path}")

    if sim_app is not None:
        sim_app.close()


def _validate_physics(args, grasps, obj_name):
    """Filter grasps via Isaac Sim physics settle check."""
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply
    from envs.anygrasp_env import AnyGraspEnvCfg
    from envs.mdp.math_utils import quat_multiply

    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    valid = []
    for i, grasp in enumerate(grasps):
        if grasp.joint_angles is None:
            continue

        # Set joints
        q = torch.tensor(grasp.joint_angles, device=device, dtype=torch.float32).unsqueeze(0)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        robot.set_joint_position_target(q, env_ids=env_ids)

        # Move object far away before FK step to prevent collision ejection
        temp_state = obj.data.default_root_state[env_ids].clone()
        temp_state[:, :3] = torch.tensor([[0, 0, -10.0]], device=device)
        temp_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(temp_state, env_ids=env_ids)
        obj.update(0.0)

        # FK step
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        # Place object at Isaac FK fingertip centroid
        from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
        ft_ids = get_fingertip_body_ids_from_env(robot, env)
        ft_pos_w = robot.data.body_pos_w[env_ids][:, ft_ids, :]
        obj_pos_w = ft_pos_w.mean(dim=1)[0]
        obj_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        obj_state = obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos_w.unsqueeze(0)
        obj_state[:, 3:7] = obj_quat_w.unsqueeze(0)
        obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        obj.update(0.0)

        # Physics settle (30 steps = 0.25s at 120Hz)
        for _ in range(30):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

        # Check stability: low velocity + object still near hand
        speed = float(torch.norm(obj.data.root_lin_vel_w[0]).item())
        obj_z = float(obj.data.root_pos_w[0, 2].item())
        # Check object didn't drift far from fingertip centroid
        ft_pos_after = robot.data.body_pos_w[env_ids][:, ft_ids, :]
        centroid_after = ft_pos_after.mean(dim=1)[0]
        obj_drift = float(torch.norm(obj.data.root_pos_w[0] - centroid_after).item())

        if speed < 0.3 and obj_z > 0.15 and obj_drift < 0.05:
            valid.append(grasp)

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(grasps)}] valid: {len(valid)}")

    env.close()
    return valid


if __name__ == "__main__":
    main()
