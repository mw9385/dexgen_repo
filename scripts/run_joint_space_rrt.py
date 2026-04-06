"""
Joint-Space RRT Grasp Generation — Isaac Sim entry point.

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_joint_space_rrt.py \
        --shapes cube --size_min 0.08 --size_max 0.08 --num_grasps 300
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Joint-Space RRT Grasp Generation")
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--size_min", type=float, default=0.06)
    p.add_argument("--size_max", type=float, default=0.08)
    p.add_argument("--num_sizes", type=int, default=1)
    p.add_argument("--num_grasps", type=int, default=300)
    p.add_argument("--num_initial", type=int, default=50)
    p.add_argument("--num_rrt_steps", type=int, default=None)
    p.add_argument("--output", type=str, default="data/grasp_graph.pkl")
    p.add_argument("--f_thresh", type=float, default=0.03)
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


def main():
    args = parse_args()
    if args.num_rrt_steps is None:
        args.num_rrt_steps = args.num_grasps

    # Isaac Sim must be launched BEFORE importing isaaclab modules
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import trimesh
    from isaaclab.envs import ManagerBasedRLEnv

    from envs.anygrasp_env import AnyGraspEnvCfg
    from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
    from grasp_generation.joint_space_rrt import JointSpaceRRTConfig, JointSpaceRRTGenerator
    from grasp_generation.net_force_optimization import NetForceOptimizer
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

    # Create env
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    num_fingers = len(ft_ids)

    nfo = NetForceOptimizer(min_quality=args.f_thresh, fast_mode=False)

    cfg = JointSpaceRRTConfig(
        target_size=args.num_rrt_steps,
        min_quality=args.f_thresh,
    )

    sizes = np.linspace(args.size_min, args.size_max, args.num_sizes)
    multi_graph = MultiObjectGraspGraph(graphs={}, object_specs={})

    for shape in args.shapes:
        for size in sizes:
            size = float(round(size, 3))
            obj_name = f"{shape}_{int(size * 1000):03d}_f{num_fingers}"

            print(f"\n{'='*60}")
            print(f"  Generating: {obj_name} (shape={shape}, size={size}m)")
            print(f"{'='*60}")

            mesh = _make_mesh(shape, size)

            generator = JointSpaceRRTGenerator(
                env=env,
                mesh=mesh,
                nfo=nfo,
                ft_ids=ft_ids,
                cfg=cfg,
                object_name=obj_name,
                object_size=size,
            )
            graph = generator.generate()

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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(multi_graph, f)
    print(f"\nSaved to: {out_path}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
