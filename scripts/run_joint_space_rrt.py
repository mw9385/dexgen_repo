"""
Joint-Space RRT Grasp Generation — runs inside Isaac Sim.

Generates kinematically feasible grasps by exploring joint space via FK.
Every grasp in the output has valid joint_angles + object_pose — no
separate solve step needed.

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_joint_space_rrt.py \
        --shapes cube --size_min 0.08 --size_max 0.08 \
        --num_grasps 300 --output data/grasp_graph.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Joint-space RRT grasp generation")
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--size_min", type=float, default=0.06)
    p.add_argument("--size_max", type=float, default=0.08)
    p.add_argument("--num_sizes", type=int, default=1)
    p.add_argument("--num_grasps", type=int, default=300)
    p.add_argument("--output", type=str, default="data/grasp_graph.pkl")

    # Validation thresholds
    p.add_argument("--contact_threshold", type=float, default=0.015)
    p.add_argument("--min_quality", type=float, default=0.03)
    p.add_argument("--joint_noise_std", type=float, default=0.1)
    p.add_argument("--seed_attempts", type=int, default=5000)

    # Isaac Sim args
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    return p.parse_args()


def _make_mesh(shape: str, size: float) -> "trimesh.Trimesh":
    """Create a trimesh primitive for the given shape and size."""
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

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    import trimesh
    from isaaclab.envs import ManagerBasedRLEnv

    from envs.anygrasp_env import AnyGraspEnvCfg
    from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
    from grasp_generation.joint_space_rrt import (
        JointSpaceRRTConfig,
        JointSpaceRRTGenerator,
    )
    from grasp_generation.net_force_optimization import NetForceOptimizer
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

    # Create env
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    ft_ids = get_fingertip_body_ids_from_env(robot, env)

    # NFO quality scorer
    nfo = NetForceOptimizer(
        mu=0.5,
        min_quality=args.min_quality,
        fast_mode=False,
    )

    # RRT config
    rrt_cfg = JointSpaceRRTConfig(
        target_size=args.num_grasps,
        num_seed_attempts=args.seed_attempts,
        contact_threshold=args.contact_threshold,
        min_quality=args.min_quality,
        joint_noise_std=args.joint_noise_std,
    )

    # Generate for each shape × size
    sizes = np.linspace(args.size_min, args.size_max, args.num_sizes)
    multi_graph = MultiObjectGraspGraph(graphs={}, object_specs={})
    num_fingers = len(ft_ids)

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
                cfg=rrt_cfg,
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

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(multi_graph, f)
    print(f"\nSaved to: {out_path}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
