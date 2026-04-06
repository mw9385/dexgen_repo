"""
DexGen-style Grasp Generation — runs inside Isaac Sim.

Pipeline strictly follows DexGen Appendix B:
  1. HeuristicSample (Algorithm 3): Surface sampling + Net Force Optimization (Algorithm 4)
  2. GraspRRTExpand (Algorithm 5): NN interpolation in (q, p) space + collision fix

Every grasp in the output has valid joint_angles (q) + object_pose (p).

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_dexgen_grasp_generation.py \
        --shapes cube --size_min 0.08 --size_max 0.08 \
        --num_initial 50 --num_rrt_steps 300 --output data/grasp_graph.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Pipeline Grasp Generation")
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--size_min", type=float, default=0.06)
    p.add_argument("--size_max", type=float, default=0.08)
    p.add_argument("--num_sizes", type=int, default=1)
    
    # DexGen Algorithm 2 params
    p.add_argument("--num_initial", type=int, default=50, help="N in HeuristicSample")
    p.add_argument("--num_rrt_steps", type=int, default=300, help="N_RRT in GraspRRTExpand")
    
    p.add_argument("--output", type=str, default="data/grasp_graph.pkl")

    # Algorithm 4 (GraspAnalysis) Validation thresholds
    p.add_argument("--f_thresh", type=float, default=0.03, help="Threshold for Net Force Opt")
    
    # Isaac Sim args
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
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

    import trimesh
    from isaaclab.envs import ManagerBasedRLEnv
    from envs.anygrasp_env import AnyGraspEnvCfg
    from envs.mdp.sim_utils import get_fingertip_body_ids_from_env
    
    # Import DexGen-specific modules 
    from grasp_generation.net_force_optimization import NetForceOptimizer
    from grasp_generation.dexgen_pipeline import (
        HeuristicSampler,  # Implements Algorithm 3
        GraspRRTExpander,  # Implements Algorithm 5
        MultiObjectGraspGraph
    )

    # Create env
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    num_fingers = len(ft_ids)

    # NFO for GraspAnalysis (Algorithm 4)
    # Optimizes f_i to minimize net force.
    nfo = NetForceOptimizer(
        min_quality=args.f_thresh, 
        fast_mode=False
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

            # 1. Algorithm 3: HeuristicSample
            sampler = HeuristicSampler(
                env=env,
                mesh=mesh,
                nfo=nfo,
                ft_ids=ft_ids,
                num_samples=args.num_initial,
            )
            print("  [Step 1] Running Heuristic Sampling...")
            initial_grasp_set = sampler.generate_seeds()
            
            if len(initial_grasp_set) == 0:
                print(f"  [Warning] Failed to generate initial seeds for {obj_name}. Skipping.")
                continue

            # 2. Algorithm 5: GraspRRTExpand
            expander = GraspRRTExpander(
                env=env,
                mesh=mesh,
                rrt_steps=args.num_rrt_steps,
            )
            print(f"  [Step 2] Running RRT Expansion ({args.num_rrt_steps} steps)...")
            # expand() function should handle NearestNeighbor, Interpolate, and FixContactAndCollision
            final_graph = expander.expand(initial_grasp_set)

            multi_graph.graphs[obj_name] = final_graph
            multi_graph.object_specs[obj_name] = {
                "name": obj_name,
                "shape_type": shape,
                "size": size,
                "num_fingers": num_fingers,
            }

    # Summary
    print(f"\n{'='*60}")
    print("  GENERATION SUMMARY (DexGen Pipeline)")
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
