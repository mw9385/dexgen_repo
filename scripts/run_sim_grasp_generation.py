"""
Simulation-based Grasp Generation for Shadow Hand.

Generates collision-free grasps using Isaac Sim physics directly.
Unlike DexGraspNet optimization (MJCF FK + analytical SDF), this approach:
  - Uses Isaac Sim FK → no MJCF-Isaac FK mismatch
  - Validates via PhysX physics settle → no penetration
  - Joint angles are in 24-DOF Isaac format directly

Usage:
    # Basic (single object)
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_sim_grasp_generation.py \\
        --shapes cube --size_min 0.06 --size_max 0.06 --num_grasps 300

    # Full object pool
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_sim_grasp_generation.py \\
        --shapes cube sphere cylinder \\
        --size_min 0.05 --size_max 0.08 --num_sizes 3 \\
        --num_grasps 300 --num_envs 64

    # High quality with NFO filtering
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_sim_grasp_generation.py \\
        --shapes cube --num_grasps 500 --nfo_min_quality 0.03
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Sim-based Grasp Generation")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=["cube"],
                   help="Object shapes to generate grasps for")
    p.add_argument("--size_min", type=float, default=0.06)
    p.add_argument("--size_max", type=float, default=0.08)
    p.add_argument("--num_sizes", type=int, default=1)

    # Generation targets
    p.add_argument("--num_grasps", type=int, default=300,
                   help="Target number of grasps per object")
    p.add_argument("--max_rounds", type=int, default=500,
                   help="Max sampling rounds per object (each tests num_envs candidates)")
    p.add_argument("--output", type=str, default="data/grasp_graph_sim.pkl")

    # Sampler parameters
    p.add_argument("--noise_std", type=float, default=0.25,
                   help="Joint noise std (fraction of joint range)")
    p.add_argument("--settle_steps", type=int, default=8,
                   help="Physics settle steps after object placement")
    p.add_argument("--vel_threshold", type=float, default=0.25,
                   help="Max object velocity after settle (m/s)")
    p.add_argument("--contact_threshold", type=float, default=0.015,
                   help="Max fingertip-to-surface distance for contact (m)")
    p.add_argument("--min_contact_fingers", type=int, default=3,
                   help="Minimum fingertips in contact with object")
    p.add_argument("--penetration_margin", type=float, default=0.005,
                   help="Max finger-mesh penetration depth (m)")
    p.add_argument("--nfo_min_quality", type=float, default=0.0,
                   help="Minimum NFO quality (0 = disabled)")

    # Graph construction
    p.add_argument("--delta_max", type=float, default=0.04,
                   help="Max edge distance for graph connectivity")

    # Isaac Sim / Isaac Lab arguments
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num_envs", type=int, default=64,
                   help="Number of parallel environments for batched sampling")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _build_graph(grasps, object_name, num_fingers, delta_max):
    """Build GraspGraph from flat grasp list using fingertip distance."""
    from grasp_generation.grasp_sampler import GraspSet
    from grasp_generation.rrt_expansion import GraspGraph

    grasp_set = GraspSet(grasps=list(grasps), object_name=object_name)
    n = len(grasp_set)
    if n == 0:
        return GraspGraph(
            grasp_set=grasp_set, object_name=object_name,
            num_fingers=num_fingers,
        )

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


def _build_env_cfg(shape: str, size: float, num_envs: int):
    """Build AnyGraspEnvCfg with specific object spawner."""
    import isaaclab.sim as sim_utils
    from envs.anygrasp_env import AnyGraspEnvCfg

    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = num_envs

    # Material for good grasp physics
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.8,
        restitution=0.0,
        friction_combine_mode="max",
    )
    # High depenetration velocity for detecting bad grasps
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
        enable_gyroscopic_forces=True,
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=0.05)
    collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.002, rest_offset=0.0,
    )
    visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color={"cube": (0.8, 0.2, 0.2),
                       "sphere": (0.2, 0.6, 0.9),
                       "cylinder": (0.3, 0.8, 0.3)}.get(shape, (0.7, 0.7, 0.7)),
    )

    if shape == "cube":
        spawner = sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=rigid_props, mass_props=mass_props,
            collision_props=collision_props, physics_material=material,
            visual_material=visual_material,
        )
    elif shape == "sphere":
        spawner = sim_utils.SphereCfg(
            radius=size / 2.0,
            rigid_props=rigid_props, mass_props=mass_props,
            collision_props=collision_props, physics_material=material,
            visual_material=visual_material,
        )
    elif shape == "cylinder":
        spawner = sim_utils.CylinderCfg(
            radius=size / 2.0, height=size,
            rigid_props=rigid_props, mass_props=mass_props,
            collision_props=collision_props, physics_material=material,
            visual_material=visual_material,
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")

    env_cfg.scene.object.spawn = spawner
    return env_cfg


def main():
    args = parse_args()

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from grasp_generation.sim_grasp_sampler import SimGraspSampler
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

    num_fingers = 5  # Shadow Hand
    sizes = np.linspace(args.size_min, args.size_max, args.num_sizes)
    multi_graph = MultiObjectGraspGraph(graphs={}, object_specs={})

    for shape in args.shapes:
        for size in sizes:
            size = float(round(size, 3))
            obj_name = f"{shape}_{int(size * 1000):03d}_f{num_fingers}"

            print(f"\n{'='*60}")
            print(f"  Sim Grasp Generation: {obj_name}")
            print(f"  shape={shape}, size={size}m, num_envs={args.num_envs}")
            print(f"{'='*60}")

            # Build environment with this specific object
            env_cfg = _build_env_cfg(shape, size, args.num_envs)
            env = ManagerBasedRLEnv(env_cfg)

            # Warm up: step once to initialise physics
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            # Create sampler and run
            sampler = SimGraspSampler(
                env=env,
                object_name=obj_name,
                object_shape=shape,
                object_size=size,
                num_fingers=num_fingers,
                settle_steps=args.settle_steps,
                vel_threshold=args.vel_threshold,
                contact_threshold=args.contact_threshold,
                min_contact_fingers=args.min_contact_fingers,
                penetration_margin=args.penetration_margin,
                noise_std=args.noise_std,
                nfo_min_quality=args.nfo_min_quality,
                seed=args.seed,
            )

            grasp_set = sampler.sample(
                num_grasps=args.num_grasps,
                max_rounds=args.max_rounds,
            )
            grasps = list(grasp_set.grasps)

            # Close environment before creating next one
            env.close()

            if len(grasps) == 0:
                print(f"  WARNING: 0 grasps for {obj_name}")
                continue

            # Build graph
            graph = _build_graph(grasps, obj_name, num_fingers, args.delta_max)

            multi_graph.graphs[obj_name] = graph
            multi_graph.object_specs[obj_name] = {
                "name": obj_name,
                "shape_type": shape,
                "size": size,
                "num_fingers": num_fingers,
            }

            # Incremental save after each object
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(multi_graph, f)
            print(f"  [Saved] {len(grasps)} grasps → {out_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("  GENERATION SUMMARY")
    print(f"{'='*60}")
    for name, g in multi_graph.graphs.items():
        n = len(g.grasp_set)
        e = len(g.edges)
        print(f"  {name:30s}  {n:4d} grasps  {e:4d} edges")
    print(f"{'='*60}")

    # Final save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(multi_graph, f)
    print(f"\nSaved to: {out_path}")

    sim_app.close()


if __name__ == "__main__":
    main()
