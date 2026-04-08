"""
Grasp Generation with Physics Validation.

Pipeline:
  1. HeuristicSampler generates candidates (Isaac Sim FK + NFO quality)
  2. SimGraspValidator filters via PhysX physics settle

Usage:
    # With visualization
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_sim_grasp_generation.py \\
        --shapes cube --size_min 0.06 --size_max 0.06 --num_grasps 300

    # Headless (faster)
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_sim_grasp_generation.py \\
        --shapes cube sphere cylinder \\
        --size_min 0.05 --size_max 0.08 --num_sizes 3 \\
        --num_grasps 300 --num_envs 64 --headless
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Grasp Generation + Physics Validation")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--size_min", type=float, default=0.06)
    p.add_argument("--size_max", type=float, default=0.08)
    p.add_argument("--num_sizes", type=int, default=1)

    # Generation targets
    p.add_argument("--num_grasps", type=int, default=300)
    p.add_argument("--output", type=str, default="data/grasp_graph_sim.pkl")

    # Phase 1: HeuristicSampler
    p.add_argument("--num_candidates", type=int, default=20000,
                   help="Max FK candidates to try")
    p.add_argument("--noise_std", type=float, default=0.3)
    p.add_argument("--contact_threshold", type=float, default=0.03,
                   help="Fingertip-to-surface distance for contact (m)")
    p.add_argument("--min_contact_fingers", type=int, default=3)
    p.add_argument("--penetration_margin", type=float, default=0.008)
    p.add_argument("--nfo_min_quality", type=float, default=0.03,
                   help="NFO force-closure quality threshold")

    # Phase 2: Physics validation
    p.add_argument("--settle_steps", type=int, default=15)
    p.add_argument("--vel_threshold", type=float, default=0.3)

    # Graph
    p.add_argument("--delta_max", type=float, default=0.04)

    # Isaac Sim
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _build_graph(grasps, object_name, num_fingers, delta_max):
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
    import isaaclab.sim as sim_utils
    from envs.anygrasp_env import AnyGraspEnvCfg

    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = num_envs

    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=0.8,
        restitution=0.0, friction_combine_mode="max",
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, max_depenetration_velocity=5.0,
        enable_gyroscopic_forces=True,
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=0.05)
    collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.002, rest_offset=0.0,
    )
    color = {"cube": (0.8, 0.2, 0.2), "sphere": (0.2, 0.6, 0.9),
             "cylinder": (0.3, 0.8, 0.3)}.get(shape, (0.7, 0.7, 0.7))
    vis = sim_utils.PreviewSurfaceCfg(diffuse_color=color)

    if shape == "cube":
        spawner = sim_utils.CuboidCfg(
            size=(size, size, size), rigid_props=rigid_props,
            mass_props=mass_props, collision_props=collision_props,
            physics_material=material, visual_material=vis,
        )
    elif shape == "sphere":
        spawner = sim_utils.SphereCfg(
            radius=size / 2.0, rigid_props=rigid_props,
            mass_props=mass_props, collision_props=collision_props,
            physics_material=material, visual_material=vis,
        )
    elif shape == "cylinder":
        spawner = sim_utils.CylinderCfg(
            radius=size / 2.0, height=size, rigid_props=rigid_props,
            mass_props=mass_props, collision_props=collision_props,
            physics_material=material, visual_material=vis,
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")

    env_cfg.scene.object.spawn = spawner
    return env_cfg


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from grasp_generation.sim_grasp_sampler import generate_and_validate
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph

    render = not args.headless
    num_fingers = 5
    sizes = np.linspace(args.size_min, args.size_max, args.num_sizes)
    multi_graph = MultiObjectGraspGraph(graphs={}, object_specs={})

    for shape in args.shapes:
        for size in sizes:
            size = float(round(size, 3))
            obj_name = f"{shape}_{int(size * 1000):03d}_f{num_fingers}"

            print(f"\n{'='*60}")
            print(f"  {obj_name} (shape={shape}, size={size}m)")
            print(f"{'='*60}")

            env_cfg = _build_env_cfg(shape, size, args.num_envs)
            env = ManagerBasedRLEnv(env_cfg)

            # Warm up
            env.sim.step(render=render)
            env.scene.update(dt=env.physics_dt)

            grasp_set = generate_and_validate(
                env=env,
                object_name=obj_name,
                object_shape=shape,
                object_size=size,
                num_grasps=args.num_grasps,
                num_candidates=args.num_candidates,
                noise_std=args.noise_std,
                contact_threshold=args.contact_threshold,
                min_contact_fingers=args.min_contact_fingers,
                penetration_margin=args.penetration_margin,
                nfo_min_quality=args.nfo_min_quality,
                settle_steps=args.settle_steps,
                vel_threshold=args.vel_threshold,
                render=render,
                seed=args.seed,
            )

            env.close()
            grasps = list(grasp_set.grasps)

            if len(grasps) == 0:
                print(f"  WARNING: 0 grasps for {obj_name}")
                continue

            graph = _build_graph(grasps, obj_name, num_fingers, args.delta_max)
            multi_graph.graphs[obj_name] = graph
            multi_graph.object_specs[obj_name] = {
                "name": obj_name, "shape_type": shape,
                "size": size, "num_fingers": num_fingers,
            }

            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(multi_graph, f)
            print(f"  [Saved] {len(grasps)} grasps → {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, g in multi_graph.graphs.items():
        print(f"  {name:30s}  {len(g.grasp_set):4d} grasps  "
              f"{len(g.edges):4d} edges")
    print(f"{'='*60}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(multi_graph, f)
    print(f"\nSaved to: {out_path}")

    sim_app.close()


if __name__ == "__main__":
    main()
