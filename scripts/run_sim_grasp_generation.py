"""
DexterityGen Grasp Generation Pipeline.

  1. Algorithm 3: sampleSurface → NFO → randomPose → IK → collision
  2. RRT Expand: nearest neighbor → steer → NFO → IK → collision
  3. Physics validation: PhysX settle test

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_sim_grasp_generation.py \\
        --shapes cube --size_min 0.05 --size_max 0.05 --num_grasps 50

    /workspace/IsaacLab/isaaclab.sh -p scripts/run_sim_grasp_generation.py \\
        --shapes cube sphere cylinder --size_min 0.05 --size_max 0.08 \\
        --num_sizes 3 --n_pts 5 --headless
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="DexterityGen Grasp Generation")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--size_min", type=float, default=0.06)
    p.add_argument("--size_max", type=float, default=0.08)
    p.add_argument("--num_sizes", type=int, default=1)

    # Grasp configuration
    p.add_argument("--n_pts", type=int, default=5, choices=[3, 4, 5],
                   help="Number of contact points per grasp (3, 4, or 5)")
    p.add_argument("--num_grasps", type=int, default=300,
                   help="Final target number of grasps")
    p.add_argument("--output", type=str, default="data/grasp_graph_sim.pkl")

    # Algorithm 3: seed generation
    p.add_argument("--num_seeds", type=int, default=50,
                   help="Seed grasps from Algorithm 3")
    p.add_argument("--max_candidates", type=int, default=50000)
    p.add_argument("--num_pose_samples", type=int, default=10)
    p.add_argument("--nfo_min_quality", type=float, default=0.03)
    p.add_argument("--collision_margin", type=float, default=0.001)

    # RRT Expand
    p.add_argument("--rrt_target", type=int, default=300,
                   help="Target graph size after RRT expansion")
    p.add_argument("--delta_pos", type=float, default=0.008)
    p.add_argument("--delta_max", type=float, default=0.04)
    p.add_argument("--max_attempts", type=int, default=30)
    p.add_argument("--rrt_collision_threshold", type=float, default=0.002)

    # Physics validation
    p.add_argument("--settle_steps", type=int, default=40)
    p.add_argument("--vel_threshold", type=float, default=0.01)

    # Isaac Sim
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


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
        disable_gravity=False, max_depenetration_velocity=0.1,
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
            physics_material=material, visual_material=vis)
    elif shape == "sphere":
        spawner = sim_utils.SphereCfg(
            radius=size / 2.0, rigid_props=rigid_props,
            mass_props=mass_props, collision_props=collision_props,
            physics_material=material, visual_material=vis)
    elif shape == "cylinder":
        spawner = sim_utils.CylinderCfg(
            radius=size / 2.0, height=size, rigid_props=rigid_props,
            mass_props=mass_props, collision_props=collision_props,
            physics_material=material, visual_material=vis)
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
    from grasp_generation.sim_grasp_sampler import (
        generate_grasp_set,
        make_primitive_mesh,
        SimGraspValidator,
    )
    from grasp_generation.rrt_expansion import (
        rrt_expand,
        MultiObjectGraspGraph,
    )

    render = not args.headless
    sizes = np.linspace(args.size_min, args.size_max, args.num_sizes)
    multi_graph = MultiObjectGraspGraph(graphs={}, object_specs={})

    for shape in args.shapes:
        for size in sizes:
            size = float(round(size, 3))
            obj_name = f"{shape}_{int(size * 1000):03d}_f{args.n_pts}"

            print(f"\n{'='*60}")
            print(f"  {obj_name} (n_pts={args.n_pts})")
            print(f"{'='*60}")

            env_cfg = _build_env_cfg(shape, size, args.num_envs)
            env = ManagerBasedRLEnv(env_cfg)
            env.sim.step(render=render)
            env.scene.update(dt=env.physics_dt)

            mesh = make_primitive_mesh(shape, size)

            # ── Phase 1: Algorithm 3 → seed grasps ──────────────
            print(f"\n  Phase 1: Algorithm 3 (seeds)")
            seeds = generate_grasp_set(
                env=env, mesh=mesh,
                object_name=obj_name, object_size=size,
                num_grasps=args.num_seeds,
                max_candidates=args.max_candidates,
                nfo_min_quality=args.nfo_min_quality,
                collision_margin=args.collision_margin,
                render=render, seed=args.seed,
            )

            if len(seeds) == 0:
                print(f"  WARNING: 0 seeds for {obj_name}")
                env.close()
                continue

            # ── Phase 2: RRT Expand ──────────────────────────────
            print(f"\n  Phase 2: RRT Expand")
            from grasp_generation.net_force_optimization import NetForceOptimizer
            nfo = NetForceOptimizer(
                mu=0.5, num_edges=8, min_quality=args.nfo_min_quality,
            )

            graph = rrt_expand(
                seed_grasps=seeds,
                mesh=mesh, nfo=nfo, env=env,
                target_size=args.rrt_target,
                delta_pos=args.delta_pos,
                delta_max=args.delta_max,
                min_quality=args.nfo_min_quality,
                max_attempts_per_step=args.max_attempts,
                collision_threshold=args.rrt_collision_threshold,
                object_size=size,
                render=render, seed=args.seed,
            )

            # ── Phase 3: Physics validation ──────────────────────
            print(f"\n  Phase 3: Physics validation")
            validator = SimGraspValidator(
                env=env,
                settle_steps=args.settle_steps,
                vel_threshold=args.vel_threshold,
                render=render,
            )
            valid_grasps = validator.validate(
                graph.grasp_set.grasps, verbose=True,
            )

            env.close()

            if len(valid_grasps) == 0:
                print(f"  WARNING: 0 physics-valid grasps for {obj_name}")
                continue

            # Rebuild graph with only valid grasps
            from grasp_generation.grasp_sampler import GraspSet
            from grasp_generation.rrt_expansion import GraspGraph
            valid_set = GraspSet(grasps=valid_grasps, object_name=obj_name)

            # Re-compute edges for valid grasps
            n = len(valid_grasps)
            eff_delta = args.delta_max * (1 + 0.35 * max(args.n_pts - 1, 0))
            edges = []
            fps = np.stack([g.fingertip_positions.flatten() for g in valid_grasps])
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(np.linalg.norm(fps[i] - fps[j])) / max(args.n_pts, 1)
                    if d < eff_delta:
                        edges.append((i, j))

            valid_graph = GraspGraph(
                grasp_set=valid_set, edges=edges,
                object_name=obj_name, num_fingers=args.n_pts,
            )

            multi_graph.graphs[obj_name] = valid_graph
            multi_graph.object_specs[obj_name] = {
                "name": obj_name, "shape_type": shape,
                "size": size, "num_fingers": args.n_pts,
            }

            # Save incrementally
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(multi_graph, f)
            print(f"  [Saved] {len(valid_grasps)} grasps, "
                  f"{len(edges)} edges → {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, g in multi_graph.graphs.items():
        print(f"  {name:30s}  {len(g.grasp_set):4d} grasps  "
              f"{len(g.edges):4d} edges")
    print(f"{'='*60}")

    out_path = Path(args.output)
    with open(out_path, "wb") as f:
        pickle.dump(multi_graph, f)
    print(f"\nSaved to: {out_path}")
    sim_app.close()


if __name__ == "__main__":
    main()
