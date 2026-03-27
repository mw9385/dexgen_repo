"""
Stage 0 – Grasp Set Generation (Multi-Object)
===============================================
Generates a diverse set of quality grasps for a *pool* of random objects
and saves a MultiObjectGraspGraph for use in Stage 1 RL training.

Each object in the pool (cube / sphere / cylinder, various sizes) gets its
own GraspGraph. At RL training time the environment randomly selects an
object + grasp pair from this combined graph every episode.

Pipeline (per object):
  1. Sample candidate grasps on the object surface (GraspSampler)
  2. Score and filter by NFO quality (NetForceOptimizer)
  3. Expand with RRT to reach target_size (RRTGraspExpander)
  4. Merge all per-object graphs → MultiObjectGraspGraph

Usage:
    # Default: cube + sphere + cylinder, 3 sizes each
    python scripts/run_grasp_generation.py

    # Custom object pool
    python scripts/run_grasp_generation.py \\
        --shapes cube sphere \\
        --size_min 0.04 --size_max 0.09 --num_sizes 4 \\
        --num_grasps 300

    # Single custom mesh
    python scripts/run_grasp_generation.py --mesh_path assets/mug.obj
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from grasp_generation import (
    GraspSampler, NetForceOptimizer, RRTGraspExpander,
    ObjectPool, MultiObjectGraspGraph,
)


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 0: Multi-Object Grasp Generation")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=["cube", "sphere", "cylinder"],
                   choices=["cube", "sphere", "cylinder"],
                   help="Primitive shapes to include in the object pool")
    p.add_argument("--size_min", type=float, default=0.04,
                   help="Minimum object size in metres (default: 4 cm)")
    p.add_argument("--size_max", type=float, default=0.09,
                   help="Maximum object size in metres (default: 9 cm)")
    p.add_argument("--num_sizes", type=int, default=3,
                   help="Number of size steps per shape (default: 3)")
    p.add_argument("--mesh_path", type=str, default=None,
                   help="Single custom mesh file instead of pool (.obj/.stl/.ply)")
    p.add_argument("--mesh_dir", type=str, default=None,
                   help="Directory of mesh files — all loaded as pool objects")

    # Grasp generation quality
    p.add_argument("--num_seed_grasps", type=int, default=300,
                   help="Seed grasps per object before NFO filtering")
    p.add_argument("--num_grasps", type=int, default=300,
                   help="Target grasps per object after RRT expansion")
    p.add_argument("--min_quality", type=float, default=0.005,
                   help="Minimum NFO quality score")
    p.add_argument("--mu", type=float, default=0.5,
                   help="Friction coefficient for NFO")
    p.add_argument("--fast_nfo", action="store_true",
                   help="Fast SVD approximation for NFO (less accurate)")

    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent.parent / "configs" / "grasp_generation.yaml"),
        help="Path to grasp_generation.yaml (hand config, pool settings, etc.)",
    )
    p.add_argument(
        "--num_fingers", type=int, default=None,
        help="Number of contact points per grasp (overrides config file hand.num_fingers)",
    )
    return p.parse_args()


def process_one_object(spec, args, seed_offset: int, num_fingers: int = 4) -> tuple:
    """
    Run the full grasp generation pipeline for one ObjectSpec.
    Returns (GraspGraph, isaac_lab_spec) or (None, None) on failure.
    """
    from grasp_generation.grasp_sampler import GraspSampler

    print(f"\n{'='*55}")
    print(f"  Object:      {spec.name}  (shape={spec.shape_type}, size={spec.size:.3f}m)")
    print(f"  num_fingers: {num_fingers}")
    print(f"{'='*55}")

    # Step 1: Sample seed grasps
    sampler = GraspSampler(
        mesh=spec.mesh,
        object_name=spec.name,
        object_scale=spec.size / 0.06,   # relative to 6 cm reference
        num_candidates=args.num_seed_grasps * 20,
        num_grasps=args.num_seed_grasps,
        num_fingers=num_fingers,
        seed=args.seed + seed_offset,
    )
    seed_set = sampler.sample()

    if len(seed_set) == 0:
        print(f"  [!] No seed grasps sampled for {spec.name}, skipping.")
        return None, None

    # Step 2: NFO quality filter
    nfo = NetForceOptimizer(mu=args.mu, min_quality=args.min_quality,
                            fast_mode=args.fast_nfo)
    filtered_set = nfo.evaluate_set(seed_set, verbose=True)

    if len(filtered_set) < 10:
        print(f"  [!] Only {len(filtered_set)} grasps passed NFO for {spec.name}, skipping.")
        return None, None

    # Step 3: RRT expansion
    expander = RRTGraspExpander(
        nfo=NetForceOptimizer(min_quality=args.min_quality, fast_mode=True),
        target_size=args.num_grasps,
        # Scale delta_pos with object size so perturbations are proportional
        delta_pos=spec.size * 0.12,
        delta_max=spec.size * 0.60,
        seed=args.seed + seed_offset,
    )
    graph = expander.expand(filtered_set)

    # Isaac Lab spawn spec for this object
    isaac_spec = {
        "name": spec.name,
        "shape_type": spec.shape_type,
        "size": spec.size,
        "mass": spec.mass,
        "color": spec.color,
    }

    return graph, isaac_spec


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config file and resolve num_fingers
    # Priority: CLI --num_fingers > config hand.num_fingers > default 4
    # ------------------------------------------------------------------
    cfg_file = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg_file = yaml.safe_load(f) or {}
        print(f"[Stage 0] Config: {cfg_path}")
    else:
        print(f"[Stage 0] Config not found at {cfg_path}, using defaults.")

    hand_cfg = cfg_file.get("hand", {})
    if args.num_fingers is not None:
        num_fingers = args.num_fingers   # CLI overrides everything
    else:
        num_fingers = hand_cfg.get("num_fingers", 4)

    print(f"[Stage 0] Hand:       {hand_cfg.get('name', 'unknown')}  "
          f"(num_fingers={num_fingers})")

    # ------------------------------------------------------------------
    # Build object pool
    # ------------------------------------------------------------------
    if args.mesh_path:
        import trimesh
        from grasp_generation.grasp_sampler import ObjectSpec
        mesh = trimesh.load(args.mesh_path, force="mesh")
        size = float(max(mesh.bounding_box.extents))
        pool = ObjectPool([ObjectSpec(
            name=Path(args.mesh_path).stem,
            mesh=mesh,
            shape_type="custom",
            size=size,
        )])
    elif args.mesh_dir:
        pool = ObjectPool.from_mesh_dir(args.mesh_dir)
    else:
        pool = ObjectPool.from_config(
            shape_types=args.shapes,
            size_range=(args.size_min, args.size_max),
            num_sizes=args.num_sizes,
            seed=args.seed,
        )

    print(f"\n[Stage 0] Processing {len(pool)} objects in pool")

    # ------------------------------------------------------------------
    # Generate grasps for each object
    # ------------------------------------------------------------------
    multi_graph = MultiObjectGraspGraph()
    success_count = 0

    for i, spec in enumerate(pool):
        graph, isaac_spec = process_one_object(spec, args, seed_offset=i * 100,
                                               num_fingers=num_fingers)
        if graph is not None:
            multi_graph.add(graph, isaac_spec)
            success_count += 1

    if success_count == 0:
        print("\nERROR: No objects produced valid grasps. "
              "Try --fast_nfo or lower --min_quality.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    graph_path = output_dir / "grasp_graph.pkl"
    multi_graph.save(graph_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f" Stage 0 Complete")
    print(f"{'='*55}")
    multi_graph.summary()
    print(f"\n  Saved: {graph_path}")
    print(f"\nNext step:")
    print(f"  python scripts/train_rl.py --grasp_graph {graph_path}")


if __name__ == "__main__":
    main()
