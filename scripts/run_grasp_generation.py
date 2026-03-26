"""
Stage 0 – Grasp Set Generation
================================
Generates a diverse set of quality grasps for the target object
and saves a GraspGraph for use in Stage 1 RL training.

Pipeline:
  1. Sample candidate grasps on the object surface (GraspSampler)
  2. Score and filter by NFO quality (NetForceOptimizer)
  3. Expand with RRT to reach target_size (RRTGraspExpander)
  4. Save grasp_graph.pkl

Usage:
    python scripts/run_grasp_generation.py --object cube --num_grasps 500
    python scripts/run_grasp_generation.py --mesh_path assets/mug.obj --num_grasps 1000
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grasp_generation import GraspSampler, NetForceOptimizer, RRTGraspExpander
from grasp_generation.grasp_sampler import make_default_object_mesh, sample_grasps_for_mesh


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 0: Grasp Generation")
    p.add_argument("--object", default="cube",
                   choices=["cube", "sphere", "cylinder"],
                   help="Default object type (used when --mesh_path is not given)")
    p.add_argument("--object_size", type=float, default=0.06,
                   help="Object bounding box size in metres (default: 6 cm)")
    p.add_argument("--mesh_path", type=str, default=None,
                   help="Path to object mesh file (.obj, .stl, .ply)")
    p.add_argument("--num_seed_grasps", type=int, default=300,
                   help="Number of seed grasps to sample before filtering")
    p.add_argument("--num_grasps", type=int, default=500,
                   help="Target number of grasps in final GraspGraph")
    p.add_argument("--min_quality", type=float, default=0.005,
                   help="Minimum NFO quality score to keep a grasp")
    p.add_argument("--mu", type=float, default=0.5,
                   help="Friction coefficient for NFO")
    p.add_argument("--fast_nfo", action="store_true",
                   help="Use fast SVD approximation instead of LP for NFO")
    p.add_argument("--output_dir", type=str, default="data",
                   help="Directory to save grasp_set.pkl and grasp_graph.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load or create mesh
    # ------------------------------------------------------------------
    if args.mesh_path:
        print(f"[Stage 0] Loading mesh from {args.mesh_path}")
        import trimesh
        mesh = trimesh.load(args.mesh_path, force="mesh")
        object_name = Path(args.mesh_path).stem
    else:
        print(f"[Stage 0] Using default object: {args.object} (size={args.object_size}m)")
        mesh = make_default_object_mesh(args.object, args.object_size)
        object_name = args.object

    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # ------------------------------------------------------------------
    # Step 2: Sample seed grasps
    # ------------------------------------------------------------------
    print(f"\n[Stage 0] Step 1/3: Sampling {args.num_seed_grasps} seed grasps...")
    sampler = GraspSampler(
        mesh=mesh,
        object_name=object_name,
        num_candidates=args.num_seed_grasps * 20,
        num_grasps=args.num_seed_grasps,
        seed=args.seed,
    )
    seed_set = sampler.sample()
    print(f"  Sampled {len(seed_set)} seed grasps")

    # ------------------------------------------------------------------
    # Step 3: Score and filter by NFO quality
    # ------------------------------------------------------------------
    print(f"\n[Stage 0] Step 2/3: Scoring with NFO (mu={args.mu}, fast={args.fast_nfo})...")
    nfo = NetForceOptimizer(
        mu=args.mu,
        min_quality=args.min_quality,
        fast_mode=args.fast_nfo,
    )
    filtered_set = nfo.evaluate_set(seed_set, verbose=True)

    if len(filtered_set) == 0:
        print("ERROR: No grasps passed quality filter. "
              "Try lowering --min_quality or increasing --num_seed_grasps.")
        sys.exit(1)

    # Save filtered seed set
    seed_path = output_dir / "grasp_set.pkl"
    filtered_set.save(seed_path)

    # ------------------------------------------------------------------
    # Step 4: RRT expansion
    # ------------------------------------------------------------------
    print(f"\n[Stage 0] Step 3/3: RRT expansion → target {args.num_grasps} grasps...")
    expander = RRTGraspExpander(
        nfo=NetForceOptimizer(min_quality=args.min_quality, fast_mode=True),
        target_size=args.num_grasps,
        seed=args.seed,
    )
    grasp_graph = expander.expand(filtered_set)

    # Save graph
    graph_path = output_dir / "grasp_graph.pkl"
    grasp_graph.save(graph_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Stage 0 Complete ===")
    print(f"  Grasp set  : {seed_path}  ({len(filtered_set)} grasps)")
    print(f"  Grasp graph: {graph_path}  "
          f"({len(grasp_graph)} nodes, {grasp_graph.num_edges} edges)")
    print(f"\nNext: train RL policy")
    print(f"  python scripts/train_rl.py --grasp_graph {graph_path}")


if __name__ == "__main__":
    main()
