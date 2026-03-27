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
  4. Solve approximate IK for each grasp → store joint_angles in Grasp
  5. Merge all per-object graphs → MultiObjectGraspGraph

IK approach (heuristic, no full robot model needed):
  For each grasp, we compute a per-finger joint angle vector by mapping
  the fingertip distance from the object centroid to a joint opening scale.
  This is the same heuristic used in events.py as a fallback, but here we
  pre-compute and store it so the RL reset is deterministic and consistent.

  For higher accuracy, replace _solve_ik_for_grasp() with a proper IK
  solver (e.g. pink, pinocchio, or Isaac Lab's built-in IK).

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

import numpy as np
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


# ---------------------------------------------------------------------------
# Heuristic IK: fingertip positions → joint angles
# ---------------------------------------------------------------------------

def _solve_ik_for_grasp(
    fingertip_positions: np.ndarray,   # (num_fingers, 3) in object frame
    num_dof: int = 16,
    dof_per_finger: int = 4,
    q_low: float = 0.0,
    q_high: float = 1.57,
) -> np.ndarray:
    """
    Heuristic IK: map fingertip distance from object centroid to joint angles.

    This is a pre-computation of the same heuristic used in events.py
    _set_robot_to_fingertip_config(). By storing it in the Grasp object,
    the RL reset is deterministic and consistent across episodes.

    Mapping:
      dist from object center → how "open" the hand is
      [0.02 m, 0.12 m] → joint scale [0.1, 0.9]
      joint_angle = q_low + scale * (q_high - q_low)

    For Allegro Hand (16 DOF, 4 fingers × 4 joints):
      - Finger order: index(0-3), middle(4-7), ring(8-11), thumb(12-15)
      - All joints in a finger get the same scale (simplified)

    Args:
        fingertip_positions: (num_fingers, 3) in object frame
        num_dof:             total robot DOF (16 for Allegro)
        dof_per_finger:      joints per finger (4 for Allegro)
        q_low:               lower joint limit (rad)
        q_high:              upper joint limit (rad)

    Returns:
        joint_angles: (num_dof,) float32 array
    """
    num_fingers = fingertip_positions.shape[0]
    joint_angles = np.zeros(num_dof, dtype=np.float32)

    # Distance from object centroid (origin in object frame) to each fingertip
    ft_dist = np.linalg.norm(fingertip_positions, axis=-1)  # (num_fingers,)

    # Map [0.02, 0.12] → scale [0.1, 0.9]
    scale = np.clip((ft_dist - 0.02) / 0.10, 0.0, 1.0) * 0.8 + 0.1  # (num_fingers,)

    for f in range(min(num_fingers, num_dof // dof_per_finger)):
        s = float(scale[f])
        start = f * dof_per_finger
        end   = start + dof_per_finger
        joint_angles[start:end] = q_low + s * (q_high - q_low)

    return joint_angles


def _attach_joint_angles_to_graph(graph, num_dof: int = 16, dof_per_finger: int = 4):
    """
    Solve heuristic IK for every grasp in a GraspGraph and store the result
    in grasp.joint_angles.

    This is called after RRT expansion so all nodes (including RRT-generated
    ones) get joint angles.
    """
    count = 0
    for grasp in graph.grasp_set.grasps:
        if grasp.joint_angles is None:
            grasp.joint_angles = _solve_ik_for_grasp(
                grasp.fingertip_positions,
                num_dof=num_dof,
                dof_per_finger=dof_per_finger,
            )
            count += 1
    print(f"  [IK] Solved heuristic IK for {count} grasps in '{graph.object_name}'")


# ---------------------------------------------------------------------------
# Per-object pipeline
# ---------------------------------------------------------------------------

def process_one_object(
    spec,
    args,
    seed_offset: int,
    num_fingers: int = 4,
    num_dof: int = 16,
    dof_per_finger: int = 4,
) -> tuple:
    """
    Run the full grasp generation pipeline for one ObjectSpec.
    Returns (GraspGraph, isaac_lab_spec) or (None, None) on failure.
    """
    from grasp_generation.grasp_sampler import GraspSampler

    print(f"\n{'='*55}")
    print(f"  Object:      {spec.name}  (shape={spec.shape_type}, size={spec.size:.3f}m)")
    print(f"  num_fingers: {num_fingers}  |  num_dof: {num_dof}")
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

    # Step 4: Solve heuristic IK for all grasps and store joint_angles
    _attach_joint_angles_to_graph(graph, num_dof=num_dof, dof_per_finger=dof_per_finger)

    # Verify: check that joint_angles are stored
    n_with_joints = sum(1 for g in graph.grasp_set.grasps if g.joint_angles is not None)
    print(f"  [IK] {n_with_joints}/{len(graph)} grasps have joint_angles stored")

    # Isaac Lab spawn spec for this object
    isaac_spec = {
        "name": spec.name,
        "shape_type": spec.shape_type,
        "size": spec.size,
        "mass": spec.mass,
        "color": spec.color,
    }

    return graph, isaac_spec


# ---------------------------------------------------------------------------
# Validation: verify grasp_graph.pkl is usable by RL
# ---------------------------------------------------------------------------

def validate_graph(graph_path: Path):
    """
    Load the saved graph and run basic sanity checks:
      - At least 2 nodes per object
      - At least 1 edge per object
      - joint_angles stored in all grasps
      - Nearest neighbor distance is reasonable (< delta_max)
    """
    import pickle
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph, GraspGraph

    print(f"\n{'='*55}")
    print(f"  Validating {graph_path}")
    print(f"{'='*55}")

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    if isinstance(graph, GraspGraph):
        graphs = {"single": graph}
    else:
        graphs = graph.graphs

    all_ok = True
    for obj_name, g in graphs.items():
        N = len(g)
        E = g.num_edges
        n_with_joints = sum(1 for gr in g.grasp_set.grasps if gr.joint_angles is not None)

        # Nearest neighbor distance check
        all_fps = g.grasp_set.as_array()  # (N, F*3)
        nn_dists = []
        for i in range(min(N, 50)):  # sample 50 grasps
            dists = np.linalg.norm(all_fps - all_fps[i], axis=-1)
            dists[i] = np.inf
            nn_dists.append(dists.min())
        nn_mean = float(np.mean(nn_dists)) if nn_dists else 0.0

        ok = N >= 2 and E >= 1 and n_with_joints == N
        status = "✓" if ok else "✗"
        print(f"  [{status}] {obj_name}: {N} nodes, {E} edges, "
              f"{n_with_joints}/{N} with joints, "
              f"mean NN dist={nn_mean:.4f}m")

        if not ok:
            all_ok = False
            if N < 2:
                print(f"      → ERROR: need at least 2 grasps for RL (start + goal)")
            if E < 1:
                print(f"      → ERROR: no edges — increase --num_grasps or decrease delta_max")
            if n_with_joints < N:
                print(f"      → WARNING: {N - n_with_joints} grasps missing joint_angles")

    if all_ok:
        print(f"\n  ✓ Graph validation PASSED — ready for RL training")
    else:
        print(f"\n  ✗ Graph validation FAILED — fix issues above before training")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config file and resolve num_fingers / num_dof
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
        num_fingers = args.num_fingers
    else:
        num_fingers = hand_cfg.get("num_fingers", 4)

    num_dof        = hand_cfg.get("num_dof", 16)
    dof_per_finger = hand_cfg.get("dof_per_finger", 4)

    print(f"[Stage 0] Hand:       {hand_cfg.get('name', 'allegro')}  "
          f"(num_fingers={num_fingers}, num_dof={num_dof})")

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
        graph, isaac_spec = process_one_object(
            spec, args,
            seed_offset=i * 100,
            num_fingers=num_fingers,
            num_dof=num_dof,
            dof_per_finger=dof_per_finger,
        )
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
    # Validate the saved graph
    # ------------------------------------------------------------------
    validate_graph(graph_path)

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
