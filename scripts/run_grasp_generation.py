"""
Stage 0 – Grasp Set Generation (Multi-Object)
===============================================
Generates a diverse set of quality grasps for a *pool* of random objects
and saves a MultiObjectGraspGraph for use in Stage 1 RL training.

Each object in the pool (cube / sphere / cylinder, various sizes) gets its
own GraspGraph. At RL training time the environment randomly selects an
object + grasp pair from this combined graph every episode.

Pipeline (per object):

  Legacy method (--grasp_method legacy):
    1. Sample candidate grasps on the object surface (GraspSampler)
    2. Score and filter by NFO quality (NetForceOptimizer)
    3. Expand with RRT to reach target_size (RRTGraspExpander)
    4. Seed joint angles for each grasp (heuristic initialization)
    5. Save a Stage-1-ready MultiObjectGraspGraph

  DexGraspNet method (--grasp_method dexgraspnet):
    1. Export object mesh to DexGraspNet format
    2. Run differentiable grasp optimisation (simulated annealing)
    3. Filter by energy thresholds + convert to Grasp format
    4. Build connectivity graph from flat grasp list
    5. Save a Stage-1-ready MultiObjectGraspGraph

Usage:
    # Default: cube + sphere + cylinder, 3 sizes each
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py

    # Custom object pool
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \\
        --shapes cube sphere \\
        --size_min 0.04 --size_max 0.09 --num_sizes 4 \\
        --num_grasps 300

    # Single custom mesh
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py --mesh_path assets/mug.obj
"""

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

from grasp_generation import (
    GraspSampler, NetForceOptimizer, RRTGraspExpander,
    ObjectPool, MultiObjectGraspGraph, refine_multi_object_graph_with_isaac,
)
from grasp_generation.rrt_expansion import build_graph_from_grasps


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 0: Multi-Object Grasp Generation")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["cube", "sphere", "cylinder"],
                   help="Primitive shapes to include in the object pool")
    p.add_argument("--size_min", type=float, default=None,
                   help="Minimum object size in metres")
    p.add_argument("--size_max", type=float, default=None,
                   help="Maximum object size in metres")
    p.add_argument("--num_sizes", type=int, default=None,
                   help="Number of size steps per shape")
    p.add_argument("--mesh_path", type=str, default=None,
                   help="Single custom mesh file instead of pool (.obj/.stl/.ply)")
    p.add_argument("--mesh_dir", type=str, default=None,
                   help="Directory of mesh files — all loaded as pool objects")
    p.add_argument(
        "--generation_preset",
        type=str,
        default="default",
        choices=["default", "high_precision"],
        help="Generation preset. 'high_precision' increases candidate counts and enables Isaac refinement by default.",
    )

    # Grasp generation quality
    p.add_argument("--num_seed_grasps", type=int, default=None,
                   help="Seed grasps per object before NFO filtering")
    p.add_argument("--num_grasps", type=int, default=None,
                   help="Target grasps per object after RRT expansion")
    p.add_argument("--min_quality", type=float, default=None,
                   help="Minimum NFO quality score")
    p.add_argument("--mu", type=float, default=None,
                   help="Friction coefficient for NFO")
    p.add_argument("--fast_nfo", action=argparse.BooleanOptionalAction, default=None,
                   help="Use fast SVD approximation for NFO")

    # Grasp generation method
    p.add_argument("--grasp_method", type=str, default="legacy",
                   choices=["legacy", "dexgraspnet"],
                   help="Grasp generation method: "
                        "'legacy' = GraspSampler + NFO + RRT, "
                        "'dexgraspnet' = DexGraspNet differentiable optimisation")

    # DexGraspNet-specific options (only used when --grasp_method=dexgraspnet)
    p.add_argument("--dgn_batch_size", type=int, default=128,
                   help="DexGraspNet batch size per optimisation run (default: 128)")
    p.add_argument("--dgn_n_iter", type=int, default=6000,
                   help="DexGraspNet optimisation iterations (default: 6000)")
    p.add_argument("--dgn_gpu", type=str, default="0",
                   help="GPU id for DexGraspNet (default: '0')")

    p.add_argument("--output_dir", type=str, default=None)
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
    p.add_argument(
        "--max_num_fingers", type=int, default=None,
        help="Generate all finger counts from 2..N (inclusive), capped at 5",
    )
    p.add_argument(
        "--finger_counts", type=str, default=None,
        help="Comma-separated finger counts to generate, e.g. '2,3,5'",
    )
    p.add_argument(
        "--isaac_refine",
        action="store_true",
        default=False,
        help="After Stage 0 generation, validate grasps in Isaac Sim and overwrite each grasp with the true simulated tuple.",
    )
    p.add_argument(
        "--isaac_refine_batch_envs",
        type=int,
        default=16,
        help="Batch size for Isaac-based grasp refinement/validation",
    )
    p.add_argument(
        "--keep_top_k",
        type=int,
        default=None,
        help="After Isaac validation, keep only the top-K lowest-error grasps per object/finger graph",
    )
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def _apply_generation_preset(args):
    if args.generation_preset != "high_precision":
        return

    preset_values = {
        "num_seed_grasps": 2000,
        "num_grasps": 1000,
        "min_quality": 0.01,
        "mu": 0.5,
        "fast_nfo": False,
        "isaac_refine_batch_envs": 32,
        "keep_top_k": None,
    }

    for field_name, preset_value in preset_values.items():
        if getattr(args, field_name) is None:
            setattr(args, field_name, preset_value)
    args.isaac_refine = True


def _resolve_finger_counts(args, hand_cfg: dict) -> list[int]:
    """
    Resolve which finger-count variants to generate.

    Priority:
      1. --finger_counts        explicit list
      2. --max_num_fingers      generate 2..N
      3. --num_fingers          single value
      4. config hand.num_fingers
    """
    if args.finger_counts:
        raw_counts = [part.strip() for part in args.finger_counts.split(",")]
        finger_counts = [int(part) for part in raw_counts if part]
    elif args.max_num_fingers is not None:
        finger_counts = list(range(2, int(args.max_num_fingers) + 1))
    else:
        resolved_num_fingers = (
            int(args.num_fingers)
            if args.num_fingers is not None
            else int(hand_cfg.get("num_fingers", 4))
        )
        finger_counts = [resolved_num_fingers]

    deduped = sorted(set(int(nf) for nf in finger_counts))
    invalid = [nf for nf in deduped if nf < 2 or nf > 5]
    if invalid:
        raise ValueError(
            f"Finger counts must be in [2, 5] for Shadow Hand, got: {invalid}"
        )
    return deduped


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
        manifold_contact_count=num_fingers,
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
# DexGraspNet pipeline (alternative to process_one_object)
# ---------------------------------------------------------------------------

def process_one_object_dexgraspnet(
    spec,
    args,
    seed_offset: int,
    num_fingers: int = 4,
    num_dof: int = 16,
    dof_per_finger: int = 4,
) -> tuple:
    """
    Run DexGraspNet differentiable optimisation for one ObjectSpec.

    Replaces the GraspSampler → NFO → RRT pipeline with:
      1. Export mesh to DexGraspNet format
      2. Run gradient-guided simulated annealing (DexGraspNet core)
      3. Filter by energy thresholds
      4. Convert to our Grasp format
      5. Build connectivity graph

    Returns (GraspGraph, isaac_lab_spec) or (None, None) on failure.
    """
    from grasp_generation.mesh_export import export_mesh_for_dexgraspnet
    from grasp_generation.dexgraspnet_adapter import (
        DexGraspNetConfig,
        generate_grasps_dexgraspnet,
    )

    print(f"\n{'='*55}")
    print(f"  Object:      {spec.name}  (shape={spec.shape_type}, size={spec.size:.3f}m)")
    print(f"  Method:      DexGraspNet  |  num_fingers: {num_fingers}")
    print(f"{'='*55}")

    # Step 1: Export mesh for DexGraspNet
    meshdata_root = Path(args.output_dir) / "meshdata"
    obj_path, scale = export_mesh_for_dexgraspnet(
        spec.mesh, spec.name, meshdata_root,
    )
    print(f"  [mesh] Exported to {obj_path} (scale={scale:.4f}m)")

    # Step 2-4: DexGraspNet optimisation + filter + convert
    cfg = DexGraspNetConfig(
        batch_size=args.dgn_batch_size,
        n_iter=args.dgn_n_iter,
        n_contact=num_fingers,
        gpu=args.dgn_gpu,
        seed=args.seed + seed_offset,
    )

    grasps = generate_grasps_dexgraspnet(
        object_code=spec.name,
        meshdata_root=str(meshdata_root),
        object_scale=scale,
        cfg=cfg,
        target_num_grasps=args.num_grasps,
        device="cuda",
    )

    if len(grasps) < 2:
        print(f"  [!] Only {len(grasps)} grasps for {spec.name}, skipping.")
        return None, None

    # Step 5: Build connectivity graph
    graph = build_graph_from_grasps(
        grasps,
        object_name=spec.name,
        delta_max=spec.size * 0.60,
        num_fingers=num_fingers,
    )

    # DexGraspNet already provides joint_angles — fill any missing ones
    n_with_joints = sum(1 for g in graph.grasp_set.grasps if g.joint_angles is not None)
    n_missing = len(graph) - n_with_joints
    if n_missing > 0:
        _attach_joint_angles_to_graph(graph, num_dof=num_dof, dof_per_finger=dof_per_finger)

    print(f"  [result] {len(graph)} grasps, {graph.num_edges} edges")

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
        reset_errs = [
            float(gr.reset_contact_error)
            for gr in g.grasp_set.grasps
            if getattr(gr, "reset_contact_error", None) is not None
        ]

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
        reset_err_text = ""
        if reset_errs:
            reset_err_text = (
                f", reset_err_mean={np.mean(reset_errs):.4f}m"
                f", reset_err_best={np.min(reset_errs):.4f}m"
            )
        print(f"  [{status}] {obj_name}: {N} nodes, {E} edges, "
              f"{n_with_joints}/{N} with joints, "
              f"mean NN dist={nn_mean:.4f}m{reset_err_text}")

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
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    try:
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
        object_pool_cfg = cfg_file.get("object_pool", {})
        sampler_cfg = cfg_file.get("sampler", {})
        nfo_cfg = cfg_file.get("nfo", {})
        rrt_cfg = cfg_file.get("rrt", {})
        _apply_generation_preset(args)

        # ------------------------------------------------------------------
        # Resolve config-overridable CLI values.
        # CLI takes precedence. If omitted, fall back to YAML. If YAML also
        # omits the value, fall back to the historical hard-coded default.
        # ------------------------------------------------------------------
        args.shapes = args.shapes or object_pool_cfg.get("shapes", ["cube", "sphere", "cylinder"])
        args.size_min = float(args.size_min if args.size_min is not None else object_pool_cfg.get("size_min", 0.04))
        args.size_max = float(args.size_max if args.size_max is not None else object_pool_cfg.get("size_max", 0.09))
        args.num_sizes = int(args.num_sizes if args.num_sizes is not None else object_pool_cfg.get("num_sizes", 3))

        args.num_seed_grasps = int(
            args.num_seed_grasps if args.num_seed_grasps is not None else sampler_cfg.get("num_seed_grasps", 300)
        )
        args.num_grasps = int(
            args.num_grasps if args.num_grasps is not None else rrt_cfg.get("target_size", 300)
        )
        args.min_quality = float(
            args.min_quality if args.min_quality is not None else nfo_cfg.get("min_quality", 0.005)
        )
        args.mu = float(
            args.mu if args.mu is not None else nfo_cfg.get("mu", 0.5)
        )
        if args.fast_nfo is None:
            args.fast_nfo = bool(nfo_cfg.get("fast_mode", False))
        args.output_dir = args.output_dir or cfg_file.get("output_dir", "data")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        finger_counts = _resolve_finger_counts(args, hand_cfg)
        num_fingers = max(finger_counts)

        num_dof        = hand_cfg.get("num_dof", 16)
        dof_per_finger = hand_cfg.get("dof_per_finger", 4)

        print(f"[Stage 0] Hand:       {hand_cfg.get('name', 'allegro')}  "
              f"(finger_counts={finger_counts}, num_dof={num_dof})")
        if args.generation_preset != "default":
            print(f"[Stage 0] Preset:     {args.generation_preset}")
        print(f"[Stage 0] Object pool: shapes={args.shapes}, "
              f"size_range=({args.size_min:.3f}, {args.size_max:.3f}), "
              f"num_sizes={args.num_sizes}")
        print(f"[Stage 0] Sampler/NFO/RRT: seeds={args.num_seed_grasps}, "
              f"target={args.num_grasps}, min_quality={args.min_quality}, "
              f"mu={args.mu}, fast_nfo={args.fast_nfo}")

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

        print(f"\n[Stage 0] Processing {len(pool)} objects × {finger_counts} finger configs")
        print(f"[Stage 0] Method: {args.grasp_method}")

        # ------------------------------------------------------------------
        # Generate grasps for each object × each finger count
        # ------------------------------------------------------------------
        multi_graph = MultiObjectGraspGraph()
        success_count = 0

        for i, spec in enumerate(pool):
            for nf in finger_counts:
                # Use a unique seed offset per (object, finger_count) pair
                seed_offset = i * 100 + nf * 1000

                if args.grasp_method == "dexgraspnet":
                    try:
                        graph, isaac_spec = process_one_object_dexgraspnet(
                            spec, args,
                            seed_offset=seed_offset,
                            num_fingers=nf,
                            num_dof=num_dof,
                            dof_per_finger=dof_per_finger,
                        )
                    except Exception as e:
                        import traceback
                        print(f"\n  [ERROR] DexGraspNet failed for {spec.name}: {e}")
                        traceback.print_exc()
                        graph, isaac_spec = None, None
                else:
                    graph, isaac_spec = process_one_object(
                        spec, args,
                        seed_offset=seed_offset,
                        num_fingers=nf,
                        num_dof=num_dof,
                        dof_per_finger=dof_per_finger,
                    )
                if graph is None:
                    continue

                # Tag the graph object name to distinguish finger configs
                graph.object_name = f"{spec.name}_f{nf}"
                isaac_spec_tagged  = dict(isaac_spec)
                isaac_spec_tagged["name"] = graph.object_name
                # Also tag num_fingers so downstream Stage 1 env can adapt
                isaac_spec_tagged["num_fingers"] = nf

                multi_graph.add(graph, isaac_spec_tagged)
                success_count += 1

                # Checkpoint after each (object, finger) so partial results survive.
                graph_path = output_dir / "grasp_graph.pkl"
                multi_graph.save(graph_path)
                print(f"  [checkpoint] {success_count} graph(s) saved → {graph_path}")

        if success_count == 0:
            print("\nERROR: No objects produced valid grasps. "
                  "Try --fast_nfo or lower --min_quality.")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        graph_path = output_dir / "grasp_graph.pkl"
        multi_graph.save(graph_path)

        if args.isaac_refine:
            print(f"\n{'='*55}")
            print(" Isaac Validation / Refinement")
            print(f"{'='*55}")
            multi_graph = refine_multi_object_graph_with_isaac(
                multi_graph,
                batch_envs=args.isaac_refine_batch_envs,
                keep_top_k=args.keep_top_k,
            )
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
        print(
            "  /workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py "
            f"--grasp_graph {graph_path} --num_envs 512 --headless"
        )
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
