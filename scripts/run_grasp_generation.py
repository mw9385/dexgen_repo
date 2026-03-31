"""
Stage 0 – Grasp Set Generation (DexGraspNet-based)
====================================================
Generates diverse grasp sets using DexGraspNet's Simulated Annealing
optimization. No Isaac Sim dependency — runs on pure PyTorch.

Pipeline (per object):
  1. Build Shadow Hand model (MJCF FK via pytorch_kinematics)
  2. Build object model (analytical SDF for primitives)
  3. Initialize hand poses around object (convex hull approach)
  4. Optimize via Simulated Annealing + RMSProp (DexGraspNet energy)
  5. Filter by energy thresholds (E_fc, E_dis, E_pen)
  6. Build GraspGraph for RL training

Usage:
    python scripts/run_grasp_generation.py
    python scripts/run_grasp_generation.py --shapes cube sphere --num_grasps 300
    python scripts/run_grasp_generation.py --batch_size 256 --num_iterations 3000
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Auto-initialize DexGraspNet submodule if missing
# ---------------------------------------------------------------------------
_DEXGRASPNET_DIR = Path(__file__).parent.parent / "third_party" / "DexGraspNet"
_DEXGRASPNET_MJCF = _DEXGRASPNET_DIR / "grasp_generation" / "mjcf" / "shadow_hand_wrist_free.xml"

if not _DEXGRASPNET_MJCF.exists():
    print("[Stage 0] DexGraspNet submodule not initialized. Running git submodule update...")
    try:
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "third_party/DexGraspNet"],
            cwd=str(Path(__file__).parent.parent),
        )
        print("[Stage 0] DexGraspNet submodule initialized successfully.")
    except Exception as e:
        print(f"[Stage 0] WARNING: Failed to initialize DexGraspNet submodule: {e}")
        print("  Run manually: git submodule update --init third_party/DexGraspNet")

from grasp_generation import (
    GraspSampler, NetForceOptimizer, RRTGraspExpander,
    ObjectPool, MultiObjectGraspGraph,
    GraspOptimizer, build_hand_model, build_object_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 0: Grasp Generation (no Isaac Sim)")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["cube", "sphere", "cylinder"])
    p.add_argument("--size_min", type=float, default=None)
    p.add_argument("--size_max", type=float, default=None)
    p.add_argument("--num_sizes", type=int, default=None)
    p.add_argument("--mesh_path", type=str, default=None)
    p.add_argument("--mesh_dir", type=str, default=None)

    # Grasp generation
    p.add_argument("--num_grasps", type=int, default=None,
                   help="Target grasps per object (default: 300)")
    p.add_argument("--num_seed_grasps", type=int, default=None)
    p.add_argument("--min_quality", type=float, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--fast_nfo", action=argparse.BooleanOptionalAction, default=None)

    # Method
    p.add_argument("--method", type=str, default="optimization",
                   choices=["optimization", "surface_sampling"])

    # Optimization params
    p.add_argument("--num_iterations", type=int, default=None,
                   help="SA iterations per batch (default: 2000)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Parallel grasp candidates per batch (default: 256)")
    p.add_argument("--n_contact", type=int, default=None,
                   help="Contact points per grasp (default: 4)")

    # Finger config
    p.add_argument("--num_fingers", type=int, default=None)
    p.add_argument("--finger_counts", type=str, default=None)
    p.add_argument("--max_num_fingers", type=int, default=None)

    # Isaac refinement (runs automatically after generation)
    p.add_argument("--no-isaac_refine", action="store_true", default=False,
                   help="Skip Isaac Sim refinement (not recommended)")
    p.add_argument("--refine_batch_envs", type=int, default=16,
                   help="Batch size for Isaac refinement environments")

    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "grasp_generation.yaml"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def _resolve_finger_counts(args, hand_cfg):
    if args.finger_counts:
        return sorted(set(int(x.strip()) for x in args.finger_counts.split(",")))
    elif args.max_num_fingers is not None:
        return list(range(2, int(args.max_num_fingers) + 1))
    else:
        nf = int(args.num_fingers) if args.num_fingers else int(hand_cfg.get("num_fingers", 4))
        return [nf]


def process_one_object(spec, args, num_fingers, num_dof, hand_name, device):
    """Generate grasps for one object using DexGraspNet optimization."""
    from grasp_generation.rrt_expansion import build_graph_from_grasps

    print(f"\n{'='*55}")
    print(f"  Object:      {spec.name}  (shape={spec.shape_type}, size={spec.size:.3f}m)")
    print(f"  num_fingers: {num_fingers}  |  method: optimization")
    print(f"{'='*55}")

    # Build hand model (reuse across objects for speed)
    hand_model = build_hand_model(hand_name, device=device)

    # Read config
    opt_cfg = getattr(args, '_opt_cfg', {})

    optimizer = GraspOptimizer(
        hand_model=hand_model,
        mesh=spec.mesh,
        shape_type=spec.shape_type,
        size=spec.size,
        w_dis=opt_cfg.get("w_dis", 100.0),
        w_pen=opt_cfg.get("w_pen", 100.0),
        w_spen=opt_cfg.get("w_spen", 10.0),
        w_joints=opt_cfg.get("w_joints", 1.0),
        w_pose=opt_cfg.get("w_pose", 10.0),
        n_iter=args.num_iterations,
        batch_size=args.batch_size,
        n_contact=args.n_contact,
        step_size=opt_cfg.get("step_size", 0.005),
        starting_temperature=opt_cfg.get("starting_temperature", 18.0),
        temperature_decay=opt_cfg.get("temperature_decay", 0.95),
        thres_fc=opt_cfg.get("thres_fc", 0.3),
        thres_dis=opt_cfg.get("thres_dis", 0.005),
        thres_pen=opt_cfg.get("thres_pen", 0.02),
        device=device,
    )

    grasp_set = optimizer.optimize(num_grasps=args.num_grasps, verbose=True)

    if len(grasp_set) < 2:
        print(f"  [!] Only {len(grasp_set)} grasps for {spec.name}, skipping.")
        return None, None

    # Tag object info
    for g in grasp_set.grasps:
        g.object_name = spec.name
        g.object_scale = spec.size / 0.06

    # Expand 22-DOF DexGraspNet joints to 24-DOF Isaac Lab format
    _expand_dexgraspnet_joints(grasp_set, num_dof)

    # NFO post-filter
    nfo = NetForceOptimizer(mu=args.mu, min_quality=args.min_quality, fast_mode=True)
    grasp_set = nfo.evaluate_set(grasp_set, verbose=True)

    if len(grasp_set) < 2:
        print(f"  [!] Only {len(grasp_set)} grasps passed NFO filter, skipping.")
        return None, None

    # Build graph
    graph = build_graph_from_grasps(
        grasp_set.grasps,
        object_name=spec.name,
        delta_max=spec.size * 0.30,
        num_fingers=num_fingers,
    )

    n_with_joints = sum(1 for g in graph.grasp_set.grasps if g.joint_angles is not None)
    print(f"  [Done] {len(graph)} grasps, {graph.num_edges} edges, "
          f"{n_with_joints} with joints")

    isaac_spec = {
        "name": spec.name,
        "shape_type": spec.shape_type,
        "size": spec.size,
        "mass": spec.mass,
        "color": spec.color,
    }
    return graph, isaac_spec


def _expand_dexgraspnet_joints(grasp_set, num_dof=24):
    """Expand 22-DOF DexGraspNet joints to 24-DOF Isaac Lab (insert 2 wrist zeros)."""
    for grasp in grasp_set.grasps:
        if grasp.joint_angles is None:
            continue
        q22 = grasp.joint_angles
        if len(q22) >= num_dof:
            continue
        q24 = np.zeros(num_dof, dtype=np.float32)
        if len(q22) == 22:
            q24[2:] = q22  # WRJ1=0, WRJ0=0 at front
        else:
            q24[:len(q22)] = q22
        grasp.joint_angles = q24


def validate_graph(graph_path):
    """Validate saved graph for RL readiness."""
    import pickle
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph, GraspGraph

    print(f"\n{'='*55}")
    print(f"  Validating {graph_path}")
    print(f"{'='*55}")

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    graphs = graph.graphs if isinstance(graph, MultiObjectGraspGraph) else {"single": graph}

    all_ok = True
    for obj_name, g in graphs.items():
        N, E = len(g), g.num_edges
        n_with_joints = sum(1 for gr in g.grasp_set.grasps if gr.joint_angles is not None)
        ok = N >= 2 and E >= 1 and n_with_joints == N
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {obj_name}: {N} nodes, {E} edges, {n_with_joints}/{N} with joints")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\n  Graph validation PASSED")
    else:
        print(f"\n  Graph validation FAILED")
    return all_ok


def main():
    args = parse_args()

    # Load config
    cfg_file = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg_file = yaml.safe_load(f) or {}
        print(f"[Stage 0] Config: {cfg_path}")

    hand_cfg = cfg_file.get("hand", {})
    object_pool_cfg = cfg_file.get("object_pool", {})
    sampler_cfg = cfg_file.get("sampler", {})
    nfo_cfg = cfg_file.get("nfo", {})
    rrt_cfg = cfg_file.get("rrt", {})
    opt_cfg = cfg_file.get("optimization", {})
    args._opt_cfg = opt_cfg

    # Resolve params (CLI > YAML > default)
    args.shapes = args.shapes or object_pool_cfg.get("shapes", ["cube", "sphere", "cylinder"])
    args.size_min = float(args.size_min if args.size_min is not None else object_pool_cfg.get("size_min", 0.05))
    args.size_max = float(args.size_max if args.size_max is not None else object_pool_cfg.get("size_max", 0.08))
    args.num_sizes = int(args.num_sizes if args.num_sizes is not None else object_pool_cfg.get("num_sizes", 3))
    args.num_grasps = int(args.num_grasps if args.num_grasps is not None else rrt_cfg.get("target_size", 300))
    args.num_seed_grasps = int(args.num_seed_grasps if args.num_seed_grasps is not None else sampler_cfg.get("num_seed_grasps", 300))
    args.min_quality = float(args.min_quality if args.min_quality is not None else nfo_cfg.get("min_quality", 0.005))
    args.mu = float(args.mu if args.mu is not None else nfo_cfg.get("mu", 0.5))
    if args.fast_nfo is None:
        args.fast_nfo = bool(nfo_cfg.get("fast_mode", False))
    args.output_dir = args.output_dir or cfg_file.get("output_dir", "data")

    # Optimization params
    args.num_iterations = int(args.num_iterations if args.num_iterations is not None else opt_cfg.get("num_iterations", 2000))
    args.batch_size = int(args.batch_size if args.batch_size is not None else opt_cfg.get("batch_size", 256))
    args.n_contact = int(args.n_contact if args.n_contact is not None else opt_cfg.get("n_contact", 4))

    if args.method == "optimization" and "method" in cfg_file:
        args.method = cfg_file["method"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    finger_counts = _resolve_finger_counts(args, hand_cfg)
    num_dof = hand_cfg.get("num_dof", 24)
    hand_name = hand_cfg.get("name", "shadow")

    # Print config
    print(f"[Stage 0] Hand:       {hand_name}  (finger_counts={finger_counts}, num_dof={num_dof})")
    print(f"[Stage 0] Object pool: shapes={args.shapes}, "
          f"size_range=({args.size_min:.3f}, {args.size_max:.3f}), num_sizes={args.num_sizes}")
    print(f"[Stage 0] Method:     {args.method} (NO Isaac Sim — full GPU for optimization)")
    print(f"[Stage 0] Optimizer:  iterations={args.num_iterations}, "
          f"batch_size={args.batch_size}, n_contact={args.n_contact}")
    print(f"[Stage 0] Device:     {args.device}")

    # Build object pool
    if args.mesh_path:
        import trimesh
        from grasp_generation.grasp_sampler import ObjectSpec
        mesh = trimesh.load(args.mesh_path, force="mesh")
        size = float(max(mesh.bounding_box.extents))
        pool = ObjectPool([ObjectSpec(name=Path(args.mesh_path).stem, mesh=mesh,
                                      shape_type="custom", size=size)])
    elif args.mesh_dir:
        pool = ObjectPool.from_mesh_dir(args.mesh_dir)
    else:
        pool = ObjectPool.from_config(
            shape_types=args.shapes,
            size_range=(args.size_min, args.size_max),
            num_sizes=args.num_sizes,
            seed=args.seed,
        )

    print(f"\n[Stage 0] Processing {len(pool)} objects x {finger_counts} finger configs")

    # Generate grasps
    multi_graph = MultiObjectGraspGraph()
    success_count = 0

    for i, spec in enumerate(pool):
        for nf in finger_counts:
            graph, isaac_spec = process_one_object(
                spec, args, num_fingers=nf, num_dof=num_dof,
                hand_name=hand_name, device=args.device,
            )
            if graph is None:
                continue

            graph.object_name = f"{spec.name}_f{nf}"
            isaac_spec_tagged = dict(isaac_spec)
            isaac_spec_tagged["name"] = graph.object_name
            isaac_spec_tagged["num_fingers"] = nf

            multi_graph.add(graph, isaac_spec_tagged)
            success_count += 1

            # Checkpoint
            graph_path = output_dir / "grasp_graph.pkl"
            multi_graph.save(graph_path)
            print(f"  [checkpoint] {success_count} graph(s) saved -> {graph_path}")

    if success_count == 0:
        print("\nERROR: No objects produced valid grasps.")
        sys.exit(1)

    # Final save
    graph_path = output_dir / "grasp_graph.pkl"
    multi_graph.save(graph_path)
    validate_graph(graph_path)

    print(f"\n{'='*55}")
    print(f" Stage 0 Grasp Generation Complete")
    print(f"{'='*55}")
    multi_graph.summary()
    print(f"\n  Saved: {graph_path}")

    # ── Isaac Refinement (default: ON) ──────────────────────────────────────
    # Corrects FK mismatch between pytorch_kinematics and Isaac Sim.
    # Without this, grasps may not align perfectly in simulation.
    skip_refine = getattr(args, 'no_isaac_refine', False)
    if not skip_refine:
        print(f"\n{'='*55}")
        print(f" Stage 0.5: Isaac Refinement")
        print(f"{'='*55}")
        refine_script = Path(__file__).parent / "refine_grasps.py"
        isaaclab_sh = Path("/workspace/IsaacLab/isaaclab.sh")

        # Determine python executable
        if isaaclab_sh.exists():
            cmd = [str(isaaclab_sh), "-p", str(refine_script),
                   "--grasp_graph", str(graph_path),
                   "--batch_envs", str(args.refine_batch_envs),
                   "--headless"]
        else:
            cmd = [sys.executable, str(refine_script),
                   "--grasp_graph", str(graph_path),
                   "--batch_envs", str(args.refine_batch_envs),
                   "--headless"]

        print(f"  Running: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            print(f"\n  Isaac refinement completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n  WARNING: Isaac refinement failed (exit code {e.returncode}).")
            print(f"  Grasps are saved without refinement. You can retry manually:")
            print(f"  /workspace/IsaacLab/isaaclab.sh -p scripts/refine_grasps.py "
                  f"--grasp_graph {graph_path} --headless")
        except FileNotFoundError:
            print(f"\n  WARNING: Isaac Sim not found. Skipping refinement.")
            print(f"  Run manually when available:")
            print(f"  /workspace/IsaacLab/isaaclab.sh -p scripts/refine_grasps.py "
                  f"--grasp_graph {graph_path} --headless")
    else:
        print(f"\n  [INFO] Isaac refinement skipped (--no-isaac_refine).")
        print(f"  Grasps use pytorch_kinematics FK — may differ slightly from Isaac Sim.")

    print(f"\nNext step:")
    print(f"  /workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py "
          f"--grasp_graph {graph_path} --num_envs 512 --headless")


if __name__ == "__main__":
    main()
