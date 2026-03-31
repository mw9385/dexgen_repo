"""
Stage 0 – Grasp Generation + Isaac Refinement
===============================================
Generates grasp sets via DexGraspNet SA optimization, then refines
them in Isaac Sim so all stored data (joint_angles, object_pos_hand,
object_quat_hand) is consistent with Isaac Sim's FK.

Pipeline:
  1. DexGraspNet optimization (pure PyTorch, full GPU)
  2. Isaac Sim refinement (differential IK, overwrites grasp data)
  3. Output: data/grasp_graph.pkl (ready for RL training)

Usage:
    /isaac-sim/python.sh scripts/run_grasp_generation.py
    /isaac-sim/python.sh scripts/run_grasp_generation.py --shapes cube --num_sizes 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Auto-initialize DexGraspNet submodule
_DEXGRASPNET_DIR = Path(__file__).parent.parent / "third_party" / "DexGraspNet"
_DEXGRASPNET_MJCF = _DEXGRASPNET_DIR / "grasp_generation" / "mjcf" / "shadow_hand_wrist_free.xml"

if not _DEXGRASPNET_MJCF.exists():
    print("[Stage 0] Initializing DexGraspNet submodule...")
    try:
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "third_party/DexGraspNet"],
            cwd=str(Path(__file__).parent.parent),
        )
    except Exception as e:
        print(f"  WARNING: {e}")

from grasp_generation import (
    NetForceOptimizer, ObjectPool, MultiObjectGraspGraph,
    GraspOptimizer, build_hand_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 0: Grasp Generation")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["cube", "sphere", "cylinder"])
    p.add_argument("--size_min", type=float, default=None)
    p.add_argument("--size_max", type=float, default=None)
    p.add_argument("--num_sizes", type=int, default=None)
    p.add_argument("--mesh_path", type=str, default=None)
    p.add_argument("--mesh_dir", type=str, default=None)

    # Grasp generation
    p.add_argument("--num_grasps", type=int, default=None)
    p.add_argument("--min_quality", type=float, default=None)
    p.add_argument("--mu", type=float, default=None)

    # Optimization
    p.add_argument("--num_iterations", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_contact", type=int, default=None)

    # Finger config
    p.add_argument("--num_fingers", type=int, default=None)

    # Refinement
    p.add_argument("--refine_batch_envs", type=int, default=16)

    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "grasp_generation.yaml"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def generate_grasps(spec, args, num_fingers, num_dof, hand_name, device):
    """Generate grasps for one object via DexGraspNet optimization."""
    from grasp_generation.rrt_expansion import build_graph_from_grasps

    print(f"\n{'='*55}")
    print(f"  Object: {spec.name} ({spec.shape_type}, {spec.size:.3f}m)")
    print(f"{'='*55}")

    hand_model = build_hand_model(hand_name, device=device)
    opt_cfg = getattr(args, '_opt_cfg', {})

    optimizer = GraspOptimizer(
        hand_model=hand_model,
        mesh=spec.mesh,
        shape_type=spec.shape_type,
        size=spec.size,
        w_dis=opt_cfg.get("w_dis", 500.0),
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
        thres_fc=opt_cfg.get("thres_fc", 3.0),
        thres_dis=opt_cfg.get("thres_dis", 0.005),
        thres_pen=opt_cfg.get("thres_pen", 0.03),
        device=device,
    )

    grasp_set = optimizer.optimize(num_grasps=args.num_grasps, verbose=True)
    if len(grasp_set) < 2:
        return None, None

    for g in grasp_set.grasps:
        g.object_name = spec.name
        g.object_scale = spec.size / 0.06

    # 22 DOF → 24 DOF (insert wrist zeros)
    for g in grasp_set.grasps:
        if g.joint_angles is not None and len(g.joint_angles) == 22:
            q24 = np.zeros(num_dof, dtype=np.float32)
            q24[2:] = g.joint_angles
            g.joint_angles = q24

    # NFO post-filter
    nfo = NetForceOptimizer(mu=args.mu, min_quality=args.min_quality, fast_mode=True)
    grasp_set = nfo.evaluate_set(grasp_set, verbose=True)
    if len(grasp_set) < 2:
        return None, None

    graph = build_graph_from_grasps(
        grasp_set.grasps, object_name=spec.name,
        delta_max=spec.size * 0.30, num_fingers=num_fingers,
    )
    print(f"  [Done] {len(graph)} grasps, {graph.num_edges} edges")

    return graph, {
        "name": spec.name, "shape_type": spec.shape_type,
        "size": spec.size, "mass": spec.mass, "color": spec.color,
    }


def run_refinement(graph_path, batch_envs):
    """Run Isaac Sim refinement as subprocess."""
    print(f"\n{'='*55}")
    print(f" Isaac Refinement")
    print(f"{'='*55}")

    refine_script = Path(__file__).parent / "refine_grasps.py"
    isaaclab_sh = Path("/workspace/IsaacLab/isaaclab.sh")

    if isaaclab_sh.exists():
        cmd = [str(isaaclab_sh), "-p", str(refine_script),
               "--grasp_graph", str(graph_path),
               "--batch_envs", str(batch_envs), "--headless"]
    else:
        cmd = [sys.executable, str(refine_script),
               "--grasp_graph", str(graph_path),
               "--batch_envs", str(batch_envs), "--headless"]

    print(f"  {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"  Refinement complete. Grasp data updated in {graph_path}")


def main():
    args = parse_args()

    # Load config
    cfg_file = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg_file = yaml.safe_load(f) or {}

    hand_cfg = cfg_file.get("hand", {})
    opt_cfg = cfg_file.get("optimization", {})
    args._opt_cfg = opt_cfg

    # Resolve params
    pool_cfg = cfg_file.get("object_pool", {})
    nfo_cfg = cfg_file.get("nfo", {})
    args.shapes = args.shapes or pool_cfg.get("shapes", ["cube", "sphere", "cylinder"])
    args.size_min = float(args.size_min or pool_cfg.get("size_min", 0.05))
    args.size_max = float(args.size_max or pool_cfg.get("size_max", 0.08))
    args.num_sizes = int(args.num_sizes or pool_cfg.get("num_sizes", 3))
    args.num_grasps = int(args.num_grasps or cfg_file.get("rrt", {}).get("target_size", 300))
    args.min_quality = float(args.min_quality or nfo_cfg.get("min_quality", 0.005))
    args.mu = float(args.mu or nfo_cfg.get("mu", 0.5))
    args.num_iterations = int(args.num_iterations or opt_cfg.get("num_iterations", 5000))
    args.batch_size = int(args.batch_size or opt_cfg.get("batch_size", 512))
    args.n_contact = int(args.n_contact or opt_cfg.get("n_contact", 4))
    args.output_dir = args.output_dir or cfg_file.get("output_dir", "data")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_fingers = int(args.num_fingers or hand_cfg.get("num_fingers", 5))
    num_dof = hand_cfg.get("num_dof", 24)
    hand_name = hand_cfg.get("name", "shadow")

    print(f"[Stage 0] Hand: {hand_name} (fingers={num_fingers}, dof={num_dof})")
    print(f"[Stage 0] Objects: {args.shapes}, size=({args.size_min:.3f}, {args.size_max:.3f})")
    print(f"[Stage 0] Optimizer: iter={args.num_iterations}, batch={args.batch_size}")
    print(f"[Stage 0] Device: {args.device}")

    # Build object pool
    if args.mesh_path:
        import trimesh
        from grasp_generation.grasp_sampler import ObjectSpec
        mesh = trimesh.load(args.mesh_path, force="mesh")
        pool = ObjectPool([ObjectSpec(name=Path(args.mesh_path).stem, mesh=mesh,
                                      shape_type="custom", size=float(max(mesh.bounding_box.extents)))])
    elif args.mesh_dir:
        pool = ObjectPool.from_mesh_dir(args.mesh_dir)
    else:
        pool = ObjectPool.from_config(
            shape_types=args.shapes,
            size_range=(args.size_min, args.size_max),
            num_sizes=args.num_sizes, seed=args.seed,
        )

    # ── Step 1: Generate grasps (pure PyTorch) ──
    multi_graph = MultiObjectGraspGraph()
    for i, spec in enumerate(pool):
        graph, isaac_spec = generate_grasps(
            spec, args, num_fingers=num_fingers, num_dof=num_dof,
            hand_name=hand_name, device=args.device,
        )
        if graph is None:
            continue

        graph.object_name = f"{spec.name}_f{num_fingers}"
        isaac_spec["name"] = graph.object_name
        isaac_spec["num_fingers"] = num_fingers
        multi_graph.add(graph, isaac_spec)

    if len(multi_graph) == 0:
        print("\nERROR: No valid grasps generated.")
        sys.exit(1)

    graph_path = output_dir / "grasp_graph.pkl"
    multi_graph.save(graph_path)
    multi_graph.summary()

    # ── Step 2: Isaac Sim refinement (overwrites grasp data) ──
    try:
        run_refinement(graph_path, args.refine_batch_envs)
    except Exception as e:
        print(f"\n  WARNING: Refinement failed: {e}")
        print(f"  Retry: /workspace/IsaacLab/isaaclab.sh -p scripts/refine_grasps.py "
              f"--grasp_graph {graph_path} --headless")

    print(f"\n{'='*55}")
    print(f" Complete: {graph_path}")
    print(f"{'='*55}")
    print(f"\nNext: /workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py "
          f"--grasp_graph {graph_path} --num_envs 512 --headless")


if __name__ == "__main__":
    main()
