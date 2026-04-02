"""
Stage 0 – Grasp Generation
============================
Generates grasp sets via DexGraspNet SA optimization.

Pipeline:
  1. DexGraspNet optimization (pure PyTorch, full GPU)
  2. NFO quality filtering
  3. Graph construction (kNN edges)
  4. Output: data/grasp_graph.pkl (ready for RL training)

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

    # 22 DOF → 24 DOF: remap DexGraspNet MJCF joints to Isaac Sim USD layout.
    #
    # DexGraspNet MJCF (22 DOF, shadow_hand_wrist_free.xml):
    #   [0-3]   FF: FFJ3, FFJ2, FFJ1, FFJ0(coupled DIP)
    #   [4-7]   MF: MFJ3, MFJ2, MFJ1, MFJ0(coupled DIP)
    #   [8-11]  RF: RFJ3, RFJ2, RFJ1, RFJ0(coupled DIP)
    #   [12-16] LF: LFJ4, LFJ3, LFJ2, LFJ1, LFJ0(coupled DIP)
    #   [17-21] TH: THJ4, THJ3, THJ2, THJ1, THJ0
    #
    # Isaac Sim USD (24 DOF):
    #   [0-1]   WR: WRJ1, WRJ0          (wrist, zeroed)
    #   [2-5]   FF: FFJ4(passive), FFJ3, FFJ2, FFJ1
    #   [6-9]   MF: MFJ4(passive), MFJ3, MFJ2, MFJ1
    #   [10-13] RF: RFJ4(passive), RFJ3, RFJ2, RFJ1
    #   [14-18] LF: LFJ5(passive), LFJ4, LFJ3, LFJ2, LFJ1
    #   [19-23] TH: THJ4, THJ3, THJ2, THJ1, THJ0
    #
    # Key differences:
    #   - MJCF has J0 (coupled DIP, driven by tendon) → not in Isaac (skip)
    #   - Isaac has J4/J5 (passive spread) → not in MJCF (zero)
    #   - Thumb: identical 5 joints in both (THJ4-THJ0)
    for g in grasp_set.grasps:
        if g.joint_angles is not None and len(g.joint_angles) == 22:
            q24 = np.zeros(num_dof, dtype=np.float32)
            # FF: MJCF[0:3]=(FFJ3,2,1) → Isaac[3:6], skip MJCF[3]=FFJ0
            q24[3:6] = g.joint_angles[0:3]
            # MF: MJCF[4:7]=(MFJ3,2,1) → Isaac[7:10], skip MJCF[7]=MFJ0
            q24[7:10] = g.joint_angles[4:7]
            # RF: MJCF[8:11]=(RFJ3,2,1) → Isaac[11:14], skip MJCF[11]=RFJ0
            q24[11:14] = g.joint_angles[8:11]
            # LF: MJCF[12:16]=(LFJ4,3,2,1) → Isaac[15:19], skip MJCF[16]=LFJ0
            q24[15:19] = g.joint_angles[12:16]
            # TH: MJCF[17:22]=(THJ4,3,2,1,0) → Isaac[19:24] (direct match)
            q24[19:24] = g.joint_angles[17:22]
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

    output_path = output_dir / "grasp_graph.pkl"
    multi_graph.save(output_path)
    multi_graph.summary()

    print(f"\n{'='*55}")
    print(f" Complete")
    print(f"   Output: {output_path}")
    print(f"{'='*55}")
    print(f"\nNext: /workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py "
          f"--grasp_graph {output_path} --num_envs 512 --headless")


if __name__ == "__main__":
    main()
