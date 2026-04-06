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
    GraspSampler, SurfaceRRTGraspExpander,
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


def generate_grasps_surface_rrt(spec, args, num_fingers, seed):
    """Generate grasps for one object via surface-projected RRT."""
    srrt_cfg = getattr(args, '_srrt_cfg', {})

    print(f"\n{'='*55}")
    print(f"  Object: {spec.name} ({spec.shape_type}, {spec.size:.3f}m)")
    print(f"  Method: surface_rrt")
    print(f"{'='*55}")

    # 1. Sample seed grasps on mesh surface
    num_seeds = int(srrt_cfg.get("num_seed_grasps", 500))
    sampler = GraspSampler(
        mesh=spec.mesh,
        object_name=spec.name,
        object_scale=spec.size / 0.06,
        num_grasps=num_seeds,
        num_candidates=max(5000, num_seeds * 10),
        num_fingers=num_fingers,
        seed=seed,
    )
    seed_set = sampler.sample()
    if len(seed_set) < 2:
        print(f"  [SurfaceRRT] Too few seeds ({len(seed_set)}), skipping")
        return None, None

    # 2. NFO pre-filter on seeds
    nfo = NetForceOptimizer(
        mu=args.mu, min_quality=args.min_quality, fast_mode=True,
    )
    seed_set = nfo.evaluate_set(seed_set, verbose=True)
    if len(seed_set) < 2:
        print(f"  [SurfaceRRT] Too few seeds after NFO ({len(seed_set)}), skipping")
        return None, None

    # 3. RRT expansion with surface projection
    # [수정] delta_pos가 과도하게 커지는 논리적 오류 수정 (Contact manifold 이탈 방지)
    obj_size = float(np.max(spec.mesh.bounding_box.extents))
    base_delta = float(srrt_cfg.get("delta_pos", 0.002)) 
    delta_pos = min(base_delta, obj_size * 0.05) # 물체 크기의 5%를 상한선으로 제한

    expander = SurfaceRRTGraspExpander(
        mesh=spec.mesh,
        nfo=nfo,
        delta_pos=delta_pos,
        delta_max=float(srrt_cfg.get("delta_max", 0.04)),
        min_quality=args.min_quality,
        target_size=args.num_grasps,
        max_attempts_per_step=int(srrt_cfg.get("max_attempts_per_step", 30)),
        collision_threshold=float(srrt_cfg.get("collision_threshold", 0.002)),
        min_finger_spacing=float(srrt_cfg.get("min_finger_spacing", 0.01)),
        seed=seed,
    )
    graph = expander.expand(seed_set)
    print(f"  [Done] {len(graph)} grasps, {graph.num_edges} edges")

    return graph, {
        "name": spec.name, "shape_type": spec.shape_type,
        "size": spec.size, "mass": spec.mass, "color": spec.color,
    }


def generate_grasps(spec, args, num_fingers, num_dof, hand_name, device):
    """Generate grasps for one object via DexGraspNet optimization."""
    from grasp_generation.rrt_expansion import build_graph_from_grasps

    print(f"\n{'='*55}")
    print(f"  Object: {spec.name} ({spec.shape_type}, {spec.size:.3f}m)")
    print(f"  Method: optimization")
    print(f"{'='*55}")

    hand_model = build_hand_model(hand_name, device=device)
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
        n_iter=args.num_iterations,
        batch_size=args.batch_size,
        n_contact=args.n_contact,
        step_size=opt_cfg.get("step_size", 0.005),
        starting_temperature=opt_cfg.get("starting_temperature", 18.0),
        temperature_decay=opt_cfg.get("temperature_decay", 0.95),
        thres_fc=opt_cfg.get("thres_fc", 0.3),
        thres_dis=opt_cfg.get("thres_dis", 0.005),
        thres_pen=opt_cfg.get("thres_pen", 0.001),
        device=device,
    )

    grasp_set = optimizer.optimize(num_grasps=args.num_grasps, verbose=True)
    if len(grasp_set) < 2:
        return None, None

    for g in grasp_set.grasps:
        g.object_name = spec.name
        g.object_scale = spec.size / 0.06

    for g in grasp_set.grasps:
        if g.joint_angles is not None and len(g.joint_angles) == 22:
            q24 = np.zeros(num_dof, dtype=np.float32)
            q24[3:6] = g.joint_angles[0:3]
            q24[7:10] = g.joint_angles[4:7]
            q24[11:14] = g.joint_angles[8:11]
            q24[15:19] = g.joint_angles[12:16]
            q24[19:24] = g.joint_angles[17:22]
            g.joint_angles = q24

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

    cfg_file = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg_file = yaml.safe_load(f) or {}

    hand_cfg = cfg_file.get("hand", {})
    opt_cfg = cfg_file.get("optimization", {})
    args._opt_cfg = opt_cfg

    pool_cfg = cfg_file.get("object_pool", {})
    nfo_cfg = cfg_file.get("nfo", {})
    args.shapes = args.shapes or pool_cfg.get("shapes", ["cube", "sphere", "cylinder"])
    args.size_min = float(args.size_min or pool_cfg.get("size_min", 0.05))
    args.size_max = float(args.size_max or pool_cfg.get("size_max", 0.08))
    args.num_sizes = int(args.num_sizes or pool_cfg.get("num_sizes", 3))
    args.num_grasps = int(args.num_grasps or cfg_file.get("rrt", {}).get("target_size", 300))
    args.min_quality = float(args.min_quality or nfo_cfg.get("min_quality", 0.005))
    args.mu = float(args.mu or nfo_cfg.get("mu", 0.5))
    args.num_iterations = int(args.num_iterations or opt_cfg.get("num_iterations", 6000))
    args.batch_size = int(args.batch_size or opt_cfg.get("batch_size", 500))
    args.n_contact = int(args.n_contact or opt_cfg.get("n_contact", 4))
    args.output_dir = args.output_dir or cfg_file.get("output_dir", "data")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_fingers = int(args.num_fingers or hand_cfg.get("num_fingers", 5))
    num_dof = hand_cfg.get("num_dof", 24)
    hand_name = hand_cfg.get("name", "shadow")

    method = cfg_file.get("method", "optimization")
    srrt_cfg = cfg_file.get("surface_rrt", {})
    args._srrt_cfg = srrt_cfg

    print(f"[Stage 0] Method: {method}")
    print(f"[Stage 0] Hand: {hand_name} (fingers={num_fingers}, dof={num_dof})")
    print(f"[Stage 0] Objects: {args.shapes}, size=({args.size_min:.3f}, {args.size_max:.3f})")
    if method == "optimization":
        print(f"[Stage 0] Optimizer: iter={args.num_iterations}, batch={args.batch_size}")
    print(f"[Stage 0] Device: {args.device}")

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

    multi_graph = MultiObjectGraspGraph()
    for i, spec in enumerate(pool):
        if method == "surface_rrt":
            if args.num_grasps is None or args.num_grasps == 300:
                args.num_grasps = int(srrt_cfg.get("target_size", 300))
            graph, isaac_spec = generate_grasps_surface_rrt(
                spec, args, num_fingers=num_fingers, seed=args.seed + i,
            )
        else:
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
