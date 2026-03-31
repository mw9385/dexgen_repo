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
  4. Seed joint angles for each grasp (heuristic initialization)
  5. Save a Stage-1-ready MultiObjectGraspGraph

Usage:
    # Default: cube + sphere + cylinder, 3 sizes each
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py

    # Custom object pool
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \\
        --shapes cube sphere \\
        --size_min 0.03 --size_max 0.08 --num_sizes 3 \\
        --num_grasps 300

    # Single custom mesh
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py --mesh_path assets/mug.obj
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

from grasp_generation import (
    GraspSampler, NetForceOptimizer, RRTGraspExpander,
    ObjectPool, MultiObjectGraspGraph, refine_multi_object_graph_with_isaac,
    GraspOptimizer, build_hand_model, build_object_model,
)


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
    p.add_argument(
        "--method", type=str, default="optimization",
        choices=["optimization", "surface_sampling"],
        help=(
            "Grasp generation method. "
            "'optimization': DexGraspNet-style differentiable optimization (recommended). "
            "'surface_sampling': legacy surface sampling + NFO filtering."
        ),
    )
    # Optimization-specific params
    p.add_argument("--opt_iterations", type=int, default=None,
                   help="Number of optimization iterations per batch (default: 200)")
    p.add_argument("--opt_batch_size", type=int, default=None,
                   help="Batch size for parallel grasp optimization (default: 256)")
    p.add_argument("--opt_lr", type=float, default=None,
                   help="Learning rate for grasp optimizer (default: 0.005)")
    p.add_argument("--sdf_resolution", type=int, default=None,
                   help="SDF voxel grid resolution (default: 64)")

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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After Stage 0, validate grasps in Isaac Sim and overwrite each grasp "
            "with the true simulated (joint_angles, object_pos_hand, object_quat_hand). "
            "DEFAULT: True — without this, object_pos_hand is None and the RL reset "
            "cannot reproduce the correct hand-object relative pose; the hand and object "
            "will be placed at random positions each episode.  "
            "Use --no-isaac_refine only for quick smoke-tests."
        ),
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
        "num_seed_grasps": 8000,   # many surface samples → top-K selection replaces RRT
        "num_grasps": 1000,         # keep top-1000 quality surface grasps
        "min_quality": 0.005,
        "mu": 0.5,
        "fast_nfo": False,
        "isaac_refine_batch_envs": 32,
        # After Isaac simulation, keep only the 300 grasps with lowest contact
        # error. This filters out grasps where the simulated hand and object
        # don't actually align, which would cause the object to fall immediately.
        "keep_top_k": 300,
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
    num_dof: int = 24,
    hand_name: str = "shadow",
) -> np.ndarray:
    """
    Heuristic IK: fingertip distance → joint angles.

    Supported hands
    ---------------
    shadow (24-DOF USD layout, Isaac Lab Shadow Hand E-Series):
      Indices in the 24-joint articulation array:
        [0]     WRJ1   (wrist, always 0)
        [1]     WRJ0   (wrist, always 0)
        [2]     FFJ4   (passive spread, always 0)
        [3-5]   FFJ3, FFJ2, FFJ1   (distal→proximal flexion)
        [6]     MFJ4   (passive spread, always 0)
        [7-9]   MFJ3, MFJ2, MFJ1
        [10]    RFJ4   (passive spread, always 0)
        [11-13] RFJ3, RFJ2, RFJ1
        [14]    LFJ5   (passive coupling, always 0)
        [15]    LFJ4   (spread, always 0)
        [16-18] LFJ3, LFJ2, LFJ1
        [19]    THJ4   (abduction)
        [20-23] THJ3, THJ2, THJ1, THJ0  (flexion, distal→proximal)
      NOTE: THJ5 does NOT exist in the Shadow Hand USD (unlike some models).
      The 4 passive joints (FFJ4, MFJ4, RFJ4, LFJ5) are kept at 0.

    allegro / generic (16 DOF, 4 fingers × 4 joints):
      All 4 joints per finger = scale × 1.57

    The scale is derived from fingertip distance from the object centroid:
      [0.02 m, 0.12 m] → scale [0.1, 0.9]

    Returns
    -------
    joint_angles: (num_dof,) float32 array
    """
    num_fingers = fingertip_positions.shape[0]
    joint_angles = np.zeros(num_dof, dtype=np.float32)

    # Distance from object centroid (origin in object frame) to each fingertip
    ft_dist = np.linalg.norm(fingertip_positions, axis=-1)  # (num_fingers,)

    # [0.02 m, 0.12 m] → scale [0.1, 0.9]
    scale = np.clip((ft_dist - 0.02) / 0.10, 0.0, 1.0) * 0.8 + 0.1  # (num_fingers,)

    if hand_name == "shadow" or num_dof in (22, 24):
        # ── Shadow Hand 24-DOF USD layout ────────────────────────────────────
        # Per-finger block layout in the 24-DOF joint array:
        #   WR:  [0-1]   = WRJ1, WRJ0       (wrist; always 0)
        #   FF:  [2-5]   = FFJ4(passive), FFJ3, FFJ2, FFJ1
        #   MF:  [6-9]   = MFJ4(passive), MFJ3, MFJ2, MFJ1
        #   RF:  [10-13] = RFJ4(passive), RFJ3, RFJ2, RFJ1
        #   LF:  [14-18] = LFJ5(passive), LFJ4, LFJ3, LFJ2, LFJ1
        #   TH:  [19-23] = THJ4, THJ3, THJ2, THJ1, THJ0  (NO THJ5)
        #
        # Passive/spread joints (FFJ4=2, MFJ4=6, RFJ4=10, LFJ5=14, LFJ4=15)
        # stay at 0; only flexion joints are set from scale.

        # Finger start indices and per-finger sizes in the 24-DOF array:
        FINGER_STARTS = [2, 6, 10, 14, 19]  # FF, MF, RF, LF, TH
        FINGER_DOFS   = [4, 4,  4,  5,  5]

        if num_fingers >= 5:
            finger_to_block = [0, 1, 2, 3, 4]
        elif num_fingers == 4:
            # 4-finger: [FF, MF, RF, TH] → skip LF block (3)
            finger_to_block = [0, 1, 2, 4]
        elif num_fingers == 3:
            finger_to_block = [0, 1, 4]
        elif num_fingers == 2:
            finger_to_block = [0, 4]
        else:
            finger_to_block = list(range(num_fingers))

        for fi in range(min(num_fingers, len(finger_to_block))):
            s = float(scale[fi])
            block = finger_to_block[fi]
            start = FINGER_STARTS[block]
            ndof  = FINGER_DOFS[block]

            if block == 4:
                # Thumb: [THJ4, THJ3, THJ2, THJ1, THJ0] at indices [19-23]
                # NO THJ5 — Shadow Hand USD has 5 thumb joints starting at THJ4.
                if start + 5 <= num_dof:
                    joint_angles[start + 0] = s * 0.50     # THJ4 (abduction)
                    joint_angles[start + 1] = s * 1.00     # THJ3 (flexion)
                    joint_angles[start + 2] = s * 0.80     # THJ2 (flexion)
                    joint_angles[start + 3] = s * 0.60     # THJ1 (flexion)
                    joint_angles[start + 4] = s * 0.40     # THJ0 (flexion)
            elif block == 3:
                # LF: [LFJ5(passive), LFJ4, LFJ3, LFJ2, LFJ1] at [14-18]
                if start + 5 <= num_dof:
                    joint_angles[start + 0] = 0.0          # LFJ5 passive, always 0
                    joint_angles[start + 1] = 0.0          # LFJ4 spread, neutral
                    joint_angles[start + 2] = s * 0.80     # LFJ3 (distal)
                    joint_angles[start + 3] = s * 1.20     # LFJ2 (middle)
                    joint_angles[start + 4] = s * 1.50     # LFJ1 (proximal)
            else:
                # Regular finger (4 DOF: [passive-spread, J3, J2, J1])
                if start + 4 <= num_dof:
                    joint_angles[start + 0] = 0.0          # J4 passive spread, always 0
                    joint_angles[start + 1] = s * 0.80     # J3 distal flexion
                    joint_angles[start + 2] = s * 1.20     # J2 middle flexion
                    joint_angles[start + 3] = s * 1.50     # J1 proximal flexion

        # [0-1] WRJ1, WRJ0: always 0 (wrist is positioned by root pose, not joints)
    else:
        # ── Generic / Allegro fallback: uniform scale per finger ────────────
        dof_per_finger = max(1, num_dof // max(num_fingers, 1))
        for f in range(min(num_fingers, num_dof // dof_per_finger)):
            s = float(scale[f])
            start = f * dof_per_finger
            end   = start + dof_per_finger
            joint_angles[start:end] = s * 1.57

    return joint_angles


def _attach_joint_angles_to_graph(graph, num_dof: int = 24, dof_per_finger: int = 4,
                                   hand_name: str = "shadow"):
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
                hand_name=hand_name,
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
    num_dof: int = 24,
    dof_per_finger: int = 4,
    hand_name: str = "shadow",
) -> tuple:
    """
    Run the full grasp generation pipeline for one ObjectSpec.
    Returns (GraspGraph, isaac_lab_spec) or (None, None) on failure.

    Supports two methods:
      - 'optimization': DexGraspNet-style differentiable optimization (recommended)
      - 'surface_sampling': legacy surface sampling + NFO filtering
    """
    from grasp_generation.grasp_sampler import GraspSampler
    from grasp_generation.rrt_expansion import build_graph_from_grasps

    print(f"\n{'='*55}")
    print(f"  Object:      {spec.name}  (shape={spec.shape_type}, size={spec.size:.3f}m)")
    print(f"  num_fingers: {num_fingers}  |  num_dof: {num_dof}")
    print(f"  method:      {args.method}")
    print(f"{'='*55}")

    method = getattr(args, "method", "optimization")

    if method == "optimization":
        graph, isaac_spec = _process_one_object_optimization(
            spec, args, seed_offset, num_fingers, num_dof, dof_per_finger, hand_name,
        )
    else:
        graph, isaac_spec = _process_one_object_surface_sampling(
            spec, args, seed_offset, num_fingers, num_dof, dof_per_finger, hand_name,
        )

    return graph, isaac_spec


def _process_one_object_optimization(
    spec, args, seed_offset, num_fingers, num_dof, dof_per_finger, hand_name,
) -> tuple:
    """
    DexGraspNet-based grasp generation via Simulated Annealing.

    Pipeline:
      1. Build DexGraspNet hand model (MJCF FK via pytorch_kinematics)
      2. Build primitive object model (analytical SDF)
      3. Initialize hand poses around object (convex hull approach)
      4. Optimize via Simulated Annealing + RMSProp
      5. Filter by energy thresholds (E_fc, E_dis, E_pen)
      6. Build GraspGraph with edges between similar grasps
    """
    from grasp_generation.rrt_expansion import build_graph_from_grasps

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build DexGraspNet hand model
    hand_model = build_hand_model(hand_name, device=device)

    # Resolve optimization parameters
    opt_iterations = getattr(args, "opt_iterations", None) or 2000
    opt_batch_size = getattr(args, "opt_batch_size", None) or 128

    # Read energy weights from config
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
        n_iter=opt_iterations,
        batch_size=opt_batch_size,
        n_contact=min(num_fingers, 4),
        step_size=opt_cfg.get("step_size", 0.005),
        starting_temperature=opt_cfg.get("starting_temperature", 18.0),
        temperature_decay=opt_cfg.get("temperature_decay", 0.95),
        thres_fc=opt_cfg.get("thres_fc", 0.3),
        thres_dis=opt_cfg.get("thres_dis", 0.005),
        thres_pen=opt_cfg.get("thres_pen", 0.02),
        device=device,
    )

    # Run optimization
    grasp_set = optimizer.optimize(
        num_grasps=args.num_grasps,
        verbose=True,
    )

    if len(grasp_set) < 2:
        print(f"  [!] Only {len(grasp_set)} grasps from optimization for {spec.name}, skipping.")
        return None, None

    # Tag object info on grasps
    for g in grasp_set.grasps:
        g.object_name = spec.name
        g.object_scale = spec.size / 0.06

    # DexGraspNet's HandModel outputs 22 DOF (Shadow Hand active joints).
    # Isaac Lab expects 24 DOF (22 active + WRJ0, WRJ1 wrist).
    # Expand by inserting wrist joints (zeros) at the front.
    _expand_dexgraspnet_joints_to_isaac(grasp_set, num_dof)

    # NFO post-filter for additional quality assurance
    effective_min_quality = _effective_nfo_threshold(num_fingers, args.min_quality)
    nfo = NetForceOptimizer(mu=args.mu, min_quality=effective_min_quality,
                            fast_mode=True)
    grasp_set = nfo.evaluate_set(grasp_set, verbose=True)

    if len(grasp_set) < 2:
        print(f"  [!] Only {len(grasp_set)} grasps passed NFO post-filter, skipping.")
        return None, None

    # Build graph
    delta_max_base = spec.size * 0.30
    graph = build_graph_from_grasps(
        grasp_set.grasps,
        object_name=spec.name,
        delta_max=delta_max_base,
        num_fingers=num_fingers,
    )

    n_with_joints = sum(1 for g in graph.grasp_set.grasps if g.joint_angles is not None)
    print(f"  [Opt] {n_with_joints}/{len(graph)} grasps have joint_angles + object_pose stored")

    isaac_spec = {
        "name": spec.name,
        "shape_type": spec.shape_type,
        "size": spec.size,
        "mass": spec.mass,
        "color": spec.color,
    }

    return graph, isaac_spec


def _process_one_object_surface_sampling(
    spec, args, seed_offset, num_fingers, num_dof, dof_per_finger, hand_name,
) -> tuple:
    """Legacy surface sampling + NFO filtering pipeline."""
    from grasp_generation.grasp_sampler import GraspSampler

    # Step 1: Sample seed grasps
    sampler = GraspSampler(
        mesh=spec.mesh,
        object_name=spec.name,
        object_scale=spec.size / 0.06,
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
    effective_min_quality = _effective_nfo_threshold(num_fingers, args.min_quality)

    nfo = NetForceOptimizer(mu=args.mu, min_quality=effective_min_quality,
                            fast_mode=args.fast_nfo)
    filtered_set = nfo.evaluate_set(seed_set, verbose=True)

    if len(filtered_set) < 10:
        print(f"  [!] Only {len(filtered_set)} grasps passed NFO for {spec.name}, skipping.")
        return None, None

    # Step 3: Select top-K by quality
    if len(filtered_set) > args.num_grasps:
        filtered_set.grasps = sorted(
            filtered_set.grasps, key=lambda g: g.quality, reverse=True
        )[:args.num_grasps]
        print(f"  [Select] Kept top-{len(filtered_set)} surface grasps by quality")

    # Build graph
    delta_max_base = spec.size * 0.30
    delta_pos_base = delta_max_base / 3.0
    expander = RRTGraspExpander(
        nfo=NetForceOptimizer(min_quality=effective_min_quality, fast_mode=True),
        target_size=len(filtered_set),
        delta_pos=delta_pos_base,
        delta_max=delta_max_base,
        manifold_contact_count=num_fingers,
        seed=args.seed + seed_offset,
    )
    graph = expander.expand(filtered_set)

    # Step 4: Solve heuristic IK
    _attach_joint_angles_to_graph(graph, num_dof=num_dof, dof_per_finger=dof_per_finger,
                                  hand_name=hand_name)

    n_with_joints = sum(1 for g in graph.grasp_set.grasps if g.joint_angles is not None)
    print(f"  [IK] {n_with_joints}/{len(graph)} grasps have joint_angles stored")

    isaac_spec = {
        "name": spec.name,
        "shape_type": spec.shape_type,
        "size": spec.size,
        "mass": spec.mass,
        "color": spec.color,
    }

    return graph, isaac_spec


def _expand_dexgraspnet_joints_to_isaac(grasp_set, num_dof: int = 24):
    """
    Expand DexGraspNet's 22-DOF joint angles to Isaac Lab's 24-DOF.

    DexGraspNet Shadow Hand: 22 DOF (no wrist)
      FFJ3,FFJ2,FFJ1,FFJ0, MFJ3,MFJ2,MFJ1,MFJ0,
      RFJ3,RFJ2,RFJ1,RFJ0, LFJ4,LFJ3,LFJ2,LFJ1,LFJ0,
      THJ4,THJ3,THJ2,THJ1,THJ0

    Isaac Lab Shadow Hand: 24 DOF
      WRJ1,WRJ0, FFJ4,FFJ3,FFJ2,FFJ1, MFJ4,MFJ3,MFJ2,MFJ1,
      RFJ4,RFJ3,RFJ2,RFJ1, LFJ5,LFJ4,LFJ3,LFJ2,LFJ1,
      THJ4,THJ3,THJ2,THJ1,THJ0

    We insert WRJ1=0, WRJ0=0 at front, and FFJ4=0, MFJ4=0, RFJ4=0, LFJ5=0
    at appropriate positions.
    """
    for grasp in grasp_set.grasps:
        if grasp.joint_angles is None:
            continue
        q22 = grasp.joint_angles
        if len(q22) >= num_dof:
            continue  # already full size

        q24 = np.zeros(num_dof, dtype=np.float32)
        # Map 22-DOF DexGraspNet → 24-DOF Isaac Lab
        # DexGraspNet order: [FF(4), MF(4), RF(4), LF(5), TH(5)] = 22
        # Isaac Lab order:   [WR(2), FF(4), MF(4), RF(4), LF(5), TH(5)] = 24
        # Just insert 2 wrist zeros at the front
        if len(q22) == 22:
            q24[0] = 0.0   # WRJ1
            q24[1] = 0.0   # WRJ0
            q24[2:] = q22   # rest maps 1:1
        else:
            q24[:len(q22)] = q22
        grasp.joint_angles = q24


def _effective_nfo_threshold(num_fingers: int, min_quality: float) -> float:
    """Finger-count-aware NFO quality threshold."""
    if num_fingers <= 2:
        return 0.5
    elif num_fingers == 3:
        return max(min_quality * 0.5, 0.002)
    else:
        return min_quality


def _expand_optimizer_joints(
    grasp_set,
    hand_model,
    num_dof: int,
    num_fingers: int,
    hand_name: str,
):
    """
    Map optimizer joint angles (active finger DOF only) to the full
    num_dof vector expected by Isaac Lab.

    The optimizer's hand_model has num_dof = sum of active finger joints,
    which may differ from the full USD joint count (e.g., Shadow Hand
    has passive/spread joints not in the optimizer).
    """
    if hand_name == "shadow" or num_dof in (22, 24):
        # Shadow Hand mapping: optimizer joints → full 24-DOF vector
        # Optimizer finger order matches build_shadow_hand():
        #   num_fingers=5: [FF(3), MF(3), RF(3), LF(3), TH(4)] = 16 active
        #   num_fingers=4: [FF(3), MF(3), RF(3), TH(4)] = 13 active
        #   num_fingers=3: [FF(3), MF(3), TH(4)] = 10 active
        #   num_fingers=2: [FF(3), TH(4)] = 7 active

        # Finger block starts in the 24-DOF array:
        #   FF: [2-5] → active: [3,4,5] (skip FFJ4=2)
        #   MF: [6-9] → active: [7,8,9] (skip MFJ4=6)
        #   RF: [10-13] → active: [11,12,13] (skip RFJ4=10)
        #   LF: [14-18] → active: [16,17,18] (skip LFJ5=14, LFJ4=15)
        #   TH: [19-23] → active: [19,20,21,22] (THJ4,THJ3,THJ2,THJ1; skip THJ0=23)
        finger_to_full_indices = {
            5: {
                0: [3, 4, 5],        # FF
                1: [7, 8, 9],        # MF
                2: [11, 12, 13],     # RF
                3: [16, 17, 18],     # LF
                4: [19, 20, 21, 22], # TH
            },
            4: {
                0: [3, 4, 5],
                1: [7, 8, 9],
                2: [11, 12, 13],
                3: [19, 20, 21, 22],
            },
            3: {
                0: [3, 4, 5],
                1: [7, 8, 9],
                2: [19, 20, 21, 22],
            },
            2: {
                0: [3, 4, 5],
                1: [19, 20, 21, 22],
            },
        }

        mapping = finger_to_full_indices.get(num_fingers, {})

        for grasp in grasp_set.grasps:
            if grasp.joint_angles is None:
                continue
            opt_q = grasp.joint_angles
            full_q = np.zeros(num_dof, dtype=np.float32)

            offset = 0
            for fi in sorted(mapping.keys()):
                indices = mapping[fi]
                n_joints = len(indices)
                if offset + n_joints <= len(opt_q):
                    for j, idx in enumerate(indices):
                        if idx < num_dof:
                            full_q[idx] = opt_q[offset + j]
                offset += n_joints

            grasp.joint_angles = full_q
    else:
        # Generic / Allegro: optimizer DOF matches full DOF (all active)
        for grasp in grasp_set.grasps:
            if grasp.joint_angles is None:
                continue
            if len(grasp.joint_angles) < num_dof:
                full_q = np.zeros(num_dof, dtype=np.float32)
                full_q[:len(grasp.joint_angles)] = grasp.joint_angles
                grasp.joint_angles = full_q


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
        opt_cfg = cfg_file.get("optimization", {})
        _apply_generation_preset(args)

        # Store optimization energy weights for process_one_object
        args._opt_cfg = opt_cfg

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

        # Resolve optimization params from YAML if not set via CLI
        if args.opt_iterations is None:
            args.opt_iterations = int(opt_cfg.get("num_iterations", 200))
        if args.opt_batch_size is None:
            args.opt_batch_size = int(opt_cfg.get("batch_size", 256))
        if args.opt_lr is None:
            args.opt_lr = float(opt_cfg.get("lr", 0.005))
        if args.sdf_resolution is None:
            args.sdf_resolution = int(opt_cfg.get("sdf_resolution", 64))

        # Resolve method from YAML if CLI default
        if args.method == "optimization" and "method" in cfg_file:
            args.method = cfg_file["method"]

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
        print(f"[Stage 0] Method:     {args.method}")
        print(f"[Stage 0] Generation: target={args.num_grasps}, "
              f"min_quality={args.min_quality}, mu={args.mu}")
        if args.method == "optimization":
            print(f"[Stage 0] Optimizer:  iterations={getattr(args, 'opt_iterations', None) or 200}, "
                  f"batch_size={getattr(args, 'opt_batch_size', None) or 256}, "
                  f"lr={getattr(args, 'opt_lr', None) or 0.005}")
        else:
            print(f"[Stage 0] Sampler:    seeds={args.num_seed_grasps}, "
                  f"fast_nfo={args.fast_nfo}")

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

        # ------------------------------------------------------------------
        # Generate grasps for each object × each finger count
        # ------------------------------------------------------------------
        multi_graph = MultiObjectGraspGraph()
        success_count = 0

        for i, spec in enumerate(pool):
            for nf in finger_counts:
                # Use a unique seed offset per (object, finger_count) pair
                seed_offset = i * 100 + nf * 1000

                graph, isaac_spec = process_one_object(
                    spec, args,
                    seed_offset=seed_offset,
                    num_fingers=nf,
                    num_dof=num_dof,
                    dof_per_finger=dof_per_finger,
                    hand_name=hand_cfg.get("name", "shadow"),
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
        else:
            print(
                "\n[WARNING] --no-isaac_refine: grasp.object_pos_hand / object_quat_hand "
                "are NOT set.\n"
                "  The RL reset will place hand and object at RANDOM relative positions "
                "each episode\n"
                "  because it cannot reproduce the correct hand-object pose without the\n"
                "  simulated FK result.  Re-run with --isaac_refine for correct training data.\n"
            )

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
