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
        --size_min 0.04 --size_max 0.09 --num_sizes 4 \\
        --num_grasps 300

    # Single custom mesh
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py --mesh_path assets/mug.obj
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

from grasp_generation import (
    GraspSampler, NetForceOptimizer, RRTGraspExpander,
    ObjectPool, MultiObjectGraspGraph, refine_multi_object_graph_with_isaac,
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
        "num_seed_grasps": 2000,   # 7× more seed candidates (better coverage)
        "num_grasps": 1000,         # 3× more RRT nodes (denser graph)
        "min_quality": 0.005,       # same as default — do NOT raise this
        # min_quality=0.01 was incorrectly stricter; NFO ε-metric values are
        # typically 0.001–0.008 on primitive objects so 0.01 filters 100% of
        # candidates and RRT generates zero grasps.  The quality improvement
        # in high_precision comes from MORE grasps, not a stricter threshold.
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
    num_dof: int = 22,
    hand_name: str = "shadow",
) -> np.ndarray:
    """
    Heuristic IK: fingertip distance → joint angles.

    Supported hands
    ---------------
    shadow (22 actuated DOF = Isaac Lab rh_* joints):
      Layout (Isaac Lab ordering):
        0-3   rh_FFJ4, rh_FFJ3, rh_FFJ2, rh_FFJ1   (forefinger: J4=spread)
        4-7   rh_MFJ4, rh_MFJ3, rh_MFJ2, rh_MFJ1   (middle finger)
        8-11  rh_RFJ4, rh_RFJ3, rh_RFJ2, rh_RFJ1   (ring finger)
        12-16 rh_LFJ5, rh_LFJ4, rh_LFJ3, rh_LFJ2, rh_LFJ1  (little finger, 5 DOF)
        17-21 rh_THJ5, rh_THJ4, rh_THJ3, rh_THJ2, rh_THJ1  (thumb, 5 DOF)
      NOTE: exact joint ordering varies by Isaac Lab version; we use a
      simplified per-finger flexion heuristic that is robust to reordering:
        - J1/J2/J3 (flexion): scale × [1.5, 1.2, 0.8]
        - J4 (spread/abduction): 0.0 (neutral)
        - J5 (LF only – extra MCP): 0.0

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
        # ── Shadow Hand layout ──────────────────────────────────────────────
        # For each of the 4 "long" fingers (FF/MF/RF/LF), joints per finger:
        #   [spread, J3, J2, J1]  where J1 is most proximal flexion joint.
        # Thumb (5 joints): [THJ5, THJ4, THJ3, THJ2, THJ1].
        # We set flexion joints (J1-J3) proportional to scale and leave spread=0.
        #
        # Finger order in grasps: FF(0), MF(1), RF(2), LF(3), TH(4)
        # DOF block sizes: FF=4, MF=4, RF=4, LF=5, TH=5 → total 22

        # For each regular finger (4 DOF: [spread, J3, J2, J1]):
        #   J3 (distal)  = scale * 0.80 rad
        #   J2 (middle)  = scale * 1.20 rad
        #   J1 (proximal)= scale * 1.50 rad
        #   spread = 0.0
        FINGER_DOFS = [4, 4, 4, 5, 5]      # FF, MF, RF, LF, TH
        n_regular_fingers = min(num_fingers, 4)  # FF/MF/RF/LF
        has_thumb = (num_fingers >= 4)  # last finger slot = thumb

        finger_map = list(range(min(num_fingers, 4)))  # which grasps → which DOF blocks
        if num_fingers >= 5:
            # 5-finger: [FF, MF, RF, LF, TH] → blocks [0,1,2,3,4]
            finger_to_block = list(range(5))
        elif num_fingers == 4:
            # 4-finger: [FF, MF, RF, TH] → blocks [0,1,2,4] (skip LF block)
            finger_to_block = [0, 1, 2, 4]
        elif num_fingers == 3:
            # 3-finger: [FF, MF, TH] → blocks [0,1,4]
            finger_to_block = [0, 1, 4]
        elif num_fingers == 2:
            # 2-finger: [FF, TH] → blocks [0,4]
            finger_to_block = [0, 4]
        else:
            finger_to_block = list(range(num_fingers))

        # Build DOF start offsets from FINGER_DOFS blocks
        dof_starts = [0]
        for d in FINGER_DOFS:
            dof_starts.append(dof_starts[-1] + d)
        # dof_starts = [0, 4, 8, 12, 17, 22]

        for fi in range(min(num_fingers, len(finger_to_block))):
            s = float(scale[fi])
            block = finger_to_block[fi]
            start = dof_starts[block]
            ndof  = FINGER_DOFS[block]

            if block == 4:
                # Thumb (5 DOF: [THJ5, THJ4, THJ3, THJ2, THJ1])
                # THJ5 = rotation ~0; THJ4 = abduction; THJ1-THJ3 = flexion
                if start + 5 <= num_dof:
                    joint_angles[start + 0] = 0.0          # THJ5 (rotation, neutral)
                    joint_angles[start + 1] = s * 0.50     # THJ4 (abduction ≈ 0.5 max)
                    joint_angles[start + 2] = s * 1.00     # THJ3 (flexion)
                    joint_angles[start + 3] = s * 0.80     # THJ2 (flexion)
                    joint_angles[start + 4] = s * 0.60     # THJ1 (flexion)
            elif block == 3:
                # LF (5 DOF: [LFJ5, LFJ4, LFJ3, LFJ2, LFJ1])
                if start + 5 <= num_dof:
                    joint_angles[start + 0] = 0.0          # LFJ5 (MCP extra, neutral)
                    joint_angles[start + 1] = 0.0          # LFJ4 (spread, neutral)
                    joint_angles[start + 2] = s * 0.80     # LFJ3 (distal)
                    joint_angles[start + 3] = s * 1.20     # LFJ2 (middle)
                    joint_angles[start + 4] = s * 1.50     # LFJ1 (proximal)
            else:
                # Regular finger (4 DOF: [spread, J3, J2, J1])
                if start + 4 <= num_dof:
                    joint_angles[start + 0] = 0.0          # J4 spread, neutral
                    joint_angles[start + 1] = s * 0.80     # J3 distal flexion
                    joint_angles[start + 2] = s * 1.20     # J2 middle flexion
                    joint_angles[start + 3] = s * 1.50     # J1 proximal flexion

        # Wrist DOF 22-23 (if present): keep at 0
    else:
        # ── Generic / Allegro fallback: uniform scale per finger ────────────
        dof_per_finger = max(1, num_dof // max(num_fingers, 1))
        for f in range(min(num_fingers, num_dof // dof_per_finger)):
            s = float(scale[f])
            start = f * dof_per_finger
            end   = start + dof_per_finger
            joint_angles[start:end] = s * 1.57

    return joint_angles


def _attach_joint_angles_to_graph(graph, num_dof: int = 22, dof_per_finger: int = 4,
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
    num_dof: int = 22,
    dof_per_finger: int = 4,
    hand_name: str = "shadow",
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
    #
    # delta_max budget analysis:
    #   multiplier = 0.30 gives effective delta_max of:
    #     4cm object:  0.04 * 0.30 * 2.4 = 2.9 cm   (5-finger effective mean)
    #     6cm object:  0.06 * 0.30 * 2.4 = 4.3 cm
    #     9cm object:  0.09 * 0.30 * 2.4 = 6.5 cm
    #
    #   Previous 0.60 gave 8.6 cm mean for 6cm/5-finger.  8.6 cm in object frame
    #   → each finger moves ~8.6 cm → requires several rad of joint change →
    #   unreachable in 10-second episode.
    #
    #   With 0.30: ~4 cm mean displacement → ~0.6–0.8 rad L2 joint change →
    #   clearly achievable in 300 steps while remaining tight enough for a
    #   300-node graph to be well-connected (many edges per node).
    #
    #   We tried 0.15 (even tighter) but it required 2000+ grasps for connectivity.
    #   0.30 is the sweet spot: reachable goals + good graph connectivity at 300 nodes.
    #
    # delta_pos (RRT step size) = delta_max / 3:
    #   Each RRT step perturbs by ≤ delta_max/3 so multiple steps per edge.
    delta_max_base = spec.size * 0.30          # was 0.60 — 2× tighter
    delta_pos_base = delta_max_base / 3.0      # step ≤ 1/3 of edge budget
    expander = RRTGraspExpander(
        nfo=NetForceOptimizer(min_quality=args.min_quality, fast_mode=True),
        target_size=args.num_grasps,
        delta_pos=delta_pos_base,
        delta_max=delta_max_base,
        manifold_contact_count=num_fingers,
        seed=args.seed + seed_offset,
    )
    graph = expander.expand(filtered_set)

    # Step 4: Solve heuristic IK for all grasps and store joint_angles
    _attach_joint_angles_to_graph(graph, num_dof=num_dof, dof_per_finger=dof_per_finger,
                                  hand_name=hand_name)

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
