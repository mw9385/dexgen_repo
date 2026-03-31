"""
Isaac Refinement — Post-process grasp_graph.pkl in Isaac Sim
=============================================================
Loads a grasp graph and refines each grasp in Isaac Sim to correct
FK discrepancies between pytorch_kinematics and Isaac Sim.

What it does (per grasp):
  1. Set hand to stored joint_angles in Isaac Sim
  2. Place object at fingertip centroid
  3. Run 3 rounds of differential IK (30 total IK steps) to refine
     joint angles so fingertips match target positions in Isaac Sim FK
  4. Overwrite grasp data with Isaac Sim's actual values:
     - joint_angles: from Isaac Sim joint state
     - object_pos_hand: Isaac Sim FK-based relative position
     - object_quat_hand: Isaac Sim FK-based relative rotation
     - reset_contact_error: measured fingertip-to-target distance (m)

After refinement, all grasp data is consistent with Isaac Sim FK,
so reward functions (fingertip_tracking, finger_joint_goal, etc.)
get coherent targets.

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/refine_grasps.py \\
        --grasp_graph data/grasp_graph.pkl --headless
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser(description="Refine grasps in Isaac Sim")
    p.add_argument("--grasp_graph", type=str, required=True,
                   help="Path to grasp_graph.pkl from run_grasp_generation.py")
    p.add_argument("--batch_envs", type=int, default=16,
                   help="Batch size for Isaac refinement environments")
    p.add_argument("--keep_top_k", type=int, default=None,
                   help="Keep only top-K lowest-error grasps per object")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def _snapshot_grasps(grasps):
    """Save before-refinement values for comparison."""
    snapshot = []
    for g in grasps:
        snapshot.append({
            'joint_angles': g.joint_angles.copy() if g.joint_angles is not None else None,
            'object_pos_hand': g.object_pos_hand.copy() if g.object_pos_hand is not None else None,
            'object_quat_hand': g.object_quat_hand.copy() if g.object_quat_hand is not None else None,
            'fingertip_positions': g.fingertip_positions.copy(),
        })
    return snapshot


def _print_comparison(name, grasps, before_snapshot):
    """Print before/after comparison for a single object's grasps."""
    n = len(grasps)
    if n == 0:
        return

    # Joint angle differences
    joint_diffs = []
    for i, g in enumerate(grasps):
        bq = before_snapshot[i]['joint_angles']
        aq = g.joint_angles
        if bq is not None and aq is not None:
            joint_diffs.append(float(np.linalg.norm(aq - bq)))

    # Object position differences
    pos_diffs = []
    for i, g in enumerate(grasps):
        bp = before_snapshot[i]['object_pos_hand']
        ap = g.object_pos_hand
        if bp is not None and ap is not None:
            pos_diffs.append(float(np.linalg.norm(ap - bp)))

    # Object quaternion differences
    quat_diffs = []
    for i, g in enumerate(grasps):
        bq = before_snapshot[i]['object_quat_hand']
        aq = g.object_quat_hand
        if bq is not None and aq is not None:
            dot = float(np.clip(abs(np.dot(bq, aq)), 0.0, 1.0))
            angle = 2.0 * np.arccos(dot)
            quat_diffs.append(float(np.degrees(angle)))

    # Contact errors (only available after refinement)
    contact_errs = [
        float(g.reset_contact_error)
        for g in grasps
        if getattr(g, 'reset_contact_error', None) is not None
    ]

    print(f"\n  ── {name}: Before vs After refinement ({n} grasps) ──")

    if joint_diffs:
        print(f"  Joint angle change (L2 norm):")
        print(f"    mean={np.mean(joint_diffs):.4f} rad, "
              f"max={np.max(joint_diffs):.4f} rad, "
              f"min={np.min(joint_diffs):.4f} rad")

    if pos_diffs:
        print(f"  Object position change (hand frame):")
        print(f"    mean={np.mean(pos_diffs)*1000:.1f} mm, "
              f"max={np.max(pos_diffs)*1000:.1f} mm, "
              f"min={np.min(pos_diffs)*1000:.1f} mm")

    if quat_diffs:
        print(f"  Object rotation change:")
        print(f"    mean={np.mean(quat_diffs):.1f} deg, "
              f"max={np.max(quat_diffs):.1f} deg, "
              f"min={np.min(quat_diffs):.1f} deg")

    if contact_errs:
        print(f"  Final contact error (fingertip-to-target distance):")
        print(f"    mean={np.mean(contact_errs)*1000:.1f} mm, "
              f"best={np.min(contact_errs)*1000:.1f} mm, "
              f"worst={np.max(contact_errs)*1000:.1f} mm")
        good = sum(1 for e in contact_errs if e < 0.01)
        print(f"    {good}/{n} grasps within 10mm contact error")


def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    try:
        from grasp_generation import refine_multi_object_graph_with_isaac
        from grasp_generation.rrt_expansion import MultiObjectGraspGraph

        graph_path = Path(args.grasp_graph)
        if not graph_path.exists():
            print(f"ERROR: {graph_path} not found")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f" Isaac Refinement")
        print(f" Input: {graph_path}")
        print(f"{'='*60}")

        graph = MultiObjectGraspGraph.load(str(graph_path))
        graph.summary()

        # Snapshot before-refinement values
        before_snapshots = {}
        for name, g in graph.graphs.items():
            before_snapshots[name] = _snapshot_grasps(g.grasp_set.grasps)

        print(f"\nRefining with batch_envs={args.batch_envs}...")
        print(f"  (3 rounds × 10 IK iterations = 30 differential IK steps per grasp)")
        graph = refine_multi_object_graph_with_isaac(
            graph,
            batch_envs=args.batch_envs,
            keep_top_k=args.keep_top_k,
        )

        # Print before/after comparison
        print(f"\n{'='*60}")
        print(f" Refinement Results")
        print(f"{'='*60}")
        for name, g in graph.graphs.items():
            snap = before_snapshots.get(name, [])
            # After keep_top_k, grasps may be fewer than snapshots
            snap = snap[:len(g.grasp_set.grasps)]
            _print_comparison(name, g.grasp_set.grasps, snap)

        # Save refined graph (overwrite)
        graph.save(str(graph_path))

        print(f"\n{'='*60}")
        print(f" Refinement Complete")
        print(f"{'='*60}")
        graph.summary()
        print(f"\n  Saved: {graph_path}")
        print(f"\nNext step:")
        print(f"  /workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py "
              f"--grasp_graph {graph_path} --action_mode hold")
        print(f"  /workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py "
              f"--grasp_graph {graph_path} --num_envs 512 --headless")

    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
