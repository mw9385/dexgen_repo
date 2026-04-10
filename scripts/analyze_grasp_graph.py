"""
Grasp Graph Dataset Analysis
=============================
Analyzes the orientation/position distribution of a grasp graph .npy file
to check goal diversity and kNN neighbor availability.

Usage:
    python scripts/analyze_grasp_graph.py data/sharpa_grasp_cube_050.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_grasp_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load .npy grasp file → (joint_angles, object_pos, object_quat)."""
    data = np.load(path)
    print(f"Loaded {path}: shape={data.shape}")

    # .npy format: (N, 29) = joint(22) + pos(3) + quat(4)
    if data.shape[1] < 29:
        print(f"ERROR: Expected at least 29 columns, got {data.shape[1]}")
        sys.exit(1)

    joints = data[:, :22]
    pos = data[:, 22:25]
    quat = data[:, 25:29]  # (w, x, y, z)

    # Normalize quaternions
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / (norms + 1e-8)

    return joints, pos, quat


def pairwise_orn_dists(quats: np.ndarray) -> np.ndarray:
    """Compute pairwise orientation distances (rad). Returns (N, N) matrix."""
    dots = np.abs(quats @ quats.T)
    np.clip(dots, 0.0, 1.0, out=dots)
    np.fill_diagonal(dots, 0.0)
    dists = 2.0 * np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    return dists


def analyze_orientation(quats: np.ndarray):
    """Print orientation distribution statistics."""
    N = len(quats)
    print(f"\n{'='*60}")
    print(f"ORIENTATION ANALYSIS ({N} grasps)")
    print(f"{'='*60}")

    dists = pairwise_orn_dists(quats)

    # Per-grasp nearest neighbor distance
    nn_dists = np.min(dists, axis=1)

    # All pairwise (upper triangle, excluding diagonal)
    triu = dists[np.triu_indices(N, k=1)]

    print(f"\n--- All pairwise distances ({len(triu)} pairs) ---")
    print(f"  Min:    {np.min(triu):.4f} rad  ({np.degrees(np.min(triu)):.2f}°)")
    print(f"  Max:    {np.max(triu):.4f} rad  ({np.degrees(np.max(triu)):.2f}°)")
    print(f"  Mean:   {np.mean(triu):.4f} rad  ({np.degrees(np.mean(triu)):.2f}°)")
    print(f"  Median: {np.median(triu):.4f} rad  ({np.degrees(np.median(triu)):.2f}°)")
    print(f"  Std:    {np.std(triu):.4f} rad  ({np.degrees(np.std(triu)):.2f}°)")

    print(f"\n--- Nearest-neighbor distances ({N} grasps) ---")
    print(f"  Min:    {np.min(nn_dists):.4f} rad  ({np.degrees(np.min(nn_dists)):.2f}°)")
    print(f"  Max:    {np.max(nn_dists):.4f} rad  ({np.degrees(np.max(nn_dists)):.2f}°)")
    print(f"  Mean:   {np.mean(nn_dists):.4f} rad  ({np.degrees(np.mean(nn_dists)):.2f}°)")
    print(f"  Median: {np.median(nn_dists):.4f} rad  ({np.degrees(np.median(nn_dists)):.2f}°)")

    # Percentiles
    print(f"\n--- Pairwise distance percentiles ---")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(triu, p)
        print(f"  {p:3d}th: {val:.4f} rad  ({np.degrees(val):.2f}°)")

    # How many neighbors pass min_orn filter
    print(f"\n--- Neighbors passing min_orn filter (per grasp, mean count) ---")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0]:
        counts = np.sum(dists >= threshold, axis=1) - 1  # exclude self (inf)
        mean_ct = np.mean(counts)
        min_ct = np.min(counts)
        zero_ct = np.sum(counts == 0)
        print(f"  min_orn={threshold:.1f}rad ({np.degrees(threshold):5.1f}°): "
              f"mean={mean_ct:.0f}  min={min_ct}  grasps_with_0_neighbors={zero_ct}")


def analyze_position(pos: np.ndarray):
    """Print position distribution statistics."""
    N = len(pos)
    print(f"\n{'='*60}")
    print(f"POSITION ANALYSIS ({N} grasps)")
    print(f"{'='*60}")

    # Per-axis stats
    for i, axis in enumerate(["X", "Y", "Z"]):
        print(f"  {axis}: min={pos[:, i].min():.4f}  max={pos[:, i].max():.4f}  "
              f"mean={pos[:, i].mean():.4f}  std={pos[:, i].std():.4f}")

    # Pairwise position distances
    diffs = pos[:, None, :] - pos[None, :, :]
    pdists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(pdists, np.inf)

    nn_pdists = np.min(pdists, axis=1)
    triu = pdists[np.triu_indices(N, k=1)]

    print(f"\n--- Pairwise position distances ---")
    print(f"  Min:    {np.min(triu):.5f} m")
    print(f"  Max:    {np.max(triu):.5f} m")
    print(f"  Mean:   {np.mean(triu):.5f} m")
    print(f"  Median: {np.median(triu):.5f} m")

    print(f"\n--- Nearest-neighbor position distances ---")
    print(f"  Min:    {np.min(nn_pdists):.5f} m")
    print(f"  Max:    {np.max(nn_pdists):.5f} m")
    print(f"  Mean:   {np.mean(nn_pdists):.5f} m")

    # max_pos filter
    print(f"\n--- Neighbors passing max_pos filter (per grasp, mean count) ---")
    for threshold in [0.01, 0.02, 0.05, 0.10, 0.20]:
        counts = np.sum(pdists <= threshold, axis=1)
        mean_ct = np.mean(counts)
        min_ct = np.min(counts)
        zero_ct = np.sum(counts == 0)
        print(f"  max_pos={threshold:.2f}m: "
              f"mean={mean_ct:.0f}  min={min_ct}  grasps_with_0_neighbors={zero_ct}")


def analyze_combined(pos: np.ndarray, quats: np.ndarray):
    """Analyze how many grasps pass BOTH min_orn AND max_pos filters."""
    N = len(pos)
    print(f"\n{'='*60}")
    print(f"COMBINED FILTER ANALYSIS (min_orn AND max_pos)")
    print(f"{'='*60}")

    orn_dists = pairwise_orn_dists(quats)

    diffs = pos[:, None, :] - pos[None, :, :]
    pos_dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(pos_dists, np.inf)

    max_pos = 0.05  # 5cm, hardcoded in _sample_nearby_goal_index

    print(f"  max_pos = {max_pos:.2f}m (fixed)")
    print()
    for min_orn in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        both_mask = (orn_dists >= min_orn) & (pos_dists <= max_pos)
        counts = np.sum(both_mask, axis=1)
        mean_ct = np.mean(counts)
        min_ct = np.min(counts)
        zero_ct = np.sum(counts == 0)
        print(f"  min_orn={min_orn:.1f}rad ({np.degrees(min_orn):5.1f}°) + max_pos={max_pos}m: "
              f"mean={mean_ct:.1f}  min={min_ct}  grasps_with_0_valid={zero_ct}/{N} "
              f"({zero_ct/N*100:.1f}%)")


def analyze_joints(joints: np.ndarray):
    """Print joint angle statistics."""
    N, D = joints.shape
    print(f"\n{'='*60}")
    print(f"JOINT ANGLE ANALYSIS ({N} grasps, {D} DOF)")
    print(f"{'='*60}")
    print(f"  Overall min: {joints.min():.4f}  max: {joints.max():.4f}")
    print(f"  Per-DOF range:")
    for i in range(D):
        rng = joints[:, i].max() - joints[:, i].min()
        print(f"    DOF {i:2d}: [{joints[:, i].min():.3f}, {joints[:, i].max():.3f}]  "
              f"range={rng:.3f}  std={joints[:, i].std():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze grasp graph dataset")
    parser.add_argument("path", type=str, help="Path to .npy grasp graph file")
    parser.add_argument("--no-joints", action="store_true", help="Skip joint analysis")
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"ERROR: File not found: {args.path}")
        sys.exit(1)

    joints, pos, quats = load_grasp_data(args.path)

    analyze_orientation(quats)
    analyze_position(pos)
    analyze_combined(pos, quats)
    if not args.no_joints:
        analyze_joints(joints)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
