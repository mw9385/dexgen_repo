"""
Grasp Graph Orientation Augmentation
=====================================
Expands a .npy grasp graph by applying cube symmetry rotations to the
object quaternion. Since a cube is invariant under 24 rotations, the
same hand configuration (joint angles, fingertip contacts) remains
physically valid after rotating the object by any symmetry element.

This increases the orientation coverage from ~60° (original dataset)
to the full rotation space, enabling the policy to learn large
reorientations including full flips.

Usage:
    python scripts/augment_grasp_graph.py \
        data/sharpa_grasp_cube_050.npy \
        data/sharpa_grasp_cube_050_aug.npy

    # Subsample to limit size
    python scripts/augment_grasp_graph.py \
        data/sharpa_grasp_cube_050.npy \
        data/sharpa_grasp_cube_050_aug.npy \
        --max_grasps 50000

    # Add extra random rotations on top of symmetry
    python scripts/augment_grasp_graph.py \
        data/sharpa_grasp_cube_050.npy \
        data/sharpa_grasp_cube_050_aug.npy \
        --extra_random 4 --random_std_deg 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Cube symmetry rotations (24 elements of the rotation group)
# ---------------------------------------------------------------------------

def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Quaternion (w,x,y,z) from axis-angle."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half = angle * 0.5
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def cube_symmetry_quaternions() -> list[np.ndarray]:
    """
    Generate the 24 rotation quaternions of a cube's symmetry group.

    Decomposition:
      - Identity: 1
      - Face rotations (90°, 180°, 270° around x, y, z): 9
      - Edge rotations (180° around face diagonals): 6
      - Vertex rotations (120°, 240° around body diagonals): 8
      Total: 24
    """
    quats = []

    # Identity
    quats.append(np.array([1, 0, 0, 0], dtype=np.float64))

    # 90°, 180°, 270° around each axis (x, y, z)
    for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
        for angle in [np.pi/2, np.pi, 3*np.pi/2]:
            quats.append(_quat_from_axis_angle(axis, angle))

    # 180° around face diagonals (6 rotations)
    face_diags = [
        np.array([1, 1, 0]), np.array([1, -1, 0]),
        np.array([1, 0, 1]), np.array([1, 0, -1]),
        np.array([0, 1, 1]), np.array([0, 1, -1]),
    ]
    for axis in face_diags:
        quats.append(_quat_from_axis_angle(axis, np.pi))

    # 120° and 240° around body diagonals (8 rotations)
    body_diags = [
        np.array([1, 1, 1]), np.array([1, 1, -1]),
        np.array([1, -1, 1]), np.array([1, -1, -1]),
    ]
    for axis in body_diags:
        for angle in [2*np.pi/3, 4*np.pi/3]:
            quats.append(_quat_from_axis_angle(axis, angle))

    assert len(quats) == 24, f"Expected 24 symmetry rotations, got {len(quats)}"

    # Normalize
    quats = [q / (np.linalg.norm(q) + 1e-8) for q in quats]
    return quats


def random_rotation_quaternion(rng: np.random.Generator, std_rad: float) -> np.ndarray:
    """Small random rotation quaternion with Gaussian angle."""
    axis = rng.standard_normal(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = rng.standard_normal() * std_rad
    return _quat_from_axis_angle(axis, angle)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_grasps(
    data: np.ndarray,
    extra_random: int = 0,
    random_std_deg: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Augment grasp data by applying cube symmetry rotations.

    Args:
        data: (N, 29) original grasp data
        extra_random: additional random rotations per symmetry rotation
        random_std_deg: std of random rotation noise in degrees
        seed: random seed

    Returns:
        (N_aug, 29) augmented data
    """
    rng = np.random.default_rng(seed)
    N = data.shape[0]
    sym_quats = cube_symmetry_quaternions()
    random_std_rad = np.radians(random_std_deg)

    augmented = []

    for sym_q in sym_quats:
        for i in range(N):
            row = data[i].copy()
            obj_quat = row[25:29].astype(np.float64)  # (w, x, y, z)

            # Apply symmetry rotation: new_quat = old_quat * R_sym
            new_quat = _quat_multiply(obj_quat, sym_q)
            new_quat = new_quat / (np.linalg.norm(new_quat) + 1e-8)

            row[25:29] = new_quat.astype(np.float32)
            # object_pos_hand (22:25) stays the same — cube center doesn't move
            # joint_angles (0:22) stay the same — hand configuration unchanged
            augmented.append(row)

            # Extra random perturbations
            for _ in range(extra_random):
                noise_q = random_rotation_quaternion(rng, random_std_rad)
                noisy_quat = _quat_multiply(new_quat, noise_q)
                noisy_quat = noisy_quat / (np.linalg.norm(noisy_quat) + 1e-8)
                noisy_row = row.copy()
                noisy_row[25:29] = noisy_quat.astype(np.float32)
                augmented.append(noisy_row)

    return np.stack(augmented, axis=0)


def verify_augmentation(original: np.ndarray, augmented: np.ndarray):
    """Print statistics comparing original and augmented datasets."""
    def _orn_stats(data):
        quats = data[:, 25:29].astype(np.float64)
        norms = np.linalg.norm(quats, axis=-1, keepdims=True)
        quats = quats / (norms + 1e-8)
        # Subsample for pairwise computation if too large
        N = len(quats)
        if N > 5000:
            idx = np.random.default_rng(0).choice(N, 5000, replace=False)
            quats = quats[idx]
        dots = np.abs(quats @ quats.T)
        np.clip(dots, 0, 1, out=dots)
        np.fill_diagonal(dots, 0)
        dists = 2.0 * np.arccos(dots)
        triu = dists[np.triu_indices(len(quats), k=1)]
        return triu

    print(f"\n{'='*60}")
    print(f"AUGMENTATION VERIFICATION")
    print(f"{'='*60}")
    print(f"  Original:  {len(original)} grasps")
    print(f"  Augmented: {len(augmented)} grasps")

    print(f"\n--- Original orientation distances ---")
    orig_dists = _orn_stats(original)
    print(f"  Min:  {np.min(orig_dists):.4f} rad  ({np.degrees(np.min(orig_dists)):.1f}°)")
    print(f"  Max:  {np.max(orig_dists):.4f} rad  ({np.degrees(np.max(orig_dists)):.1f}°)")
    print(f"  Mean: {np.mean(orig_dists):.4f} rad  ({np.degrees(np.mean(orig_dists)):.1f}°)")

    print(f"\n--- Augmented orientation distances ---")
    aug_dists = _orn_stats(augmented)
    print(f"  Min:  {np.min(aug_dists):.4f} rad  ({np.degrees(np.min(aug_dists)):.1f}°)")
    print(f"  Max:  {np.max(aug_dists):.4f} rad  ({np.degrees(np.max(aug_dists)):.1f}°)")
    print(f"  Mean: {np.mean(aug_dists):.4f} rad  ({np.degrees(np.mean(aug_dists)):.1f}°)")

    print(f"\n--- Neighbor availability (augmented) ---")
    quats = augmented[:, 25:29].astype(np.float64)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / (norms + 1e-8)
    N = len(quats)
    if N > 5000:
        idx = np.random.default_rng(0).choice(N, 5000, replace=False)
        quats_sub = quats[idx]
    else:
        quats_sub = quats
    dots = np.abs(quats_sub @ quats_sub.T)
    np.clip(dots, 0, 1, out=dots)
    np.fill_diagonal(dots, 0)
    dists = 2.0 * np.arccos(dots)
    for threshold in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        counts = np.sum(dists >= threshold, axis=1)
        zero_ct = np.sum(counts == 0)
        print(f"  min_orn={threshold:.1f}rad ({np.degrees(threshold):5.1f}°): "
              f"mean_neighbors={np.mean(counts):.0f}  "
              f"grasps_with_0={zero_ct}/{len(quats_sub)}")


def main():
    parser = argparse.ArgumentParser(description="Augment grasp graph with cube symmetry rotations")
    parser.add_argument("input", type=str, help="Input .npy grasp file")
    parser.add_argument("output", type=str, help="Output .npy augmented file")
    parser.add_argument("--max_grasps", type=int, default=None,
                        help="Subsample to this many grasps after augmentation")
    parser.add_argument("--extra_random", type=int, default=0,
                        help="Extra random rotations per symmetry rotation")
    parser.add_argument("--random_std_deg", type=float, default=10.0,
                        help="Std of random rotation noise (degrees)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Print augmentation statistics")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    original = np.load(args.input)
    print(f"Loaded {args.input}: {original.shape}")

    augmented = augment_grasps(
        original,
        extra_random=args.extra_random,
        random_std_deg=args.random_std_deg,
        seed=args.seed,
    )
    print(f"Augmented: {original.shape[0]} × {24 * (1 + args.extra_random)} = {len(augmented)} grasps")

    if args.max_grasps and len(augmented) > args.max_grasps:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(augmented), args.max_grasps, replace=False)
        augmented = augmented[idx]
        print(f"Subsampled to {len(augmented)} grasps")

    if args.verify:
        verify_augmentation(original, augmented)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, augmented)
    print(f"\nSaved: {args.output} ({augmented.shape})")


if __name__ == "__main__":
    main()
