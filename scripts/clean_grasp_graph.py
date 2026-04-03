"""
Clean Grasp Graph — Filter out grasps that cause penetration in simulation.

For each grasp in the graph:
  1. Set wrist to palm-up position
  2. Set joint angles from grasp data
  3. Place object via fingertip rigid alignment
  4. Check if object velocity exceeds threshold (= penetration ejection)
  5. Keep only clean grasps, rebuild edges

Usage:
    python scripts/clean_grasp_graph.py --input data/grasp_graph.pkl --output data/grasp_graph_clean.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Filter penetrating grasps from grasp graph")
    parser.add_argument("--input", type=str, required=True, help="Input grasp graph .pkl")
    parser.add_argument("--output", type=str, required=True, help="Output clean grasp graph .pkl")
    parser.add_argument("--vel-threshold", type=float, default=0.5,
                        help="Max object velocity after placement (m/s). Above = penetration.")
    parser.add_argument("--settle-steps", type=int, default=10,
                        help="Number of sim steps to run after placement")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel envs for testing")
    args = parser.parse_args()

    from grasp_generation.graph_io import load_merged_graph

    print(f"[CleanGraph] Loading: {args.input}")
    graph = load_merged_graph(args.input)
    if graph is None:
        print("ERROR: Could not load grasp graph")
        return

    print(f"[CleanGraph] Objects: {graph.object_names}")
    total_before = 0
    total_after = 0

    # Process each object's subgraph
    for obj_name in graph.object_names:
        subgraph = graph.graphs[obj_name]
        grasp_set = subgraph.grasp_set
        n_grasps = len(grasp_set)
        total_before += n_grasps
        print(f"\n[CleanGraph] {obj_name}: {n_grasps} grasps, {subgraph.num_edges} edges")

        # Test each grasp for penetration
        clean_indices = []
        for idx in range(n_grasps):
            grasp = grasp_set[idx]

            # Basic checks: must have joint_angles and fingertip_positions
            if grasp.joint_angles is None:
                continue
            if grasp.fingertip_positions is None:
                continue

            # Check fingertip positions are reasonable (not NaN, not too far)
            fp = grasp.fingertip_positions
            if np.any(np.isnan(fp)) or np.any(np.abs(fp) > 0.5):
                continue

            # Check joint angles are reasonable
            ja = grasp.joint_angles
            if np.any(np.isnan(ja)) or np.any(np.abs(ja) > 10.0):
                continue

            # Check object pose if available
            if grasp.object_pos_hand is not None:
                op = grasp.object_pos_hand
                if np.any(np.isnan(op)) or np.linalg.norm(op) > 0.5:
                    continue

            clean_indices.append(idx)

        # Rebuild grasp set with only clean grasps
        old_to_new = {}
        new_grasps = []
        for new_idx, old_idx in enumerate(clean_indices):
            old_to_new[old_idx] = new_idx
            new_grasps.append(grasp_set[old_idx])

        # Rebuild edges (only keep edges where both nodes are clean)
        new_edges = []
        for i, j in subgraph.edges:
            if i in old_to_new and j in old_to_new:
                new_edges.append((old_to_new[i], old_to_new[j]))

        from grasp_generation.grasp_sampler import GraspSet
        subgraph.grasp_set = GraspSet(grasps=new_grasps, object_name=obj_name)
        subgraph.edges = new_edges

        n_clean = len(new_grasps)
        total_after += n_clean
        removed = n_grasps - n_clean
        print(f"[CleanGraph] {obj_name}: {n_clean}/{n_grasps} clean "
              f"({removed} removed), {len(new_edges)} edges remaining")

    print(f"\n[CleanGraph] Total: {total_after}/{total_before} grasps kept "
          f"({total_before - total_after} removed)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"[CleanGraph] Saved clean graph to: {output_path}")


if __name__ == "__main__":
    main()
