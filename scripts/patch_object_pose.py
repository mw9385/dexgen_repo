"""
Patch existing grasp_graph.pkl: recompute obj_pos_hand as fingertip centroid in hand frame.

Usage:
    python scripts/patch_object_pose.py --input data/grasp_graph.pkl --output data/grasp_graph.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/grasp_graph.pkl")
    p.add_argument("--output", type=str, default="data/grasp_graph.pkl")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        graph = pickle.load(f)

    patched = 0
    for obj_name, subgraph in graph.graphs.items():
        for grasp in subgraph.grasp_set.grasps:
            if grasp.fingertip_positions is None or grasp.joint_angles is None:
                continue
            # fingertip_positions are in object frame (object at origin in optimizer).
            # In the default hand frame (identity wrist rotation), the fingertip
            # centroid IS the object position relative to the hand.
            centroid = grasp.fingertip_positions.mean(axis=0)
            grasp.object_pos_hand = centroid.astype(np.float32)
            grasp.object_quat_hand = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            grasp.object_pose_frame = "hand_root"
            patched += 1

    with open(args.output, "wb") as f:
        pickle.dump(graph, f)
    print(f"Patched {patched} grasps → {args.output}")


if __name__ == "__main__":
    main()
