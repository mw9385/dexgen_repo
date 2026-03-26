"""
Stage 2 – Dataset Collection
==============================
Rolls out the trained RL policy on all grasp pairs from the GraspGraph
and records trajectories for DexGen controller training.

Each trajectory records:
  - keypoint_traj:   (T, 12)  fingertip positions in object frame
  - joint_traj:      (T, 16)  joint positions
  - action_traj:     (T, 16)  RL policy actions
  - robot_state:     (T, 32)  joint pos + vel
  - object_pose:     (T, 7)   object pos (3) + quat (4)
  - start_grasp_idx: int
  - goal_grasp_idx:  int
  - success:         bool     (all fingertips within threshold at end)

Dataset is saved as HDF5 for efficient I/O during Stage 3 training.

Usage:
    python scripts/collect_data.py \\
        --checkpoint logs/rl/allegro_anygrasp/checkpoints/model_30000.pt \\
        --grasp_graph data/grasp_graph.pkl \\
        --num_episodes 50000 \\
        --output data/dataset.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 2: Dataset Collection")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained RL policy checkpoint")
    p.add_argument("--grasp_graph", type=str, default="data/grasp_graph.pkl",
                   help="Path to GraspGraph from Stage 0")
    p.add_argument("--num_envs", type=int, default=64,
                   help="Parallel environments for collection")
    p.add_argument("--num_episodes", type=int, default=50000,
                   help="Total episodes to collect")
    p.add_argument("--episode_length", type=int, default=200,
                   help="Max steps per episode")
    p.add_argument("--output", type=str, default="data/dataset.h5",
                   help="Output HDF5 file path")
    p.add_argument("--min_success_rate", type=float, default=0.3,
                   help="Minimum success rate; warn if below this")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def collect_episode_batch(env, policy, episode_length: int, device: str) -> list:
    """
    Run one batch of episodes and return trajectory dicts.
    Returns list of trajectory dicts (one per env).
    """
    import torch

    obs, _ = env.reset()
    num_envs = env.num_envs

    # Storage
    kp_trajs = []      # (T, N, 12)
    jq_trajs = []      # (T, N, 16)
    act_trajs = []     # (T, N, 16)
    rs_trajs = []      # (T, N, 32)
    obj_trajs = []     # (T, N, 7)

    for _ in range(episode_length):
        with torch.no_grad():
            action = policy(obs["policy"])

        # Store state before step
        robot = env.scene["robot"]
        obj = env.scene["object"]
        q = robot.data.joint_pos.cpu().numpy()
        dq = robot.data.joint_vel.cpu().numpy()
        rs = np.concatenate([q, dq], axis=-1)          # (N, 32)

        obj_pos = obj.data.root_pos_w.cpu().numpy()    # (N, 3)
        obj_quat = obj.data.root_quat_w.cpu().numpy()  # (N, 4)
        obj_pose = np.concatenate([obj_pos, obj_quat], axis=-1)   # (N, 7)

        from envs.mdp.observations import fingertip_positions_in_object_frame
        kp = fingertip_positions_in_object_frame(env).cpu().numpy()  # (N, 12)

        kp_trajs.append(kp)
        jq_trajs.append(q)
        act_trajs.append(action.cpu().numpy())
        rs_trajs.append(rs)
        obj_trajs.append(obj_pose)

        obs, _, terminated, truncated, info = env.step(action)
        done = (terminated | truncated).cpu().numpy()

    # Stack: (T, N, dim)
    kp_trajs = np.stack(kp_trajs, axis=0)
    jq_trajs = np.stack(jq_trajs, axis=0)
    act_trajs = np.stack(act_trajs, axis=0)
    rs_trajs = np.stack(rs_trajs, axis=0)
    obj_trajs = np.stack(obj_trajs, axis=0)

    # Check success: final fingertip dist < 1 cm
    target = env.extras.get("target_fingertip_pos").cpu().numpy()   # (N, 12)
    final_kp = kp_trajs[-1]                                          # (N, 12)
    success = (np.linalg.norm(
        final_kp.reshape(-1, 4, 3) - target.reshape(-1, 4, 3),
        axis=-1
    ).mean(axis=-1) < 0.01)   # (N,) bool

    # Package per-env trajectories
    trajectories = []
    for i in range(num_envs):
        trajectories.append({
            "keypoint_traj": kp_trajs[:, i, :].astype(np.float32),
            "joint_traj": jq_trajs[:, i, :].astype(np.float32),
            "action_traj": act_trajs[:, i, :].astype(np.float32),
            "robot_state": rs_trajs[:, i, :].astype(np.float32),
            "object_pose": obj_trajs[:, i, :].astype(np.float32),
            "success": bool(success[i]),
        })
    return trajectories


def main():
    args = parse_args()

    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.grasp_graph).exists():
        print(f"ERROR: GraspGraph not found: {args.grasp_graph}")
        sys.exit(1)

    try:
        import isaaclab  # noqa: F401
        import torch
    except ImportError:
        print("ERROR: Isaac Lab or PyTorch not found.")
        sys.exit(1)

    from isaaclab.envs import ManagerBasedRLEnv
    from envs import AnyGraspEnvCfg, register_anygrasp_env

    register_anygrasp_env()

    # Build env
    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.grasp_graph_path = args.grasp_graph
    env = ManagerBasedRLEnv(env_cfg)

    # Load policy (rl_games format)
    from rl_games.algos_torch import players
    agent = players.PpoPlayerContinuous({"params": {"config": {
        "name": "DexGen-AnyGrasp-Allegro",
        "env_name": "rlgpu",
        "device": args.device,
    }}})
    agent.restore(args.checkpoint)
    policy = agent.get_action

    # Collection loop
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 2] Collecting {args.num_episodes} episodes...")
    print(f"[Stage 2] Checkpoint: {args.checkpoint}")
    print(f"[Stage 2] Output: {output_path}")
    print("-" * 60)

    total_collected = 0
    total_success = 0
    num_batches = (args.num_episodes + args.num_envs - 1) // args.num_envs

    with h5py.File(output_path, "w") as f:
        # Dataset metadata
        f.attrs["num_episodes"] = args.num_episodes
        f.attrs["episode_length"] = args.episode_length
        f.attrs["checkpoint"] = args.checkpoint
        f.attrs["grasp_graph"] = args.grasp_graph

        traj_grp = f.create_group("trajectories")

        for batch_idx in range(num_batches):
            trajs = collect_episode_batch(
                env, policy, args.episode_length, args.device
            )
            for traj in trajs:
                if total_collected >= args.num_episodes:
                    break
                ep_grp = traj_grp.create_group(str(total_collected))
                for key, val in traj.items():
                    if isinstance(val, np.ndarray):
                        ep_grp.create_dataset(key, data=val)
                    else:
                        ep_grp.attrs[key] = val
                if traj["success"]:
                    total_success += 1
                total_collected += 1

            success_rate = total_success / max(total_collected, 1)
            if (batch_idx + 1) % 10 == 0:
                print(f"  [{total_collected}/{args.num_episodes}] "
                      f"success rate: {success_rate:.1%}")

        f.attrs["total_collected"] = total_collected
        f.attrs["success_rate"] = total_success / max(total_collected, 1)

    env.close()

    success_rate = total_success / max(total_collected, 1)
    print(f"\n=== Stage 2 Complete ===")
    print(f"  Collected : {total_collected} episodes")
    print(f"  Success   : {total_success} ({success_rate:.1%})")
    print(f"  Dataset   : {output_path}")

    if success_rate < args.min_success_rate:
        print(f"\nWARNING: Success rate {success_rate:.1%} < {args.min_success_rate:.1%}")
        print("Consider training RL policy for more iterations before collecting data.")

    print(f"\nNext: train DexGen controller")
    print(f"  python scripts/train_dexgen.py --data {output_path}")


if __name__ == "__main__":
    main()
