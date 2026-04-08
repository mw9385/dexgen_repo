# DexGen

Reproduction work for **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity** on the **Sharpa Wave Hand** in Isaac Lab.

This repo is currently centered on the Stage 0 -> Stage 1 pipeline:

```text
Stage 0  Grasp generation (.npy, hand-frame pose)  -> data/sharpa_grasp_*.npy
Stage 1  RL training (PPO, Isaac Lab + rl_games)   -> logs/rl/sharpa_anygrasp_v1/
```

Older optimization / `pkl`-based paths are not the active workflow.

## Current Status

- Hand: Sharpa Wave, 22 DOF, 5 fingertips
- Grasp cache format: `.npy`
- Stored grasp state: `joint_pos(22) + object_pos_hand(3) + object_quat_hand(4)`
- RL reset: start grasp sampled from the saved `.npy`
- RL goal: K-nearest-neighbor goal sampled from the same `.npy`, constrained to stay near the sampled start pose
- RL logging/checkpoints: `logs/rl/sharpa_anygrasp_v1`

## Setup

```bash
./setup_isaaclab.sh
./docker/run.sh up
./docker/run.sh exec
```

Inside the container:

```bash
/isaac-sim/python.sh -m pip install transforms3d transformations lxml
```

Working directory in the container:

```bash
/workspace/dexgen
```

## Stage 0: Generate Grasp Cache

Generate Sharpa grasp data and save it directly to `data/` as `.npy`.

Example:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/gen_grasp.py \
    --shape cube \
    --size 0.05 \
    --num_grasps 1000 \
    --headless
```

Output example:

```text
data/sharpa_grasp_cube_050.npy
```

The saved format is:

```text
[joint_pos(22) | object_pos_hand(3) | object_quat_hand(4)]
```

Notes:

- Default output directory comes from [grasp_generation.yaml](/home/mw/ws/dexgen_repo/configs/grasp_generation.yaml) and is currently `data`.
- The `.npy` now stores object pose in **hand frame**. This is required for Stage 1 reset and goal generation to work correctly.

## Stage 1: Train RL

Train PPO from the generated `.npy` grasp cache:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/sharpa_grasp_cube_050.npy \
    --num_envs 4096 \
    --headless
```

Quick sanity check:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/sharpa_grasp_cube_050.npy \
    --num_envs 1 \
    --max_iterations 1 \
    --headless
```

Verified current behavior:

- The `.npy` file is loaded successfully by `train_rl.py`
- Start grasp is sampled from the saved cache
- Goal grasp is selected by KNN from the same cache
- Goal is kept near the start pose in hand-frame pose space
- Checkpoints and summaries are written under `logs/rl/sharpa_anygrasp_v1`

## Observation

The policy and critic both use the same 101D observation:

- joint positions: 22
- joint velocities: 22
- object position in hand frame: 3
- object quaternion in hand frame: 4
- target object position in hand frame: 3
- target object quaternion in hand frame: 4
- object linear velocity in hand frame: 3
- object angular velocity in hand frame: 3
- fingertip contact forces: 15
- last action: 22

## Reset / Goal Logic

Current reset flow:

1. Sample a start grasp from the saved `.npy`
2. Sample a nearby goal grasp from the same file using KNN in pose space
3. Reset the Sharpa hand to the sampled joint configuration
4. Reconstruct the object pose in sim
5. Rebase the target goal pose from the sampled start->goal delta

Rolling goal update:

- When the current goal is reached, a new nearby goal is selected again from the same grasp cache

## Logging

Training outputs are written here:

```text
logs/rl/sharpa_anygrasp_v1/
```

Typical contents:

```text
logs/rl/sharpa_anygrasp_v1/nn/
logs/rl/sharpa_anygrasp_v1/summaries/
```

## Important Files

- [scripts/gen_grasp.py](/home/mw/ws/dexgen_repo/scripts/gen_grasp.py)
- [scripts/train_rl.py](/home/mw/ws/dexgen_repo/scripts/train_rl.py)
- [envs/anygrasp_env.py](/home/mw/ws/dexgen_repo/envs/anygrasp_env.py)
- [envs/mdp/events.py](/home/mw/ws/dexgen_repo/envs/mdp/events.py)
- [configs/grasp_generation.yaml](/home/mw/ws/dexgen_repo/configs/grasp_generation.yaml)
- [configs/rl_training.yaml](/home/mw/ws/dexgen_repo/configs/rl_training.yaml)

## Troubleshooting

| Problem | Action |
|---|---|
| `train_rl.py` goal distance is extremely large | Re-generate the `.npy` so it stores hand-frame pose |
| Logs appear under `runs/` instead of `logs/rl/...` | Use the updated `train_rl.py`; current config writes to `logs/rl/sharpa_anygrasp_v1` |
| Grasp generation is slow at startup | Isaac Sim scene creation and simulation start take time, especially at `4096` envs |
| `ModuleNotFoundError: transformations` | `/isaac-sim/python.sh -m pip install transformations` |
| No grasps appear early in Stage 0 | Let it run; early progress can stay at `0/1000` before filling rapidly |
