# DexGen – DexterityGen Reproduction

Minimal reproduction of the **DexterityGen** pipeline for dexterous in-hand manipulation.

> *DexterityGen: Foundation Controller for Unprecedented Dexterity*
> Yin et al., 2025

## Pipeline Overview

```
Stage 0: Grasp Generation  →  data/grasp_graph.pkl
Stage 1: RL Training       →  logs/rl/allegro_anygrasp/
Stage 2: Dataset Collection →  data/dataset.h5
Stage 3: DexGen Controller  →  logs/dexgen/
```

### Key Insight (from paper)
The DexGen controller operates in **object-centric fingertip space**:
1. A *diffusion model* plans a keypoint trajectory k_{0:T} from start to goal grasp
2. An *inverse dynamics model* maps (k_t, k_{t+1}) → joint actions at runtime
3. This allows generalisation to **new objects** never seen during training

## Requirements

- Ubuntu 22.04, NVIDIA GPU (driver ≥ 525), CUDA 12.x, Python 3.10

```bash
# Install Isaac Lab 5.1.0 + Isaac Sim 5.1.0
./setup_isaaclab.sh

# Additional Python dependencies (no GPU needed for Stage 0 & 3)
pip install trimesh scipy h5py torch torchvision
```

## Quick Start

```bash
# Step 0: Generate grasp set (no GPU required)
python scripts/run_grasp_generation.py --object cube --num_grasps 500

# Step 1: Train RL policy (GPU required)
python scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 512 --headless

# Step 2: Collect dataset
python scripts/collect_data.py \
    --checkpoint logs/rl/allegro_anygrasp/checkpoints/model_30000.pt \
    --num_episodes 50000

# Step 3: Train DexGen controller
python scripts/train_dexgen.py --data data/dataset.h5

# Verify Isaac Lab AllegroHand baseline
python scripts/run_allegro_hand.py --mode test
```

## Project Structure

```
dexgen_repo/
├── setup_isaaclab.sh          # Isaac Lab 5.1.0 installation
├── configs/                   # YAML configs for all stages
├── grasp_generation/          # Stage 0: NFO-based grasp sampling + RRT
├── envs/                      # Stage 1: Isaac Lab AnyGrasp RL environment
│   └── mdp/                   #   observations, rewards, events
├── models/                    # Stage 3: diffusion + inverse dynamics
└── scripts/                   # Run scripts for all stages
```

## Stage Details

### Stage 0 – Grasp Generation
- Surface-sample candidate grasps on object mesh
- Score with **Net Force Optimization** (ε-metric, force-closure LP)
- Expand with **RRT** to build a connected `GraspGraph`
- Grasp representation: 4 fingertip positions in object frame (12-dim)

### Stage 1 – RL Training
- Task: transition between arbitrary grasps in the GraspGraph
- Environment: Isaac Lab `ManagerBasedRLEnv`, Allegro Hand (16 DoF)
- Observation: joint pos/vel, fingertip pos, target fingertip pos (all object-centric)
- Algorithm: PPO via `rl_games`

### Stage 2 – Dataset Collection
- Roll out trained RL policy on all grasp pairs
- Record `(keypoint_traj, joint_traj, action_traj, robot_state)` per episode
- Save as HDF5 for efficient training

### Stage 3 – DexGen Controller
- **Diffusion model** (DDPM): plans k_{0:T} conditioned on (k_start, k_goal)
- **Inverse dynamics** (MLP): maps (k_t, k_{t+1}, robot_state) → joint action
- Combined `DexGenController` class for deployment

## Configuration

All hyperparameters in `configs/`:
- `grasp_generation.yaml` – NFO, RRT settings
- `rl_training.yaml` – PPO hyperparameters
- `dexgen.yaml` – diffusion + inverse dynamics model settings
- `allegro_hand.yaml` – Isaac Lab baseline AllegroHand settings
