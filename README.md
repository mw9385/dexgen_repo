# DexGen – DexterityGen Reproduction

Minimal reproduction of the **DexterityGen** pipeline for dexterous in-hand manipulation.

> *DexterityGen: Foundation Controller for Unprecedented Dexterity*
> Yin et al., 2025

## Pipeline Overview

```
Stage 0: Grasp Generation   →  data/grasp_graph.pkl
Stage 1: RL Training        →  logs/rl/allegro_anygrasp/
Stage 2: Dataset Collection →  data/dataset.h5
Stage 3: DexGen Controller  →  logs/dexgen/
```

### Key Insight (from paper)
The DexGen controller operates in **object-centric fingertip space**:
1. A *diffusion model* plans a keypoint trajectory k_{0:T} from start to goal grasp
2. An *inverse dynamics model* maps (k_t, k_{t+1}) → joint actions at runtime
3. This allows generalisation to **new objects** never seen during training

---

## Environment Setup (Docker)

Direct pip installation of Isaac Sim produces dependency conflicts.
**Use Docker** for a reproducible environment.

### Host Requirements

| Requirement | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 / 24.04 |
| NVIDIA GPU driver | ≥ 525.60 |
| Docker | ≥ 24.x |
| NVIDIA Container Toolkit | latest |
| NGC account + API key | [ngc.nvidia.com](https://ngc.nvidia.com) |

### One-time Setup

```bash
# Installs Docker, nvidia-container-toolkit, logs in to NGC, builds image
./setup_isaaclab.sh
```

This script:
1. Installs Docker (if missing)
2. Installs `nvidia-container-toolkit`
3. Guides you through NGC login (needed to pull Isaac Sim 5.1.0)
4. Builds the `dexgen:latest` Docker image (~20 GB, takes 20–40 min first time)

Base image: `nvcr.io/nvidia/isaac-sim:5.1.0` + Isaac Lab `v2.3.2`

---

## Quick Start

```bash
# Build image (first time only)
./docker/run.sh build

# Start container
./docker/run.sh up

# Verify Isaac Lab + GPU
./docker/run.sh test_allegro
```

### Full Pipeline

```bash
# Stage 0 — Generate grasp set
./docker/run.sh gen_grasps

# Stage 1 — Train RL policy
./docker/run.sh train_rl -- --num_envs 512 --headless

# Stage 2 — Collect dataset
./docker/run.sh collect_data

# Stage 3 — Train DexGen controller
./docker/run.sh train_dexgen
```

Or run manually inside the container:

```bash
./docker/run.sh exec bash

# Inside container — use isaaclab.sh -p (standard Isaac Lab way)
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 512 --headless
/workspace/IsaacLab/isaaclab.sh -p scripts/collect_data.py \
    --checkpoint logs/rl/allegro_anygrasp/checkpoints/model_30000.pt
/workspace/IsaacLab/isaaclab.sh -p scripts/train_dexgen.py \
    --data data/dataset.h5
```

Output files appear directly on the host:
- `data/grasp_graph.pkl` — Stage 0 output
- `logs/rl/` — Stage 1 checkpoints & tensorboard logs
- `data/dataset.h5` — Stage 2 dataset
- `logs/dexgen/` — Stage 3 model weights

---

## Project Structure

```
dexgen_repo/
├── setup_isaaclab.sh          # Host setup (Docker + NGC + build)
├── requirements.txt           # DexGen Python deps (installed in Docker)
├── docker/
│   ├── Dockerfile             # Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + DexGen
│   ├── docker-compose.yml     # GPU passthrough, bind mount
│   └── run.sh                 # Helper: build / up / exec / pipeline shortcuts
├── configs/
│   ├── grasp_generation.yaml  # Hand config (num_fingers, links), NFO, RRT
│   ├── rl_training.yaml       # PPO hyperparameters + Domain Randomization ranges
│   └── dexgen.yaml            # Diffusion + inverse dynamics model settings
├── grasp_generation/          # Stage 0: NFO-based grasp sampling + RRT expansion
├── envs/                      # Stage 1: Isaac Lab AnyGrasp RL environment
│   └── mdp/                   #   observations, rewards, events, domain_rand
├── models/                    # Stage 3: diffusion + inverse dynamics
└── scripts/                   # Entry-point scripts for all stages
```

---

## Stage Details

### Stage 0 – Grasp Generation

- Surface-sample candidate grasps on object mesh (greedy spacing-aware)
- Score with **Net Force Optimization** (ε-metric, force-closure LP)
- Expand with **RRT** to build a connected `GraspGraph` per object
- Merge all per-object graphs → `MultiObjectGraspGraph`
- Grasp representation: `num_fingers` fingertip positions in object frame

**Hand / finger count** is set in `configs/grasp_generation.yaml`:

```yaml
hand:
  name: allegro
  num_fingers: 4        # 2 / 3 / 4 / 5
  num_dof: 16
  dof_per_finger: 4
  fingertip_links:
    - link_3.0_tip
    - link_7.0_tip
    - link_11.0_tip
    - link_15.0_tip
```

Or override at runtime:
```bash
./docker/run.sh gen_grasps -- --num_fingers 3
```

### Stage 1 – RL Training

- Task: transition between arbitrary grasps in the GraspGraph
- Environment: Isaac Lab `ManagerBasedRLEnv`, Allegro Hand (16 DoF)
- Algorithm: PPO via `rl_games`
- **Asymmetric Actor-Critic**:
  - Actor (76 dims): joint pos/vel, fingertip pos, target, contact binary, last action
  - Critic (104 dims): actor obs + true object state, full contact forces, DR params
- **Domain Randomization** (configurable in `configs/rl_training.yaml`):
  - Object mass, friction, restitution
  - Joint damping, armature
  - Action delay (0–N steps)
  - Observation noise (joint pos/vel, fingertip pos)
- **Tactile sensing**: ContactSensorCfg on fingertip links → binary contact (actor) + full 3D forces (critic)
- **Random object pool**: cube / sphere / cylinder at multiple sizes
- **Random wrist pose**: hemisphere sampling per episode

### Stage 2 – Dataset Collection

- Roll out trained RL policy on all grasp pairs in the GraspGraph
- Record `(keypoint_traj, joint_traj, action_traj, robot_state)` per episode
- Save as HDF5

### Stage 3 – DexGen Controller

- **Diffusion model** (DDPM): plans k_{0:T} conditioned on (k_start, k_goal)
- **Inverse dynamics** (MLP): maps (k_t, k_{t+1}, robot_state) → joint action
- `DexGenController` class for deployment on new objects

---

## Configuration

### Domain Randomization (`configs/rl_training.yaml`)

```yaml
domain_randomization:
  object_physics:
    mass_range:        [0.03, 0.30]   # kg
    friction_range:    [0.30, 1.20]
    restitution_range: [0.00, 0.40]
  robot_physics:
    damping_range:     [0.01, 0.30]
    armature_range:    [0.001, 0.03]
  action_delay:
    max_delay: 2                       # steps
  obs_noise:
    joint_pos_std:     0.005           # rad
    joint_vel_std:     0.04
    fingertip_pos_std: 0.003           # m
```

### Hand Configuration (`configs/grasp_generation.yaml`)

Supports any dexterous hand — change `num_fingers`, `num_dof`, `dof_per_finger`,
and `fingertip_links` to adapt to Shadow, LEAP, or custom hands.
The env automatically reads `hand` config from `AnyGraspEnvCfg.hand`.
