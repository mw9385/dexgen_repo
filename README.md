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

Direct pip installation of Isaac Sim produces too many dependency conflicts.
**Use Docker** for a reproducible environment.

### Host Requirements

| Requirement | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
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

---

## Quick Start

```bash
# Start container
./docker/run.sh up

# Verify Isaac Lab + GPU
./docker/run.sh test_allegro

# Open shell inside container
./docker/run.sh exec bash
```

### Full Pipeline (inside container or via run.sh)

```bash
# Stage 0 — Generate grasp set (no GPU needed inside container)
./docker/run.sh gen_grasps --object cube --num_grasps 500

# Stage 1 — Train RL policy
./docker/run.sh train_rl -- --num_envs 512 --headless

# Stage 2 — Collect dataset
./docker/run.sh collect_data

# Stage 3 — Train DexGen controller
./docker/run.sh train_dexgen -- --diffusion_epochs 500
```

Or run manually inside the container:

```bash
./docker/run.sh exec bash
# now inside container:
python scripts/run_grasp_generation.py --object cube --num_grasps 500
python scripts/train_rl.py --grasp_graph data/grasp_graph.pkl --num_envs 512 --headless
python scripts/collect_data.py --checkpoint logs/rl/.../model_30000.pt
python scripts/train_dexgen.py --data data/dataset.h5
```

---

## Project Structure

```
dexgen_repo/
├── setup_isaaclab.sh          # Host setup (Docker + NGC + build)
├── requirements.txt           # DexGen Python deps (installed in Docker)
├── docker/
│   ├── Dockerfile             # Isaac Sim 5.1.0 + Isaac Lab v5.1.0 + DexGen
│   ├── docker-compose.yml     # GPU passthrough, volumes
│   └── run.sh                 # Helper: build / up / exec / pipeline shortcuts
├── configs/                   # YAML configs for all stages
│   ├── allegro_hand.yaml
│   ├── grasp_generation.yaml
│   ├── rl_training.yaml
│   └── dexgen.yaml
├── grasp_generation/          # Stage 0: NFO-based grasp sampling + RRT
├── envs/                      # Stage 1: Isaac Lab AnyGrasp RL environment
│   └── mdp/                   #   observations, rewards, events
├── models/                    # Stage 3: diffusion + inverse dynamics
└── scripts/                   # Entry-point scripts for all stages
```

---

## Stage Details

### Stage 0 – Grasp Generation
- Surface-sample candidate grasps on object mesh
- Score with **Net Force Optimization** (ε-metric, force-closure LP)
- Expand with **RRT** to build a connected `GraspGraph`
- Grasp representation: 4 fingertip positions in object frame (12-dim)

### Stage 1 – RL Training
- Task: transition between arbitrary grasps in the GraspGraph
- Environment: Isaac Lab `ManagerBasedRLEnv`, Allegro Hand (16 DoF)
- Observation: joint pos/vel, fingertip pos, target fingertip pos (object-centric)
- Algorithm: PPO via `rl_games`

### Stage 2 – Dataset Collection
- Roll out trained RL policy on all grasp pairs
- Record `(keypoint_traj, joint_traj, action_traj, robot_state)` per episode
- Save as HDF5

### Stage 3 – DexGen Controller
- **Diffusion model** (DDPM): plans k_{0:T} conditioned on (k_start, k_goal)
- **Inverse dynamics** (MLP): maps (k_t, k_{t+1}, robot_state) → joint action
- `DexGenController` class for deployment on new objects

---

## Configuration

All hyperparameters in `configs/`:
- `grasp_generation.yaml` – NFO, RRT settings
- `rl_training.yaml` – PPO hyperparameters
- `dexgen.yaml` – diffusion + inverse dynamics model settings
- `allegro_hand.yaml` – Isaac Lab baseline AllegroHand settings
