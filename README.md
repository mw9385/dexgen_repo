# DexGen

Reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Paper: [arXiv](https://arxiv.org/abs/2404.08603) | Project: [zhaohengyin.github.io/dexteritygen](https://zhaohengyin.github.io/dexteritygen/)

## Pipeline Overview

```
Stage 0  Grasp Generation (DexGraspNet SA)  →  data/grasp_graph.pkl
  0.5    Isaac Refinement (optional)        →  data/grasp_graph.pkl (refined)
Stage 1  RL Policy Training                 →  logs/rl/shadow_anygrasp_v1/
Stage 2  Dataset Collection                 →  data/dataset.h5
Stage 3  DexGen Controller                  →  logs/dexgen/
```

## Quick Start

```bash
# 1. Build & launch container
./setup_isaaclab.sh
./docker/run.sh up && ./docker/run.sh exec

# 2. Install DexGraspNet dependencies (inside container)
/isaac-sim/python.sh -m pip install transforms3d transformations lxml
git submodule update --init third_party/DexGraspNet

# 3. Generate grasps (Stage 0 — no Isaac Sim, full GPU)
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --size_min 0.06 --size_max 0.06

# 4. (Optional) Refine grasps in Isaac Sim (Stage 0.5)
/workspace/IsaacLab/isaaclab.sh -p scripts/refine_grasps.py \
    --grasp_graph data/grasp_graph.pkl --headless

# 5. Visualize grasps
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl --action_mode hold

# 6. Train RL policy (Stage 1)
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 512 --headless
```

## Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 20.04+ |
| GPU | NVIDIA (driver installed) |
| Docker | 20.10+ |
| NVIDIA Container Toolkit | latest |
| NGC account | required for `nvcr.io` login |

Verify: `nvidia-smi`, `docker info`, `docker login nvcr.io`

## Installation

```bash
./setup_isaaclab.sh
```

This handles Docker check, NVIDIA toolkit, NGC login, image build, and GPU verification.

Manual alternative:

```bash
docker login nvcr.io
./docker/run.sh build   # Isaac Sim 5.1.0 + Isaac Lab v2.3.2
./docker/run.sh up
./docker/run.sh exec
```

## Stage 0: Grasp Generation (DexGraspNet-based)

Generates grasp sets using DexGraspNet's Simulated Annealing optimization.
Runs on **pure PyTorch** — no Isaac Sim required — so the full GPU is available.

```bash
# Single object (fast test)
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --size_min 0.06 --size_max 0.06

# All primitives (full pool)
/isaac-sim/python.sh scripts/run_grasp_generation.py

# Custom parameters
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube sphere cylinder --num_grasps 300 \
    --batch_size 512 --num_iterations 5000
```

### Algorithm

Uses DexGraspNet's exact energy formulation with Simulated Annealing + RMSProp:

| Energy | Description | Weight |
|--------|-------------|--------|
| E_fc | Force closure (wrench Jacobian stability) | 1.0 |
| E_dis | Contact distance (fingertips to surface) | 500.0 |
| E_pen | Reverse penetration (object into hand) | 100.0 |
| E_spen | Self-penetration (finger collisions) | 10.0 |
| E_joints | Joint limit violation | 1.0 |
| E_pose | Natural pose deviation (prevents unnatural bends) | 10.0 |

The hand model uses **pytorch_kinematics** (MJCF-based FK from DexGraspNet's
third_party) for accurate Shadow Hand forward kinematics. Object collision
uses analytical SDFs (capsule + box) — no `torchsdf` or `pytorch3d` needed.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--shapes` | cube sphere cylinder | Object primitives |
| `--size_min` / `--size_max` | 0.05 / 0.08 | Object size range (m) |
| `--num_sizes` | 3 | Sizes per shape |
| `--num_grasps` | 300 | Target grasps per object |
| `--batch_size` | 256 | Parallel grasp candidates |
| `--num_iterations` | 5000 | SA iterations per batch |
| `--n_contact` | 4 | Contact points per grasp |
| `--device` | cuda | Compute device |

**Output:** `data/grasp_graph.pkl`

### Quality Tuning

Key parameters in `configs/grasp_generation.yaml`:

```yaml
optimization:
  w_dis: 500.0        # Higher = fingertips closer to surface
  w_pose: 10.0        # Higher = more natural finger poses
  thres_dis: 0.01     # Strict: only grasps within 10mm contact
  num_iterations: 5000 # More iterations = better convergence
```

## Stage 0.5: Isaac Refinement (Optional)

Corrects FK discrepancy between pytorch_kinematics and Isaac Sim by
re-evaluating each grasp in simulation and overwriting stored poses.

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/refine_grasps.py \
    --grasp_graph data/grasp_graph.pkl --headless

# With filtering (keep only best grasps)
/workspace/IsaacLab/isaaclab.sh -p scripts/refine_grasps.py \
    --grasp_graph data/grasp_graph.pkl --keep_top_k 200 --headless
```

| Flag | Default | Description |
|------|---------|-------------|
| `--grasp_graph` | required | Path to grasp_graph.pkl |
| `--batch_envs` | 16 | Isaac Sim parallel environments |
| `--keep_top_k` | None | Keep only top-K lowest-error grasps |

## Visualization

```bash
# Hold mode (freeze at initial grasp pose) — recommended for checking quality
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl --action_mode hold

# Zero actions (hand drifts due to gravity)
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl --action_mode zero

# Random actions
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl --action_mode random
```

| Action Mode | Behavior |
|-------------|----------|
| `hold` | Maintains initial grasp joint positions (frozen pose) |
| `zero` | Sends zero actions (joints drift toward default) |
| `random` | Sends random `[-1, 1]` actions |

## Stage 1: RL Training

Trains an AnyGrasp-to-AnyGrasp transition policy using asymmetric actor-critic PPO (rl_games).

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 512 --headless
```

### How It Works

1. **Reset:** Sample a start grasp from the graph, place the object via SVD rigid alignment of fingertip positions
2. **Goal:** Nearest-neighbor grasp in fingertip space becomes the target
3. **Policy:** Actor sees 107-dim obs → outputs normalized `[-1, 1]` joint targets
4. **Reward:** Dense fingertip tracking + sparse grasp success bonus

### Hand Configuration

| Property | Value |
|----------|-------|
| Hand | Shadow Hand E-Series (right) |
| Fingers | FF, MF, RF, LF, TH (5 fingers) |
| DOF | 22 actuated (FF x4, MF x4, RF x4, LF x5, TH x5) |
| Isaac Lab root | `robot0_forearm` (2 wrist joints: WRJ0, WRJ1) |

### Resume Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --resume logs/rl/shadow_anygrasp_v1/checkpoints/model_10000.pt \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 512 --headless
```

## Stage 2: Dataset Collection

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/collect_data.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt \
    --num_episodes 50000
```

## Stage 3: DexGen Controller

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_dexgen.py --data data/dataset.h5
```

## Dependencies

DexGraspNet integration requires these additional packages (inside Docker):

```bash
/isaac-sim/python.sh -m pip install transforms3d transformations lxml
git submodule update --init third_party/DexGraspNet
```

`pytorch_kinematics` is bundled in `third_party/DexGraspNet/thirdparty/`.
No `torchsdf` or `pytorch3d` needed.

## Container Management

```bash
./docker/run.sh build    # Build image (~20 GB)
./docker/run.sh up       # Start container
./docker/run.sh exec     # Open shell
./docker/run.sh down     # Stop container
```

## Troubleshooting

**`DexGraspNet assets not found`** — Run `git submodule update --init third_party/DexGraspNet`

**`ModuleNotFoundError: transformations`** — Run `/isaac-sim/python.sh -m pip install transformations`

**GPU OOM during grasp generation** — Reduce `--batch_size` (e.g., 256 or 128)

**Object falls from hand in visualization** — Run Isaac refinement (`refine_grasps.py`) or increase `--num_iterations`

**Fingers in unnatural poses** — Increase `w_pose` in `configs/grasp_generation.yaml`
