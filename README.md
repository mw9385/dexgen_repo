# DexGen

Reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Paper: [arXiv](https://arxiv.org/abs/2404.08603) | Project: [zhaohengyin.github.io/dexteritygen](https://zhaohengyin.github.io/dexteritygen/)

## Pipeline Overview

```
Stage 0  Grasp Generation       →  data/grasp_graph.pkl
Stage 1  RL Policy Training     →  logs/rl/allegro_anygrasp_v2/
Stage 2  Dataset Collection     →  data/dataset.h5
Stage 3  DexGen Controller      →  logs/dexgen/
```

## Quick Start

```bash
# 1. Build & launch container
./setup_isaaclab.sh
./docker/run.sh up && ./docker/run.sh exec

# 2. Generate refined grasps per object (Stage 0)
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless --shapes cube --num_sizes 1 --isaac_refine --output_dir data/cube_graph
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless --shapes sphere --num_sizes 1 --isaac_refine --output_dir data/sphere_graph
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless --shapes cylinder --num_sizes 1 --isaac_refine --output_dir data/cylinder_graph

# 3. Train RL policy (Stage 1)
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/cube_graph/grasp_graph.pkl \
    --grasp_graph data/sphere_graph/grasp_graph.pkl \
    --grasp_graph data/cylinder_graph/grasp_graph.pkl \
    --num_envs 512 --headless
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

## Stage 0: Grasp Generation

Generates a grasp graph by sampling force-closure grasps on primitive objects, then expanding with RRT.

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py --headless
```

**Pipeline:** Sample heuristic seeds → NFO quality filter → RRT expansion → IK joint seeding → Isaac Lab refinement → Save graph

For training-quality graphs, Isaac refinement should be treated as the default path. Running without `--isaac_refine` is mainly useful for quick debug runs.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--shapes` | cube sphere cylinder | Object primitives |
| `--size_min` / `--size_max` | 0.04 / 0.09 | Object size range (m) |
| `--num_sizes` | 3 | Sizes per shape |
| `--num_seed_grasps` | 300 | Initial candidates |
| `--num_grasps` | 300 | Target after RRT |
| `--num_fingers` | from config | Generate one finger count (2 to 5) |
| `--max_num_fingers` | unset | Generate all finger counts from 2..N |
| `--finger_counts` | unset | Explicit list, e.g. `2,3,5` |
| `--generation_preset` | default | `high_precision` enables larger candidate sets and Isaac refinement defaults |
| `--fast_nfo` | off | Fast SVD approximation |
**Output:** `data/grasp_graph.pkl` — each grasp stores `(joint_angles, object_pos_hand, object_quat_hand, fingertip_positions, quality)`.

**Debug run:**

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless --shapes cube --num_sizes 1 --num_seed_grasps 24 --num_grasps 12 --fast_nfo
```

**High-precision generation:**

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless --generation_preset high_precision
```

This preset currently defaults to `num_seed_grasps=2000`, `num_grasps=1000`,
`min_quality=0.01`, `fast_nfo=False`, `isaac_refine=True`,
and `isaac_refine_batch_envs=32`.

**Recommended training workflow:**

- Treat `--isaac_refine` as the default for graphs that will be used for RL training.
- For larger multi-object runs, Isaac refinement is more stable when each object is generated in a separate Stage-0 run.
- In practice, run Stage 0 separately for `cube`, `sphere`, and `cylinder`, each with `--isaac_refine`, then train RL by passing multiple `--grasp_graph` files in a single training command.

Example end-to-end workflow:

```bash
cd /workspace/dexgen

# 1. Generate one refined graph per object
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless \
    --shapes cube \
    --size_min 0.06 --size_max 0.06 --num_sizes 1 \
    --num_fingers 5 \
    --num_seed_grasps 400 \
    --num_grasps 80 \
    --min_quality 0.005 \
    --isaac_refine \
    --output_dir data/cube_graph

/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless \
    --shapes sphere \
    --size_min 0.06 --size_max 0.06 --num_sizes 1 \
    --num_fingers 5 \
    --num_seed_grasps 400 \
    --num_grasps 80 \
    --min_quality 0.005 \
    --isaac_refine \
    --output_dir data/sphere_graph

/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless \
    --shapes cylinder \
    --size_min 0.06 --size_max 0.06 --num_sizes 1 \
    --num_fingers 5 \
    --num_seed_grasps 400 \
    --num_grasps 80 \
    --min_quality 0.005 \
    --isaac_refine \
    --output_dir data/cylinder_graph
```

Example training command with per-object refined graphs:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/cube_graph/grasp_graph.pkl \
    --grasp_graph data/sphere_graph/grasp_graph.pkl \
    --grasp_graph data/cylinder_graph/grasp_graph.pkl \
    --num_envs 16 --headless
```

## Stage 1: RL Training

Trains an AnyGrasp-to-AnyGrasp transition policy using asymmetric actor-critic PPO (rl_games).

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/cube_graph/grasp_graph.pkl \
    --grasp_graph data/sphere_graph/grasp_graph.pkl \
    --grasp_graph data/cylinder_graph/grasp_graph.pkl \
    --num_envs 512 --headless
```

### How It Works

1. **Reset:** Sample a start grasp from the graph, reconstruct `(joint positions, object pose)` directly in the simulator
2. **Goal:** Nearest-neighbor grasp in fingertip space becomes the target
3. **Policy:** Actor sees 107-dim obs → outputs normalized `[-1, 1]` joint targets
4. **Reward:** Dense fingertip tracking + sparse grasp success bonus

### Architecture

| | Input | Network | Output |
|-|-------|---------|--------|
| Actor | 107-dim policy obs | [512, 512, 256, 128] ELU | 24-dim actions |
| Critic | 138-dim privileged obs | [512, 512, 256] ELU | value |

**Policy observations (107):** joint pos (24) + joint vel (24) + fingertip pos (15) + target fingertip pos (15) + contact binary (5) + last action (24)

**Critic adds (31):** object world pose (7) + object velocity (6) + fingertip forces (15) + DR params (3)

### Key Hyperparameters

See `configs/rl_training.yaml` for the full config.

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-4 (adaptive) |
| Gamma / Lambda | 0.99 / 0.95 |
| Horizon | 16 steps |
| Minibatch | 16384 |
| Reward scale | 0.1 |
| Episode length | 10 s |
| Control freq | 30 Hz (120 Hz sim / 4) |

### Reward Weights

| Term | Weight | Type |
|------|--------|------|
| fingertip_tracking | 10.0 | dense exp(-20d) |
| grasp_success | 50.0 | sparse (all tips < 1cm) |
| fingertip_contact | 2.0 | maintain contact |
| action_rate | -0.01 | jerk penalty |
| object_velocity | -0.5 | anti-fling |
| object_drop | -200.0 | drop penalty |
| joint_limit | -0.1 | soft limit |
| wrist_height | -1.0 | table collision |

### Domain Randomization

**Per episode reset:**
- Object mass: U(0.03, 0.30) kg
- Friction: U(0.30, 1.20)
- Restitution: U(0.00, 0.40)
- Joint damping: U(0.01, 0.30)
- Joint armature: U(0.001, 0.03)
- Action delay: 0–2 steps

**Per step (obs noise):**
- Joint position: N(0, 0.005) rad
- Joint velocity: N(0, 0.04)
- Fingertip position: N(0, 0.003) m

### Resume Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --resume logs/rl/allegro_anygrasp_v2/checkpoints/model_10000.pt \
    --grasp_graph data/cube_graph/grasp_graph.pkl \
    --grasp_graph data/sphere_graph/grasp_graph.pkl \
    --grasp_graph data/cylinder_graph/grasp_graph.pkl \
    --num_envs 512 --headless
```

## Stage 2: Dataset Collection

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/collect_data.py \
    --checkpoint logs/rl/allegro_anygrasp_v2/checkpoints/model_30000.pt \
    --num_episodes 50000
```

Output: `data/dataset.h5`

## Stage 3: DexGen Controller

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_dexgen.py --data data/dataset.h5
```

Output: `logs/dexgen/`

## Repository Structure

```
dexgen_repo/
├── configs/
│   ├── allegro_hand.yaml          # Hand kinematics & fingertip links
│   ├── grasp_generation.yaml      # Stage 0 settings
│   ├── rl_training.yaml           # Stage 1 PPO + DR config
│   └── dexgen.yaml                # Stage 3 settings
├── docker/
│   ├── Dockerfile                 # Isaac Sim 5.1.0 + Lab v2.3.2
│   ├── docker-compose.yml         # GPU passthrough, X11, bind mount
│   └── run.sh                     # Container management CLI
├── envs/
│   ├── anygrasp_env.py            # ManagerBasedRLEnvCfg (scene, obs, actions, rewards)
│   └── mdp/
│       ├── observations.py        # Actor (107d) + Critic (138d) obs functions
│       ├── rewards.py             # 8 reward terms
│       ├── events.py              # Reset logic (grasp → NN goal), terminations
│       └── domain_rand.py         # Physics/action/noise randomization
├── grasp_generation/
│   ├── grasp_sampler.py           # Heuristic grasp sampling on primitives
│   ├── net_force_optimization.py  # NFO quality scoring (wrench space)
│   └── rrt_expansion.py           # RRT graph expansion + GraspGraph class
├── models/
│   ├── diffusion.py               # Diffusion policy (Stage 3)
│   └── inverse_dynamics.py        # Inverse dynamics model (Stage 3)
├── scripts/
│   ├── run_grasp_generation.py    # Stage 0 entry point
│   ├── train_rl.py                # Stage 1 entry point
│   ├── collect_data.py            # Stage 2 entry point
│   └── train_dexgen.py            # Stage 3 entry point
├── data/                          # Generated grasp graphs and datasets
├── requirements.txt               # DexGen-specific Python deps
└── setup_isaaclab.sh              # One-command Docker setup
```

## Container Management

```bash
./docker/run.sh build    # Build image
./docker/run.sh up       # Start container (detached)
./docker/run.sh exec     # Open shell in container
./docker/run.sh down     # Stop and remove
./docker/run.sh logs     # View container logs
./docker/run.sh status   # Check container state
```

All pipeline commands run inside the container at `/workspace/dexgen`.

## Troubleshooting

**`GraspGraph not found`** — Run Stage 0 first and pass the generated `grasp_graph.pkl` path(s) to Stage 1.

**GUI not appearing** — Check `echo $DISPLAY` and `echo $XAUTHORITY` on host, then `./docker/run.sh down && ./docker/run.sh up`.

**GPU OOM** — Multiple Isaac processes may hold VRAM. Restart the container: `./docker/run.sh down && ./docker/run.sh up`.

**Smoke test (headless):**

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless --shapes cube --num_sizes 1 --num_grasps 12

/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 1 --max_iterations 1 --headless
```
