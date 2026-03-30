# DexGen

Reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Paper: [arXiv](https://arxiv.org/abs/2404.08603) | Project: [zhaohengyin.github.io/dexteritygen](https://zhaohengyin.github.io/dexteritygen/)

## Pipeline Overview

```
Stage 0  Grasp Generation       →  data/grasp_graph.pkl
Stage 1  RL Policy Training     →  logs/rl/shadow_anygrasp_v1/
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

Two methods are available:

| Method | Flag | Description |
|--------|------|-------------|
| Legacy | `--grasp_method legacy` (default) | Heuristic sampling + NFO filter + RRT expansion |
| DexGraspNet | `--grasp_method dexgraspnet` | Differentiable optimization via simulated annealing |

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

`high_precision` preset values:

| Parameter | default | high_precision |
|-----------|---------|----------------|
| `num_seed_grasps` | 300 | 2000 |
| `num_grasps` (RRT target) | 300 | 1000 |
| `fast_nfo` | False | False |
| `isaac_refine` | True | True (forced) |
| `isaac_refine_batch_envs` | 16 | 32 |

**Multi-finger generation:**

`--max_num_fingers N`을 지정하면 2부터 N까지의 모든 손가락 개수에 대해 자동으로 grasp를 생성합니다.

| Flag | 생성되는 finger count |
|------|----------------------|
| `--max_num_fingers 5` | 2, 3, 4, 5 |
| `--num_fingers 4` | 4 (단일) |
| `--finger_counts 2,3,5` | 2, 3, 5 (지정) |

각 `(object × finger_count)` 조합마다 별도 GraspGraph가 생성되며, 이름은 `"cube_0.035_f2"`, `"cube_0.035_f3"` 등으로 태깅됩니다.

**Multi-finger + High-precision (권장):**

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless \
    --max_num_fingers 5 \
    --generation_preset high_precision
```

이 명령 하나로 모든 shape × size × finger_count 조합(예: 3 shape × 3 size × 4 finger = 36개 graph)을 고품질로 생성합니다. object별로 체크포인트가 저장되므로 중간에 중단돼도 결과가 보존됩니다.

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
| Fingertip links | `rh_fftip`, `rh_mftip`, `rh_rftip`, `rh_lftip`, `rh_thtip` |

> **Frame convention note:** DexGraspNet stores grasps in the palm body frame (`robot0:palm`),
> while Isaac Lab's articulation root is `robot0_forearm`. The reset logic resolves this mismatch
> by always computing object placement from fingertip positions via SVD rigid alignment,
> bypassing stored frame-dependent transforms entirely.

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
- Action delay: 0-2 steps

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

### View A Saved Checkpoint

First confirm that a checkpoint exists:

```bash
find logs/rl/allegro_anygrasp_v2 -type f | grep model_
```

Then load the saved policy in non-headless inference mode:

```bash
DISPLAY=:1 /workspace/IsaacLab/isaaclab.sh -p scripts/view_rl_checkpoint.py \
    --checkpoint logs/rl/allegro_anygrasp_v2/checkpoints/model_1000.pt \
    --grasp_graph data/cube_graph/grasp_graph.pkl \
    --grasp_graph data/sphere_graph/grasp_graph.pkl \
    --grasp_graph data/cylinder_graph/grasp_graph.pkl \
    --num_envs 16
```

This viewer runs policy inference only. It does not resume training.

## Stage 2: Dataset Collection

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/collect_data.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt \
    --num_episodes 50000
```

Output: `data/dataset.h5`

## Stage 3: DexGen Controller

Trains a keypoint diffusion model and inverse dynamics model on the collected dataset.

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_dexgen.py --data data/dataset.h5
```

| Component | Architecture | Input | Output |
|-----------|-------------|-------|--------|
| Keypoint Diffusion | DDPM (512-dim, 4 layers, 100 steps) | condition [k_start, k_goal] | T x 12-dim trajectory |
| Inverse Dynamics | MLP (256-dim, 3 layers) | k_t + k_{t+1} + robot_state (56-dim) | 16-dim joint actions |

Output: `logs/dexgen/`

## Monitoring with TensorBoard

The Docker container runs with `network_mode: host`, meaning container ports are directly accessible on the host machine without port mapping.

**From host (outside the container):**

```bash
# Install tensorboard on host if not already installed
pip install tensorboard

# Point to the logs directory (bind-mounted from container)
tensorboard --logdir=logs/rl/shadow_anygrasp_v1 --port=6006
```

Since the repo is bind-mounted into the container at `/workspace/dexgen`, the `logs/` directory is shared between host and container. Running TensorBoard from the host avoids needing to install it inside the Isaac Sim container.

**From inside the container** (if preferred):

```bash
pip install tensorboard
tensorboard --logdir=/workspace/dexgen/logs/rl/shadow_anygrasp_v1 --port=6006 --bind_all
```

Access at: `http://localhost:6006`

**Key metrics to watch:**
- `solve_mean_err` -- object placement error at reset (should be < 0.01 m)
- `rewards/fingertip_tracking` -- dense per-fingertip distance reward
- `rewards/grasp_success` -- sparse success rate (all tips within 1 cm)

## Repository Structure

```
dexgen_repo/
├── configs/
│   ├── shadow_hand.yaml              # Shadow Hand kinematics (5 fingers, 22 DOF)
│   ├── grasp_generation.yaml         # Stage 0 settings
│   ├── rl_training.yaml              # Stage 1 PPO + DR config
│   └── dexgen.yaml                   # Stage 3 diffusion + inverse dynamics
├── docker/
│   ├── Dockerfile                    # Isaac Sim 5.1.0 + Lab v2.3.2
│   ├── docker-compose.yml            # GPU passthrough, host network, bind mount
│   └── run.sh                        # Container management CLI
├── envs/
│   ├── anygrasp_env.py               # ManagerBasedRLEnvCfg (scene, obs, actions, rewards)
│   └── mdp/
│       ├── observations.py        # Actor (107d) + Critic (138d) obs functions
│       ├── rewards.py             # 8 reward terms
│       ├── events.py              # Reset logic (grasp → NN goal), terminations
│       └── domain_rand.py         # Physics/action/noise randomization
├── grasp_generation/
│   ├── grasp_sampler.py              # Heuristic grasp sampling on primitives
│   ├── net_force_optimization.py     # NFO quality scoring (wrench space)
│   ├── rrt_expansion.py              # RRT graph expansion + GraspGraph class
│   ├── dexgraspnet_adapter.py        # DexGraspNet differentiable optimization adapter
│   └── mesh_export.py                # Mesh I/O utilities
├── models/
│   ├── diffusion.py                  # Keypoint diffusion model (Stage 3)
│   └── inverse_dynamics.py           # Inverse dynamics model (Stage 3)
├── scripts/
│   ├── run_grasp_generation.py    # Stage 0 entry point
│   ├── train_rl.py                # Stage 1 entry point
│   ├── view_rl_checkpoint.py      # Stage 1 checkpoint viewer
│   ├── collect_data.py            # Stage 2 entry point
│   └── train_dexgen.py            # Stage 3 entry point
├── data/                          # Generated grasp graphs and datasets
├── requirements.txt               # DexGen-specific Python deps
└── setup_isaaclab.sh              # One-command Docker setup
```

## Container Management

```bash
./docker/run.sh build    # Build image (~20 GB, takes 20-40 min)
./docker/run.sh up       # Start container (detached)
./docker/run.sh exec     # Open shell in container
./docker/run.sh down     # Stop and remove
./docker/run.sh logs     # View container logs
./docker/run.sh status   # Check container state
```

All pipeline commands run inside the container at `/workspace/dexgen`.

## Troubleshooting

**`GraspGraph not found`** — Run Stage 0 first and pass the generated `grasp_graph.pkl` path(s) to Stage 1.

**GUI not appearing** -- Check `echo $DISPLAY` and `echo $XAUTHORITY` on host, then `./docker/run.sh down && ./docker/run.sh up`.

**GPU OOM** -- Multiple Isaac processes may hold VRAM. Restart the container: `./docker/run.sh down && ./docker/run.sh up`.

**Smoke test (headless):**

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless --shapes cube --num_sizes 1 --num_grasps 12

/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 1 --max_iterations 1 --headless
```
