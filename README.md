# DexGen Reproduction

This repository is a reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Project page: <https://zhaohengyin.github.io/dexteritygen/>

This codebase is organized as a 4-stage pipeline:

```text
Stage 0: Grasp Generation      -> data/grasp_graph.pkl
Stage 1: RL Training           -> logs/rl/allegro_anygrasp_v2/
Stage 2: Dataset Collection    -> data/dataset.h5
Stage 3: DexGen Training       -> logs/dexgen/
```

Current implementation highlights:

- Stage 0 stores each grasp as a tuple of `(hand joint position, object pose)`.
- Stage 1 resets directly from the stored grasp tuple.
- Goal grasps are selected from nearby nearest neighbors.
- `num_fingers=2/3/4` is supported.
- Policy actions live in normalized `[-1, 1]`.
- With `action_scale=1.0`, the Allegro hand uses the full soft joint range.

## 1. Requirements

Host requirements:

- Ubuntu 20.04 / 22.04 / 24.04
- NVIDIA GPU
- NVIDIA driver
- Docker
- NVIDIA Container Toolkit
- NGC account + API key

Recommended checks:

- `nvidia-smi` works on the host
- `docker info` works on the host
- `docker login nvcr.io` is already configured

## 2. Installation

The easiest setup path is:

```bash
cd /path/to/dexgen_repo
./setup_isaaclab.sh
```

This script does the following:

- checks Docker
- checks NVIDIA Container Toolkit
- guides NGC login
- builds the Docker image
- verifies GPU access inside the container

Manual setup is also possible:

```bash
docker login nvcr.io
./docker/run.sh build
```

## 3. Start the Container

Start the container and open a shell:

```bash
./docker/run.sh up
./docker/run.sh exec
```

After that, run all pipeline commands inside `/workspace/dexgen` in the container.

Container management commands:

```bash
./docker/run.sh build
./docker/run.sh up
./docker/run.sh down
./docker/run.sh exec
./docker/run.sh logs
./docker/run.sh status
```

## 4. Stage 0: Grasp Generation

Default run:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py
```

Fast smoke test:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless \
    --shapes cube \
    --num_sizes 1 \
    --size_min 0.06 \
    --size_max 0.06 \
    --num_seed_grasps 24 \
    --num_grasps 12 \
    --fast_nfo
```

Generate graphs with different contact counts:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --num_fingers 2

/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --num_fingers 3

/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --num_fingers 4
```

Useful options:

- `--shapes cube sphere cylinder`
- `--size_min`, `--size_max`, `--num_sizes`
- `--num_seed_grasps`
- `--num_grasps`
- `--num_fingers`
- `--fast_nfo`
- `--headless`

Output:

- `data/grasp_graph.pkl`

Current Stage 0 behavior:

1. sample heuristic grasp seeds on the object surface
2. filter stable grasps with NFO
3. expand the grasp set with generalized `GraspRRTExpand`
4. attach heuristic joint seeds
5. refine the grasp inside Isaac Lab
6. save refined `joint_angles`, `object_pos_hand`, `object_quat_hand`, and `fingertip_positions`

The saved graph is directly usable by Stage 1.

## 5. Stage 1: RL Training

Default headless training:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 512 \
    --headless
```

Visual check:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 1 \
    --max_iterations 100
```

Current Stage 1 reset behavior:

1. sample a start grasp from the grasp graph
2. choose a nearby nearest-neighbor grasp as the goal
3. reconstruct the stored `(joint, object pose)` tuple directly
4. start the RL episode with the object already grasped in-hand

Current action behavior:

- action space is normalized `[-1, 1]`
- implemented with `JointPositionToLimitsActionCfg`
- `action_scale=1.0` means full Allegro soft joint range

The current reset reconstruction path has been validated for `num_fingers=2/3/4`.

## 6. Stage 2: Dataset Collection

Collect RL rollouts into a dataset:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/collect_data.py \
    --checkpoint logs/rl/allegro_anygrasp_v2/checkpoints/model_30000.pt \
    --num_episodes 50000
```

Output:

- `data/dataset.h5`

## 7. Stage 3: DexGen Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_dexgen.py \
    --data data/dataset.h5
```

Output:

- `logs/dexgen/`

## 8. Important Config Files

### `configs/grasp_generation.yaml`

Stage 0 grasp generation settings.

Key section:

```yaml
hand:
  name: allegro
  num_fingers: 4
  num_dof: 16
  dof_per_finger: 4
  fingertip_links:
    - index_link_3
    - middle_link_3
    - ring_link_3
    - thumb_link_3
```

### `configs/rl_training.yaml`

Stage 1 RL settings.

Key section:

```yaml
env:
  num_envs: 512
  episode_length_s: 10.0
  action_scale: 1.0
  decimation: 4
```

### `configs/allegro_hand.yaml`

Base Allegro hand settings.

## 9. Repository Layout

```text
dexgen_repo/
├── README.md
├── requirements.txt
├── setup_isaaclab.sh
├── configs/
│   ├── grasp_generation.yaml
│   ├── rl_training.yaml
│   ├── allegro_hand.yaml
│   └── dexgen.yaml
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── run.sh
├── envs/
│   ├── anygrasp_env.py
│   └── mdp/
├── grasp_generation/
├── models/
├── scripts/
│   ├── run_grasp_generation.py
│   ├── train_rl.py
│   ├── collect_data.py
│   └── train_dexgen.py
└── data/
```

## 10. Docker and Dependency Notes

Current dependency layout:

- Isaac Sim comes from `nvcr.io/nvidia/isaac-sim:5.1.0`
- Isaac Lab is cloned and installed at build time with tag `v2.3.2`
- project-specific Python packages are installed from `requirements.txt`

This repository does not currently use `pyproject.toml` or a separate conda environment.

### `requirements.txt`

This file only covers DexGen-side Python dependencies.

- Isaac Sim is not installed from `requirements.txt`
- Isaac Lab is not installed from `requirements.txt`
- PyTorch is already present in the Isaac Sim image

### `docker/Dockerfile`

Current build flow:

1. start from Isaac Sim 5.1.0
2. clone Isaac Lab `v2.3.2`
3. run `./isaaclab.sh --install`
4. install `requirements.txt`
5. bind-mount the repository for live development

### `docker/docker-compose.yml`

Current compose behavior:

- `entrypoint: /bin/bash -lc`
- `command: sleep infinity`
- X11 and `XAUTHORITY` are forwarded for GUI mode
- the repository is bind-mounted to `/workspace/dexgen`

The container stays alive, and actual work is run with `docker exec` or `./docker/run.sh exec`.

## 11. GUI Notes

To launch Isaac Sim with a visible window:

- `DISPLAY` must be valid on the host
- `XAUTHORITY` must be valid on the host
- the container must be recreated if compose changes were made

Restart if needed:

```bash
./docker/run.sh down
./docker/run.sh up
```

GUI RL example:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 1 \
    --max_iterations 100
```

## 12. Troubleshooting

### `GraspGraph not found`

Run Stage 0 first:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py
```

### GUI window does not appear

Check:

- `echo $DISPLAY`
- `echo $XAUTHORITY`
- `./docker/run.sh down && ./docker/run.sh up`

### Isaac / GPU OOM

If multiple Isaac processes accumulate during debugging, GPU memory may stay occupied.

The simplest cleanup is:

```bash
./docker/run.sh down
./docker/run.sh up
```

### Headless smoke test

Minimal Stage 0 and Stage 1 verification:

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --headless \
    --shapes cube \
    --num_sizes 1 \
    --num_grasps 12

/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 1 \
    --max_iterations 1 \
    --headless
```
