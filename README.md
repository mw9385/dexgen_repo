# DexGen

Reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Paper: [arXiv](https://arxiv.org/abs/2404.08603) | Project: [zhaohengyin.github.io/dexteritygen](https://zhaohengyin.github.io/dexteritygen/)

## Pipeline

```
Stage 0    Grasp Generation + Isaac Refinement  →  data/grasp_graph.pkl
Stage 1    RL Policy Training                   →  logs/rl/shadow_anygrasp_v1/
Stage 2    Dataset Collection                   →  data/dataset.h5
Stage 3    DexGen Controller                    →  logs/dexgen/
```

## Quick Start

```bash
# Setup
./setup_isaaclab.sh
./docker/run.sh up && ./docker/run.sh exec

# Install dependencies (inside container)
/isaac-sim/python.sh -m pip install transforms3d transformations lxml
git submodule update --init third_party/DexGraspNet

# Stage 0: Generate grasps + refine in Isaac Sim
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --size_min 0.06 --size_max 0.06

# Visualize
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl

# Stage 1: Train RL
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 512 --headless
```

## Stage 0: Grasp Generation

Two-step process:
1. **DexGraspNet optimization** (pure PyTorch, full GPU) — generates grasp candidates
2. **Isaac Sim refinement** (automatic) — corrects FK mismatch, overwrites grasp data

After refinement, all stored data (`joint_angles`, `object_pos_hand`, `object_quat_hand`) is consistent with Isaac Sim's FK, ensuring reward functions get coherent targets during RL training.

```bash
# Full pool (9 objects)
/isaac-sim/python.sh scripts/run_grasp_generation.py

# Single object
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --size_min 0.06 --size_max 0.06

# Custom parameters
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes sphere --batch_size 512 --num_iterations 5000 --num_grasps 300
```

### Configuration

Key parameters in `configs/grasp_generation.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iterations` | 5000 | SA optimization steps per batch |
| `batch_size` | 512 | Parallel grasp candidates |
| `w_dis` | 500.0 | Contact distance weight (higher = tighter contact) |
| `w_pose` | 10.0 | Natural pose weight (prevents unnatural finger bends) |
| `thres_fc` | 3.0 | Force closure threshold |
| `thres_dis` | 0.005 | Contact distance threshold (5mm) |

## Stage 1: RL Training

Symmetric actor-critic PPO with full observation (132 dims).

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 512 --headless
```

### Observation (132 dims, shared by actor and critic)

| Component | Dims |
|-----------|------|
| Joint positions (normalized) | 22 |
| Joint velocities | 22 |
| Fingertip positions (object frame) | 15 |
| Relative fingertip to goal | 15 |
| Fingertip contact binary | 5 |
| Last action | 22 |
| Object position (world) | 3 |
| Object quaternion | 4 |
| Object linear velocity | 3 |
| Object angular velocity | 3 |
| Contact forces | 15 |
| Domain randomization params | 3 |

### Reward Function

| Term | Weight | Description |
|------|--------|-------------|
| `fingertip_tracking` | +8.0 | exp(-20 * dist) per fingertip |
| `grasp_success` | +125.0 | All fingertips within threshold |
| `finger_joint_goal` | +8.0 | exp(-2 * joint error) |
| `object_pose_goal` | +5.0 | exp(-10 * pos_err - 5 * rot_err) |
| `fingertip_contact` | +2.0 | Maintain contact |
| `object_drop` | -100.0 | Object falls below table |
| `object_left_hand` | -50.0 | Object escapes hand |

### Resume Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --resume logs/rl/shadow_anygrasp_v1/checkpoints/model_10000.pt \
    --grasp_graph data/grasp_graph.pkl --num_envs 512 --headless
```

## Visualization

```bash
# Zero actions (default)
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl

# Hold initial pose
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl --action_mode hold
```

## Stage 2 & 3

```bash
# Collect dataset from trained policy
/workspace/IsaacLab/isaaclab.sh -p scripts/collect_data.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt

# Train DexGen controller
/workspace/IsaacLab/isaaclab.sh -p scripts/train_dexgen.py --data data/dataset.h5
```

## Dependencies

```bash
/isaac-sim/python.sh -m pip install transforms3d transformations lxml
git submodule update --init third_party/DexGraspNet
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `DexGraspNet assets not found` | `git submodule update --init third_party/DexGraspNet` |
| `ModuleNotFoundError: transformations` | `/isaac-sim/python.sh -m pip install transformations` |
| GPU OOM during generation | Reduce `--batch_size` |
| Object falls in visualization | Run refinement or increase `--num_iterations` |
