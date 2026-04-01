# DexGen

Reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Paper: [arXiv](https://arxiv.org/abs/2404.08603) | Project: [zhaohengyin.github.io/dexteritygen](https://zhaohengyin.github.io/dexteritygen/)

## Pipeline

```
Stage 0    Grasp Generation (DexGraspNet SA)    →  data/grasp_graph.pkl
Stage 1    RL Policy Training (PPO)             →  logs/rl/shadow_anygrasp_v1/
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

# Stage 0: Generate grasps
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

DexGraspNet simulated annealing optimization (pure PyTorch, full GPU). Generates grasp candidates, filters by force closure quality (NFO), and builds a kNN graph for RL training.

DIP (J0) joints are zeroed during fingertip extraction to match Isaac Sim's joint layout (22 MJCF DOF → 24 USD DOF mapping).

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

### Evaluate Trained Policy

```bash
# Headless evaluation (metrics only, fast)
/workspace/IsaacLab/isaaclab.sh -p scripts/evaluate_policy.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt \
    --num_episodes 100

# Visual evaluation (opens viewer)
/workspace/IsaacLab/isaaclab.sh -p scripts/evaluate_policy.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt \
    --num_episodes 20 --no-headless

# View policy playback (no metrics, just visual)
/workspace/IsaacLab/isaaclab.sh -p scripts/view_rl_checkpoint.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt
```

`evaluate_policy.py` reports: success rate, fingertip tracking error (mm), rolling goal updates/episode, drop rate, left-hand rate. DR is disabled for clean measurement.

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
| Object falls in visualization | Increase `--num_iterations` or `--num_grasps` |
