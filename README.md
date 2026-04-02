# DexGen

Reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Paper: [arXiv](https://arxiv.org/abs/2502.04307)

## Pipeline

```
Stage 0    Grasp Generation (DexGraspNet SA)    ->  data/grasp_graph.pkl
Stage 1    RL Policy Training (PPO)             ->  logs/rl/shadow_anygrasp_v1/
Stage 2    Dataset Collection                   ->  data/dataset.h5
Stage 3    DexGen Controller                    ->  logs/dexgen/
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
    --shapes cube --num_sizes 1 --size_min 0.04 --size_max 0.04

# Visualize
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph.pkl

# Stage 1: Train RL
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 4096 --headless
```

## Stage 0: Grasp Generation

DexGraspNet simulated annealing optimization (pure PyTorch, full GPU).
Generates grasp candidates, filters by force closure quality, and builds a kNN graph for RL training.

Energy function (matches PKU-EPIC/DexGraspNet exactly):
```
E = E_fc + 100*E_dis + 100*E_pen + 10*E_spen + 1*E_joints
```

DIP (J0) joints are zeroed during fingertip extraction to match Isaac Sim's joint layout (22 MJCF DOF -> 24 USD DOF mapping).

```bash
# Full pool (9 objects: cube/sphere/cylinder x 3 sizes)
/isaac-sim/python.sh scripts/run_grasp_generation.py

# Single object
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --size_min 0.04 --size_max 0.04
```

### Configuration

Key parameters in `configs/grasp_generation.yaml` (DexGraspNet defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iterations` | 6000 | SA optimization steps per batch |
| `batch_size` | 500 | Parallel grasp candidates |
| `n_contact` | 4 | Contact points per grasp |
| `w_dis` | 100.0 | Contact distance weight |
| `w_pen` | 100.0 | Penetration weight |
| `w_spen` | 10.0 | Self-penetration weight |
| `w_joints` | 1.0 | Joint limit violation weight |
| `thres_fc` | 0.3 | Force closure threshold |
| `thres_dis` | 0.005 | Contact distance threshold (5mm) |
| `thres_pen` | 0.001 | Penetration threshold |

## Stage 1: RL Training

Symmetric actor-critic PPO. Both actor and critic receive the same 132-dim observation.

Hand orientation: palm-down (hand grasps from above). Wrist tilt randomization (+/-15 deg) forces the policy to learn active grasping instead of passive balancing.

Object pool: 3-5cm cube/sphere/cylinder (9 variants), mass 20-100g. Sized for single-hand precision grasp and in-hand rotation.

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl --num_envs 4096 --headless
```

### Observation (132 dims)

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

### Reward Function (DexterityGen Eq. 4-9)

All terms normalized to [-1,1] or [0,1]. Weights control relative importance.

| Term | Range | Weight | Description |
|------|-------|--------|-------------|
| `object_position` | (0, 1] | 1.0 | exp(-20 * \|\|pos_err\|\|^2) in hand frame |
| `object_orientation` | (0, 1] | 1.0 | exp(-10 * rot_err) in hand frame |
| `joint_tracking` | [-1, 0] | 0.5 | -tanh(2 * \|\|q - q_target\|\|) |
| `goal_bonus` | {0, 1} | 5.0 | 1 if pos < 2cm AND rot < 0.1rad |
| `work` | [-1, 0] | 0.001 | -tanh(0.01 * \|torque\| * \|vel\|) |
| `action` | [-1, 0] | 0.001 | -tanh(0.5 * \|\|a\|\|^2) |
| `torque` | [-1, 0] | 0.001 | -tanh(0.005 * \|\|tau\|\|^2) |

### Termination

- `object_drop`: object height < 0.2m (no penalty, just terminates)
- `object_left_hand`: palm-object distance > 20cm
- `time_out`: episode length limit

### Rolling Goal

When the object reaches the current goal (pos < 2cm, rot < 0.1rad), a new nearby goal is selected via kNN from the grasp graph. Same criteria as `goal_bonus`.

### Reset

Object placement uses stored `object_pos_hand` / `object_quat_hand` from grasp optimization (exact hand-object relative pose), not fingertip rigid alignment.

### Resume Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --resume logs/rl/shadow_anygrasp_v1/checkpoints/model_10000.pt \
    --grasp_graph data/grasp_graph.pkl --num_envs 4096 --headless
```

### Evaluate Trained Policy

```bash
# Headless evaluation (metrics only)
/workspace/IsaacLab/isaaclab.sh -p scripts/evaluate_policy.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt \
    --num_episodes 100

# Visual evaluation (opens viewer)
/workspace/IsaacLab/isaaclab.sh -p scripts/evaluate_policy.py \
    --checkpoint logs/rl/shadow_anygrasp_v1/checkpoints/model_30000.pt
```

### Tensorboard Metrics

- `Performance/drop_ratio`: fraction of envs where object dropped (accumulated per epoch)
- `Performance/left_hand_ratio`: fraction of envs where object left hand
- `Performance/success_ratio`: rolling goal updates / num_envs
- `Performance/rolling_goal_updates`: absolute count of goal transitions

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
| Object falls at reset | Check grasp generation quality, increase `--num_iterations` |
