# DexGen

Reproduction of **DEXTERITYGEN: Foundation Controller for Unprecedented Dexterity**.

Paper: [arXiv](https://arxiv.org/abs/2502.04307)

## Pipeline

```
Stage 0a   Grasp Generation (Surface RRT)          ->  data/grasp_graph.pkl
Stage 0b   IK Solve + Validation (Isaac Sim)        ->  data/grasp_graph_solved.pkl
Stage 1    RL Policy Training (PPO)                 ->  logs/rl/shadow_anygrasp_v1/
Stage 2    Dataset Collection                       ->  data/dataset.h5
Stage 3    DexGen Controller                        ->  logs/dexgen/
```

## Quick Start

```bash
# Setup
./setup_isaaclab.sh
./docker/run.sh up && ./docker/run.sh exec

# Install dependencies (inside container)
/isaac-sim/python.sh -m pip install transforms3d transformations lxml
git submodule update --init third_party/DexGraspNet

# Stage 0a: Generate contact-set grasp graph (Surface RRT)
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --size_min 0.04 --size_max 0.04

# Stage 0b: Solve IK + validate grasps in Isaac Sim
/isaac-sim/python.sh scripts/solve_grasp_graph.py \
    --input data/grasp_graph.pkl \
    --output data/grasp_graph_solved.pkl

# Visualize
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph_solved.pkl

# Stage 1: Train RL (use solved graph)
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph_solved.pkl --num_envs 4096 --headless
```

## Stage 0: Grasp Generation

### 0a: Surface-Projected RRT (`surface_rrt.py`)

Generates grasp candidates via RRT expansion with all fingertip positions
constrained to the object mesh surface. Builds a connectivity graph for RL.

Output: `data/grasp_graph.pkl` — contact-set graph (fingertip_positions +
contact_normals only, no joint_angles).

```bash
# Full pool (9 objects: cube/sphere/cylinder x 3 sizes)
/isaac-sim/python.sh scripts/run_grasp_generation.py

# Single object
/isaac-sim/python.sh scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --size_min 0.04 --size_max 0.04
```

### 0b: IK Solve + Validation (`solve_grasp_graph.py`)

Converts the contact-set graph into a robot-state graph by solving per-finger
IK in Isaac Sim and validating physics stability.

Per grasp:
1. Place object at fixed position, compute wrist from fingertip arrangement
2. Adaptive initial joint pose based on object size
3. Per-finger differential IK (null-space + SVD fallback)
4. Fingertip error check (reject if mean > 8mm)
5. Physics settle + object stability check (reject if ejected/dropped)
6. Store: `joint_angles`, `object_pos_hand`, `object_quat_hand`, `reset_contact_error`

Output: `data/grasp_graph_solved.pkl` — robot-state graph.

```bash
/isaac-sim/python.sh scripts/solve_grasp_graph.py \
    --input data/grasp_graph.pkl \
    --output data/grasp_graph_solved.pkl \
    --error-threshold 0.008 \
    --vel-threshold 0.3
```

## Stage 1: RL Training

Symmetric actor-critic PPO. Both actor and critic receive the same 161-dim observation.

Hand orientation: palm-up (object rests on palm). Wrist tilt randomization (+/-15 deg) forces the policy to learn active grasping instead of passive balancing.

Object pool: 3-5cm cube/sphere/cylinder (9 variants), mass 20-100g. Sized for single-hand precision grasp and in-hand rotation.

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph_solved.pkl --num_envs 4096 --headless
```

### Observation (161 dims)

| Component | Dims |
|-----------|------|
| Joint positions (normalized) | 22 |
| Joint velocities | 22 |
| Fingertip positions (object frame) | 15 |
| Relative fingertip to goal | 15 |
| Fingertip contact binary | 5 |
| Last action | 22 |
| Target object pos (hand frame) | 3 |
| Target object quat (hand frame) | 4 |
| Target joint angles (normalized) | 22 |
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

Two paths depending on graph type:

**Solved graph** (`grasp_graph_solved.pkl`, recommended):
1. Set wrist at default world position
2. Set stored `joint_angles` directly (skip IK)
3. FK update
4. Compute object world pose from stored `object_pos_hand`/`object_quat_hand` (exact hand-relative pose)
5. Palm-up rotation + tilt noise

**Unsolved graph** (`grasp_graph.pkl`, fallback):
1. Place object at fixed position near palm
2. Compute wrist from fingertip centroid
3. Adaptive initial joints + per-finger differential IK
4. Palm-up rotation + tilt noise

The `has_stored_reset` check requires ALL grasps in the batch to have `joint_angles`, `object_pos_hand`, and `object_quat_hand`. Use `solve_grasp_graph.py` to prepare the graph.

### Resume Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --resume logs/rl/shadow_anygrasp_v1/checkpoints/model_10000.pt \
    --grasp_graph data/grasp_graph_solved.pkl --num_envs 4096 --headless
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
    --grasp_graph data/grasp_graph_solved.pkl

# Hold initial pose
/workspace/IsaacLab/isaaclab.sh -p scripts/visualize_env.py \
    --grasp_graph data/grasp_graph_solved.pkl --action_mode hold
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
| Object falls at reset | Use `grasp_graph_solved.pkl` (run `solve_grasp_graph.py`), or lower `--error-threshold` |
| Low solve rate | Increase RRT `--num_grasps`, or relax `--error-threshold` |
