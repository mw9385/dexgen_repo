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

Symmetric actor-critic PPO. Both actor and critic receive the same 101-dim observation (all spatial quantities in hand root frame).

Each episode samples a random start grasp and a nearby goal grasp from the grasp graph. The start and goal are distinct from step 0 — the policy must reorient the object toward the goal immediately. When the goal is achieved (pos < 2cm, rot < 0.1rad), a new nearby goal is selected via kNN (rolling goal).

Hand orientation: palm-up base + diverse wrist poses (±15° tilt noise, 5mm position jitter). This forces the policy to learn active prehensile manipulation under varied gravity directions, not just passive balancing.

Object pool: 3-5cm cube/sphere/cylinder (9 variants), mass 50-100g. Sized for single-hand precision grasp and in-hand rotation.

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph_solved.pkl --num_envs 4096 --headless
```

### Observation (101 dims, hand root frame)

| Component | Dims | Description |
|-----------|------|-------------|
| Joint positions (normalized) | 22 | Finger joints, normalized to [-1, 1] |
| Joint velocities (normalized) | 22 | Finger joints, clipped by 5 rad/s |
| Object position | 3 | Current object pos in hand frame |
| Object quaternion | 4 | Current object quat in hand frame |
| Target object position | 3 | Goal object pos in hand frame |
| Target object quaternion | 4 | Goal object quat in hand frame |
| Object linear velocity | 3 | In hand frame |
| Object angular velocity | 3 | In hand frame |
| Fingertip contact forces | 15 | 3-D force per fingertip (5x3), normalized by 10N |
| Last action | 22 | Previous joint position targets |

### Reward Function (DexterityGen Eq. 4-9)

All terms normalized to [-1,1] or [0,1]. Weights control relative importance.

| Term | Range | Weight | Description |
|------|-------|--------|-------------|
| `object_orientation` | (0, 1] | 10.0 | exp(-2 * rot_err) in hand frame (PRIMARY) |
| `object_position` | (0, 1] | 0.5 | exp(-10 * \|\|pos_err\|\|) in hand frame |
| `goal_bonus` | {0, 1} | 10.0 | 1 if pos < 2cm AND rot < 0.1rad |
| `work` | [-1, 0] | 0.01 | -tanh(0.01 * \|torque\| * \|vel\|) |
| `action` | [-1, 0] | 0.01 | -tanh(0.5 * \|\|a\|\|^2) |
| `torque` | [-1, 0] | 0.01 | -tanh(0.005 * \|\|tau\|\|^2) |

### Termination

- `object_drop`: object height < 0.2m (no penalty, just terminates)
- `object_left_hand`: palm-object distance > 20cm
- `no_fingertip_contact`: no fingertip touches object for 30 consecutive steps (~1s)
- `time_out`: episode length limit

### Rolling Goal

When the object reaches the current goal (pos < 2cm, rot < 0.1rad), a new nearby goal is selected via kNN from the grasp graph. Same criteria as `goal_bonus`.

### Reset

Two paths depending on graph type, then common steps:

**Solved graph** (`grasp_graph_solved.pkl`, recommended):
1. Set wrist at default world position
2. Set stored `joint_angles` directly (skip IK)
3. Compute object world pose from stored hand-relative pose

**Unsolved graph** (`grasp_graph.pkl`, fallback):
1. Place object at fixed position near palm
2. Compute wrist from fingertip centroid
3. Adaptive initial joints + per-finger differential IK

**Common steps (both paths):**
4. Palm-up rotation
5. Wrist pose diversity: position jitter (5mm) + tilt noise (±15°)
6. Compute goal object pose as delta(start→goal) applied to actual sim state
   - Start ≠ goal from step 0; policy must reorient immediately

The `has_stored_reset` check requires ALL grasps in the batch to have `joint_angles`, `object_pos_hand`, `object_quat_hand`, and `object_pose_frame == "hand_root"`. Use `solve_grasp_graph.py` to prepare the graph.

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
