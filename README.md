# DexGen — Sharpa Wave Hand

In-hand object reorientation with the **Sharpa Wave Hand** (22 DOF) in Isaac Lab.

Based on:
- [DeXtreme (Handa et al., ICRA 2023)](https://arxiv.org/abs/2210.13702) — reward, action, hyperparameters
- [DexterityGen (Yin et al., 2025)](https://arxiv.org/abs/2502.04307) — grasp graph AnyGrasp-to-AnyGrasp
- [sharpa-rl-lab](https://github.com/sharpa-robotics/sharpa-rl-lab) — Sharpa Hand env + tactile temporal stacking

## Pipeline

```
Stage 0   gen_grasp.py    →  data/sharpa_grasp_*.npy
Stage 1   train_rl.py     →  logs/rl/sharpa_anygrasp_v1/
          evaluate.py     →  policy playback / metrics
```

## Quick Start

The Docker container exposes an `isaacpy` alias that runs `/isaac-sim/python.sh`:

```bash
# Stage 0: Generate grasp cache (per shape×size, random object orientation)
isaacpy scripts/gen_grasp.py --shapes cube --sizes 0.06 --num_grasps 10000 --num_envs 4096 --headless

# Stage 1: Train RL (multiple grasp files → multi-object)
isaacpy scripts/train_rl.py \
    --grasp_graph data/sharpa_grasp_cube_060.npy \
    --grasp_graph data/sharpa_grasp_sphere_060.npy \
    --grasp_graph data/sharpa_grasp_cylinder_060.npy \
    --num_envs 16384 --headless

# Evaluate
isaacpy scripts/evaluate.py \
    --checkpoint logs/rl/sharpa_anygrasp_v1/<run>/nn/last.pth \
    --num_envs 16 --num_episodes 50
```

`/workspace/IsaacLab/isaaclab.sh -p <script>` is the equivalent long form.

## Hand

**Sharpa Wave Hand** — 22 DOF, 5 fingers, no wrist joints.

| Finger | DOF | Joints |
|--------|-----|--------|
| Thumb | 5 | CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP |
| Index | 4 | MCP_FE, MCP_AA, PIP, DIP |
| Middle | 4 | MCP_FE, MCP_AA, PIP, DIP |
| Ring | 4 | MCP_FE, MCP_AA, PIP, DIP |
| Pinky | 5 | CMC, MCP_FE, MCP_AA, PIP, DIP |

Contact sensors: 5 elastomer (one per fingertip).

## Stage 0: Grasp Generation

1. Default pre-grasp pose + 0.15 rad random joint noise
2. Object spawned with **uniform random SO(3) orientation** (full rotation coverage)
3. Small position jitter (±1 cm) on object spawn
4. Step physics with 6-direction gravity cycling (40 steps each)
5. Validate: all fingertips < 0.1 m from object + ≥3 fingers with contact > 0.5 N
6. Episode survives → save state

```bash
# Per shape×size (one file each)
isaacpy scripts/gen_grasp.py --shapes cube     --sizes 0.06 --num_grasps 10000 --headless
isaacpy scripts/gen_grasp.py --shapes sphere   --sizes 0.06 --num_grasps 10000 --headless
isaacpy scripts/gen_grasp.py --shapes cylinder --sizes 0.06 --num_grasps 10000 --headless

# Output: data/sharpa_grasp_{shape}_{size_mm}.npy
# Format: (N, 29) = [joint_pos(22) | obj_pos_hand(3) | obj_quat_hand(4)]
```

### Dataset analysis

```bash
isaacpy scripts/analyze_grasp_graph.py --path data/sharpa_grasp_cube_060.npy
```

Reports orientation / position distribution and effective kNN neighbor
count at various `min_orn` thresholds.

## Stage 1: RL Training

### Observation — 309 dims (symmetric actor = critic)

**Temporal block** (3 frames × 86 dims = 258) — sharpa-rl-lab style stacking:
```
joint_pos_normalized    22   ([-1,1] + noise)
joint_vel_normalized    22   (÷5.0, clamp [-1,1])
joint_targets           22   (current action targets)
sensed_contacts          5   (smoothed force magnitude per fingertip)
contact_positions       15   (5 × 3D contact point in tactile frame)
```

**Non-temporal** (51 dims):
```
last_action             22   (previous step's action)
object_pose_hand         7   (obj pos+quat in hand frame)
object_vel_hand          6   (lin vel + ang vel × 0.2)
target_obj_pos_hand      3
target_obj_quat_hand     4
goal_relative_rotation   4   (quat: object → target)
rotation_distance        2   (current + best rot error)
dr_params                3   (mass, friction, damping — normalised)
```

### Action (absolute joint position, DeXtreme-style)

```
policy output ∈ [-1, 1]²² → JointPositionToLimitsAction → joint targets
smoothed = 0.3 × action + 0.7 × prev_action            (fixed EMA)
```

- `action_mode: absolute`, `action_scale: 1.0` (full joint range)
- `actions_moving_average: 0.3` (fixed, no annealing)
- Control rate: **30 Hz** (240 Hz sim / decimation 8)

### Reward (DeXtreme, exact weights)

```
r_t = -10.0   × ||pos_err||                distance (penalty)
    +  1.0   × 1/(|rot_err| + 0.1)         rotation alignment (dense positive)
    +  -0.0001 × Σ(a²)                     action magnitude penalty
    +  -0.01   × Σ(Δa²)                    action smoothness penalty
    +  -0.05   × Σ((v/4)²)                 joint velocity penalty
    + 250.0   × (rot_err < 0.4 rad)        sparse goal bonus
```

- **No drop penalty** — episode termination is the implicit cost (you lose future +250 bonuses)
- **No contact gating** — rotation reward is dense positive
- **Goal condition = orientation only** (`rot_err < 0.4 rad`) — DeXtreme exact
- All values are wall-clock normalised via Isaac Lab's `dt` multiplication

### Termination

| Condition | Trigger |
|---|---|
| `object_drop` | palm-to-object distance > **0.24 m** (DeXtreme `fall_dist`) |
| `time_out` | episode length > **8 s** (240 steps at 30 Hz) |

### Reset / Rolling goal

1. **Reset**: sample start grasp from `.npy` + kNN goal with `min_orn ≥ 0.7 rad`
2. **Rolling goal**: when `rot_err < 0.4` and object not dropped, sample a new
   kNN goal with the same `min_orn` constraint
3. Goal poses are absolute (start / goal grasps come from the same graph)
4. **No curriculum** — gravity and `min_orn` are fixed for the entire run

### PPO (rl_games)

```
Network:  MLP [512, 512] + ELU, symmetric actor/critic (no LSTM)
Sigma:    fixed, log-sigma init -1.0
Input:    309-dim observation (normalized)
Output:   22-dim action (normalized)

learning_rate:     5e-4 (adaptive)
gamma / tau:       0.998 / 0.95
e_clip:            0.2
entropy_coef:      0.0
bounds_loss_coef:  0.005
critic_coef:       4.0

horizon_length:    16
minibatch_size:    16384   (batch = num_envs × horizon)
mini_epochs:       4
max_iterations:    50000
```

LSTM is disabled (`use_rnn: false`): temporal stacking already provides
short-term history and the sim observation includes ground-truth
velocities / contact forces, so the MDP is fully observable. Enable LSTM
later for sim-to-real where partial observation matters.

### Training

```bash
isaacpy scripts/train_rl.py \
    --grasp_graph data/sharpa_grasp_cube_060.npy \
    --grasp_graph data/sharpa_grasp_sphere_060.npy \
    --num_envs 16384 --headless
```

### Evaluation

```bash
# Auto-detects checkpoint epoch and applies matching curriculum state
isaacpy scripts/evaluate.py \
    --checkpoint logs/rl/sharpa_anygrasp_v1/<run>/nn/last.pth \
    --grasp_graph data/sharpa_grasp_cube_060.npy \
    --num_envs 16 --num_episodes 50

# Watch in viewer (4 envs, GUI)
isaacpy scripts/evaluate.py --checkpoint <path> --num_envs 4
```

## Files

```
scripts/
  gen_grasp.py              Grasp cache generation (one shape×size per run)
  analyze_grasp_graph.py    Orientation / position distribution analysis
  train_rl.py               RL training (PPO + rl_games)
  evaluate.py               Policy playback + metrics
  train_dexgen.py           DexGen controller training (Stage 2)

envs/
  anygrasp_env.py           ManagerBasedRLEnv config (Sharpa Hand)
  mdp/
    observations.py         Temporal stacking + tactile
    rewards.py              DeXtreme rewards (dist, rot, action, velocity, bonus)
    events.py               Reset + rolling goal + (no-op) curriculum
    sim_utils.py            FK, IK, palm body lookup
    math_utils.py           Quaternion ops, rotation noise
    domain_rand.py          DR (object/robot physics, action delay)

grasp_generation/
  graph_io.py               .npy loader + MultiObjectGraspGraph

assets/
  SharpaWave/               Hand USD
  cylinder/                 Cylinder USD

configs/
  grasp_generation.yaml
  rl_training.yaml
  dexgen.yaml

docker/
  Dockerfile                Exposes `isaacpy`, `islab`, `proj` aliases
```

## Config

Key settings in `configs/rl_training.yaml`:

```yaml
env:
  action_mode: "absolute"
  actions_moving_average: 0.3
  decimation: 8                      # 30 Hz control

  training_curriculum:
    enabled: false                   # fixed values, no ramp
    start_gravity: 9.81
    end_gravity: 9.81
    min_orn_start: 0.70              # kNN goal min orientation distance
    min_orn_end: 0.70

  rewards:
    distance_weight: -10.0
    rotation_weight: 1.0
    rotation_eps: 0.1
    action_penalty_weight: -0.0001
    action_delta_penalty_weight: -0.01
    velocity_penalty_weight: -0.05
    goal_bonus: 250.0
    goal_thresh: 0.4                 # rad

  terminations:
    object_drop_max_dist: 0.24
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| All episodes end on step 1 | `object_drop_max_dist` too tight. Object resting distance from palm exceeds it — check with `analyze_grasp_graph.py` |
| `min_orn_start` < 0.4 | Goals are instantly reached (`min_orn` must stay above `goal_thresh` 0.4 rad) |
| `bounds_loss` keeps climbing | Raise `bounds_loss_coef` (already 0.005) or lower `learning_rate` |
| Policy learns "do nothing" | Check that `rotation_reward` dominates — with DeXtreme weights a stationary object still gets `1/(rot_err+0.1)` baseline; large enough gradient pulls toward the goal |
| "performance/" duplicate metrics in TensorBoard | `_FilteredWriter` drops lowercase rl_games metrics, only uppercase `Performance/` passes |
| Slow env startup with multi-object | `replicate_physics` is auto-set to False for multi-object runs |
| Grasp gen stuck at 0 | Early steps fail frequently; success rate ramps up over time |
