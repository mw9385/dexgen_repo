# DexGen — Sharpa Wave Hand

In-hand object reorientation with the **Sharpa Wave Hand** (22 DOF) in Isaac Lab.

Based on:
- [DeXtreme (Handa et al., 2023)](https://arxiv.org/abs/2210.13702) — reward + observation structure
- [DexterityGen](https://arxiv.org/abs/2502.04307) — grasp graph + curriculum RL
- [sharpa-rl-lab](https://github.com/sharpa-robotics/sharpa-rl-lab) — Sharpa Hand env + tactile temporal stacking

## Pipeline

```
Stage 0   gen_grasp.py    →  data/sharpa_grasp_*.npy
Stage 1   train_rl.py     →  logs/rl/sharpa_anygrasp_v1/
          evaluate.py     →  policy playback / metrics
```

## Quick Start

```bash
# Alias (optional)
echo 'alias ilab="/workspace/IsaacLab/isaaclab.sh -p"' >> ~/.bashrc && source ~/.bashrc

# Stage 0: Generate grasp cache (per shape×size, random object orientation)
ilab scripts/gen_grasp.py --shapes cube --sizes 0.06 --num_grasps 10000 --num_envs 4096 --headless

# Stage 1: Train RL (multiple grasp files → multi-object)
ilab scripts/train_rl.py \
    --grasp_graph data/sharpa_grasp_cube_060.npy \
    --grasp_graph data/sharpa_grasp_sphere_060.npy \
    --grasp_graph data/sharpa_grasp_cylinder_060.npy \
    --num_envs 4096 --headless

# Evaluate
ilab scripts/evaluate.py \
    --checkpoint logs/rl/sharpa_anygrasp_v1/<run>/nn/last.pth \
    --num_envs 16 --num_episodes 50
```

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
5. Validate: all fingertips < 0.1m from object + ≥3 fingers with contact > 0.5N
6. Episode survives → save state

```bash
# Per shape×size (one file each)
ilab scripts/gen_grasp.py --shapes cube     --sizes 0.06 --num_grasps 10000 --headless
ilab scripts/gen_grasp.py --shapes sphere   --sizes 0.06 --num_grasps 10000 --headless
ilab scripts/gen_grasp.py --shapes cylinder --sizes 0.06 --num_grasps 10000 --headless

# Output: data/sharpa_grasp_{shape}_{size_mm}.npy
# Format: (N, 29) = [joint_pos(22) | obj_pos_hand(3) | obj_quat_hand(4)]
```

### Dataset analysis

```bash
python scripts/analyze_grasp_graph.py --path data/sharpa_grasp_cube_060.npy
```

Reports orientation / position distribution and effective kNN neighbor
count at various curriculum thresholds.

## Stage 1: RL Training

### Observation (309 dims, symmetric actor = critic)

**Temporal block** (3 frames × 86 dims = 258) — sharpa-rl-lab style:
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
- Control rate: 20 Hz (240 Hz sim / decimation 12)

### Reward (DeXtreme, Handa et al. 2023)

```
r_t = -10   × ||pos_err||                distance (penalty)
    +  1   × 1/(|rot_err| + 0.1)         rotation alignment (dense positive)
    +  -0.0002 × Σ(a²)                   action magnitude penalty
    +  -0.01   × Σ(Δa²)                  action smoothness penalty
    +  -0.05   × Σ((v/4)²)               joint velocity penalty
    + 250   × (rot_err < 0.4)            sparse goal bonus
```

- **No drop penalty** — episode termination is the implicit cost (you lose future +250 bonuses)
- **No contact gating** — rotation reward is dense positive, avoids "do nothing" local optimum
- **Goal = orientation only** (no position check)

### Termination

| Condition | Trigger |
|---|---|
| `object_drop` | palm-to-object distance > **0.24 m** (DeXtreme fall_dist) |
| `time_out` | episode length > **10 s** (200 steps at 20 Hz) |

### Reset / Rolling goal

1. **Reset**: sample start grasp from `.npy` + kNN goal with `min_orn ≥ curriculum value`
2. **Rolling goal**: when `rot_err < 0.4` and object not dropped, sample new kNN goal
3. Goals are re-based: `target = actual_pos + (new_goal - old_goal)` (not absolute)

### Curriculum (time-based linear ramp)

```
warmup_ratio = 0.30  →  reach max difficulty at epoch 3000/10000
gravity:   1.0  →  9.81 m/s²   (+0.003/epoch)
min_orn:   0.80 →  3.14 rad    (+0.0008/epoch)
```

Slow and predictable so the policy has time to adapt. Configured under
`training_curriculum` in `configs/rl_training.yaml`.

### PPO (rl_games)

```
Network:  MLP [512, 512] + ELU, symmetric actor/critic
Sigma:    fixed, learned log-sigma init -1.0
Input:    309-dim observation (normalized)
Output:   22-dim action (normalized)

learning_rate:     5e-4 (adaptive)
gamma / tau:       0.99 / 0.95
e_clip:            0.2
entropy_coef:      0.0
bounds_loss_coef:  0.0001
critic_coef:       4.0

horizon_length:    16
minibatch_size:    4096
mini_epochs:       8
max_iterations:    10000
```

### Training

```bash
ilab scripts/train_rl.py \
    --grasp_graph data/sharpa_grasp_cube_060.npy \
    --grasp_graph data/sharpa_grasp_sphere_060.npy \
    --num_envs 4096 --headless
```

### Evaluation

```bash
# Auto-detects checkpoint epoch and applies matching curriculum state
ilab scripts/evaluate.py \
    --checkpoint logs/rl/sharpa_anygrasp_v1/<run>/nn/last.pth \
    --grasp_graph data/sharpa_grasp_cube_060.npy \
    --num_envs 16 --num_episodes 50

# Watch in viewer (4 envs, GUI)
ilab scripts/evaluate.py --checkpoint <path> --num_envs 4
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
    rewards.py              DeXtreme rewards (dist, rot, penalties, bonus)
    events.py               Reset + rolling goal + time-based curriculum
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
```

## Config

Key settings in `configs/rl_training.yaml`:

```yaml
env:
  action_mode: "absolute"
  actions_moving_average: 0.3
  decimation: 12                    # 20 Hz control

  training_curriculum:
    enabled: true
    warmup_ratio: 0.30              # time-based linear ramp
    start_gravity: 1.0
    end_gravity: 9.81
    min_orn_start: 0.80
    min_orn_end: 3.14

  rewards:
    distance_weight: -10.0
    rotation_weight: 1.0
    rotation_eps: 0.1
    action_penalty_weight: -0.0002
    action_delta_penalty_weight: -0.01
    velocity_penalty_weight: -0.05
    goal_bonus: 250.0
    goal_thresh: 0.4

  terminations:
    object_drop_max_dist: 0.24
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| All episodes end on step 1 | `object_drop_max_dist` too tight — object resting distance from palm exceeds it. Check with `analyze_grasp_graph.py` |
| Goal hits = 0/0 | `min_orn_start` < `goal_thresh` → goals instantly reached. Ensure `min_orn_start > rot_thresh` |
| Reward collapses mid-training | Curriculum advanced too fast. Increase `warmup_ratio` |
| Policy learns "do nothing" | Happens with sparse-only rewards. DeXtreme reward avoids this via dense `1/(rot_err+0.1)` positive signal |
| "performance/" duplicate metrics in TensorBoard | `_FilteredWriter` drops lowercase rl_games metrics, only uppercase `Performance/` passes |
| Slow env startup with multi-object | Set `replicate_physics=False` auto when loading multi-object graphs |
| Grasp gen stuck at 0 | Early steps fail frequently, success rate ramps up over time |
```