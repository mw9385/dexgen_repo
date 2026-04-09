# DexGen — Sharpa Wave Hand

In-hand object reorientation with the **Sharpa Wave Hand** (22 DOF) in Isaac Lab.

Based on:
- [DexterityGen](https://arxiv.org/abs/2502.04307) — grasp graph + curriculum RL
- [sharpa-rl-lab](https://github.com/sharpa-robotics/sharpa-rl-lab) — Sharpa Hand env + tactile
- [OpenAI In-Hand Manipulation](https://arxiv.org/abs/1808.00177) — reward function

## Pipeline

```
Stage 0   gen_grasp.py    →  data/sharpa_grasp_*.npy
Stage 1   train_rl.py     →  logs/rl/sharpa_anygrasp_v1/
```

## Quick Start

```bash
# Alias (optional)
echo 'alias ilab="/workspace/IsaacLab/isaaclab.sh -p"' >> ~/.bashrc && source ~/.bashrc

# Stage 0: Generate grasp cache
ilab scripts/gen_grasp.py --shapes cube --sizes 0.05 --num_grasps 1000 --num_envs 4096 --headless

# Stage 1: Train RL
ilab scripts/train_rl.py --grasp_graph data/sharpa_grasp_cube_050.npy --num_envs 512 --headless
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

Based on [sharpa-rl-lab grasp env](https://github.com/sharpa-robotics/sharpa-rl-lab):

1. Default pre-grasp pose + 0.15 random noise
2. Object placed within hand grasp
3. Step physics with gravity cycling (6 directions)
4. Validate: all fingertips < 0.1m + 3+ contacts > 0.5N + rotation < 30°
5. Episode survives → save state

```bash
# Single object
ilab scripts/gen_grasp.py --shapes cube --sizes 0.05 --num_grasps 1000

# Multiple objects
ilab scripts/gen_grasp.py --shapes cube sphere cylinder --sizes 0.04 0.05 0.06

# Output: data/sharpa_grasp_{shape}_{size_mm}.npy
# Format: (N, 29) = [joint_pos(22) | obj_pos_hand(3) | obj_quat_hand(4)]
```

## Stage 1: RL Training

### Observation (199 dims)

Sharpa-rl-lab temporal stacking + DexGen target info:

```
Per-step block (64 dims, × 3 temporal = 192):
  joint_pos_normalized    22   (unscale + noise)
  joint_targets           22   (current position targets)
  tactile_forces           5   (smoothed elastomer contact)
  contact_positions       15   (5×3, tactile frame)

+ Target info (7 dims, non-temporal):
  target_obj_pos_hand      3
  target_obj_quat_hand     4
```

Tactile processing matches sharpa-rl-lab exactly:
smoothing → latency simulation → binary/continuous mode.

### Reward (OpenAI)

```
r_t = (d_t - d_{t+1})           rotation error reduction
    + 1.0  if rot_dist < 0.4    goal bonus
    - 2.0  if object dropped    fall penalty
```

### Reset / Goal

1. Sample start grasp from `.npy` cache
2. KNN goal: nearby grasp with `min_orn ≥ 0.5 rad` (curriculum: 0.5 → 1.5 rad)
3. Rolling goal: when `rot_dist < 0.4 rad`, pick new goal via KNN

### Training

```bash
ilab scripts/train_rl.py \
    --grasp_graph data/sharpa_grasp_cube_050.npy \
    --num_envs 512 \
    --headless
```

## Files

```
scripts/
  gen_grasp.py          Grasp cache generation
  train_rl.py           RL training (PPO + rl_games)
  train_dexgen.py       DexGen controller training

envs/
  anygrasp_env.py       ManagerBasedRLEnv (Sharpa Hand)
  mdp/
    observations.py     Sharpa tactile + temporal stacking
    rewards.py          OpenAI reward (delta + bonus + penalty)
    events.py           Reset + rolling goal + curriculum
    sim_utils.py        FK, IK, palm utilities
    math_utils.py       Quaternion operations
    domain_rand.py      Domain randomization

grasp_generation/
  graph_io.py           .npy/.pkl loader + GraspGraph structures

assets/
  SharpaWave/           Hand USD
  cylinder/             Cylinder USD

configs/
  grasp_generation.yaml
  rl_training.yaml
  dexgen.yaml
```

## Config

Key settings in `configs/rl_training.yaml`:

```yaml
env:
  task: DexGen-AnyGrasp-Sharpa-v0
  action_mode: "delta"
  delta_scale: 0.0625        # 1/16 per control step
  decimation: 12              # 20 Hz control
  gravity_curriculum:
    enabled: true
    start_gravity: 0.05
    end_gravity: 9.81
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Slow env startup | `replicate_physics=False` + multi-object is slow. Use `--num_envs 128` or single object |
| No rolling goal updates | Check `rot_threshold` in `update_rolling_goal` matches `goal_bonus` thresh |
| Reward too high | Reduce `bonus`/`penalty` values in `AnyGraspRewardsCfg` |
| Grasp gen stuck at 0 | Let it run — early steps fail frequently, fills up later |
