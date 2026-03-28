“””
Reward functions for the AnyGrasp-to-AnyGrasp environment.

All functions follow the Isaac Lab RewardTerm signature:
func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  (num_envs,)

# =======================================================================
DexGen 논문 기반 3축 보상 구조

1. GOAL-RELATED REWARDS (목표 도달)
   object_pose_goal        dense    object position + orientation 추적
   finger_joint_goal       dense    joint-space goal 추적
   fingertip_tracking      dense    fingertip position 추적 (기존 유지)
   grasp_success           sparse   all tips within threshold (기존 유지)
1. STYLE REWARD (조작 스타일)
   fingertip_velocity      penalty  fingertip velocity → 부드러운 조작 유도
1. REGULARIZATION (정규화)
   action_scale            penalty  action 크기 제한
   applied_torque          penalty  torque 최소화
   mechanical_work         penalty  에너지 소비 최소화
   action_rate             penalty  jerk 최소화 (기존 유지)
1. SAFETY (공학적 안정화, 논문 외 추가)
   fingertip_contact       reward   접촉 유지
   object_velocity         penalty  물체 속도 제한
   object_drop             penalty  낙하 페널티
   joint_limit             penalty  관절 한계 페널티
   wrist_height            penalty  테이블 충돌 방지

# =======================================================================
권장 가중치 (configs/rl_training.yaml에 반영)

# Goal-related (핵심)

object_pose_goal:       15.0    ← 가장 중요한 항, 높은 가중치
finger_joint_goal:       5.0
fingertip_tracking:     10.0
grasp_success:          50.0

# Style

fingertip_velocity:     -0.5

# Regularization

action_scale:           -0.005
applied_torque:         -0.001
mechanical_work:        -0.002
action_rate:            -0.01

# Safety

# fingertip_contact:       2.0
object_velocity:        -0.5
object_drop:          -200.0
joint_limit:            -0.1
wrist_height:           -1.0

“””

from **future** import annotations

import torch

from .observations import (
fingertip_positions_in_object_frame,
target_fingertip_positions,
_get_fingertip_body_ids,
_get_num_fingers,
_sensor_force_vectors,
)

# ═══════════════════════════════════════════════════════════

# 1. GOAL-RELATED REWARDS

# ═══════════════════════════════════════════════════════════

def object_pose_goal_reward(env, pos_scale: float = 10.0,
rot_scale: float = 5.0) -> torch.Tensor:
“””
[논문 핵심 - 기존 코드에 없었음]

```
Object pose가 목표 grasp의 object pose에 얼마나 가까운지.
DexGen goal-related reward의 핵심 축.

Position: L2 distance in hand frame
Rotation: quaternion geodesic distance d(q1,q2) = 1 - |<q1,q2>|

reward = exp(-pos_scale * d_pos) + exp(-rot_scale * d_rot)

Returns: (N,)  ∈ [0, 2]
"""
obj = env.scene["object"]

# 현재 object pose
current_pos = obj.data.root_pos_w       # (N, 3)
current_quat = obj.data.root_quat_w     # (N, 4) [w, x, y, z]

# 목표 object pose (env.goal_object_pos, env.goal_object_quat에 저장)
goal_pos = env.goal_object_pos          # (N, 3)
goal_quat = env.goal_object_quat        # (N, 4)

# Position distance
d_pos = torch.norm(current_pos - goal_pos, dim=-1)   # (N,)

# Rotation distance: geodesic on SO(3)
quat_dot = torch.sum(current_quat * goal_quat, dim=-1).abs()
d_rot = 1.0 - quat_dot.clamp(0.0, 1.0)              # (N,)

r_pos = torch.exp(-pos_scale * d_pos)
r_rot = torch.exp(-rot_scale * d_rot)

return r_pos + r_rot
```

def finger_joint_goal_reward(env, scale: float = 5.0) -> torch.Tensor:
“””
[논문 핵심 - 기존 코드에 없었음]

```
Finger joint positions가 목표 grasp의 joint configuration에 가까운지.

fingertip position 추적만으로는 joint configuration ambiguity가 남음.
(같은 fingertip 위치를 만드는 여러 joint 조합 존재)
논문은 goal-related reward에 finger joint positions를 명시적으로 포함.

reward = exp(-scale * ||q - q*||)

Returns: (N,)  ∈ [0, 1]
"""
robot = env.scene["robot"]
current_joints = robot.data.joint_pos          # (N, 16)
goal_joints = env.goal_joint_positions          # (N, 16)

d_joint = torch.norm(current_joints - goal_joints, dim=-1)  # (N,)
return torch.exp(-scale * d_joint)
```

def fingertip_tracking_reward(env, alpha: float = 20.0) -> torch.Tensor:
“””
Dense tracking reward: exp(-α * ||p_i - p*_i||) per fingertip, averaged.

```
Provides a continuous gradient signal throughout the episode.
α = 20 gives ~0.82 at 1 cm distance, ~0.37 at 5 cm.

Returns: (N,)
"""
nf = _get_num_fingers(env)
current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
target = target_fingertip_positions(env).reshape(-1, nf, 3)
dist = torch.norm(current - target, dim=-1)             # (N, F)
return torch.exp(-alpha * dist).mean(dim=-1)            # (N,)
```

def grasp_success_reward(env, threshold: float = 0.01) -> torch.Tensor:
“””
Sparse success bonus: +1 when ALL fingertips are within `threshold`
of their targets simultaneously.

```
threshold = 0.01 m (1 cm) by default.
Returns: (N,) binary float
"""
nf = _get_num_fingers(env)
current = fingertip_positions_in_object_frame(env).reshape(-1, nf, 3)
target = target_fingertip_positions(env).reshape(-1, nf, 3)
dist = torch.norm(current - target, dim=-1)             # (N, F)
return (dist < threshold).all(dim=-1).float()           # (N,)
```

# ═══════════════════════════════════════════════════════════

# 2. STYLE REWARD

# ═══════════════════════════════════════════════════════════

def fingertip_velocity_penalty(env) -> torch.Tensor:
“””
[논문 핵심 - 기존 코드에 없었음]

```
논문의 style reward: fingertip velocity penalty.
부드럽고 다양한 manipulation style을 유도.

기존 object_velocity_penalty와는 다름:
  - object_velocity_penalty: "물체를 던지지 말라" (safety)
  - fingertip_velocity_penalty: "손가락을 부드럽게 움직여라" (style)

penalty = Σ_i ||v_fingertip_i||²

Returns: (N,)  ≥ 0
"""
robot = env.scene["robot"]
tip_ids = _get_fingertip_body_ids(env)

# Fingertip body velocities
# body_lin_vel_w: (N, num_bodies, 3)
tip_vels = robot.data.body_lin_vel_w[:, tip_ids, :]     # (N, F, 3)

# Sum of squared velocities across all fingertips
vel_sq = (tip_vels ** 2).sum(dim=-1)                    # (N, F)
return vel_sq.sum(dim=-1)                               # (N,)
```

# ═══════════════════════════════════════════════════════════

# 3. REGULARIZATION

# ═══════════════════════════════════════════════════════════

def action_scale_penalty(env) -> torch.Tensor:
“””
[논문 핵심 - 기존 코드에 없었음]

```
논문의 regularization: action scale penalty.
큰 action을 억제하여 안전한 범위 내에서 조작.

기존 action_rate_penalty(Δa의 크기)와 다름:
  - action_rate: 연속 action 간 변화량 (jerk)
  - action_scale: action 자체의 크기 (magnitude)

penalty = ||a||²

Returns: (N,)  ≥ 0
"""
current_act = env.extras.get("current_action")
if current_act is None:
    return torch.zeros(env.num_envs, device=env.device)
return (current_act ** 2).sum(dim=-1)                   # (N,)
```

def applied_torque_penalty(env) -> torch.Tensor:
“””
[논문 핵심 - 기존 코드에 없었음]

```
논문의 regularization: applied torque penalty.
관절 토크를 최소화하여 에너지 효율적인 조작 유도.

penalty = ||τ||²

Returns: (N,)  ≥ 0
"""
robot = env.scene["robot"]
torques = robot.data.applied_torque                     # (N, 16)
return (torques ** 2).sum(dim=-1)                       # (N,)
```

def mechanical_work_penalty(env) -> torch.Tensor:
“””
[논문 핵심 - 기존 코드에 없었음]

```
논문의 regularization: work penalty.
토크 × 관절 속도 = 기계적 일률(power)을 최소화.

penalty = Σ |τ_i * q̇_i|

Returns: (N,)  ≥ 0
"""
robot = env.scene["robot"]
torques = robot.data.applied_torque                     # (N, 16)
velocities = robot.data.joint_vel                       # (N, 16)
power = (torques * velocities).abs()                    # (N, 16)
return power.sum(dim=-1)                                # (N,)
```

def action_rate_penalty(env) -> torch.Tensor:
“””
Penalise large changes between consecutive actions (jerk penalty).

```
Encourages smooth, continuous motions rather than jerky behaviour.
Uses L2 norm of action delta normalised by action dimension.

Returns: (N,)  ≥ 0
"""
last_act = env.extras.get("last_action")
current_act = env.extras.get("current_action")
if last_act is None or current_act is None:
    return torch.zeros(env.num_envs, device=env.device)
delta = current_act - last_act                          # (N, 16)
return torch.norm(delta, dim=-1)                        # (N,)
```

# ═══════════════════════════════════════════════════════════

# 4. SAFETY REWARDS (논문 외 공학적 안정화)

# ═══════════════════════════════════════════════════════════

def fingertip_contact_reward(env) -> torch.Tensor:
“””
Reward for maintaining contact between fingertips and the object
during the grasp transition.

```
Uses ContactSensor if available, otherwise returns 0.

Returns: (N,)  ∈ [0, 1]  (fraction of tips in contact)
"""
sensor = env.scene.sensors.get("fingertip_contact_sensor")
if sensor is None:
    return torch.zeros(env.num_envs, device=env.device)

nf = _get_num_fingers(env)
forces = _sensor_force_vectors(sensor)
forces = forces[:, :nf, :]
force_mag = torch.norm(forces, dim=-1)                  # (N, F)
in_contact = (force_mag > 0.5).float()                  # 0.5 N threshold
return in_contact.mean(dim=-1)                          # (N,)
```

def object_velocity_penalty(env, lin_thresh: float = 0.1,
ang_thresh: float = 1.0) -> torch.Tensor:
“””
Penalise excessive object linear and angular velocity.

```
Note: 이것은 논문의 style reward(fingertip velocity)와는 별개.
이 항은 safety 용도로 물체가 던져지는 것을 방지.

Returns: (N,)  ≥ 0
"""
obj = env.scene["object"]
v_lin = torch.norm(obj.data.root_lin_vel_w, dim=-1)     # (N,)
v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)     # (N,)
penalty = torch.relu(v_lin - lin_thresh) + 0.1 * torch.relu(v_ang - ang_thresh)
return penalty
```

def object_drop_penalty(env, min_height: float = 0.2) -> torch.Tensor:
“””
Binary penalty when the object falls below `min_height`.

```
Returns: (N,)  0 or 1
"""
obj = env.scene["object"]
height = obj.data.root_pos_w[:, 2]
return (height < min_height).float()
```

def joint_limit_penalty(env) -> torch.Tensor:
“””
Soft penalty that increases as joints approach their limits.
Activates within 5% of the joint range at either end.

```
Returns: (N,)  ≥ 0
"""
robot = env.scene["robot"]
q = robot.data.joint_pos
q_low = robot.data.soft_joint_pos_limits[..., 0]
q_high = robot.data.soft_joint_pos_limits[..., 1]
q_range = (q_high - q_low).clamp(min=1e-6)
q_norm = (q - q_low) / q_range                         # [0, 1]

penalty = torch.relu(0.05 - q_norm) + torch.relu(q_norm - 0.95)
return penalty.sum(dim=-1)                              # (N,)
```

def wrist_height_penalty(env, min_height: float = 0.1) -> torch.Tensor:
“””
Penalty if the wrist (robot base) drops below `min_height`.

```
Returns: (N,)  ≥ 0
"""
robot = env.scene["robot"]
wrist_z = robot.data.root_pos_w[:, 2]                   # (N,)
return torch.relu(min_height - wrist_z)
```
