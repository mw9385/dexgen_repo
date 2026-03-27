# DexGen – DexterityGen Reproduction

Minimal reproduction of the **DexterityGen** pipeline for dexterous in-hand manipulation.

> *DexterityGen: Foundation Controller for Unprecedented Dexterity*
> Yin et al., 2025

## Pipeline Overview

```
Stage 0: Grasp Generation   →  data/grasp_graph.pkl
Stage 1: RL Training        →  logs/rl/allegro_anygrasp/
Stage 2: Dataset Collection →  data/dataset.h5
Stage 3: DexGen Controller  →  logs/dexgen/
```

---

## Environment Setup (Docker)

**Base image**: `nvcr.io/nvidia/isaac-sim:5.1.0` + Isaac Lab `v2.3.2`

### Host Requirements

| Requirement | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 / 24.04 |
| NVIDIA GPU driver | ≥ 525.60 |
| Docker | ≥ 24.x |
| NVIDIA Container Toolkit | latest |
| NGC account + API key | [ngc.nvidia.com](https://ngc.nvidia.com) |

### 최초 설정 (1회)

```bash
# Docker + nvidia-container-toolkit 설치, NGC 로그인, 이미지 빌드
./setup_isaaclab.sh
```

또는 수동으로:

```bash
# NGC 로그인
docker login nvcr.io
#   Username: $oauthtoken
#   Password: <NGC API key>

# 이미지 빌드
./docker/run.sh build
```

---

## 사용 방법 — 컨테이너 내부에서 실행

모든 파이프라인 작업은 **컨테이너 안에서** 수행합니다.

### 1. 컨테이너 시작 및 진입

```bash
./docker/run.sh up     # 컨테이너 백그라운드 시작
./docker/run.sh exec   # bash로 진입
```

이후 모든 명령은 컨테이너 내부(`/workspace/dexgen`)에서 실행합니다.

---

### Stage 0 – Grasp Generation

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py
```

옵션:

```bash
# 물체 종류/크기 지정
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --shapes cube sphere cylinder \
    --num_sizes 3 \
    --num_grasps 300

# 손가락 수 변경 (config 파일 또는 CLI)
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --num_fingers 3

# 빠른 테스트 (물체 1개, grasp 50개)
/workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \
    --shapes cube --num_sizes 1 --num_grasps 50 --fast_nfo
```

출력: `data/grasp_graph.pkl`

---

### Stage 1 – RL Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 512 \
    --headless
```

옵션:

```bash
# 환경 수 조절 (GPU 메모리에 따라)
--num_envs 64        # 테스트용
--num_envs 512       # 본 학습

# 학습 재개
--resume logs/rl/allegro_anygrasp/checkpoints/model_10000.pt

# 최대 iteration
--max_iterations 30000
```

출력: `logs/rl/allegro_anygrasp/`

---

### Stage 2 – Dataset Collection

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/collect_data.py \
    --checkpoint logs/rl/allegro_anygrasp/checkpoints/model_30000.pt \
    --num_episodes 50000
```

출력: `data/dataset.h5`

---

### Stage 3 – DexGen Controller Training

```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/train_dexgen.py \
    --data data/dataset.h5
```

출력: `logs/dexgen/`

---

## 컨테이너 관리 (host에서)

```bash
./docker/run.sh build    # 이미지 빌드
./docker/run.sh up       # 컨테이너 시작
./docker/run.sh exec     # bash 진입
./docker/run.sh down     # 컨테이너 중지
./docker/run.sh logs     # 로그 확인
./docker/run.sh status   # 상태 확인
```

---

## 파일 구조

```
dexgen_repo/
├── setup_isaaclab.sh          # 최초 설정 스크립트
├── requirements.txt           # Python 의존성
├── docker/
│   ├── Dockerfile             # Isaac Sim 5.1.0 + Isaac Lab v2.3.2
│   ├── docker-compose.yml     # GPU passthrough, 볼륨 마운트
│   └── run.sh                 # 컨테이너 관리 (build/up/down/exec)
├── configs/
│   ├── grasp_generation.yaml  # Hand 설정 (num_fingers, links), NFO, RRT
│   ├── rl_training.yaml       # PPO 하이퍼파라미터 + DR 범위
│   └── dexgen.yaml            # Diffusion + Inverse Dynamics 설정
├── grasp_generation/          # Stage 0: 파지 샘플링 + RRT 확장
├── envs/                      # Stage 1: Isaac Lab RL 환경
│   └── mdp/                   #   observations, rewards, events, domain_rand
├── models/                    # Stage 3: Diffusion + Inverse Dynamics
└── scripts/                   # 각 Stage 실행 스크립트
```

---

## Configuration

### Hand / 손가락 수 설정 (`configs/grasp_generation.yaml`)

```yaml
hand:
  name: allegro
  num_fingers: 4        # 2 / 3 / 4 / 5
  num_dof: 16
  dof_per_finger: 4
  fingertip_links:
    - link_3.0_tip
    - link_7.0_tip
    - link_11.0_tip
    - link_15.0_tip
```

### Domain Randomization (`configs/rl_training.yaml`)

```yaml
domain_randomization:
  object_physics:
    mass_range:        [0.03, 0.30]
    friction_range:    [0.30, 1.20]
    restitution_range: [0.00, 0.40]
  robot_physics:
    damping_range:     [0.01, 0.30]
    armature_range:    [0.001, 0.03]
  action_delay:
    max_delay: 2
  obs_noise:
    joint_pos_std:     0.005
    joint_vel_std:     0.04
    fingertip_pos_std: 0.003
```

---

## Stage 상세

### Stage 0 – Grasp Generation
- 물체 표면에서 후보 파지점 샘플링 (greedy spacing-aware)
- **Net Force Optimization** (ε-metric, force-closure LP)으로 품질 평가
- **RRT**로 확장 → 물체별 `GraspGraph` 생성
- 전체 물체 통합 → `MultiObjectGraspGraph`

### Stage 1 – RL Training
- Task: GraspGraph 내 임의의 두 grasp 사이를 전환
- **Asymmetric Actor-Critic**:
  - Actor (76 dims): joint pos/vel, fingertip pos, target pos, contact binary, last action
  - Critic (104 dims): actor obs + 실제 물체 상태, 완전 contact force, DR params
- **Domain Randomization**: 물체 물리, 관절 동역학, 액션 딜레이, 관측 노이즈
- **Tactile**: ContactSensorCfg → binary contact (actor) / 3D force (critic)
- 매 에피소드: 랜덤 물체 + 랜덤 wrist 위치

### Stage 2 – Dataset Collection
- 학습된 RL 정책으로 GraspGraph의 모든 grasp 쌍 롤아웃
- `(keypoint_traj, joint_traj, action_traj, robot_state)` 기록 → HDF5

### Stage 3 – DexGen Controller
- **Diffusion model** (DDPM): (k_start, k_goal) 조건부 키포인트 궤적 생성
- **Inverse dynamics** (MLP): (k_t, k_{t+1}, robot_state) → joint action
- `DexGenController`: 새 물체에 바로 배포 가능
