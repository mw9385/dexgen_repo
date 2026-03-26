# DexGen Project Context

## 목표
DexterityGen (Yin et al., 2025) 파이프라인을 Isaac Lab 5.1.0 기반으로 구현.
단일 환경 (Allegro Hand + cube)에서의 재현에 초점.

## 현재 진행 상황

### Stage 0 ✅ 구현 완료: Grasp Generation
- `grasp_generation/grasp_sampler.py`: 물체 표면 샘플링 + 손가락 배치 휴리스틱
- `grasp_generation/net_force_optimization.py`: NFO 품질 점수 (LP 기반 wrench space)
- `grasp_generation/rrt_expansion.py`: RRT 기반 grasp set 확장 + GraspGraph 생성

### Stage 1 ✅ 구현 완료: RL Training
- `envs/anygrasp_env.py`: Isaac Lab ManagerBasedRLEnv (AnyGrasp-to-AnyGrasp 태스크)
- `envs/mdp/observations.py`: 물체 프레임 기준 fingertip 관측
- `envs/mdp/rewards.py`: 지수 tracking reward + 성공 보너스 + drop 패널티
- `envs/mdp/events.py`: GraspGraph에서 무작위 (start, goal) 쌍으로 reset

### Stage 2 ✅ 구현 완료: Dataset Collection
- `scripts/collect_data.py`: 학습된 RL policy로 trajectory 수집 → HDF5 저장

### Stage 3 ✅ 구현 완료: DexGen Controller
- `models/diffusion.py`: DDPM 기반 keypoint trajectory 생성 (조건: k_start, k_goal)
- `models/inverse_dynamics.py`: (k_t, k_{t+1}, robot_state) → joint action MLP
- `models/inverse_dynamics.py::DexGenController`: 두 모델 결합한 inference controller

## 환경 설정

### Isaac Lab 셋업
- Isaac Lab v5.1.0 (~/IsaacLab)
- Isaac Sim 5.1.0 (pip install)
- RL framework: rl_games (PPO)

```bash
# 설치
./setup_isaaclab.sh

# AllegroHand 기본 환경 테스트 (GPU 머신에서)
python scripts/run_allegro_hand.py --mode test
```

## 전체 파이프라인 실행 순서

```bash
# Step 0: Grasp set 생성 (GPU 불필요)
python scripts/run_grasp_generation.py \
    --object cube --num_grasps 500

# Step 1: RL 학습 (GPU 필요)
python scripts/train_rl.py \
    --grasp_graph data/grasp_graph.pkl \
    --num_envs 512 --headless

# Step 2: Dataset 수집
python scripts/collect_data.py \
    --checkpoint logs/rl/allegro_anygrasp/checkpoints/model_30000.pt \
    --num_episodes 50000

# Step 3: DexGen controller 학습
python scripts/train_dexgen.py \
    --data data/dataset.h5 \
    --diffusion_epochs 500
```

## 파일 구조

```
dexgen_repo/
├── CLAUDE_CODE_CONTEXT.md
├── README.md
├── setup_isaaclab.sh                    # Isaac Lab 5.1.0 설치
├── configs/
│   ├── allegro_hand.yaml                # Isaac-Repose-Cube-Allegro-v0 설정
│   ├── grasp_generation.yaml            # Stage 0 설정
│   ├── rl_training.yaml                 # Stage 1 PPO 설정
│   └── dexgen.yaml                      # Stage 3 모델 설정
├── grasp_generation/
│   ├── __init__.py
│   ├── grasp_sampler.py                 # 표면 샘플링 + Grasp 데이터 구조
│   ├── net_force_optimization.py        # NFO 품질 평가 (LP)
│   └── rrt_expansion.py                 # RRT 확장 + GraspGraph
├── envs/
│   ├── __init__.py
│   ├── anygrasp_env.py                  # Isaac Lab ManagerBasedRLEnv
│   └── mdp/
│       ├── __init__.py
│       ├── observations.py              # 관측 함수 (물체 프레임)
│       ├── rewards.py                   # 보상 함수
│       └── events.py                    # Reset/termination 이벤트
├── models/
│   ├── __init__.py
│   ├── diffusion.py                     # DDPM keypoint 궤적 생성
│   └── inverse_dynamics.py              # inv_dyn MLP + DexGenController
└── scripts/
    ├── run_allegro_hand.py              # Isaac Lab 기본 AllegroHand 환경 테스트
    ├── run_grasp_generation.py          # Stage 0 실행
    ├── train_rl.py                      # Stage 1 실행
    ├── collect_data.py                  # Stage 2 실행
    └── train_dexgen.py                  # Stage 3 실행
```

## 논문 핵심 개념 → 구현 매핑

| 논문 §  | 개념                    | 구현 파일                                      |
|--------|------------------------|-----------------------------------------------|
| §3.1   | Grasp set G = {g_i}    | `grasp_generation/grasp_sampler.py`           |
| §3.1   | NFO quality score ε    | `grasp_generation/net_force_optimization.py`  |
| §3.1   | RRT expansion          | `grasp_generation/rrt_expansion.py`           |
| §3.2   | AnyGrasp-to-AnyGrasp   | `envs/anygrasp_env.py`                        |
| §3.2   | Object-centric obs     | `envs/mdp/observations.py`                    |
| §3.2   | Fingertip tracking rew | `envs/mdp/rewards.py`                         |
| §3.3   | Diffusion planner      | `models/diffusion.py`                         |
| §3.3   | Inverse dynamics       | `models/inverse_dynamics.py`                  |
| §3.3   | DexGen controller      | `models/inverse_dynamics.py::DexGenController`|

## 다음 작업
1. GPU 머신에서 `./setup_isaaclab.sh` 실행
2. `python scripts/run_allegro_hand.py --mode test` 로 Isaac Lab 환경 확인
3. `python scripts/run_grasp_generation.py` 로 grasp set 생성 (GPU 불필요)
4. RL 학습 실행 후 결과 확인
