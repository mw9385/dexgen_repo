# DexGen Project Context

## 목표
DexterityGen (Yin et al., 2025) 파이프라인을 Isaac Lab 기반으로 구현.
단일 환경에서의 재현에 초점.

## 현재 진행 상황

### Stage 0 (완료 예정): Isaac Lab AllegroHand 환경 검증
- Isaac Lab의 기존 `Isaac-Repose-Cube-Allegro-v0` 태스크로 환경 셋업 확인
- 이후 커스텀 AnyGrasp-to-AnyGrasp 환경 구현 기반으로 활용

### Stage 1 (예정): Grasp Generation
- Net Force Optimization 기반 grasp 샘플링
- RRT 기반 grasp set 확장

### Stage 2 (예정): RL Training
- Isaac Lab 위에서 PPO로 anygrasp-to-anygrasp 학습

### Stage 3 (예정): DexGen Controller
- Diffusion 기반 keypoint motion 생성
- Inverse dynamics로 joint action 변환

## 환경

### Isaac Lab 셋업
- Isaac Lab v2.1.0 (~/IsaacLab)
- Isaac Sim 4.5.0 (pip install)
- RL framework: rl_games (PPO)

```bash
# 설치
./setup_isaaclab.sh

# AllegroHand 환경 테스트 (GPU 머신에서 실행)
python scripts/run_allegro_hand.py --mode test

# AllegroHand 학습
python scripts/run_allegro_hand.py --mode train --num_envs 512 --headless

# 학습된 policy 실행
python scripts/run_allegro_hand.py --mode play --checkpoint logs/rl_games/allegro_hand/model.pth
```

### Isaac Lab AllegroHand 태스크 정보
- **Task**: `Isaac-Repose-Cube-Allegro-v0`
- **DoF**: 16 (Allegro Hand)
- **Goal**: 큐브를 목표 방향으로 재배향 (cube reorientation)
- **Obs**: joint pos/vel, fingertip pos, object pose/vel, goal orientation (~77 dims)
- **Action**: 16 joint position targets
- **Source**: `~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/`

## 파일 구조

```
dexgen_repo/
├── CLAUDE_CODE_CONTEXT.md        # 이 파일
├── README.md
├── setup_isaaclab.sh             # Isaac Lab 설치 스크립트
├── configs/
│   └── allegro_hand.yaml         # AllegroHand 환경/학습 설정
├── scripts/
│   └── run_allegro_hand.py       # AllegroHand 실행 (train/play/test)
└── envs/                         # 커스텀 환경 (Stage 2에서 구현)
```

## 다음 작업
1. GPU 머신에서 `setup_isaaclab.sh` 실행
2. `python scripts/run_allegro_hand.py --mode test` 로 환경 동작 확인
3. `--mode train` 으로 baseline 학습 실행
4. 학습 결과 확인 후 커스텀 AnyGrasp 환경 개발 시작

## 주요 참고 경로 (Isaac Lab 설치 후)

```
~/IsaacLab/
├── scripts/reinforcement_learning/rl_games/
│   ├── train.py          # rl_games 학습 엔트리포인트
│   └── play.py           # 학습된 policy 실행
└── source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/
    ├── inhand_env_cfg.py           # 환경 기본 설정
    ├── mdp/                        # rewards, observations, events
    └── config/
        └── allegro_hand/
            ├── allegro_env_cfg.py           # AllegroHand 환경 설정
            └── rl_games_ppo_cfg.yaml        # rl_games PPO 하이퍼파라미터
```
