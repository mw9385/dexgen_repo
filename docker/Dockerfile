# syntax=docker/dockerfile:1.6

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=5.1.0
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}

SHELL ["/bin/bash", "-lc"]

# Isaac Lab 및 프로젝트 설정
ARG ISAACLAB_TAG=v2.3.2
ENV ISAACSIM_PATH=/isaac-sim \
    ISAACSIM_PYTHON_EXE=/isaac-sim/python.sh \
    ISAACLAB_DIR=/workspace/IsaacLab \
    DEXGEN_DIR=/workspace/dexgen \
    PYTHONUNBUFFERED=1 \
    TERM=xterm \
    DISPLAY=""

USER root

# 필수 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates unzip rsync vim \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Isaac Lab 설치
RUN git clone --depth 1 --branch ${ISAACLAB_TAG} https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_DIR}
WORKDIR ${ISAACLAB_DIR}
RUN ln -sfn /isaac-sim _isaac_sim \
    && ./isaaclab.sh --install

# ------------------------------------------------------------------------------
# DexGen 의존성 주입 (작동이 확인된 구조 유지)
# ------------------------------------------------------------------------------
WORKDIR ${DEXGEN_DIR}
COPY requirements.txt .

# 1. Isaac Sim 내장 패키지 보호를 위해 pip 업데이트를 생략하고 의존성만 설치
# flatdict 4.0.1의 pkg_resources 종속성을 해결하기 위해 빌드 격리 해제 옵션 사용
RUN ${ISAACSIM_PYTHON_EXE} -m pip install --no-cache-dir \
    pyyaml pyzmq einops flaky junitparser pytest pytest-mock \
    && ${ISAACSIM_PYTHON_EXE} -m pip install --no-cache-dir --no-build-isolation flatdict==4.0.1

# 2. 나머지 프로젝트 요구사항 설치
RUN ${ISAACSIM_PYTHON_EXE} -m pip install --no-cache-dir -r requirements.txt

# 소스 복사 및 환경 변수 설정
COPY . .

# DexGraspNet 서브모듈 초기화 (grasp optimization에 필요)
RUN if [ ! -f third_party/DexGraspNet/grasp_generation/mjcf/shadow_hand_wrist_free.xml ]; then \
        git submodule update --init third_party/DexGraspNet || true; \
    fi

# DexGraspNet 의존성 (transforms3d, pytorch_kinematics는 third_party에 번들)
RUN ${ISAACSIM_PYTHON_EXE} -m pip install --no-cache-dir transforms3d>=0.4.1

ENV PYTHONPATH="${DEXGEN_DIR}:${DEXGEN_DIR}/third_party/DexGraspNet/thirdparty/pytorch_kinematics:${ISAACLAB_DIR}/source/isaaclab:${ISAACLAB_DIR}/source/isaaclab_assets:${PYTHONPATH}"

# 편리성을 위한 Alias 설정
RUN printf 'alias isaacpy="/isaac-sim/python.sh"\n' >> /root/.bashrc \
    && printf 'alias islab="cd /workspace/IsaacLab"\n' >> /root/.bashrc \
    && printf 'alias proj="cd /workspace/dexgen"\n' >> /root/.bashrc

WORKDIR ${DEXGEN_DIR}
CMD ["bash"]
