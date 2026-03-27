# Base: NVIDIA Isaac Sim 5.1.0 (Ubuntu 24.04 Base)
FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

# 필수 시스템 패키지 설치 (libgl1 적용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    htop \
    tmux \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Isaac Lab (Isaac Sim 5.1.0과 호환되는 v2.3.2 브랜치 명시)
# ------------------------------------------------------------------------------
ARG ISAACLAB_VERSION=v2.3.2
ENV ISAACLAB_DIR=/workspace/IsaacLab

RUN git clone --depth 1 --branch ${ISAACLAB_VERSION} \
    https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_DIR}

WORKDIR ${ISAACLAB_DIR}

ENV TERM=xterm

# isaac sim path (오타 수정: PTYHON -> PYTHON)
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACSIM_PYTHON_EXE=${ISAACSIM_PATH}/python.sh
RUN ln -s ${ISAACSIM_PATH} ${ISAACLAB_DIR}/_isaac_sim

# Install Isaac Lab core + RL frameworks
RUN ./isaaclab.sh --install
RUN ./isaaclab.sh --install rl_games
RUN ./isaaclab.sh --install rsl_rl
RUN ./isaaclab.sh --install skrl

# pip, setuptools, wheel 사전 업데이트 (의존성 충돌 및 빌드 에러 방지)
RUN ${ISAACSIM_PYTHON_EXE} -m pip install --upgrade pip setuptools wheel

# warp is bundled with Isaac Sim but must be pip-installed so isaaclab.sh -p can find it
RUN ${ISAACSIM_PYTHON_EXE} -m pip install --no-cache-dir warp-lang

# ------------------------------------------------------------------------------
# DexGen Python dependencies
# ------------------------------------------------------------------------------
WORKDIR /workspace/dexgen

COPY requirements.txt .
# Install into Isaac Sim Python
RUN ${ISAACSIM_PYTHON_EXE} -m pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Copy DexGen source & Set Environment
# ------------------------------------------------------------------------------
COPY . .

ENV PYTHONPATH=/workspace/dexgen:${ISAACLAB_DIR}/source/isaaclab:${ISAACLAB_DIR}/source/isaaclab_assets:${PYTHONPATH}
ENV ISAACLAB_DIR=${ISAACLAB_DIR}

# Headless rendering (EGL)
ENV DISPLAY=""
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

WORKDIR /workspace/dexgen

CMD ["/bin/bash"]
