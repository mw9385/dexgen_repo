# ==============================================================================
# DexGen Docker Image
# ==============================================================================
# Base: NVIDIA Isaac Sim 5.1.0 (from NGC)
#
# Prerequisites on host:
#   - Docker >= 24.x
#   - NVIDIA Container Toolkit (nvidia-docker2)
#   - NGC account + docker login (see README)
#
# Build:
#   docker compose -f docker/docker-compose.yml build
#
# Run:
#   ./docker/run.sh up
# ==============================================================================

ARG ISAACSIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

# ------------------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------------------
# Isaac Sim 5.1.0 runs as a non-root user by default — switch to root for apt.
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Isaac Lab v5.1.0
# ------------------------------------------------------------------------------
ARG ISAACLAB_VERSION=v5.1.0
ENV ISAACLAB_DIR=/workspace/IsaacLab

RUN git clone --depth 1 --branch ${ISAACLAB_VERSION} \
    https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_DIR}

WORKDIR ${ISAACLAB_DIR}

# Install Isaac Lab core + RL frameworks
RUN ./isaaclab.sh --install 2>&1 | tee /tmp/isaaclab_install.log
RUN ./isaaclab.sh --install rl_games  2>&1 | tee /tmp/rl_games_install.log
RUN ./isaaclab.sh --install rsl_rl    2>&1 | tee /tmp/rsl_rl_install.log
RUN ./isaaclab.sh --install skrl      2>&1 | tee /tmp/skrl_install.log

# ------------------------------------------------------------------------------
# DexGen Python dependencies
# ------------------------------------------------------------------------------
WORKDIR /workspace/dexgen

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Copy DexGen source
# ------------------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------------------
ENV PYTHONPATH=/workspace/dexgen:${ISAACLAB_DIR}/source/isaaclab:${PYTHONPATH}
ENV ISAACLAB_DIR=${ISAACLAB_DIR}

# Headless rendering (EGL) — override with -e DISPLAY=:0 for GUI
ENV DISPLAY=""
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Isaac Sim kit config
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACSIM_PYTHON_EXE=${ISAACSIM_PATH}/python.sh

# Default working directory
WORKDIR /workspace/dexgen

# Default entrypoint: bash (override for specific tasks)
CMD ["/bin/bash"]
