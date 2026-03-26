#!/usr/bin/env bash
# ==============================================================================
# Isaac Lab Setup Script for DexGen
# ==============================================================================
# This script installs Isaac Sim + Isaac Lab for the AllegroHand environment.
#
# Requirements:
#   - Ubuntu 22.04
#   - NVIDIA GPU with driver >= 525.60
#   - CUDA 12.x
#   - Python 3.10
#
# Isaac Lab AllegroHand task: Isaac-Repose-Cube-Allegro-v0
# ==============================================================================

set -e

ISAACLAB_VERSION="v2.1.0"
ISAACLAB_DIR="${HOME}/IsaacLab"

echo "=== DexGen: Isaac Lab Setup ==="
echo "Installing Isaac Lab ${ISAACLAB_VERSION} to ${ISAACLAB_DIR}"

# ------------------------------------------------------------------------------
# 1. Check prerequisites
# ------------------------------------------------------------------------------
echo "[1/5] Checking prerequisites..."

if ! command -v python3.10 &> /dev/null; then
    echo "ERROR: Python 3.10 is required."
    echo "  sudo apt install python3.10 python3.10-dev python3.10-venv"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU not found. Isaac Lab requires an NVIDIA GPU."
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  GPU Driver: ${DRIVER_VERSION}"

# ------------------------------------------------------------------------------
# 2. Install Isaac Sim via pip (Isaac Sim 4.x supports pip install)
# ------------------------------------------------------------------------------
echo "[2/5] Installing Isaac Sim..."
pip install isaacsim==4.5.0 \
    --extra-index-url https://pypi.nvidia.com \
    isaacsim-rl \
    isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk

# ------------------------------------------------------------------------------
# 3. Clone Isaac Lab
# ------------------------------------------------------------------------------
echo "[3/5] Cloning Isaac Lab ${ISAACLAB_VERSION}..."
if [ -d "${ISAACLAB_DIR}" ]; then
    echo "  ${ISAACLAB_DIR} already exists, pulling latest..."
    cd "${ISAACLAB_DIR}" && git fetch origin && git checkout ${ISAACLAB_VERSION}
else
    git clone --branch ${ISAACLAB_VERSION} \
        https://github.com/isaac-sim/IsaacLab.git \
        "${ISAACLAB_DIR}"
fi

# ------------------------------------------------------------------------------
# 4. Install Isaac Lab
# ------------------------------------------------------------------------------
echo "[4/5] Installing Isaac Lab..."
cd "${ISAACLAB_DIR}"
./isaaclab.sh --install  # installs isaaclab + all extensions

# Install RL frameworks
./isaaclab.sh --install rl_games   # RL Games (used for AllegroHand)
./isaaclab.sh --install rsl_rl     # RSL RL (alternative)
./isaaclab.sh --install skrl       # SKRL (alternative)

# ------------------------------------------------------------------------------
# 5. Verify installation
# ------------------------------------------------------------------------------
echo "[5/5] Verifying Isaac Lab installation..."
python -c "import isaaclab; print(f'Isaac Lab version: {isaaclab.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To test AllegroHand environment:"
echo "  cd ${ISAACLAB_DIR}"
echo "  python scripts/reinforcement_learning/rl_games/train.py \\"
echo "      --task Isaac-Repose-Cube-Allegro-v0 \\"
echo "      --num_envs 512 \\"
echo "      --headless"
echo ""
echo "Or use our wrapper script:"
echo "  python scripts/run_allegro_hand.py --mode train"
echo "  python scripts/run_allegro_hand.py --mode play --checkpoint <path>"
