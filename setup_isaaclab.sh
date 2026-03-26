#!/usr/bin/env bash
# ==============================================================================
# DexGen Setup — Docker-based (recommended)
# ==============================================================================
# Direct pip installation of Isaac Sim has too many system-level dependencies.
# Use Docker instead for a reproducible, error-free environment.
#
# Requirements (host machine):
#   - Ubuntu 20.04 / 22.04
#   - NVIDIA GPU with driver >= 525.60
#   - Docker >= 24.x  (https://docs.docker.com/engine/install/ubuntu/)
#   - NVIDIA Container Toolkit  (nvidia-docker2)
#   - NGC account + API key     (https://ngc.nvidia.com)
# ==============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
error() { echo -e "${RED}[setup]${NC} $*"; exit 1; }

# ------------------------------------------------------------------------------
# Step 1 — Install Docker (if missing)
# ------------------------------------------------------------------------------
install_docker() {
    if command -v docker &>/dev/null; then
        info "Docker already installed: $(docker --version)"
        return
    fi
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "${USER}"
    warn "Docker installed. Log out and back in (or run 'newgrp docker') to use without sudo."
}

# ------------------------------------------------------------------------------
# Step 2 — Install NVIDIA Container Toolkit (if missing)
# ------------------------------------------------------------------------------
install_nvidia_container_toolkit() {
    if dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
        info "nvidia-container-toolkit already installed."
        return
    fi
    info "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    info "NVIDIA Container Toolkit installed."
}

# ------------------------------------------------------------------------------
# Step 3 — NGC login for Isaac Sim image
# ------------------------------------------------------------------------------
ngc_login() {
    if grep -q "nvcr.io" ~/.docker/config.json 2>/dev/null; then
        info "NGC login already configured."
        return
    fi
    info "NGC login required to pull Isaac Sim 5.1.0 image."
    echo ""
    echo "  1. Go to https://ngc.nvidia.com → Account → Setup → API Key"
    echo "  2. Run: docker login nvcr.io"
    echo "     Username: \$oauthtoken"
    echo "     Password: <your NGC API key>"
    echo ""
    read -rp "Press Enter after completing NGC login, or Ctrl+C to cancel..."
    docker login nvcr.io || error "NGC login failed."
}

# ------------------------------------------------------------------------------
# Step 4 — Build DexGen Docker image
# ------------------------------------------------------------------------------
build_image() {
    info "Building DexGen Docker image..."
    info "Base: nvcr.io/nvidia/isaac-sim:5.1.0 + Isaac Lab v5.1.0"
    warn "First build downloads ~20 GB from NGC. This will take 20–40 minutes."
    echo ""
    docker compose -f docker/docker-compose.yml build
    info "Image built successfully: dexgen:latest"
}

# ------------------------------------------------------------------------------
# Step 5 — Verify GPU inside container
# ------------------------------------------------------------------------------
verify() {
    info "Verifying GPU access inside container..."
    docker run --rm --gpus all dexgen:latest nvidia-smi \
        && info "GPU OK" \
        || error "GPU not accessible inside container. Check nvidia-container-toolkit."

    info "Verifying Isaac Lab..."
    docker run --rm --gpus all dexgen:latest \
        python3 -c "import isaaclab; print(f'Isaac Lab: {isaaclab.__version__}')" \
        && info "Isaac Lab OK" \
        || warn "Isaac Lab import failed — may need DISPLAY for full init."
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
echo "========================================"
echo " DexGen Setup (Docker)"
echo "========================================"
echo ""

# Check GPU first
if ! command -v nvidia-smi &>/dev/null; then
    error "NVIDIA GPU not detected. A CUDA-capable GPU is required."
fi
info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

install_docker
install_nvidia_container_toolkit
ngc_login
build_image
verify

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  ./docker/run.sh up                    # start container"
echo "  ./docker/run.sh test_allegro          # smoke test"
echo "  ./docker/run.sh gen_grasps            # Stage 0"
echo "  ./docker/run.sh train_rl              # Stage 1"
echo "  ./docker/run.sh exec bash             # open shell"
echo ""
echo "Full pipeline docs: README.md"
