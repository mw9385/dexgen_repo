#!/usr/bin/env bash
# ==============================================================================
# DexGen Docker Helper Script
# ==============================================================================
# Wraps common docker compose commands and provides shortcuts for
# DexGen pipeline stages.
#
# Usage:
#   ./docker/run.sh <command> [options]
#
# Commands:
#   build           Build the DexGen Docker image
#   up              Start the container (detached)
#   down            Stop and remove the container
#   exec [cmd]      Execute command inside running container (default: bash)
#   logs            Tail container logs
#   status          Show container status
#
#   # DexGen pipeline shortcuts (run inside container):
#   test_allegro    Run Isaac Lab AllegroHand smoke test
#   gen_grasps      Stage 0: Generate grasp set
#   train_rl        Stage 1: Train RL policy (pass extra args after --)
#   collect_data    Stage 2: Collect dataset
#   train_dexgen    Stage 3: Train DexGen controller
#
# Examples:
#   ./docker/run.sh build
#   ./docker/run.sh up
#   ./docker/run.sh exec bash
#   ./docker/run.sh train_rl -- --num_envs 512 --headless
#   ./docker/run.sh exec python scripts/run_allegro_hand.py --mode test
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
CONTAINER_NAME="dexgen"

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[dexgen]${NC} $*"; }
warn()  { echo -e "${YELLOW}[dexgen]${NC} $*"; }
error() { echo -e "${RED}[dexgen]${NC} $*"; exit 1; }

# ------------------------------------------------------------------------------
# Prerequisite checks
# ------------------------------------------------------------------------------
check_docker() {
    command -v docker &>/dev/null || error "Docker not found. Install Docker first."
    docker info &>/dev/null       || error "Docker daemon not running."
}

check_nvidia() {
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found. GPU passthrough may not work."
        return
    fi
    if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 \
            nvidia-smi &>/dev/null 2>&1; then
        warn "nvidia-container-toolkit may not be installed."
        warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

check_ngc_login() {
    # Isaac Sim image requires NGC login
    if ! grep -q "nvcr.io" ~/.docker/config.json 2>/dev/null; then
        warn "NGC registry login not found."
        warn "Run: docker login nvcr.io"
        warn "  Username: \$oauthtoken"
        warn "  Password: <your NGC API key from https://ngc.nvidia.com>"
    fi
}

is_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"
}

# ------------------------------------------------------------------------------
# Core commands
# ------------------------------------------------------------------------------
cmd_build() {
    info "Building DexGen image (Isaac Sim 5.1.0 + Isaac Lab v5.1.0)..."
    info "Note: first build pulls ~20 GB from NGC. This will take a while."
    check_ngc_login
    docker compose -f "${COMPOSE_FILE}" build "$@"
    info "Build complete. Run './docker/run.sh up' to start."
}

cmd_up() {
    info "Starting DexGen container..."
    docker compose -f "${COMPOSE_FILE}" up -d "$@"
    info "Container '${CONTAINER_NAME}' is running."
    info "Use './docker/run.sh exec bash' to enter."
}

cmd_down() {
    info "Stopping DexGen container..."
    docker compose -f "${COMPOSE_FILE}" down "$@"
}

cmd_exec() {
    is_running || error "Container '${CONTAINER_NAME}' is not running. Run './docker/run.sh up' first."
    local cmd="${1:-bash}"
    shift 2>/dev/null || true
    docker exec -it "${CONTAINER_NAME}" ${cmd} "$@"
}

cmd_logs() {
    docker compose -f "${COMPOSE_FILE}" logs -f "$@"
}

cmd_status() {
    docker compose -f "${COMPOSE_FILE}" ps
}

# ------------------------------------------------------------------------------
# DexGen pipeline shortcuts
# ------------------------------------------------------------------------------
cmd_gen_grasps() {
    info "Stage 0: Generating grasp set..."
    is_running || error "Container not running."
    # Grasp generation only needs trimesh/scipy — use python3 (no Isaac Lab needed)
    docker exec -it "${CONTAINER_NAME}" \
        python3 scripts/run_grasp_generation.py "$@"
}

cmd_train_rl() {
    info "Stage 1: Training RL policy..."
    is_running || error "Container not running."
    # Isaac Lab scripts must use the Isaac Sim python interpreter
    docker exec -it "${CONTAINER_NAME}" \
        /isaac-sim/python.sh scripts/train_rl.py \
            --grasp_graph data/grasp_graph.pkl \
            --num_envs 512 \
            --headless \
            "$@"
}

cmd_collect_data() {
    info "Stage 2: Collecting dataset..."
    is_running || error "Container not running."
    local ckpt
    ckpt=$(docker exec "${CONTAINER_NAME}" \
        bash -c "ls logs/rl/allegro_anygrasp/checkpoints/*.pt 2>/dev/null | sort | tail -1")
    [ -z "${ckpt}" ] && error "No checkpoint found. Train RL first."
    docker exec -it "${CONTAINER_NAME}" \
        /isaac-sim/python.sh scripts/collect_data.py \
            --checkpoint "${ckpt}" \
            --num_episodes 50000 \
            "$@"
}

cmd_train_dexgen() {
    info "Stage 3: Training DexGen controller..."
    is_running || error "Container not running."
    docker exec -it "${CONTAINER_NAME}" \
        python3 scripts/train_dexgen.py \
            --data data/dataset.h5 \
            "$@"
}

cmd_test_allegro() {
    info "Running AllegroHand smoke test..."
    is_running || error "Container not running. Run './docker/run.sh up' first."
    docker exec -it "${CONTAINER_NAME}" \
        /isaac-sim/python.sh scripts/run_allegro_hand.py --mode test
}

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
cmd_help() {
    cat <<EOF
DexGen Docker helper

Usage: ./docker/run.sh <command> [args]

Setup:
  build             Build Docker image (first run: ~20 GB download from NGC)
  up                Start container in background
  down              Stop container
  exec [cmd]        Shell into running container (default: bash)
  logs              Tail container logs
  status            Show container status

Pipeline:
  test_allegro      Run Isaac Lab AllegroHand smoke test
  gen_grasps [args] Stage 0: Generate grasp set
  train_rl [args]   Stage 1: Train RL policy (--num_envs, --max_iterations, ...)
  collect_data [..]  Stage 2: Collect dataset
  train_dexgen [..]  Stage 3: Train DexGen controller

Quick start:
  ./docker/run.sh build
  ./docker/run.sh up
  ./docker/run.sh test_allegro
  ./docker/run.sh gen_grasps
  ./docker/run.sh train_rl -- --num_envs 512
EOF
}

# ------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------
check_docker

case "${1:-help}" in
    build)         shift; cmd_build "$@"        ;;
    up)            shift; cmd_up "$@"           ;;
    down)          shift; cmd_down "$@"         ;;
    exec)          shift; cmd_exec "$@"         ;;
    logs)          shift; cmd_logs "$@"         ;;
    status)        cmd_status                   ;;
    test_allegro)  shift; cmd_test_allegro "$@" ;;
    gen_grasps)    shift; cmd_gen_grasps "$@"   ;;
    train_rl)      shift; cmd_train_rl "$@"     ;;
    collect_data)  shift; cmd_collect_data "$@" ;;
    train_dexgen)  shift; cmd_train_dexgen "$@" ;;
    help|--help|-h) cmd_help                    ;;
    *) error "Unknown command: $1. Run './docker/run.sh help'";;
esac
