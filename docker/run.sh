#!/usr/bin/env bash
# ==============================================================================
# DexGen Docker Helper Script
# ==============================================================================
# 컨테이너 생명주기 관리 전용 (build / up / down / exec / logs / status)
#
# 파이프라인 실행은 컨테이너 내부에서 직접 수행:
#   ./docker/run.sh up
#   ./docker/run.sh exec          ← bash로 진입
#   # 컨테이너 안에서:
#   /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py
#   /workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py --num_envs 512 --headless
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
CONTAINER_NAME="dexgen"

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[dexgen]${NC} $*"; }
warn()  { echo -e "${YELLOW}[dexgen]${NC} $*"; }
error() { echo -e "${RED}[dexgen]${NC} $*"; exit 1; }

check_docker() {
    command -v docker &>/dev/null || error "Docker not found."
    docker info &>/dev/null       || error "Docker daemon not running."
}

is_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"
}

# ------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------
cmd_build() {
    info "Building DexGen image (Isaac Sim 5.1.0 + Isaac Lab v2.3.2)..."
    info "First build downloads ~20 GB from NGC — takes 20–40 min."
    if ! grep -q "nvcr.io" ~/.docker/config.json 2>/dev/null; then
        warn "NGC login not found. Run: docker login nvcr.io"
        warn "  Username: \$oauthtoken  /  Password: <NGC API key>"
    fi
    docker compose -f "${COMPOSE_FILE}" build "$@"
    info "Build complete. Run './docker/run.sh up' to start."
}

cmd_up() {
    info "Starting DexGen container..."
    docker compose -f "${COMPOSE_FILE}" up -d "$@"
    info "Container '${CONTAINER_NAME}' started."
    info "Enter with: ./docker/run.sh exec"
}

cmd_down() {
    info "Stopping DexGen container..."
    docker compose -f "${COMPOSE_FILE}" down "$@"
}

cmd_exec() {
    is_running || error "Container not running. Run './docker/run.sh up' first."
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

cmd_help() {
    cat <<EOF
DexGen Docker helper — 컨테이너 관리 전용

Usage: ./docker/run.sh <command>

  build     Docker 이미지 빌드 (최초 1회, ~20 GB 다운로드)
  up        컨테이너 백그라운드 시작
  down      컨테이너 중지
  exec      컨테이너 bash 진입  (기본값)
  exec <cmd> 특정 명령 실행
  logs      컨테이너 로그 실시간 출력
  status    컨테이너 상태 확인

파이프라인은 컨테이너 내부에서 실행:
  ./docker/run.sh up
  ./docker/run.sh exec
  # 컨테이너 안에서 isaaclab.sh -p 로 실행
EOF
}

# ------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------
check_docker

case "${1:-help}" in
    build)          shift; cmd_build "$@"  ;;
    up)             shift; cmd_up "$@"     ;;
    down)           shift; cmd_down "$@"   ;;
    exec)           shift; cmd_exec "$@"   ;;
    logs)           shift; cmd_logs "$@"   ;;
    status)         cmd_status             ;;
    help|--help|-h) cmd_help               ;;
    *) error "Unknown command: $1. Run './docker/run.sh help'" ;;
esac
