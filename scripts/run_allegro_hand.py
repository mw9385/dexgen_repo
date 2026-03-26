"""
Run Isaac Lab's built-in AllegroHand cube reorientation environment.

Isaac Lab task: Isaac-Repose-Cube-Allegro-v0

Usage:
    # Train from scratch
    python scripts/run_allegro_hand.py --mode train --num_envs 512 --headless

    # Train with rendering
    python scripts/run_allegro_hand.py --mode train --num_envs 64

    # Play with a trained checkpoint
    python scripts/run_allegro_hand.py --mode play --checkpoint logs/rl_games/allegro_hand/model.pth

    # Quick smoke test (1 env, 100 steps, headless)
    python scripts/run_allegro_hand.py --mode test

Notes:
    - Requires Isaac Lab installed via setup_isaaclab.sh
    - ISAACLAB_DIR env var or ~/IsaacLab path expected
    - Wraps Isaac Lab's rl_games trainer
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Isaac Lab discovery
# ---------------------------------------------------------------------------

def find_isaaclab_dir() -> Path:
    """Locate Isaac Lab installation directory."""
    # 1. Environment variable
    env_path = os.environ.get("ISAACLAB_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. Default location
    default = Path.home() / "IsaacLab"
    if default.exists():
        return default

    raise FileNotFoundError(
        "Isaac Lab not found. Please run ./setup_isaaclab.sh first, "
        "or set the ISAACLAB_DIR environment variable."
    )


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

# Isaac Lab task name for AllegroHand cube reorientation
ALLEGRO_TASK = "Isaac-Repose-Cube-Allegro-v0"

# rl_games config name (matches Isaac Lab's task registry)
RL_GAMES_CONFIG = "allegro_hand_rl_games"

# Default training hyperparameters
TRAIN_DEFAULTS = {
    "num_envs": 512,
    "max_iterations": 30000,
    "seed": 42,
}

PLAY_DEFAULTS = {
    "num_envs": 16,
    "num_steps": 1000,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_train(args, isaaclab_dir: Path):
    """Launch rl_games training for AllegroHand."""
    train_script = isaaclab_dir / "scripts" / "reinforcement_learning" / "rl_games" / "train.py"

    if not train_script.exists():
        # Older Isaac Lab path structure
        train_script = isaaclab_dir / "scripts" / "rl_games" / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"rl_games train script not found in {isaaclab_dir}")

    log_dir = Path("logs") / "rl_games" / "allegro_hand"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_script),
        f"--task={ALLEGRO_TASK}",
        f"--num_envs={args.num_envs}",
        f"--seed={args.seed}",
        f"--log_root_path={log_dir}",
    ]

    if args.headless:
        cmd.append("--headless")

    if args.max_iterations:
        cmd += ["--max_iterations", str(args.max_iterations)]

    print(f"[run_allegro_hand] Task: {ALLEGRO_TASK}")
    print(f"[run_allegro_hand] Num envs: {args.num_envs}")
    print(f"[run_allegro_hand] Command: {' '.join(cmd)}")
    print("-" * 60)

    subprocess.run(cmd, check=True)


def run_play(args, isaaclab_dir: Path):
    """Run a trained AllegroHand policy."""
    if not args.checkpoint:
        # Try to find the latest checkpoint
        log_dir = Path("logs") / "rl_games" / "allegro_hand"
        checkpoints = sorted(log_dir.glob("**/model_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                "No checkpoint found. Train first with --mode train, "
                "or specify --checkpoint <path>"
            )
        args.checkpoint = str(checkpoints[-1])
        print(f"[run_allegro_hand] Auto-selected checkpoint: {args.checkpoint}")

    play_script = isaaclab_dir / "scripts" / "reinforcement_learning" / "rl_games" / "play.py"

    if not play_script.exists():
        play_script = isaaclab_dir / "scripts" / "rl_games" / "play.py"

    if not play_script.exists():
        raise FileNotFoundError(f"rl_games play script not found in {isaaclab_dir}")

    cmd = [
        sys.executable,
        str(play_script),
        f"--task={ALLEGRO_TASK}",
        f"--num_envs={args.num_envs}",
        f"--checkpoint={args.checkpoint}",
    ]

    if args.headless:
        cmd.append("--headless")

    print(f"[run_allegro_hand] Playing with: {args.checkpoint}")
    subprocess.run(cmd, check=True)


def run_test(args, isaaclab_dir: Path):
    """Quick smoke test: create env, step 100 times, no training."""
    print(f"[run_allegro_hand] Smoke test: {ALLEGRO_TASK}")
    print("[run_allegro_hand] Creating environment with 1 env, 100 steps...")

    smoke_script = Path(__file__).parent / "_allegro_smoke_test.py"
    _write_smoke_test_script(smoke_script, isaaclab_dir)

    cmd = [sys.executable, str(smoke_script)]
    result = subprocess.run(cmd)
    smoke_script.unlink(missing_ok=True)

    if result.returncode == 0:
        print("[run_allegro_hand] Smoke test PASSED")
    else:
        print("[run_allegro_hand] Smoke test FAILED")
        sys.exit(result.returncode)


def _write_smoke_test_script(path: Path, isaaclab_dir: Path):
    """Write a temporary smoke test script that imports Isaac Lab and steps the env."""
    script = f"""
import sys
sys.path.insert(0, "{isaaclab_dir}")

import gymnasium as gym
import torch

# Isaac Lab registers tasks when imported
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.inhand  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv

print("Creating environment: {ALLEGRO_TASK}")
env = gym.make(
    "{ALLEGRO_TASK}",
    num_envs=1,
    render_mode=None,
)
obs, info = env.reset()
print(f"Observation shape: {{obs['policy'].shape}}")
print(f"Action shape: {{env.action_space.shape}}")

print("Stepping environment 100 times...")
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if (i + 1) % 20 == 0:
        print(f"  Step {{i+1:3d}}: reward={{reward.mean():.4f}}")

env.close()
print("Smoke test complete.")
"""
    path.write_text(script)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Isaac Lab AllegroHand cube reorientation environment"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play", "test"],
        default="test",
        help="train: run RL training | play: run trained policy | test: smoke test (default: test)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without rendering",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for play mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAIN_DEFAULTS["seed"],
        help="Random seed",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Max training iterations (default: 30000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply mode-specific defaults for num_envs
    if args.num_envs is None:
        if args.mode == "train":
            args.num_envs = TRAIN_DEFAULTS["num_envs"]
        else:
            args.num_envs = PLAY_DEFAULTS["num_envs"]

    try:
        isaaclab_dir = find_isaaclab_dir()
        print(f"[run_allegro_hand] Isaac Lab found at: {isaaclab_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.mode == "train":
        run_train(args, isaaclab_dir)
    elif args.mode == "play":
        run_play(args, isaaclab_dir)
    elif args.mode == "test":
        run_test(args, isaaclab_dir)


if __name__ == "__main__":
    main()
