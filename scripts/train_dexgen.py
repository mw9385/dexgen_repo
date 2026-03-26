"""
Stage 3 – DexGen Controller Training
======================================
Trains the two-component DexGen controller:
  (1) Keypoint diffusion model   → plans fingertip trajectories
  (2) Inverse dynamics model     → maps keypoint deltas to joint actions

Both models are trained on the dataset from Stage 2.

Paper reference (DexterityGen §3.3):
  - Diffusion model: DDPM trained to denoise keypoint trajectories
    conditioned on (k_start, k_goal)
  - Inverse dynamics: MLP trained with supervised learning on
    (k_t, k_{t+1}, robot_state) → action pairs from the RL dataset

Usage:
    python scripts/train_dexgen.py \\
        --data data/dataset.h5 \\
        --epochs 200 \\
        --batch_size 256

    # Train only one component
    python scripts/train_dexgen.py --data data/dataset.h5 --only diffusion
    python scripts/train_dexgen.py --data data/dataset.h5 --only inv_dyn
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    KeypointDiffusionModel, DiffusionConfig,
    InverseDynamicsModel, InverseDynamicsConfig,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GraspTransitionDataset(Dataset):
    """
    Loads the Stage 2 HDF5 dataset.

    Each item is a trajectory, subsampled to `horizon` keyframes.
    For inverse dynamics, we additionally expose individual (t, t+1) pairs.
    """

    def __init__(
        self,
        h5_path: str,
        horizon: int = 50,
        success_only: bool = True,
    ):
        import h5py
        self.horizon = horizon
        self.episodes = []

        with h5py.File(h5_path, "r") as f:
            for key in f["trajectories"]:
                ep = f["trajectories"][key]
                success = bool(ep.attrs.get("success", True))
                if success_only and not success:
                    continue
                self.episodes.append({
                    "keypoint_traj": ep["keypoint_traj"][:],   # (T, 12)
                    "joint_traj": ep["joint_traj"][:],          # (T, 16)
                    "action_traj": ep["action_traj"][:],        # (T, 16)
                    "robot_state": ep["robot_state"][:],        # (T, 32)
                })

        print(f"[Dataset] Loaded {len(self.episodes)} episodes from {h5_path}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        T_orig = ep["keypoint_traj"].shape[0]

        # Subsample to horizon keyframes
        indices = np.round(np.linspace(0, T_orig - 1, self.horizon)).astype(int)
        kp = ep["keypoint_traj"][indices]         # (H, 12)
        jq = ep["joint_traj"][indices]             # (H, 16)
        act = ep["action_traj"][indices]           # (H, 16)
        rs = ep["robot_state"][indices]            # (H, 32)

        return {
            "keypoint_traj": torch.tensor(kp, dtype=torch.float32),
            "joint_traj": torch.tensor(jq, dtype=torch.float32),
            "action_traj": torch.tensor(act, dtype=torch.float32),
            "robot_state": torch.tensor(rs, dtype=torch.float32),
            "k_start": torch.tensor(kp[0], dtype=torch.float32),
            "k_goal": torch.tensor(kp[-1], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_diffusion(
    model: KeypointDiffusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    save_dir: Path,
):
    """Train the keypoint diffusion model."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    model = model.to(device)

    best_val_loss = float("inf")
    print(f"[Diffusion] Training for {epochs} epochs on {device}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            traj = batch["keypoint_traj"].to(device)    # (B, H, 12)
            k_start = batch["k_start"].to(device)       # (B, 12)
            k_goal = batch["k_goal"].to(device)         # (B, 12)

            optimiser.zero_grad()
            loss = model.compute_loss(traj, k_start, k_goal)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    traj = batch["keypoint_traj"].to(device)
                    k_start = batch["k_start"].to(device)
                    k_goal = batch["k_goal"].to(device)
                    loss = model.compute_loss(traj, k_start, k_goal)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            lr_cur = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_cur:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(str(save_dir / "diffusion_best.pt"))

    model.save(str(save_dir / "diffusion_final.pt"))
    print(f"[Diffusion] Best val loss: {best_val_loss:.4f}")
    return model


def train_inv_dyn(
    model: InverseDynamicsModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    save_dir: Path,
):
    """Train the inverse dynamics model."""
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=model.cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    model = model.to(device)

    best_val_loss = float("inf")
    print(f"[InvDyn] Training for {epochs} epochs on {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            kp = batch["keypoint_traj"].to(device)      # (B, H, 12)
            act = batch["action_traj"].to(device)        # (B, H, 16)
            rs = batch["robot_state"].to(device)         # (B, H, 32)

            B, H, _ = kp.shape
            # All consecutive pairs (B*(H-1), ...)
            k_curr = kp[:, :-1, :].reshape(B * (H - 1), -1)
            k_next = kp[:, 1:, :].reshape(B * (H - 1), -1)
            a_gt = act[:, :-1, :].reshape(B * (H - 1), -1)
            a_prev = act[:, :-1, :].reshape(B * (H - 1), -1)   # for smoothness
            robot_s = rs[:, :-1, :].reshape(B * (H - 1), -1)

            optimiser.zero_grad()
            loss = model.compute_loss(k_curr, k_next, robot_s, a_gt, a_prev)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    kp = batch["keypoint_traj"].to(device)
                    act = batch["action_traj"].to(device)
                    rs = batch["robot_state"].to(device)
                    B, H, _ = kp.shape
                    k_curr = kp[:, :-1].reshape(B * (H - 1), -1)
                    k_next = kp[:, 1:].reshape(B * (H - 1), -1)
                    a_gt = act[:, :-1].reshape(B * (H - 1), -1)
                    robot_s = rs[:, :-1].reshape(B * (H - 1), -1)
                    loss = model.compute_loss(k_curr, k_next, robot_s, a_gt)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(str(save_dir / "inv_dyn_best.pt"))

    model.save(str(save_dir / "inv_dyn_final.pt"))
    print(f"[InvDyn] Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 3: Controller Training")
    p.add_argument("--data", type=str, default="data/dataset.h5",
                   help="Path to HDF5 dataset from Stage 2")
    p.add_argument("--only", choices=["diffusion", "inv_dyn", "both"], default="both",
                   help="Which component to train")
    # Diffusion
    p.add_argument("--diffusion_epochs", type=int, default=500)
    p.add_argument("--diffusion_lr", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=50,
                   help="Keyframe trajectory length T")
    p.add_argument("--diffusion_steps", type=int, default=100,
                   help="DDPM denoising steps")
    # Inverse dynamics
    p.add_argument("--inv_dyn_epochs", type=int, default=300)
    p.add_argument("--inv_dyn_lr", type=float, default=3e-4)
    # Common
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--success_only", action="store_true", default=True,
                   help="Only train on successful episodes")
    p.add_argument("--output_dir", type=str, default="logs/dexgen")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not Path(args.data).exists():
        print(f"ERROR: Dataset not found at {args.data}")
        print("Run Stage 2 first: python scripts/collect_data.py")
        sys.exit(1)

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = GraspTransitionDataset(
        args.data,
        horizon=args.horizon,
        success_only=args.success_only,
    )

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Stage 3] Dataset: {len(dataset)} episodes "
          f"(train={train_size}, val={val_size})")
    print(f"[Stage 3] Device: {args.device}")
    print(f"[Stage 3] Output: {save_dir}")

    # ------------------------------------------------------------------
    # Train diffusion model
    # ------------------------------------------------------------------
    if args.only in ("diffusion", "both"):
        diff_cfg = DiffusionConfig(
            horizon=args.horizon,
            num_train_timesteps=args.diffusion_steps,
            num_diffusion_steps=args.diffusion_steps,
        )
        diff_model = KeypointDiffusionModel(diff_cfg)
        n_params = sum(p.numel() for p in diff_model.parameters())
        print(f"\n[Diffusion] Parameters: {n_params:,}")
        train_diffusion(
            diff_model, train_loader, val_loader,
            epochs=args.diffusion_epochs,
            lr=args.diffusion_lr,
            device=args.device,
            save_dir=save_dir,
        )

    # ------------------------------------------------------------------
    # Train inverse dynamics
    # ------------------------------------------------------------------
    if args.only in ("inv_dyn", "both"):
        inv_cfg = InverseDynamicsConfig(learning_rate=args.inv_dyn_lr)
        inv_model = InverseDynamicsModel(inv_cfg)
        n_params = sum(p.numel() for p in inv_model.parameters())
        print(f"\n[InvDyn] Parameters: {n_params:,}")
        train_inv_dyn(
            inv_model, train_loader, val_loader,
            epochs=args.inv_dyn_epochs,
            lr=args.inv_dyn_lr,
            device=args.device,
            save_dir=save_dir,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n=== Stage 3 Complete ===")
    print(f"  Checkpoints: {save_dir}/")
    if args.only in ("diffusion", "both"):
        print(f"  Diffusion  : diffusion_best.pt, diffusion_final.pt")
    if args.only in ("inv_dyn", "both"):
        print(f"  Inv Dyn    : inv_dyn_best.pt, inv_dyn_final.pt")
    print(f"\nTo use DexGen controller:")
    print(f"""
    from models.inverse_dynamics import DexGenController
    controller = DexGenController.from_checkpoints(
        diffusion_ckpt="{save_dir}/diffusion_best.pt",
        inv_dyn_ckpt="{save_dir}/inv_dyn_best.pt",
        device="cuda",
    )
    controller.plan(k_start, k_goal)
    while True:
        action = controller.act(robot_state)
        if action is None:
            break
        env.step(action)
    """)


if __name__ == "__main__":
    main()
