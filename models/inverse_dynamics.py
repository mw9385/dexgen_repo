"""
Stage 3 – Inverse Dynamics Model
==================================
Maps consecutive keypoint states to joint actions.

Paper reference (DexterityGen §3.3):
  Given a keypoint trajectory k_{0:T} from the diffusion model,
  the inverse dynamics model computes the joint action at each step:

      a_t = f_inv(k_t, k_{t+1}, x_t)

  where:
    k_t       = current fingertip positions (12-dim, object frame)
    k_{t+1}   = next fingertip positions    (12-dim, object frame)
    x_t       = current robot state: joint pos + vel (32-dim)

  Output:
    a_t       = joint position targets (16-dim, Allegro Hand)

  Architecture (per paper Appendix B):
    - MLP: 3 hidden layers of 256 units, ELU activations
    - Input dim:  12 + 12 + 32 = 56
    - Output dim: 16
    - Trained with supervised learning on Stage 2 dataset

  Training objective:
    L = MSE(â_t, a_t^{RL})  + λ_smooth * ||a_t - a_{t-1}||^2

  where a_t^{RL} is the action taken by the RL policy in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InverseDynamicsConfig:
    """Hyperparameters for the inverse dynamics model."""

    # I/O dimensions
    keypoint_dim: int = 12       # 4 fingertips × 3
    robot_state_dim: int = 32    # joint pos (16) + joint vel (16)
    action_dim: int = 16         # Allegro joint targets

    # Network
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.05

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    smooth_weight: float = 0.01   # smoothness regularisation λ

    @property
    def input_dim(self) -> int:
        return self.keypoint_dim * 2 + self.robot_state_dim


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InverseDynamicsModel(nn.Module):
    """
    Predicts joint actions from keypoint transitions.

    Usage:
        model = InverseDynamicsModel(InverseDynamicsConfig())

        # Single-step prediction
        action = model(k_t, k_next, robot_state)  # (B, 16)

        # Rollout from trajectory
        actions = model.rollout(trajectory, robot_states)  # (B, T-1, 16)
    """

    def __init__(self, cfg: InverseDynamicsConfig):
        super().__init__()
        self.cfg = cfg

        # Build MLP
        dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.num_layers + [cfg.action_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers)

        # Output normalisation (action scale ≈ [-1, 1] after tanh)
        self.action_scale = nn.Parameter(torch.ones(cfg.action_dim) * 0.3)

    def forward(
        self,
        k_current: torch.Tensor,     # (B, 12)  current fingertip pos
        k_next: torch.Tensor,        # (B, 12)  next fingertip pos
        robot_state: torch.Tensor,   # (B, 32)  joint pos + vel
    ) -> torch.Tensor:
        """
        Returns predicted joint position targets (B, 16).
        Output is clipped to [-π, π] (full joint range of Allegro).
        """
        x = torch.cat([k_current, k_next, robot_state], dim=-1)
        raw = self.net(x)                                        # (B, 16)
        # Tanh + learned scale → bounded output
        action = torch.tanh(raw) * self.action_scale.abs()
        return action

    def rollout(
        self,
        trajectory: torch.Tensor,     # (B, T, 12)  keypoint trajectory
        robot_states: torch.Tensor,   # (B, T, 32)  robot states along traj
    ) -> torch.Tensor:
        """
        Predict joint actions for an entire trajectory.

        Returns:
            actions: (B, T-1, 16)
        """
        B, T, _ = trajectory.shape
        actions = []
        for t in range(T - 1):
            a = self.forward(
                trajectory[:, t, :],
                trajectory[:, t + 1, :],
                robot_states[:, t, :],
            )
            actions.append(a)
        return torch.stack(actions, dim=1)   # (B, T-1, 16)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        k_current: torch.Tensor,     # (B, 12)
        k_next: torch.Tensor,        # (B, 12)
        robot_state: torch.Tensor,   # (B, 32)
        action_gt: torch.Tensor,     # (B, 16)  ground truth RL action
        prev_action: Optional[torch.Tensor] = None,  # (B, 16) for smoothness
    ) -> torch.Tensor:
        """
        Supervised loss:
            L = MSE(â, a_gt) + λ * ||â - a_prev||^2
        """
        a_pred = self.forward(k_current, k_next, robot_state)
        loss = F.mse_loss(a_pred, action_gt)

        if prev_action is not None:
            smooth_loss = F.mse_loss(a_pred, prev_action)
            loss = loss + self.cfg.smooth_weight * smooth_loss

        return loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({"model": self.state_dict(), "cfg": self.cfg}, path)
        print(f"[InvDyn] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "InverseDynamicsModel":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["model"])
        print(f"[InvDyn] Loaded from {path}")
        return model


# ---------------------------------------------------------------------------
# DexGen Controller: diffusion + inverse dynamics combined
# ---------------------------------------------------------------------------

class DexGenController:
    """
    Full DexGen controller at inference time.

    Given a new object and a target grasp goal:
      1. Estimate current grasp k_0 from robot/object state
      2. Run diffusion to plan keypoint trajectory k_{0:T}
      3. At each step, run inverse dynamics to get joint action a_t

    This controller generalises to new objects (out-of-distribution)
    because it operates in the object-centric fingertip space.
    """

    def __init__(
        self,
        diffusion: "KeypointDiffusionModel",  # noqa: F821
        inv_dyn: InverseDynamicsModel,
        device: str = "cpu",
    ):
        self.diffusion = diffusion.to(device)
        self.inv_dyn = inv_dyn.to(device)
        self.device = device
        self.diffusion.eval()
        self.inv_dyn.eval()

        self._planned_traj: Optional[torch.Tensor] = None
        self._traj_step: int = 0

    def plan(
        self,
        k_start: torch.Tensor,   # (12,) current fingertip positions (obj frame)
        k_goal: torch.Tensor,    # (12,) target fingertip positions  (obj frame)
    ):
        """
        Plan a keypoint trajectory from k_start to k_goal.
        Stores the trajectory for use in act().
        """
        with torch.no_grad():
            traj = self.diffusion.sample(
                k_start.to(self.device),
                k_goal.to(self.device),
            )   # (1, T, 12)
        self._planned_traj = traj.squeeze(0)    # (T, 12)
        self._traj_step = 0
        print(f"[DexGen] Planned trajectory: {traj.shape[1]} steps")

    def act(
        self,
        robot_state: torch.Tensor,   # (32,) joint pos + vel
    ) -> Optional[torch.Tensor]:
        """
        Get the next joint action from the planned trajectory.

        Returns None when the trajectory is exhausted (goal reached).
        """
        if self._planned_traj is None:
            raise RuntimeError("Call plan() before act()")

        T = self._planned_traj.shape[0]
        if self._traj_step >= T - 1:
            return None   # trajectory complete

        k_curr = self._planned_traj[self._traj_step].unsqueeze(0).to(self.device)
        k_next = self._planned_traj[self._traj_step + 1].unsqueeze(0).to(self.device)
        rs = robot_state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.inv_dyn(k_curr, k_next, rs).squeeze(0)

        self._traj_step += 1
        return action   # (16,)

    @property
    def progress(self) -> float:
        """Fraction of planned trajectory executed [0, 1]."""
        if self._planned_traj is None:
            return 0.0
        return self._traj_step / max(self._planned_traj.shape[0] - 1, 1)

    @classmethod
    def from_checkpoints(
        cls,
        diffusion_ckpt: str,
        inv_dyn_ckpt: str,
        device: str = "cpu",
    ) -> "DexGenController":
        """Load both models from checkpoint files."""
        from .diffusion import KeypointDiffusionModel
        diffusion = KeypointDiffusionModel.load(diffusion_ckpt)
        inv_dyn = InverseDynamicsModel.load(inv_dyn_ckpt)
        return cls(diffusion, inv_dyn, device)
