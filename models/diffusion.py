"""
Stage 3 – Keypoint Motion Diffusion Model
==========================================
Generates fingertip keypoint trajectories conditioned on start and goal grasps.

Paper reference (DexterityGen §3.3):
  The DexGen controller consists of two components:
    (1) Diffusion model: given (k_0, k_T), generates full keypoint trajectory
        k_{0:T} = [k_0, k_1, ..., k_T]  where k_t ∈ R^12 (4 fingertips × 3)
    (2) Inverse dynamics: maps consecutive keypoints (k_t, k_{t+1}) → joint action

  Diffusion model details:
    - DDPM (Ho et al., 2020) with T=100 denoising steps
    - Denoiser: MLP with sinusoidal time embedding
    - Conditioning: concatenate (k_0, k_T) as context c
    - Training: predict noise ε given noisy trajectory x_τ and context c
    - Inference: start from Gaussian noise, denoise to get trajectory

  Architecture (per paper Appendix B):
    - Input:  [x_τ (trajectory), τ (timestep), c (k_0 + k_T)]
    - Hidden: 4 × 512 MLP with SiLU activations + LayerNorm
    - Output: ε̂ (predicted noise), same shape as x_τ

  Trajectory representation:
    - T = 50 keyframes (steps), each is 12-dim fingertip positions
    - Expressed in object frame (object-pose invariant)
    - x ∈ R^{T × 12}  flattened to R^{600}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Hyperparameters for the keypoint diffusion model."""

    # Trajectory
    horizon: int = 50           # number of keyframes T
    keypoint_dim: int = 12      # 4 fingertips × 3 coords

    # Diffusion
    num_diffusion_steps: int = 100          # denoising steps at inference
    num_train_timesteps: int = 100          # noise schedule steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"           # "linear" or "cosine"

    # Network
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1

    # Conditioning
    cond_dim: int = 24   # k_0 (12) + k_T (12)

    @property
    def traj_dim(self) -> int:
        return self.horizon * self.keypoint_dim


# ---------------------------------------------------------------------------
# Noise Schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from Improved DDPM (Nichol & Dhariwal, 2021).
    Smoother than linear, avoids too much noise near t=0.
    """
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0001, 0.9999)


def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


# ---------------------------------------------------------------------------
# Sinusoidal Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timestep τ.
    Produces a rich representation of the noise level.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer timestep indices
        Returns: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        emb = t[:, None].float() * freqs[None, :]     # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Denoiser Network
# ---------------------------------------------------------------------------

class DenoiseMLP(nn.Module):
    """
    MLP denoiser for the diffusion model.

    Input:  [x_noisy (traj_dim), τ_emb (hidden), c (cond_dim)]
    Output: ε̂ (traj_dim)  – predicted noise

    Architecture: residual MLP blocks with time/condition injection.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        traj_dim = cfg.traj_dim
        hidden = cfg.hidden_dim

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(hidden)

        # Condition embedding (k_0 + k_T → hidden)
        self.cond_emb = nn.Sequential(
            nn.Linear(cfg.cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Input projection
        self.input_proj = nn.Linear(traj_dim, hidden)

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            _ResidualBlock(hidden, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, traj_dim),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, traj_dim) noisy trajectory
        t: torch.Tensor,          # (B,) diffusion timestep
        cond: torch.Tensor,       # (B, cond_dim) = [k_0, k_T]
    ) -> torch.Tensor:
        """Returns predicted noise ε̂ of shape (B, traj_dim)."""
        # Embeddings
        t_emb = self.time_emb(t)           # (B, hidden)
        c_emb = self.cond_emb(cond)        # (B, hidden)
        h = self.input_proj(x)             # (B, hidden)

        # Add time and condition to every residual block
        ctx = t_emb + c_emb               # (B, hidden)
        for block in self.blocks:
            h = block(h, ctx)

        return self.output_proj(h)         # (B, traj_dim)


class _ResidualBlock(nn.Module):
    """MLP residual block with context injection."""

    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.ctx_proj = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # Pre-norm + context injection
        h = self.norm1(x + self.ctx_proj(ctx))
        h = self.drop(self.act(self.fc1(h)))
        h = self.fc2(h)
        return x + self.norm2(h)


# ---------------------------------------------------------------------------
# DDPM Diffusion Model
# ---------------------------------------------------------------------------

class KeypointDiffusionModel(nn.Module):
    """
    DDPM-based keypoint trajectory diffusion model.

    Training:
        loss = MSE(ε̂, ε)   where ε ~ N(0, I) is added noise

    Inference:
        Given (k_0, k_T), generate trajectory k_{0:T} via reverse diffusion.

    Usage:
        model = KeypointDiffusionModel(DiffusionConfig())

        # Training step
        loss = model.compute_loss(trajectory, k_start, k_goal)

        # Inference
        trajectory = model.sample(k_start, k_goal)  # (B, T, 12)
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoiser = DenoiseMLP(cfg)

        # Build noise schedule
        T = cfg.num_train_timesteps
        if cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T, cfg.beta_start, cfg.beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        trajectory: torch.Tensor,    # (B, T, 12)  ground truth
        k_start: torch.Tensor,        # (B, 12)
        k_goal: torch.Tensor,         # (B, 12)
    ) -> torch.Tensor:
        """
        Standard DDPM training loss.
          1. Sample random timestep τ ~ Uniform(1, T)
          2. Add noise: x_τ = sqrt(ᾱ_τ) x_0 + sqrt(1-ᾱ_τ) ε
          3. Predict ε̂ = denoiser(x_τ, τ, cond)
          4. Loss = MSE(ε̂, ε)
        """
        B = trajectory.shape[0]
        device = trajectory.device

        # Flatten trajectory: (B, T*12)
        x0 = trajectory.reshape(B, -1)

        # Random timestep
        t = torch.randint(0, self.cfg.num_train_timesteps, (B,), device=device)

        # Forward diffusion: add noise
        eps = torch.randn_like(x0)
        x_t = (
            self.sqrt_alphas_cumprod[t, None] * x0
            + self.sqrt_one_minus_alphas_cumprod[t, None] * eps
        )

        # Condition
        cond = torch.cat([k_start, k_goal], dim=-1)   # (B, 24)

        # Predict noise
        eps_hat = self.denoiser(x_t, t, cond)

        return F.mse_loss(eps_hat, eps)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        k_start: torch.Tensor,    # (B, 12) or (12,)
        k_goal: torch.Tensor,     # (B, 12) or (12,)
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a keypoint trajectory from k_start to k_goal.

        Returns:
            trajectory: (B, T, 12)  fingertip positions at each keyframe
        """
        if k_start.dim() == 1:
            k_start = k_start.unsqueeze(0)
            k_goal = k_goal.unsqueeze(0)

        B = k_start.shape[0]
        device = k_start.device
        T_total = self.cfg.num_train_timesteps
        num_steps = num_steps or self.cfg.num_diffusion_steps
        cond = torch.cat([k_start, k_goal], dim=-1)   # (B, 24)

        # Start from pure noise
        x = torch.randn(B, self.cfg.traj_dim, device=device)

        # DDPM reverse process
        step_size = T_total // num_steps
        timesteps = list(range(0, T_total, step_size))[::-1]

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps_hat = self.denoiser(x, t, cond)

            # Reverse step (DDPM)
            alpha = self.alphas_cumprod[t_val]
            alpha_prev = self.alphas_cumprod_prev[t_val] if t_val > 0 else torch.tensor(1.0)
            beta = self.betas[t_val]

            # Predicted x_0
            x0_pred = (x - (1 - alpha).sqrt() * eps_hat) / alpha.sqrt()
            x0_pred = x0_pred.clamp(-3.0, 3.0)

            # Posterior mean
            coef1 = beta * alpha_prev.sqrt() / (1 - alpha)
            coef2 = (1 - alpha_prev) * (1 - beta).sqrt() / (1 - alpha)
            mean = coef1 * x0_pred + coef2 * x

            # Add noise (except at final step)
            if t_val > 0:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t_val].clamp(min=1e-10)
                x = mean + var.sqrt() * noise
            else:
                x = mean

        # Reshape to (B, T, 12)
        traj = x.reshape(B, self.cfg.horizon, self.cfg.keypoint_dim)

        # Enforce boundary conditions: first = k_start, last = k_goal
        traj[:, 0, :] = k_start
        traj[:, -1, :] = k_goal

        return traj

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({"model": self.state_dict(), "cfg": self.cfg}, path)
        print(f"[Diffusion] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "KeypointDiffusionModel":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["model"])
        print(f"[Diffusion] Loaded from {path}")
        return model
