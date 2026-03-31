"""
Differentiable Hand Kinematic Model
====================================
Provides forward kinematics (FK) for Shadow Hand and Allegro Hand
using PyTorch, enabling gradient-based grasp optimization.

Based on DexGraspNet approach:
  - Joint angles → fingertip positions (differentiable FK)
  - Used by GraspOptimizer to optimize hand configuration
  - Supports Shadow Hand (24 DOF, 5 fingers) and Allegro Hand (16 DOF, 4 fingers)

The FK is simplified (chain of revolute joints with fixed DH parameters)
but sufficient for grasp optimization. Isaac Sim refinement later corrects
any small kinematic errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DH-parameter-based link for revolute joints
# ---------------------------------------------------------------------------

@dataclass
class LinkParam:
    """Modified DH parameters for a single revolute joint."""
    name: str
    a: float       # link length (m)
    d: float       # link offset (m)
    alpha: float   # link twist (rad)
    theta_offset: float = 0.0   # fixed offset added to joint angle
    joint_min: float = -0.5
    joint_max: float = 1.8


def _dh_transform(a: float, d: float, alpha: float, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute 4x4 homogeneous transform from Modified DH parameters.
    Batched: theta shape (...,), returns (..., 4, 4).
    """
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = float(np.cos(alpha))
    sa = float(np.sin(alpha))

    T = theta.new_zeros(theta.shape + (4, 4))
    T[..., 0, 0] = ct
    T[..., 0, 1] = -st
    T[..., 0, 2] = 0.0
    T[..., 0, 3] = a

    T[..., 1, 0] = st * ca
    T[..., 1, 1] = ct * ca
    T[..., 1, 2] = -sa
    T[..., 1, 3] = -d * sa

    T[..., 2, 0] = st * sa
    T[..., 2, 1] = ct * sa
    T[..., 2, 2] = ca
    T[..., 2, 3] = d * ca

    T[..., 3, 3] = 1.0
    return T


# ---------------------------------------------------------------------------
# Finger kinematic chain
# ---------------------------------------------------------------------------

class FingerChain:
    """
    A single finger's kinematic chain: base transform + N revolute joints.
    Forward kinematics maps joint angles → fingertip position.
    """

    def __init__(self, name: str, links: List[LinkParam],
                 base_transform: Optional[np.ndarray] = None):
        self.name = name
        self.links = links
        self.num_joints = len(links)
        if base_transform is not None:
            self._base_T = torch.tensor(base_transform, dtype=torch.float32)
        else:
            self._base_T = torch.eye(4, dtype=torch.float32)

    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute fingertip position from joint angles.

        Args:
            joint_angles: (..., num_joints)

        Returns:
            fingertip_pos: (..., 3) in hand root frame
        """
        batch_shape = joint_angles.shape[:-1]
        base = self._base_T.to(joint_angles.device)
        T = base.expand(batch_shape + (4, 4)).clone()

        for i, link in enumerate(self.links):
            theta = joint_angles[..., i] + link.theta_offset
            theta = theta.clamp(link.joint_min, link.joint_max)
            dh_T = _dh_transform(link.a, link.d, link.alpha, theta)
            T = T @ dh_T

        return T[..., :3, 3]

    @property
    def joint_limits_lower(self) -> np.ndarray:
        return np.array([l.joint_min for l in self.links], dtype=np.float32)

    @property
    def joint_limits_upper(self) -> np.ndarray:
        return np.array([l.joint_max for l in self.links], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hand model (collection of finger chains)
# ---------------------------------------------------------------------------

class HandModel(nn.Module):
    """
    Differentiable hand model for grasp optimization.

    Provides:
      - FK: joint_angles → fingertip_positions (differentiable)
      - Joint limit clamping
      - Hand pose (translation + rotation) applied to all fingertips

    The hand pose is parameterized as:
      - translation: (3,) hand root position relative to object
      - rotation: (6,) continuous rotation representation (first two columns of R)
    """

    def __init__(self, fingers: List[FingerChain], name: str = "hand"):
        super().__init__()
        self.fingers = nn.ModuleList()  # placeholder for module registration
        self._finger_chains = fingers
        self.name = name
        self.num_fingers = len(fingers)
        self.num_dof = sum(f.num_joints for f in fingers)

        # Collect joint limits
        lowers, uppers = [], []
        for f in fingers:
            lowers.append(f.joint_limits_lower)
            uppers.append(f.joint_limits_upper)
        self.register_buffer("joint_lower", torch.tensor(np.concatenate(lowers)))
        self.register_buffer("joint_upper", torch.tensor(np.concatenate(uppers)))

    def forward_kinematics(
        self,
        joint_angles: torch.Tensor,
        hand_translation: torch.Tensor,
        hand_rotation_6d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fingertip positions in object frame.

        Args:
            joint_angles: (B, num_dof) joint angles
            hand_translation: (B, 3) hand root position relative to object
            hand_rotation_6d: (B, 6) continuous rotation (Zhou et al.)

        Returns:
            fingertip_positions: (B, num_fingers, 3) in object frame
        """
        B = joint_angles.shape[0]
        R = rotation_6d_to_matrix(hand_rotation_6d)  # (B, 3, 3)

        tips = []
        offset = 0
        for chain in self._finger_chains:
            n = chain.num_joints
            q = joint_angles[:, offset:offset + n]
            tip_hand = chain.forward(q)  # (B, 3) in hand frame
            # Transform to object frame: p_obj = R^T @ (p_hand - t) → p_obj = R @ p_hand + t
            tip_obj = torch.einsum("bij,bj->bi", R, tip_hand) + hand_translation
            tips.append(tip_obj)
            offset += n

        return torch.stack(tips, dim=1)  # (B, F, 3)

    def clamp_joints(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Clamp joint angles to valid range."""
        return torch.clamp(joint_angles, self.joint_lower, self.joint_upper)

    def sample_random_joints(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample uniformly random joint angles within limits."""
        low = self.joint_lower.to(device)
        high = self.joint_upper.to(device)
        u = torch.rand(batch_size, self.num_dof, device=device)
        return low + u * (high - low)


# ---------------------------------------------------------------------------
# Rotation utilities (differentiable)
# ---------------------------------------------------------------------------

def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        r6d: (..., 6) first two columns of rotation matrix

    Returns:
        R: (..., 3, 3) rotation matrix
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:]

    # Gram-Schmidt orthogonalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """Extract 6D representation from rotation matrix."""
    return R[..., :2].reshape(R.shape[:-2] + (6,))


def random_rotation_6d(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample random rotations in 6D representation."""
    # Random rotation via QR decomposition of random matrix
    M = torch.randn(batch_size, 3, 3, device=device)
    Q, _ = torch.linalg.qr(M)
    # Ensure proper rotation (det = +1)
    det = torch.det(Q)
    Q[:, :, 0] *= det.unsqueeze(-1).sign()
    return matrix_to_rotation_6d(Q)


# ---------------------------------------------------------------------------
# Pre-built hand models
# ---------------------------------------------------------------------------

def build_shadow_hand(num_fingers: int = 5) -> HandModel:
    """
    Build a simplified Shadow Hand model for grasp optimization.

    The DH parameters are approximate but capture the essential kinematics.
    Isaac Sim refinement corrects residual errors.

    Shadow Hand finger layout:
      0: Forefinger (FF) - 3 active joints (FFJ3, FFJ2, FFJ1)
      1: Middle (MF) - 3 active joints
      2: Ring (RF) - 3 active joints
      3: Little (LF) - 3 active joints
      4: Thumb (TH) - 4 active joints (THJ4, THJ3, THJ2, THJ1)
    """
    # Approximate link lengths (meters) from Shadow Hand specs
    FF_BASE = np.eye(4, dtype=np.float32)
    FF_BASE[:3, 3] = [0.033, 0.0, 0.095]  # FF base relative to palm

    MF_BASE = np.eye(4, dtype=np.float32)
    MF_BASE[:3, 3] = [0.011, 0.0, 0.099]

    RF_BASE = np.eye(4, dtype=np.float32)
    RF_BASE[:3, 3] = [-0.011, 0.0, 0.095]

    LF_BASE = np.eye(4, dtype=np.float32)
    LF_BASE[:3, 3] = [-0.033, 0.0, 0.086]

    TH_BASE = np.eye(4, dtype=np.float32)
    TH_BASE[:3, 3] = [0.034, -0.009, 0.029]
    # Thumb base has a rotation (abduction plane)
    TH_BASE[:3, :3] = _rot_y(-np.pi / 6) @ _rot_z(np.pi / 4)

    def _make_finger_links(name_prefix: str, proximal: float = 0.045,
                           middle: float = 0.025, distal: float = 0.026):
        return [
            LinkParam(f"{name_prefix}_J1", a=0.0, d=0.0, alpha=-np.pi / 2,
                      joint_min=0.0, joint_max=1.57),
            LinkParam(f"{name_prefix}_J2", a=proximal, d=0.0, alpha=0.0,
                      joint_min=0.0, joint_max=1.57),
            LinkParam(f"{name_prefix}_J3", a=middle, d=0.0, alpha=0.0,
                      joint_min=0.0, joint_max=1.57),
        ]

    def _make_thumb_links():
        return [
            LinkParam("TH_J4", a=0.0, d=0.0, alpha=np.pi / 2,
                      joint_min=-0.3, joint_max=1.2),   # abduction
            LinkParam("TH_J3", a=0.038, d=0.0, alpha=0.0,
                      joint_min=0.0, joint_max=1.2),
            LinkParam("TH_J2", a=0.032, d=0.0, alpha=0.0,
                      joint_min=-0.2, joint_max=1.2),
            LinkParam("TH_J1", a=0.027, d=0.0, alpha=0.0,
                      joint_min=-0.2, joint_max=1.2),
        ]

    all_fingers = [
        FingerChain("FF", _make_finger_links("FF"), FF_BASE),
        FingerChain("MF", _make_finger_links("MF"), MF_BASE),
        FingerChain("RF", _make_finger_links("RF"), RF_BASE),
        FingerChain("LF", _make_finger_links("LF", proximal=0.040, middle=0.025, distal=0.022), LF_BASE),
        FingerChain("TH", _make_thumb_links(), TH_BASE),
    ]

    # Select fingers based on num_fingers
    finger_subsets = {
        2: [0, 4],           # FF + TH
        3: [0, 1, 4],        # FF + MF + TH
        4: [0, 1, 2, 4],     # FF + MF + RF + TH
        5: [0, 1, 2, 3, 4],  # All
    }
    indices = finger_subsets.get(num_fingers, list(range(min(num_fingers, 5))))
    selected = [all_fingers[i] for i in indices]

    return HandModel(selected, name="shadow")


def build_allegro_hand(num_fingers: int = 4) -> HandModel:
    """
    Build a simplified Allegro Hand model for grasp optimization.

    Allegro Hand: 16 DOF, 4 fingers × 4 joints each.
    """
    # Approximate DH parameters for Allegro Hand
    FINGER_BASES = [
        np.array([[1, 0, 0, 0.0435], [0, 1, 0, -0.0017], [0, 0, 1, 0.0164], [0, 0, 0, 1]], dtype=np.float32),  # index
        np.array([[1, 0, 0, 0.0007], [0, 1, 0, 0.0003], [0, 0, 1, 0.0584], [0, 0, 0, 1]], dtype=np.float32),  # middle
        np.array([[1, 0, 0, -0.0435], [0, 1, 0, -0.0017], [0, 0, 1, 0.0164], [0, 0, 0, 1]], dtype=np.float32),  # ring
        np.array([[1, 0, 0, -0.0182], [0, 1, 0, -0.019], [0, 0, 1, -0.045], [0, 0, 0, 1]], dtype=np.float32),  # thumb
    ]

    def _make_allegro_finger(name: str, link_lengths=(0.054, 0.054, 0.054, 0.027)):
        return [
            LinkParam(f"{name}_J0", a=0.0, d=0.0, alpha=np.pi / 2,
                      joint_min=-0.47, joint_max=0.47),
            LinkParam(f"{name}_J1", a=link_lengths[0], d=0.0, alpha=0.0,
                      joint_min=-0.196, joint_max=1.61),
            LinkParam(f"{name}_J2", a=link_lengths[1], d=0.0, alpha=0.0,
                      joint_min=-0.174, joint_max=1.71),
            LinkParam(f"{name}_J3", a=link_lengths[2], d=0.0, alpha=0.0,
                      joint_min=-0.227, joint_max=1.618),
        ]

    all_fingers = [
        FingerChain("index", _make_allegro_finger("index"), FINGER_BASES[0]),
        FingerChain("middle", _make_allegro_finger("middle"), FINGER_BASES[1]),
        FingerChain("ring", _make_allegro_finger("ring"), FINGER_BASES[2]),
        FingerChain("thumb", _make_allegro_finger("thumb", (0.054, 0.054, 0.054, 0.027)), FINGER_BASES[3]),
    ]

    indices = list(range(min(num_fingers, 4)))
    if num_fingers < 4:
        # Always include thumb (index 3) for stable grasps
        if 3 not in indices:
            indices[-1] = 3
    selected = [all_fingers[i] for i in indices]

    return HandModel(selected, name="allegro")


def build_hand_model(hand_name: str = "shadow", num_fingers: int = 4) -> HandModel:
    """Factory function to build a hand model by name."""
    if hand_name == "shadow":
        return build_shadow_hand(num_fingers)
    elif hand_name == "allegro":
        return build_allegro_hand(num_fingers)
    else:
        raise ValueError(f"Unknown hand: {hand_name}. Supported: shadow, allegro")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
