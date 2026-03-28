"""
DexGraspNet Adapter
===================
Wraps the DexGraspNet differentiable grasp optimization pipeline
(PKU-EPIC/DexGraspNet) and converts its output to the :class:`Grasp` /
:class:`GraspGraph` format used by our RL training pipeline.

DexGraspNet's algorithm:
  1. Initialise hand poses on an inflated convex hull of the object
  2. Run gradient-guided simulated annealing (6 000 iters default)
     minimising: E_fc + w_dis·E_dis + w_pen·E_pen + w_spen·E_spen + w_joints·E_joints
  3. Filter results by energy thresholds
  4. Output: (wrist_translation, wrist_rotation_6d, joint_angles) per grasp

Design for extensibility:
  - :class:`HandConfig` encapsulates all hand-specific paths and naming
    so that switching from Allegro to Shadow or a custom hand only
    requires a new HandConfig instance.
  - DexGraspNet code is called as-is from the submodule; we only
    consume its outputs.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# DexGraspNet import setup
# ---------------------------------------------------------------------------
_DEXGRASPNET_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "DexGraspNet"
_DEXGRASPNET_GEN = _DEXGRASPNET_ROOT / "grasp_generation"

_TRANSLATION_NAMES = ["WRJTx", "WRJTy", "WRJTz"]
_ROTATION_NAMES = ["WRJRx", "WRJRy", "WRJRz"]


def _ensure_dexgraspnet_on_path():
    """Add DexGraspNet to sys.path and ensure __init__.py files exist.

    DexGraspNet's source tree ships without __init__.py in its utils/
    directory.  We create them at runtime so that ``from utils.hand_model
    import HandModel`` works when the grasp_generation dir is on sys.path.
    """
    gen_str = str(_DEXGRASPNET_GEN)
    if gen_str not in sys.path:
        sys.path.insert(0, gen_str)

    # Create missing __init__.py files so Python treats dirs as packages
    for subdir in [_DEXGRASPNET_GEN, _DEXGRASPNET_GEN / "utils"]:
        init_file = subdir / "__init__.py"
        if not init_file.exists() and subdir.is_dir():
            init_file.write_text("")


# ---------------------------------------------------------------------------
# Hand configuration (extensible for any dexterous hand)
# ---------------------------------------------------------------------------

@dataclass
class HandConfig:
    """
    Hand-specific configuration for DexGraspNet.

    To add a new hand:
      1. Create a URDF file in DexGraspNet format
      2. Create a contact_points.json with per-link contact candidates
      3. Instantiate HandConfig with the correct paths and link names

    Attributes
    ----------
    urdf_path : Path
        Path to the hand URDF (DexGraspNet format).
    contact_points_path : Path
        Path to the contact_points.json for this hand.
    fingertip_links : list[str]
        Ordered list of fingertip link names in the URDF.
        Order must match our convention: [index, middle, ring, thumb, ...]
    joint_names : list[str]
        Ordered list of joint names as used in DexGraspNet's qpos dict.
    n_dof : int
        Number of actuated DOF.
    isaac_lab_joint_mapping : list[int] | None
        If DexGraspNet joint ordering differs from Isaac Lab, provide
        a mapping: ``isaac_joint[i] = dgn_joint[mapping[i]]``.
        None means same ordering.
    """
    urdf_path: Path = field(default_factory=lambda: (
        _DEXGRASPNET_GEN / "allegro_hand_description"
        / "allegro_hand_description_right.urdf"
    ))
    contact_points_path: Path = field(default_factory=lambda: (
        _DEXGRASPNET_GEN / "allegro_hand_description" / "contact_points.json"
    ))
    fingertip_links: List[str] = field(default_factory=lambda: [
        "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip",
    ])
    joint_names: List[str] = field(default_factory=lambda: [
        f"joint_{i}.0" for i in range(16)
    ])
    n_dof: int = 16
    isaac_lab_joint_mapping: Optional[List[int]] = None

    @classmethod
    def allegro(cls) -> "HandConfig":
        """Pre-configured for Wonik Allegro Hand (right, 4 fingers, 16 DOF)."""
        return cls()

    @classmethod
    def from_custom(
        cls,
        urdf_path: str | Path,
        contact_points_path: str | Path,
        fingertip_links: List[str],
        joint_names: List[str],
        n_dof: int,
        isaac_lab_joint_mapping: Optional[List[int]] = None,
    ) -> "HandConfig":
        """Create a HandConfig for a custom dexterous hand."""
        return cls(
            urdf_path=Path(urdf_path),
            contact_points_path=Path(contact_points_path),
            fingertip_links=fingertip_links,
            joint_names=joint_names,
            n_dof=n_dof,
            isaac_lab_joint_mapping=isaac_lab_joint_mapping,
        )


# ---------------------------------------------------------------------------
# Optimisation configuration
# ---------------------------------------------------------------------------

@dataclass
class DexGraspNetConfig:
    """Hyperparameters for DexGraspNet grasp optimisation."""
    # Batch & iteration
    batch_size: int = 128
    n_iter: int = 6000
    n_contact: int = 4

    # Annealing
    switch_possibility: float = 0.5
    starting_temperature: float = 18.0
    temperature_decay: float = 0.95
    annealing_period: int = 30
    noise_size: float = 0.005
    stepsize_period: int = 50
    mu: float = 0.98

    # Energy weights
    w_dis: float = 100.0
    w_pen: float = 100.0
    w_spen: float = 30.0
    w_joints: float = 1.0

    # Initialisation
    jitter_strength: float = 0.1
    distance_lower: float = 0.2
    distance_upper: float = 0.3
    theta_lower: float = -math.pi / 6
    theta_upper: float = math.pi / 6

    # Filtering thresholds
    thres_fc: float = 0.3
    thres_dis: float = 0.005
    thres_pen: float = 0.001

    # Device
    gpu: str = "0"

    # Seed
    seed: int = 42

    # Hand configuration
    hand: HandConfig = field(default_factory=HandConfig.allegro)


# ---------------------------------------------------------------------------
# Core runner — calls DexGraspNet as-is
# ---------------------------------------------------------------------------

def run_dexgraspnet_optimization(
    object_code: str,
    meshdata_root: str | Path,
    object_scale: float,
    cfg: DexGraspNetConfig | None = None,
    device: str = "cuda",
) -> List[dict]:
    """
    Run DexGraspNet's differentiable grasp optimisation on a single object.

    Uses DexGraspNet's own HandModel, ObjectModel, energy functions, and
    Annealing optimizer as-is from the submodule.  We only set up the
    inputs and collect the outputs.

    Parameters
    ----------
    object_code : str
        Object identifier (must match a directory in ``meshdata_root``).
    meshdata_root : path-like
        Root directory containing ``{object_code}/coacd/decomposed.obj``.
    object_scale : float
        Scale returned by :func:`mesh_export.export_mesh_for_dexgraspnet`.
    cfg : DexGraspNetConfig, optional
    device : str

    Returns
    -------
    results : list[dict]
        One dict per grasp: ``qpos``, ``energy``, ``E_fc``, ``E_dis``,
        ``E_pen``, ``E_spen``, ``E_joints``, ``scale``, ``_hand_pose``.
    """
    import os
    import torch
    from tqdm import tqdm

    _ensure_dexgraspnet_on_path()

    from utils.hand_model import HandModel
    from utils.object_model import ObjectModel
    from utils.initializations import initialize_convex_hull
    from utils.energy import cal_energy
    from utils.optimizer import Annealing
    from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
    import transforms3d

    if cfg is None:
        cfg = DexGraspNetConfig()

    hand_cfg = cfg.hand

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    torch_device = torch.device(device)

    # ---- Hand model (uses HandConfig for extensibility) ----
    hand_model = HandModel(
        urdf_path=str(hand_cfg.urdf_path),
        contact_points_path=str(hand_cfg.contact_points_path),
        n_surface_points=1000,
        device=torch_device,
    )

    # ---- Object model ----
    object_model = ObjectModel(
        data_root_path=str(meshdata_root),
        batch_size_each=cfg.batch_size,
        num_samples=2000,
        device=torch_device,
    )
    object_model.initialize([object_code])
    # Override scale to match our specific object
    object_model.object_scale_tensor = torch.full(
        (1, cfg.batch_size), object_scale, dtype=torch.float, device=torch_device,
    )

    # ---- Initialisation ----
    class _Args:
        pass
    init_args = _Args()
    init_args.n_contact = cfg.n_contact
    init_args.jitter_strength = cfg.jitter_strength
    init_args.distance_lower = cfg.distance_lower
    init_args.distance_upper = cfg.distance_upper
    init_args.theta_lower = cfg.theta_lower
    init_args.theta_upper = cfg.theta_upper

    initialize_convex_hull(hand_model, object_model, init_args)
    hand_pose_st = hand_model.hand_pose.detach().clone()

    # ---- Optimizer ----
    optimizer = Annealing(
        hand_model,
        switch_possibility=cfg.switch_possibility,
        starting_temperature=cfg.starting_temperature,
        temperature_decay=cfg.temperature_decay,
        annealing_period=cfg.annealing_period,
        noise_size=cfg.noise_size,
        stepsize_period=cfg.stepsize_period,
        mu=cfg.mu,
        device=torch_device,
    )

    weight_dict = dict(
        w_dis=cfg.w_dis, w_pen=cfg.w_pen,
        w_spen=cfg.w_spen, w_joints=cfg.w_joints,
    )

    # ---- Initial energy ----
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(
        hand_model, object_model, verbose=True, **weight_dict,
    )
    energy.sum().backward(retain_graph=True)

    # ---- Optimisation loop (DexGraspNet as-is) ----
    for step in tqdm(range(1, cfg.n_iter + 1), desc=f"DexGraspNet [{object_code}]"):
        optimizer.try_step()
        optimizer.zero_grad()

        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = \
            cal_energy(hand_model, object_model, verbose=True, **weight_dict)
        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)
            energy[accept] = new_energy[accept]
            E_fc[accept] = new_E_fc[accept]
            E_dis[accept] = new_E_dis[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]

    # ---- Collect results ----
    joint_names = hand_cfg.joint_names
    results = []
    for j in range(cfg.batch_size):
        scale = object_model.object_scale_tensor[0][j].item()
        hand_pose = hand_model.hand_pose[j].detach().cpu()

        qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        euler = transforms3d.euler.mat2euler(rot.numpy(), axes="sxyz")
        qpos.update(dict(zip(_ROTATION_NAMES, euler)))
        qpos.update(dict(zip(_TRANSLATION_NAMES, hand_pose[:3].tolist())))

        results.append(dict(
            scale=scale,
            qpos=qpos,
            energy=energy[j].item(),
            E_fc=E_fc[j].item(),
            E_dis=E_dis[j].item(),
            E_pen=E_pen[j].item(),
            E_spen=E_spen[j].item(),
            E_joints=E_joints[j].item(),
            _hand_pose=hand_pose,
        ))

    return results


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_by_energy(
    results: List[dict],
    thres_fc: float = 0.3,
    thres_dis: float = 0.005,
    thres_pen: float = 0.001,
) -> List[dict]:
    """Filter by per-component energy thresholds."""
    filtered = [
        r for r in results
        if r["E_fc"] <= thres_fc
        and r["E_dis"] <= thres_dis
        and r["E_pen"] <= thres_pen
    ]
    print(f"[DexGraspNet filter] {len(filtered)}/{len(results)} passed "
          f"(fc≤{thres_fc}, dis≤{thres_dis}, pen≤{thres_pen})")
    return filtered


# ---------------------------------------------------------------------------
# Conversion to our Grasp format
# ---------------------------------------------------------------------------

def _extract_fingertip_positions(
    hand_model,
    hand_pose_tensor,
    fingertip_links: List[str],
    device,
) -> np.ndarray:
    """
    Run FK on a single hand_pose and extract fingertip positions in world frame.

    Returns (num_fingers, 3) array.
    """
    import torch
    _ensure_dexgraspnet_on_path()
    from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

    hp = hand_pose_tensor.to(device).unsqueeze(0)
    translation = hp[:, :3]
    rotation = robust_compute_rotation_matrix_from_ortho6d(hp[:, 3:9])
    fk_status = hand_model.chain.forward_kinematics(hp[:, 9:])

    tips = []
    for link_name in fingertip_links:
        tip_local = hand_model.mesh[link_name]["contact_candidates"]
        tip_homo = torch.cat([
            tip_local,
            torch.ones(tip_local.shape[0], 1, dtype=torch.float, device=device),
        ], dim=1)
        T = fk_status[link_name].get_matrix()
        tip_wrist = (T @ tip_homo.unsqueeze(-1))[:, :3, 0]
        tip_world = tip_wrist @ rotation.transpose(1, 2) + translation
        tips.append(tip_world[0].detach().cpu().numpy())

    return np.stack(tips, axis=0)


def _compute_contact_normals(
    fingertip_positions: np.ndarray,
    mesh: trimesh.Trimesh,
    scale: float,
) -> np.ndarray:
    """Compute outward surface normals at closest mesh points to each fingertip."""
    pts_normalised = fingertip_positions / scale
    _, _, face_ids = mesh.nearest.on_surface(pts_normalised)
    return mesh.face_normals[face_ids].astype(np.float32)


def convert_results_to_grasps(
    results: List[dict],
    object_code: str,
    mesh: trimesh.Trimesh,
    hand_cfg: HandConfig | None = None,
    hand_model=None,
    device=None,
) -> "List[Grasp]":
    """
    Convert DexGraspNet results → our Grasp format.

    Parameters
    ----------
    results : list[dict]
        From :func:`run_dexgraspnet_optimization`.
    object_code : str
    mesh : trimesh.Trimesh
        Unit-normalised mesh.
    hand_cfg : HandConfig, optional
    hand_model : HandModel, optional
        Reuse an existing model to avoid re-loading URDF.
    device : optional
    """
    import torch
    _ensure_dexgraspnet_on_path()
    import transforms3d

    from grasp_generation.grasp_sampler import Grasp

    if hand_cfg is None:
        hand_cfg = HandConfig.allegro()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hand_model is None:
        from utils.hand_model import HandModel
        hand_model = HandModel(
            urdf_path=str(hand_cfg.urdf_path),
            contact_points_path=str(hand_cfg.contact_points_path),
            n_surface_points=0,
            device=device,
        )

    joint_names = hand_cfg.joint_names
    fingertip_links = hand_cfg.fingertip_links
    mapping = hand_cfg.isaac_lab_joint_mapping

    grasps = []
    for r in results:
        qpos = r["qpos"]
        scale = r["scale"]

        # Extract joint angles
        dgn_joints = np.array(
            [qpos[name] for name in joint_names], dtype=np.float32,
        )
        # Apply mapping if DexGraspNet ordering differs from Isaac Lab
        joint_angles = dgn_joints[mapping] if mapping is not None else dgn_joints

        # Wrist transform
        translation = np.array(
            [qpos[n] for n in _TRANSLATION_NAMES], dtype=np.float32,
        )
        euler = [qpos[n] for n in _ROTATION_NAMES]
        rot_matrix = np.array(
            transforms3d.euler.euler2mat(*euler, axes="sxyz"), dtype=np.float32,
        )

        # FK → fingertip positions in world frame (= object frame)
        hand_pose = r.get("_hand_pose")
        if hand_pose is not None:
            tips_world = _extract_fingertip_positions(
                hand_model, hand_pose, fingertip_links, device,
            )
        else:
            rot6d = rot_matrix[:, :2].T.flatten()
            hand_pose = torch.tensor(
                np.concatenate([translation, rot6d, dgn_joints]),
                dtype=torch.float,
            )
            tips_world = _extract_fingertip_positions(
                hand_model, hand_pose, fingertip_links, device,
            )

        fingertip_positions = tips_world.astype(np.float32)
        contact_normals = _compute_contact_normals(fingertip_positions, mesh, scale)

        energy = r.get("energy", 0.0)
        quality = 1.0 / (1.0 + max(0.0, energy))

        # Object pose in wrist frame (object at origin)
        object_pos_hand = (-rot_matrix.T @ translation).astype(np.float32)
        object_quat_hand = np.array(
            transforms3d.quaternions.mat2quat(rot_matrix.T), dtype=np.float32,
        )

        grasps.append(Grasp(
            fingertip_positions=fingertip_positions,
            contact_normals=contact_normals,
            quality=quality,
            object_name=object_code,
            object_scale=scale,
            joint_angles=joint_angles,
            object_pos_hand=object_pos_hand,
            object_quat_hand=object_quat_hand,
        ))

    print(f"[DexGraspNet adapter] Converted {len(grasps)} grasps for {object_code}")
    return grasps


# ---------------------------------------------------------------------------
# High-level API: generate + filter + convert
# ---------------------------------------------------------------------------

def generate_grasps_dexgraspnet(
    object_code: str,
    meshdata_root: str | Path,
    object_scale: float,
    cfg: DexGraspNetConfig | None = None,
    target_num_grasps: int = 300,
    device: str = "cuda",
) -> "List[Grasp]":
    """
    End-to-end DexGraspNet grasp generation for a single object.

    Replaces the GraspSampler → NFO → RRT pipeline:
      1. Run DexGraspNet optimisation (simulated annealing)
      2. Filter by energy thresholds
      3. Convert to our Grasp format

    If fewer than ``target_num_grasps`` pass the filter, thresholds
    are progressively relaxed.
    """
    if cfg is None:
        cfg = DexGraspNetConfig()

    # Overshoot ~3x to account for filtering
    batches_needed = max(1, (target_num_grasps * 3) // cfg.batch_size + 1)

    all_results: List[dict] = []
    filtered: List[dict] = []

    for batch_idx in range(batches_needed):
        batch_cfg = DexGraspNetConfig(**{
            **cfg.__dict__,
            "seed": cfg.seed + batch_idx,
        })
        results = run_dexgraspnet_optimization(
            object_code=object_code,
            meshdata_root=meshdata_root,
            object_scale=object_scale,
            cfg=batch_cfg,
            device=device,
        )
        all_results.extend(results)

        filtered = filter_by_energy(
            all_results,
            thres_fc=cfg.thres_fc,
            thres_dis=cfg.thres_dis,
            thres_pen=cfg.thres_pen,
        )
        if len(filtered) >= target_num_grasps:
            break

    # Progressive threshold relaxation
    if len(filtered) < target_num_grasps:
        for relax_factor in [2.0, 5.0, 10.0]:
            filtered = filter_by_energy(
                all_results,
                thres_fc=cfg.thres_fc * relax_factor,
                thres_dis=cfg.thres_dis * relax_factor,
                thres_pen=cfg.thres_pen * relax_factor,
            )
            if len(filtered) >= target_num_grasps:
                break

    # Best grasps first
    filtered.sort(key=lambda r: r["energy"])
    filtered = filtered[:target_num_grasps]

    # Convert to our format
    meshdata_root = Path(meshdata_root)
    unit_mesh_path = meshdata_root / object_code / "coacd" / "decomposed.obj"
    unit_mesh = trimesh.load(str(unit_mesh_path), force="mesh", process=False)

    grasps = convert_results_to_grasps(
        filtered, object_code, unit_mesh,
        hand_cfg=cfg.hand, device=device,
    )
    return grasps
