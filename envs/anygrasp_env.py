"""
Stage 1 – AnyGrasp-to-AnyGrasp Isaac Lab Environment
======================================================
Implements the core RL environment from DexterityGen §3.2,
extended with:
  - Random object pool (cube / sphere / cylinder, various sizes)
    via Isaac Lab MultiAssetSpawnerCfg
  - Random Allegro Hand wrist position at every episode reset

Task definition:
  - Start: Allegro Hand holds the object in grasp g_start
  - Goal:  Transition to grasp g_goal (different fingertip configuration)
  - Object AND wrist pose are re-randomised every episode

Observation (object-centric frame):
  ┌──────────────────────────────────────────────────────┐
  │ joint positions          (16,)                       │
  │ joint velocities         (16,)                       │
  │ fingertip positions      (4×3=12,)   in obj frame    │
  │ target fingertip pos     (4×3=12,)   in obj frame    │
  │ object linear velocity   (3,)                        │
  │ object angular velocity  (3,)                        │
  │ last action              (16,)                       │
  └──────────────────────────────────────────────────────┘
  Total: 78 dims

Action: 16 joint position targets (Δ from current, scale=0.1 rad)
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import List, Optional

import numpy as np
import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import (
        EventTermCfg as EventTerm,
        ObservationGroupCfg as ObsGroup,
        ObservationTermCfg as ObsTerm,
        RewardTermCfg as RewTerm,
        TerminationTermCfg as DoneTerm,
    )
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    try:
        from isaaclab_assets.robots.allegro_hand import ALLEGRO_HAND_CFG
    except ImportError:
        from isaaclab_assets import ALLEGRO_HAND_CFG

    _ISAACLAB_AVAILABLE = True

except ImportError:
    _ISAACLAB_AVAILABLE = False
    def configclass(cls):
        return cls

from .mdp import rewards as mdp_rewards
from .mdp import observations as mdp_obs
from .mdp import events as mdp_events


# ---------------------------------------------------------------------------
# Helper: build MultiAssetSpawnerCfg from object pool specs
# ---------------------------------------------------------------------------

def _build_object_spawner(object_pool_specs: Optional[List[dict]] = None):
    """
    Build an Isaac Lab spawner for the object.

    If object_pool_specs is provided (list of dicts with shape_type/size/mass/color),
    returns a MultiAssetSpawnerCfg so each env instance gets one of the pool objects.
    Otherwise falls back to a default 6 cm cube.
    """
    if not _ISAACLAB_AVAILABLE:
        return None

    if not object_pool_specs:
        # Default: single 6 cm cube
        return sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        )

    # Build one spawner config per object spec
    spawner_list = []
    for spec in object_pool_specs:
        shape = spec["shape_type"]
        s = spec["size"]
        color = tuple(spec.get("color", (0.7, 0.7, 0.7)))
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        )
        mass_props = sim_utils.MassPropertiesCfg(mass=spec.get("mass", 0.1))
        col_props = sim_utils.CollisionPropertiesCfg()
        vis_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color)

        if shape == "cube":
            cfg = sim_utils.CuboidCfg(
                size=(s, s, s),
                rigid_props=rigid_props, mass_props=mass_props,
                collision_props=col_props, visual_material=vis_mat,
            )
        elif shape == "sphere":
            cfg = sim_utils.SphereCfg(
                radius=s / 2,
                rigid_props=rigid_props, mass_props=mass_props,
                collision_props=col_props, visual_material=vis_mat,
            )
        elif shape == "cylinder":
            cfg = sim_utils.CylinderCfg(
                radius=s / 2,
                height=s,
                rigid_props=rigid_props, mass_props=mass_props,
                collision_props=col_props, visual_material=vis_mat,
            )
        else:
            continue
        spawner_list.append(cfg)

    if len(spawner_list) == 1:
        return spawner_list[0]

    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=spawner_list,
        random_choice=True,   # each env gets a random shape at spawn
    )


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspSceneCfg(InteractiveSceneCfg):
        """
        Scene: ground + Allegro Hand + randomised object pool.

        The object spawner is set to MultiAssetSpawnerCfg so each parallel
        env instance gets one of the pool objects at scene creation.
        The wrist (robot base) pose and object pose are further randomised
        per-episode in the reset event.
        """

        ground = sim_utils.GroundPlaneCfg()

        # Allegro Hand — base pose is overridden every reset by events.py
        robot: ArticulationCfg = ALLEGRO_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/AllegroHand",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.6),          # default above-table position
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Object — spawner is replaced at env build time via AnyGraspEnvCfg
        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(         # placeholder, overridden below
                size=(0.06, 0.06, 0.06),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        num_envs: int = MISSING
        env_spacing: float = 0.6   # slightly larger spacing for wrist randomization


# ---------------------------------------------------------------------------
# MDP Manager configurations
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspObservationsCfg:
        @configclass
        class PolicyObs(ObsGroup):
            joint_pos        = ObsTerm(func=mdp_obs.joint_positions_normalized)
            joint_vel        = ObsTerm(func=mdp_obs.joint_velocities_normalized)
            fingertip_pos    = ObsTerm(func=mdp_obs.fingertip_positions_in_object_frame)
            target_fingertip = ObsTerm(func=mdp_obs.target_fingertip_positions)
            object_lin_vel   = ObsTerm(func=mdp_obs.object_linear_velocity)
            object_ang_vel   = ObsTerm(func=mdp_obs.object_angular_velocity)
            last_action      = ObsTerm(func=mdp_obs.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyObs = PolicyObs()


    @configclass
    class AnyGraspRewardsCfg:
        fingertip_tracking = RewTerm(
            func=mdp_rewards.fingertip_tracking_reward,
            weight=10.0,
            params={"alpha": 20.0},
        )
        grasp_success = RewTerm(
            func=mdp_rewards.grasp_success_reward,
            weight=50.0,
            params={"threshold": 0.01},
        )
        action_smoothness = RewTerm(
            func=mdp_rewards.action_smoothness_penalty,
            weight=-0.01,
        )
        object_drop = RewTerm(
            func=mdp_rewards.object_drop_penalty,
            weight=-200.0,
            params={"min_height": 0.2},
        )
        joint_limit = RewTerm(
            func=mdp_rewards.joint_limit_penalty,
            weight=-0.1,
        )


    @configclass
    class AnyGraspTerminationsCfg:
        time_out = DoneTerm(func=mdp_events.time_out, time_out=True)
        object_drop = DoneTerm(
            func=mdp_events.object_dropped,
            params={"min_height": 0.2},
        )


    @configclass
    class AnyGraspEventsCfg:
        reset_all = EventTerm(
            func=mdp_events.reset_to_random_grasp,
            mode="reset",
        )


# ---------------------------------------------------------------------------
# Top-level Environment Config
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspEnvCfg(ManagerBasedRLEnvCfg):
        """
        Full environment config for AnyGrasp-to-AnyGrasp with:
          - Random object pool (shape + size randomised per env)
          - Random wrist position per episode

        Key config fields:
            grasp_graph_path:    path to MultiObjectGraspGraph
            object_pool_specs:   list of dicts from ObjectPool.get_isaac_lab_specs()
            wrist_randomization: dict with pos_radius, pos_height, rot_std params
        """

        scene: AnyGraspSceneCfg = AnyGraspSceneCfg(num_envs=MISSING, env_spacing=0.6)

        observations:  AnyGraspObservationsCfg = AnyGraspObservationsCfg()
        rewards:       AnyGraspRewardsCfg      = AnyGraspRewardsCfg()
        terminations:  AnyGraspTerminationsCfg = AnyGraspTerminationsCfg()
        events:        AnyGraspEventsCfg       = AnyGraspEventsCfg()

        # Grasp graph
        grasp_graph_path: str = "data/grasp_graph.pkl"

        # Object pool specs — set via set_object_pool() below
        # Each entry: {name, shape_type, size, mass, color}
        object_pool_specs: list = None   # type: ignore

        # Wrist randomization range (see events._randomise_wrist_pose)
        wrist_randomization: dict = None   # type: ignore

        episode_length_s: float = 10.0
        action_scale: float = 0.1
        decimation: int = 4

        def __post_init__(self):
            super().__post_init__()
            self.sim.dt = 1.0 / 120.0
            self.sim.render_interval = self.decimation

            # If object pool specs are provided, replace the default cube spawner
            if self.object_pool_specs:
                spawner = _build_object_spawner(self.object_pool_specs)
                self.scene.object = self.scene.object.replace(spawn=spawner)

            # Default wrist randomization if not set
            if self.wrist_randomization is None:
                self.wrist_randomization = {
                    "pos_radius_min": 0.12,   # min distance from object centre (m)
                    "pos_radius_max": 0.22,   # max distance
                    "pos_height_min": 0.08,   # min height above object
                    "pos_height_max": 0.20,   # max height above object
                    "rot_std_deg": 15.0,      # wrist orientation noise (degrees)
                }

    def set_object_pool(cfg: AnyGraspEnvCfg, pool) -> AnyGraspEnvCfg:
        """Convenience helper: attach an ObjectPool to the env config."""
        cfg.object_pool_specs = pool.get_isaac_lab_specs()
        spawner = _build_object_spawner(cfg.object_pool_specs)
        cfg.scene.object = cfg.scene.object.replace(spawn=spawner)
        return cfg


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

def register_anygrasp_env():
    if not _ISAACLAB_AVAILABLE:
        raise RuntimeError("Isaac Lab not installed. Run ./setup_isaaclab.sh first.")

    import gymnasium as gym
    gym.register(
        id="DexGen-AnyGrasp-Allegro-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"cfg": AnyGraspEnvCfg()},
        disable_env_checker=True,
    )
    print("[AnyGraspEnv] Registered 'DexGen-AnyGrasp-Allegro-v0'")
