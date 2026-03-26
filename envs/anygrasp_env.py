"""
Stage 1 – AnyGrasp-to-AnyGrasp Isaac Lab Environment
======================================================
Implements the core RL environment from DexterityGen §3.2.

Task definition (per paper):
  - Start: Allegro Hand holds the object in grasp g_start
  - Goal:  Transition to grasp g_goal (different fingertip configuration)
  - Both g_start and g_goal are drawn from the pre-built GraspGraph

Observation (object-centric frame):
  ┌──────────────────────────────────────────────────────┐
  │ joint positions          (16,)                       │
  │ joint velocities         (16,)                       │
  │ fingertip positions      (4×3 = 12,)  in obj frame   │
  │ target fingertip pos     (4×3 = 12,)  in obj frame   │
  │ object linear velocity   (3,)                        │
  │ object angular velocity  (3,)                        │
  │ last action              (16,)                       │
  └──────────────────────────────────────────────────────┘
  Total: 78 dims

Action: 16 joint position targets (Δ from current, clipped to ±0.2 rad)

Reward (per paper §3.2):
  r = r_fingertip + r_grasp_stable + r_smooth - r_drop
  where:
    r_fingertip   = exp(-α ||p_finger - p_target||)   tracking reward
    r_grasp_stable = bonus when all fingertips within ε of target
    r_smooth       = -β ||a_t - a_{t-1}||             action smoothness
    r_drop         = large penalty if object is dropped

Isaac Lab integration:
  - Uses ManagerBasedRLEnv (manager-based API, Isaac Lab 5.x)
  - Allegro Hand: isaaclab_assets ALLEGRO_HAND_CFG
  - Object: RigidObject (cube 6 cm)
  - Scene: table surface
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Optional

import numpy as np
import torch

# Isaac Lab imports (available after ./setup_isaaclab.sh)
try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import (
        EventTermCfg as EventTerm,
        ObservationGroupCfg as ObsGroup,
        ObservationTermCfg as ObsTerm,
        RewardTermCfg as RewTerm,
        SceneEntityCfg,
        TerminationTermCfg as DoneTerm,
    )
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    # Official Allegro Hand asset
    try:
        from isaaclab_assets.robots.allegro_hand import ALLEGRO_HAND_CFG  # noqa: F401
    except ImportError:
        # Fallback path for older asset package layout
        from isaaclab_assets import ALLEGRO_HAND_CFG  # noqa: F401

    _ISAACLAB_AVAILABLE = True

except ImportError:
    _ISAACLAB_AVAILABLE = False
    # Stubs so the rest of the file parses cleanly outside Isaac Lab
    def configclass(cls):  # noqa: F811
        return cls

from .mdp import rewards as mdp_rewards
from .mdp import observations as mdp_obs
from .mdp import events as mdp_events


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspSceneCfg(InteractiveSceneCfg):
        """Scene: table + Allegro Hand (mounted) + manipulated object."""

        # Ground / table
        ground = sim_utils.GroundPlaneCfg()

        # Allegro Hand mounted on a fixed pedestal
        # The hand is mounted above the table, palm facing down
        robot: ArticulationCfg = ALLEGRO_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/AllegroHand",
        )

        # Object to manipulate (default: 6 cm cube)
        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
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
                pos=(0.0, 0.0, 0.5),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Contact sensors on fingertips (optional, for stability check)
        num_envs: int = MISSING
        env_spacing: float = 0.5


# ---------------------------------------------------------------------------
# MDP Manager configurations
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspObservationsCfg:
        """Observation groups for the AnyGrasp-to-AnyGrasp task."""

        @configclass
        class PolicyObs(ObsGroup):
            """Concatenated observation fed to the RL policy."""

            joint_pos = ObsTerm(func=mdp_obs.joint_positions_normalized)
            joint_vel = ObsTerm(func=mdp_obs.joint_velocities_normalized)
            fingertip_pos = ObsTerm(func=mdp_obs.fingertip_positions_in_object_frame)
            target_fingertip_pos = ObsTerm(func=mdp_obs.target_fingertip_positions)
            object_lin_vel = ObsTerm(func=mdp_obs.object_linear_velocity)
            object_ang_vel = ObsTerm(func=mdp_obs.object_angular_velocity)
            last_action = ObsTerm(func=mdp_obs.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyObs = PolicyObs()


    @configclass
    class AnyGraspRewardsCfg:
        """Reward terms for AnyGrasp-to-AnyGrasp task."""

        # Primary: fingertip tracking (exponential distance reward)
        fingertip_tracking = RewTerm(
            func=mdp_rewards.fingertip_tracking_reward,
            weight=10.0,
            params={"alpha": 20.0},
        )

        # Success bonus: all fingertips within 1 cm of target
        grasp_success = RewTerm(
            func=mdp_rewards.grasp_success_reward,
            weight=50.0,
            params={"threshold": 0.01},
        )

        # Action smoothness penalty
        action_smoothness = RewTerm(
            func=mdp_rewards.action_smoothness_penalty,
            weight=-0.01,
        )

        # Object drop penalty
        object_drop = RewTerm(
            func=mdp_rewards.object_drop_penalty,
            weight=-200.0,
            params={"min_height": 0.3},
        )

        # Joint limit penalty
        joint_limit = RewTerm(
            func=mdp_rewards.joint_limit_penalty,
            weight=-0.1,
        )


    @configclass
    class AnyGraspTerminationsCfg:
        """Episode termination conditions."""

        time_out = DoneTerm(func=mdp_events.time_out, time_out=True)
        object_drop = DoneTerm(
            func=mdp_events.object_dropped,
            params={"min_height": 0.3},
        )


    @configclass
    class AnyGraspEventsCfg:
        """Reset / randomisation events."""

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
        Full environment config for AnyGrasp-to-AnyGrasp.

        Usage:
            from envs import AnyGraspEnvCfg
            cfg = AnyGraspEnvCfg()
            cfg.scene.num_envs = 512
            cfg.grasp_graph_path = "data/grasp_graph.pkl"
            env = ManagerBasedRLEnv(cfg)
        """

        # Scene
        scene: AnyGraspSceneCfg = AnyGraspSceneCfg(num_envs=MISSING, env_spacing=0.5)

        # Managers
        observations: AnyGraspObservationsCfg = AnyGraspObservationsCfg()
        rewards: AnyGraspRewardsCfg = AnyGraspRewardsCfg()
        terminations: AnyGraspTerminationsCfg = AnyGraspTerminationsCfg()
        events: AnyGraspEventsCfg = AnyGraspEventsCfg()

        # Task-specific
        grasp_graph_path: str = "data/grasp_graph.pkl"
        episode_length_s: float = 10.0     # 10 seconds per episode
        action_scale: float = 0.1          # joint position delta scale
        decimation: int = 4                # control freq = sim_freq / decimation

        def __post_init__(self):
            super().__post_init__()
            self.sim.dt = 1.0 / 120.0     # 120 Hz physics
            self.sim.render_interval = self.decimation


# ---------------------------------------------------------------------------
# Gymnasium registration helper
# ---------------------------------------------------------------------------

def register_anygrasp_env():
    """
    Register the AnyGrasp environment with gymnasium.
    Call this before creating the env.

    Example:
        register_anygrasp_env()
        env = gym.make("DexGen-AnyGrasp-Allegro-v0", cfg=cfg)
    """
    if not _ISAACLAB_AVAILABLE:
        raise RuntimeError("Isaac Lab is not installed. Run ./setup_isaaclab.sh first.")

    import gymnasium as gym

    gym.register(
        id="DexGen-AnyGrasp-Allegro-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"cfg": AnyGraspEnvCfg()},
        disable_env_checker=True,
    )
    print("[AnyGraspEnv] Registered 'DexGen-AnyGrasp-Allegro-v0'")
