"""
Stage 1 – AnyGrasp-to-AnyGrasp Isaac Lab Environment
======================================================
Implements the core RL environment from DexterityGen §3.2 with:
  - Asymmetric Actor-Critic (separate policy / critic observation groups)
  - Tactile sensing via ContactSensorCfg on 4 fingertip links
  - Domain Randomization (object physics, joint dynamics, action delay)
  - Random object pool (cube / sphere / cylinder, multiple sizes)
  - Random Allegro Hand wrist position per episode

=======================================================================
  OBSERVATION SPLIT  (see mdp/observations.py for full details)
=======================================================================

  ACTOR (policy) — 76 dims
  ─────────────────────────────────────────────────────────────────
  joint_pos_normalized       16   (encoder, normalised)
  joint_vel_normalized       16   (encoder derivative)
  fingertip_pos_obj_frame    12   (FK in object-centric frame)
  target_fingertip_pos       12   (goal from GraspGraph)
  fingertip_contact_binary    4   (tactile: binary contact per tip)
  last_action                16   (previous joint targets)
  ─────────────────────────────────────────────────────────────────
  Total: 76

  CRITIC (privileged) — 104 dims
  ─────────────────────────────────────────────────────────────────
  [actor obs]                76
  object_pos_world            3   (true 3-D position)
  object_quat_world           4   (true orientation)
  object_lin_vel              3   (true linear velocity)
  object_ang_vel              3   (true angular velocity)
  fingertip_contact_forces   12   (full 3-D forces per tip)
  dr_params                   3   (mass / friction / damping)
  ─────────────────────────────────────────────────────────────────
  Total: 104

=======================================================================
  DOMAIN RANDOMIZATION  (see mdp/domain_rand.py for ranges)
=======================================================================
  Per episode reset:
    - Object mass:         Uniform(0.03, 0.30) kg
    - Object friction:     Uniform(0.30, 1.20)
    - Joint damping:       per-joint Uniform(0.01, 0.30)
    - Joint armature:      per-joint Uniform(0.001, 0.03)
    - Action delay:        0–2 steps
  Per step (obs corruption):
    - joint_pos noise:     N(0, 0.005) rad
    - joint_vel noise:     N(0, 0.04)  (normalised space)
    - fingertip pos noise: N(0, 0.003) m
=======================================================================
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import List, Optional

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
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

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
from .mdp import domain_rand as mdp_dr


# ---------------------------------------------------------------------------
# Helper: build MultiAssetSpawnerCfg from object pool
# ---------------------------------------------------------------------------

def _build_object_spawner(object_pool_specs: Optional[List[dict]] = None):
    if not _ISAACLAB_AVAILABLE:
        return None

    if not object_pool_specs:
        return sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        )

    spawner_list = []
    for spec in object_pool_specs:
        shape, s = spec["shape_type"], spec["size"]
        color = tuple(spec.get("color", (0.7, 0.7, 0.7)))
        rp = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0)
        mp = sim_utils.MassPropertiesCfg(mass=spec.get("mass", 0.1))
        cp = sim_utils.CollisionPropertiesCfg()
        vm = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        if shape == "cube":
            cfg = sim_utils.CuboidCfg(size=(s, s, s), rigid_props=rp, mass_props=mp, collision_props=cp, visual_material=vm)
        elif shape == "sphere":
            cfg = sim_utils.SphereCfg(radius=s/2, rigid_props=rp, mass_props=mp, collision_props=cp, visual_material=vm)
        elif shape == "cylinder":
            cfg = sim_utils.CylinderCfg(radius=s/2, height=s, rigid_props=rp, mass_props=mp, collision_props=cp, visual_material=vm)
        else:
            continue
        spawner_list.append(cfg)

    if len(spawner_list) == 1:
        return spawner_list[0]
    return sim_utils.MultiAssetSpawnerCfg(assets_cfg=spawner_list, random_choice=True)


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspSceneCfg(InteractiveSceneCfg):
        """
        Scene: ground + Allegro Hand + randomised object pool
               + ContactSensors on 4 fingertip links.
        """

        ground = sim_utils.GroundPlaneCfg()

        # Allegro Hand (wrist pose randomised per episode in events.py)
        robot: ArticulationCfg = ALLEGRO_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/AllegroHand",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.6),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Manipulated object (spawner replaced by pool at cfg build time)
        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.06, 0.06),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4), rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # -----------------------------------------------------------------------
        # Tactile ContactSensors on 4 fingertip links
        # -----------------------------------------------------------------------
        # The standard Allegro Hand has no real tactile sensors.
        # We attach ContactSensors to the 4 fingertip links to get physics-based
        # contact force measurements.
        #
        # The sensor pattern matches all four fingertip links:
        #   link_3.0_tip, link_7.0_tip, link_11.0_tip, link_15.0_tip
        #
        # From this we derive:
        #   - Binary contact indicator  → actor obs (4 dims)
        #   - Full 3-D contact forces   → critic privileged obs (12 dims)
        # -----------------------------------------------------------------------
        fingertip_contact_sensor: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/AllegroHand/.*_tip",
            update_period=0.0,       # update every physics step
            history_length=1,        # only need current forces
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        num_envs:    int   = MISSING
        env_spacing: float = 0.6


# ---------------------------------------------------------------------------
# Asymmetric Actor-Critic Observation groups
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspObservationsCfg:
        """
        Two separate observation groups:
          policy  → actor  (76 dims, available at deployment)
          critic  → critic (104 dims, privileged, training only)
        """

        @configclass
        class PolicyObs(ObsGroup):
            """Actor observation — what the real robot can sense."""

            joint_pos = ObsTerm(
                func=mdp_obs.joint_positions_normalized,
                noise=GaussianNoise(std=0.005),   # encoder noise ±0.005 rad
            )
            joint_vel = ObsTerm(
                func=mdp_obs.joint_velocities_normalized,
                noise=GaussianNoise(std=0.04),    # 0.04 in normalised space ≈ 0.2 rad/s
            )
            fingertip_pos = ObsTerm(
                func=mdp_obs.fingertip_positions_in_object_frame,
                noise=GaussianNoise(std=0.003),   # FK position uncertainty ±3 mm
            )
            target_fingertip_pos = ObsTerm(func=mdp_obs.target_fingertip_positions)
            fingertip_contact    = ObsTerm(func=mdp_obs.fingertip_contact_binary)
            last_action          = ObsTerm(func=mdp_obs.last_action)

            def __post_init__(self):
                self.enable_corruption  = True   # apply noise above
                self.concatenate_terms  = True

        @configclass
        class CriticObs(ObsGroup):
            """
            Critic (privileged) observation — includes simulation-only info.
            No noise applied to privileged terms.
            """

            # Re-use actor terms (with noise for consistency)
            joint_pos = ObsTerm(
                func=mdp_obs.joint_positions_normalized,
                noise=GaussianNoise(std=0.005),
            )
            joint_vel = ObsTerm(
                func=mdp_obs.joint_velocities_normalized,
                noise=GaussianNoise(std=0.04),
            )
            fingertip_pos = ObsTerm(
                func=mdp_obs.fingertip_positions_in_object_frame,
                noise=GaussianNoise(std=0.003),
            )
            target_fingertip_pos = ObsTerm(func=mdp_obs.target_fingertip_positions)
            fingertip_contact    = ObsTerm(func=mdp_obs.fingertip_contact_binary)
            last_action          = ObsTerm(func=mdp_obs.last_action)

            # Privileged: true object state
            object_pos   = ObsTerm(func=mdp_obs.object_position_world)
            object_quat  = ObsTerm(func=mdp_obs.object_orientation_world)
            object_linvel = ObsTerm(func=mdp_obs.object_linear_velocity)
            object_angvel = ObsTerm(func=mdp_obs.object_angular_velocity)

            # Privileged: full contact forces from sensor
            contact_forces = ObsTerm(func=mdp_obs.fingertip_contact_forces)

            # Privileged: DR parameters
            dr_params = ObsTerm(func=mdp_obs.domain_randomization_params)

            def __post_init__(self):
                self.enable_corruption  = True
                self.concatenate_terms  = True

        policy: PolicyObs = PolicyObs()
        critic: CriticObs = CriticObs()


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspRewardsCfg:

        # --- Positive ---
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
        fingertip_contact = RewTerm(
            func=mdp_rewards.fingertip_contact_reward,
            weight=2.0,
        )

        # --- Negative ---
        action_rate = RewTerm(
            func=mdp_rewards.action_rate_penalty,
            weight=-0.01,
        )
        object_velocity = RewTerm(
            func=mdp_rewards.object_velocity_penalty,
            weight=-0.5,
            params={"lin_thresh": 0.1, "ang_thresh": 1.0},
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
        wrist_height = RewTerm(
            func=mdp_rewards.wrist_height_penalty,
            weight=-1.0,
            params={"min_height": 0.1},
        )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspTerminationsCfg:
        time_out = DoneTerm(func=mdp_events.time_out, time_out=True)
        object_drop = DoneTerm(
            func=mdp_events.object_dropped,
            params={"min_height": 0.2},
        )


# ---------------------------------------------------------------------------
# Events (reset + domain randomization)
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspEventsCfg:
        """
        Reset events fire at every episode reset (mode="reset").
        The order matters: randomise physics first, then set poses.
        """

        # 1. Randomise object physics (mass, friction, restitution)
        randomize_object_physics = EventTerm(
            func=mdp_dr.randomize_object_physics,
            mode="reset",
            params={
                "mass_range":        (0.03, 0.30),
                "friction_range":    (0.30, 1.20),
                "restitution_range": (0.00, 0.40),
            },
        )

        # 2. Randomise robot joint dynamics (damping, armature)
        randomize_robot_physics = EventTerm(
            func=mdp_dr.randomize_robot_physics,
            mode="reset",
            params={
                "damping_range":  (0.01, 0.30),
                "armature_range": (0.001, 0.03),
            },
        )

        # 3. Randomise action delay (0–2 steps)
        randomize_action_delay = EventTerm(
            func=mdp_dr.randomize_action_delay,
            mode="reset",
            params={"max_delay": 2},
        )

        # 4. Randomise wrist position + set start/goal grasps
        reset_to_random_grasp = EventTerm(
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
        Full environment config for asymmetric-AC AnyGrasp-to-AnyGrasp.

        Actor sees 76-dim obs; critic sees 104-dim privileged obs.
        """

        scene:        AnyGraspSceneCfg        = AnyGraspSceneCfg(num_envs=MISSING, env_spacing=0.6)
        observations: AnyGraspObservationsCfg = AnyGraspObservationsCfg()
        rewards:      AnyGraspRewardsCfg      = AnyGraspRewardsCfg()
        terminations: AnyGraspTerminationsCfg = AnyGraspTerminationsCfg()
        events:       AnyGraspEventsCfg       = AnyGraspEventsCfg()

        grasp_graph_path:    str  = "data/grasp_graph.pkl"
        object_pool_specs:   list = None   # type: ignore
        wrist_randomization: dict = None   # type: ignore

        episode_length_s: float = 10.0
        action_scale:     float = 0.1
        decimation:       int   = 4

        def __post_init__(self):
            super().__post_init__()
            self.sim.dt = 1.0 / 120.0
            self.sim.render_interval = self.decimation

            if self.object_pool_specs:
                spawner = _build_object_spawner(self.object_pool_specs)
                self.scene.object = self.scene.object.replace(spawn=spawner)

            if self.wrist_randomization is None:
                self.wrist_randomization = {
                    "pos_radius_min": 0.12,
                    "pos_radius_max": 0.22,
                    "pos_height_min": 0.08,
                    "pos_height_max": 0.20,
                    "rot_std_deg":    15.0,
                }

    def set_object_pool(cfg: AnyGraspEnvCfg, pool) -> AnyGraspEnvCfg:
        """Attach an ObjectPool to the env config."""
        cfg.object_pool_specs = pool.get_isaac_lab_specs()
        spawner = _build_object_spawner(cfg.object_pool_specs)
        cfg.scene.object = cfg.scene.object.replace(spawn=spawner)
        return cfg


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

def register_anygrasp_env():
    if not _ISAACLAB_AVAILABLE:
        raise RuntimeError("Isaac Lab not installed.")

    import gymnasium as gym
    gym.register(
        id="DexGen-AnyGrasp-Allegro-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"cfg": AnyGraspEnvCfg()},
        disable_env_checker=True,
    )
    print("[AnyGraspEnv] Registered 'DexGen-AnyGrasp-Allegro-v0'")
