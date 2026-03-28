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
import re

import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import (
        ActionTermCfg as ActionTerm,
        EventTermCfg as EventTerm,
        ObservationGroupCfg as ObsGroup,
        ObservationTermCfg as ObsTerm,
        RewardTermCfg as RewTerm,
        TerminationTermCfg as DoneTerm,
    )
    # Isaac Lab v2.x Action Configuration
    from isaaclab.envs.mdp import JointPositionToLimitsActionCfg
    
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

    try:
        from isaaclab.envs.mdp import JointPositionToLimitsActionCfg
    except ImportError:
        from isaaclab.envs.mdp.actions import JointPositionToLimitsActionCfg

    try:
        from isaaclab_assets.robots.allegro_hand import ALLEGRO_HAND_CFG
    except ImportError:
        from isaaclab_assets import ALLEGRO_HAND_CFG

    # Patch USD path if the nucleus asset root resolved to None.
    # This happens in headless containers without a running Nucleus server;
    # NUCLEUS_ASSET_ROOT_DIR in isaaclab/utils/assets.py is read from a carb
    # setting that defaults to None in that environment.
    _allegro_usd = ALLEGRO_HAND_CFG.spawn.usd_path
    if str(_allegro_usd).startswith("None"):
        _S3_ROOT = (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/5.0"
        )
        ALLEGRO_HAND_CFG = ALLEGRO_HAND_CFG.replace(
            spawn=ALLEGRO_HAND_CFG.spawn.replace(
                usd_path=(
                    f"{_S3_ROOT}/Isaac/Robots/WonikRobotics/AllegroHand/"
                    "allegro_hand_instanceable.usd"
                )
            )
        )

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
        ground = AssetBaseCfg(
                    prim_path="/World/ground",
                    spawn=sim_utils.GroundPlaneCfg()
                )
        robot: ArticulationCfg = ALLEGRO_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/AllegroHand",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.6),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "thumb_joint_0": 0.28,  # within [0.279, 1.571]
                },
            ),
            actuators={
                **ALLEGRO_HAND_CFG.actuators,
            },
            spawn=ALLEGRO_HAND_CFG.spawn.replace(
                activate_contact_sensors=True,
            ),
        )

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

        fingertip_contact_sensor_index: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/AllegroHand/index_link_3",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_middle: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/AllegroHand/middle_link_3",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_ring: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/AllegroHand/ring_link_3",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_thumb: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/AllegroHand/thumb_link_3",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        num_envs:    int   = MISSING
        env_spacing: float = 1.5
        replicate_physics: bool = False


# ---------------------------------------------------------------------------
# Asymmetric Actor-Critic Observation groups
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspObservationsCfg:
        @configclass
        class PolicyObs(ObsGroup):
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

            def __post_init__(self):
                self.enable_corruption  = True
                self.concatenate_terms  = True

        @configclass
        class CriticObs(ObsGroup):
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

            object_pos   = ObsTerm(func=mdp_obs.object_position_world)
            object_quat  = ObsTerm(func=mdp_obs.object_orientation_world)
            object_linvel = ObsTerm(func=mdp_obs.object_linear_velocity)
            object_angvel = ObsTerm(func=mdp_obs.object_angular_velocity)
            contact_forces = ObsTerm(func=mdp_obs.fingertip_contact_forces)
            dr_params = ObsTerm(func=mdp_obs.domain_randomization_params)

            def __post_init__(self):
                self.enable_corruption  = True
                self.concatenate_terms  = True

        policy: PolicyObs = PolicyObs()
        critic: CriticObs = CriticObs()


# ---------------------------------------------------------------------------
# Action configuration
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspActionsCfg:
        """Normalized joint-position actions mapped to Allegro joint limits."""
        joint_pos = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            rescale_to_limits=True,
        )


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspRewardsCfg:
        # --- Goal-related (DexGen paper §3.2 goal reward) ---
        object_pose = RewTerm(
            func=mdp_rewards.object_pose_goal_reward,   # pos+rot to goal
            weight=15.0,
            params={"pos_scale": 10.0, "rot_scale": 5.0},
        )
        finger_joint_goal = RewTerm(
            func=mdp_rewards.finger_joint_goal_reward,
            weight=8.0,
            params={"scale": 5.0},
        )
        fingertip_tracking = RewTerm(
            func=mdp_rewards.fingertip_tracking_reward,
            weight=5.0,
            params={"alpha": 20.0},
        )
        grasp_success = RewTerm(
            func=mdp_rewards.grasp_success_reward,
            weight=50.0,
            params={"threshold": 0.01},
        )
        # --- Style reward (DexGen paper: fingertip velocity) ---
        fingertip_velocity = RewTerm(
            func=mdp_rewards.fingertip_velocity_penalty,
            weight=-0.5,
        )
        # --- Contact reward ---
        fingertip_contact = RewTerm(
            func=mdp_rewards.fingertip_contact_reward,
            weight=2.0,
        )
        # --- Regularization (DexGen paper: torque, work, action scale) ---
        action_scale = RewTerm(
            func=mdp_rewards.action_scale_penalty,
            weight=-0.001,
        )
        torque = RewTerm(
            func=mdp_rewards.applied_torque_penalty,
            weight=-0.002,
        )
        mechanical_work = RewTerm(
            func=mdp_rewards.mechanical_work_penalty,
            weight=-0.001,
        )
        action_rate = RewTerm(
            func=mdp_rewards.action_rate_penalty,
            weight=-0.01,
        )
        # --- Stability / safety ---
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
        object_left_hand = RewTerm(
            func=mdp_rewards.object_left_hand_penalty,
            weight=-100.0,
            params={"max_dist": 0.25},
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
# Terminations & Events
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspTerminationsCfg:
        time_out = DoneTerm(func=mdp_events.time_out, time_out=True)
        object_drop = DoneTerm(func=mdp_events.object_dropped, params={"min_height": 0.2})
        object_left_hand = DoneTerm(func=mdp_events.object_left_hand, params={"max_dist": 0.25})

    @configclass
    class AnyGraspEventsCfg:
        randomize_object_physics = EventTerm(
            func=mdp_dr.randomize_object_physics,
            mode="reset",
            params={"mass_range": (0.03, 0.30), "friction_range": (0.30, 1.20), "restitution_range": (0.00, 0.40)},
        )
        randomize_robot_physics = EventTerm(
            func=mdp_dr.randomize_robot_physics,
            mode="reset",
            params={"damping_range": (0.01, 0.30), "armature_range": (0.001, 0.03)},
        )
        randomize_action_delay = EventTerm(
            func=mdp_dr.randomize_action_delay,
            mode="reset",
            params={"max_delay": 2},
        )
        reset_to_random_grasp = EventTerm(func=mdp_events.reset_to_random_grasp, mode="reset")


# ---------------------------------------------------------------------------
# Top-level Environment Config
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspEnvCfg(ManagerBasedRLEnvCfg):
        scene:        AnyGraspSceneCfg        = AnyGraspSceneCfg(num_envs=MISSING, env_spacing=1.5)
        observations: AnyGraspObservationsCfg = AnyGraspObservationsCfg()
        actions:      AnyGraspActionsCfg      = AnyGraspActionsCfg()
        rewards:      AnyGraspRewardsCfg      = AnyGraspRewardsCfg()
        terminations: AnyGraspTerminationsCfg = AnyGraspTerminationsCfg()
        events:       AnyGraspEventsCfg       = AnyGraspEventsCfg()

        grasp_graph_path:    str  = "data/grasp_graph.pkl"
        object_pool_specs:   list = None   # type: ignore
        wrist_randomization: dict = None   # type: ignore
        reset_randomization: dict = None   # type: ignore
        reset_refinement:   dict = None   # type: ignore
        hand:                dict = None   # type: ignore

        episode_length_s: float = 10.0
        action_scale:     float = 1.0
        decimation:       int   = 4

        def __post_init__(self):
            super().__post_init__()
            self.sim.dt = 1.0 / 120.0
            self.sim.render_interval = self.decimation

            # Build multi-object spawner.
            # Priority: explicit object_pool_specs > default diverse pool.
            # "Default diverse pool" covers cube/sphere/cylinder at 3 sizes so
            # even without running Stage 0 the env uses varied objects.
            _DEFAULT_POOL = [
                {"shape_type": "cube",     "size": 0.045, "mass": 0.08, "color": (0.9, 0.2, 0.2)},
                {"shape_type": "cube",     "size": 0.060, "mass": 0.10, "color": (0.8, 0.2, 0.2)},
                {"shape_type": "cube",     "size": 0.075, "mass": 0.13, "color": (0.7, 0.2, 0.2)},
                {"shape_type": "sphere",   "size": 0.045, "mass": 0.07, "color": (0.2, 0.5, 0.9)},
                {"shape_type": "sphere",   "size": 0.060, "mass": 0.10, "color": (0.2, 0.4, 0.8)},
                {"shape_type": "sphere",   "size": 0.075, "mass": 0.13, "color": (0.2, 0.3, 0.7)},
                {"shape_type": "cylinder", "size": 0.045, "mass": 0.08, "color": (0.2, 0.8, 0.3)},
                {"shape_type": "cylinder", "size": 0.060, "mass": 0.10, "color": (0.2, 0.7, 0.3)},
                {"shape_type": "cylinder", "size": 0.075, "mass": 0.13, "color": (0.2, 0.6, 0.3)},
            ]
            specs = self.object_pool_specs or _DEFAULT_POOL
            spawner = _build_object_spawner(specs)
            self.scene.object = self.scene.object.replace(spawn=spawner)

            if self.wrist_randomization is None:
                self.wrist_randomization = {
                    "pos_radius_min": 0.12, "pos_radius_max": 0.22,
                    "pos_height_min": 0.08, "pos_height_max": 0.20,
                    "rot_std_deg": 15.0,
                }

            if self.reset_randomization is None:
                # Wrist jitter is intentionally non-zero so the policy learns to
                # be robust to small variations in the initial grasp pose.
                # Stage 0 data is valid for any wrist orientation because
                # fingertip and object poses are stored in hand-relative frames.
                self.reset_randomization = {
                    "object_pos_jitter_std": 0.005,   # 5 mm position jitter
                    "object_rot_jitter_deg": 5.0,     # ±5° object orientation jitter
                    "wrist_pos_jitter_std": 0.01,     # 1 cm wrist position jitter
                    "wrist_rot_std_deg": 20.0,        # ±20° wrist yaw randomization
                    "align_palm_up": True,
                }

            if self.reset_refinement is None:
                self.reset_refinement = {
                    "enabled": True,
                    "iterations": 3,
                    "step_gain": 0.6,
                    "damping": 0.05,
                    "max_delta": 0.2,
                    "pos_threshold": 0.005,
                }

            # Finger link subsets must MATCH GraspSampler._FINGER_SUBSETS so
            # fingertip_positions order in grasps aligns with sensor/obs order.
            #   2-finger: index + thumb   (pinch — thumb always required)
            #   3-finger: index + middle + thumb
            #   4-finger: index + middle + ring + thumb   (Allegro default)
            _TIP_LINK_SUBSETS = {
                2: ["index_link_3", "thumb_link_3"],
                3: ["index_link_3", "middle_link_3", "thumb_link_3"],
                4: ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"],
                5: ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3", "pinky_link_3"],
            }

            if self.hand is None:
                self.hand = {
                    "name": "allegro", "num_fingers": 4, "num_dof": 16, "dof_per_finger": 4,
                    "fingertip_links": _TIP_LINK_SUBSETS[4],
                }
            else:
                self.hand = dict(self.hand)

            requested_fingers = int(self.hand.get("num_fingers", 4))
            # Use pre-defined subset when possible; fall back to first-N of 4-finger list
            default_tips = _TIP_LINK_SUBSETS.get(
                requested_fingers,
                _TIP_LINK_SUBSETS[4][:requested_fingers],
            )
            tip_links = list(self.hand.get("fingertip_links", default_tips))
            # If caller provided a full list, trim to requested count
            if len(tip_links) > requested_fingers:
                tip_links = tip_links[:requested_fingers]
            self.hand["fingertip_links"] = tip_links

            sensor_attr_by_link = {
                "index_link_3": "fingertip_contact_sensor_index",
                "middle_link_3": "fingertip_contact_sensor_middle",
                "ring_link_3": "fingertip_contact_sensor_ring",
                "thumb_link_3": "fingertip_contact_sensor_thumb",
            }
            for link_name, sensor_attr in sensor_attr_by_link.items():
                sensor_cfg = getattr(self.scene, sensor_attr)
                setattr(
                    self.scene,
                    sensor_attr,
                    sensor_cfg.replace(
                        prim_path=f"{{ENV_REGEX_NS}}/AllegroHand/{link_name}",
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )
            
            # action_scale=1.0 maps policy output [-1, 1] to the full soft joint range.
            self.actions.joint_pos.scale = self.action_scale


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
