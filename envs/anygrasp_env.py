"""
Stage 1 – AnyGrasp-to-AnyGrasp Isaac Lab Environment
======================================================
Implements the core RL environment from DexterityGen §3.2 with:
  - Asymmetric Actor-Critic (separate policy / critic observation groups)
  - Tactile sensing via ContactSensorCfg on 5 fingertip links (Shadow Hand)
  - Domain Randomization (object physics, joint dynamics, action delay)
  - Random object pool (cube / sphere / cylinder, multiple sizes)
  - Random Shadow Hand wrist position per episode

Shadow Hand E-Series: 5 fingers, 20 actuated DOF + 4 passive = 24 total USD joints
  Actuated (20): WRJ1/WRJ0(wrist) + FFJ3/J2/J1 + MFJ3/J2/J1 + RFJ3/J2/J1 + LFJ4/J3/J2/J1 + THJ4/J3/J2/J1/J0
  Passive  ( 4): FFJ4, MFJ4, RFJ4 (spread), LFJ5 (coupling) — driven by tendons, not directly controlled
  Policy action space: 22 finger joints (excl. wrist WRJ1/WRJ0); wrist is fixed at reset pose
  Fingertip links: robot0_ffdistal, robot0_mfdistal, robot0_rfdistal, robot0_lfdistal, robot0_thdistal

=======================================================================
  OBSERVATION SPLIT  (see mdp/observations.py for full details)
=======================================================================

  ACTOR (policy) — 101 dims
  ─────────────────────────────────────────────────────────────────
  joint_pos_normalized       22   (finger joints only, excl. wrist)
  joint_vel_normalized       22   (finger joints only, excl. wrist)
  fingertip_pos_obj_frame    15   (FK in object-centric frame, 5×3)
  rel_fingertip_to_goal      15   (goal−current in obj frame, 5×3)
  fingertip_contact_binary    5   (tactile: binary contact per tip)
  last_action                22   (previous joint targets, excl. wrist)
  ─────────────────────────────────────────────────────────────────
  Total: 22+22+15+15+5+22 = 101

  CRITIC (privileged) — 132 dims
  ─────────────────────────────────────────────────────────────────
  [actor obs]               101   (incl. rel_fingertip_to_goal)
  object_pos_world            3   (true 3-D position)
  object_quat_world           4   (true orientation)
  object_lin_vel              3   (true linear velocity)
  object_ang_vel              3   (true angular velocity)
  fingertip_contact_forces   15   (full 3-D forces per tip, 5×3)
  dr_params                   3   (mass / friction / damping)
  ─────────────────────────────────────────────────────────────────
  Total: 101+3+4+3+3+15+3 = 132

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
        from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
    except ImportError:
        from isaaclab_assets import SHADOW_HAND_CFG

    # Resolve Shadow Hand USD path.
    # Use non-instanceable shadow_hand.usd for proper visual mesh rendering.
    _shadow_usd = str(SHADOW_HAND_CFG.spawn.usd_path)
    if _shadow_usd.startswith("None"):
        _S3_ROOT = (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/5.0"
        )
        _shadow_usd = f"{_S3_ROOT}/Isaac/Robots/ShadowRobot/ShadowHand/shadow_hand.usd"
    elif "instanceable" in _shadow_usd:
        _shadow_usd = _shadow_usd.replace("shadow_hand_instanceable.usd", "shadow_hand.usd")
    SHADOW_HAND_CFG = SHADOW_HAND_CFG.replace(
        spawn=SHADOW_HAND_CFG.spawn.replace(usd_path=_shadow_usd)
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
            size=(0.065, 0.065, 0.065),
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
        robot: ArticulationCfg = SHADOW_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/ShadowHand",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.45),     # wrist above object
                rot=(1.0, 0.0, 0.0, 0.0), # aligned by _align_wrist_palm_down at reset
                joint_pos={
                    "robot0_THJ4": 0.5,   # thumb rotation: natural resting pose
                    "robot0_THJ3": 0.3,
                },
            ),
            actuators={
                **SHADOW_HAND_CFG.actuators,
            },
            spawn=SHADOW_HAND_CFG.spawn.replace(
                activate_contact_sensors=True,
            ),
        )

        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.065, 0.065, 0.065),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.035), rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Shadow Hand 5-finger contact sensors (distal links in Isaac USD)
        fingertip_contact_sensor_ff: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ShadowHand/robot0_ffdistal",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_mf: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ShadowHand/robot0_mfdistal",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_rf: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ShadowHand/robot0_rfdistal",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_lf: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ShadowHand/robot0_lfdistal",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_th: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ShadowHand/robot0_thdistal",
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
            # Full observation: actor sees everything critic sees
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
            rel_fingertip_to_goal = ObsTerm(func=mdp_obs.relative_fingertip_to_goal)
            fingertip_contact     = ObsTerm(func=mdp_obs.fingertip_contact_binary)
            last_action           = ObsTerm(func=mdp_obs.last_action)

            # Previously critic-only — now shared with actor
            object_pos    = ObsTerm(func=mdp_obs.object_position_world)
            object_quat   = ObsTerm(func=mdp_obs.object_orientation_world)
            object_linvel = ObsTerm(func=mdp_obs.object_linear_velocity)
            object_angvel = ObsTerm(func=mdp_obs.object_angular_velocity)
            contact_forces = ObsTerm(func=mdp_obs.fingertip_contact_forces)
            dr_params      = ObsTerm(func=mdp_obs.domain_randomization_params)

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
            rel_fingertip_to_goal = ObsTerm(func=mdp_obs.relative_fingertip_to_goal)
            fingertip_contact     = ObsTerm(func=mdp_obs.fingertip_contact_binary)
            last_action           = ObsTerm(func=mdp_obs.last_action)

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
        """
        Normalized joint-position actions for Shadow Hand finger joints only.

        Wrist joints (WRJ1, WRJ0) are excluded from the policy action space.
        The wrist is positioned at reset and remains fixed during the episode.
        Action dim = 22 (all finger joints incl. passive spread joints at 0).
        """
        joint_pos = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=[
                "robot0_FFJ.*",   # FFJ4(passive), FFJ3, FFJ2, FFJ1  (4 joints)
                "robot0_MFJ.*",   # MFJ4(passive), MFJ3, MFJ2, MFJ1  (4 joints)
                "robot0_RFJ.*",   # RFJ4(passive), RFJ3, RFJ2, RFJ1  (4 joints)
                "robot0_LFJ.*",   # LFJ5(passive), LFJ4, LFJ3, LFJ2, LFJ1  (5 joints)
                "robot0_THJ.*",   # THJ4, THJ3, THJ2, THJ1, THJ0  (5 joints)
            ],
            rescale_to_limits=True,
        )


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspRewardsCfg:
        # ══════════════════════════════════════════════════════════════
        # DexterityGen reward structure (arXiv:2502.04307 §3.2):
        #   r = r_goal + r_style + r_reg
        #
        # All functions output [-1, 1]. Weights are positive.
        # Drop/escape handled by termination (no explicit penalty).
        # ══════════════════════════════════════════════════════════════

        # ── r_goal: fingertip tracking (func → [0, 1]) ───────────────
        # Primary dense signal. Fingertips in object frame, so correct
        # object pose is implicitly required.
        fingertip_tracking = RewTerm(
            func=mdp_rewards.fingertip_tracking_reward,
            weight=1.0,
            params={"alpha": 10.0},
        )

        # ── r_style: fingertip velocity — DISABLED for Stage 1 ──────────
        # DexterityGen §3.2: velocity penalty only in Stage 2 (fine-tuning).
        # Enabling it from the start rewards "stay still" over "reach goal".
        # fingertip_velocity = RewTerm(
        #     func=mdp_rewards.fingertip_velocity_penalty,
        #     weight=0.05,
        # )

        # ── r_reg: action/torque/work (func → [-1, 0]) ───────────────
        action_scale = RewTerm(
            func=mdp_rewards.action_scale_penalty,
            weight=0.0001,
        )
        torque = RewTerm(
            func=mdp_rewards.applied_torque_penalty,
            weight=0.0001,
        )
        mechanical_work = RewTerm(
            func=mdp_rewards.mechanical_work_penalty,
            weight=0.0001,
        )


# ---------------------------------------------------------------------------
# Terminations & Events
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspTerminationsCfg:
        time_out = DoneTerm(func=mdp_events.time_out, time_out=True)
        object_drop = DoneTerm(func=mdp_events.object_dropped, params={"min_height": 0.2})
        object_left_hand = DoneTerm(func=mdp_events.object_left_hand, params={"max_dist": 0.20})

    @configclass
    class AnyGraspEventsCfg:
        randomize_object_physics = EventTerm(
            func=mdp_dr.randomize_object_physics,
            mode="reset",
            params={"mass_range": (0.05, 0.15), "friction_range": (0.5, 1.0), "restitution_range": (0.00, 0.20)},
        )
        randomize_robot_physics = EventTerm(
            func=mdp_dr.randomize_robot_physics,
            mode="reset",
            params={"damping_range": (0.01, 0.10), "armature_range": (0.001, 0.01)},
        )
        randomize_action_delay = EventTerm(
            func=mdp_dr.randomize_action_delay,
            mode="reset",
            params={"max_delay": 1},
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
            # Increase PhysX GPU buffers for large num_envs (prevents patch buffer overflow).
            # scene.num_envs may be MISSING at config construction time, so guard it.
            _n = self.scene.num_envs if isinstance(self.scene.num_envs, int) else 4096
            self.sim.physx.gpu_max_rigid_patch_count = 4 * _n * 1024

            # Build multi-object spawner.
            # Priority: explicit object_pool_specs > default diverse pool.
            # "Default diverse pool" covers cube/sphere/cylinder at 3 sizes so
            # even without running Stage 0 the env uses varied objects.
            # Mixed pool: medium (transitional) → large (power grasp)
            # Shadow Hand palm ~8-9 cm, finger span ~16-20 cm
            # Minimum 5 cm so objects are clearly visible in simulation
            _DEFAULT_POOL = [
                {"shape_type": "cube",     "size": 0.050, "mass": 0.06, "color": (1.0, 0.3, 0.3)},
                {"shape_type": "cube",     "size": 0.065, "mass": 0.10, "color": (0.8, 0.2, 0.2)},
                {"shape_type": "cube",     "size": 0.080, "mass": 0.15, "color": (0.6, 0.1, 0.1)},
                {"shape_type": "sphere",   "size": 0.050, "mass": 0.05, "color": (0.4, 0.6, 1.0)},
                {"shape_type": "sphere",   "size": 0.065, "mass": 0.09, "color": (0.2, 0.4, 0.9)},
                {"shape_type": "sphere",   "size": 0.080, "mass": 0.13, "color": (0.1, 0.3, 0.8)},
                {"shape_type": "cylinder", "size": 0.050, "mass": 0.06, "color": (0.3, 0.9, 0.4)},
                {"shape_type": "cylinder", "size": 0.065, "mass": 0.10, "color": (0.2, 0.7, 0.3)},
                {"shape_type": "cylinder", "size": 0.080, "mass": 0.15, "color": (0.1, 0.6, 0.2)},
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
                    "object_pos_jitter_std": 0.005,         # 5 mm position jitter
                    "object_rot_jitter_deg": 5.0,           # ±5° object orientation jitter
                    "wrist_pos_jitter_std": 0.01,           # 1 cm wrist position jitter
                    "wrist_rot_std_deg": 20.0,              # ±20° wrist orientation noise
                    "align_palm_toward_object": True,       # palm faces object (not up)
                }

            if self.reset_refinement is None:
                self.reset_refinement = {
                    "enabled": False,
                }

            # Finger link subsets — Shadow Hand Isaac Lab link names.
            # Order must MATCH GraspSampler._FINGER_SUBSETS so that fingertip
            # positions in grasps align with sensor/obs order.
            #   2-finger: FF + TH  (pinch — thumb always required)
            #   3-finger: FF + MF + TH
            #   4-finger: FF + MF + RF + TH
            #   5-finger: FF + MF + RF + LF + TH  (Shadow Hand default)
            _TIP_LINK_SUBSETS = {
                2: ["robot0_ffdistal", "robot0_thdistal"],
                3: ["robot0_ffdistal", "robot0_mfdistal", "robot0_thdistal"],
                4: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_thdistal"],
                5: ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"],
            }

            if self.hand is None:
                self.hand = {
                    "name": "shadow", "num_fingers": 5, "num_dof": 24, "dof_per_finger": 4,
                    "fingertip_links": _TIP_LINK_SUBSETS[5],
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
                "robot0_ffdistal": "fingertip_contact_sensor_ff",
                "robot0_mfdistal": "fingertip_contact_sensor_mf",
                "robot0_rfdistal": "fingertip_contact_sensor_rf",
                "robot0_lfdistal": "fingertip_contact_sensor_lf",
                "robot0_thdistal": "fingertip_contact_sensor_th",
            }
            for link_name, sensor_attr in sensor_attr_by_link.items():
                sensor_cfg = getattr(self.scene, sensor_attr)
                setattr(
                    self.scene,
                    sensor_attr,
                    sensor_cfg.replace(
                        prim_path=f"{{ENV_REGEX_NS}}/ShadowHand/{link_name}",
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
        id="DexGen-AnyGrasp-Shadow-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"cfg": AnyGraspEnvCfg()},
        disable_env_checker=True,
    )
    print("[AnyGraspEnv] Registered 'DexGen-AnyGrasp-Shadow-v0'")
