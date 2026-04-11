"""
Stage 1 – AnyGrasp-to-AnyGrasp Isaac Lab Environment (Sharpa Wave Hand)
=========================================================================
Implements the core RL environment from DexterityGen §3.2 with:
  - Sharpa Wave Hand: 5 fingers, 22 actuated DOF (no wrist joints)
  - Tactile sensing via ContactSensorCfg on 5 elastomer fingertip links
  - Domain Randomization (object physics, joint dynamics)
  - Object pool (cube / sphere / cylinder)
  - DeXtreme-aligned reward, obs, and termination structure

Sharpa Wave Hand: 5 fingers, 22 actuated DOF
  Thumb (5):  right_thumb_CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
  Index (4):  right_index_MCP_FE, MCP_AA, PIP, DIP
  Middle (4): right_middle_MCP_FE, MCP_AA, PIP, DIP
  Ring (4):   right_ring_MCP_FE, MCP_AA, PIP, DIP
  Pinky (5):  right_pinky_CMC, MCP_FE, MCP_AA, PIP, DIP

  Fingertip links: right_thumb_fingertip, right_index_fingertip, etc.
  Contact sensors: right_thumb_elastomer, right_index_elastomer, etc.

=======================================================================
  OBSERVATION SPLIT  (see mdp/observations.py for full details)
=======================================================================

  ACTOR = CRITIC — 309 dims (symmetric, DR params included)

  Temporal (3 frames × 86-dim per step = 258):
  ─────────────────────────────────────────────────────────────────
  joint_pos_normalized       22   (all joints, [-1,1] + noise)
  joint_vel_normalized       22   (÷5.0, clamp [-1,1])
  joint_targets              22   (current action targets)
  sensed_contacts             5   (smoothed force magnitude per tip)
  contact_positions           15  (5 fingers × 3D position)
  ─────────────────────────────────────────────────────────────────
  Per-step: 22+22+22+5+15 = 86  ×  3 frames = 258

  Non-temporal (appended once):
  ───────────────────────────────��────────────────────────────────��
  last_action                 22  (previous step's action)
  object_pose_hand             7  (current obj pos+quat in hand frame)
  object_vel_hand              6  (lin vel + ang vel × 0.2)
  target_object_pos_hand       3  (goal object position)
  target_object_quat_hand      4  (goal object quaternion)
  goal_relative_rotation       4  (quat diff: object→target)
  rotation_distance            2  (current + best rot error)
  dr_params                    3  (mass, friction, damping normalised)
  ─────────────────────────────────────────────────────────────────
  Total: 258 + 22 + 7 + 6 + 3 + 4 + 4 + 2 + 3 = 309

=======================================================================
"""
from __future__ import annotations

import math
import os
from dataclasses import MISSING
from pathlib import Path
from typing import List, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHARPA_USD = str(_REPO_ROOT / "assets" / "SharpaWave" / "right_sharpa_wave.usda")
_CYLINDER_USD = str(_REPO_ROOT / "assets" / "cylinder" / "cylinder.usd")

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
    from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import (
        ActionTermCfg as ActionTerm,
        EventTermCfg as EventTerm,
        ObservationGroupCfg as ObsGroup,
        ObservationTermCfg as ObsTerm,
        RewardTermCfg as RewTerm,
        TerminationTermCfg as DoneTerm,
    )
    from isaaclab.envs.mdp import JointPositionToLimitsActionCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
    _ISAACLAB_AVAILABLE = True

except ImportError:
    _ISAACLAB_AVAILABLE = False
    def configclass(cls):
        return cls

from .mdp import rewards as mdp_rewards
from .mdp import observations as mdp_obs
from .mdp import events as mdp_events
from .mdp import domain_rand as mdp_dr


# ── Sharpa Wave Hand default joint positions (from sharpa-rl-lab) ──
SHARPA_DEFAULT_JOINT_POS = {
    "right_thumb_CMC_FE": math.pi / 180 * 95.12771,
    "right_thumb_CMC_AA": math.pi / 180 * -3.11244,
    "right_thumb_MCP_FE": math.pi / 180 * 14.81626,
    "right_thumb_MCP_AA": math.pi / 180 * -1.03493,
    "right_thumb_IP": math.pi / 180 * 12.23986,
    "right_index_MCP_FE": math.pi / 180 * 65.21091,
    "right_index_MCP_AA": math.pi / 180 * 6.1133,
    "right_index_PIP": math.pi / 180 * 15.58495,
    "right_index_DIP": math.pi / 180 * 5.90325,
    "right_middle_MCP_FE": math.pi / 180 * 31.74149,
    "right_middle_MCP_AA": math.pi / 180 * -0.95812,
    "right_middle_PIP": math.pi / 180 * 41.88173,
    "right_middle_DIP": math.pi / 180 * 12.844,
    "right_ring_MCP_FE": math.pi / 180 * 31.72383,
    "right_ring_MCP_AA": math.pi / 180 * 9.84458,
    "right_ring_PIP": math.pi / 180 * 35.22366,
    "right_ring_DIP": math.pi / 180 * 18.02839,
    "right_pinky_CMC": math.pi / 180 * 10.9712,
    "right_pinky_MCP_FE": math.pi / 180 * 68.30895,
    "right_pinky_MCP_AA": math.pi / 180 * 7.99151,
    "right_pinky_PIP": math.pi / 180 * 5.89626,
    "right_pinky_DIP": math.pi / 180 * 5.89875,
}

SHARPA_FINGERTIP_LINKS = [
    "right_thumb_fingertip",
    "right_index_fingertip",
    "right_middle_fingertip",
    "right_ring_fingertip",
    "right_pinky_fingertip",
]

SHARPA_ELASTOMER_LINKS = [
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
]

# Object default position (within Sharpa Hand grasp, from sharpa-rl-lab)
OBJECT_DEFAULT_POS = (-0.09559, -0.00517, 0.61906)


# ---------------------------------------------------------------------------
# Helper: build MultiAssetSpawnerCfg from object pool
# ---------------------------------------------------------------------------

def _build_object_spawner(object_pool_specs: Optional[List[dict]] = None):
    if not _ISAACLAB_AVAILABLE:
        return None

    _DEFAULT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8, dynamic_friction=0.5,
        restitution=0.0, friction_combine_mode="max",
    )

    if not object_pool_specs:
        return sim_utils.CuboidCfg(
            size=(0.040, 0.040, 0.040),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1000.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0,
            ),
            physics_material=_DEFAULT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        )

    spawner_list = []
    for spec in object_pool_specs:
        shape, s = spec["shape_type"], spec["size"]
        color = tuple(spec.get("color", (0.7, 0.7, 0.7)))
        rp = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=1000.0,
            enable_gyroscopic_forces=True,
        )
        mp = sim_utils.MassPropertiesCfg(mass=spec.get("mass", 0.05))
        cp = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
        pm = _DEFAULT_MATERIAL
        vm = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        if shape == "cube":
            cfg = sim_utils.CuboidCfg(size=(s, s, s), rigid_props=rp, mass_props=mp,
                                       collision_props=cp, physics_material=pm, visual_material=vm)
        elif shape == "sphere":
            cfg = sim_utils.SphereCfg(radius=s/2, rigid_props=rp, mass_props=mp,
                                       collision_props=cp, physics_material=pm, visual_material=vm)
        elif shape == "cylinder":
            cfg = sim_utils.CylinderCfg(radius=s/2, height=s, rigid_props=rp, mass_props=mp,
                                          collision_props=cp, physics_material=pm, visual_material=vm)
        else:
            continue
        spawner_list.append(cfg)

    if len(spawner_list) == 1:
        return spawner_list[0]
    return sim_utils.MultiAssetSpawnerCfg(assets_cfg=spawner_list, random_choice=True)


# ---------------------------------------------------------------------------
# Scene configuration — Sharpa Wave Hand
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/SharpaHand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_SHARPA_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    angular_damping=0.01,
                    max_depenetration_velocity=1000.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.5),
                rot=(0.819152, 0.0, -0.5735764, 0.0),
                joint_pos=SHARPA_DEFAULT_JOINT_POS,
            ),
            actuators={
                "joints": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=None, damping=None,
                ),
            },
            soft_joint_pos_limit_factor=0.9,
        )

        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.040, 0.040, 0.040),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=1000.0,
                    enable_gyroscopic_forces=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.5,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=OBJECT_DEFAULT_POS, rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Sharpa Hand 5-finger elastomer contact sensors
        fingertip_contact_sensor_thumb: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SharpaHand/right_thumb_elastomer",
            update_period=0.0, history_length=3, debug_vis=False,
            track_contact_points=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_index: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SharpaHand/right_index_elastomer",
            update_period=0.0, history_length=3, debug_vis=False,
            track_contact_points=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_middle: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SharpaHand/right_middle_elastomer",
            update_period=0.0, history_length=3, debug_vis=False,
            track_contact_points=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_ring: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SharpaHand/right_ring_elastomer",
            update_period=0.0, history_length=3, debug_vis=False,
            track_contact_points=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        fingertip_contact_sensor_pinky: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SharpaHand/right_pinky_elastomer",
            update_period=0.0, history_length=3, debug_vis=False,
            track_contact_points=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        num_envs: int = MISSING
        env_spacing: float = 0.75
        replicate_physics: bool = False


# ---------------------------------------------------------------------------
# Observation groups — 309 dims (DeXtreme-aligned)
# 258 = 86×3 temporal (joint_pos + joint_vel + targets + tactile + contact_pos)
# + 51 = last_action(22) + obj_pose(7) + obj_vel(6) + target_pos(3)
#         + target_quat(4) + goal_rel_rot(4) + rot_dist(2) + dr_params(3)
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspObservationsCfg:
        @configclass
        class PolicyObs(ObsGroup):
            # 258-dim: sharpa 3-step temporal (joint_pos + joint_vel + targets + tactile)
            temporal = ObsTerm(func=mdp_obs.sharpa_observation_temporal)
            # Non-temporal (appended once)
            last_action = ObsTerm(func=mdp_obs.last_action)                         # 22
            object_pose = ObsTerm(func=mdp_obs.object_pose_in_hand_frame_obs)       # 7
            object_vel = ObsTerm(func=mdp_obs.object_vel_in_hand_frame)             # 6
            target_obj_pos = ObsTerm(func=mdp_obs.target_object_pos_in_hand_frame)  # 3
            target_obj_quat = ObsTerm(func=mdp_obs.target_object_quat_in_hand_frame)  # 4
            goal_rel_rot = ObsTerm(func=mdp_obs.goal_relative_rotation)             # 4
            rot_dist = ObsTerm(func=mdp_obs.rotation_distance_obs)                  # 2
            dr_params = ObsTerm(func=mdp_obs.domain_randomization_params)           # 3

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class CriticObs(ObsGroup):
            temporal = ObsTerm(func=mdp_obs.sharpa_observation_temporal)
            last_action = ObsTerm(func=mdp_obs.last_action)                         # 22
            object_pose = ObsTerm(func=mdp_obs.object_pose_in_hand_frame_obs)       # 7
            object_vel = ObsTerm(func=mdp_obs.object_vel_in_hand_frame)             # 6
            target_obj_pos = ObsTerm(func=mdp_obs.target_object_pos_in_hand_frame)  # 3
            target_obj_quat = ObsTerm(func=mdp_obs.target_object_quat_in_hand_frame)  # 4
            goal_rel_rot = ObsTerm(func=mdp_obs.goal_relative_rotation)             # 4
            rot_dist = ObsTerm(func=mdp_obs.rotation_distance_obs)                  # 2
            dr_params = ObsTerm(func=mdp_obs.domain_randomization_params)           # 3

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyObs = PolicyObs()
        critic: CriticObs = CriticObs()


# ---------------------------------------------------------------------------
# Action configuration — Sharpa Hand 22-DOF
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspActionsCfg:
        """
        Normalized joint-position actions for all 22 Sharpa Hand joints.
        No wrist joints (Sharpa Hand has no wrist DOF).
        """
        joint_pos = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["right_.*"],   # All 22 joints
            rescale_to_limits=True,
        )


# ---------------------------------------------------------------------------
# Reward configuration (DeXtreme-aligned)
# r_t = dist_rew + rot_rew + action_penalty + action_delta_penalty
#        + velocity_penalty + reach_goal_bonus
# No drop penalty — episode termination is the implicit penalty.
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspRewardsCfg:
        # Dense: goal_dist × weight (negative → penalises distance)
        distance = RewTerm(
            func=mdp_rewards.distance_reward,
            weight=-10.0,
            params={},
        )
        # Dense: 1/(|rot_dist| + eps) × weight (positive → rewards alignment)
        rotation = RewTerm(
            func=mdp_rewards.rotation_reward,
            weight=1.0,
            params={"rot_eps": 0.1},
        )
        # Dense: Σ(actions²) × weight (negative → penalises large actions)
        action_penalty = RewTerm(
            func=mdp_rewards.action_penalty,
            weight=-0.0002,
            params={},
        )
        # Dense: Σ(Δactions²) × weight (negative → penalises jerky motion)
        action_delta_penalty = RewTerm(
            func=mdp_rewards.action_delta_penalty,
            weight=-0.01,
            params={},
        )
        # Dense: Σ((dof_vel/4)²) × weight (negative → penalises high velocity)
        velocity_penalty = RewTerm(
            func=mdp_rewards.velocity_penalty,
            weight=-0.05,
            params={},
        )
        # Sparse: +250 when rot_dist < 0.4 rad (rotation only, no pos check)
        goal_bonus = RewTerm(
            func=mdp_rewards.goal_bonus,
            weight=1.0,
            params={"rot_thresh": 0.4, "bonus": 250.0},
        )


# ---------------------------------------------------------------------------
# Terminations & Events
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspTerminationsCfg:
        time_out = DoneTerm(func=mdp_events.time_out, time_out=True)
        # DeXtreme: fall_dist = 0.24 m — episode ends when object is too far from palm.
        # No reward penalty; opportunity cost of missing +250 goal bonus is the signal.
        object_drop = DoneTerm(func=mdp_events.object_dropped, params={"max_dist": 0.24})

    @configclass
    class AnyGraspEventsCfg:
        reset_to_random_grasp = EventTerm(func=mdp_events.reset_to_random_grasp, mode="reset")
        randomize_object_physics = EventTerm(func=mdp_dr.randomize_object_physics, mode="reset")
        randomize_robot_physics = EventTerm(func=mdp_dr.randomize_robot_physics, mode="reset")


# ---------------------------------------------------------------------------
# Top-level Environment Config
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    @configclass
    class AnyGraspEnvCfg(ManagerBasedRLEnvCfg):
        scene: AnyGraspSceneCfg = AnyGraspSceneCfg(num_envs=MISSING, env_spacing=0.75)
        observations: AnyGraspObservationsCfg = AnyGraspObservationsCfg()
        actions: AnyGraspActionsCfg = AnyGraspActionsCfg()
        rewards: AnyGraspRewardsCfg = AnyGraspRewardsCfg()
        terminations: AnyGraspTerminationsCfg = AnyGraspTerminationsCfg()
        events: AnyGraspEventsCfg = AnyGraspEventsCfg()

        grasp_graph_path: str = "data/grasp_graph.pkl"
        object_pool_specs: list = None   # type: ignore
        reset_refinement: dict = None    # type: ignore
        training_curriculum: dict = None  # type: ignore
        hand: dict = None                # type: ignore

        episode_length_s: float = 20.0
        action_scale: float = 1.0         # full joint range (absolute position target)
        decimation: int = 12              # 240Hz sim / 12 = 20Hz control

        def __post_init__(self):
            super().__post_init__()
            self.sim.dt = 1.0 / 240.0
            self.sim.render_interval = self.decimation
            self.sim.gravity = (0.0, 0.0, -0.05)  # reduced gravity (sharpa default)

            # --- PhysX GPU buffer sizing ---
            # In-hand manip: ~20 rigid bodies/env, ~80 contacts/env at peak.
            # Scale buffers gently with num_envs, with sane ceilings so we
            # don't OOM the GPU on larger runs. Previous setting
            # (4 * n * 1024 patches) asked for 16M patches at n=4096, which
            # is ~1GB GPU memory just for rigid patches and caused sim
            # startup to hang/crash on typical GPUs.
            _n = self.scene.num_envs if isinstance(self.scene.num_envs, int) else 4096
            self.sim.physx.gpu_max_rigid_patch_count = max(163840, 256 * _n)
            self.sim.physx.gpu_max_rigid_contact_count = max(524288, 256 * _n)
            self.sim.physx.gpu_found_lost_pairs_capacity = max(65536, 128 * _n)
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = max(262144, 256 * _n)
            self.sim.physx.gpu_total_aggregate_pairs_capacity = max(65536, 128 * _n)
            self.sim.physx.gpu_heap_capacity = max(67108864, 16384 * _n)  # 64MB min
            self.sim.physx.gpu_temp_buffer_capacity = max(16777216, 4096 * _n)  # 16MB min

            # Object pool
            _DEFAULT_POOL = [
                {"shape_type": "cube", "size": 0.040, "mass": 0.05, "color": (0.8, 0.2, 0.2)},
                {"shape_type": "cube", "size": 0.050, "mass": 0.05, "color": (1.0, 0.3, 0.3)},
                {"shape_type": "sphere", "size": 0.040, "mass": 0.05, "color": (0.2, 0.6, 0.9)},
                {"shape_type": "sphere", "size": 0.050, "mass": 0.05, "color": (0.4, 0.6, 1.0)},
                {"shape_type": "cylinder", "size": 0.040, "mass": 0.05, "color": (0.3, 0.9, 0.4)},
                {"shape_type": "cylinder", "size": 0.050, "mass": 0.05, "color": (0.2, 0.7, 0.3)},
            ]
            specs = self.object_pool_specs or _DEFAULT_POOL
            spawner = _build_object_spawner(specs)
            self.scene.object = self.scene.object.replace(spawn=spawner)

            # Fast-path: when every env spawns the same rigid asset (single
            # shape/size — the typical single-.npy case), Isaac Lab can clone
            # physics across envs instead of rebuilding them per-env.
            # This dramatically speeds up env startup (minutes → seconds).
            # MultiAssetSpawnerCfg requires replicate_physics=False, so we
            # only enable it when there is exactly one spec.
            is_single_asset = (len(specs) == 1) if specs else True
            self.scene.replicate_physics = bool(is_single_asset)

            if self.reset_refinement is None:
                self.reset_refinement = {
                    "enabled": True,
                    "iterations": 15,
                    "step_gain": 0.8,
                    "damping": 0.05,
                    "pos_threshold": 0.005,
                }

            if self.hand is None:
                self.hand = {
                    "name": "sharpa",
                    "num_fingers": 5,
                    "num_dof": 22,
                    "fingertip_links": SHARPA_FINGERTIP_LINKS,
                }
            else:
                self.hand = dict(self.hand)

            if self.training_curriculum is None:
                self.training_curriculum = {
                    "enabled": False,
                    "start_gravity": 0.05,
                    "end_gravity": 9.81,
                    "warmup_ratio": 0.10,
                    "min_orn_start": 0.10,
                    "min_orn_end": 3.14,
                }
            else:
                self.training_curriculum = dict(self.training_curriculum)

            # Sensor link mapping for Sharpa Hand
            sensor_attr_by_link = {
                "right_thumb_elastomer": "fingertip_contact_sensor_thumb",
                "right_index_elastomer": "fingertip_contact_sensor_index",
                "right_middle_elastomer": "fingertip_contact_sensor_middle",
                "right_ring_elastomer": "fingertip_contact_sensor_ring",
                "right_pinky_elastomer": "fingertip_contact_sensor_pinky",
            }
            for link_name, sensor_attr in sensor_attr_by_link.items():
                sensor_cfg = getattr(self.scene, sensor_attr)
                setattr(
                    self.scene, sensor_attr,
                    sensor_cfg.replace(
                        prim_path=f"{{ENV_REGEX_NS}}/SharpaHand/{link_name}",
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )

            self.actions.joint_pos.scale = self.action_scale


def register_anygrasp_env():
    if not _ISAACLAB_AVAILABLE:
        raise RuntimeError("Isaac Lab not installed.")

    import gymnasium as gym
    gym.register(
        id="DexGen-AnyGrasp-Sharpa-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"cfg": AnyGraspEnvCfg()},
        disable_env_checker=True,
    )
    print("[AnyGraspEnv] Registered 'DexGen-AnyGrasp-Sharpa-v0'")
