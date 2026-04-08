"""
Sharpa Hand Grasp Generation (sharpa-rl-lab 방식)
==================================================
Based on: https://github.com/sharpa-robotics/sharpa-rl-lab
Reference: sharpa_wave_grasp_env.py

Algorithm:
  1. default_joint_pos + 0.15 * random noise → hand DOF
  2. Object at default position (within hand grasp)
  3. Step physics with gravity cycling (6 directions)
  4. Validate per episode:
     - All fingertips within 0.1m of object
     - Contact force > 0.5N on >= 3 fingers
     - Object angular displacement < 30°
  5. Episode survives to max length → save state
  6. (N, 29) = 22 joints + 3 obj_pos + 4 obj_quat → .npy

Supports cube, sphere, cylinder objects.

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/gen_grasp.py \\
        --shape cube --size 0.05 --num_grasps 1000 --num_envs 4096

    /workspace/IsaacLab/isaaclab.sh -p scripts/gen_grasp.py \\
        --shape sphere --size 0.04 --num_grasps 1000 --headless
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Asset paths (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHARPA_USD = str(_REPO_ROOT / "assets" / "SharpaWave" / "right_sharpa_wave.usda")
_CYLINDER_USD = str(_REPO_ROOT / "assets" / "cylinder" / "cylinder.usd")


def parse_args():
    p = argparse.ArgumentParser(description="Sharpa Hand grasp generation")
    p.add_argument("--shape", type=str, default="cube",
                   choices=["cube", "sphere", "cylinder"])
    p.add_argument("--size", type=float, default=0.05,
                   help="Object size in metres")
    p.add_argument("--num_grasps", type=int, default=1000)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--episode_steps", type=int, default=50,
                   help="Steps per episode (at decimation=12, ~2.5s)")
    p.add_argument("--output_dir", type=str, default="cache")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Sharpa Hand 22-DOF default pre-grasp (from sharpa_wave_env_cfg.py) ──
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

SHARPA_HAND_INIT_POS = (0.0, 0.0, 0.5)
SHARPA_HAND_INIT_ROT = (0.819152, 0.0, -0.5735764, 0.0)

# Object default position (within hand grasp) from sharpa config
OBJECT_DEFAULT_POS = (-0.09559, -0.00517, 0.61906)

FINGERTIP_BODY_NAMES = [
    "right_thumb_fingertip",
    "right_index_fingertip",
    "right_middle_fingertip",
    "right_ring_fingertip",
    "right_pinky_fingertip",
]

# 6 gravity directions for cycling (from sharpa_wave_grasp_env.py)
GRAVITY_DIRECTIONS = [
    (0.0, 0.0, 9.81),
    (0.0, 0.0, -9.81),
    (0.0, 9.81, 0.0),
    (0.0, -9.81, 0.0),
    (9.81, 0.0, 0.0),
    (-9.81, 0.0, 0.0),
]


def build_env_cfg(shape: str, size: float, num_envs: int):
    """Build DirectRLEnv config for grasp generation."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.utils import configclass

    # Object spawner based on shape
    if shape == "cylinder":
        obj_spawner = sim_utils.UsdFileCfg(
            usd_path=_CYLINDER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        )
    elif shape == "cube":
        obj_spawner = sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        )
    elif shape == "sphere":
        obj_spawner = sim_utils.SphereCfg(
            radius=size / 2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 0.9),
            ),
        )

    @configclass
    class GraspGenEnvCfg(DirectRLEnvCfg):
        episode_length_s = 12.0
        action_space = 22
        observation_space = 22  # minimal, not used
        state_space = 0
        decimation = 12
        sim = SimulationCfg(
            dt=1 / 240,
            render_interval=2,
            gravity=(0.0, 0.0, -9.81),
            physx=PhysxCfg(
                solver_type=1,
                max_position_iteration_count=8,
                max_velocity_iteration_count=0,
                bounce_threshold_velocity=0.2,
                gpu_max_rigid_contact_count=2**23,
            ),
        )
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_SHARPA_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
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
                pos=SHARPA_HAND_INIT_POS,
                rot=SHARPA_HAND_INIT_ROT,
                joint_pos=SHARPA_DEFAULT_JOINT_POS,
            ),
            actuators={
                "joints": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=None, damping=None,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )
        # Contact sensors (elastomer only — 5 fingertips)
        contact_sensor = [
            ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/right_{finger}_elastomer",
                history_length=3,
                filter_prim_paths_expr=["/World/envs/env_.*/object"],
            )
            for finger in ["thumb", "index", "middle", "ring", "pinky"]
        ]
        object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=obj_spawner,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=OBJECT_DEFAULT_POS,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        scene = InteractiveSceneCfg(
            num_envs=num_envs, env_spacing=0.75,
            replicate_physics=False,
        )

    return GraspGenEnvCfg()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import carb
    import torch
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sensors import ContactSensor
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

    render = not args.headless

    # Build env
    cfg = build_env_cfg(args.shape, args.size, args.num_envs)

    # We use DirectRLEnv with a minimal wrapper
    class GraspGenEnv(DirectRLEnv):
        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)
            self.num_hand_dofs = self.hand.num_joints
            self.finger_bodies = [
                self.hand.body_names.index(name) for name in FINGERTIP_BODY_NAMES
            ]
            limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
            self.dof_lower = limits[..., 0] * 0.9  # safety margin like sharpa
            self.dof_upper = limits[..., 1] * 0.9

        def _setup_scene(self):
            self.hand = Articulation(self.cfg.robot_cfg)
            self.object = RigidObject(self.cfg.object_cfg)
            spawn_ground_plane("/World/ground", GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions()
            self.scene.articulations["robot"] = self.hand
            self.scene.rigid_objects["object"] = self.object
            self._contact_sensors = []
            for i, sensor_cfg in enumerate(self.cfg.contact_sensor):
                sensor = ContactSensor(sensor_cfg)
                self._contact_sensors.append(sensor)
                self.scene.sensors[f"contact_{i}"] = sensor
            from isaaclab.sim import SimulationCfg
            import isaaclab.sim as sim_utils
            light = sim_utils.DomeLightCfg(intensity=2000.0)
            light.func("/World/Light", light)

        def _pre_physics_step(self, actions):
            pass

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {"policy": torch.zeros(self.num_envs, 22, device=self.device)}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
                   torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

    env = GraspGenEnv(cfg)
    hand = env.hand
    obj = env.object
    device = env.device
    N = env.num_envs
    all_env_ids = torch.arange(N, device=device, dtype=torch.long)

    import omni.physics.tensors.impl.api as physx
    import isaaclab.sim as sim_utils
    physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view

    # ── Grasp generation loop ──────────────────────────────────
    saved = torch.zeros((0, 29), dtype=torch.float32, device=device)
    gravity_id = 0
    step_counter = 0
    reset_angle_diff = 30 / 180 * math.pi

    print(f"\n[GenGrasp] shape={args.shape}, size={args.size}, "
          f"num_envs={N}, target={args.num_grasps}")
    print(f"[GenGrasp] episode_steps={args.episode_steps}")

    # Initial reset
    env.reset()

    while len(saved) < args.num_grasps:
        step_counter += 1

        # Zero actions — PD controller holds current targets
        actions = torch.zeros(N, 22, device=device)
        env.step(actions)

        # Gravity cycling every 40 steps (from sharpa)
        if step_counter % 40 == 0:
            grav = GRAVITY_DIRECTIONS[gravity_id]
            physics_sim_view.set_gravity(carb.Float3(*grav))
            gravity_id = (gravity_id + 1) % len(GRAVITY_DIRECTIONS)

        # ── Validation (every episode end) ─────────────────────
        # Check which envs reached max episode length
        at_end = (env.episode_length_buf == env.max_episode_length - 1)

        if at_end.any():
            # Read current state
            fingertip_pos = hand.data.body_pos_w[:, env.finger_bodies, :]  # (N, 5, 3)
            object_pos = obj.data.root_pos_w[:, :3]  # (N, 3)
            object_rot = obj.data.root_quat_w  # (N, 4)
            hand_dof_pos = hand.data.joint_pos  # (N, 22)

            # Condition 1: all fingertips within 0.1m
            ft_dists = torch.norm(
                fingertip_pos - object_pos.unsqueeze(1), dim=-1, p=2,
            )  # (N, 5)
            cond1 = (ft_dists < 0.1).all(dim=-1)  # (N,)

            # Condition 2: contact force > 0.5N on >= 3 fingers
            forces = []
            for sensor in env._contact_sensors:
                f = sensor.data.force_matrix_w[:, 0, 0, :]  # (N, 3)
                forces.append(torch.norm(f, dim=-1, p=2))  # (N,)
            contact_forces = torch.stack(forces, dim=1)  # (N, 5)
            cond2 = (contact_forces > 0.5).sum(dim=-1) >= 3  # (N,)

            # Condition 3: object rotation < 30°
            default_rot = obj.data.default_root_state[:, 3:7]
            delta_rot = quat_mul(object_rot, quat_conjugate(default_rot))
            delta_rot = delta_rot / (torch.norm(delta_rot, dim=-1, keepdim=True) + 1e-8)
            angle = 2 * torch.acos(delta_rot[:, 0].clamp(-1, 1))
            cond3 = angle < reset_angle_diff

            success = at_end & cond1 & cond2 & cond3

            if success.any():
                # Save: (22 joints + 3 pos + 4 quat) = 29
                states = torch.cat([
                    hand_dof_pos[success],
                    object_pos[success],
                    object_rot[success],
                ], dim=1)
                saved = torch.cat([saved, states], dim=0)

        # ── Reset envs that ended ─────────────────────────────
        done_ids = torch.where(
            env.episode_length_buf >= env.max_episode_length - 1
        )[0]

        if len(done_ids) > 0:
            # Reset hand DOF: default + 0.15 * noise (from sharpa)
            rand = 2.0 * torch.rand(
                (len(done_ids), hand.num_joints), device=device,
            ) - 1.0
            dof_pos = hand.data.default_joint_pos[done_ids] + 0.15 * rand
            dof_pos = saturate(dof_pos, env.dof_lower[done_ids], env.dof_upper[done_ids])

            hand.write_joint_state_to_sim(
                dof_pos,
                torch.zeros_like(dof_pos),
                env_ids=done_ids,
            )
            hand.set_joint_position_target(dof_pos, env_ids=done_ids)

            # Reset object
            obj_state = obj.data.default_root_state[done_ids].clone()
            obj_state[:, :3] += env.scene.env_origins[done_ids]
            obj_state[:, 7:] = 0.0
            obj.write_root_pose_to_sim(obj_state[:, :7], env_ids=done_ids)
            obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=done_ids)

            env.episode_length_buf[done_ids] = 0

        # Progress log
        if step_counter % 100 == 0:
            n_success = len(saved)
            print(f"  [{time.strftime('%H:%M:%S')}] step={step_counter}, "
                  f"grasps={n_success}/{args.num_grasps}")

    # ── Save ───────────────────────────────────────────────────
    data = saved[:args.num_grasps].cpu().numpy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sharpa_grasp_{args.shape}_{int(args.size * 1000):03d}.npy"
    np.save(out_path, data)

    print(f"\n[GenGrasp] Saved {len(data)} grasps to {out_path}")
    print(f"  shape: {data.shape}")
    print(f"  format: [joint_pos(22) | obj_pos(3) | obj_quat(4)]")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
