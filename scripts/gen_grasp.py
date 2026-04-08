"""
HORA-style Grasp Generation for Shadow Hand.

Ref: In-Hand Object Rotation via Rapid Motor Adaptation (Qi et al.)
     https://github.com/HaozhiQi/hora

Algorithm:
  1. Define canonical grasp pose (fingers partially closed)
  2. For N parallel envs:
     a. q = canonical_pose + random noise
     b. Place object at palm center
     c. Step physics (PD controller holds pose)
     d. Validate: fingertips near object + object not dropped
  3. Save valid grasps as (N, 31) numpy array

Usage:
    /workspace/IsaacLab/isaaclab.sh -p scripts/gen_grasp.py \\
        --shape cube --size 0.05 --num_grasps 1000 --num_envs 64

    /workspace/IsaacLab/isaaclab.sh -p scripts/gen_grasp.py \\
        --shape sphere --size 0.05 --num_grasps 1000 --headless
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="HORA-style grasp generation")
    p.add_argument("--shape", type=str, default="cube", choices=["cube", "sphere", "cylinder"])
    p.add_argument("--size", type=float, default=0.05)
    p.add_argument("--num_grasps", type=int, default=1000)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--hold_steps", type=int, default=50)
    p.add_argument("--noise_std", type=float, default=0.3)
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--physics_gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Shadow Hand 24-DOF canonical grasp pose
# Fingers moderately closed, thumb in opposition
SHADOW_CANONICAL_POSE = np.zeros(24, dtype=np.float32)
# Wrist: 0
SHADOW_CANONICAL_POSE[0] = 0.0    # WRJ1
SHADOW_CANONICAL_POSE[1] = 0.0    # WRJ0
# Forefinger
SHADOW_CANONICAL_POSE[3] = 0.4    # FFJ3 (MCP)
SHADOW_CANONICAL_POSE[4] = 0.8    # FFJ2 (PIP)
SHADOW_CANONICAL_POSE[5] = 0.4    # FFJ1 (DIP)
# Middle finger
SHADOW_CANONICAL_POSE[7] = 0.4    # MFJ3
SHADOW_CANONICAL_POSE[8] = 0.8    # MFJ2
SHADOW_CANONICAL_POSE[9] = 0.4    # MFJ1
# Ring finger
SHADOW_CANONICAL_POSE[11] = 0.4   # RFJ3
SHADOW_CANONICAL_POSE[12] = 0.8   # RFJ2
SHADOW_CANONICAL_POSE[13] = 0.4   # RFJ1
# Little finger
SHADOW_CANONICAL_POSE[16] = 0.4   # LFJ3
SHADOW_CANONICAL_POSE[17] = 0.8   # LFJ2
SHADOW_CANONICAL_POSE[18] = 0.4   # LFJ1
# Thumb (opposition)
SHADOW_CANONICAL_POSE[19] = 1.0   # THJ4
SHADOW_CANONICAL_POSE[20] = 0.4   # THJ3
SHADOW_CANONICAL_POSE[21] = 0.3   # THJ2
SHADOW_CANONICAL_POSE[22] = 0.5   # THJ1
SHADOW_CANONICAL_POSE[23] = 0.0   # THJ0


def build_env(shape, size, num_envs):
    import isaaclab.sim as sim_utils
    from envs.anygrasp_env import AnyGraspEnvCfg

    cfg = AnyGraspEnvCfg()
    cfg.scene.num_envs = num_envs

    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=0.8,
        restitution=0.0, friction_combine_mode="max",
    )
    rp = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, max_depenetration_velocity=0.1,
        enable_gyroscopic_forces=True,
    )
    mp = sim_utils.MassPropertiesCfg(mass=0.05)
    cp = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
    color = {"cube": (0.8, 0.2, 0.2), "sphere": (0.2, 0.6, 0.9),
             "cylinder": (0.3, 0.8, 0.3)}[shape]
    vis = sim_utils.PreviewSurfaceCfg(diffuse_color=color)

    if shape == "cube":
        cfg.scene.object.spawn = sim_utils.CuboidCfg(
            size=(size, size, size), rigid_props=rp, mass_props=mp,
            collision_props=cp, physics_material=mat, visual_material=vis)
    elif shape == "sphere":
        cfg.scene.object.spawn = sim_utils.SphereCfg(
            radius=size / 2, rigid_props=rp, mass_props=mp,
            collision_props=cp, physics_material=mat, visual_material=vis)
    elif shape == "cylinder":
        cfg.scene.object.spawn = sim_utils.CylinderCfg(
            radius=size / 2, height=size, rigid_props=rp, mass_props=mp,
            collision_props=cp, physics_material=mat, visual_material=vis)

    return cfg


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from envs.mdp.sim_utils import (
        get_fingertip_body_ids_from_env,
        get_palm_body_id_from_env,
        set_robot_root_pose,
        get_local_palm_normal,
    )
    from envs.mdp.math_utils import quat_from_two_vectors, quat_multiply
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    render = not args.headless
    rng = np.random.default_rng(args.seed)

    # Build env
    env_cfg = build_env(args.shape, args.size, args.num_envs)
    env = ManagerBasedRLEnv(env_cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = env.device
    N = args.num_envs

    env_ids = torch.arange(N, device=device, dtype=torch.long)
    ft_ids = get_fingertip_body_ids_from_env(robot, env)
    palm_id = get_palm_body_id_from_env(robot, env)

    # Joint limits
    q_low = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    q_high = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    q_range = q_high - q_low
    num_dof = len(q_low)

    # Warm up
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)

    # ── Compute palm-up wrist pose ─────────────────────────────
    root_state = robot.data.default_root_state[env_ids, :7].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    set_robot_root_pose(env, env_ids, root_state[:, :3], root_state[:, 3:7])

    # Set canonical pose for palm normal computation
    canonical_t = torch.tensor(SHADOW_CANONICAL_POSE[:num_dof],
                               device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1)
    robot.write_joint_state_to_sim(canonical_t, torch.zeros_like(canonical_t), env_ids=env_ids)
    robot.set_joint_position_target(canonical_t, env_ids=env_ids)
    env.sim.step(render=render)
    env.scene.update(dt=env.physics_dt)

    wrist_pos = robot.data.root_pos_w[env_ids].clone()
    wrist_quat = robot.data.root_quat_w[env_ids].clone()
    palm_n = get_local_palm_normal(robot, env).unsqueeze(0).expand(N, 3)
    palm_w = quat_apply(wrist_quat, palm_n)
    target_up = torch.tensor([0, 0, 1.0], device=device).expand(N, 3)
    correction = quat_from_two_vectors(palm_w, target_up)

    ft_w = robot.data.body_pos_w[env_ids][:, ft_ids, :]
    pivot = ft_w.mean(dim=1)
    palmup_quat = quat_multiply(correction, wrist_quat)
    palmup_quat = palmup_quat / (torch.norm(palmup_quat, dim=-1, keepdim=True) + 1e-8)
    palmup_pos = quat_apply(correction, wrist_pos - pivot) + pivot

    print(f"Palm-up computed. Starting grasp generation...")
    print(f"  shape={args.shape}, size={args.size}, num_envs={N}, "
          f"hold_steps={args.hold_steps}, noise_std={args.noise_std}")

    # ── Main loop ──────────────────────────────────────────────
    saved = []  # list of (31,) arrays: 24 joints + 3 pos + 4 quat
    total_tested = 0
    round_idx = 0

    while len(saved) < args.num_grasps:
        round_idx += 1

        # 1. Palm-up wrist
        set_robot_root_pose(env, env_ids, palmup_pos, palmup_quat)

        # 2. Canonical pose + noise
        noise = rng.normal(0, args.noise_std, size=(N, num_dof)).astype(np.float32)
        noise[:, :2] = 0  # wrist fixed
        q_np = SHADOW_CANONICAL_POSE[:num_dof] + noise * q_range
        q_np = np.clip(q_np, q_low, q_high)
        q_t = torch.tensor(q_np, device=device, dtype=torch.float32)

        robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), env_ids=env_ids)
        robot.set_joint_position_target(q_t, env_ids=env_ids)

        # 3. Place object at palm center
        env.sim.step(render=render)
        env.scene.update(dt=env.physics_dt)

        palm_pos = robot.data.body_pos_w[env_ids, palm_id, :].clone()
        obj_pos = palm_pos.clone()
        obj_pos[:, 2] += args.size * 0.5  # slightly above palm

        obj_state = obj.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos
        obj_state[:, 3:7] = torch.tensor([1, 0, 0, 0], device=device,
                                          dtype=torch.float32).expand(N, -1)
        obj_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
        obj.update(0.0)

        # 4. Step physics — PD controller holds pose
        for _ in range(args.hold_steps):
            robot.set_joint_position_target(q_t, env_ids=env_ids)
            env.sim.step(render=render)
            env.scene.update(dt=env.physics_dt)

        # 5. Validate
        ft_pos = robot.data.body_pos_w[env_ids][:, ft_ids, :]  # (N, 5, 3)
        obj_pos_after = obj.data.root_pos_w[env_ids]            # (N, 3)
        obj_quat_after = obj.data.root_quat_w[env_ids]          # (N, 4)
        obj_vel = torch.norm(obj.data.root_lin_vel_w[env_ids], dim=-1)  # (N,)

        # Fingertip distance to object center
        ft_to_obj = ft_pos - obj_pos_after.unsqueeze(1)  # (N, 5, 3)
        ft_dists = torch.norm(ft_to_obj, dim=-1)          # (N, 5)

        # Check criteria
        all_near = (ft_dists < 0.08).all(dim=1)           # all within 8cm
        n_contact = (ft_dists < 0.02).sum(dim=1)           # within 2cm = contact
        enough_contact = n_contact >= 2
        not_dropped = obj_pos_after[:, 2] > 0.3
        low_vel = obj_vel < 0.1

        valid = all_near & enough_contact & not_dropped & low_vel

        # 6. Save valid grasps
        actual_q = robot.data.joint_pos[env_ids]  # (N, 24)

        # Object pose in hand frame
        rp = robot.data.root_pos_w[env_ids]
        rq = robot.data.root_quat_w[env_ids]

        for i in range(N):
            if valid[i]:
                q_i = actual_q[i].cpu().numpy()
                # Object pos in hand frame
                rel = obj_pos_after[i] - rp[i]
                pos_hand = quat_apply_inverse(
                    rq[i].unsqueeze(0), rel.unsqueeze(0),
                )[0].cpu().numpy()
                quat_hand = quat_multiply(
                    torch.tensor([[rq[i][0], -rq[i][1], -rq[i][2], -rq[i][3]]],
                                 device=device),  # conjugate
                    obj_quat_after[i].unsqueeze(0),
                )[0].cpu().numpy()

                # (31,) = 24 joints + 3 pos + 4 quat
                state = np.concatenate([q_i, pos_hand, quat_hand])
                saved.append(state)

        total_tested += N
        n_valid = int(valid.sum().item())
        rate = len(saved) / total_tested * 100

        print(f"  round {round_idx}: +{n_valid} valid, "
              f"total={len(saved)}/{args.num_grasps} "
              f"({rate:.1f}% acceptance, tested={total_tested})")

    # ── Save ───────────────────────────────────────────────────
    saved = saved[:args.num_grasps]
    data = np.stack(saved)  # (num_grasps, 31)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grasp_{args.shape}_{int(args.size * 1000):03d}.npy"
    np.save(out_path, data)

    print(f"\nSaved {len(data)} grasps to {out_path}")
    print(f"  shape: {data.shape}")
    print(f"  format: [joint_angles(24) | obj_pos_hand(3) | obj_quat_hand(4)]")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
