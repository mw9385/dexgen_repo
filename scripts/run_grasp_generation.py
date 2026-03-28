"""
Stage 0 – Grasp Set Generation (Multi-Object)
===============================================
Generates a diverse set of quality grasps for a *pool* of random objects
and saves a MultiObjectGraspGraph for use in Stage 1 RL training.

Each object in the pool (cube / sphere / cylinder, various sizes) gets its
own GraspGraph. At RL training time the environment randomly selects an
object + grasp pair from this combined graph every episode.

Pipeline (per object):
  1. Sample candidate grasps on the object surface (GraspSampler)
  2. Score and filter by NFO quality (NetForceOptimizer)
  3. Expand with RRT to reach target_size (RRTGraspExpander)
  4. Seed joint angles for each grasp (heuristic initialization)
  5. Refine stored joint angles inside Isaac Lab with multi-fingertip IK
  6. Save a Stage-1-ready MultiObjectGraspGraph

Usage:
    # Default: cube + sphere + cylinder, 3 sizes each
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py

    # Custom object pool
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py \\
        --shapes cube sphere \\
        --size_min 0.04 --size_max 0.09 --num_sizes 4 \\
        --num_grasps 300

    # Single custom mesh
    /workspace/IsaacLab/isaaclab.sh -p scripts/run_grasp_generation.py --mesh_path assets/mug.obj
"""

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

from grasp_generation import (
    GraspSampler, NetForceOptimizer, RRTGraspExpander,
    ObjectPool, MultiObjectGraspGraph,
)


def parse_args():
    p = argparse.ArgumentParser(description="DexGen Stage 0: Multi-Object Grasp Generation")

    # Object pool
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["cube", "sphere", "cylinder"],
                   help="Primitive shapes to include in the object pool")
    p.add_argument("--size_min", type=float, default=None,
                   help="Minimum object size in metres")
    p.add_argument("--size_max", type=float, default=None,
                   help="Maximum object size in metres")
    p.add_argument("--num_sizes", type=int, default=None,
                   help="Number of size steps per shape")
    p.add_argument("--mesh_path", type=str, default=None,
                   help="Single custom mesh file instead of pool (.obj/.stl/.ply)")
    p.add_argument("--mesh_dir", type=str, default=None,
                   help="Directory of mesh files — all loaded as pool objects")

    # Grasp generation quality
    p.add_argument("--num_seed_grasps", type=int, default=None,
                   help="Seed grasps per object before NFO filtering")
    p.add_argument("--num_grasps", type=int, default=None,
                   help="Target grasps per object after RRT expansion")
    p.add_argument("--min_quality", type=float, default=None,
                   help="Minimum NFO quality score")
    p.add_argument("--mu", type=float, default=None,
                   help="Friction coefficient for NFO")
    p.add_argument("--fast_nfo", action=argparse.BooleanOptionalAction, default=None,
                   help="Use fast SVD approximation for NFO")

    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent.parent / "configs" / "grasp_generation.yaml"),
        help="Path to grasp_generation.yaml (hand config, pool settings, etc.)",
    )
    p.add_argument(
        "--num_fingers", type=int, default=None,
        help="Number of contact points per grasp (overrides config file hand.num_fingers)",
    )
    p.add_argument("--refine_iterations", type=int, default=25,
                   help="Isaac refinement iterations per inner IK loop (default: 25)")
    p.add_argument("--refine_step_gain", type=float, default=0.6,
                   help="Differential IK step gain (default: 0.6)")
    p.add_argument("--refine_damping", type=float, default=0.01,
                   help="Damped least-squares coefficient (default: 0.01, lower = more aggressive)")
    p.add_argument("--refine_max_delta", type=float, default=0.12,
                   help="Max per-iteration joint delta (default: 0.12 rad)")
    p.add_argument("--refine_pos_threshold", type=float, default=0.001,
                   help="Per-env convergence threshold in metres (default: 1 mm)")
    p.add_argument("--refine_outer_loops", type=int, default=20,
                   help="Outer IK loop count (default: 20, was 6)")
    p.add_argument("--refine_retry_passes", type=int, default=3,
                   help="Re-init & retry passes for high-residual grasps (default: 3)")
    p.add_argument("--refine_discard_threshold", type=float, default=0.005,
                   help="Discard grasps whose final residual exceeds this (default: 5 mm)")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Heuristic IK: fingertip positions → joint angles
# ---------------------------------------------------------------------------

def _solve_ik_for_grasp(
    fingertip_positions: np.ndarray,   # (num_fingers, 3) in object frame
    num_dof: int = 16,
    dof_per_finger: int = 4,
    q_low: float = 0.0,
    q_high: float = 1.57,
) -> np.ndarray:
    """
    Heuristic IK: map fingertip distance from object centroid to joint angles.

    This is a pre-computation of the same heuristic used in events.py
    _set_robot_to_fingertip_config(). By storing it in the Grasp object,
    the RL reset is deterministic and consistent across episodes.

    Mapping:
      dist from object center → how "open" the hand is
      [0.02 m, 0.12 m] → joint scale [0.1, 0.9]
      joint_angle = q_low + scale * (q_high - q_low)

    For Allegro Hand (16 DOF, 4 fingers × 4 joints):
      - Finger order: index(0-3), middle(4-7), ring(8-11), thumb(12-15)
      - All joints in a finger get the same scale (simplified)

    Args:
        fingertip_positions: (num_fingers, 3) in object frame
        num_dof:             total robot DOF (16 for Allegro)
        dof_per_finger:      joints per finger (4 for Allegro)
        q_low:               lower joint limit (rad)
        q_high:              upper joint limit (rad)

    Returns:
        joint_angles: (num_dof,) float32 array
    """
    num_fingers = fingertip_positions.shape[0]
    joint_angles = np.zeros(num_dof, dtype=np.float32)

    # Distance from object centroid (origin in object frame) to each fingertip
    ft_dist = np.linalg.norm(fingertip_positions, axis=-1)  # (num_fingers,)

    # Map [0.02, 0.12] → scale [0.1, 0.9]
    scale = np.clip((ft_dist - 0.02) / 0.10, 0.0, 1.0) * 0.8 + 0.1  # (num_fingers,)

    for f in range(min(num_fingers, num_dof // dof_per_finger)):
        s = float(scale[f])
        start = f * dof_per_finger
        end   = start + dof_per_finger
        joint_angles[start:end] = q_low + s * (q_high - q_low)

    return joint_angles


def _attach_joint_angles_to_graph(graph, num_dof: int = 16, dof_per_finger: int = 4):
    """
    Solve heuristic IK for every grasp in a GraspGraph and store the result
    in grasp.joint_angles.

    This is called after RRT expansion so all nodes (including RRT-generated
    ones) get joint angles.
    """
    count = 0
    for grasp in graph.grasp_set.grasps:
        if grasp.joint_angles is None:
            grasp.joint_angles = _solve_ik_for_grasp(
                grasp.fingertip_positions,
                num_dof=num_dof,
                dof_per_finger=dof_per_finger,
            )
            count += 1
    print(f"  [IK] Solved heuristic IK for {count} grasps in '{graph.object_name}'")


def _make_env_for_object(spec: dict, args, num_fingers: int, num_envs: int = 1):
    from isaaclab.envs import ManagerBasedRLEnv
    from envs import AnyGraspEnvCfg, register_anygrasp_env

    register_anygrasp_env()

    env_cfg = AnyGraspEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.object_pool_specs = [spec]
    env_cfg.reset_randomization["object_pos_jitter_std"] = 0.0
    env_cfg.reset_randomization["object_rot_jitter_deg"] = 0.0
    env_cfg.reset_randomization["wrist_pos_jitter_std"] = 0.0
    env_cfg.reset_randomization["wrist_rot_std_deg"] = 0.0
    env_cfg.reset_refinement["enabled"] = True
    env_cfg.reset_refinement["iterations"] = int(args.refine_iterations)
    env_cfg.reset_refinement["step_gain"] = float(args.refine_step_gain)
    env_cfg.reset_refinement["damping"] = float(args.refine_damping)
    env_cfg.reset_refinement["max_delta"] = float(args.refine_max_delta)
    env_cfg.reset_refinement["pos_threshold"] = float(args.refine_pos_threshold)
    env_cfg.reset_debug = {"enabled": False}
    # Stage 0 refinement should not consume an existing Stage 1 grasp graph.
    # We drive the hand/object state directly inside this script.
    env_cfg.grasp_graph_path = "/tmp/stage0_refine_no_graph.pkl"
    env_cfg.hand = dict(env_cfg.hand or {})
    env_cfg.hand["num_fingers"] = int(num_fingers)
    # Must match GraspSampler._FINGER_SUBSETS so fingertip_positions order aligns
    _TIP_LINK_SUBSETS = {
        2: ["index_link_3", "thumb_link_3"],
        3: ["index_link_3", "middle_link_3", "thumb_link_3"],
        4: ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"],
    }
    tip_links = _TIP_LINK_SUBSETS.get(
        int(num_fingers),
        ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"][: int(num_fingers)],
    )
    env_cfg.hand["fingertip_links"] = tip_links
    sensor_attr_by_link = {
        "index_link_3": "fingertip_contact_sensor_index",
        "middle_link_3": "fingertip_contact_sensor_middle",
        "ring_link_3": "fingertip_contact_sensor_ring",
        "thumb_link_3": "fingertip_contact_sensor_thumb",
    }
    for link_name, sensor_attr in sensor_attr_by_link.items():
        sensor_cfg = getattr(env_cfg.scene, sensor_attr)
        setattr(
            env_cfg.scene,
            sensor_attr,
            sensor_cfg.replace(
                prim_path=f"{{ENV_REGEX_NS}}/AllegroHand/{link_name}",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            ),
        )
    if getattr(args, "headless", False):
        env_cfg.viewer = None

    env = ManagerBasedRLEnv(env_cfg)
    env.reset()
    return env


# Allegro fingertip tip-link sphere radius: body center is this far outside
# the object surface when the finger is in contact.  Applying this offset
# converts surface contact points (from GraspSampler) to fingertip body-
# center targets so the differential IK does not drive tips *into* the mesh.
_FINGERTIP_STANDOFF = 0.012   # metres


# Number of environments created for parallel grasp refinement.
# Larger values use more GPU memory; 16 works well for most GPUs.
_REFINE_BATCH_SIZE = 16


def _refine_object_graph_joints(graph, spec: dict, args) -> tuple[float, float]:
    """
    Refine joint angles and object pose for every grasp in *graph* using
    batched parallel Isaac Lab simulation.

    Key fixes vs. the previous single-env sequential version:
      1. Standoff offset: GraspSampler produces surface contact points; we
         add contact_normal * _FINGERTIP_STANDOFF so the IK target is the
         fingertip *body centre*, not the surface.  This prevents the tip
         from being driven into the mesh.
      2. Batch parallelism: _REFINE_BATCH_SIZE envs run in parallel so GPU
         utilisation is much higher than one-at-a-time.
      3. Viewer update: after each batch the viewer is advanced one render
         step so the user can watch grasps being refined in real time.
    """
    from envs.mdp import events as mdp_events
    import torch

    num_fingers = int(getattr(graph, "num_fingers", 4))
    headless    = getattr(args, "headless", False)
    discard_thresh = float(getattr(args, "refine_discard_threshold", 0.005))

    env = _make_env_for_object(
        spec, args, num_fingers=num_fingers, num_envs=_REFINE_BATCH_SIZE
    )

    grasps = graph.grasp_set.grasps
    residuals: list[float] = []

    try:
        total = len(grasps)
        for batch_start in range(0, total, _REFINE_BATCH_SIZE):
            batch = grasps[batch_start : batch_start + _REFINE_BATCH_SIZE]
            n = len(batch)
            env_ids = torch.arange(n, device=env.device, dtype=torch.long)

            # ----------------------------------------------------------------
            # Build batched tensors for this batch
            # ----------------------------------------------------------------
            fps_arr  = np.stack([g.fingertip_positions for g in batch])   # (n,F,3)
            norm_arr = np.stack([g.contact_normals      for g in batch])   # (n,F,3)

            # Surface contact points in object frame (from GraspSampler)
            start_fps_surface = torch.tensor(fps_arr,  device=env.device, dtype=torch.float32)
            contact_norms     = torch.tensor(norm_arr, device=env.device, dtype=torch.float32)

            # Fingertip body-centre targets: surface + outward_normal * standoff
            # This is what the IK and SVD alignment must reach — not the surface.
            start_fps_body = start_fps_surface + contact_norms * _FINGERTIP_STANDOFF

            # ----------------------------------------------------------------
            # Initialise each environment in the batch
            # ----------------------------------------------------------------
            from isaaclab.utils.math import quat_apply as _quat_apply

            # Place objects at their per-env world positions.
            mdp_events._randomise_object_pose(env, env_ids)

            obj   = env.scene["object"]
            robot = env.scene["robot"]
            obj_pos_w  = obj.data.root_pos_w[env_ids].clone()    # (n, 3) world
            obj_quat_w = obj.data.root_quat_w[env_ids].clone()   # (n, 4) world

            # Compute grasp approach direction: average outward normal, object→world.
            avg_norm_obj = contact_norms.mean(dim=1)              # (n, 3) obj frame
            avg_norm_obj = avg_norm_obj / (avg_norm_obj.norm(dim=-1, keepdim=True) + 1e-8)
            avg_norm_w   = _quat_apply(obj_quat_w, avg_norm_obj)  # (n, 3) world frame

            # Place wrist root along the approach direction from the object centroid.
            # Standoff is computed adaptively per-env: max fingertip-to-center radius
            # in object frame + 0.05 m approach margin, clamped to [0.08, 0.14] m.
            # A fixed 0.20 m was too large for the Allegro Hand whose longest finger
            # reaches only ~13 cm — fingertips could not physically reach contact
            # points (3-4 cm residual IK errors, all grasps discarded).
            fp_max_radius = start_fps_body.norm(dim=-1).max(dim=-1).values  # (n,)
            standoff_s0   = (fp_max_radius + 0.05).clamp(0.08, 0.14)        # (n,) m
            wrist_pos_w   = obj_pos_w + avg_norm_w * standoff_s0.unsqueeze(-1)  # (n,3)
            # Apply per-grasp uniform yaw so Stage 0 data covers the full
            # wrist_rot_std_deg=20° range used in Stage 1 training.
            # Bounded uniform is safer than Gaussian here: IK convergence is
            # sensitive to large initial yaw offsets, so capping at ±20° keeps
            # all envs in a reachable starting configuration.
            # Wrist yaw is safe to randomize because fingertip and object poses
            # are stored in hand-relative frames, so the grasp geometry is
            # frame-invariant.
            base_quat    = robot.data.default_root_state[env_ids, 3:7].clone()
            yaw_noise_rad = np.deg2rad(20.0)
            yaw_angles = (
                torch.rand(n, device=env.device) * 2.0 * yaw_noise_rad - yaw_noise_rad
            )  # Uniform(-20°, +20°) — bounded to keep hand in IK-reachable region
            # Build axis-angle around Z (yaw): [sin(θ/2)*ẑ, cos(θ/2)]
            _half = yaw_angles * 0.5
            _zero = torch.zeros(n, device=env.device)
            yaw_quats = torch.stack(
                [torch.cos(_half), _zero, _zero, torch.sin(_half)], dim=-1
            )  # (n, 4) wxyz
            # Compose: q_final = yaw_quat * base_quat
            from isaaclab.utils.math import quat_mul as _quat_mul
            wrist_quat_w = _quat_mul(yaw_quats, base_quat)

            mdp_events._set_robot_root_pose(env, env_ids, wrist_pos_w, wrist_quat_w)

            joint_list = [g.joint_angles for g in batch]
            if any(j is not None for j in joint_list):
                mdp_events._set_robot_joints_direct(env, env_ids, joint_list)
            else:
                mdp_events._set_robot_to_fingertip_config(env, env_ids, start_fps_body)

            # ----------------------------------------------------------------
            # Differential IK refinement — multi-pass with retry
            # ----------------------------------------------------------------
            # Strategy:
            #   1. Run up to refine_outer_loops × inner IK passes.
            #      Early-stop only when ALL envs are below threshold
            #      (not mean-based, which lets outliers hide).
            #   2. After each pass, identify high-residual envs and re-init
            #      them with a fresh wrist orientation drawn from a grid of
            #      yaw angles.  Keep the best result per env (lowest max-err).
            #   3. After all passes, discard grasps that still exceed the
            #      discard_threshold — they will not produce good RL episodes.

            outer_loops      = int(getattr(args, "refine_outer_loops",     20))
            retry_passes     = int(getattr(args, "refine_retry_passes",     3))
            discard_thresh   = float(getattr(args, "refine_discard_threshold", 0.005))
            pos_thresh       = float(args.refine_pos_threshold)

            # Candidate yaw offsets for retry (spread around the circle)
            _RETRY_YAWS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

            # Save ORIGINAL object poses before any IK or snapping.
            # These are used to restore the object to its canonical position
            # before each retry re-init so that the wrist standoff computation
            # is relative to the real object, not a displaced snapped position.
            orig_obj_pos_w  = obj.data.root_pos_w[env_ids].clone()    # (n, 3)
            orig_obj_quat_w = obj.data.root_quat_w[env_ids].clone()   # (n, 4)

            # Best IK result seen so far (track joints only — object pose is
            # determined by _place_object_in_hand after all passes complete).
            best_max_err   = torch.full((n,), float("inf"), device=env.device)
            best_joint_pos = robot.data.joint_pos[env_ids].clone()   # (n, D)

            for retry_idx in range(retry_passes):
                # ---- inner IK passes (no object snapping) ------------------
                # The object stays at its CANONICAL position throughout so that
                # the IK targets (start_fps_body in object frame, converted to
                # world frame via current object pose) are always correct.
                for _ in range(outer_loops):
                    mdp_events._refine_hand_to_start_grasp(env, env_ids, start_fps_body)
                    _, cur_max_err = mdp_events._measure_grasp_contact_error(
                        env, env_ids, start_fps_body
                    )
                    # Strict per-env early-stop: ALL envs must converge
                    if float(cur_max_err.max().item()) <= pos_thresh:
                        break

                # Measure without snapping (object still at canonical position)
                cur_mean_err, cur_max_err = mdp_events._measure_grasp_contact_error(
                    env, env_ids, start_fps_body
                )

                # Update best-so-far joints for each env
                improved = cur_max_err < best_max_err
                if improved.any():
                    best_max_err[improved]   = cur_max_err[improved].clone()
                    best_joint_pos[improved] = robot.data.joint_pos[env_ids][improved].clone()

                # All envs converged — no need for more passes
                if float(best_max_err.max().item()) <= pos_thresh:
                    break

                if retry_idx < retry_passes - 1:
                    # Identify still-struggling envs and re-init with a new yaw
                    high_mask = best_max_err > pos_thresh
                    if not high_mask.any():
                        break
                    retry_env_ids = env_ids[high_mask]
                    nr = int(high_mask.sum().item())

                    yaw_deg = float(_RETRY_YAWS_DEG[retry_idx % len(_RETRY_YAWS_DEG)])
                    yaw_rad = np.deg2rad(yaw_deg)
                    half = torch.full((nr,), yaw_rad / 2.0, device=env.device)
                    zero = torch.zeros(nr, device=env.device)
                    retry_yaw_q = torch.stack(
                        [torch.cos(half), zero, zero, torch.sin(half)], dim=-1
                    )
                    from isaaclab.utils.math import quat_mul as _qmul
                    base_q = robot.data.default_root_state[retry_env_ids, 3:7].clone()
                    new_quat = _qmul(retry_yaw_q, base_q)

                    # Restore ORIGINAL object pose for high-residual envs before
                    # recomputing wrist standoff.  Without this, the standoff is
                    # computed relative to the IK-displaced object position, which
                    # places the wrist far from the real target (cascade failure).
                    restore_state = obj.data.root_state_w[retry_env_ids].clone()
                    restore_state[:, :3] = orig_obj_pos_w[high_mask]
                    restore_state[:, 3:7] = orig_obj_quat_w[high_mask]
                    restore_state[:, 7:] = 0.0
                    obj.write_root_state_to_sim(restore_state, env_ids=retry_env_ids)
                    obj.update(0.0)

                    # Compute new wrist position relative to RESTORED object
                    obj_p_retry  = orig_obj_pos_w[high_mask]
                    obj_q_retry  = orig_obj_quat_w[high_mask]
                    cn_retry     = contact_norms[high_mask]
                    avg_n_obj_r  = cn_retry.mean(dim=1)
                    avg_n_obj_r  = avg_n_obj_r / (avg_n_obj_r.norm(dim=-1, keepdim=True) + 1e-8)
                    avg_n_w_r    = _quat_apply(obj_q_retry, avg_n_obj_r)
                    new_pos      = obj_p_retry + avg_n_w_r * standoff_s0[high_mask].unsqueeze(-1)

                    mdp_events._set_robot_root_pose(env, retry_env_ids, new_pos, new_quat)
                    mdp_events._set_robot_to_fingertip_config(
                        env, retry_env_ids, start_fps_body[high_mask]
                    )

            # Restore best joint positions found across all passes
            robot.write_joint_state_to_sim(
                best_joint_pos,
                torch.zeros_like(best_joint_pos),
                env_ids=env_ids,
            )
            # Restore canonical object position before the final snap
            obj_state = obj.data.root_state_w[env_ids].clone()
            obj_state[:, :3] = orig_obj_pos_w
            obj_state[:, 3:7] = orig_obj_quat_w
            obj_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
            robot.update(0.0)
            obj.update(0.0)

            # Snap object to the best-found hand configuration (single final snap)
            mdp_events._place_object_in_hand(env, env_ids, start_fps_body)

            solve_mean_err, solve_max_err = mdp_events._measure_grasp_contact_error(
                env, env_ids, start_fps_body
            )

            # Advance the viewer so the user can see the captured grasp state.
            if not headless:
                env.sim.step(render=True)

            # ----------------------------------------------------------------
            # Extract per-env results and store back into the Grasp objects
            # ----------------------------------------------------------------
            robot = env.scene["robot"]
            obj   = env.scene["object"]
            ft_ids = mdp_events._get_fingertip_body_ids_from_env(robot, env)

            ft_world   = robot.data.body_pos_w[env_ids][:, ft_ids, :].clone()  # (n,F,3)
            obj_pos    = obj.data.root_pos_w[env_ids].clone()                  # (n,3)
            obj_quat   = obj.data.root_quat_w[env_ids].clone()                 # (n,4)
            robot_pos  = robot.data.root_pos_w[env_ids].clone()                # (n,3)
            robot_quat = robot.data.root_quat_w[env_ids].clone()               # (n,4)

            discarded = 0
            for i, grasp in enumerate(batch):
                # Grasps whose final residual exceeds the discard threshold
                # are marked as invalid (joint_angles=None) so downstream
                # code skips them.  This prevents polluting the graph with
                # grasps that never converged.
                if float(solve_max_err[i].item()) > discard_thresh:
                    grasp.joint_angles = None   # signal: unusable
                    residuals.append(float(solve_max_err[i].item()))
                    discarded += 1
                    continue

                # Fingertip body-centre positions in object frame
                ft_obj_i = mdp_events._world_to_local_points(
                    ft_world[i : i + 1], obj_pos[i : i + 1], obj_quat[i : i + 1]
                )[0]
                grasp.fingertip_positions = (
                    ft_obj_i.detach().cpu().numpy().astype(np.float32)
                )
                # Refined joint angles
                grasp.joint_angles = (
                    robot.data.joint_pos[env_ids[i] : env_ids[i] + 1][0]
                    .detach().cpu().numpy().astype(np.float32)
                )
                # Object pose in hand frame
                obj_pos_hand = mdp_events._world_to_local_points(
                    obj_pos[i : i + 1].unsqueeze(1),
                    robot_pos[i : i + 1],
                    robot_quat[i : i + 1],
                ).squeeze(1)[0]
                obj_quat_hand = mdp_events._quat_multiply(
                    mdp_events._quat_conjugate(robot_quat[i : i + 1]),
                    obj_quat[i : i + 1],
                )[0]
                grasp.object_pos_hand = (
                    obj_pos_hand.detach().cpu().numpy().astype(np.float32)
                )
                grasp.object_quat_hand = (
                    obj_quat_hand.detach().cpu().numpy().astype(np.float32)
                )
                residuals.append(float(solve_mean_err[i].item()))

            done = min(batch_start + _REFINE_BATCH_SIZE, total)
            converged_n = n - discarded
            print(
                f"  [batch] {done}/{total}  "
                f"converged={converged_n}/{n}  "
                f"discarded={discarded}  "
                f"batch_mean={solve_mean_err.mean():.4f}m  "
                f"batch_max={solve_max_err.max():.4f}m"
            )
    finally:
        env.close()

    # Remove discarded grasps (joint_angles=None means IK failed to converge)
    before_count = len(grasps)
    valid_grasps = [g for g in grasps if g.joint_angles is not None]
    removed = before_count - len(valid_grasps)
    if removed:
        graph.grasp_set.grasps = valid_grasps
        # Rebuild edges — existing edge indices are now stale
        from grasp_generation.rrt_expansion import GraspSet
        old_edges = graph.edges
        valid_set = set(range(len(valid_grasps)))
        # Map old indices to new ones via the surviving grasps
        old_to_new = {}
        vi = 0
        for old_i, g in enumerate(grasps):
            if g.joint_angles is not None:
                old_to_new[old_i] = vi
                vi += 1
        new_edges = [
            (old_to_new[a], old_to_new[b])
            for a, b in old_edges
            if a in old_to_new and b in old_to_new
        ]
        graph.edges = new_edges
        print(
            f"  [filter] Removed {removed} unconverged grasps "
            f"({len(valid_grasps)} remain, {len(new_edges)} edges)"
        )

    good_residuals = [r for r in residuals if r <= discard_thresh]
    mean_residual = float(np.mean(good_residuals)) if good_residuals else 0.0
    max_residual  = float(np.max(good_residuals))  if good_residuals else 0.0
    print(
        f"  [IK] {graph.object_name}: {len(valid_grasps)} valid grasps, "
        f"{removed} discarded, "
        f"mean residual={mean_residual:.5f}m, max residual={max_residual:.5f}m"
    )
    return mean_residual, max_residual


# ---------------------------------------------------------------------------
# Per-object pipeline
# ---------------------------------------------------------------------------

def process_one_object(
    spec,
    args,
    seed_offset: int,
    num_fingers: int = 4,
    num_dof: int = 16,
    dof_per_finger: int = 4,
) -> tuple:
    """
    Run the full grasp generation pipeline for one ObjectSpec.
    Returns (GraspGraph, isaac_lab_spec) or (None, None) on failure.
    """
    from grasp_generation.grasp_sampler import GraspSampler

    print(f"\n{'='*55}")
    print(f"  Object:      {spec.name}  (shape={spec.shape_type}, size={spec.size:.3f}m)")
    print(f"  num_fingers: {num_fingers}  |  num_dof: {num_dof}")
    print(f"{'='*55}")

    # Step 1: Sample seed grasps
    sampler = GraspSampler(
        mesh=spec.mesh,
        object_name=spec.name,
        object_scale=spec.size / 0.06,   # relative to 6 cm reference
        num_candidates=args.num_seed_grasps * 20,
        num_grasps=args.num_seed_grasps,
        num_fingers=num_fingers,
        seed=args.seed + seed_offset,
    )
    seed_set = sampler.sample()

    if len(seed_set) == 0:
        print(f"  [!] No seed grasps sampled for {spec.name}, skipping.")
        return None, None

    # Step 2: NFO quality filter
    nfo = NetForceOptimizer(mu=args.mu, min_quality=args.min_quality,
                            fast_mode=args.fast_nfo)
    filtered_set = nfo.evaluate_set(seed_set, verbose=True)

    if len(filtered_set) < 10:
        print(f"  [!] Only {len(filtered_set)} grasps passed NFO for {spec.name}, skipping.")
        return None, None

    # Step 3: RRT expansion
    expander = RRTGraspExpander(
        nfo=NetForceOptimizer(min_quality=args.min_quality, fast_mode=True),
        target_size=args.num_grasps,
        # Scale delta_pos with object size so perturbations are proportional
        delta_pos=spec.size * 0.12,
        delta_max=spec.size * 0.60,
        manifold_contact_count=num_fingers,
        seed=args.seed + seed_offset,
    )
    graph = expander.expand(filtered_set)

    # Step 4: Solve heuristic IK for all grasps and store joint_angles
    _attach_joint_angles_to_graph(graph, num_dof=num_dof, dof_per_finger=dof_per_finger)

    # Verify: check that joint_angles are stored
    n_with_joints = sum(1 for g in graph.grasp_set.grasps if g.joint_angles is not None)
    print(f"  [IK] {n_with_joints}/{len(graph)} grasps have joint_angles stored")

    # Isaac Lab spawn spec for this object
    isaac_spec = {
        "name": spec.name,
        "shape_type": spec.shape_type,
        "size": spec.size,
        "mass": spec.mass,
        "color": spec.color,
    }

    return graph, isaac_spec


# ---------------------------------------------------------------------------
# Validation: verify grasp_graph.pkl is usable by RL
# ---------------------------------------------------------------------------

def validate_graph(graph_path: Path):
    """
    Load the saved graph and run basic sanity checks:
      - At least 2 nodes per object
      - At least 1 edge per object
      - joint_angles stored in all grasps
      - Nearest neighbor distance is reasonable (< delta_max)
    """
    import pickle
    from grasp_generation.rrt_expansion import MultiObjectGraspGraph, GraspGraph

    print(f"\n{'='*55}")
    print(f"  Validating {graph_path}")
    print(f"{'='*55}")

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    if isinstance(graph, GraspGraph):
        graphs = {"single": graph}
    else:
        graphs = graph.graphs

    all_ok = True
    for obj_name, g in graphs.items():
        N = len(g)
        E = g.num_edges
        n_with_joints = sum(1 for gr in g.grasp_set.grasps if gr.joint_angles is not None)

        # Nearest neighbor distance check
        all_fps = g.grasp_set.as_array()  # (N, F*3)
        nn_dists = []
        for i in range(min(N, 50)):  # sample 50 grasps
            dists = np.linalg.norm(all_fps - all_fps[i], axis=-1)
            dists[i] = np.inf
            nn_dists.append(dists.min())
        nn_mean = float(np.mean(nn_dists)) if nn_dists else 0.0

        ok = N >= 2 and E >= 1 and n_with_joints == N
        status = "✓" if ok else "✗"
        print(f"  [{status}] {obj_name}: {N} nodes, {E} edges, "
              f"{n_with_joints}/{N} with joints, "
              f"mean NN dist={nn_mean:.4f}m")

        if not ok:
            all_ok = False
            if N < 2:
                print(f"      → ERROR: need at least 2 grasps for RL (start + goal)")
            if E < 1:
                print(f"      → ERROR: no edges — increase --num_grasps or decrease delta_max")
            if n_with_joints < N:
                print(f"      → WARNING: {N - n_with_joints} grasps missing joint_angles")

    if all_ok:
        print(f"\n  ✓ Graph validation PASSED — ready for RL training")
    else:
        print(f"\n  ✗ Graph validation FAILED — fix issues above before training")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    try:
        # ------------------------------------------------------------------
        # Load config file and resolve num_fingers / num_dof
        # Priority: CLI --num_fingers > config hand.num_fingers > default 4
        # ------------------------------------------------------------------
        cfg_file = {}
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg_file = yaml.safe_load(f) or {}
            print(f"[Stage 0] Config: {cfg_path}")
        else:
            print(f"[Stage 0] Config not found at {cfg_path}, using defaults.")

        hand_cfg = cfg_file.get("hand", {})
        object_pool_cfg = cfg_file.get("object_pool", {})
        sampler_cfg = cfg_file.get("sampler", {})
        nfo_cfg = cfg_file.get("nfo", {})
        rrt_cfg = cfg_file.get("rrt", {})

        # ------------------------------------------------------------------
        # Resolve config-overridable CLI values.
        # CLI takes precedence. If omitted, fall back to YAML. If YAML also
        # omits the value, fall back to the historical hard-coded default.
        # ------------------------------------------------------------------
        args.shapes = args.shapes or object_pool_cfg.get("shapes", ["cube", "sphere", "cylinder"])
        args.size_min = float(args.size_min if args.size_min is not None else object_pool_cfg.get("size_min", 0.04))
        args.size_max = float(args.size_max if args.size_max is not None else object_pool_cfg.get("size_max", 0.09))
        args.num_sizes = int(args.num_sizes if args.num_sizes is not None else object_pool_cfg.get("num_sizes", 3))

        args.num_seed_grasps = int(
            args.num_seed_grasps if args.num_seed_grasps is not None else sampler_cfg.get("num_seed_grasps", 300)
        )
        args.num_grasps = int(
            args.num_grasps if args.num_grasps is not None else rrt_cfg.get("target_size", 300)
        )
        args.min_quality = float(
            args.min_quality if args.min_quality is not None else nfo_cfg.get("min_quality", 0.005)
        )
        args.mu = float(
            args.mu if args.mu is not None else nfo_cfg.get("mu", 0.5)
        )
        if args.fast_nfo is None:
            args.fast_nfo = bool(nfo_cfg.get("fast_mode", False))
        args.output_dir = args.output_dir or cfg_file.get("output_dir", "data")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.num_fingers is not None:
            num_fingers = args.num_fingers
        else:
            num_fingers = hand_cfg.get("num_fingers", 4)

        num_dof        = hand_cfg.get("num_dof", 16)
        dof_per_finger = hand_cfg.get("dof_per_finger", 4)

        print(f"[Stage 0] Hand:       {hand_cfg.get('name', 'allegro')}  "
              f"(num_fingers={num_fingers}, num_dof={num_dof})")
        print(f"[Stage 0] Object pool: shapes={args.shapes}, "
              f"size_range=({args.size_min:.3f}, {args.size_max:.3f}), "
              f"num_sizes={args.num_sizes}")
        print(f"[Stage 0] Sampler/NFO/RRT: seeds={args.num_seed_grasps}, "
              f"target={args.num_grasps}, min_quality={args.min_quality}, "
              f"mu={args.mu}, fast_nfo={args.fast_nfo}")

        # ------------------------------------------------------------------
        # Build object pool
        # ------------------------------------------------------------------
        if args.mesh_path:
            import trimesh
            from grasp_generation.grasp_sampler import ObjectSpec
            mesh = trimesh.load(args.mesh_path, force="mesh")
            size = float(max(mesh.bounding_box.extents))
            pool = ObjectPool([ObjectSpec(
                name=Path(args.mesh_path).stem,
                mesh=mesh,
                shape_type="custom",
                size=size,
            )])
        elif args.mesh_dir:
            pool = ObjectPool.from_mesh_dir(args.mesh_dir)
        else:
            pool = ObjectPool.from_config(
                shape_types=args.shapes,
                size_range=(args.size_min, args.size_max),
                num_sizes=args.num_sizes,
                seed=args.seed,
            )

        # Respect the configured/default finger count unless the caller
        # explicitly overrides it via CLI.
        finger_counts = [int(num_fingers)]

        print(f"\n[Stage 0] Processing {len(pool)} objects × {finger_counts} finger configs")

        # ------------------------------------------------------------------
        # Generate grasps for each object × each finger count
        # ------------------------------------------------------------------
        multi_graph = MultiObjectGraspGraph()
        success_count = 0

        for i, spec in enumerate(pool):
            for nf in finger_counts:
                # Use a unique seed offset per (object, finger_count) pair
                seed_offset = i * 100 + nf * 1000

                graph, isaac_spec = process_one_object(
                    spec, args,
                    seed_offset=seed_offset,
                    num_fingers=nf,
                    num_dof=num_dof,
                    dof_per_finger=dof_per_finger,
                )
                if graph is None:
                    continue

                # Tag the graph object name to distinguish finger configs
                graph.object_name = f"{spec.name}_f{nf}"
                isaac_spec_tagged  = dict(isaac_spec)
                isaac_spec_tagged["name"] = graph.object_name
                # Also tag num_fingers so downstream Stage 1 env can adapt
                isaac_spec_tagged["num_fingers"] = nf

                _refine_object_graph_joints(graph, isaac_spec, args)
                multi_graph.add(graph, isaac_spec_tagged)
                success_count += 1

                # Checkpoint after each (object, finger) so partial results survive.
                graph_path = output_dir / "grasp_graph.pkl"
                multi_graph.save(graph_path)
                print(f"  [checkpoint] {success_count} graph(s) saved → {graph_path}")

        if success_count == 0:
            print("\nERROR: No objects produced valid grasps. "
                  "Try --fast_nfo or lower --min_quality.")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        graph_path = output_dir / "grasp_graph.pkl"
        multi_graph.save(graph_path)

        # ------------------------------------------------------------------
        # Validate the saved graph
        # ------------------------------------------------------------------
        validate_graph(graph_path)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print(f"\n{'='*55}")
        print(f" Stage 0 Complete")
        print(f"{'='*55}")
        multi_graph.summary()
        print(f"\n  Saved: {graph_path}")
        print(f"\nNext step:")
        print(
            "  /workspace/IsaacLab/isaaclab.sh -p scripts/train_rl.py "
            f"--grasp_graph {graph_path} --num_envs 512 --headless"
        )
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
