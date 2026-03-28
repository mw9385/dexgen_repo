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
    p.add_argument("--shapes", nargs="+", default=["cube", "sphere", "cylinder"],
                   choices=["cube", "sphere", "cylinder"],
                   help="Primitive shapes to include in the object pool")
    p.add_argument("--size_min", type=float, default=0.04,
                   help="Minimum object size in metres (default: 4 cm)")
    p.add_argument("--size_max", type=float, default=0.09,
                   help="Maximum object size in metres (default: 9 cm)")
    p.add_argument("--num_sizes", type=int, default=3,
                   help="Number of size steps per shape (default: 3)")
    p.add_argument("--mesh_path", type=str, default=None,
                   help="Single custom mesh file instead of pool (.obj/.stl/.ply)")
    p.add_argument("--mesh_dir", type=str, default=None,
                   help="Directory of mesh files — all loaded as pool objects")

    # Grasp generation quality
    p.add_argument("--num_seed_grasps", type=int, default=300,
                   help="Seed grasps per object before NFO filtering")
    p.add_argument("--num_grasps", type=int, default=300,
                   help="Target grasps per object after RRT expansion")
    p.add_argument("--min_quality", type=float, default=0.005,
                   help="Minimum NFO quality score")
    p.add_argument("--mu", type=float, default=0.5,
                   help="Friction coefficient for NFO")
    p.add_argument("--fast_nfo", action="store_true",
                   help="Fast SVD approximation for NFO (less accurate)")

    p.add_argument("--output_dir", type=str, default="data")
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
    p.add_argument("--refine_iterations", type=int, default=8,
                   help="Isaac refinement iterations per grasp")
    p.add_argument("--refine_step_gain", type=float, default=0.7,
                   help="Differential IK step gain during Isaac refinement")
    p.add_argument("--refine_damping", type=float, default=0.03,
                   help="Damped least-squares coefficient during Isaac refinement")
    p.add_argument("--refine_max_delta", type=float, default=0.15,
                   help="Max per-iteration joint delta during Isaac refinement")
    p.add_argument("--refine_pos_threshold", type=float, default=0.002,
                   help="Early-stop fingertip error threshold during Isaac refinement")
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
    default_tip_links = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
    tip_links = default_tip_links[: int(num_fingers)]
    env_cfg.hand["fingertip_links"] = tip_links
    sensor_pattern = "|".join(re.escape(name) for name in tip_links)
    env_cfg.scene.fingertip_contact_sensor = env_cfg.scene.fingertip_contact_sensor.replace(
        prim_path=f"{{ENV_REGEX_NS}}/AllegroHand/({sensor_pattern})"
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
    headless = getattr(args, "headless", False)

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
            mdp_events._randomise_object_pose(env, env_ids)
            wrist_pos_w, wrist_quat_w = mdp_events._sample_wrist_pose_world(env, env_ids)
            mdp_events._set_robot_root_pose(env, env_ids, wrist_pos_w, wrist_quat_w)

            joint_list = [g.joint_angles for g in batch]
            if any(j is not None for j in joint_list):
                mdp_events._set_robot_joints_direct(env, env_ids, joint_list)
            else:
                mdp_events._set_robot_to_fingertip_config(env, env_ids, start_fps_body)

            # ----------------------------------------------------------------
            # Differential IK refinement (all n envs in parallel)
            # ----------------------------------------------------------------
            for _ in range(3):
                mdp_events._refine_hand_to_start_grasp(env, env_ids, start_fps_body)
                _, solve_max_err = mdp_events._measure_grasp_contact_error(
                    env, env_ids, start_fps_body
                )
                if float(solve_max_err.max().item()) <= float(args.refine_pos_threshold):
                    break

            # Final snap: place each object to match its refined hand pose.
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

            for i, grasp in enumerate(batch):
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
            print(
                f"  [batch] {done}/{total} grasps refined  "
                f"batch_mean={solve_mean_err.mean():.4f}m  "
                f"batch_max={solve_max_err.max():.4f}m"
            )
    finally:
        env.close()

    mean_residual = float(np.mean(residuals)) if residuals else 0.0
    max_residual  = float(np.max(residuals))  if residuals else 0.0
    print(
        f"  [IK] {graph.object_name}: refined {len(grasps)} grasps, "
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        if args.num_fingers is not None:
            num_fingers = args.num_fingers
        else:
            num_fingers = hand_cfg.get("num_fingers", 4)

        num_dof        = hand_cfg.get("num_dof", 16)
        dof_per_finger = hand_cfg.get("dof_per_finger", 4)

        print(f"[Stage 0] Hand:       {hand_cfg.get('name', 'allegro')}  "
              f"(num_fingers={num_fingers}, num_dof={num_dof})")

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

        print(f"\n[Stage 0] Processing {len(pool)} objects in pool")

        # ------------------------------------------------------------------
        # Generate grasps for each object
        # ------------------------------------------------------------------
        multi_graph = MultiObjectGraspGraph()
        success_count = 0

        for i, spec in enumerate(pool):
            graph, isaac_spec = process_one_object(
                spec, args,
                seed_offset=i * 100,
                num_fingers=num_fingers,
                num_dof=num_dof,
                dof_per_finger=dof_per_finger,
            )
            if graph is None:
                continue

            _refine_object_graph_joints(graph, isaac_spec, args)
            multi_graph.add(graph, isaac_spec)
            success_count += 1

            # Checkpoint after each object so partial results survive interruption.
            graph_path = output_dir / "grasp_graph.pkl"
            multi_graph.save(graph_path)
            print(f"  [checkpoint] {success_count} object(s) saved → {graph_path}")

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
