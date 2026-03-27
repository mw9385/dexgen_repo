"""
Domain Randomization for the AnyGrasp-to-AnyGrasp environment.

=======================================================================
  DR STRATEGY
=======================================================================

  Applied at episode RESET (per-episode randomization):
    1. object_mass          Uniform(0.03, 0.30) kg
    2. object_friction      Uniform(0.30, 1.20)  lateral friction
    3. object_restitution   Uniform(0.00, 0.40)  bounciness
    4. joint_damping        per-joint Uniform(0.01, 0.30) N·m·s/rad
    5. joint_armature       per-joint Uniform(0.001, 0.03) kg·m²
    6. action_delay         random 0–2 step delay buffer

  Applied at every STEP (continuous noise):
    7. obs_noise_joint_pos  N(0, 0.005) rad  — added in obs group
    8. obs_noise_joint_vel  N(0, 0.2) rad/s  — added in obs group
    9. obs_noise_fingertip  N(0, 0.003) m    — added in obs group

  Stored in env.extras["dr_params"] for critic observation:
    [0] mass       / 0.15  (normalised around default)
    [1] friction   / 0.75
    [2] damping_mean / 0.10
=======================================================================
"""

from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Reset-time DR events
# ---------------------------------------------------------------------------

def randomize_object_physics(
    env,
    env_ids: torch.Tensor,
    mass_range: tuple = (0.03, 0.30),
    friction_range: tuple = (0.30, 1.20),
    restitution_range: tuple = (0.00, 0.40),
):
    """
    Randomise object mass and contact properties at episode reset.

    Uses Isaac Lab's physics material and body properties API.
    Values are stored to env.extras["dr_params"] for the critic.
    """
    n   = len(env_ids)
    obj = env.scene["object"]

    # --- Mass ---
    masses = torch.empty(n, device="cpu").uniform_(*mass_range)
    env_ids_cpu = env_ids.cpu()
    obj.root_physx_view.set_masses(
        masses.unsqueeze(-1),               # (n, 1)
        indices=env_ids_cpu,
    )

    # --- Friction + restitution via physics material ---
    friction     = torch.empty(n, device="cpu").uniform_(*friction_range)
    restitution  = torch.empty(n, device="cpu").uniform_(*restitution_range)

    # Isaac Lab API: get_material_properties() returns (N_total, num_shapes, 3)
    # set_material_properties(mat, indices) requires indices
    mat_props = obj.root_physx_view.get_material_properties()
    mat_props[env_ids_cpu, :, 0] = friction.unsqueeze(-1)        # static friction
    mat_props[env_ids_cpu, :, 1] = friction.unsqueeze(-1) * 0.9  # dynamic friction (~10% less)
    mat_props[env_ids_cpu, :, 2] = restitution.unsqueeze(-1)
    obj.root_physx_view.set_material_properties(mat_props, env_ids_cpu)

    # --- Store normalised DR params for critic ---
    _update_dr_params(
        env,
        env_ids,
        mass=masses.to(env.device),
        friction=friction.to(env.device),
        damping_mean=None,
    )  # damping updated separately


def randomize_robot_physics(
    env,
    env_ids: torch.Tensor,
    damping_range: tuple = (0.01, 0.30),
    armature_range: tuple = (0.001, 0.03),
):
    """
    Randomise Allegro Hand joint damping and armature at episode reset.

    Damping:  friction-like resistance to joint motion
    Armature: effective rotational inertia at each joint
    Both affect how quickly the joints respond to torque commands.
    """
    n     = len(env_ids)
    robot = env.scene["robot"]
    hand_cfg = getattr(env.cfg, "hand", None) or {}
    n_dof = hand_cfg.get("num_dof", robot.data.joint_pos.shape[-1])

    env_ids_cpu = env_ids.cpu()
    damping  = torch.empty(n, n_dof, device="cpu").uniform_(*damping_range)
    armature = torch.empty(n, n_dof, device="cpu").uniform_(*armature_range)

    robot.root_physx_view.set_dof_dampings(damping,   indices=env_ids_cpu)
    robot.root_physx_view.set_dof_armatures(armature,  indices=env_ids_cpu)

    # Update dr_params with mean damping
    damping_mean = damping.mean(dim=-1).to(env.device)
    _update_dr_params(env, env_ids, damping_mean=damping_mean)


def randomize_action_delay(
    env,
    env_ids: torch.Tensor,
    max_delay: int = 2,
):
    """
    Randomise per-env action delay (0–max_delay steps).

    The delay buffer is initialised to zeros so the first actions are
    replayed from a neutral (zero-delta) start.
    """
    n = len(env_ids)

    # Initialise delay buffer if absent
    if "action_delay_buf" not in env.extras:
        hand_cfg = getattr(env.cfg, "hand", None) or {}
        n_dof = hand_cfg.get("num_dof", env.scene["robot"].data.joint_pos.shape[-1])
        env.extras["action_delay_buf"] = torch.zeros(
            env.num_envs, max_delay + 1, n_dof, device=env.device
        )
        env.extras["action_delay_steps"] = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    # Sample delay per env
    delays = torch.randint(0, max_delay + 1, (n,), device=env.device)
    env.extras["action_delay_steps"][env_ids] = delays

    # Reset buffer to zeros for reset envs
    env.extras["action_delay_buf"][env_ids] = 0.0


# ---------------------------------------------------------------------------
# Step-time helper: apply action delay
# ---------------------------------------------------------------------------

def apply_action_delay(env, action: torch.Tensor) -> torch.Tensor:
    """
    Apply per-env action delay and return the delayed action.

    Call this in the env's pre_physics_step before writing to actuators.
    The delay buffer stores the last (max_delay+1) actions per env.

    Returns: (N, 16) delayed action tensor
    """
    buf = env.extras.get("action_delay_buf")
    delays = env.extras.get("action_delay_steps")
    if buf is None or delays is None:
        return action

    # Shift buffer left and insert new action at the end
    buf[:, :-1, :] = buf[:, 1:, :].clone()
    buf[:, -1, :] = action

    # For each env, read from buf[i, -(delay+1)]
    N = env.num_envs
    max_d = buf.shape[1] - 1
    idx = (max_d - delays).clamp(0, max_d)   # (N,) index into buffer
    delayed = buf[torch.arange(N, device=env.device), idx, :]
    return delayed


# ---------------------------------------------------------------------------
# Observation noise (applied as Isaac Lab obs corruption)
# ---------------------------------------------------------------------------

# These noise values are referenced in anygrasp_env.py obs group config.
# Isaac Lab applies Gaussian noise N(0, std) to each obs term when
# enable_corruption = True.
OBS_NOISE = {
    "joint_pos":     0.005,   # rad
    "joint_vel":     0.2,     # rad/s  (before normalisation by 5)
    "fingertip_pos": 0.003,   # m
    "target_pos":    0.0,     # no noise on target (known exactly)
    "contact":       0.0,     # no noise on binary contact
    "last_action":   0.0,     # no noise on commanded action
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_dr_params(
    env,
    env_ids: torch.Tensor,
    mass:         Optional[torch.Tensor] = None,
    friction:     Optional[torch.Tensor] = None,
    damping_mean: Optional[torch.Tensor] = None,
):
    """Store normalised DR parameters for critic observation."""
    if "dr_params" not in env.extras:
        env.extras["dr_params"] = torch.zeros(
            env.num_envs, 3, device=env.device
        )

    if mass is not None:
        env.extras["dr_params"][env_ids, 0] = mass / 0.15
    if friction is not None:
        env.extras["dr_params"][env_ids, 1] = friction / 0.75
    if damping_mean is not None:
        env.extras["dr_params"][env_ids, 2] = damping_mean / 0.10
