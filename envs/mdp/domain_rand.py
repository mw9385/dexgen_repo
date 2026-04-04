"""
Domain Randomization for the AnyGrasp-to-AnyGrasp environment.
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
    n   = len(env_ids)
    obj = env.scene["object"]
    env_ids_cpu = env_ids.cpu()
    all_indices = torch.arange(env.num_envs, device="cpu")

    # --- Mass (Read-Modify-Write 방식으로 차원 에러 우회) ---
    full_masses = obj.root_physx_view.get_masses().clone()
    new_masses = torch.empty(n, 1, device=full_masses.device).uniform_(*mass_range)
    full_masses[env_ids_cpu] = new_masses
    obj.root_physx_view.set_masses(full_masses, indices=all_indices)

    # --- Friction + restitution ---
    mat_props = obj.root_physx_view.get_material_properties().clone()
    new_friction = torch.empty(n, device=mat_props.device).uniform_(*friction_range)
    new_restitution = torch.empty(n, device=mat_props.device).uniform_(*restitution_range)

    mat_props[env_ids_cpu, :, 0] = new_friction.unsqueeze(-1)
    mat_props[env_ids_cpu, :, 1] = new_friction.unsqueeze(-1) * 0.9
    mat_props[env_ids_cpu, :, 2] = new_restitution.unsqueeze(-1)
    
    obj.root_physx_view.set_material_properties(mat_props, indices=all_indices)

    # --- Store normalised DR params for critic ---
    _update_dr_params(
        env,
        env_ids,
        mass=new_masses.to(env.device).squeeze(-1),
        friction=new_friction.to(env.device),
        damping_mean=None,
    )


def randomize_robot_physics(
    env,
    env_ids: torch.Tensor,
    damping_range: tuple = (0.01, 0.30),
    armature_range: tuple = (0.001, 0.03),
):
    n     = len(env_ids)
    robot = env.scene["robot"]
    n_dof = robot.data.joint_pos.shape[-1]

    env_ids_cpu = env_ids.cpu()
    all_indices = torch.arange(env.num_envs, device="cpu")

    # Read-Modify-Write 방식
    full_damping = robot.root_physx_view.get_dof_dampings().clone()
    full_armature = robot.root_physx_view.get_dof_armatures().clone()

    new_damping  = torch.empty(n, n_dof, device=full_damping.device).uniform_(*damping_range)
    new_armature = torch.empty(n, n_dof, device=full_armature.device).uniform_(*armature_range)

    full_damping[env_ids_cpu] = new_damping
    full_armature[env_ids_cpu] = new_armature

    robot.root_physx_view.set_dof_dampings(full_damping, indices=all_indices)
    robot.root_physx_view.set_dof_armatures(full_armature, indices=all_indices)

    damping_mean = new_damping.mean(dim=-1).to(env.device)
    _update_dr_params(env, env_ids, damping_mean=damping_mean)


def randomize_action_delay(
    env,
    env_ids: torch.Tensor,
    max_delay: int = 2,
):
    n = len(env_ids)

    if "action_delay_buf" not in env.extras:
        # Use action-space dim (not full joint dim) so the buffer matches
        # the actual policy action tensor (wrist joints excluded for Shadow Hand).
        try:
            n_dof = env.action_manager.action.shape[-1]
        except (AttributeError, RuntimeError):
            n_dof = env.scene["robot"].data.joint_pos.shape[-1]
        env.extras["action_delay_buf"] = torch.zeros(
            env.num_envs, max_delay + 1, n_dof, device=env.device
        )
        env.extras["action_delay_steps"] = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    delays = torch.randint(0, max_delay + 1, (n,), device=env.device)
    env.extras["action_delay_steps"][env_ids] = delays
    env.extras["action_delay_buf"][env_ids] = 0.0


# ---------------------------------------------------------------------------
# Step-time helper: apply action delay
# ---------------------------------------------------------------------------

def apply_action_delay(env, action: torch.Tensor) -> torch.Tensor:
    buf = env.extras.get("action_delay_buf")
    delays = env.extras.get("action_delay_steps")
    if buf is None or delays is None:
        return action

    buf[:, :-1, :] = buf[:, 1:, :].clone()
    buf[:, -1, :] = action

    N = env.num_envs
    max_d = buf.shape[1] - 1
    idx = (max_d - delays).clamp(0, max_d)
    delayed = buf[torch.arange(N, device=env.device), idx, :]
    return delayed




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
