from __future__ import annotations

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg

try:
    from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
except ImportError:
    from isaaclab_assets import SHADOW_HAND_CFG


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))

    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(
        prim_path="/World/ShadowHand",
        spawn=SHADOW_HAND_CFG.spawn.replace(activate_contact_sensors=True),
    )
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    print("body_names")
    for name in robot.body_names:
        print(name)

    print("joint_names")
    for name in robot.joint_names:
        print(name)

    simulation_app.close()


if __name__ == "__main__":
    main()
