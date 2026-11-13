"""Shared Franka Panda tabletop setup (robot on a 0.75 m table, 8-D embedding)."""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.common.config import load

ROBOT = load("robots/franka")   # physical constants of the robot

from eddies.envs.controllers.franka import HOME, FrankaArm, edit_panda, panda_space
from eddies.envs.framework.contacts import ContactMonitor
from eddies.envs.framework.scene import SceneBuilder, body_geoms, gid
from eddies.envs.framework.task import Task

TABLE_H = 0.75


class FrankaTaskBase(Task):
    robot = "Franka Emika Panda"
    family = "manipulation"
    physics_dt = float(ROBOT.physics_dt)
    control_dt = float(ROBOT.control_dt)
    rho_step = float(ROBOT.rho_step)          # metric step per control step (configs/robots)
    rho_game = 1.2
    failure_budget = 2
    x_range = (0.15, 0.85)
    y_range = (-0.5, 0.5)
    z_range = (TABLE_H - 0.05, TABLE_H + 0.65)

    # ----------------------------------------------------------------- scene
    def make_builder(self, name: str) -> SceneBuilder:
        sb = SceneBuilder(name, timestep=self.physics_dt)
        sb.table("table", (0.45, 0.0), TABLE_H, (0.6, 0.65))
        sb.attach_robot("franka_emika_panda/panda.xml", "franka/", pos=(0, 0, TABLE_H), edit=edit_panda)
        # robot pedestal plate (visual + collision, static)
        sb.static_box("pedestal", (0, 0, TABLE_H - 0.005), (0.12, 0.12, 0.005), rgba=(0.2, 0.2, 0.22, 1))
        return sb

    def finish_build(self, sb: SceneBuilder) -> None:
        self.model, self.data = sb.compile()
        self.arm = FrankaArm(self.model, self.data)
        self.space = panda_space(self.x_range, self.y_range, self.z_range)
        self.table_geoms = [gid(self.model, "table_top")]

    def contact_monitor(self, objects: dict[str, list[str]]) -> ContactMonitor:
        """Pairs: fingers<->each object, arm(non-finger)<->world/objects, fingers<->table."""
        m = self.model
        fingers = self.arm.finger_geoms["left"] + self.arm.finger_geoms["right"]
        arm_only = [g for g in self.arm.robot_geoms if g not in fingers and
                    (m.geom_contype[g] or m.geom_conaffinity[g])]
        groups = {"L": self.arm.finger_geoms["left"], "R": self.arm.finger_geoms["right"],
                  "fingers": fingers, "arm": arm_only, "table": self.table_geoms}
        pairs = {"fingers_table": ("fingers", "table"), "arm_table": ("arm", "table")}
        for key, bodies in objects.items():
            groups[key] = body_geoms(m, bodies)
            pairs[f"L_{key}"] = ("L", key)
            pairs[f"R_{key}"] = ("R", key)
            pairs[f"arm_{key}"] = ("arm", key)
        return ContactMonitor(m, groups, pairs)

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        self.arm.set_configuration(HOME)
        self.settle(0.6)

    def settle(self, seconds: float) -> None:
        """Pre-episode settling under gravity with the controller holding HOME."""
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0

    # --------------------------------------------------------------- command
    def cmd(self, pos, yaw: float = 0.0, aperture: float = 0.08, force: float = 0.0) -> np.ndarray:
        """Normalized command for a top-down TCP pose with world yaw ``yaw``."""
        return self.space.normalize(np.r_[np.asarray(pos, float), 0.0, 0.0, yaw, aperture, force])

    def initial_command(self) -> np.ndarray:
        return self.arm.encode(self.space)

    def apply_command(self, e) -> None:
        self.arm.command_embedding(self.space, e)

    def achieved(self, obs: dict) -> np.ndarray:
        return obs["achieved"]

    def base_obs(self) -> dict:
        arm = self.arm
        return {"t": self.time, "q": arm.q(), "dq": arm.dq(), "tcp": arm.tcp_pos(),
                "tcp_mat": arm.tcp_mat(), "aperture": arm.aperture(),
                "achieved": arm.encode(self.space)}

    def tcp_yaw(self) -> float:
        R = self.arm.tcp_mat()
        # hand x-axis projected on the table; DOWN orientation has x along world x
        return float(np.arctan2(R[1, 0], R[0, 0]))
