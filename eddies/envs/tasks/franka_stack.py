"""Task 4 — Franka cube stacking with two differently coloured cubes.

Pick the grey cube ("red" in the code) from the table and stack it on the black
cube ("blue"), release it and retreat; colours match the real cubes.  Physics: 5 cm wooden cubes with randomized mass/friction resting on
a laminate tabletop under gravity; the grasp holds only through pad friction.

LTLf:  G !bad & F(grasped & F(stacked & released))
"""
from __future__ import annotations

import numpy as np

from eddies.envs.framework.perception import PoseEstimator, tilt_of_quat, wrap_to, yaw_of_quat
from eddies.envs.framework.scene import yaw_quat
from eddies.envs.framework.task import OK, Skill, SkillSpec, OutcomeSpec, word
from eddies.envs.tasks.franka_base import TABLE_H, FrankaTaskBase

HALF = 0.025                      # cube half size [m]
GRASP_DZ = 0.006                  # TCP above cube centre when grasping
RED_NOM = np.array([0.50, -0.12, TABLE_H + HALF])
BLUE_NOM = np.array([0.50, 0.12, TABLE_H + HALF])


class FrankaCubeStackTask(FrankaTaskBase):
    name = "franka_cube_stack"
    description = "Franka Panda stacks a grey 5 cm cube onto a black 5 cm cube (friction grasp)."
    horizon = 45.0
    success_hold = 1.0
    formula = "G !bad & F(grasped & F(stacked & released))"
    terminal_cells = ("done",)
    main_camera = "main"
    detail_camera = "detail"

    def __init__(self, seed: int, **variant):
        super().__init__(seed, **variant)
        rng = self.rng
        self.red0 = RED_NOM + np.r_[rng.uniform(-0.06, 0.06), rng.uniform(-0.05, 0.05), 0.0]
        self.blue0 = BLUE_NOM + np.r_[rng.uniform(-0.06, 0.06), rng.uniform(-0.05, 0.05), 0.0]
        self.red_yaw0, self.blue_yaw0 = rng.uniform(-0.6, 0.6, 2)
        self.red_mass, self.blue_mass = rng.uniform(0.06, 0.14), rng.uniform(0.08, 0.16)
        self.mu = rng.uniform(0.45, 0.9)
        sb = self.make_builder(self.name)
        # colours of the real cubes (slate-grey cube picked, black cube below); names kept as red/blue in the code
        for name, p, yaw, mass, rgba in (("red_cube", self.red0, self.red_yaw0, self.red_mass, (0.36, 0.47, 0.62, 1)),
                                         ("blue_cube", self.blue0, self.blue_yaw0, self.blue_mass, (0.05, 0.055, 0.065, 1))):
            body = sb.free_box(name, p + [0, 0, 0.0005], (HALF,) * 3, mass=mass, rgba=rgba,
                               quat=yaw_quat(yaw), friction=(self.mu, 0.005, 1e-4))
            body.geoms[0].priority = 1
        sb.camera("main", (1.55, -1.05, TABLE_H + 1.0), (0.38, 0.0, TABLE_H + 0.22), fovy=45)
        c = self.blue0
        sb.camera("detail", c + np.array([0.28, -0.30, 0.20]), c + np.array([0.0, -0.04, 0.05]), fovy=38)
        sb.camera("top", (0.5, 0.0, TABLE_H + 1.3), (0.5, 0.0, TABLE_H), fovy=50)
        self.finish_build(sb)
        self.cm = self.contact_monitor({"red": ["red_cube"], "blue": ["blue_cube"]})
        self.est_red = PoseEstimator(rng)
        self.est_blue = PoseEstimator(rng)
        self._contacts = {}
        self._build_skills()
        # stress-test schedule (drawn even when unused so nominal RNG streams match)
        self._push_dir = rng.uniform(-np.pi, np.pi)
        self._push_delay = rng.uniform(0.0, 0.3)
        self._drop_delay = rng.uniform(0.3, 0.8)
        self._pert = {"push": None, "dropout": None}
        self._grip_range = self.model.actuator_forcerange[self.arm.act_grip].copy()

    # ---------------------------------------------------------------- state
    def _cube(self, name):
        b = self.data.body(name)
        return b.xpos.copy(), b.xquat.copy(), self.data.cvel[b.id][3:].copy()

    def _forces(self):
        if self._contacts.get("t") != self.data.time:
            self._contacts = self.cm.measure(self.data)
            self._contacts["t"] = self.data.time
        return self._contacts

    def _geometry(self):
        red, rq, rv = self._cube("red_cube")
        blue, bq, bv = self._cube("blue_cube")
        f = self._forces()
        grasp = f["L_red"] > 1.0 and f["R_red"] > 1.0
        on_blue = (np.linalg.norm(red[:2] - blue[:2]) < 0.02 and 0.044 < red[2] - blue[2] < 0.056
                   and blue[2] < TABLE_H + HALF + 0.005)
        return {"red": red, "rq": rq, "rv": rv, "blue": blue, "bq": bq, "bv": bv, "grasp": grasp,
                "on_blue": on_blue, "f": f}

    def observe(self) -> dict:
        obs = self.base_obs()
        g = self._geometry()
        tcp = obs["tcp"]
        rp, ry, rn = self.est_red.estimate(g["red"], yaw_of_quat(g["rq"]), tcp)
        bp, by, bn = self.est_blue.estimate(g["blue"], yaw_of_quat(g["bq"]), tcp)
        f = g["f"]
        obs.update({"red_est": rp, "red_yaw": wrap_to(ry, np.pi / 2), "blue_est": bp,
                    "blue_yaw": wrap_to(by, np.pi / 2), "red_near_est": rn,
                    "contact_L": f["L_red"], "contact_R": f["R_red"], "grasp": g["grasp"],
                    "red_true": g["red"], "blue_true": g["blue"], "on_blue": g["on_blue"],
                    "red_speed": float(np.linalg.norm(g["rv"]))})
        return obs

    def labels(self) -> set:
        g = self._geometry()
        f = g["f"]
        out = set()
        if g["grasp"] and g["red"][2] > TABLE_H + HALF + 0.008:
            out.add("grasped")
        if g["on_blue"] and np.linalg.norm(g["rv"]) < 0.03:
            out.add("stacked")
        if f["L_red"] + f["R_red"] < 0.3 and self.arm.aperture() > 0.055:
            out.add("released")
        fell = min(g["red"][2], g["blue"][2]) < TABLE_H - 0.02
        smash = f["arm_table"] > 25.0 or f["fingers_table"] > 60.0 or f["arm_red"] > 40.0 or f["arm_blue"] > 40.0
        if fell or smash or tilt_of_quat(g["bq"]) > np.radians(35):
            out.add("bad")
        return out

    def task_success(self) -> bool:
        g = self._geometry()
        tcp = self.arm.tcp_pos()
        return bool(g["on_blue"] and not g["grasp"] and np.linalg.norm(g["rv"]) < 0.03
                    and tcp[2] > g["red"][2] + HALF + 0.05 and g["f"]["L_red"] + g["f"]["R_red"] < 0.3)

    def cell(self, obs: dict, labels: frozenset) -> str:
        red, blue, tcp = obs["red_true"], obs["blue_true"], obs["tcp"]
        grasp, ap = obs["grasp"], obs["aperture"]
        if red[2] < TABLE_H - 0.02:
            return "red_lost"
        if obs["on_blue"] and not grasp:
            return "done" if tcp[2] > red[2] + HALF + 0.05 else "released_on_blue"
        if obs["on_blue"] and grasp:
            return "placed_held"
        if grasp:
            above_blue = (np.linalg.norm(red[:2] - blue[:2]) < 0.012 and red[2] > blue[2] + 2 * HALF + 0.02)
            if above_blue:
                return "over_blue"
            if red[2] > TABLE_H + HALF + 0.06:
                return "lifted"
            return "grasped"
        dxy = np.linalg.norm(tcp[:2] - red[:2])
        if ap > 0.065 and dxy < 0.015 and abs(tcp[2] - (red[2] + GRASP_DZ)) < 0.012:
            return "at_red"
        if ap > 0.065 and dxy < 0.02 and 0.06 < tcp[2] - red[2] < 0.16:
            return "above_red"
        return "free"

    # --------------------------------------------------------------- skills
    def _build_skills(self):
        H_ABOVE = 0.12
        S = {}

        def near(obs, pos, tol):
            return np.linalg.norm(obs["tcp"] - pos) < tol

        def grasp_yaw(obs):
            # cube is 90-degree symmetric: of the two finger-closing axes pick the one
            # most perpendicular to the neighbouring (blue) cube, so a finger never
            # lands on it when the cubes are close.
            y0 = obs["red_yaw"]
            d = obs["blue_est"][:2] - obs["red_est"][:2]
            d = d / (np.linalg.norm(d) + 1e-9)
            cands = [y0, wrap_to(y0 + np.pi / 2, np.pi)]
            return min(cands, key=lambda y: abs(np.sin(y) * d[0] - np.cos(y) * d[1]) - 1e-3 * abs(y))

        # 1 reach above the red cube (re-planned from the live estimate)
        def t_reach(obs, st):
            return self.cmd(obs["red_est"] + [0, 0, H_ABOVE], grasp_yaw(obs), 0.08)

        def o_reach(obs, labels, st):
            p = obs["red_est"] + [0, 0, H_ABOVE]
            return OK if near(obs, p, 0.008) and obs["aperture"] > 0.07 else None

        # 2 descend around the cube
        def t_descend(obs, st):
            return self.cmd(obs["red_est"] + [0, 0, GRASP_DZ], grasp_yaw(obs), 0.08)

        def o_descend(obs, labels, st):
            return OK if near(obs, obs["red_est"] + [0, 0, GRASP_DZ], 0.005) else None

        # 3 close the gripper (grasp is decided by MEASURED pad contact)
        def e_grasp(obs, st):
            st.memory["hold"] = obs["tcp"].copy()
            st.memory["yaw"] = self.tcp_yaw()

        def t_grasp(obs, st):
            return self.cmd(st.memory["hold"], st.memory["yaw"], 0.0)

        def o_grasp(obs, labels, st):
            if obs["grasp"] and obs["contact_L"] > 3 and obs["contact_R"] > 3 and st.elapsed(obs["t"]) > 0.5:
                return OK
            if obs["aperture"] < 2 * HALF - 0.012 and not obs["grasp"]:
                return "missed"
            return None

        # 4 lift
        def e_lift(obs, st):
            st.memory["hold"] = obs["tcp"].copy()
            st.memory["yaw"] = self.tcp_yaw()
            st.memory["offset"] = obs["red_true"] - obs["tcp"]

        def t_lift(obs, st):
            return self.cmd(st.memory["hold"] + [0, 0, 0.16], st.memory["yaw"], 0.0)

        def dropped(obs, st):
            # Dropped = the cube has physically left the fingers: no pad contact AND its
            # centre is away from the TCP.  A transient force dip is not a drop.
            held = (obs["contact_L"] + obs["contact_R"] > 0.2 or
                    np.linalg.norm(obs["red_true"] - obs["tcp"]) < 0.03)
            if held:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1

        def o_lift(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            return OK if obs["red_true"][2] > TABLE_H + HALF + 0.10 else None

        # 5 carry over the blue cube (cube faces aligned with the blue cube)
        def carry_tcp(obs, clearance):
            off = obs["red_true"] - obs["tcp"]
            top = obs["blue_est"][2] + HALF
            return np.r_[obs["blue_est"][:2] - off[:2], top + clearance + HALF - off[2]]

        def t_move(obs, st):
            return self.cmd(carry_tcp(obs, 0.05), obs["blue_yaw"], 0.0)

        def o_move(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            red, blue = obs["red_true"], obs["blue_est"]
            ok = (np.linalg.norm(red[:2] - blue[:2]) < 0.006 and
                  np.linalg.norm(obs["tcp"] - carry_tcp(obs, 0.05)) < 0.008)
            return OK if ok else None

        # 6 lower onto the blue cube until physical contact
        def t_lower(obs, st):
            return self.cmd(carry_tcp(obs, -0.003), obs["blue_yaw"], 0.0)

        def o_lower(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            red, blue = obs["red_true"], obs["blue_true"]
            seated = (np.linalg.norm(red[:2] - blue[:2]) < 0.008 and red[2] - blue[2] < 2 * HALF + 0.003)
            return OK if seated and st.elapsed(obs["t"]) > 0.3 else None

        # 7 open the gripper
        def e_hold(obs, st):
            st.memory["hold"] = obs["tcp"].copy()
            st.memory["yaw"] = self.tcp_yaw()

        def t_release(obs, st):
            return self.cmd(st.memory["hold"], st.memory["yaw"], 0.08)

        def o_release(obs, labels, st):
            return OK if obs["aperture"] > 0.065 and obs["contact_L"] + obs["contact_R"] < 0.3 else None

        # 8 retreat upward
        def t_retreat(obs, st):
            return self.cmd(st.memory["hold"] + [0, 0, 0.15], st.memory["yaw"], 0.08)

        def o_retreat(obs, labels, st):
            return OK if obs["tcp"][2] > obs["red_true"][2] + HALF + 0.08 else None

        # Anchors at the nominal goal of each token; trust radius covers the approach
        # from the previous token's goal plus the randomization range.
        nominal = {
            "reach_above_red": self.cmd(RED_NOM + [0, 0, H_ABOVE], 0, 0.08),
            "descend_red": self.cmd(RED_NOM + [0, 0, GRASP_DZ], 0, 0.08),
            "grasp_red": self.cmd(RED_NOM + [0, 0, GRASP_DZ], 0, 0.0),
            "lift_red": self.cmd(RED_NOM + [0, 0, GRASP_DZ + 0.16], 0, 0.0),
            "move_over_blue": self.cmd(BLUE_NOM + [0, 0, 2 * HALF + 0.05 + GRASP_DZ], 0, 0.0),
            "lower_onto_blue": self.cmd(BLUE_NOM + [0, 0, 2 * HALF + GRASP_DZ], 0, 0.0),
            "release": self.cmd(BLUE_NOM + [0, 0, 2 * HALF + GRASP_DZ], 0, 0.08),
            "retreat": self.cmd(BLUE_NOM + [0, 0, 2 * HALF + GRASP_DZ + 0.15], 0, 0.08),
        }
        home = self.cmd([0.474, 0.0, TABLE_H + 0.41], 0, 0.08)
        prev = home
        deltas = {}
        for name in self.nominal_plan():
            deltas[name] = self.space.norm(nominal[name] - prev) + 0.32
            prev = nominal[name]
        spec = [
            ("reach_above_red", t_reach, o_reach, None, 10.0, 2.5),
            ("descend_red", t_descend, o_descend, None, 6.0, 1.5),
            ("grasp_red", t_grasp, o_grasp, e_grasp, 3.0, 1.2),
            ("lift_red", t_lift, o_lift, e_lift, 6.0, 1.5),
            ("move_over_blue", t_move, o_move, None, 10.0, 2.5),
            ("lower_onto_blue", t_lower, o_lower, None, 6.0, 1.5),
            ("release", t_release, o_release, e_hold, 3.0, 1.0),
            ("retreat", t_retreat, o_retreat, e_hold, 6.0, 1.5),
        ]
        for name, tgt, out, enter, timeout, dur in spec:
            S[name] = Skill(name, nominal[name], deltas[name], tgt, out, timeout=timeout,
                            nominal_duration=dur, on_enter=enter)
        self._skills = S

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        return ["reach_above_red", "descend_red", "grasp_red", "lift_red", "move_over_blue",
                "lower_onto_blue", "release", "retreat"]

    def skill_graph(self) -> list:
        e = word("")
        r = word("released")
        g = word("grasped")
        return [
            SkillSpec("reach_above_red", "free", (OutcomeSpec(OK, "above_red", word("", "released")),)),
            SkillSpec("reach_above_red", "at_red", (OutcomeSpec(OK, "above_red", r),)),
            SkillSpec("reach_above_red", "grasped", (OutcomeSpec(OK, "above_red", word("", "released")),)),
            SkillSpec("descend_red", "above_red", (OutcomeSpec(OK, "at_red", r),)),
            SkillSpec("grasp_red", "at_red", (OutcomeSpec(OK, "grasped", word("released", "")),
                                              OutcomeSpec("missed", "free", word("released", ""), failure=True))),
            SkillSpec("lift_red", "grasped", (OutcomeSpec(OK, "lifted", word("", "grasped")),
                                              OutcomeSpec("dropped", "free", e, failure=True))),
            SkillSpec("move_over_blue", "lifted", (OutcomeSpec(OK, "over_blue", g),
                                                   OutcomeSpec("dropped", "free", word("grasped", ""), failure=True))),
            SkillSpec("lower_onto_blue", "over_blue", (OutcomeSpec(OK, "placed_held", g),
                                                       OutcomeSpec("dropped", "free", word("grasped", ""), failure=True))),
            SkillSpec("release", "placed_held", (OutcomeSpec(OK, "released_on_blue", word("", {"stacked", "released"})),)),
            SkillSpec("retreat", "released_on_blue", (OutcomeSpec(OK, "done", word({"stacked", "released"})),)),
        ]

    # ------------------------------------------------------------- stress test
    def disturb(self) -> None:
        """(1) a sideways push (sized for a 3-5 cm slide) on the red cube as the gripper descends onto it
        (object-pose perturbation); (2) a 0.3 s gripper force dropout to +-1 N while
        the cube is carried (deliberate grasp failure).  Both from the paper's
        robustness protocol; triggered by measured state, identical for all methods."""
        d, t = self.data, self.data.time
        bid_red = self.model.body("red_cube").id
        d.xfrc_applied[bid_red, :3] = 0.0
        red = d.body("red_cube").xpos
        tcp = self.arm.tcp_pos()
        P = self._pert
        if P["push"] is None:
            if (np.linalg.norm(tcp[:2] - red[:2]) < 0.02 and 0.0 < tcp[2] - red[2] < 0.06
                    and self.arm.aperture() > 0.06):
                P["push"] = {"t_trigger": t, "t_start": t + self._push_delay, "dir": self._push_dir}
        elif P["push"].get("done") is None:
            p = P["push"]
            # force sized from mass/friction so the cube slides ~3-5 cm (a table bump)
            T = 0.1
            slide = 0.03 + 0.02 * (0.5 + 0.5 * np.sin(3 * p["dir"]))
            v0 = np.sqrt(2 * self.mu * 9.81 * slide)
            F = self.red_mass * (v0 / T + self.mu * 9.81)
            p["force_N"], p["target_slide_m"] = float(F), float(slide)
            if p["t_start"] <= t < p["t_start"] + T:
                d.xfrc_applied[bid_red, :2] = F * np.array([np.cos(p["dir"]), np.sin(p["dir"])])
            elif t >= p["t_start"] + T:
                p["done"] = True
                p["displacement_m"] = None
        g = self._geometry()
        if P["dropout"] is None and g["grasp"] and red[2] > TABLE_H + HALF + 0.08:
            P["dropout"] = {"t_trigger": t, "t_start": t + self._drop_delay}
        drop = P["dropout"]
        lim = self.model.actuator_forcerange[self.arm.act_grip]
        if drop is not None and drop["t_start"] <= t < drop["t_start"] + 0.3:
            lim[:] = (-1.0, 1.0)
        else:
            lim[:] = self._grip_range

    def perturbation_info(self) -> dict:
        return self._pert

    # ---------------------------------------------------------- diagnostics
    def step_diagnostics(self) -> dict:
        f = self._forces()
        return {"arm_table_force_N": f["arm_table"], "finger_table_force_N": f["fingers_table"],
                "finger_red_force_N": f["L_red"] + f["R_red"],
                "blue_displacement_m": float(np.linalg.norm(self.data.body("blue_cube").xpos[:2] - self.blue0[:2]))}

    def log_signals(self) -> dict:
        return {"red": self.data.body("red_cube").xpos.copy(), "blue": self.data.body("blue_cube").xpos.copy(),
                "tcp": self.arm.tcp_pos(), "aperture": self.arm.aperture()}

    def episode_metrics(self) -> dict:
        red = self.data.body("red_cube").xpos
        blue = self.data.body("blue_cube").xpos
        return {"final_red_blue_xy_offset_m": float(np.linalg.norm(red[:2] - blue[:2])),
                "final_red_height_above_blue_m": float(red[2] - blue[2])}

    def variant_info(self) -> dict:
        return {"red_init": self.red0, "blue_init": self.blue0, "red_yaw": self.red_yaw0,
                "blue_yaw": self.blue_yaw0, "red_mass_kg": self.red_mass, "blue_mass_kg": self.blue_mass,
                "cube_friction": self.mu, "perception_bias_red_m": self.est_red.b_far,
                "perception_bias_blue_m": self.est_blue.b_far}
