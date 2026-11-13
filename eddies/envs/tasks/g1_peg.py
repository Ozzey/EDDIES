"""Task 5 - Unitree G1 humanoid peg-in-hole insertion with a Dex3-1 hand.

"G1 takes the peg and inserts it into the hole using its robotic hands and
fingers."  The free-standing G1 (balanced by its own whole-body controller,
``controllers/g1.py``) faces a table.  A wooden/plastic peg (3 cm diameter,
14 cm long, 55-110 g) stands upright on the table; a hole fixture bolted to the
table (static) has a round hole (16-gon of box segments, 2.0-3.0 mm diametral
clearance, 6 cm deep, 45-degree chamfer of 2.5-3.5 mm at the entrance).  The
right hand grasps the peg with the Dex3 power synergy (thumb + index + middle +
palm, friction only), lifts it, aligns it over the hole using the wrist-camera
estimates, inserts it >= 4 cm deep, releases it and retreats, while standing.

Failure handling (contracts with explicit failure outcomes, budget 2):
grasp miss -> re-approach; drop -> re-approach (if the peg is still upright);
jam (no insertion progress) -> back off 1.5 cm and re-align with a small spiral
search offset, then insert again.

LTLf:  G !bad & F(grasped & F(inserted & released))
bad = robot fall (pelvis height/tilt, non-foot floor contact), peg off the
table / knocked over, or excessive forbidden contact (robot body vs table/
fixture/peg, hand vs table or fixture).

Deviation from the request ("e.g. 10-12 cm long"): the peg is 14 cm long.  The
Dex3-1 power grasp spans 8.8 cm along the peg (index finger at +2.85 cm, middle
finger at -2.85 cm, palm +-4.4 cm around the grasp centre); a symmetric
four-contact grasp (index AND middle on the peg, otherwise the unbalanced
finger torque rotates the peg in the hand, measured 4 -> 15 deg) plus >= 4 cm
insertion plus finger clearance above the fixture needs >= 13 cm.
"""
from __future__ import annotations

import numpy as np

from eddies.envs.controllers.g1 import PREGRASP_OFFSET
from eddies.envs.framework.perception import PoseEstimator, tilt_of_quat
from eddies.envs.framework.physics import DENSITY, FRICTION
from eddies.envs.framework.scene import GEOM
from eddies.envs.framework.task import OK, Skill, SkillSpec, OutcomeSpec, word
from eddies.envs.tasks.g1_base import TABLE_H, G1TaskBase

PEG_HH = 0.07                 # peg half length [m] (14 cm peg, see module docstring)
G_TOP = 0.040                 # grasp-site height below the peg top when grasping [m]
                              # (index finger fully beside the peg, clear of the top rim; chosen
                              # with the former convex-hull finger colliders, kept with the
                              # primitive colliders of controllers/g1.py)
ABOVE = 0.075                 # grasp-site height above the peg top for the approach [m]
BLOCK_PLATE = 0.010           # fixture base plate thickness [m]
HOLE_DEPTH = 0.060            # hole depth above the plate [m]
INSERT_GOAL = 0.045           # commanded insertion depth [m] (finger clearance 1.1 cm)
INSERT_OK = 0.040             # success insertion depth [m]
CARRY_CLEAR = 0.016           # peg-bottom clearance above the fixture top while carrying [m]
N_SEG = 16
HAND_PEG_FORCE_REF = 400.0    # [N] reference for the force-transient diagnostic (step_diagnostics)
PEG_SOLREF = (0.004, 1.0)
PEG_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
PEG_NOM = np.array([0.330, -0.250])
HOLE_NOM = np.array([0.360, -0.100])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _min_rotation(a, b) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s2 = float(v @ v)
    if s2 < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s2)


def _quat_from_mat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


class G1PegInsertionTask(G1TaskBase):
    name = "g1_peg_insertion"
    description = ("Unitree G1 humanoid (free-standing, balanced) grasps a 3 cm x 14 cm peg with its "
                   "Dex3-1 hand and inserts it >= 4 cm into a chamfered hole (2-3 mm clearance).")
    horizon = 60.0
    success_hold = 1.0
    formula = "G !bad & F(grasped & F(inserted & released))"
    terminal_cells = ("done",)
    main_camera = "main"
    detail_camera = "detail"

    def __init__(self, seed: int, **variant):
        super().__init__(seed, **variant)
        rng = self.rng
        # ---------------- randomized physical parameters (realistic ranges)
        self.table_h = TABLE_H + rng.uniform(-0.015, 0.015)
        self.peg0 = np.r_[PEG_NOM + rng.uniform(-0.02, 0.02, 2), self.table_h + PEG_HH]
        self.hole_xy = HOLE_NOM + rng.uniform(-0.02, 0.02, 2)
        self.peg_r = 0.015 + rng.uniform(-0.0003, 0.0003)
        self.clearance = rng.uniform(0.0020, 0.0030)           # diametral
        self.hole_r = self.peg_r + self.clearance / 2
        self.chamfer = rng.uniform(0.0025, 0.0035)
        material = "wood" if rng.uniform() < 0.5 else "plastic"
        vol = np.pi * self.peg_r ** 2 * 2 * PEG_HH
        self.peg_mass = float(np.clip(vol * DENSITY[material] * rng.uniform(0.95, 1.15), 0.05, 0.12))
        self.peg_mu = FRICTION[material][0] * rng.uniform(0.9, 1.15)
        self.peg_material = material
        self.block_top = self.table_h + BLOCK_PLATE + HOLE_DEPTH
        # ---------------- scene
        sb = self.make_builder(self.name)
        self.add_table(sb, self.table_h)
        peg = sb.free_cylinder("peg", self.peg0 + [0, 0, 0.0005], self.peg_r, PEG_HH, mass=self.peg_mass,
                               rgba=(0.88, 0.55, 0.16, 1) if material == "wood" else (0.95, 0.75, 0.1, 1),
                               friction=(self.peg_mu, 0.005, 1e-4), condim=4)
        pg = peg.geoms[0]
        pg.priority = 1
        # rigid wood/plastic part: stiff contact (as the finger pads, DEVELOPER_GUIDE 2.4);
        # with MuJoCo's default 20 ms time constant a wedged peg penetrated the hole
        # walls by ~1.3 mm (5.7 deg tilt inside a 1.25 mm radial clearance).
        pg.solref = np.array(PEG_SOLREF)
        pg.solimp = np.array(PEG_SOLIMP)
        self._build_fixture(sb)
        hc = np.r_[self.hole_xy, self.block_top]
        pc = self.peg0
        mid = 0.5 * (hc + pc)
        # main: side profile (whole robot incl. legs/feet, arm and table; leaves the
        # top-right free for the detail inset); detail: hand + peg + fixture close-up
        sb.camera("main", (0.40, -2.30, 1.30), (0.32, -0.05, 0.78), fovy=45)
        sb.camera("detail", mid + np.array([0.28, -0.30, 0.12]), mid + np.array([0.0, 0.0, 0.03]), fovy=45)
        sb.camera("front", (1.20, -0.85, 1.30), (0.20, 0.02, 0.85), fovy=45)
        self.finish_build(sb)
        self.cm = self.contact_monitor({"peg": ["peg"], "block": ["hole_block"]})
        self.est_peg = PoseEstimator(rng, near_distance=0.15)
        self.est_hole = PoseEstimator(rng, near_distance=0.17)
        self.axis_noise = np.radians(0.4)
        self._contacts = {}
        self._search_k = 0
        self._build_skills()
        # stress-test schedule (drawn even when unused so nominal RNG streams match)
        self._push_sign = 1.0 if rng.uniform() < 0.5 else -1.0
        self._push_force = rng.uniform(40.0, 55.0)
        self._push_delay = rng.uniform(0.0, 0.3)
        self._nudge_dir = rng.uniform(-np.pi, np.pi)
        self._nudge_delay = rng.uniform(0.1, 0.4)
        self._nudge_slide = rng.uniform(0.02, 0.03)
        self._pert = {"torso_push": None, "peg_nudge": None}

    # ------------------------------------------------------------- fixture
    def _build_fixture(self, sb) -> None:
        """Static hole fixture: base plate + 16 wall segments + 16 chamfer wedges."""
        body = sb.spec.worldbody.add_body(name="hole_block", pos=[self.hole_xy[0], self.hole_xy[1], self.table_h])
        mat = sb.material("fixture_mat", (0.55, 0.62, 0.70, 1), specular=0.4, shininess=0.5)
        fr = list(FRICTION["plastic"])
        t = 0.024                                   # wall radial thickness
        c = self.chamfer
        rh = self.hole_r
        z0 = BLOCK_PLATE
        H = HOLE_DEPTH
        body.add_geom(name="block_plate", type=GEOM.mjGEOM_BOX, pos=[0, 0, BLOCK_PLATE / 2],
                      size=[0.055, 0.055, BLOCK_PLATE / 2], material=mat, friction=fr)
        tan = np.tan(np.pi / N_SEG)
        for i in range(N_SEG):
            a = 2 * np.pi * i / N_SEG
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            q = _quat_from_mat(Rz)
            # lower wall: inner face at rh, z in [z0, z0 + H - c]
            hl = (H - c) / 2
            r_c = rh + t / 2
            body.add_geom(name=f"wall_lo{i}", type=GEOM.mjGEOM_BOX, pos=list(Rz @ [r_c, 0, z0 + hl]), quat=list(q),
                          size=[t / 2, (rh + t) * tan + 0.0005, hl], material=mat, friction=fr)
            # upper wall: inner face at rh + c, z in [z0 + H - c, z0 + H]
            r_u = rh + c + (t - c) / 2
            body.add_geom(name=f"wall_hi{i}", type=GEOM.mjGEOM_BOX, pos=list(Rz @ [r_u, 0, z0 + H - c / 2]),
                          quat=list(q), size=[(t - c) / 2, (rh + t) * tan + 0.0005, c / 2], material=mat,
                          friction=fr)
            # chamfer wedge: inner face on the 45-degree line (rh, H-c) -> (rh+c, H)
            tw = c / np.sqrt(2)
            mid_r, mid_z = rh + c / 2, z0 + H - c / 2
            n_out = np.array([1.0, -1.0]) / np.sqrt(2)
            cr, cz = np.array([mid_r, mid_z]) + n_out * tw / 2
            # local x -> outward normal (1, 0, -1)/sqrt2, local z -> along the chamfer face
            Rw = Rz @ _rot_y(np.pi / 4)
            body.add_geom(name=f"chamfer{i}", type=GEOM.mjGEOM_BOX, pos=list(Rz @ [cr, 0, cz]),
                          quat=list(_quat_from_mat(Rw)),
                          size=[tw / 2, (rh + c) * tan + 0.0005, c / np.sqrt(2)],
                          material=mat, friction=fr)
        # visual top marker ring (no collision) so the hole is easy to see
        body.add_geom(name="hole_marker", type=GEOM.mjGEOM_CYLINDER, pos=[0, 0, z0 + 0.0005],
                      size=[rh - 0.001, 0.0005, 0], rgba=[0.15, 0.15, 0.18, 1], contype=0, conaffinity=0)

    # ---------------------------------------------------------------- state
    def _forces(self):
        if self._contacts.get("t") != self.data.time:
            self._contacts = self.cm.measure(self.data)
            self._contacts["t"] = self.data.time
        return self._contacts

    def _peg(self):
        b = self.data.body("peg")
        R = b.xmat.reshape(3, 3)
        axis = R[:, 2].copy()
        if axis[2] < 0:
            axis = -axis
        center = b.xpos.copy()
        return {"center": center, "axis": axis, "bottom": center - PEG_HH * axis, "top": center + PEG_HH * axis,
                "tilt": tilt_of_quat(b.xquat), "vel": self.data.cvel[b.id][3:].copy(), "quat": b.xquat.copy()}

    def _geometry(self):
        p = self._peg()
        f = self._forces()
        hand = f["palm_peg"] + f["thumb_peg"] + f["index_peg"] + f["middle_peg"]
        held = (f["palm_peg"] > 1.0 and f["thumb_peg"] > 1.0 and (f["index_peg"] > 1.0 or f["middle_peg"] > 1.0))
        hole = np.r_[self.hole_xy, self.block_top]
        lateral = float(np.linalg.norm(p["bottom"][:2] - hole[:2]))
        depth = float(self.block_top - p["bottom"][2]) if lateral < self.hole_r + 0.004 else -1.0
        return {**p, "f": f, "hand_force": hand, "held": held, "lateral": lateral, "depth": depth}

    def observe(self) -> dict:
        obs = self.base_obs()
        g = self._geometry()
        gp = obs["grasp_pos"]
        pc, _, pn = self.est_peg.estimate(g["center"], 0.0, gp)
        hole_true = np.r_[self.hole_xy, self.block_top]
        hc, _, hn = self.est_hole.estimate(hole_true, 0.0, gp)
        ax = g["axis"] + self.rng.normal(0, self.axis_noise, 3)
        ax /= np.linalg.norm(ax)
        f = g["f"]
        obs.update({
            "peg_est": pc, "peg_axis_est": ax, "peg_tip_est": pc - PEG_HH * ax, "peg_near": pn,
            "hole_est": hc, "hole_near": hn,
            "peg_true": g["center"], "peg_bottom": g["bottom"], "peg_top": g["top"], "peg_tilt": g["tilt"],
            "peg_speed": float(np.linalg.norm(g["vel"])), "held": g["held"], "depth": g["depth"],
            "lateral": g["lateral"], "f_palm": f["palm_peg"], "f_thumb": f["thumb_peg"],
            "f_index": f["index_peg"], "f_middle": f["middle_peg"], "hand_force": g["hand_force"],
            "f_hand_block": f.get("hand_block", 0.0)})
        return obs

    def _peg_off_table(self, g) -> bool:
        lo = self.table_h - 0.02
        return bool(g["center"][2] < lo)

    def labels(self) -> set:
        g = self._geometry()
        f = g["f"]
        out = set()
        if g["held"] and g["bottom"][2] > self.table_h + 0.01:
            out.add("grasped")
        if g["depth"] >= INSERT_OK and g["tilt"] < np.radians(8):
            out.add("inserted")
        if g["hand_force"] < 0.2 and self.g1.synergy_closure(self.hand, self.synergy_kind) < 0.3:
            out.add("released")
        smash = (f["body_table"] > 40.0 or f["body_block"] > 40.0 or f["body_peg"] > 40.0
                 or f["hand_table"] > 60.0 or f["hand_block"] > 80.0)
        if self.fallen() or self._peg_off_table(g) or smash:
            out.add("bad")
        return out

    def task_success(self) -> bool:
        g = self._geometry()
        gp = self.g1.grasp_pos(self.hand)
        return bool(g["depth"] >= INSERT_OK and g["tilt"] < np.radians(8) and g["hand_force"] < 0.2
                    and np.linalg.norm(gp - g["top"]) > 0.06 and not self.fallen())

    def cell(self, obs: dict, labels: frozenset) -> str:
        pt, pb, gp = obs["peg_true"], obs["peg_bottom"], obs["grasp_pos"]
        held = obs["held"]
        if obs["peg_tilt"] > np.radians(30) or pt[2] < self.table_h - 0.02:
            return "peg_lost"
        if obs["depth"] >= INSERT_OK:
            if held:
                return "inserted_held"
            if obs["hand_force"] < 0.2 and obs["closure"] < 0.3:
                return "done" if np.linalg.norm(gp - obs["peg_top"]) > 0.06 else "released_in_hole"
            return "inserted_held"
        if held:
            hole = obs["hole_est"]
            tip = obs["peg_tip_est"]
            if pb[2] < self.block_top + 0.006 and obs["lateral"] < 0.03:
                return "jammed"
            if (np.linalg.norm(tip[:2] - hole[:2]) < 0.004 and
                    self.block_top + 0.006 <= pb[2] <= self.block_top + 0.035):
                return "over_hole"
            if pb[2] > self.block_top + 0.006:
                return "lifted"
            return "grasped"
        # hand not holding the peg
        if obs["closure"] < 0.15:
            pre = self._pregrasp(obs)
            dxy = np.linalg.norm(gp[:2] - pre[:2])
            if dxy < 0.008 and abs(gp[2] - pre[2]) < 0.008:
                return "at_peg"
            if dxy < 0.012 and 0.05 < gp[2] - obs["peg_top"][2] < 0.11:
                return "above_peg"
        return "free"

    # --------------------------------------------------------------- skills
    def _pregrasp(self, obs, z=None):
        """Grasp-site target that puts the peg at the pre-grasp location in the open hand."""
        pc = obs["peg_est"]
        top = pc[2] + PEG_HH
        p = np.r_[pc[:2] - PREGRASP_OFFSET[:2], top - G_TOP if z is None else z]
        return p

    def _build_skills(self):
        S = {}
        hold_rv = np.zeros(3)

        def near(obs, p, tol):
            return np.linalg.norm(obs["grasp_pos"] - p) < tol

        # 1 reach above the peg (open hand)
        def t_reach(obs, st):
            p = self._pregrasp(obs, obs["peg_est"][2] + PEG_HH + ABOVE)
            return self.cmd(p, hold_rv, 0.0)

        def o_reach(obs, labels, st):
            p = self._pregrasp(obs, obs["peg_est"][2] + PEG_HH + ABOVE)
            return OK if near(obs, p, 0.008) and obs["closure"] < 0.1 and obs["peg_speed"] < 0.01 else None

        # 2 descend around the peg; the descent pauses (hand re-centres at its current
        # height) while the peg moves or the hand is off the capture window
        def t_descend(obs, st):
            pre = self._pregrasp(obs)
            gp = obs["grasp_pos"]
            if obs["peg_speed"] > 0.01 or np.linalg.norm(gp[:2] - pre[:2]) > 0.006:
                z = max(gp[2], pre[2])
                return self.cmd(np.r_[pre[:2], z], hold_rv, 0.0)
            return self.cmd(pre, hold_rv, 0.0)

        def o_descend(obs, labels, st):
            if obs["peg_tilt"] > np.radians(12):
                return "knocked"
            return OK if near(obs, self._pregrasp(obs), 0.005) and obs["peg_speed"] < 0.01 else None

        # 3 close the power synergy (grasp decided by MEASURED contact forces)
        def e_grasp(obs, st):
            st.memory["hold"] = obs["grasp_pos"].copy()

        def t_grasp(obs, st):
            # the hand holds still while the power synergy closes; the fingers pull the
            # peg ~2 cm onto the grasp site (it may tilt a few degrees about its base;
            # tested: advancing the hand during closure caused palm misses)
            return self.cmd(st.memory["hold"], hold_rv, 1.0)

        def o_grasp(obs, labels, st):
            if obs["held"] and st.elapsed(obs["t"]) > 0.7 and obs["closure"] > 0.5:
                return OK
            if obs["closure"] > 0.85 and not obs["held"] and st.elapsed(obs["t"]) > 0.7:
                return "missed"
            return None

        def dropped(obs, st):
            held = obs["hand_force"] > 0.3 or np.linalg.norm(obs["peg_true"] - obs["grasp_pos"]) < 0.04
            if held:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1

        # 4 lift
        def e_lift(obs, st):
            st.memory["hold"] = obs["grasp_pos"].copy()

        def t_lift(obs, st):
            if "zgoal" not in st.memory:
                # raise the peg bottom to CARRY_CLEAR (+4 mm) above the fixture top
                dz = self.block_top + CARRY_CLEAR + 0.004 - obs["peg_bottom"][2]
                st.memory["zgoal"] = float(obs["grasp_pos"][2] + dz)
            return self.cmd(np.r_[st.memory["hold"][:2], st.memory["zgoal"]], hold_rv, 1.0)

        def o_lift(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            return OK if obs["peg_bottom"][2] > self.block_top + CARRY_CLEAR - 0.004 else None

        # 5 carry the peg over the hole (tip aligned on the hole axis, using wrist-camera estimates)
        def upright(obs, st):
            """Hand orientation that makes the held peg vertical (in-hand tilt compensation):
            the peg axis in the hand frame (EMA of the wrist-camera estimate) is rotated onto
            world z by the minimal rotation."""
            R = obs["grasp_mat"]
            a_h = R.T @ obs["peg_axis_est"]
            prev = st.memory.get("a_h")
            a_h = a_h if prev is None else 0.85 * prev + 0.15 * a_h
            a_h /= np.linalg.norm(a_h)
            st.memory["a_h"] = a_h
            R_des = _min_rotation(a_h, np.array([0.0, 0.0, 1.0]))
            return R_des, R_des @ R.T

        def carry_target(obs, st, clearance, search=np.zeros(2)):
            R_des, R_rel = upright(obs, st)
            gp = obs["grasp_pos"]
            off = R_rel @ (obs["peg_tip_est"] - gp)     # tip offset once the hand is re-oriented
            hole = obs["hole_est"]
            xy = hole[:2] + search - off[:2]
            z = self.block_top + clearance - off[2]
            return np.r_[xy, z], self.g1.chart.encode_matrix(R_des)

        def t_move(obs, st):
            p, rv = carry_target(obs, st, CARRY_CLEAR)
            return self.cmd(p, rv, 1.0)

        def o_move(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            ok = (np.linalg.norm(obs["peg_tip_est"][:2] - obs["hole_est"][:2]) < 0.0015 and
                  abs(obs["peg_bottom"][2] - (self.block_top + CARRY_CLEAR)) < 0.006 and
                  obs["peg_speed"] < 0.02)
            return OK if ok else None

        # 6 insert: descend slowly along the hole axis, re-aligning from the estimates
        def e_insert(obs, st):
            st.memory["z0"] = float(obs["grasp_pos"][2])
            st.memory["best"] = obs["depth"]
            st.memory["t_best"] = obs["t"]

        def t_insert(obs, st):
            goal, rv = carry_target(obs, st, -INSERT_GOAL, self._search_offset())
            # descend at 3.5 cm/s; never command more than 12 mm below the measured hand
            # height (bounded push force when the peg meets the chamfer or jams)
            z = max(goal[2], st.memory["z0"] - 0.035 * st.elapsed(obs["t"]), obs["grasp_pos"][2] - 0.012)
            return self.cmd(np.r_[goal[:2], z], rv, 1.0)

        def o_insert(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            if obs["depth"] >= INSERT_GOAL - 0.003 or (obs["depth"] >= INSERT_OK and st.elapsed(obs["t"]) > 4.0):
                return OK
            if obs["depth"] > st.memory["best"] + 0.002:
                st.memory["best"], st.memory["t_best"] = obs["depth"], obs["t"]
            if st.elapsed(obs["t"]) > 1.5 and obs["t"] - st.memory["t_best"] > 1.0:
                self._search_k += 1            # next spiral search offset for the re-alignment
                return "jammed"
            return None

        # 7 back off and re-align with a small spiral search offset (recovery from a jam)
        def t_realign(obs, st):
            p, rv = carry_target(obs, st, CARRY_CLEAR, self._search_offset())
            return self.cmd(p, rv, 1.0)

        def o_realign(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            ok = (np.linalg.norm(obs["peg_tip_est"][:2] - obs["hole_est"][:2] - self._search_offset()) < 0.0015
                  and abs(obs["peg_bottom"][2] - (self.block_top + CARRY_CLEAR)) < 0.006 and obs["peg_speed"] < 0.02)
            return OK if ok else None

        # 8 release (open the hand in place)
        def e_hold(obs, st):
            st.memory["hold"] = obs["grasp_pos"].copy()
            st.memory["rv"] = self.g1.chart.encode_matrix(obs["grasp_mat"])   # keep the wrist orientation

        def t_release(obs, st):
            return self.cmd(st.memory["hold"], st.memory["rv"], 0.0)

        def o_release(obs, labels, st):
            return OK if obs["closure"] < 0.12 and obs["hand_force"] < 0.2 and st.elapsed(obs["t"]) > 0.3 else None

        # 9 retreat: hand backwards (-x) and up, clear of the peg
        def t_retreat(obs, st):
            return self.cmd(st.memory["hold"] + [-0.07, -0.03, 0.06], st.memory["rv"], 0.0)

        def o_retreat(obs, labels, st):
            return OK if np.linalg.norm(obs["grasp_pos"] - obs["peg_top"]) > 0.08 and \
                near(obs, st.memory["hold"] + [-0.07, -0.03, 0.06], 0.02) else None

        # nominal goals (normalized) of every token -> anchors and trust radii
        pre = np.r_[PEG_NOM - PREGRASP_OFFSET[:2], TABLE_H + 2 * PEG_HH - G_TOP]
        lift_z = TABLE_H + BLOCK_PLATE + HOLE_DEPTH + CARRY_CLEAR + (2 * PEG_HH - G_TOP)
        over = np.r_[HOLE_NOM, lift_z]
        ins = np.r_[HOLE_NOM, lift_z - CARRY_CLEAR - INSERT_GOAL]
        nominal = {
            "reach_above_peg": self.cmd(pre + [0, 0, G_TOP + ABOVE], hold_rv, 0.0),
            "descend_to_peg": self.cmd(pre, hold_rv, 0.0),
            "grasp_peg": self.cmd(pre, hold_rv, 1.0),
            "lift_peg": self.cmd(np.r_[pre[:2], lift_z], hold_rv, 1.0),
            "move_over_hole": self.cmd(over, hold_rv, 1.0),
            "insert_peg": self.cmd(ins, hold_rv, 1.0),
            "realign_search": self.cmd(over, hold_rv, 1.0),
            "release_peg": self.cmd(ins, hold_rv, 0.0),
            "retreat": self.cmd(ins + [-0.07, -0.03, 0.06], hold_rv, 0.0),
        }
        self.ready_site = np.array([0.22, -0.25, TABLE_H + 0.24])
        home = self.cmd(self.ready_site, hold_rv, 0.0)
        prev = home
        deltas = {}
        for name in self.nominal_plan():
            deltas[name] = self.space.norm(nominal[name] - prev) + 0.30
            prev = nominal[name]
        deltas["realign_search"] = self.space.norm(nominal["realign_search"] - nominal["insert_peg"]) + 0.30
        spec = [
            ("reach_above_peg", t_reach, o_reach, None, 10.0, 2.0),
            ("descend_to_peg", t_descend, o_descend, None, 6.0, 1.5),
            ("grasp_peg", t_grasp, o_grasp, e_grasp, 4.0, 1.5),
            ("lift_peg", t_lift, o_lift, e_lift, 6.0, 1.5),
            ("move_over_hole", t_move, o_move, None, 10.0, 2.5),
            ("insert_peg", t_insert, o_insert, e_insert, 8.0, 2.5),
            ("realign_search", t_realign, o_realign, None, 6.0, 1.5),
            ("release_peg", t_release, o_release, e_hold, 4.0, 1.2),
            ("retreat", t_retreat, o_retreat, e_hold, 6.0, 1.5),
        ]
        for name, tgt, out, enter, timeout, dur in spec:
            S[name] = Skill(name, nominal[name], deltas[name], tgt, out, timeout=timeout,
                            nominal_duration=dur, on_enter=enter)
        self._skills = S

    def _search_offset(self) -> np.ndarray:
        k = self._search_k
        if k == 0:
            return np.zeros(2)
        r = 0.0012 * np.sqrt(k)
        a = k * np.radians(137.5)
        return r * np.array([np.cos(a), np.sin(a)])

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        return ["reach_above_peg", "descend_to_peg", "grasp_peg", "lift_peg", "move_over_hole",
                "insert_peg", "release_peg", "retreat"]

    def skill_graph(self) -> list:
        e = word("")
        r = word("released")
        g = word("grasped")
        gi = {"grasped", "inserted"}
        return [
            SkillSpec("reach_above_peg", "free", (OutcomeSpec(OK, "above_peg", word("", "released")),)),
            SkillSpec("reach_above_peg", "at_peg", (OutcomeSpec(OK, "above_peg", r),)),
            SkillSpec("descend_to_peg", "above_peg", (OutcomeSpec(OK, "at_peg", r),
                                                      OutcomeSpec("knocked", "free", r, failure=True))),
            SkillSpec("grasp_peg", "at_peg", (OutcomeSpec(OK, "grasped", word("released", "")),
                                              OutcomeSpec("missed", "free", word("released", ""), failure=True))),
            SkillSpec("lift_peg", "grasped", (OutcomeSpec(OK, "lifted", word("", "grasped")),
                                              OutcomeSpec("dropped", "free", e, failure=True),
                                              OutcomeSpec("dropped", "free", word("", "grasped", ""), failure=True))),
            SkillSpec("move_over_hole", "lifted", (OutcomeSpec(OK, "over_hole", g),
                                                   OutcomeSpec("dropped", "free", word("grasped", ""), failure=True))),
            SkillSpec("insert_peg", "over_hole", (OutcomeSpec(OK, "inserted_held", word("grasped", {"grasped", "inserted"})),
                                                  OutcomeSpec("jammed", "jammed", g, failure=True),
                                                  OutcomeSpec("dropped", "free", word("grasped", ""), failure=True))),
            SkillSpec("realign_search", "jammed", (OutcomeSpec(OK, "over_hole", g),
                                                   OutcomeSpec("dropped", "free", word("grasped", ""), failure=True))),
            SkillSpec("release_peg", "inserted_held", (OutcomeSpec(OK, "released_in_hole",
                                                                   word(gi, "inserted", {"inserted", "released"})),)),
            SkillSpec("retreat", "released_in_hole", (OutcomeSpec(OK, "done", word({"inserted", "released"})),)),
        ]

    # -------------------------------------------------------------- reset
    def ready_hand_pose(self):
        """Hand held in front of the chest (glass orientation), 24 cm above the table."""
        return np.array([0.22, -0.25, self.table_h + 0.24]), np.eye(3)

    def reset(self) -> None:
        super().reset()
        self._search_k = 0
        self._t_above, self._t_diag = 0.0, 0.0

    # ------------------------------------------------------------- stress test
    def disturb(self) -> None:
        """(1) lateral push on the torso (40-55 N for 0.2 s = 8-11 N s, sign random) while the hand
        reaches for the peg; (2) a slow nudge at the peg base (smooth sinusoidal
        acceleration profile, 2-3 cm slide, peak acceleration g r / (2 h) = half the
        tipping limit, so the 14 cm peg slides and stops upright) while the hand hovers
        above it.  Triggered by measured state + per-seed delay."""
        d, t = self.data, self.data.time
        P = self._pert
        gp = self.g1.grasp_pos(self.hand)
        g = self._peg()
        # (1) torso push
        if P["torso_push"] is None:
            pre = np.r_[g["center"][:2] - PREGRASP_OFFSET[:2]]
            dist = np.linalg.norm(gp[:2] - pre)
            if 0.04 < dist < 0.12 and t > 0.2:
                P["torso_push"] = {"t_trigger": t, "t_start": t + self._push_delay, "duration": 0.2,
                                   "force_N": [0.0, self._push_sign * self._push_force, 0.0]}
        push = P["torso_push"]
        active = push is not None and push["t_start"] <= t < push["t_start"] + push["duration"]
        self.torso_push(active, push["force_N"] if push else (0, 0, 0))
        if push is not None and push["t_start"] <= t <= push["t_start"] + 2.0:
            # measured response during the 2 s after the push onset
            pel = d.xpos[self.g1.pelvis][:2]
            if "pelvis_xy0" not in push:
                push["pelvis_xy0"] = pel.tolist()
                push.update(impulse_Ns=float(abs(push["force_N"][1]) * push["duration"]),
                            max_pelvis_disp_m=0.0, max_pelvis_tilt_deg=0.0, min_com_margin_m=1.0)
            bal = self.g1.balance_state()
            push["max_pelvis_disp_m"] = max(push["max_pelvis_disp_m"],
                                            float(np.linalg.norm(pel - np.asarray(push["pelvis_xy0"]))))
            push["max_pelvis_tilt_deg"] = max(push["max_pelvis_tilt_deg"], float(np.degrees(bal["pelvis_tilt"])))
            push["min_com_margin_m"] = min(push["min_com_margin_m"], float(bal["com_margin"]))
        # (2) peg nudge at the base
        b = self.model.body("peg").id
        nud = P["peg_nudge"]
        if nud is None:
            pre = self._pregrasp({"peg_est": g["center"]}, g["top"][2] + ABOVE)
            if np.linalg.norm(gp[:2] - pre[:2]) < 0.03 and gp[2] - g["top"][2] > 0.045:
                mu = self.pair_friction("peg_geom", "table_top")
                # peak acceleration well below the tipping limit g r / h (= 2.1 m/s^2 for the
                # 14 cm peg): the base is pushed, decelerates smoothly and stops upright
                A = 0.5 * 9.81 * self.peg_r / PEG_HH
                T = float(np.sqrt(2 * np.pi * self._nudge_slide / A))
                P["peg_nudge"] = {"t_trigger": t, "t_start": t + self._nudge_delay, "duration": T,
                                  "dir": float(self._nudge_dir), "mu": mu, "peak_acc": A,
                                  "target_slide_m": float(self._nudge_slide), "start_pos": g["center"].tolist()}
                nud = P["peg_nudge"]
        if nud is not None and nud.get("done") is None:
            if nud["t_start"] <= t < nud["t_start"] + nud["duration"]:
                s = (t - nud["t_start"]) / nud["duration"]
                a = nud["peak_acc"] * np.sin(2 * np.pi * s)
                Fm = self.peg_mass * (nud["mu"] * 9.81 + a)
                u = np.array([np.cos(nud["dir"]), np.sin(nud["dir"])])
                self.object_push("peg", Fm * u, point=g["bottom"] + np.array([0, 0, 0.002]))
            elif t >= nud["t_start"] + nud["duration"]:
                self.object_push("peg", None)
                nud["done"] = True
                nud["displacement_m"] = float(np.linalg.norm(g["center"][:2] - np.asarray(nud["start_pos"])[:2]))
                nud["tilt_after_deg"] = float(np.degrees(g["tilt"]))
            else:
                self.object_push("peg", None)
        else:
            self.object_push("peg", None)

    def perturbation_info(self) -> dict:
        return self._pert

    # ---------------------------------------------------------- diagnostics
    def step_diagnostics(self) -> dict:
        """Per-step maxima.  Besides the contact forces: ``hand_peg_time_above_400N_s`` is
        the CUMULATIVE time (so its episode maximum is the total) during which the summed
        hand-peg normal force exceeded 400 N; ``finger_contact_torque_Nm`` is the largest
        generalized constraint torque on an active-hand finger joint (``qfrc_constraint``:
        contacts J^T f + the 0.3 Nm joint friction loss).  Interpretation (measured): a
        static power grasp, where every finger joint sits at its 1.4 Nm limit, gives a
        summed normal force of 240-300 N (Streamer hand pose) up to 400-480 N (a slightly
        different pose: shorter contact levers, palm up to 330 N); the
        finger-joint constraint torque is then <= ~1.4 Nm.  Larger torques (up to ~9 Nm)
        appear only for 20-40 ms while a closing finger is stopped by the peg (impact of
        the menagerie's 0.01 kg m^2 reflected motor inertia per finger joint)."""
        f = self._forces()
        hp = f["palm_peg"] + f["thumb_peg"] + f["index_peg"] + f["middle_peg"]
        t = float(self.data.time)
        if hp > HAND_PEG_FORCE_REF:
            self._t_above += max(0.0, t - self._t_diag)
        self._t_diag = t
        hd = self.g1.hand[self.hand][1]
        out = {"body_table_force_N": f["body_table"], "hand_table_force_N": f["hand_table"],
               "hand_block_force_N": f["hand_block"], "body_block_force_N": f["body_block"],
               "hand_peg_force_N": hp, "hand_peg_time_above_400N_s": self._t_above,
               "palm_peg_force_N": f["palm_peg"], "thumb_peg_force_N": f["thumb_peg"],
               "index_peg_force_N": f["index_peg"], "middle_peg_force_N": f["middle_peg"],
               "finger_contact_torque_Nm": float(np.max(np.abs(self.data.qfrc_constraint[hd])))}
        out.update(self.balance_diagnostics())
        return out

    def log_signals(self) -> dict:
        g = self._peg()
        bal = self.g1.balance_state()
        return {"peg": g["center"], "peg_tilt": g["tilt"], "grasp": self.g1.grasp_pos(self.hand),
                "closure": self.g1.synergy_closure(self.hand, self.synergy_kind),
                "com": bal["com"], "pelvis": self.data.xpos[self.g1.pelvis].copy()}

    def episode_metrics(self) -> dict:
        g = self._geometry()
        return {"final_insertion_depth_m": g["depth"], "final_peg_tilt_deg": float(np.degrees(g["tilt"])),
                "final_peg_lateral_offset_m": g["lateral"], "jam_retries": int(self._search_k)}

    def variant_info(self) -> dict:
        return {"table_height_m": self.table_h, "peg_init": self.peg0, "hole_xy": self.hole_xy,
                "peg_radius_m": self.peg_r, "hole_radius_m": self.hole_r,
                "diametral_clearance_m": self.clearance, "chamfer_m": self.chamfer,
                "peg_mass_kg": self.peg_mass, "peg_friction": self.peg_mu, "peg_material": self.peg_material,
                "perception_bias_peg_m": self.est_peg.b_far, "perception_bias_hole_m": self.est_hole.b_far}

    def overlay_lines(self) -> list:
        g = self._geometry()
        lines = super().overlay_lines()
        lines.append(f"peg: depth {100 * max(g['depth'], 0):4.1f} cm   tilt {np.degrees(g['tilt']):4.1f} deg"
                     f"   hand force {g['hand_force']:5.1f} N")
        return lines
