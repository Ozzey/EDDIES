"""Task 7 - Unitree G1 humanoid opens a drawer, takes a caddy out of it and puts it on the table.

"G1 opens the drawer (camera placed so the opening and especially the handle are
clearly visible), takes a simple object out of the drawer with its robotic hand
and puts it on the table."

Scene
-----
The free-standing G1 (balanced by its own whole-body controller, ``controllers/g1.py``)
faces a 0.78 m table.  A one-drawer wooden cabinet (``assets/furniture.add_drawer_cabinet``,
0.30 x 0.36 m, standing on four 3 cm hardwood feet) faces the robot.  The drawer
(1.4-2.2 kg, slide joint with 5-12 N s/m damping, 1-3 N breakaway friction, 0.22 m travel)
has a 13 mm metal D-pull (0.20 m long, 65 mm projection) and a tray box whose sides are
7 cm lower than the 12.4 cm front panel (a common construction: decorative front over a
low drawer box).  Inside, right behind the front panel, stands a small wooden pen caddy
(4 x 12 x 3 cm base, two posts, a 12 mm bar handle 7.5 cm above the base, 0.12-0.25 kg).

Hand strategy (Dex3-1, one hand orientation for the whole task)
---------------------------------------------------------------
The Dex3 thumb is long and, when swung back, hangs ~6 cm below the palm; with the palm
horizontal it collides with the drawer front when the hand reaches behind it.  The hand
therefore works PITCHED 45 deg fingers-down (palm facing down-backwards) in every skill:
the swung-back thumb then stays above the bar it hooks (>= 7 mm above the caddy bar,
which is 11 mm below the front-panel top).  Both bars are taken with a "deep hook":
index + middle flex to (1.55, 1.74) rad (beyond contact, torque-limited 1.4 N m per
joint) and pull the bar into the palm, the thumb stays swung back (thumb_1 = 0.45 rad;
0.70 rad touched the forearm at high wrist flexion).  The bar is held by the curled
fingers against the palm (form closure + skin friction), nothing else.  Hook offsets
(bar centre in the hand frame) were found with closed-loop probes and 2-D collider
sections: drawer bar (0.100, 0.026) m with 0.40 pre-curl (hand front 54 mm ahead of the bar,
front panel at 65 mm), caddy bar (0.110, 0.025) m (0.08-0.09 let the bar slip out under
the closing fingertips).  Hooks are confirmed by a 1.5 cm tug and MEASURED contact force.

Task sequence (skill tokens)
----------------------------
reach_handle -> above_handle -> hook_handle -> grasp_handle -> pull_drawer (>= 12 cm) ->
release_handle -> retreat_handle -> reach_caddy (closed hand, over the drawer front) ->
preshape_caddy -> descend_caddy -> grasp_caddy -> lift_caddy (base above the drawer
sides) -> carry_caddy (to the goal disc on the table, right of the cabinet) ->
lower_caddy -> release_caddy -> retreat.

Failure outcomes (contracts, budget 3): handle missed (re-reach), handle slip while
pulling (re-reach / continue if already open), drawer found pushed back with the caddy
inside (re-open), caddy grasp miss (re-preshape), caddy dropped (re-pick from where it
rests: drawer or table).  All outcomes are decided from measured state (slide-joint
encoder, mj_contactForce contact forces, caddy pose).

Stress test (``--perturb``): (1) lateral torso push 35-50 N for 0.2 s (7-10 N s, inside
the controller's 60 N x 0.2 s envelope) while the drawer is being pulled; (2) the drawer
is pushed back in by 6-9 cm (15 N along the closing axis) shortly after the handle was
released, i.e. with the caddy inside it, so it has to be re-opened.

LTLf: G !bad & F(opened & F(grasped & F(placed & released)))
bad = robot fall, caddy off the table / knocked over, or excessive forbidden contact
(robot body vs table/cabinet/caddy, hand vs table or cabinet carcass).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from eddies.core.online import RecursiveLeastSquares
from eddies.envs.assets.furniture import add_drawer_cabinet
from eddies.envs.controllers.g1 import GRASP_SITE
from eddies.envs.framework.contacts import ContactMonitor
from eddies.envs.framework.perception import PoseEstimator, tilt_of_quat, yaw_of_quat
from eddies.envs.framework.physics import FRICTION
from eddies.envs.framework.scene import GEOM, body_geoms
from eddies.envs.framework.task import OK, Skill, SkillSpec, OutcomeSpec, word
from eddies.envs.tasks.g1_base import G1TaskBase

STIFF_SOLREF = (0.004, 1.0)
STIFF_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)

# ----------------------------------------------------------------- geometry
TABLE_H_DRAWER = 0.78
CAB_FEET = 0.03                                    # cabinet feet height
CAB_W, CAB_D = 0.30, 0.36
DRAWER_H = 0.13
TRAVEL = 0.22
BAR_R = 0.0065
BAR_LEN = 0.20
BAR_STANDOFF = 0.065                               # large-clearance D-pull (fingers are 27 mm thick)
BOX_MARGIN = 0.070                                 # drawer box sides 7 cm lower than the front panel
# caddy: base block + two posts + a tall bar handle (bar along the cabinet's lateral axis)
CADDY_HALF = np.array([0.020, 0.060, 0.015])       # base half sizes (x = depth, y = length, z)
CADDY_BAR_R = 0.006
CADDY_BAR_H = 0.075                                # bar centre above the base top
CADDY_POST = 0.004                                 # post half thickness (along y)
CADDY_GAP = 0.006                                  # caddy base to the front panel's inner face
CADDY_BAR_Z = 2 * CADDY_HALF[2] + CADDY_BAR_H      # bar centre above the caddy bottom
# hand / hook
PITCH = np.radians(45.0)                           # fingers 45 deg below horizontal in every skill
B_HANDLE = np.array([0.100, 0.026, 0.0])           # drawer bar centre in the hand frame before closing
B_CADDY = np.array([0.110, 0.025, 0.0])            # caddy bar centre in the hand frame before closing
HOOK_OPEN = np.array([0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0])
HOOK_CLOSED = np.array([0.0, 0.45, 0.0, 1.55, 1.74, 1.55, 1.74])
S_PRE_H = 0.40                                     # finger pre-curl for the handle approach
S_PRE_C = 0.20                                     # finger pre-curl for the caddy approach
S_REL = 0.35                                       # closure while letting go of the drawer bar
S_REL_C = 0.45                                     # closure while letting go of the caddy bar
SLIDE_OUT = 0.06                                   # caddy release: hand slides 6 cm along -palm normal
TUG = 0.015                                        # hook test pull [m]
PRE_H = np.array([-0.03, 0.0, 0.13])               # handle pre-pose rel. to the hook pose (fingers above the bar)
ABOVE_H = np.array([0.0, 0.0, 0.09])               # above the hook pose (fingertips in front of the bar)
# drawer
OPEN_MIN = 0.12                                    # 'opened': caddy reachable
OPEN_TARGET = 0.125                                # pull token goal (measured opening)
PULL_CMD = 0.135                                   # commanded pull (arm lag ~1 cm)
# caddy goal on the table (right of the cabinet)
GOAL_NOM = np.array([0.31, -0.34])
GOAL_TOL = 0.04
LIFT_CLEAR = 0.024                                 # commanded caddy bottom above the drawer sides
LIFT_OK = 0.010                                    # lifted: caddy bottom >= 1 cm above the drawer sides
CARRY_EXTRA = 0.02


def rot_y(a):
    return Rot.from_euler("y", a).as_matrix()


def palm_down(pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """Hand rotation: palm facing down (hand y along world -z), fingers along world +x,
    then pitched fingers-down by ``pitch`` about world y and yawed about world z."""
    return Rot.from_euler("z", yaw).as_matrix() @ rot_y(pitch) @ Rot.from_euler("x", -np.pi / 2).as_matrix()


def hook_posture(s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    return HOOK_OPEN + s * (HOOK_CLOSED - HOOK_OPEN)


class G1DrawerTask(G1TaskBase):
    name = "g1_drawer"
    description = ("Unitree G1 humanoid (free-standing, balanced) opens a drawer by its bar handle with a "
                   "Dex3-1 finger hook (measured slide-joint opening), takes a small caddy out of the drawer "
                   "by its bar handle and puts it down on the table in a goal disc (contact forces only).")
    synergy_kind = "hook"
    horizon = 75.0
    success_hold = 1.0
    formula = "G !bad & F(opened & F(grasped & F(placed & released)))"
    terminal_cells = ("done",)
    failure_budget = 3
    main_camera = "main"
    detail_camera = "detail"
    # IK null-space posture of the decoder (elbow out, forearm pronated): with the menagerie
    # READY_ARM posture the near-body pull end put the elbow/shoulder into the torso
    IK_POSTURE = np.array([-0.35, -0.45, 0.3, 0.8, 0.0, 0.0, 0.0])

    def __init__(self, seed: int, **variant):
        super().__init__(seed, **variant)
        rng = self.rng
        # ---------------- randomized physical parameters (realistic ranges)
        self.table_h = TABLE_H_DRAWER
        self.front_x = 0.458 + rng.uniform(-0.008, 0.008)
        self.cab_y = -0.09 + rng.uniform(-0.01, 0.01)
        self.cab_yaw = np.pi + rng.uniform(-0.03, 0.03)
        self.drawer_mass = rng.uniform(1.4, 2.2)
        self.drawer_damping = rng.uniform(5.0, 12.0)
        self.drawer_fric = rng.uniform(1.0, 3.0)
        self.caddy_mass = rng.uniform(0.12, 0.25)
        self.caddy_mu = FRICTION["wood"][0] * rng.uniform(0.9, 1.15)
        self.caddy_lat = rng.uniform(-0.012, 0.012)
        self.goal = GOAL_NOM + rng.uniform(-0.015, 0.015, 2)
        # ---------------- scene
        sb = self.make_builder(self.name)
        self.add_table(sb, self.table_h, x_front=0.24, depth=0.62, y_center=-0.12, width=0.84)
        n_out = np.array([np.cos(self.cab_yaw), np.sin(self.cab_yaw), 0.0])
        self.n_out = n_out
        center = np.array([self.front_x, self.cab_y, 0.0]) - (CAB_D / 2) * n_out
        self.cab_center = center + [0, 0, self.table_h + CAB_FEET]
        info = add_drawer_cabinet(sb, "cabinet", self.cab_center, self.cab_yaw, width=CAB_W, depth=CAB_D,
                                  drawer_heights=(DRAWER_H,), travel=TRAVEL, drawer_mass=self.drawer_mass,
                                  damping=self.drawer_damping, frictionloss=self.drawer_fric,
                                  handle_length=BAR_LEN, handle_radius=BAR_R, handle_standoff=BAR_STANDOFF,
                                  box_height_margin=BOX_MARGIN)
        self.cab = info
        self.dr = info["drawers"][0]
        self.Rc = Rot.from_euler("z", self.cab_yaw).as_matrix()
        floor = self.dr["interior_floor_local"]
        self.floor_z0 = self.cab_center[2] + floor[2]
        self.front_top_z = self.cab_center[2] + self.dr["front_top_local_z"]
        self.cab_top_z = self.cab_center[2] + info["size"][2]
        self.side_top_z = self.floor_z0 + self.dr["interior_height"]          # drawer box sides
        self._build_feet(sb)
        obj_local = np.array([self.dr["front_inner_local_x"] - CADDY_HALF[0] - CADDY_GAP, self.caddy_lat,
                              floor[2] + CADDY_HALF[2] + 0.0005])
        self.obj0 = self.cab_center + self.Rc @ obj_local
        self._build_caddy(sb, self.obj0, self.cab_yaw - np.pi)
        self.handle0 = self.cab_center + self.Rc @ self.dr["handle_closed_local"]
        sb.spec.worldbody.add_geom(name="goal_marker", type=GEOM.mjGEOM_CYLINDER,
                                   pos=[*self.goal, self.table_h + 0.0005], size=[GOAL_TOL, 0.0005, 0],
                                   rgba=[0.1, 0.75, 0.25, 0.45], contype=0, conaffinity=0, group=1, mass=0.0)
        h0 = self.handle0
        # main: from behind-right of the robot, looking diagonally at the drawer front (handle,
        # opening and goal disc in view, whole robot); detail: handle / opening close-up from the left
        sb.camera("main", (-0.45, -1.05, 1.55), (0.28, -0.12, 0.74), fovy=50)
        sb.camera("detail", h0 + np.array([-0.10, 0.36, 0.20]), h0 + np.array([-0.06, -0.02, -0.02]), fovy=55)
        sb.camera("side", (0.35, -1.6, 1.15), (0.35, -0.1, 0.85), fovy=40)
        self.finish_build(sb)
        self.g1.ik[self.hand].posture = self.IK_POSTURE.copy()
        m = self.model
        self.jq_drawer = int(m.jnt_qposadr[m.joint(self.dr["joint"]).id])
        self.jd_drawer = int(m.jnt_dofadr[m.joint(self.dr["joint"]).id])
        self.drawer_bid = m.body(self.dr["body"]).id
        self.caddy_bid = m.body("caddy").id
        # contact groups (physical normal forces)
        drawer_box = [self.dr["body"]]
        self.cm = self.contact_monitor({"handle": [self.dr["handle_body"]], "caddy": ["caddy"],
                                        "drawer": drawer_box, "carcass": [info["body"]]})
        caddy_g = set(body_geoms(m, ["caddy"]))
        robot_g = set(self.g1.robot_geoms)
        env = [g for g in range(m.ngeom) if g not in caddy_g and g not in robot_g
               and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.cm_obj = ContactMonitor(m, {"caddy": sorted(caddy_g), "table": self.table_geoms,
                                         "drawer": body_geoms(m, drawer_box), "env": env},
                                     {"caddy_table": ("caddy", "table"), "caddy_drawer": ("caddy", "drawer"),
                                      "caddy_env": ("caddy", "env")})
        self._contacts = {}
        self._hist = []
        # perception (shared by all methods; same per seed)
        self.est_handle = PoseEstimator(rng, bias_far=0.003, sigma_far=0.001, near_distance=0.12)
        self.est_caddy = PoseEstimator(rng, near_distance=0.15)
        self.q_noise = 0.001
        self._build_skills()
        # stress-test schedule (drawn even when unused so the nominal RNG stream is unchanged)
        self._push_sign = 1.0 if rng.uniform() < 0.5 else -1.0
        self._push_force = rng.uniform(35.0, 50.0)
        self._push_delay = rng.uniform(0.0, 0.3)
        self._dpush_delay = rng.uniform(0.2, 0.5)
        self._dpush_dist = rng.uniform(0.06, 0.09)
        self._pert = {"torso_push": None, "drawer_push": None}
        self._handle_free_since = None
        self.max_opening = 0.0

    # ------------------------------------------------------------- scene parts
    def _build_feet(self, sb):
        """Four static hardwood feet (3 cm) under the carcass corners."""
        mat = sb.material("feet_mat", (0.35, 0.25, 0.18, 1), specular=0.1, shininess=0.2)
        for k, (fx, fy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            p = self.cab_center + self.Rc @ np.array([fx * (CAB_D / 2 - 0.03), fy * (CAB_W / 2 - 0.03), 0.0])
            sb.spec.worldbody.add_geom(name=f"cabinet_foot{k}", type=GEOM.mjGEOM_CYLINDER,
                                       pos=[p[0], p[1], self.table_h + CAB_FEET / 2], size=[0.018, CAB_FEET / 2, 0],
                                       material=mat, friction=list(FRICTION["wood"]))

    def _build_caddy(self, sb, pos, yaw):
        """Wooden pen caddy: base block, two posts, bar handle (rigid, stiff contacts)."""
        b = sb.free_body("caddy", pos, quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
        mat = sb.material("caddy_mat", (0.30, 0.55, 0.80, 1), specular=0.2, shininess=0.3)
        hx, hy, hz = CADDY_HALF
        fr = [self.caddy_mu, 0.005, 1e-4]
        b.add_geom(name="caddy_base", type=GEOM.mjGEOM_BOX, size=[hx, hy, hz], material=mat, friction=fr,
                   mass=0.75 * self.caddy_mass, condim=4)
        post_h = (CADDY_BAR_H + CADDY_BAR_R) / 2
        for k, sy in enumerate((-1, 1)):
            b.add_geom(name=f"caddy_post{k}", type=GEOM.mjGEOM_BOX,
                       pos=[0, sy * (hy - CADDY_POST), hz + post_h], size=[0.006, CADDY_POST, post_h],
                       material=mat, friction=fr, mass=0.1 * self.caddy_mass, condim=4)
        b.add_geom(name="caddy_bar", type=GEOM.mjGEOM_CAPSULE, pos=[0, 0, hz + CADDY_BAR_H],
                   quat=[np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0],
                   size=[CADDY_BAR_R, hy - 2 * CADDY_POST, 0], material=mat, friction=fr,
                   mass=0.05 * self.caddy_mass, condim=4)
        for g in b.geoms:
            g.solref = np.array(STIFF_SOLREF)
            g.solimp = np.array(STIFF_SOLIMP)
            g.priority = 1

    # ---------------------------------------------------------------- decoder
    def ready_arm(self) -> dict:
        # left arm abducted 0.4 rad: with the default stand pose the swung-back left thumb
        # presses into the left hip link (kN self-contact)
        return {"left": np.array([0.2, 0.4, 0.0, 1.28, 0.0, 0.0, 0.0])}

    def ready_hand_pose(self):
        return np.array([0.30, -0.12, 1.00]), palm_down(PITCH)

    def apply_command(self, e) -> None:
        """Shared decoder: grasp-site pose by whole-body IK, closure -> deep-hook finger posture."""
        dec = self.g1.decode(self.space, self.space.clip(e))
        self.g1.command_hand(self.hand, dec["pos"], dec["mat"])
        s = float(np.clip(dec["grasp"], 0.0, 1.0))
        self.g1.synergy[self.hand] = ("hook", s)
        self.g1.hand_goal[self.hand] = hook_posture(s)

    def closure(self) -> float:
        """Measured closure: finger encoders projected onto the hook direction."""
        dq = HOOK_CLOSED - HOOK_OPEN
        q = self.g1.hand_q(self.hand)
        return float(np.clip((q - HOOK_OPEN) @ dq / (dq @ dq), 0.0, 1.0))

    def initial_command(self) -> np.ndarray:
        return self.g1.encode(self.space, self.hand, closure=self.closure())

    def base_obs(self) -> dict:
        obs = super().base_obs()
        c = self.closure()
        obs["closure"] = c
        obs["achieved"] = self.g1.encode(self.space, self.hand, closure=c)
        return obs

    def hand_R(self, yaw_est: float | None = None) -> np.ndarray:
        yaw = (self.cab_yaw if yaw_est is None else yaw_est) - np.pi
        return palm_down(PITCH, yaw)

    def hook_site(self, bar, R, B) -> np.ndarray:
        """Grasp-site target that puts bar point ``bar`` at hand-frame point ``B``."""
        return np.asarray(bar, float) + R @ (GRASP_SITE - B)

    def cmd_R(self, pos, R, s: float) -> np.ndarray:
        return self.cmd(pos, self.g1.chart.encode_matrix(R), s)

    # ------------------------------------------------------------------ state
    def opening(self) -> float:
        return float(self.data.qpos[self.jq_drawer])

    def handle_pos(self) -> np.ndarray:
        return self.data.site(self.dr["handle_site"]).xpos.copy()

    def forces(self) -> dict:
        if self._contacts.get("t") != self.data.time:
            self._contacts = self.cm.measure(self.data)
            self._contacts.update(self.cm_obj.measure(self.data))
            self._contacts["t"] = self.data.time
        return self._contacts

    def _windowed(self, hand_caddy: float, caddy_env: float, window: float = 0.25):
        """Max over the last ``window`` s of the hand-caddy and caddy-environment forces."""
        t = float(self.data.time)
        h = self._hist
        if not h or h[-1][0] != t:
            if h and t < h[-1][0]:
                h.clear()
            h.append((t, hand_caddy, caddy_env))
            while h and h[0][0] < t - window:
                h.pop(0)
        return max(x[1] for x in h), max(x[2] for x in h)

    def caddy_state(self) -> dict:
        d = self.data
        b = self.caddy_bid
        R = d.xmat[b].reshape(3, 3)
        pos = d.xpos[b].copy()
        return {"pos": pos, "quat": d.xquat[b].copy(), "tilt": tilt_of_quat(d.xquat[b]),
                "bar": pos + R @ np.array([0, 0, CADDY_HALF[2] + CADDY_BAR_H]),
                "bottom": float(pos[2] - CADDY_HALF[2]), "vel": d.cvel[b][3:].copy()}

    def in_drawer(self, p) -> bool:
        """Point (caddy centre) over the drawer's interior floor (drawer frame, measured)."""
        rel = self.Rc.T @ (np.asarray(p, float) - self.data.site(self.dr["interior_site"]).xpos)
        hx, hy = self.dr["interior_half"]
        return bool(abs(rel[0]) < hx + 0.01 and abs(rel[1]) < hy + 0.01 and rel[2] < 0.06)

    def _geometry(self) -> dict:
        f = self.forces()
        c = self.caddy_state()
        # caddy in the hand: measured hand-caddy contact with the bar at the hook (the curled
        # fingers alone carry it by form closure; palm/thumb contact is not required).  A caddy
        # hanging in the hook rocks and its stiff bar contact chatters (single physics steps with
        # zero force), so contact is taken as the maximum over the last 0.25 s.
        hc_max, env_max = self._windowed(f["hand_caddy"], f["caddy_env"])
        held = (hc_max > 0.5 and self.closure() > 0.6
                and np.linalg.norm(c["bar"] - self.g1.grasp_pos(self.hand)) < 0.07)
        in_air = env_max < 0.05
        hold_handle = f["hand_handle"] > 1.0 and self.closure() > 0.6
        in_goal = np.linalg.norm(c["pos"][:2] - self.goal) < GOAL_TOL
        on_table = f["caddy_table"] > 0.2 and abs(c["bottom"] - self.table_h) < 0.006
        touchdown = f["caddy_table"] > 1.0 and in_goal                 # resting on the table (maybe on an edge)
        upright = c["tilt"] < np.radians(10)
        return {**c, "f": f, "held": bool(held), "in_air": bool(in_air), "hold_handle": bool(hold_handle),
                "touchdown": bool(touchdown), "hand_caddy_w": float(hc_max),
                "in_goal": bool(in_goal), "on_table": bool(on_table), "upright": bool(upright),
                "placed": bool(on_table and in_goal and upright), "hand_caddy": f["hand_caddy"],
                "hand_handle": f["hand_handle"], "opening": self.opening()}

    def observe(self) -> dict:
        obs = self.base_obs()
        g = self._geometry()
        gp = obs["grasp_pos"]
        hp_true = self.handle_pos()
        hp, hy, hn = self.est_handle.estimate(hp_true, self.cab_yaw, gp)
        cp, cy, cn = self.est_caddy.estimate(g["bar"], yaw_of_quat(g["quat"]), gp)
        f = g["f"]
        obs.update({
            "handle_est": hp, "handle_true": hp_true, "cab_yaw_est": hy, "handle_near": hn,
            "caddy_bar_est": cp, "caddy_yaw_est": cy + np.pi, "caddy_near": cn,
            "caddy_bar": g["bar"], "caddy_pos": g["pos"], "caddy_bottom": g["bottom"], "caddy_tilt": g["tilt"],
            "caddy_speed": float(np.linalg.norm(g["vel"])), "caddy_in_drawer": self.in_drawer(g["pos"]),
            "opening": g["opening"], "opening_est": g["opening"] + self.rng.normal(0, self.q_noise),
            "held": g["held"], "in_air": g["in_air"], "hold_handle": g["hold_handle"], "placed": g["placed"],
            "touchdown": g["touchdown"], "hand_caddy_w": g["hand_caddy_w"],
            "on_table": g["on_table"], "hand_caddy": g["hand_caddy"], "hand_handle": g["hand_handle"],
            "f_index_c": f["index_caddy"], "f_middle_c": f["middle_caddy"], "f_palm_c": f["palm_caddy"],
            "f_index_h": f["index_handle"], "f_middle_h": f["middle_handle"], "f_palm_h": f["palm_handle"],
        })
        return obs

    def labels(self) -> set:
        g = self._geometry()
        f = g["f"]
        out = set()
        if g["opening"] >= OPEN_MIN:
            out.add("opened")
        if g["held"] and g["in_air"]:
            out.add("grasped")
        if g["placed"]:
            out.add("placed")
        if g["hand_caddy"] < 0.3:
            out.add("released")
        lost = g["pos"][2] < self.table_h - 0.05 or g["tilt"] > np.radians(60)
        smash = (f["body_table"] > 40.0 or f["body_carcass"] > 40.0 or f["body_drawer"] > 40.0
                 or f["body_handle"] > 40.0 or f["body_caddy"] > 40.0
                 or f["hand_table"] > 60.0 or f["hand_carcass"] > 150.0)
        if self.fallen() or lost or smash:
            out.add("bad")
        return out

    def task_success(self) -> bool:
        g = self._geometry()
        gp = self.g1.grasp_pos(self.hand)
        return bool(g["placed"] and g["hand_caddy"] < 0.3 and np.linalg.norm(g["vel"]) < 0.03
                    and gp[2] > g["bar"][2] + 0.05 and not self.fallen())

    # ------------------------------------------------------------ abstraction
    def _handle_hook(self, obs):
        R = self.hand_R(obs["cab_yaw_est"])
        bar = obs["handle_est"] + np.array([0.0, -0.02, 0.0])        # hook 2 cm right of the bar centre
        return self.hook_site(bar, R, B_HANDLE), R

    def _caddy_hook(self, obs):
        R = self.hand_R(obs["caddy_yaw_est"])
        return self.hook_site(obs["caddy_bar_est"], R, B_CADDY), R

    def lift_height(self, obs) -> float:
        """Grasp-site height (from the measured offset) that puts the caddy bottom above its support."""
        support = self.side_top_z if obs["caddy_in_drawer"] else self.table_h + 0.03
        off = obs["grasp_pos"][2] - obs["caddy_bottom"]
        return float(support + LIFT_CLEAR + off)

    def cell(self, obs: dict, labels: frozenset) -> str:
        gp = obs["grasp_pos"]
        if obs["caddy_pos"][2] < self.table_h - 0.05 or obs["caddy_tilt"] > np.radians(45):
            return "lost"
        clos = obs["closure"]
        free = obs["hand_caddy"] < 0.3
        if obs["placed"] and free:
            return "done" if gp[2] > obs["caddy_bar"][2] + 0.08 else "on_goal"
        if obs["placed"] and obs["held"]:
            return "placed_held"
        if obs["held"] and obs["touchdown"] and not obs["in_air"]:
            return "touchdown"                 # held caddy resting on the table in the goal, not yet flat
        if obs["held"]:
            if obs["in_air"]:
                over = (np.linalg.norm(obs["caddy_pos"][:2] - self.goal) < 0.012 and
                        obs["caddy_bottom"] > self.table_h + 0.02)
                if over:
                    return "over_goal"
                if obs["caddy_bottom"] > self.side_top_z + LIFT_OK - 0.002:
                    return "lifted"
            return "c_grasped"
        if obs["hold_handle"]:
            return "h_open" if obs["opening"] >= OPEN_TARGET else "h_grasped"
        hook, _ = self._handle_hook(obs)
        dh = np.linalg.norm(gp - hook)
        caddy_inside = obs["caddy_in_drawer"]
        if obs["opening"] < OPEN_MIN and caddy_inside:
            if dh < 0.006 and clos < 0.5:
                return "at_handle"
            if np.linalg.norm(gp - (hook + ABOVE_H)) < 0.008 and clos < 0.5:
                return "above_handle"
            if np.linalg.norm(gp - (hook + PRE_H)) < 0.010 and clos < 0.5:
                return "pre_handle"
            return "closed"
        if caddy_inside and dh < 0.06 and gp[2] < hook[2] + 0.09:
            return "at_handle_open"
        ch, _ = self._caddy_hook(obs)
        if clos < 0.4 and np.linalg.norm(gp - ch) < 0.006:
            return "at_caddy"
        if clos < 0.4 and np.linalg.norm(gp - (ch + [0, 0, 0.08])) < 0.010:
            return "preshaped"
        if clos > 0.6 and np.linalg.norm(gp - (ch + [0, 0, 0.11])) < 0.012:
            return "over_caddy"
        return "open"

    # ----------------------------------------------------------------- skills
    def _build_skills(self):
        S = {}

        def near(obs, p, tol):
            return np.linalg.norm(obs["grasp_pos"] - p) < tol

        def caddy_pushed_in(obs):
            return obs["opening"] < OPEN_MIN and obs["caddy_in_drawer"]

        # 1 pre-pose behind and above the handle (re-planned from the estimate)
        def t_reach_h(obs, st):
            p, R = self._handle_hook(obs)
            if obs["closure"] > S_PRE_H + 0.15:          # open the fingers in place first (e.g. after
                return self.cmd_R(obs["grasp_pos"], R, S_PRE_H)   # a miss: a closed hook may catch the bar)
            return self.cmd_R(p + PRE_H, R, S_PRE_H)

        def o_reach_h(obs, labels, st):
            p, _ = self._handle_hook(obs)
            return OK if near(obs, p + PRE_H, 0.008) and obs["closure"] < 0.5 else None

        # 2 move forward over the bar (fingers stay in front of the carcass face)
        def t_above_h(obs, st):
            p, R = self._handle_hook(obs)
            return self.cmd_R(p + ABOVE_H, R, S_PRE_H)

        def o_above_h(obs, labels, st):
            if obs["opening"] >= OPEN_MIN:
                return "already_open"
            p, _ = self._handle_hook(obs)
            return OK if near(obs, p + ABOVE_H, 0.008) else None

        # 3 descend: the pre-curled fingers drop into the gap between bar and front panel
        def t_hook_h(obs, st):
            p, R = self._handle_hook(obs)
            return self.cmd_R(p, R, S_PRE_H)

        def o_hook_h(obs, labels, st):
            if obs["opening"] >= OPEN_MIN:
                return "already_open"
            p, _ = self._handle_hook(obs)
            return OK if near(obs, p, 0.005) else None

        # 4 close the deep hook (decided by MEASURED finger + palm contact on the bar)
        def e_hold(obs, st):
            st.memory["hold"] = obs["grasp_pos"].copy()
            st.memory["R"] = self.g1.grasp_mat(self.hand)

        def t_grasp_h(obs, st):
            # close the hook, then tug 1.5 cm along the opening direction: the hook is
            # confirmed only by MEASURED finger/palm contact force on the bar
            p = st.memory["hold"]
            if obs["closure"] > 0.9:
                p = p + TUG * self.n_out
            return self.cmd_R(p, st.memory["R"], 1.0)

        def o_grasp_h(obs, labels, st):
            if obs["hold_handle"] and obs["closure"] > 0.8 and st.elapsed(obs["t"]) > 0.5:
                return OK
            moved = (obs["grasp_pos"] - st.memory["hold"]) @ self.n_out
            if obs["closure"] > 0.9 and moved > TUG - 0.004 and not obs["hold_handle"]:
                return "missed"
            return None

        # 5 pull along the drawer axis (nominal: estimated front normal; Streamer online: RLS of the
        #   handle's motion line p(q) = p0 + q n from the measured grasp site and opening)
        def e_pull(obs, st):
            e_hold(obs, st)
            st.memory["q0"] = obs["opening_est"]
            n = np.array([np.cos(obs["cab_yaw_est"]), np.sin(obs["cab_yaw_est"]), 0.0])
            st.memory["n"] = n
            if st.memory.get("online"):
                rls = RecursiveLeastSquares(2, 3, ridge=1.0, forgetting=0.995)
                p0 = obs["grasp_pos"] - obs["opening_est"] * n
                rls.theta = np.column_stack([p0, n])
                rls.P = np.diag([1e-4, 1e-2])
                st.memory["rls"] = rls
            st.memory["lost"] = None

        def pull_target(obs, st):
            if st.memory.get("rls") is not None:
                rls = st.memory["rls"]
                if obs["hold_handle"]:
                    rls.update(np.array([1.0, obs["opening_est"]]), obs["grasp_pos"])
                return rls.predict(np.array([1.0, PULL_CMD]))
            return st.memory["hold"] + (PULL_CMD - st.memory["q0"]) * st.memory["n"]

        def t_pull(obs, st):
            return self.cmd_R(pull_target(obs, st), st.memory["R"], 1.0)

        def o_pull(obs, labels, st):
            if obs["hold_handle"]:
                st.memory["lost"] = None
                return OK if obs["opening"] >= OPEN_TARGET else None
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            gone = np.linalg.norm(obs["handle_true"] - obs["grasp_pos"]) > 0.05
            if obs["t"] - st.memory["lost"] > 0.15 and (gone or obs["hand_handle"] < 0.3):
                return "slipped_open" if obs["opening"] >= OPEN_MIN else "slipped"
            return None

        # 6 let go: fingers half open, hand 2 cm up and 0.5 cm back
        def e_rel(obs, st):
            e_hold(obs, st)
            st.memory["target"] = obs["grasp_pos"] + [0, 0, 0.02]

        def t_rel_h(obs, st):
            return self.cmd_R(st.memory["target"], st.memory["R"], S_REL)

        def o_rel_h(obs, labels, st):
            ok = obs["hand_handle"] < 0.3 and obs["closure"] < S_REL + 0.12 and st.elapsed(obs["t"]) > 0.3
            return OK if ok else None

        # 7 rise clear of the bar and the drawer front
        def e_retreat_h(obs, st):
            e_hold(obs, st)
            st.memory["target"] = obs["grasp_pos"] + [0, 0, 0.10]

        def t_retreat_h(obs, st):
            return self.cmd_R(st.memory["target"], st.memory["R"], S_REL)

        def o_retreat_h(obs, labels, st):
            ok = near(obs, st.memory["target"], 0.015) and obs["hand_handle"] < 0.3
            return OK if ok else None

        # 8 closed hand over the caddy (passes over the drawer front with the fingers curled in)
        def t_reach_c(obs, st):
            p, R = self._caddy_hook(obs)
            return self.cmd_R(p + [0, 0, 0.11], R, 1.0)

        def o_reach_c(obs, labels, st):
            if caddy_pushed_in(obs):
                return "closed_again"
            p, _ = self._caddy_hook(obs)
            return OK if near(obs, p + [0, 0, 0.11], 0.010) and obs["closure"] > 0.6 else None

        # 9 pre-shape the hook above the caddy bar
        def t_pre_c(obs, st):
            p, R = self._caddy_hook(obs)
            return self.cmd_R(p + [0, 0, 0.08], R, S_PRE_C)

        def o_pre_c(obs, labels, st):
            if caddy_pushed_in(obs):
                return "closed_again"
            p, _ = self._caddy_hook(obs)
            return OK if near(obs, p + [0, 0, 0.08], 0.008) and obs["closure"] < 0.35 else None

        # 10 descend: fingers pass in front of the caddy bar
        def t_desc_c(obs, st):
            p, R = self._caddy_hook(obs)
            return self.cmd_R(p, R, S_PRE_C)

        def o_desc_c(obs, labels, st):
            if caddy_pushed_in(obs):
                return "closed_again"
            p, _ = self._caddy_hook(obs)
            return OK if near(obs, p, 0.005) else None

        # 11 close the deep hook on the caddy bar (measured finger + palm contact)
        def t_grasp_c(obs, st):
            # close the hook, then tug 1.5 cm up: the grasp is confirmed by the MEASURED state
            # (caddy off its support and in contact with the hand only)
            p = st.memory["hold"]
            if obs["closure"] > 0.9:
                p = p + [0, 0, TUG]
            return self.cmd_R(p, st.memory["R"], 1.0)

        def o_grasp_c(obs, labels, st):
            if obs["held"] and obs["in_air"] and obs["closure"] > 0.8 and st.elapsed(obs["t"]) > 0.5:
                return OK
            moved = obs["grasp_pos"][2] - st.memory["hold"][2]
            if obs["closure"] > 0.9 and moved > TUG - 0.004:
                st.memory["tugged"] = st.memory.get("tugged") or obs["t"]
                if obs["t"] - st.memory["tugged"] > 0.4 and not (obs["held"] and obs["in_air"]):
                    return "missed"
            return None

        def dropped(obs, st):
            held = obs["hand_caddy"] > 0.3 or np.linalg.norm(obs["caddy_bar"] - obs["grasp_pos"]) < 0.06
            if held:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1

        # 12 lift the caddy until its bottom clears the drawer sides
        def e_lift(obs, st):
            e_hold(obs, st)
            st.memory["lost"] = None

        def t_lift(obs, st):
            # target from the CURRENT measured hand-caddy offset (the caddy settles in the hook)
            z = max(self.lift_height(obs), st.memory["hold"][2] + 0.03)
            return self.cmd_R(np.r_[st.memory["hold"][:2], z], st.memory["R"], 1.0)

        def o_lift(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            ok = obs["in_air"] and obs["caddy_bottom"] > self.side_top_z + LIFT_OK
            return OK if ok else None

        # 13 carry the caddy over the goal disc (measured caddy-hand offset)
        def carry_site(st, z):
            # hand-caddy offset measured when the carry starts (low-pass refreshed while the caddy
            # is calm): the hand does not chase a swinging caddy
            return np.r_[self.goal - st.memory["off"], z]

        def track_offset(obs, st):
            off = obs["caddy_pos"][:2] - obs["grasp_pos"][:2]
            if "off" not in st.memory:
                st.memory["off"] = off
            elif obs["caddy_speed"] < 0.03:
                st.memory["off"] = 0.9 * st.memory["off"] + 0.1 * off

        def e_carry(obs, st):
            e_hold(obs, st)
            st.memory["lost"] = None
            st.memory["z"] = obs["grasp_pos"][2] + CARRY_EXTRA
            track_offset(obs, st)

        def t_carry(obs, st):
            track_offset(obs, st)
            return self.cmd_R(carry_site(st, st.memory["z"]), st.memory["R"], 1.0)

        def o_carry(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            ok = (np.linalg.norm(obs["caddy_pos"][:2] - self.goal) < 0.008 and
                  near(obs, carry_site(st, st.memory["z"]), 0.01) and obs["caddy_speed"] < 0.03)
            return OK if ok else None

        # 14 lower onto the table until physical contact
        def e_lower(obs, st):
            e_hold(obs, st)
            st.memory["lost"] = None
            track_offset(obs, st)

        def t_lower(obs, st):
            track_offset(obs, st)
            z = obs["grasp_pos"][2] - (obs["caddy_bottom"] - self.table_h) - 0.003
            return self.cmd_R(carry_site(st, z), st.memory["R"], 1.0)

        def o_lower(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            down = obs["placed"] or (obs["touchdown"] and not obs["in_air"])
            return OK if down and st.elapsed(obs["t"]) > 0.3 and obs["caddy_speed"] < 0.02 else None

        # 15 open the hook half-way, then slide the hand out along its palm normal (45 deg
        #    forward-up): the bar leaves the hook under the palm, clear of the curled fingers
        def slide_dir(R):
            return -R[:, 1]

        def t_rel_c(obs, st):
            # open in place and let the caddy settle flat (it may touch down on an edge when
            # the grip is tilted), then slide the hand out
            p = st.memory["hold"]
            if st.memory.get("slide") or (obs["closure"] < S_REL_C + 0.15 and st.elapsed(obs["t"]) > 0.5
                                          and obs["caddy_speed"] < 0.02):
                st.memory["slide"] = True
                p = p + SLIDE_OUT * slide_dir(st.memory["R"])
            return self.cmd_R(p, st.memory["R"], S_REL_C)

        def o_rel_c(obs, labels, st):
            moved = (obs["grasp_pos"] - st.memory["hold"]) @ slide_dir(st.memory["R"])
            if obs["hand_caddy_w"] < 0.3 and moved > SLIDE_OUT - 0.02 and obs["caddy_speed"] < 0.01:
                return OK if obs["placed"] else "dropped"
            return None

        # 16 retreat straight up, clear of the caddy
        def e_retreat(obs, st):
            e_hold(obs, st)
            st.memory["target"] = obs["grasp_pos"] + [0, 0, 0.08]

        def t_retreat(obs, st):
            if obs["closure"] > S_REL_C + 0.15:
                return self.cmd_R(st.memory["hold"], st.memory["R"], S_REL_C)
            return self.cmd_R(st.memory["target"], st.memory["R"], S_REL_C)

        def o_retreat(obs, labels, st):
            ok = near(obs, st.memory["target"], 0.015) and obs["grasp_pos"][2] > obs["caddy_bar"][2] + 0.08
            return OK if ok else None

        # ---- anchors (nominal goal of each token) and trust radii
        R0 = self.hand_R()
        h_nom = self.handle0 + np.array([0, -0.02, 0])
        hook_h = self.hook_site(h_nom, R0, B_HANDLE)
        pull_end = hook_h + PULL_CMD * self.n_out
        rel_end = pull_end + [0, 0, 0.02]
        ret_end = rel_end + [0, 0, 0.10]
        c_open = self.obj0 + OPEN_TARGET * self.n_out
        bar_open = c_open + np.array([0, 0, CADDY_HALF[2] + CADDY_BAR_H])
        hook_c = self.hook_site(bar_open, R0, B_CADDY)
        off_z = hook_c[2] - (c_open[2] - CADDY_HALF[2])                 # grasp site above the caddy bottom
        lift_z = self.side_top_z + LIFT_CLEAR + off_z
        goal_site = np.r_[self.goal + (hook_c[:2] - c_open[:2]), lift_z + CARRY_EXTRA]
        place = np.r_[goal_site[:2], self.table_h + off_z]
        nominal = {
            "reach_handle": self.cmd_R(hook_h + PRE_H, R0, S_PRE_H),
            "above_handle": self.cmd_R(hook_h + ABOVE_H, R0, S_PRE_H),
            "hook_handle": self.cmd_R(hook_h, R0, S_PRE_H),
            "grasp_handle": self.cmd_R(hook_h, R0, 1.0),
            "pull_drawer": self.cmd_R(pull_end, R0, 1.0),
            "release_handle": self.cmd_R(rel_end, R0, S_REL),
            "retreat_handle": self.cmd_R(ret_end, R0, S_REL),
            "reach_caddy": self.cmd_R(hook_c + [0, 0, 0.11], R0, 1.0),
            "preshape_caddy": self.cmd_R(hook_c + [0, 0, 0.08], R0, S_PRE_C),
            "descend_caddy": self.cmd_R(hook_c, R0, S_PRE_C),
            "grasp_caddy": self.cmd_R(hook_c, R0, 1.0),
            "lift_caddy": self.cmd_R(np.r_[hook_c[:2], lift_z], R0, 1.0),
            "carry_caddy": self.cmd_R(goal_site, R0, 1.0),
            "lower_caddy": self.cmd_R(place, R0, 1.0),
            "release_caddy": self.cmd_R(place - SLIDE_OUT * R0[:, 1], R0, S_REL_C),
            "retreat": self.cmd_R(place - SLIDE_OUT * R0[:, 1] + [0, 0, 0.08], R0, S_REL_C),
        }
        home = self.cmd_R(self.ready_hand_pose()[0], R0, 0.0)
        prev = home
        deltas = {}
        for name in self.nominal_plan():
            deltas[name] = self.space.norm(nominal[name] - prev) + 0.30
            prev = nominal[name]
        spec = [
            ("reach_handle", t_reach_h, o_reach_h, None, 10.0, 2.5),
            ("above_handle", t_above_h, o_above_h, None, 5.0, 1.2),
            ("hook_handle", t_hook_h, o_hook_h, None, 5.0, 1.5),
            ("grasp_handle", t_grasp_h, o_grasp_h, e_hold, 3.0, 1.2),
            ("pull_drawer", t_pull, o_pull, e_pull, 8.0, 2.5),
            ("release_handle", t_rel_h, o_rel_h, e_rel, 3.0, 0.8),
            ("retreat_handle", t_retreat_h, o_retreat_h, e_retreat_h, 5.0, 1.2),
            ("reach_caddy", t_reach_c, o_reach_c, None, 8.0, 2.0),
            ("preshape_caddy", t_pre_c, o_pre_c, None, 4.0, 1.0),
            ("descend_caddy", t_desc_c, o_desc_c, None, 5.0, 1.5),
            ("grasp_caddy", t_grasp_c, o_grasp_c, e_hold, 3.0, 1.2),
            ("lift_caddy", t_lift, o_lift, e_lift, 5.0, 1.5),
            ("carry_caddy", t_carry, o_carry, e_carry, 10.0, 3.0),
            ("lower_caddy", t_lower, o_lower, e_lower, 6.0, 2.0),
            ("release_caddy", t_rel_c, o_rel_c, e_hold, 4.0, 1.2),
            ("retreat", t_retreat, o_retreat, e_retreat, 5.0, 1.2),
        ]
        for name, tgt, out, enter, timeout, dur in spec:
            S[name] = Skill(name, nominal[name], deltas[name], tgt, out, timeout=timeout,
                            nominal_duration=dur, on_enter=enter)
        self._skills = S

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        return ["reach_handle", "above_handle", "hook_handle", "grasp_handle", "pull_drawer",
                "release_handle", "retreat_handle", "reach_caddy", "preshape_caddy", "descend_caddy",
                "grasp_caddy", "lift_caddy", "carry_caddy", "lower_caddy", "release_caddy", "retreat"]

    def skill_graph(self) -> list:
        R = "released"
        OR = {"opened", "released"}
        O = "opened"
        OG = {"opened", "grasped"}
        OP = {"opened", "placed"}
        OPR = {"opened", "placed", "released"}
        G = "grasped"
        P = {"placed"}
        PR = {"placed", "released"}
        closed_again = OutcomeSpec("closed_again", "closed", word(OR, R), failure=True)
        return [
            SkillSpec("reach_handle", "closed", (OutcomeSpec(OK, "pre_handle", word(R)),)),
            SkillSpec("above_handle", "pre_handle", (
                OutcomeSpec(OK, "above_handle", word(R)),
                OutcomeSpec("already_open", "open", word(R, OR), failure=True))),
            SkillSpec("hook_handle", "above_handle", (
                OutcomeSpec(OK, "at_handle", word(R)),
                OutcomeSpec("already_open", "open", word(R, OR), failure=True))),
            SkillSpec("grasp_handle", "at_handle", (
                OutcomeSpec(OK, "h_grasped", word(R)),
                OutcomeSpec("missed", "closed", word(R), failure=True))),
            SkillSpec("pull_drawer", "h_grasped", (
                OutcomeSpec(OK, "h_open", word(R, OR)),
                OutcomeSpec("slipped", "closed", word(R), failure=True),
                OutcomeSpec("slipped_open", "at_handle_open", word(R, OR), failure=True),
                OutcomeSpec("slipped_open", "open", word(R, OR), failure=True))),
            SkillSpec("release_handle", "h_open", (OutcomeSpec(OK, "at_handle_open", word(OR)),)),
            SkillSpec("retreat_handle", "at_handle_open", (
                OutcomeSpec(OK, "open", word(OR)),
                OutcomeSpec(OK, "closed", word(OR, R), failure=True))),
            SkillSpec("reach_caddy", "open", (
                OutcomeSpec(OK, "over_caddy", word(OR)),
                OutcomeSpec(OK, "over_caddy", word(R)),
                closed_again)),
            SkillSpec("preshape_caddy", "over_caddy", (
                OutcomeSpec(OK, "preshaped", word(OR)),
                OutcomeSpec(OK, "preshaped", word(R)),
                closed_again)),
            SkillSpec("descend_caddy", "preshaped", (
                OutcomeSpec(OK, "at_caddy", word(OR)),
                OutcomeSpec(OK, "at_caddy", word(R)),
                closed_again)),
            SkillSpec("grasp_caddy", "at_caddy", (
                OutcomeSpec(OK, "c_grasped", word(OR, O, OG)),
                OutcomeSpec(OK, "c_grasped", word(R, "", G)),
                OutcomeSpec("missed", "open", word(OR), failure=True),
                OutcomeSpec("missed", "open", word(R), failure=True))),
            SkillSpec("lift_caddy", "c_grasped", (
                OutcomeSpec(OK, "lifted", word(OG)),
                OutcomeSpec(OK, "lifted", word(G)),
                OutcomeSpec("dropped", "open", word(OG, OR), failure=True),
                OutcomeSpec("dropped", "open", word(OG, O, OR), failure=True))),
            SkillSpec("carry_caddy", "lifted", (
                OutcomeSpec(OK, "over_goal", word(OG)),
                OutcomeSpec(OK, "over_goal", word(G)),
                OutcomeSpec("dropped", "open", word(OG, OR), failure=True),
                OutcomeSpec("dropped", "open", word(G, R), failure=True))),
            SkillSpec("lower_caddy", "over_goal", (
                OutcomeSpec(OK, "placed_held", word(OG, OP)),
                OutcomeSpec(OK, "placed_held", word(G, P)),
                OutcomeSpec(OK, "on_goal", word(OG, OP, OPR)),       # hook unloaded on touchdown
                OutcomeSpec(OK, "on_goal", word(G, P, PR)),
                OutcomeSpec(OK, "touchdown", word(OG, O)),           # resting on an edge (tilted grip)
                OutcomeSpec(OK, "touchdown", word(G, "")),
                OutcomeSpec("dropped", "open", word(OG, OR), failure=True))),
            SkillSpec("release_caddy", "placed_held", (
                OutcomeSpec(OK, "on_goal", word(OP, OPR)),
                OutcomeSpec(OK, "on_goal", word(P, PR)),
                OutcomeSpec("dropped", "open", word(OP, OPR, OR), failure=True))),
            SkillSpec("release_caddy", "touchdown", (
                OutcomeSpec(OK, "on_goal", word(O, OR, OPR)),
                OutcomeSpec(OK, "on_goal", word("", R, PR)),
                OutcomeSpec("dropped", "open", word(O, OR), failure=True))),
            SkillSpec("retreat", "on_goal", (
                OutcomeSpec(OK, "done", word(OPR)),
                OutcomeSpec(OK, "done", word(PR)))),
        ]

    # -------------------------------------------------------------- reset
    def reset(self) -> None:
        super().reset()
        self._hist = []
        self.max_opening = 0.0
        self._handle_free_since = None

    # ------------------------------------------------------------- stress test
    def disturb(self) -> None:
        """(1) Lateral torso push 35-50 N for 0.2 s (sign random) once the drawer has been pulled
        4 cm open (the arm is loaded by the drawer); (2) drawer pushed back in: after the hand has
        released the handle (no hand-handle contact for 0.2 s, grasp site > 5 cm from the bar,
        drawer open, caddy inside), a 15 N push along the closing axis acts on the drawer until
        the measured displacement plus its predicted viscous/Coulomb coast reaches 6-9 cm (per
        seed), so the caddy is inside a partially closed drawer that must be re-opened.
        Triggered by measured state with per-seed delays; identical for every method."""
        d, t = self.data, self.data.time
        P = self._pert
        d.xfrc_applied[self.drawer_bid, :3] = 0.0
        op = self.opening()
        f = self.forces()
        # ---- (1) torso push
        if P["torso_push"] is None and op > 0.04 and f["hand_handle"] > 0.5:
            P["torso_push"] = {"t_trigger": t, "t_start": t + self._push_delay, "duration": 0.2,
                               "force_N": [0.0, self._push_sign * self._push_force, 0.0]}
        push = P["torso_push"]
        active = push is not None and push["t_start"] <= t < push["t_start"] + push["duration"]
        self.torso_push(active, push["force_N"] if push else (0, 0, 0))
        if push is not None and push["t_start"] <= t <= push["t_start"] + 2.0:
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
        # ---- (2) drawer pushed back after the release
        gp = self.g1.grasp_pos(self.hand)
        if P["drawer_push"] is None:
            free = f["hand_handle"] < 0.1 and np.linalg.norm(gp - self.handle_pos()) > 0.05
            if free and op >= OPEN_MIN and self.in_drawer(self.caddy_state()["pos"]):
                self._handle_free_since = self._handle_free_since or t
                if t - self._handle_free_since > 0.2:
                    P["drawer_push"] = {"t_trigger": t, "t_start": t + self._dpush_delay, "force_N": 15.0,
                                        "target_m": float(self._dpush_dist), "q_start": None}
            else:
                self._handle_free_since = None
        elif P["drawer_push"].get("done") is None:
            p = P["drawer_push"]
            if t >= p["t_start"]:
                if p["q_start"] is None:
                    p["q_start"] = op
                moved = p["q_start"] - op
                v0 = max(-float(d.qvel[self.jd_drawer]), 0.0)
                m_eff = self.drawer_mass + self.caddy_mass + 0.02
                c, fc = self.drawer_damping, self.drawer_fric
                # exact free-coast distance of m x'' = -c x' - fc (viscous + Coulomb runner friction)
                coast = (m_eff / c) * v0 - (fc * m_eff / c ** 2) * np.log1p(c * v0 / fc)
                if moved + coast < p["target_m"] and t < p["t_start"] + 1.0:
                    d.xfrc_applied[self.drawer_bid, :3] = -p["force_N"] * self.n_out
                else:
                    p["done"] = True
                    p["t_end"] = t
                    p["moved_m"] = float(moved)
        elif P["drawer_push"].get("q_final") is None and t > P["drawer_push"]["t_end"] + 1.0:
            P["drawer_push"]["q_final"] = op

    def perturbation_info(self) -> dict:
        return self._pert

    # ---------------------------------------------------------- diagnostics
    def step_diagnostics(self) -> dict:
        f = self.forces()
        self.max_opening = max(self.max_opening, self.opening())
        hd = self.g1.hand[self.hand][1]
        out = {"body_table_force_N": f["body_table"], "body_cabinet_force_N": f["body_carcass"] + f["body_drawer"],
               "hand_table_force_N": f["hand_table"], "hand_carcass_force_N": f["hand_carcass"],
               "hand_drawer_force_N": f["hand_drawer"], "hand_handle_force_N": f["hand_handle"],
               "hand_caddy_force_N": f["hand_caddy"], "drawer_opening_m": self.opening(),
               "finger_contact_torque_Nm": float(np.max(np.abs(self.data.qfrc_constraint[hd])))}
        out.update(self.balance_diagnostics())
        return out

    def log_signals(self) -> dict:
        c = self.caddy_state()
        bal = self.g1.balance_state()
        return {"caddy": c["pos"], "caddy_tilt": c["tilt"], "grasp": self.g1.grasp_pos(self.hand),
                "closure": self.closure(), "drawer_opening": self.opening(), "handle": self.handle_pos(),
                "com": bal["com"], "pelvis": self.data.xpos[self.g1.pelvis].copy()}

    def episode_metrics(self) -> dict:
        c = self.caddy_state()
        return {"final_caddy_goal_xy_error_m": float(np.linalg.norm(c["pos"][:2] - self.goal)),
                "final_caddy_height_m": float(c["bottom"] - self.table_h),
                "final_caddy_tilt_deg": float(np.degrees(c["tilt"])),
                "final_drawer_opening_m": self.opening(), "max_drawer_opening_m": self.max_opening}

    def variant_info(self) -> dict:
        return {"table_height_m": self.table_h, "cabinet_front_x": self.front_x, "cabinet_y": self.cab_y,
                "cabinet_yaw": self.cab_yaw, "drawer_mass_kg": self.drawer_mass,
                "drawer_damping_Ns_per_m": self.drawer_damping, "drawer_frictionloss_N": self.drawer_fric,
                "caddy_init": self.obj0, "caddy_mass_kg": self.caddy_mass, "caddy_friction": self.caddy_mu,
                "goal_xy": self.goal, "perception_bias_handle_m": self.est_handle.b_far,
                "perception_bias_caddy_m": self.est_caddy.b_far}

    def overlay_lines(self) -> list:
        g = self._geometry()
        state = ("in hand" if g["held"] else "placed" if g["placed"] else "on table" if g["on_table"]
                 else "in drawer" if self.in_drawer(g["pos"]) else "-")
        lines = super().overlay_lines()
        lines.append(f"drawer opening: {g['opening'] * 100:5.1f} cm   caddy: {state}   "
                     f"caddy-goal: {np.linalg.norm(g['pos'][:2] - self.goal) * 100:5.1f} cm")
        return lines
