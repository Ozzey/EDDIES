"""G1 skillbook: the primitives of the Unitree G1 (+ Dex3-1 hands) embodiment, implemented once.

Every G1 task (peg insertion, drawer, microwave; ``tasks/g1_sb.py``) is composed only of Eddies
of :data:`G1_BOOK`.  The book uses the primitive NAMES of the skillbook vocabulary
(``eddies.envs.skillbook``) -- 10 of the 11; ``turn`` is not needed -- with implementations for
the G1 embedding chart (``controllers/g1.g1_space``):

    [x, y, z]     grasp-site position (world, m)
    [rx, ry, rz]  rotation vector of the hand frame (``g1.chart``, identity = "glass" orientation)
    grasp         synergy closure in [0, 1] (the task's decoder maps it to a finger posture)
    extra         task-declared extra dims (the microwave's power/hook ``mode``)

A hand SHAPE is an argument of every primitive: ``closure`` (commanded synergy closure) and
``extra`` (commanded extra dims), e.g. the drawer's pre-curled hook is ``reach`` with
``closure=0.4`` and the microwave's fist is ``reach`` with ``closure=1, extra=(0,)``; shape
changes are therefore never primitives of their own.  ``closure_key`` names the MEASURED
closure in the observation that a primitive's outcome test reads (``closure`` / ``hook_closure``).

Frames are callables ``(obs, st) -> point`` or ``(obs, st) -> (point, R)`` (``pose``) supplied by
the task; ``orient`` is a callable ``(obs, st) -> R`` (None: keep the hand rotation measured at
entry).  Object knowledge comes through a :class:`G1View`.  Outcomes are decided from measured
state.  Each ``build_*`` returns ``(target, outcome, on_enter)`` for :meth:`SkillBook.eddy`.

The Franka builders of :mod:`primitives` read the Franka observation / command interface (TCP,
yaw, aperture, force limit) and are therefore not reused; the structure of each primitive
(hold-at-entry, measured success, failure keys, RLS line identification of ``pull``) follows them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from eddies.core.online import RecursiveLeastSquares
from eddies.envs.controllers.ik import rotation_error_mat
from eddies.envs.framework.task import OK
from eddies.envs.skillbook.book import Primitive, SkillBook

G1_BOOK = SkillBook(version="g1-1.0")


# --------------------------------------------------------------------------- object view
@dataclass
class G1View:
    """What a G1 primitive may know about the manipulated object (supplied by the task).

    held(obs): measured grasp; touch(obs): hand-object normal force [N]; ref(obs): measured object
    point used for drop detection (the object counts as in the hand while it is within ``near_r``
    of the grasp site or touched); free(obs): no hand-object contact (default: not held)."""
    held: Callable
    touch: Callable | None = None
    ref: Callable | None = None
    near_r: float = 0.06
    free: Callable | None = None

    def is_free(self, obs) -> bool:
        return bool(self.free(obs)) if self.free is not None else not self.held(obs)


def dropped(view: G1View, obs, st, after: float = 0.1) -> bool:
    """The object measurably left the hand (no contact AND away from the grasp site) for ``after`` s."""
    inhand = view.touch(obs) > 0.3 or np.linalg.norm(view.ref(obs) - obs["grasp_pos"]) < view.near_r
    if inhand:
        st.memory["lost"] = None
        return False
    st.memory["lost"] = st.memory.get("lost") or obs["t"]
    return obs["t"] - st.memory["lost"] > after


# --------------------------------------------------------------------------- helpers
def g1_cmd(task, pos, R, closure, extra=()) -> np.ndarray:
    """Normalized G1 command: grasp-site position, hand rotation (matrix -> chart), closure, extra."""
    rv = task.g1.chart.encode_matrix(np.asarray(R, float))
    return task.space.normalize(np.r_[np.asarray(pos, float), rv, float(closure), list(extra)])


def _near(obs, p, tol) -> bool:
    return bool(np.linalg.norm(obs["grasp_pos"] - np.asarray(p, float)) < tol)


def _rot_err(obs, R) -> float:
    return float(np.linalg.norm(rotation_error_mat(obs["grasp_mat"], R)))


def _enter_hold(orient=None):
    """on_enter: hold the measured grasp-site position; orientation = orient(obs, st) or measured."""
    def on_enter(obs, st):
        st.memory["hold"] = obs["grasp_pos"].copy()
        st.memory["R"] = (np.array(orient(obs, st), float) if orient is not None
                          else np.array(obs["grasp_mat"], float).copy())
        st.memory["lost"] = None
    return on_enter


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


def _upright(obs, st, axis, ema=0.85):
    """In-hand tilt compensation: hand rotation that turns the held object's axis (wrist-camera
    estimate ``axis(obs)``, low-passed in the hand frame) onto world z; returns (R_des, R_rel)."""
    R = obs["grasp_mat"]
    a_h = R.T @ axis(obs)
    prev = st.memory.get("a_h")
    a_h = a_h if prev is None else ema * prev + (1 - ema) * a_h
    a_h = a_h / np.linalg.norm(a_h)
    st.memory["a_h"] = a_h
    R_des = _min_rotation(a_h, np.array([0.0, 0.0, 1.0]))
    return R_des, R_des @ R.T


def _offset(obs, st, ref, upright=None, smooth=None, update=True):
    """Object reference point relative to the grasp site (in the commanded hand orientation when
    ``upright``).  smooth(obs): the xy offset is low-passed (0.9 / 0.1) and refreshed only while
    smooth(obs) holds (the hand does not chase a swinging object); z is always current."""
    R_des = None
    off = ref(obs) - obs["grasp_pos"]
    if upright is not None:
        if update or "R_up" not in st.memory:
            R_des, R_rel = _upright(obs, st, upright)
            st.memory["R_up"], st.memory["R_rel"] = R_des, R_rel
        R_des, R_rel = st.memory["R_up"], st.memory["R_rel"]
        off = R_rel @ off
    if smooth is not None:
        if "off" not in st.memory:
            st.memory["off"] = off[:2].copy()
        elif update and smooth(obs):
            st.memory["off"] = 0.9 * st.memory["off"] + 0.1 * off[:2]
        off = np.r_[st.memory["off"], off[2]]
    return off, R_des


def _orient(st, obs, orient, R_des=None):
    if R_des is not None:
        return R_des
    return orient(obs, st) if orient is not None else st.memory["R"]


def _goal(goal, obs, st, search):
    g = np.array(goal(obs, st), float)
    if search is not None:
        g[:2] = g[:2] + search()
    return g


# --------------------------------------------------------------------------- 1 reach
def build_reach(task, *, pose, closure=0.0, extra=(), tol=0.008, rot_tol=None, closure_key="closure",
                closure_max=None, closure_min=None, require=None, abort=None, fail_after=None,
                fail_if=None, fail_key="blocked", open_first=None, capture=None, approach=None):
    """Move the grasp site to ``pose(obs, st) = (p, R)`` with the hand shape (closure, extra).

    Success: within ``tol`` (and ``rot_tol``), measured closure below ``closure_max`` / above
    ``closure_min``, ``require(obs)``.  abort(obs) -> outcome key checked first; ``fail_key``
    when ``fail_if(obs)`` after ``fail_after`` s.  Motion options:
      open_first: the measured closure exceeds the commanded one by more than this -> change the
                  hand shape in place first (a closed hook would catch the object);
      capture=(xy_tol, hold): guarded descent -- while the hand is off the target column by more
                  than xy_tol or hold(obs), keep the height max(current, target);
      approach=(length, lead): straight-line approach along the commanded fingers axis R[:, 0],
                  ending at p; carrot ``lead`` ahead of the measured hand's projection."""
    def target(obs, st):
        p, R = pose(obs, st)
        gp = obs["grasp_pos"]
        if open_first is not None and obs[closure_key] > closure + open_first:
            return g1_cmd(task, gp, R, closure, extra)
        if capture is not None:
            xy_tol, hold = capture
            if (hold is not None and hold(obs)) or np.linalg.norm(gp[:2] - p[:2]) > xy_tol:
                return g1_cmd(task, np.r_[p[:2], max(gp[2], p[2])], R, closure, extra)
        if approach is not None:
            length, lead = approach
            a = np.asarray(R, float)[:, 0]
            s = float(np.clip((gp - (p - length * a)) @ a + lead, 0.0, length))
            p = p - (length - s) * a
        return g1_cmd(task, p, R, closure, extra)

    def outcome(obs, labels, st):
        if abort is not None:
            a = abort(obs)
            if a:
                return a
        p, R = pose(obs, st)
        ok = (_near(obs, p, tol) and (closure_max is None or obs[closure_key] < closure_max)
              and (closure_min is None or obs[closure_key] > closure_min)
              and (rot_tol is None or _rot_err(obs, R) < rot_tol) and (require is None or require(obs)))
        if ok:
            return OK
        if fail_after is not None and st.elapsed(obs["t"]) > fail_after and fail_if(obs):
            return fail_key
        return None
    return target, outcome, None


# --------------------------------------------------------------------------- 2 grasp
def build_grasp(task, *, view, closure=1.0, extra=(), orient=None, closure_key="closure", held=None,
                ok_closure=0.5, settle=0.7, miss_closure=0.85, miss_after=0.7, tug=None, tug_after=0.9,
                tug_dwell=None):
    """Close the hand (to ``closure`` / ``extra``) at the entry pose; success is the MEASURED grasp
    ``held(obs)`` (default ``view.held``) with measured closure > ok_closure after ``settle`` s.

    tug=(direction(obs) -> unit vector, distance): once the measured closure exceeds
    ``tug_after`` the hand pulls ``distance`` along the direction; 'missed' when the hand has
    moved (distance - 4 mm) without a grasp (for ``tug_dwell`` s if given).  Without a tug:
    'missed' when the closure exceeds ``miss_closure`` after ``miss_after`` s and the view is free."""
    held = held or view.held

    def on_enter(obs, st):
        _enter_hold(orient)(obs, st)
        if tug is not None:
            st.memory["tug"] = np.asarray(tug[0](obs), float)

    def target(obs, st):
        p = st.memory["hold"]
        if tug is not None and obs[closure_key] > tug_after:
            p = p + tug[1] * st.memory["tug"]
        return g1_cmd(task, p, st.memory["R"], closure, extra)

    def outcome(obs, labels, st):
        if held(obs) and (ok_closure is None or obs[closure_key] > ok_closure) and st.elapsed(obs["t"]) > settle:
            return OK
        if tug is not None:
            moved = (obs["grasp_pos"] - st.memory["hold"]) @ st.memory["tug"]
            if obs[closure_key] > tug_after and moved > tug[1] - 0.004:
                if tug_dwell is None:
                    return "missed" if view.is_free(obs) else None
                st.memory["tugged"] = st.memory.get("tugged") or obs["t"]
                if obs["t"] - st.memory["tugged"] > tug_dwell and not held(obs):
                    return "missed"
            return None
        if obs[closure_key] > miss_closure and view.is_free(obs) and st.elapsed(obs["t"]) > miss_after:
            return "missed"
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 3 lift
def build_lift(task, *, view, lifted, closure=1.0, extra=(), orient=None, bottom=None, goal=None, rise=None,
               latch=False, tol=None):
    """Lift the held object straight up from the entry pose.

    Hand height = max(entry + ``rise``, hand + goal(obs) - bottom(obs)): ``goal`` is the wanted
    height of the object's lowest point ``bottom`` (from the MEASURED hand-object offset;
    ``latch``: evaluated once).  Success: lifted(obs) (and within ``tol`` of the target)."""
    def z_goal(obs, st):
        zs = []
        if rise is not None:
            zs.append(st.memory["hold"][2] + rise)
        if goal is not None:
            # (evaluation order as in the hand-written G1 skills: bit-identical episodes)
            if latch:
                if "zgoal" not in st.memory:
                    st.memory["zgoal"] = float(obs["grasp_pos"][2] + (goal(obs) - bottom(obs)))
                zs.append(st.memory["zgoal"])
            else:
                zs.append(float(goal(obs) + (obs["grasp_pos"][2] - bottom(obs))))
        return max(zs)

    def target(obs, st):
        return g1_cmd(task, np.r_[st.memory["hold"][:2], z_goal(obs, st)], st.memory["R"], closure, extra)

    def outcome(obs, labels, st):
        if dropped(view, obs, st):
            return "dropped"
        ok = lifted(obs) and (tol is None or _near(obs, np.r_[st.memory["hold"][:2], z_goal(obs, st)], tol))
        return OK if ok else None
    return target, outcome, _enter_hold(orient)


# --------------------------------------------------------------------------- 4 transport
def build_transport(task, *, view, goal=None, ref=None, height=None, rise=0.0, shift=None, closure=1.0, extra=(),
                    orient=None, upright=None, search=None, smooth=None, meas=None, tol_xy=None, tol_z=None,
                    tol=None, still=None, done=None):
    """Carry the held object.

    goal(obs, st): goal of the object's reference point ``ref(obs)`` (measured, offset to the hand
    re-measured every step; ``smooth``: low-passed while smooth(obs)); ``height``: the reference
    point ends ``height`` above goal (None: travel at the constant hand height entry + ``rise``).
    shift(obs): instead, move the hand by this displacement from the entry pose (latched).
    upright(obs): object axis estimate -- the hand turns so that the held object is vertical.
    search(): xy offset added to the goal (recovery search).  Success: ``meas(obs)`` (default ref)
    within tol_xy of the goal (xy) and tol_z of goal + height (z), hand within ``tol`` of its
    target, still(obs), done(obs).  Fails 'dropped' (measured)."""
    hold = _enter_hold(orient)

    def on_enter(obs, st):
        hold(obs, st)
        st.memory["z"] = float(obs["grasp_pos"][2]) + rise
        if shift is not None:
            st.memory["target"] = obs["grasp_pos"] + np.asarray(shift(obs), float)
        if ref is not None and smooth is not None:
            _offset(obs, st, ref, upright, smooth)

    def hand_target(obs, st, update):
        if shift is not None:
            return st.memory["target"], st.memory["R"]
        off, R_des = _offset(obs, st, ref, upright, smooth, update)
        g = _goal(goal, obs, st, search)
        z = st.memory["z"] if height is None else g[2] + height - off[2]
        return np.r_[g[:2] - off[:2], z], _orient(st, obs, orient, R_des)

    def target(obs, st):
        p, R = hand_target(obs, st, True)
        return g1_cmd(task, p, R, closure, extra)

    def outcome(obs, labels, st):
        if dropped(view, obs, st):
            return "dropped"
        ok = True
        if tol_xy is not None or tol_z is not None:
            m = (meas or ref)(obs)
            g = _goal(goal, obs, st, search)
            if tol_xy is not None:
                ok &= bool(np.linalg.norm(m[:2] - g[:2]) < tol_xy)
            if tol_z is not None:
                ok &= bool(abs(m[2] - (g[2] + height)) < tol_z)
        if ok and tol is not None:
            ok &= _near(obs, hand_target(obs, st, False)[0], tol)
        ok = ok and (still is None or still(obs)) and (done is None or done(obs))
        return OK if ok else None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 5 place
def build_place(task, *, view, goal, ref, placed, below=0.0, closure=1.0, extra=(), orient=None, smooth=None,
                zone=None, speed=None, settle=0.3, still=None):
    """Lower the held object so that its lowest point ``ref(obs)`` goes ``below`` under the support
    point ``goal(obs, st)`` (xy: goal, offset as in transport); guarded touchdown at ``speed`` m/s
    once ``ref`` is within ``zone`` of the support.  Success: placed(obs) after ``settle`` s, still(obs)."""
    hold = _enter_hold(orient)

    def on_enter(obs, st):
        hold(obs, st)
        st.memory["tz0"] = None
        if smooth is not None:
            _offset(obs, st, ref, None, smooth)

    def target(obs, st):
        off, _ = _offset(obs, st, ref, None, smooth, True)
        g = np.asarray(goal(obs, st), float)
        p = np.r_[g[:2] - off[:2], g[2] - below - off[2]]
        if zone is not None:
            if st.memory.get("tz0") is None and ref(obs)[2] - g[2] < zone:
                st.memory["tz0"], st.memory["tt0"] = float(obs["grasp_pos"][2]), float(obs["t"])
            if st.memory.get("tz0") is not None:
                p[2] = max(p[2], st.memory["tz0"] - speed * (obs["t"] - st.memory["tt0"]))
        return g1_cmd(task, p, _orient(st, obs, orient), closure, extra)

    def outcome(obs, labels, st):
        if dropped(view, obs, st):
            return "dropped"
        ok = placed(obs) and st.elapsed(obs["t"]) > settle and (still is None or still(obs))
        return OK if ok else None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 6 release
def build_release(task, *, view, closure=0.0, extra=(), lift=(0.0, 0.0, 0.0), closure_key="closure",
                  open_tol=0.12, settle=0.3, still=None, slide=None, slide_margin=0.15, slide_delay=0.5,
                  slide_calm=None, verify=None, fail_key="dropped"):
    """Open the hand (to ``closure`` / ``extra``) in place (entry pose + ``lift``).

    Success: the view is free, measured closure < closure + open_tol, after ``settle`` s, still(obs).
    slide=(hand-frame direction, distance): once opened (closure < closure + slide_margin, after
    slide_delay s, slide_calm(obs)) the hand slides out along the direction; success then also
    needs the slide done, and verify(obs) (else ``fail_key``)."""
    def target(obs, st):
        p = st.memory["hold"] + np.asarray(lift, float)
        if slide is not None:
            if st.memory.get("slide") or (obs[closure_key] < closure + slide_margin and st.elapsed(obs["t"]) > slide_delay
                                          and (slide_calm is None or slide_calm(obs))):
                st.memory["slide"] = True
                p = p + slide[1] * (st.memory["R"] @ np.asarray(slide[0], float))
        return g1_cmd(task, p, st.memory["R"], closure, extra)

    def outcome(obs, labels, st):
        if slide is not None:
            moved = (obs["grasp_pos"] - st.memory["hold"]) @ (st.memory["R"] @ np.asarray(slide[0], float))
            if view.is_free(obs) and moved > slide[1] - 0.02 and (still is None or still(obs)):
                return OK if (verify is None or verify(obs)) else fail_key
            return None
        ok = (view.is_free(obs) and obs[closure_key] < closure + open_tol
              and (settle is None or st.elapsed(obs["t"]) > settle) and (still is None or still(obs)))
        return OK if ok else None
    return target, outcome, _enter_hold()


# --------------------------------------------------------------------------- 7 insert
def build_insert(task, *, view, goal, ref, depth, seat, ok_depth, upright=None, search=None, on_jam=None,
                 closure=1.0, extra=(), speed=0.035, max_lead=0.012, ok_time=4.0, stall_start=1.5,
                 stall_time=1.0, progress=0.002):
    """Insert the held object into the hole at ``goal(obs, st)`` (entrance; + ``search()`` xy offset):
    rate-limited descent (``speed`` m/s, never more than ``max_lead`` below the measured hand: a
    bounded push force at the chamfer or in a jam) to ``seat`` below the entrance, the object kept
    vertical (``upright``) and re-aligned from the estimates.  Success: depth(obs) >= seat - 3 mm,
    or >= ok_depth after ok_time s; 'jammed' when the depth gained < ``progress`` for stall_time s
    (after stall_start s; ``on_jam()`` then advances the task's search schedule)."""
    def on_enter(obs, st):
        st.memory["z0"] = float(obs["grasp_pos"][2])
        st.memory["best"] = depth(obs)
        st.memory["t_best"] = obs["t"]
        st.memory["lost"] = None

    def target(obs, st):
        off, R_des = _offset(obs, st, ref, upright, None, True)
        g = _goal(goal, obs, st, search)
        p = np.r_[g[:2] - off[:2], g[2] - seat - off[2]]
        z = max(p[2], st.memory["z0"] - speed * st.elapsed(obs["t"]), obs["grasp_pos"][2] - max_lead)
        R = R_des if R_des is not None else obs["grasp_mat"]
        return g1_cmd(task, np.r_[p[:2], z], R, closure, extra)

    def outcome(obs, labels, st):
        if dropped(view, obs, st):
            return "dropped"
        d = depth(obs)
        if d >= seat - 0.003 or (d >= ok_depth and st.elapsed(obs["t"]) > ok_time):
            return OK
        if d > st.memory["best"] + progress:
            st.memory["best"], st.memory["t_best"] = d, obs["t"]
        if st.elapsed(obs["t"]) > stall_start and obs["t"] - st.memory["t_best"] > stall_time:
            if on_jam is not None:
                on_jam()
            return "jammed"
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 8 push
def build_push(task, *, frame, q_est, q_meas, goal, done, lead, closure=1.0, extra=(), stall_progress=np.radians(1.0),
               stall_start=3.0, stall_time=2.0):
    """Push an articulation along its path with the closed hand (contact only, no grasp).

    frame(q, obs, st) -> (p, R): contact pose of the hand at path coordinate q; the target leads
    the estimated coordinate q_est(obs, st) by lead(q) (capped at ``goal``).  Success: done(obs);
    'stalled' when the MEASURED coordinate q_meas(obs) gains < stall_progress for stall_time s."""
    def target(obs, st):
        q = q_est(obs, st)
        p, R = frame(min(q + lead(q), goal), obs, st)
        return g1_cmd(task, p, R, closure, extra)

    def outcome(obs, labels, st):
        if done(obs):
            return OK
        q, best = q_meas(obs), st.memory.get("best", -np.inf)
        if q > best + stall_progress:
            st.memory["best"], st.memory["t_best"] = q, obs["t"]
        elif st.elapsed(obs["t"]) > stall_start and obs["t"] - st.memory.get("t_best", obs["t"]) > stall_time:
            return "stalled"
        return None
    return target, outcome, None


# --------------------------------------------------------------------------- 9 pull
def build_pull(task, *, engaged, done, slip, held=None, gone=None, slip_time=0.15, closure=1.0, extra=(),
               axis=None, progress=None, goal=None, arc=None):
    """Pull a grasped articulation (hand shape held closed) until done(obs).

    Line model (``axis``, ``progress``, ``goal``): pull along the estimated unit axis(obs) from the
    entry pose by goal - progress(obs at entry); with the method's online mode the line
    p(q) = p0 + q n is identified by recursive least squares from the measured grasp site and q.
    Arc model (``arc`` dict: center(obs, st), angle(c, obs) -> q, point(obs) observed grasp point,
    frame(point, q) -> (p, R), lead, goal, sign, prior(), on_fit(c), min_radius): the observed
    grasp point is rotated about the vertical axis through the centre by ``lead`` (target capped
    at goal); online, the centre is identified by an RLS circle fit of the observed grasp points
    (prior: the nominal model) and reported with ``on_fit``.
    The grasp is lost when not engaged(obs) for slip_time s and gone(obs): slip(obs) -> outcome."""
    held = held or engaged

    def on_enter(obs, st):
        st.memory["lost"] = None
        st.memory["rls"] = None
        if arc is None:
            _enter_hold()(obs, st)
            q0 = progress(obs)
            n = np.asarray(axis(obs), float)
            st.memory.update(q0=q0, n=n)
            if st.memory.get("online"):
                rls = RecursiveLeastSquares(2, 3, ridge=1.0, forgetting=0.995)
                rls.theta = np.column_stack([obs["grasp_pos"] - q0 * n, n])
                rls.P = np.diag([1e-4, 1e-2])
                st.memory["rls"] = rls
        elif st.memory.get("online"):
            c = np.asarray(arc["prior"](), float)
            rls = RecursiveLeastSquares(3, 1, ridge=1.0)
            r = float(np.linalg.norm(arc["point"](obs)[:2] - c))
            rls.theta = np.array([[c[0], c[1], r * r - c @ c]])
            rls.P = np.diag([1e-4, 1e-4, 2.5e-4]) / 1.5e-6
            st.memory["rls"] = rls

    def target(obs, st):
        rls = st.memory["rls"]
        if arc is None:
            if rls is not None:
                if held(obs):
                    rls.update(np.array([1.0, progress(obs)]), obs["grasp_pos"])
                p = rls.predict(np.array([1.0, goal]))
            else:
                p = st.memory["hold"] + (goal - st.memory["q0"]) * st.memory["n"]
            return g1_cmd(task, p, st.memory["R"], closure, extra)
        if rls is not None and held(obs):                       # circle fit x^2 + y^2 = 2 a x + 2 b y + c
            h = arc["point"](obs)[:2]
            rls.update(np.array([2 * h[0], 2 * h[1], 1.0]), np.array([h @ h]))
            a, b, cc = rls.theta[0]
            if cc + a * a + b * b > arc.get("min_radius", 0.2) ** 2:
                arc["on_fit"](np.array([a, b]))
        c = np.asarray(arc["center"](obs, st), float)
        h = arc["point"](obs)
        q_now = arc["angle"](c, obs)
        q_t = min(q_now + arc["lead"], arc["goal"])
        da = arc.get("sign", 1.0) * (q_t - q_now)
        Rz = np.array([[np.cos(da), -np.sin(da)], [np.sin(da), np.cos(da)]])
        p, R = arc["frame"](np.r_[c[:2] + Rz @ (h[:2] - c[:2]), h[2]], q_t)
        return g1_cmd(task, p, R, closure, extra)

    def outcome(obs, labels, st):
        if engaged(obs):
            st.memory["lost"] = None
            return OK if done(obs) else None
        st.memory["lost"] = st.memory.get("lost") or obs["t"]
        if obs["t"] - st.memory["lost"] > slip_time and (gone is None or gone(obs)):
            return slip(obs)
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 10 retreat
def build_retreat(task, *, closure=0.0, extra=(), shift=(0.0, 0.0, 0.0), hand_shift=None, pose=None, latch=False,
                  closure_key="closure", closure_max=None, open_first=None, tol=None, away=None, fail_if=None,
                  fail_key="failed"):
    """Move away from the object with the hand shape (closure, extra): to the entry pose +
    ``shift`` (world) + R ``hand_shift`` (hand frame), or to the clearance frame pose(obs, st)
    (tracked; ``latch``: evaluated at entry, keeping the measured hand rotation).  open_first: as
    in reach.  Success: within ``tol`` of the target, away(obs), measured closure < closure_max;
    on arrival ``fail_if(obs)`` -> ``fail_key``."""
    hold = _enter_hold()

    def on_enter(obs, st):
        hold(obs, st)
        t = obs["grasp_pos"] + np.asarray(shift, float)
        if hand_shift is not None:
            t = t + st.memory["R"] @ np.asarray(hand_shift, float)
        if pose is not None and latch:
            t = np.asarray(pose(obs, st)[0], float)
        st.memory["target"] = t

    def goal(obs, st):
        if pose is not None and not latch:
            return pose(obs, st)
        return st.memory["target"], st.memory["R"]

    def target(obs, st):
        p, R = goal(obs, st)
        if open_first is not None and obs[closure_key] > closure + open_first:
            p = st.memory["hold"]
        return g1_cmd(task, p, R, closure, extra)

    def outcome(obs, labels, st):
        ok = ((tol is None or _near(obs, goal(obs, st)[0], tol)) and (away is None or away(obs))
              and (closure_max is None or obs[closure_key] < closure_max))
        if not ok:
            return None
        if fail_if is not None and fail_if(obs):
            return fail_key
        return OK
    return target, outcome, on_enter


TAXONOMY = "Bogh et al. 2012; MAPLE (Nasiriany et al. 2021); RAPS (Dalal et al. 2021); FMB (Luo et al. 2024)"
for _name, _fn, _outs, _desc, _src in [
    ("reach", build_reach, ("ok", "knocked", "blocked", "already_open", "closed_again", "door_closed"),
     "move the hand to a frame with a hand shape (closure / mode)", "MAPLE reach; RAPS go-to-pose; Bogh move to"),
    ("grasp", build_grasp, ("ok", "missed"), "close the hand synergy; measured contact (optional tug test)",
     "MAPLE grasp; RAPS top grasp; Bogh pick up"),
    ("lift", build_lift, ("ok", "dropped"), "lift the held object", "RAPS lift"),
    ("transport", build_transport, ("ok", "dropped"), "carry the held object to a frame (optionally upright)",
     "Bogh move to; FMB move to board"),
    ("place", build_place, ("ok", "dropped"), "lower the held object onto a support (guarded touchdown)",
     "Bogh place; RAPS drop; FMB place on fixture"),
    ("release", build_release, ("ok", "dropped"), "open the hand in place (optional slide-out)",
     "MAPLE / RAPS / Bogh release"),
    ("insert", build_insert, ("ok", "jammed", "dropped"), "rate-limited compliant insertion with jam detection",
     "Bogh align + press; FMB insert"),
    ("push", build_push, ("ok", "stalled"), "push an articulation along its path with the closed hand",
     "MAPLE / RAPS push; Bogh press"),
    ("pull", build_pull, ("ok", "slipped", "slipped_open"), "pull a grasped articulation along its (identified) line / arc",
     "RAPS pull; Bogh open"),
    ("retreat", build_retreat, ("ok", "door_closed"), "move away from the object", "reach variant"),
]:
    G1_BOOK.register(Primitive(_name, "g1-1.0", _desc, _outs, _fn, (_src,)))
