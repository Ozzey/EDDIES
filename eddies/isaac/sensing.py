"""Contact sensing from PhysX for the unchanged task code.

The MuJoCo tasks read contact forces in four places; in Isaac all of them are
redirected (at runtime, no source change) to the PhysX contact reports of the current
physics step, collected by ``PhysXBridge.read_contacts``:

* ``ContactMonitor.measure``          grasp / support / collision labels (all tasks),
* ``contacts.geom_contact_force``     one-off pair forces,
* ``FrankaPegInsertionTask._wrist_force_raw``   wrist F/T reading (normal + friction),
* ``G1WholeBody.floor_contacts``      foot forces and centre of pressure.

Each PhysX contact pair carries the MuJoCo geom ids of its two colliders (collider
prims are mapped to geoms by ``PhysXBridge``), so the geom masks of the task code are
used unchanged.  Forces are impulses / dt (PhysX convention: force on collider 0, normal
pointing from collider 1 to collider 0).
"""
from __future__ import annotations

import numpy as np

_PATCHED: dict = {}


def _sel(mask, g, b, bridge):
    """Geom g (or, if unmapped, any geom of body b) is in the boolean geom mask."""
    if g >= 0:
        return bool(mask[g])
    return bool(np.any(mask[bridge.body_geoms.get(b, [])]))


def install(bridge) -> None:
    """(Re)point every contact reader at ``bridge`` (one bridge per episode)."""
    _PATCHED["bridge"] = bridge
    if _PATCHED.get("done"):
        return
    _PATCHED["done"] = True
    from eddies.envs.framework import contacts as C

    def br():
        return _PATCHED["bridge"]

    # ---------------------------------------------------------- ContactMonitor
    def measure(cm, data):
        b = br()
        out = {}
        for name, (ga, gb) in cm.pairs.items():
            ma, mb = cm.masks[ga], cm.masks[gb]
            tot, n = 0.0, 0
            for rec in b.contact_records:
                g0, g1, b0, b1 = rec["g0"], rec["g1"], rec["b0"], rec["b1"]
                if (_sel(ma, g0, b0, b) and _sel(mb, g1, b1, b)) or (_sel(mb, g0, b0, b) and _sel(ma, g1, b1, b)):
                    tot += rec["fn"]
                    n += rec["n"]
            out[name], out[name + "#n"] = tot, n
        return out

    C.ContactMonitor.measure = measure

    def geom_contact_force(model, data, geoms_a, geoms_b):
        b = br()
        tot = 0.0
        for rec in b.contact_records:
            g0, g1, b0, b1 = rec["g0"], rec["g1"], rec["b0"], rec["b1"]
            if (_sel(geoms_a, g0, b0, b) and _sel(geoms_b, g1, b1, b)) or \
                    (_sel(geoms_b, g0, b0, b) and _sel(geoms_a, g1, b1, b)):
                tot += rec["fn"]
        return tot

    C.geom_contact_force = geom_contact_force

    # ------------------------------------------------------- Franka wrist F/T
    try:
        from eddies.envs.tasks.franka_peg import FrankaPegInsertionTask

        def _wrist_force_raw(task):
            """Contact force acting ON hand + fingers (world frame): PhysX normal + friction."""
            b = br()
            F = np.zeros(3)
            for rec in b.contact_records:
                h0 = _sel(task.hand_mask, rec["g0"], rec["b0"], b)
                h1 = _sel(task.hand_mask, rec["g1"], rec["b1"], b)
                if h0 and not h1:
                    F += rec["f0"]
                elif h1 and not h0:
                    F -= rec["f0"]
            return F

        FrankaPegInsertionTask._wrist_force_raw = _wrist_force_raw
    except ImportError:
        pass

    # ------------------------------------------------------------- G1 feet
    try:
        from eddies.envs.controllers.g1 import G1WholeBody

        def floor_contacts(ctrl):
            """(foot force, non-foot robot force against the floor, centre of pressure)."""
            b = br()
            if ctrl.floor_geom < 0:
                return 0.0, 0.0, None
            foot_f, other_f, cop_num = 0.0, 0.0, np.zeros(2)
            for rec in b.contact_records:
                for me, other, ob in ((rec["g0"], rec["g1"], rec["b1"]), (rec["g1"], rec["g0"], rec["b0"])):
                    if me != ctrl.floor_geom:
                        continue
                    if not _sel(ctrl._robot_mask, other, ob, b):
                        continue
                    if _sel(ctrl._foot_mask, other, ob, b):
                        foot_f += rec["fn"]
                        for pos, fn in rec["points"]:
                            cop_num += fn * np.asarray(pos)[:2]
                    else:
                        other_f += rec["fn"]
            pts = sum(fn for rec in b.contact_records for _, fn in rec["points"]
                      if ctrl.floor_geom in (rec["g0"], rec["g1"]) and
                      _sel(ctrl._foot_mask, rec["g1"] if rec["g0"] == ctrl.floor_geom else rec["g0"],
                           rec["b1"] if rec["g0"] == ctrl.floor_geom else rec["b0"], b))
            cop = cop_num / pts if pts > 1e-6 else None
            return foot_f, other_f, cop

        G1WholeBody.floor_contacts = floor_contacts
    except ImportError:
        pass
