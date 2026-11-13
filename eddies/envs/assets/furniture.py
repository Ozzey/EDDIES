"""Parametric, physically articulated furniture built with mjSpec (shared asset module).

Two builders are provided; both add a STATIC carcass (a body without joints, i.e.
fixed to its parent) and ARTICULATED parts on real 1-DoF joints with viscous
``damping``, Coulomb ``frictionloss`` and hard joint-range stops.  Every part is a
rigid body with realistic mass (computed from part volumes) and finite friction;
handles are separate child bodies (rigidly attached, no joint) so contact
monitors can group them.  Nothing is kinematic, welded by an equality, or moved
by writing poses: a robot opens a drawer/door only by contact forces.

Common conventions
------------------
* ``sb`` is a :class:`eddies.envs.framework.scene.SceneBuilder`; ``name`` prefixes
  every element (bodies, joints, geoms, sites) so several instances can coexist.
* ``pos`` = world position of the centre of the carcass FOOTPRINT (bottom face,
  i.e. on the supporting surface: a table top, the floor, ...).  ``yaw`` [rad]
  rotates about world z.  In the LOCAL frame the FRONT faces +x, the carcass
  spans x in [-depth/2, depth/2], y in [-width/2, width/2], z in [0, height].
  A robot standing at world -x of the furniture therefore uses ``yaw = pi``.
* ``parent`` (optional MjsBody) attaches the furniture to another body instead
  of the world body (e.g. a counter); pos/yaw are then in that body's frame.
* "left"/"right" are as seen by a person standing IN FRONT of the furniture,
  looking at it (local -y is that person's left).
* Geom sizes in this module are full extents unless named ``*_half``.
* Returned dicts contain only plain names/floats/arrays so tasks can look up ids
  after ``sb.compile()`` (``framework.scene.jid/bid/gid/sid``).

``add_drawer_cabinet(sb, name, pos, yaw, ...)`` -> dict
    A chest of ``len(drawer_heights)`` stacked drawers (index 0 = bottom).
    Carcass: bottom, top, two sides, back panel and horizontal dividers (real
    walls with collision).  Each drawer is a hollow box (bottom + 4 walls; the
    front wall is the visible front panel) on a SLIDE joint along the local +x
    (opening) axis, range [0, travel] (hard stops: closed and fully extended),
    with damping [N s/m] and frictionloss [N].  A metal bar handle (capsule) is
    mounted on two standoffs in front of the front panel (all with collision).
    Contacts carcass<->drawer are excluded: the slide joint IS the runner/rail
    model (the drawer box has a few mm clearance to the carcass anyway).
    Returned keys::

        body            carcass body name
        carcass_geoms   list of carcass geom names
        front_normal    world unit vector of the front (opening direction)
        quat            world quaternion of the carcass frame
        size            (depth, width, height) [m]
        drawers         list (bottom->top) of dicts:
            body, joint            drawer body / slide-joint names (q = opening [m])
            handle_body            handle child body (bar + standoffs)
            handle_geom            the bar geom (capsule along local y)
            handle_site            site at the bar centre (moves with the drawer)
            interior_site          site at the centre of the interior floor
            front_geom             front panel geom
            geoms                  all drawer geoms (box walls + front)
            travel, mass           [m], [kg] (drawer box incl. handle)
            interior_half          (half depth x, half width y) of the interior [m]
            interior_height        inner wall height above the interior floor [m]
            handle_length, handle_radius, handle_standoff   [m]
            handle_closed_local    bar centre in the carcass frame when closed
            interior_floor_local   interior floor centre in the carcass frame when closed
            front_inner_local_x    carcass-frame x of the front panel's inner face (closed)
            front_top_local_z, opening_top_local_z   heights of the front panel top edge
                                   and of the carcass opening top (underside of the rail)

``add_microwave(sb, name, pos, yaw, ...)`` -> dict
    A countertop microwave oven: static shell (feet, floor, ceiling, back,
    hinge-side wall and the solid control-panel block, which forms the other
    cavity side wall), so the CAVITY is hollow and open only at the front.  The
    door (default 1.5 kg) hangs on a vertical HINGE joint at the hinge-side
    front corner (q = opening angle [rad], range [0, open_range_deg]), with
    damping [N m s/rad] and frictionloss [N m]; a vertical metal bar handle on
    two standoffs sits near the door's free edge.  An optional light closing
    spring (``detent_stiffness`` [N m/rad] with ``detent_preload`` [rad], i.e. a
    joint spring with springref = -preload) presses the closed door against its
    stop like a magnetic catch; choose it below the hinge frictionloss divided
    by the max angle if the door must stay where it is released.
    Contacts shell<->door are excluded (the hinge is the mechanism; the door has
    a 2 mm gap to the shell front).  Returned keys::

        body, shell_geoms, front_normal, quat, size     as above
        door_body, hinge_joint, door_geom, door_geoms
        handle_body, handle_geom, handle_site
        hinge_axis_world    unit axis such that a positive angle opens the door
        hinge_point_world   a point on the hinge axis (at the door bottom)
        door_width, door_thickness, door_mass, open_range [rad]
        handle_radius_from_hinge   horizontal distance hinge axis -> bar centre
        handle_angle_offset   angle [rad] of (bar centre - hinge) w.r.t. the
                              closed door plane (bar is in front of the door)
        cavity_site         site at the centre of the cavity floor
        cavity_half         (half depth x, half width y) of the cavity [m]
        cavity_height       [m]
        cavity_center_local cavity floor centre in the shell frame
        hinge_side          "left" | "right"

Realistic default parameters (documented sources: typical kitchen hardware)
---------------------------------------------------------------------------
* drawer box 1.5-3 kg (12 mm plywood walls), runner damping 5-15 N s/m,
  breakaway friction 1-4 N, travel 0.30 m; bar pull 12-14 mm diameter,
  ~35 mm projection (standoff to bar centre), 128-160 mm long;
* microwave 0.46 x 0.36 x 0.27 m outside, ~0.30 x 0.29 x 0.20 m cavity, door
  1-2 kg, hinge damping ~0.1-0.5 N m s/rad, friction ~0.1-0.3 N m.
"""
from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from eddies.envs.framework.physics import DENSITY, FRICTION
from eddies.envs.framework.scene import SceneBuilder, yaw_quat

GEOM = mujoco.mjtGeom

# capsule/cylinder axis (local z) -> local y / local x
_Q_Z_TO_Y = np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0])
_Q_Z_TO_X = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])


def _fr(f) -> list:
    if isinstance(f, str):
        return list(FRICTION[f])
    return [float(x) for x in f]


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _box(body, name, lo, hi, *, material=None, rgba=None, friction="wood", mass=None,
         collide=True, condim=3, group=0):
    """Axis-aligned box geom from its min/max corners (body frame)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    kw = dict(name=name, type=GEOM.mjGEOM_BOX, pos=list((lo + hi) / 2),
              size=list((hi - lo) / 2), friction=_fr(friction), condim=condim, group=group)
    if material:
        kw["material"] = material
    if rgba is not None:
        kw["rgba"] = list(rgba)
    if mass is not None:
        kw["mass"] = float(mass)
    if not collide:
        kw.update(contype=0, conaffinity=0, mass=0.0, group=max(group, 1))
    return body.add_geom(**kw)


def _vol(lo, hi) -> float:
    return float(np.prod(np.asarray(hi, float) - np.asarray(lo, float)))


# =============================================================== drawer cabinet
def add_drawer_cabinet(sb: SceneBuilder, name: str, pos: Sequence[float], yaw: float = 0.0, *,
                       parent=None, width: float = 0.44, depth: float = 0.40,
                       drawer_heights: Sequence[float] = (0.12, 0.12), wall: float = 0.016,
                       gap: float = 0.003, side_clearance: float = 0.004,
                       front_thickness: float = 0.018, box_wall: float = 0.012,
                       box_bottom: float = 0.008, box_height_margin: float = 0.012,
                       travel: float = 0.30, drawer_mass: float | Sequence[float] = 2.0,
                       damping: float | Sequence[float] = 8.0,
                       frictionloss: float | Sequence[float] = 2.0,
                       handle_length: float = 0.16, handle_radius: float = 0.0065,
                       handle_standoff: float = 0.035, handle_mass: float = 0.12,
                       handle_height_frac: float = 0.5,
                       carcass_rgba=(0.80, 0.76, 0.68, 1.0), front_rgba=(0.60, 0.42, 0.27, 1.0),
                       box_rgba=(0.86, 0.74, 0.55, 1.0), handle_rgba=(0.78, 0.79, 0.82, 1.0),
                       friction="wood", handle_friction="metal") -> dict:
    """Static carcass + stacked drawers on slide joints (see module docstring)."""
    n = len(drawer_heights)

    def per(v, i):
        return float(v[i]) if isinstance(v, (list, tuple, np.ndarray)) else float(v)

    height = 2 * wall + float(np.sum(drawer_heights)) + (n - 1) * wall
    if travel > depth - wall - front_thickness - 0.06:
        raise ValueError("drawer travel too long for the carcass depth")
    pbody = parent or sb.spec.worldbody
    quat = yaw_quat(yaw)
    body = pbody.add_body(name=name, pos=[float(x) for x in pos], quat=list(quat))
    cmat = sb.material(f"{name}_carcass_mat", carcass_rgba, specular=0.15, shininess=0.2)
    fmat = sb.material(f"{name}_front_mat", front_rgba, specular=0.25, shininess=0.35)
    bmat = sb.material(f"{name}_box_mat", box_rgba, specular=0.1, shininess=0.2)
    hmat = sb.material(f"{name}_handle_mat", handle_rgba, specular=0.8, shininess=0.8,
                       reflectance=0.15)
    hx, hy = depth / 2, width / 2
    carcass = []

    def cbox(tag, lo, hi):
        carcass.append(f"{name}_{tag}")
        _box(body, f"{name}_{tag}", lo, hi, material=cmat, friction=friction)

    cbox("bottom", (-hx, -hy, 0.0), (hx, hy, wall))
    cbox("top", (-hx, -hy, height - wall), (hx, hy, height))
    cbox("side_l", (-hx, -hy, wall), (hx, -hy + wall, height - wall))
    cbox("side_r", (-hx, hy - wall, wall), (hx, hy, height - wall))
    cbox("back", (-hx, -hy + wall, wall), (-hx + wall, hy - wall, height - wall))
    z = wall
    openings = []
    for i, h in enumerate(drawer_heights):
        openings.append((z, z + float(h)))
        z += float(h)
        if i < n - 1:
            cbox(f"divider{i}", (-hx + wall, -hy + wall, z), (hx, hy - wall, z + wall))
            z += wall

    w_in = width - 2 * wall                      # opening width
    box_depth = depth - wall - front_thickness - 0.010
    drawers = []
    for i, (z0, z1) in enumerate(openings):
        dn = f"{name}_drawer{i}"
        h_open = z1 - z0
        front_h = h_open - 2 * gap
        box_h = front_h - box_height_margin
        d = body.add_body(name=dn, pos=[hx, 0.0, z0])
        d.add_joint(name=f"{dn}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                    range=[0.0, float(travel)], limited=1, damping=per(damping, i),
                    frictionloss=per(frictionloss, i), armature=0.02)
        yo = w_in / 2 - gap                          # front panel half width
        yb = w_in / 2 - side_clearance               # box outer half width
        xb0, xb1 = -front_thickness - box_depth, -front_thickness
        parts = {
            "front": ((-front_thickness, -yo, gap), (0.0, yo, gap + front_h), fmat),
            "bottom": ((xb0, -yb, gap), (xb1, yb, gap + box_bottom), bmat),
            "wall_l": ((xb0, -yb, gap + box_bottom), (xb1, -yb + box_wall, gap + box_h), bmat),
            "wall_r": ((xb0, yb - box_wall, gap + box_bottom), (xb1, yb, gap + box_h), bmat),
            "wall_b": ((xb0, -yb + box_wall, gap + box_bottom), (xb0 + box_wall, yb - box_wall, gap + box_h), bmat),
        }
        m_box = per(drawer_mass, i) - handle_mass
        vtot = sum(_vol(lo, hi) for lo, hi, _ in parts.values())
        geoms = []
        for tag, (lo, hi, mat) in parts.items():
            gname = f"{dn}_{tag}"
            _box(d, gname, lo, hi, material=mat, friction=friction, mass=m_box * _vol(lo, hi) / vtot)
            geoms.append(gname)
        # handle: bar on two standoffs, centred on the front panel
        zc = gap + handle_height_frac * front_h
        hb = d.add_body(name=f"{dn}_handle", pos=[float(handle_standoff), 0.0, float(zc)])
        r_so = 0.0045
        so_len = handle_standoff - 0.5 * handle_radius
        bar_v = np.pi * handle_radius ** 2 * handle_length
        so_v = 2 * np.pi * r_so ** 2 * so_len
        bar_m = handle_mass * bar_v / (bar_v + so_v)
        hb.add_geom(name=f"{dn}_handle_bar", type=GEOM.mjGEOM_CAPSULE, quat=list(_Q_Z_TO_Y),
                    size=[handle_radius, handle_length / 2 - handle_radius, 0], material=hmat,
                    friction=_fr(handle_friction), condim=4, mass=bar_m)
        for k, sy in enumerate((-1, 1)):
            yy = sy * (handle_length / 2 - 0.016)
            hb.add_geom(name=f"{dn}_handle_post{k}", type=GEOM.mjGEOM_CYLINDER,
                        quat=list(_Q_Z_TO_X), pos=[-so_len / 2 - 0.5 * handle_radius, yy, 0.0],
                        size=[r_so, so_len / 2, 0], material=hmat, friction=_fr(handle_friction),
                        mass=(handle_mass - bar_m) / 2)
        hb.add_site(name=f"{dn}_handle_site", pos=[0, 0, 0], size=[0.004] * 3, rgba=[1, 0.3, 0.1, 0])
        d.add_site(name=f"{dn}_interior", pos=[(xb0 + box_wall + xb1) / 2, 0.0, gap + box_bottom],
                   size=[0.005] * 3, rgba=[0.1, 0.8, 0.2, 0])
        sb.spec.add_exclude(bodyname1=name, bodyname2=dn)
        drawers.append({
            "body": dn, "joint": f"{dn}_slide", "handle_body": f"{dn}_handle",
            "handle_geom": f"{dn}_handle_bar", "handle_site": f"{dn}_handle_site",
            "interior_site": f"{dn}_interior", "front_geom": f"{dn}_front", "geoms": geoms,
            "travel": float(travel), "mass": per(drawer_mass, i),
            "interior_half": ((xb1 - xb0 - box_wall) / 2, yb - box_wall),
            "interior_height": box_h - box_bottom,
            "handle_length": handle_length, "handle_radius": handle_radius,
            "handle_standoff": handle_standoff,
            "handle_closed_local": np.array([hx + handle_standoff, 0.0, z0 + zc]),
            "interior_floor_local": np.array([hx + (xb0 + box_wall + xb1) / 2, 0.0, z0 + gap + box_bottom]),
            "front_inner_local_x": hx - front_thickness,
            "front_top_local_z": z0 + gap + front_h,
            "opening_top_local_z": z1,
            "damping": per(damping, i), "frictionloss": per(frictionloss, i),
        })
    R = _rot(yaw)
    return {"body": name, "carcass_geoms": carcass, "drawers": drawers,
            "front_normal": R @ np.array([1.0, 0.0, 0.0]), "quat": quat,
            "size": (depth, width, height)}


# ==================================================================== microwave
def add_microwave(sb: SceneBuilder, name: str, pos: Sequence[float], yaw: float = 0.0, *,
                  parent=None, width: float = 0.46, depth: float = 0.36, height: float = 0.27,
                  wall: float = 0.012, feet: float = 0.012, cavity_width: float = 0.30,
                  hinge_side: str = "left", door_thickness: float = 0.03, door_gap: float = 0.002,
                  door_mass: float = 1.5, damping: float = 0.2, frictionloss: float = 0.15,
                  open_range_deg: float = 115.0, handle_length: float = 0.14,
                  handle_radius: float = 0.007, handle_standoff: float = 0.035,
                  handle_inset: float = 0.035, handle_mass: float = 0.10,
                  detent_stiffness: float = 0.0, detent_preload: float = 0.25,
                  shell_rgba=(0.88, 0.88, 0.86, 1.0), cavity_rgba=(0.93, 0.93, 0.90, 1.0),
                  door_rgba=(0.20, 0.21, 0.23, 1.0), handle_rgba=(0.80, 0.80, 0.82, 1.0),
                  friction="metal", cavity_friction="glass", handle_friction="metal") -> dict:
    """Static microwave shell with a hollow cavity and a hinged door (see module docstring)."""
    if hinge_side not in ("left", "right"):
        raise ValueError("hinge_side must be 'left' or 'right'")
    side = -1.0 if hinge_side == "left" else 1.0        # local y sign of the hinge side
    pbody = parent or sb.spec.worldbody
    quat = yaw_quat(yaw)
    body = pbody.add_body(name=name, pos=[float(x) for x in pos], quat=list(quat))
    smat = sb.material(f"{name}_shell_mat", shell_rgba, specular=0.4, shininess=0.5)
    cmat = sb.material(f"{name}_cavity_mat", cavity_rgba, specular=0.2, shininess=0.3)
    dmat = sb.material(f"{name}_door_mat", door_rgba, specular=0.6, shininess=0.7, reflectance=0.1)
    gmat = sb.material(f"{name}_glass_mat", (0.05, 0.06, 0.07, 1.0), specular=0.9, shininess=0.9,
                       reflectance=0.25)
    hmat = sb.material(f"{name}_handle_mat", handle_rgba, specular=0.8, shininess=0.8, reflectance=0.15)
    pmat = sb.material(f"{name}_panel_mat", (0.12, 0.12, 0.13, 1.0), specular=0.5, shininess=0.6)
    hx, hy = depth / 2, width / 2
    z_floor, z_ceil = feet + wall, height - wall
    if cavity_width > width - 2 * wall - 0.06:
        raise ValueError("cavity too wide for the shell (need room for the control panel)")
    shell = []

    def sbox(tag, lo, hi, mat=smat, fric=friction):
        shell.append(f"{name}_{tag}")
        _box(body, f"{name}_{tag}", lo, hi, material=mat, friction=fric)

    # y-extents (local) of hinge wall, cavity and control-panel block
    if side < 0:
        y_hw = (-hy, -hy + wall)
        y_cav = (-hy + wall, -hy + wall + cavity_width)
        y_pan = (y_cav[1], hy)
    else:
        y_hw = (hy - wall, hy)
        y_cav = (hy - wall - cavity_width, hy - wall)
        y_pan = (-hy, y_cav[0])
    for k, (fx, fy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        body.add_geom(name=f"{name}_foot{k}", type=GEOM.mjGEOM_CYLINDER,
                      pos=[fx * (hx - 0.03), fy * (hy - 0.03), feet / 2], size=[0.012, feet / 2, 0],
                      rgba=[0.1, 0.1, 0.1, 1], friction=_fr("rubber"), condim=3)
        shell.append(f"{name}_foot{k}")
    sbox("floor", (-hx, -hy, feet), (hx, hy, z_floor), mat=cmat, fric=cavity_friction)
    sbox("ceiling", (-hx, -hy, z_ceil), (hx, hy, height))
    sbox("back", (-hx, -hy, z_floor), (-hx + wall, hy, z_ceil))
    sbox("side_hinge", (-hx + wall, y_hw[0], z_floor), (hx, y_hw[1], z_ceil))
    sbox("panel_block", (-hx + wall, y_pan[0], z_floor), (hx, y_pan[1], z_ceil))
    # cosmetic: control-panel face + buttons (visual only)
    yc = (y_pan[0] + y_pan[1]) / 2
    _box(body, f"{name}_panel_face", (hx, y_pan[0] + 0.008, z_floor + 0.01),
         (hx + 0.002, y_pan[1] - 0.008, z_ceil - 0.01), material=pmat, collide=False)
    for k in range(3):
        zz = z_ceil - 0.05 - 0.035 * k
        body.add_geom(name=f"{name}_button{k}", type=GEOM.mjGEOM_CYLINDER, quat=list(_Q_Z_TO_X),
                      pos=[hx + 0.004, yc, zz], size=[0.011, 0.002, 0], rgba=[0.75, 0.75, 0.78, 1],
                      contype=0, conaffinity=0, group=1, mass=0.0)
    # inner light (visual cue for the cavity)
    cav_c = np.array([(-hx + wall + hx) / 2, (y_cav[0] + y_cav[1]) / 2, z_floor])
    body.add_site(name=f"{name}_cavity", pos=list(cav_c), size=[0.005] * 3, rgba=[0.2, 0.8, 0.2, 0])

    # ---------------------------------------------------------------- door
    door_w = cavity_width + wall + 0.012          # covers hinge wall + cavity + small overlap
    hinge_local = np.array([hx + door_gap, side * hy, feet + 0.004])
    dz = (height - 0.004) - (feet + 0.004)
    door = body.add_body(name=f"{name}_door", pos=list(hinge_local))
    # local door frame: hinge axis at origin, door extends along u = -side*y, thickness along +x
    u = -side
    axis = [0.0, 0.0, side]        # positive angle swings the free edge toward +x (opening)
    # mjSpec interprets joint ranges/springrefs in the compiler's angle unit (degrees by default)
    to_unit = (lambda rad: float(np.degrees(rad))) if sb.spec.compiler.degree else float
    jkw = dict(name=f"{name}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis,
               range=[0.0, to_unit(np.radians(open_range_deg))], limited=1, damping=float(damping),
               frictionloss=float(frictionloss), armature=0.002)
    if detent_stiffness > 0:
        jkw.update(stiffness=float(detent_stiffness), springref=-to_unit(detent_preload))
    door.add_joint(**jkw)
    y_lo, y_hi = sorted([0.0, u * door_w])
    slab_lo, slab_hi = (0.0, y_lo, 0.0), (door_thickness, y_hi, dz)
    door_geoms = [f"{name}_door_slab"]
    m_slab = door_mass - handle_mass
    _box(door, f"{name}_door_slab", slab_lo, slab_hi, material=dmat, friction="plastic", mass=m_slab)
    # glass window (visual) on the front face
    m = 0.035
    wy = sorted([u * (m + 0.01), u * (door_w - 0.075)])
    _box(door, f"{name}_door_window", (door_thickness, wy[0], m + 0.01),
         (door_thickness + 0.0015, wy[1], dz - m), material=gmat, collide=False)
    # handle (vertical bar) near the free edge
    y_bar = u * (door_w - handle_inset)
    hb = door.add_body(name=f"{name}_handle", pos=[door_thickness + handle_standoff, y_bar, dz / 2])
    r_so = 0.005
    so_len = handle_standoff - 0.5 * handle_radius
    bar_v = np.pi * handle_radius ** 2 * handle_length
    so_v = 2 * np.pi * r_so ** 2 * so_len
    bar_m = handle_mass * bar_v / (bar_v + so_v)
    hb.add_geom(name=f"{name}_handle_bar", type=GEOM.mjGEOM_CAPSULE,
                size=[handle_radius, handle_length / 2 - handle_radius, 0], material=hmat,
                friction=_fr(handle_friction), condim=4, mass=bar_m)
    for k, sz in enumerate((-1, 1)):
        hb.add_geom(name=f"{name}_handle_post{k}", type=GEOM.mjGEOM_CYLINDER, quat=list(_Q_Z_TO_X),
                    pos=[-so_len / 2 - 0.5 * handle_radius, 0.0, sz * (handle_length / 2 - 0.014)],
                    size=[r_so, so_len / 2, 0], material=hmat, friction=_fr(handle_friction),
                    mass=(handle_mass - bar_m) / 2)
    hb.add_site(name=f"{name}_handle_site", pos=[0, 0, 0], size=[0.004] * 3, rgba=[1, 0.3, 0.1, 0])
    sb.spec.add_exclude(bodyname1=name, bodyname2=f"{name}_door")

    R = _rot(yaw)
    p0 = np.asarray(pos, float)
    bar_rel = np.array([door_thickness + handle_standoff, y_bar])
    return {
        "body": name, "shell_geoms": shell, "front_normal": R @ np.array([1.0, 0.0, 0.0]),
        "quat": quat, "size": (depth, width, height),
        "door_body": f"{name}_door", "hinge_joint": f"{name}_hinge", "door_geom": f"{name}_door_slab",
        "door_geoms": door_geoms, "handle_body": f"{name}_handle", "handle_geom": f"{name}_handle_bar",
        "handle_site": f"{name}_handle_site",
        "hinge_axis_world": R @ np.array(axis), "hinge_point_world": p0 + R @ hinge_local,
        "door_width": door_w, "door_thickness": door_thickness, "door_mass": door_mass,
        "door_height": dz, "open_range": float(np.radians(open_range_deg)),
        "handle_radius_from_hinge": float(np.hypot(*bar_rel)),
        "handle_angle_offset": float(np.arctan2(bar_rel[0], abs(bar_rel[1]))),
        "handle_height_local": float(hinge_local[2] + dz / 2),
        "handle_length": handle_length, "handle_bar_radius": handle_radius,
        "cavity_site": f"{name}_cavity", "cavity_half": ((2 * hx - wall) / 2, cavity_width / 2),
        "cavity_height": z_ceil - z_floor, "cavity_center_local": cav_c,
        "cavity_y_local": y_cav, "floor_z_local": z_floor, "hinge_side": hinge_side,
        "damping": damping, "frictionloss": frictionloss,
    }
