"""Procedural peg-in-hole hardware (mjSpec boxes only; no meshes).

* :func:`add_hole_block` - a fixture block bolted to the table (static geometry, like a
  peg board clamped to a bench) with a blind SQUARE hole and a 45-degree entry chamfer.
  The hole is the empty column between four walls; the chamfer is cut into the top inner
  edge of each wall.  Boxes can only add material, so each wall is split into a lower
  part (up to the chamfer), an upper part (outside the chamfer) and a 45-degree rotated
  square prism whose only exposed face is the bevel plane (its other three faces are
  buried inside the lower/upper parts).  For a yaw-aligned square peg this gives exactly
  the support surface of a machined chamfer (the corner seams are overlaps of two bevels).
  Every hole-block geom collides as box-box (MuJoCo's dedicated box-box collider, up to 8
  contact points per pair), which keeps a sliding peg stable in the hole.
* :func:`add_square_peg` - a free square aluminium peg (box geom, real mass/inertia).

All sizes are SI (m); MuJoCo geom sizes are half-sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from eddies.envs.framework.scene import SceneBuilder, yaw_quat

GEOM = mujoco.mjtGeom


@dataclass
class HoleBlockGeometry:
    """Geometry of a built hole block (world frame)."""
    name: str
    center_xy: np.ndarray            # block (and hole axis) centre on the table
    yaw: float                       # block yaw about world z [rad]
    base_z: float                    # table surface height [m]
    outer: float                     # outer side length of the square block [m]
    height: float                    # block height [m]
    bottom: float                    # thickness of the bottom plate (blind hole floor) [m]
    hole_side: float                 # side length of the square hole [m]
    chamfer: float                   # 45-degree chamfer leg length [m]
    geom_names: list = field(default_factory=list)

    @property
    def top_z(self) -> float:
        return self.base_z + self.height

    @property
    def floor_z(self) -> float:
        return self.base_z + self.bottom

    @property
    def depth(self) -> float:
        return self.height - self.bottom

    @property
    def entrance(self) -> np.ndarray:
        """Centre of the hole entrance (top surface, on the hole axis)."""
        return np.r_[self.center_xy, self.top_z]

    def to_local(self, p) -> np.ndarray:
        """World point -> block frame (x,y rotated by -yaw about the hole axis; z from table)."""
        p = np.asarray(p, float)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        dx, dy = p[0] - self.center_xy[0], p[1] - self.center_xy[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, p[2] - self.base_z])


def add_hole_block(sb: SceneBuilder, name: str, center_xy, yaw: float, base_z: float, *,
                   hole_side: float, outer: float = 0.09, height: float = 0.062,
                   bottom: float = 0.012, chamfer: float = 0.0025,
                   friction=(0.3, 0.003, 1e-4), rgba=(0.62, 0.80, 0.93, 0.38),
                   floor_rgba=(0.25, 0.28, 0.32, 1.0), chamfer_rgba=(0.95, 0.62, 0.15, 0.85)
                   ) -> HoleBlockGeometry:
    """Static block with a blind square hole of side ``hole_side`` and a 45-degree chamfer.

    The block is attached to the world (bolted to the table): it is not a kinematic object
    and has no joint; contacts with it are ordinary rigid contacts.  The walls are
    rendered semi-transparent (clear acrylic) so the peg is visible inside the hole; the
    chamfer bevels are tinted so the entry funnel is visible in the close-up camera.
    """
    W, H, tb, s, ch = float(outer), float(height), float(bottom), float(hole_side), float(chamfer)
    if not (0 < ch < (W - s) / 4 and s < W - 4 * ch and tb < H - 2 * ch):
        raise ValueError("inconsistent hole-block dimensions")
    body = sb.spec.worldbody.add_body(name=name, pos=[float(center_xy[0]), float(center_xy[1]), float(base_z)],
                                      quat=list(yaw_quat(yaw)))
    wall_mat = sb.material(name + "_acrylic", rgba, specular=0.6, shininess=0.8, reflectance=0.0)
    floor_mat = sb.material(name + "_floor", floor_rgba, specular=0.2, shininess=0.2)
    bevel_mat = sb.material(name + "_bevel", chamfer_rgba, specular=0.5, shininess=0.6)
    fr = [float(x) for x in friction]
    names = []

    def box(gname, center, half, mat, quat=(1, 0, 0, 0)):
        body.add_geom(name=f"{name}_{gname}", type=GEOM.mjGEOM_BOX, pos=[float(v) for v in center],
                      size=[float(v) for v in half], quat=[float(v) for v in quat], material=mat,
                      friction=fr, condim=3, contype=1, conaffinity=1)
        names.append(f"{name}_{gname}")

    hs, hw = s / 2, W / 2
    # floor plate (blind hole bottom)
    box("floor", (0, 0, tb / 2), (hw, hw, tb / 2), floor_mat)
    # lower walls: z in [tb, H - ch]
    zl, hzl = (tb + H - ch) / 2, (H - ch - tb) / 2
    for sgn, tag in ((1, "px"), (-1, "nx")):
        box(f"wall_lo_{tag}", (sgn * (hs + hw) / 2, 0, zl), ((hw - hs) / 2, hw, hzl), wall_mat)
    for sgn, tag in ((1, "py"), (-1, "ny")):
        box(f"wall_lo_{tag}", (0, sgn * (hs + hw) / 2, zl), (hs, (hw - hs) / 2, hzl), wall_mat)
    # upper walls (outside the chamfer): z in [H - ch, H]
    zu, hzu = H - ch / 2, ch / 2
    a = hs + ch
    for sgn, tag in ((1, "px"), (-1, "nx")):
        box(f"wall_up_{tag}", (sgn * (a + hw) / 2, 0, zu), ((hw - a) / 2, hw, hzu), wall_mat)
    for sgn, tag in ((1, "py"), (-1, "ny")):
        box(f"wall_up_{tag}", (0, sgn * (a + hw) / 2, zu), (a, (hw - a) / 2, hzu), wall_mat)
    # 45-degree bevels: square prisms rotated 45 deg about the edge, centred on the buried
    # corner (hs + ch, H - ch); exposed face = the plane through (hs, H-ch) and (hs+ch, H).
    r = ch / np.sqrt(2.0)
    q_about_y = (np.cos(np.pi / 8), 0.0, np.sin(np.pi / 8), 0.0)
    q_about_x = (np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0)
    for sgn, tag in ((1, "px"), (-1, "nx")):
        box(f"bevel_{tag}", (sgn * (hs + ch), 0, H - ch), (r, hs + ch, r), bevel_mat, q_about_y)
    for sgn, tag in ((1, "py"), (-1, "ny")):
        box(f"bevel_{tag}", (0, sgn * (hs + ch), H - ch), (hs + ch, r, r), bevel_mat, q_about_x)
    return HoleBlockGeometry(name, np.asarray(center_xy, float)[:2].copy(), float(yaw), float(base_z),
                             W, H, tb, s, ch, names)


def add_square_peg(sb: SceneBuilder, name: str, bottom_center, yaw: float, *, side: float,
                   length: float, mass: float, friction, rgba=(0.72, 0.74, 0.78, 1.0),
                   stripe_rgba=(0.85, 0.15, 0.12, 1.0)) -> mujoco.MjsBody:
    """Free square peg standing upright with its bottom face centre at ``bottom_center``.

    One collision box (real mass, box inertia).  A thin non-colliding red ring marks the
    top end so the grasp is visible in video; it carries no mass.
    """
    p = np.asarray(bottom_center, float) + np.array([0, 0, length / 2 + 0.0005])
    body = sb.free_body(name, p, quat=yaw_quat(yaw))
    mat = sb.material(name + "_alu", rgba, specular=0.7, shininess=0.7)
    g = body.add_geom(name=name + "_geom", type=GEOM.mjGEOM_BOX, size=[side / 2, side / 2, length / 2],
                      mass=float(mass), material=mat, friction=[float(x) for x in friction], condim=4)
    g.priority = 1          # the peg's (anodized aluminium) friction governs its contacts
    stripe = sb.material(name + "_stripe", stripe_rgba, specular=0.3, shininess=0.3)
    body.add_geom(name=name + "_mark", type=GEOM.mjGEOM_BOX,
                  size=[side / 2 + 0.0004, side / 2 + 0.0004, 0.003],
                  pos=[0, 0, length / 2 - 0.004], material=stripe, contype=0, conaffinity=0,
                  mass=0.0, group=1)
    return body
