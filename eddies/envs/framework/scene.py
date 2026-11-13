"""mjSpec helpers for building physical scenes around menagerie robots.

Robots are loaded from the pinned mujoco_menagerie checkout and attached with a
name prefix, so their meshes, defaults, actuators and keyframes are preserved.
Objects are ordinary rigid bodies (free joints, real mass/inertia, finite
friction) or articulated mechanisms (hinge/slide joints with damping and
friction loss).  Helper geometry sizes are MuJoCo HALF-sizes.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import mujoco
import numpy as np

from eddies.envs.framework.physics import FRICTION, configure_options, configure_visual
from eddies.envs.paths import menagerie

GEOM = mujoco.mjtGeom


def look_at_quat(pos: Sequence[float], target: Sequence[float],
                 up: Sequence[float] = (0, 0, 1)) -> np.ndarray:
    """Quaternion (wxyz) of a MuJoCo camera at ``pos`` looking at ``target``.

    MuJoCo cameras look along their local -z axis with +y up.
    """
    pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:          # looking straight down/up
        x = np.cross((0, 1, 0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).reshape(-1))
    return quat


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


class SceneBuilder:
    """Thin convenience layer over :class:`mujoco.MjSpec`."""

    def __init__(self, name: str, *, timestep: float = 0.002, cone: str = "elliptic",
                 impratio: float = 10.0, air: bool = True, width: int = 1920,
                 height: int = 1080, floor: bool = True, floor_size: float = 6.0,
                 floor_friction: str = "floor", skybox: bool = True):
        self.spec = mujoco.MjSpec()
        self.spec.modelname = name
        configure_options(self.spec, timestep=timestep, cone=cone, impratio=impratio, air=air)
        configure_visual(self.spec, width, height)
        self._materials: set[str] = set()
        if skybox:
            self.spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                                  builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                                  rgb1=[0.52, 0.66, 0.82], rgb2=[0.95, 0.96, 0.98],
                                  width=512, height=3072)
        self.texture_material("floor_mat", builtin="checker", rgb1=(0.80, 0.80, 0.78),
                              rgb2=(0.68, 0.68, 0.66), texrepeat=(8, 8), reflectance=0.05)
        if floor:
            self.spec.worldbody.add_geom(name="floor", type=GEOM.mjGEOM_PLANE,
                                         size=[floor_size, floor_size, 0.05],
                                         material="floor_mat",
                                         friction=list(FRICTION[floor_friction]), condim=3)
        self.spec.worldbody.add_light(name="key_light", pos=[1.5, -1.5, 3.5],
                                      dir=[-0.35, 0.35, -1.0], diffuse=[0.65, 0.65, 0.65],
                                      specular=[0.2, 0.2, 0.2], castshadow=True)
        self.spec.worldbody.add_light(name="fill_light", pos=[-2.0, 2.0, 3.0],
                                      dir=[0.5, -0.5, -1.0], diffuse=[0.3, 0.3, 0.3],
                                      specular=[0.0, 0.0, 0.0], castshadow=False)

    # ------------------------------------------------------------------ materials
    def material(self, name: str, rgba: Sequence[float], *, specular: float = 0.3,
                 shininess: float = 0.4, reflectance: float = 0.0) -> str:
        if name not in self._materials:
            self.spec.add_material(name=name, rgba=list(rgba), specular=specular,
                                   shininess=shininess, reflectance=reflectance)
            self._materials.add(name)
        return name

    def texture_material(self, name: str, *, builtin: str = "checker",
                         rgb1=(0.8, 0.8, 0.8), rgb2=(0.6, 0.6, 0.6), texrepeat=(4, 4),
                         reflectance: float = 0.0) -> str:
        if name in self._materials:
            return name
        kind = {"checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT}[builtin]
        self.spec.add_texture(name=name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                              builtin=kind, rgb1=list(rgb1), rgb2=list(rgb2),
                              width=512, height=512)
        mat = self.spec.add_material(name=name, texrepeat=list(texrepeat),
                                     reflectance=reflectance, texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name + "_tex"
        self._materials.add(name)
        return name

    # ------------------------------------------------------------------- robots
    def attach_robot(self, relative_xml: str, prefix: str, *, pos=(0, 0, 0), quat=(1, 0, 0, 0),
                     edit: Callable[[mujoco.MjSpec], None] | None = None,
                     parent: mujoco.MjsBody | None = None) -> mujoco.MjSpec:
        """Attach a menagerie robot (optionally edited first) at a world frame."""
        child = mujoco.MjSpec.from_file(menagerie(relative_xml))
        if edit is not None:
            edit(child)
        frame = (parent or self.spec.worldbody).add_frame(pos=list(map(float, pos)),
                                                          quat=list(map(float, quat)))
        self.spec.attach(child, prefix=prefix, frame=frame)
        return child

    def attach_spec(self, child: mujoco.MjSpec, prefix: str, *, pos=(0, 0, 0),
                    quat=(1, 0, 0, 0), parent: mujoco.MjsBody | None = None) -> None:
        frame = (parent or self.spec.worldbody).add_frame(pos=list(map(float, pos)),
                                                          quat=list(map(float, quat)))
        self.spec.attach(child, prefix=prefix, frame=frame)

    # ------------------------------------------------------------------ objects
    def static_box(self, name: str, pos, half, *, rgba=(0.6, 0.6, 0.6, 1), quat=(1, 0, 0, 0),
                   friction: str | Sequence[float] = "table", material: str | None = None,
                   parent: mujoco.MjsBody | None = None, condim: int = 3,
                   contype: int = 1, conaffinity: int = 1) -> mujoco.MjsGeom:
        body = parent or self.spec.worldbody
        return body.add_geom(name=name, type=GEOM.mjGEOM_BOX, pos=list(map(float, pos)),
                             quat=list(map(float, quat)), size=list(map(float, half)),
                             rgba=list(rgba), friction=_friction(friction), condim=condim,
                             contype=contype, conaffinity=conaffinity,
                             **({"material": material} if material else {}))

    def table(self, name: str, center_xy, top_height: float, half_xy, *,
              thickness: float = 0.03, rgba=(0.55, 0.38, 0.22, 1), leg_rgba=(0.25, 0.25, 0.27, 1),
              friction: str = "table") -> mujoco.MjsBody:
        """Static table: a slab top plus four legs (top surface at ``top_height``)."""
        cx, cy = map(float, center_xy)
        hx, hy = map(float, half_xy)
        body = self.spec.worldbody.add_body(name=name, pos=[cx, cy, 0.0])
        mat = self.material(name + "_top_mat", rgba, specular=0.2, shininess=0.3)
        body.add_geom(name=name + "_top", type=GEOM.mjGEOM_BOX,
                      pos=[0, 0, top_height - thickness / 2], size=[hx, hy, thickness / 2],
                      material=mat, friction=_friction(friction), condim=3)
        leg_h = (top_height - thickness) / 2
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            body.add_geom(name=f"{name}_leg{i}", type=GEOM.mjGEOM_BOX,
                          pos=[sx * (hx - 0.04), sy * (hy - 0.04), leg_h],
                          size=[0.025, 0.025, leg_h], rgba=list(leg_rgba), contype=1, conaffinity=1)
        return body

    def free_body(self, name: str, pos, *, quat=(1, 0, 0, 0)) -> mujoco.MjsBody:
        body = self.spec.worldbody.add_body(name=name, pos=list(map(float, pos)),
                                            quat=list(map(float, quat)))
        body.add_freejoint(name=name + "_free")
        return body

    def free_box(self, name: str, pos, half, *, mass: float, rgba=(0.8, 0.2, 0.2, 1),
                 quat=(1, 0, 0, 0), friction: str | Sequence[float] = "wood",
                 condim: int = 4) -> mujoco.MjsBody:
        body = self.free_body(name, pos, quat=quat)
        body.add_geom(name=name + "_geom", type=GEOM.mjGEOM_BOX, size=list(map(float, half)),
                      mass=float(mass), rgba=list(rgba), friction=_friction(friction),
                      condim=condim)
        return body

    def free_cylinder(self, name: str, pos, radius: float, half_height: float, *, mass: float,
                      rgba=(0.2, 0.4, 0.8, 1), quat=(1, 0, 0, 0),
                      friction: str | Sequence[float] = "plastic", condim: int = 4) -> mujoco.MjsBody:
        body = self.free_body(name, pos, quat=quat)
        body.add_geom(name=name + "_geom", type=GEOM.mjGEOM_CYLINDER,
                      size=[float(radius), float(half_height), 0], mass=float(mass),
                      rgba=list(rgba), friction=_friction(friction), condim=condim)
        return body

    def camera(self, name: str, pos, target, *, fovy: float = 45.0,
               parent: mujoco.MjsBody | None = None) -> mujoco.MjsCamera:
        body = parent or self.spec.worldbody
        return body.add_camera(name=name, pos=list(map(float, pos)),
                               quat=list(look_at_quat(pos, target)), fovy=float(fovy))

    def site(self, name: str, pos, *, size=0.01, rgba=(1, 0, 0, 0.0),
             parent: mujoco.MjsBody | None = None) -> mujoco.MjsSite:
        body = parent or self.spec.worldbody
        return body.add_site(name=name, pos=list(map(float, pos)), size=[size, size, size],
                             rgba=list(rgba))

    # ------------------------------------------------------------------ compile
    def compile(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model = self.spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data


def _friction(f) -> list[float]:
    if isinstance(f, str):
        return list(FRICTION[f])
    f = [float(x) for x in f]
    return f + list(FRICTION["plastic"][len(f):])


# --------------------------------------------------------------- name lookups
def jid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise KeyError(f"joint {name!r} not in model")
    return i


def bid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"body {name!r} not in model")
    return i


def gid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if i < 0:
        raise KeyError(f"geom {name!r} not in model")
    return i


def sid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise KeyError(f"site {name!r} not in model")
    return i


def aid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        raise KeyError(f"actuator {name!r} not in model")
    return i


def qadr(model, joint_name: str) -> int:
    return int(model.jnt_qposadr[jid(model, joint_name)])


def dadr(model, joint_name: str) -> int:
    return int(model.jnt_dofadr[jid(model, joint_name)])


def names(model, obj: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj, i) or "" for i in range(count)]


def subtree_geoms(model, root_body: str | int) -> list[int]:
    """All geom ids attached to ``root_body`` and its descendants."""
    root = bid(model, root_body) if isinstance(root_body, str) else int(root_body)
    bodies = {root}
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p in bodies:
                bodies.add(b)
                break
            p = int(model.body_parentid[p])
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def body_geoms(model, bodies: Iterable[str | int]) -> list[int]:
    ids = {bid(model, b) if isinstance(b, str) else int(b) for b in bodies}
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in ids]


def free_qadr(model, body_name: str) -> int:
    """qpos address of a body's free joint."""
    b = bid(model, body_name)
    j = int(model.body_jntadr[b])
    if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
        raise KeyError(f"body {body_name!r} has no free joint")
    return int(model.jnt_qposadr[j])
