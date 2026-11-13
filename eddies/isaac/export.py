"""Build a task exactly as the MuJoCo runner does and convert its scene to USD.

``make_task_and_usd(task, seed, perturb)`` returns the (unchanged) eddies.envs task
object — used as the task-logic/kinematics mirror — and the path of the PhysX USD scene
converted from the very same MjSpec by NVIDIA's MJCF importer.  Conversions are cached
in ``.cache/isaac_usd/<task>/seed_XX`` ($EDDIES_ISAAC_CACHE), keyed by a hash of the scene XML.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from eddies.isaac.env import CACHE

_LAST_SPECS: list = []
_PATCHED = False


def _patch_scene_builder() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from eddies.envs.framework import scene as S
    orig = S.SceneBuilder.compile

    def compile_and_keep(self):
        _LAST_SPECS.append(self.spec)
        del _LAST_SPECS[:-4]
        return orig(self)

    S.SceneBuilder.compile = compile_and_keep
    _PATCHED = True


def _externalize_inline_meshes(spec, out: Path) -> str:
    """The USD converter needs mesh FILES: write meshes given inline (vertices[/faces]) as
    OBJ files (faces = convex hull when none are given, exactly what MuJoCo collides with)."""
    import numpy as np
    from scipy.spatial import ConvexHull
    for mesh in spec.meshes:
        if mesh.file:
            continue
        v = np.asarray(mesh.uservert, float).reshape(-1, 3)
        if not len(v):
            continue
        f = np.asarray(mesh.userface, int).reshape(-1, 3) if len(mesh.userface) else None
        if f is None:
            hull = ConvexHull(v)
            f = hull.simplices.copy()
            c = v.mean(0)                                    # outward winding
            for i, (a, b, d) in enumerate(f):
                if np.dot(np.cross(v[b] - v[a], v[d] - v[a]), v[a] - c) < 0:
                    f[i] = (a, d, b)
        path = out / f"inline_{mesh.name.replace('/', '_')}.obj"
        lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in v] + [f"f {a + 1} {b + 1} {d + 1}" for a, b, d in f]
        path.write_text("\n".join(lines) + "\n")
        mesh.uservert = []
        mesh.userface = []
        mesh.file = str(path)
    return spec.to_xml()


_POSE_ATTRS = ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")
CONVERT_VERSION = "4"        # bump to invalidate cached conversions after converter-side changes


def _split_multi_joint_bodies(xml: str) -> str:
    """PhysX articulation links have ONE joint each, and the converter turns a MuJoCo body
    with several hinge/slide joints (e.g. the screw bolt hinge+slide) into
    a generic D6 joint that leaves the remaining axes free.  Equivalent exact chain: each
    extra joint gets its own massless-in-practice link (1e-4 kg) whose frame coincides
    with the body frame, joints applied in MuJoCo's order (first joint nearest the parent).
    Only the PhysX scene is restructured; the task's MuJoCo model is unchanged."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    parent_of = {c: par for par in root.iter() for c in par}

    def eff_childclass(el):
        while el is not None:
            if el.get("childclass"):
                return el.get("childclass")
            el = parent_of.get(el)
        return None

    for body in list(root.iter("body")):
        joints = [c for c in body if c.tag == "joint" and c.get("type", "hinge") in ("hinge", "slide")]
        if len(joints) < 2 or any(c.tag in ("freejoint",) or (c.tag == "joint" and c.get("type") in ("free", "ball"))
                                  for c in body):
            continue
        cls = eff_childclass(body)
        par = parent_of[body]
        idx = list(par).index(body)
        par.remove(body)
        holder, insert_at = par, idx
        for k, j in enumerate(joints[:-1]):
            d = ET.Element("body", {"name": f"{body.get('name', 'body')}__isaac_dof{k}"})
            if k == 0:
                for a in _POSE_ATTRS:
                    if body.get(a) is not None:
                        d.set(a, body.get(a))
                        del body.attrib[a]
            if cls and not j.get("class"):
                j.set("class", cls)
            if cls:
                d.set("childclass", cls)
            body.remove(j)
            d.append(ET.Element("inertial", {"pos": "0 0 0", "mass": "0.0001", "diaginertia": "1e-08 1e-08 1e-08"}))
            d.append(j)
            holder.insert(insert_at, d)
            holder, insert_at = d, len(d)
        holder.insert(insert_at, body)
        parent_of = {c: p_ for p_ in root.iter() for c in p_}
    return ET.tostring(root, encoding="unicode")


def _ellipsoids_to_meshes(xml: str, out: Path) -> str:
    """PhysX has no ellipsoid shape (the converter drops such colliders, e.g. the steering
    wheel hub): collidable ellipsoids become a finely tessellated convex mesh of the same
    semi-axes (MuJoCo itself collides with convex hulls of meshes)."""
    import xml.etree.ElementTree as ET
    import numpy as np
    root = ET.fromstring(xml)
    geoms = [g for g in root.iter("geom") if g.get("type") == "ellipsoid" and g.get("size")
             and not (g.get("contype") == "0" and g.get("conaffinity") == "0")]
    if not geoms:
        return xml
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    nu, nv = 48, 24
    for i, g in enumerate(geoms):
        a, b, c = (float(x) for x in g.get("size").split()[:3])
        th = np.linspace(0, np.pi, nv + 1)[1:-1]
        ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)
        v = [(0.0, 0.0, c)] + [(a * np.sin(t) * np.cos(f), b * np.sin(t) * np.sin(f), c * np.cos(t))
                               for t in th for f in ph] + [(0.0, 0.0, -c)]
        faces = []
        for k in range(nu):                                  # caps and bands (outward winding)
            faces.append((0, 1 + k, 1 + (k + 1) % nu))
            base = 1 + (nv - 2) * nu
            faces.append((len(v) - 1, base + (k + 1) % nu, base + k))
        for r in range(nv - 2):
            for k in range(nu):
                p0, p1 = 1 + r * nu + k, 1 + r * nu + (k + 1) % nu
                q0, q1 = p0 + nu, p1 + nu
                faces += [(p0, q0, q1), (p0, q1, p1)]
        name = f"isaac_ellipsoid_{i}_{g.get('name', 'geom').replace('/', '_')}"
        path = out / f"{name}.obj"
        path.write_text("\n".join([f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in v] +
                                  [f"f {p + 1} {q + 1} {r + 1}" for p, q, r in faces]) + "\n")
        asset.append(ET.Element("mesh", {"name": name, "file": str(path)}))
        g.set("type", "mesh")
        g.set("mesh", name)
        del g.attrib["size"]
    return ET.tostring(root, encoding="unicode")


def _quiet_jointless_overlay() -> None:
    """The importer's last step (merging over-constrained joints into D6 joints) reads the
    PhysX overlay layer, which the converter only writes for scenes WITH joints; for
    joint-less scenes (free rigid bodies only) it logs an error
    although there is nothing to correct.  Skip that step quietly when the layer is absent
    (runtime wrapper in this process; the Isaac Sim installation is not modified)."""
    import os
    from isaacsim.asset.importer.utils.impl import mjc_to_physx_conversion_utils as U
    if getattr(U.combine_overconstrained_joints_in_physx_layer, "_bistl_wrapped", False):
        return
    orig = U.combine_overconstrained_joints_in_physx_layer

    def combine(physx_layer_path: str) -> int:
        if not os.path.exists(physx_layer_path):
            return 0                       # no joints -> no PhysX overlay -> nothing to merge
        return orig(physx_layer_path)

    combine._bistl_wrapped = True
    U.combine_overconstrained_joints_in_physx_layer = combine


def make_task_and_usd(task_name: str, seed: int, perturb: bool = False):
    _patch_scene_builder()
    from eddies.envs.tasks import make_task
    task = make_task(task_name, seed, **({"perturb": True} if perturb else {}))
    xml = _LAST_SPECS[-1].to_xml()
    key = hashlib.sha256((CONVERT_VERSION + xml).encode()).hexdigest()[:16]
    out = CACHE / "usd" / task_name / f"seed_{seed:02d}_{key}"
    done = out / "DONE"
    if done.exists():
        return task, done.read_text().strip()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    xml = _split_multi_joint_bodies(_externalize_inline_meshes(_LAST_SPECS[-1], out))
    xml = _ellipsoids_to_meshes(xml, out)
    src = out / f"{task_name}.xml"
    src.write_text(xml)
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    _quiet_jointless_overlay()
    usd = MJCFImporter().import_mjcf(MJCFImporterConfig(
        mjcf_path=str(src), usd_path=str(out), import_scene=False, allow_self_collision=False,
        fix_base=None, run_asset_transformer=True, run_multi_physics_conversion=True))
    done.write_text(str(usd))
    return task, str(usd)
