"""PhysX <-> task bridge: Isaac Sim simulates, the unchanged task logic decides.

Every physics step (dt identical to MuJoCo, 2 ms):

1. **PhysX steps** rigid-body dynamics, joints, contacts and friction of the USD scene
   converted from the task's MJCF (same robots, objects, masses, inertias, frictions).
2. **State mirror.** Joint/root/object states are read from PhysX tensor views and
   written into the task's ``MjData`` (``qpos``/``qvel``); ``mj_forward`` then provides
   kinematics (sites, frames, cameras) for the unchanged task code, which observes,
   labels, decides and commands exactly as in MuJoCo.  The mirror never integrates.
3. **Contact sensing from PhysX.** ``ContactMonitor.measure`` (used by the tasks for grasp,
   support and collision labels) is redirected to PhysX contact reports: normal force =
   |normal impulse| / dt summed per body pair.
4. **Actuation.** Single-joint affine servos (position/velocity actuators: Franka arm,
   G1, LEAP, UR5e, ...) become native, implicitly integrated PhysX joint
   drives with the MuJoCo gains (stiffness = gear^2 kp, damping = gear^2 kv + joint
   damping, target = gear (gain ctrl + bias0)/stiffness, max force = force range).  All
   other actuation — motors, tendon/site actuators (Franka gripper
   tendon, Aero-hand tendons), actuator-routed gravity
   compensation — is the generalized force MuJoCo's own actuator model computes
   (``qfrc_actuator`` minus the drive part), applied as joint efforts / body wrenches.
   Joint springs, fluid drag and non-actuated gravity compensation are applied the same
   way; joint friction loss as PhysX implicit joint friction effort; external disturbances
   (``xfrc_applied``) as PhysX body wrenches at the centre of mass.
5. Joint-coupling equality constraints (e.g. the Franka finger mimic, screw threads)
   become PhysX mimic joints.
6. Contact model.  Every PhysX collider is mapped to its MuJoCo geom; it gets a physics
   material with that geom's sliding friction (condim 1 -> frictionless, condim >= 4 ->
   torsional patch radius from the torsional coefficient), restitution 0, and a friction
   combine mode chosen per geom so that every pair gets MuJoCo's coefficient (max of the
   two for equal priority, the higher-priority geom's otherwise).  MuJoCo's collision filtering is
   reproduced exactly with PhysX filtered pairs: contype/conaffinity bitmasks, <exclude>
   pairs, same-weld and parent/child bodies (filterparent); robot self-collision is
   enabled like in MuJoCo.  Contact sensing (sensing.py) uses the PhysX contact report
   incl. friction anchors.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

_ALNUM = re.compile(r"[^0-9a-z]")


def _key(name: str) -> str:
    return _ALNUM.sub("", name.lower())


def _decode(prim_name: str) -> str:
    n = prim_name[4:] if prim_name.startswith("tn__") else prim_name
    n = re.sub(r"_[A-Za-z0-9]{2}$", "", n) if prim_name.startswith("tn__") else n
    return _key(n)


class _EnumInts:
    """An mjt* enum class whose members read as plain ints."""
    def __init__(self, enum_cls):
        self._e = enum_cls

    def __getattr__(self, name):
        return int(getattr(self._e, name))


class _MjInts:
    """``mujoco`` module proxy for the bridge: enum members are plain ints.  Under numpy 2.x
    (Isaac Sim 6.0.1 pip build) ``np.int32 in (mjtJoint.mjJNT_HINGE, ...)`` is False although
    ``==`` is True (the tuple test calls the enum's __eq__, which rejects numpy integers), so every
    joint/actuator type test silently failed.  MuJoCo functions accept ints for enum arguments."""
    def __init__(self, mod):
        self._m = mod

    def __getattr__(self, name):
        attr = getattr(self._m, name)
        return _EnumInts(attr) if name.startswith("mjt") else attr


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_to_mat(q):  # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PhysXBridge:
    def __init__(self, task, usd_path: str, *, physics_iterations: int = 32):
        import mujoco
        import omni.usd
        from isaacsim.core.api import World
        from pxr import PhysxSchema, Usd, UsdPhysics
        self.mj = _MjInts(mujoco)
        self.task = task
        self.m, self.d = task.model, task.data
        m = self.m
        ctx = omni.usd.get_context()
        ctx.open_stage(usd_path)
        self.stage = stage = ctx.get_stage()
        self._deinstance()
        self.dt = float(m.opt.timestep)
        World.clear_instance()                     # one fresh World per episode
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt, rendering_dt=1.0 / 30.0,
                           backend="numpy", device="cpu")
        # ---------------------------------------------------------- physics scene
        scene = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
        UsdPhysics.Scene(scene).CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        UsdPhysics.Scene(scene).CreateGravityMagnitudeAttr().Set(float(-m.opt.gravity[2]))
        ps = PhysxSchema.PhysxSceneAPI.Apply(scene)
        ps.CreateSolverTypeAttr().Set("TGS")
        ps.CreateEnableCCDAttr().Set(True)
        ps.CreateEnableStabilizationAttr().Set(False)
        ps.CreateFrictionCorrelationDistanceAttr().Set(0.0025)
        ps.CreateFrictionOffsetThresholdAttr().Set(0.004)
        ps.CreateBounceThresholdAttr().Set(0.2)
        ps.CreateTimeStepsPerSecondAttr().Set(int(round(1.0 / self.dt)))
        # -------------------------------------------------- prim <-> MuJoCo maps
        bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
        self.body_path: dict[int, str] = {}
        self.path_body: dict[str, int] = {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetMetadata("displayName") or prim.GetName()
                b = bodies.get(name)
                if b is None:
                    k = _decode(prim.GetName())
                    b = next((i for n, i in bodies.items() if n and _key(n) == k), None)
                if b is not None:
                    self.body_path[b] = str(prim.GetPath())
                    self.path_body[str(prim.GetPath())] = b
        # Bodies MuJoCo welds to the world (no joint, world parent, not an articulation link)
        # are STATIC there: make them true PhysX static colliders (remove RigidBodyAPI and
        # the converter's anchoring fixed joint) instead of jointed dynamic bodies.
        self.static_bodies = set()
        self._dropped_roots: set[str] = set()      # joint-less "articulation" roots (no view)
        for b, path in list(self.body_path.items()):
            prim = stage.GetPrimAtPath(path)
            if m.body_jntnum[b] == 0 and m.body_parentid[b] == 0 and not self._in_articulation(path):
                for c in list(prim.GetChildren()):
                    if c.IsA(UsdPhysics.FixedJoint):
                        c.SetActive(False)
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if prim.HasAPI(UsdPhysics.MassAPI):
                    prim.RemoveAPI(UsdPhysics.MassAPI)
                self.static_bodies.add(b)
        for b in self.static_bodies:
            self.path_body[self.body_path[b]] = b      # contacts on static colliders still map
        self._anchor_fixed_bases()
        self._map_colliders()
        self._bind_materials()
        self._filter_pairs()
        # contact reports on every rigid body; articulation solver iterations
        for b, path in self.body_path.items():
            if b in self.static_bodies:
                continue
            prim = stage.GetPrimAtPath(path)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                # scenes of free-flying bodies only get a root without any joint:
                # those bodies are plain PhysX rigid bodies, not an articulation
                if not any(p.IsA(UsdPhysics.Joint) and p.IsActive() for p in Usd.PrimRange(prim)):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    self._dropped_roots.add(str(prim.GetPath()))
                    continue
                ap = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                ap.CreateSolverPositionIterationCountAttr().Set(physics_iterations)
                ap.CreateSolverVelocityIterationCountAttr().Set(4)
                ap.CreateEnabledSelfCollisionsAttr().Set(True)     # MuJoCo collides robot links too
        self._mimic_joints(UsdPhysics, PhysxSchema)
        self.world.reset()
        self.view = self.world.physics_sim_view
        self._build_views()
        self._classify_actuators()
        self._contacts: dict[tuple[int, int], float] = {}
        self.contact_records: list[dict] = []
        self._patch_contact_monitor()

    # ---------------------------------------------------------------- helpers
    def _in_articulation(self, path: str) -> bool:
        from pxr import UsdPhysics
        prim = self.stage.GetPrimAtPath(path)
        while prim and prim.IsValid():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return True
            prim = prim.GetParent()
        return False

    def _deinstance(self) -> None:
        """Make instanced (shared) mesh prims editable so colliders get materials/filters."""
        for _ in range(8):
            inst = [p for p in self.stage.Traverse() if p.IsInstance()]
            if not inst:
                return
            for p in inst:
                p.SetInstanceable(False)

    def _anchor_fixed_bases(self) -> None:
        """World-welded MuJoCo bodies that root an articulation (e.g. the steering column of
        the wheel tasks) must be a FIXED-BASE PhysX articulation.  The converter puts the
        articulation root on the body itself, which makes PhysX treat the articulation as
        floating and hold it with a compliant world joint (it sags by millimetres under
        load).  The root API is moved onto the world-anchoring fixed joint (PhysX's
        fixed-base convention) and that joint's frame is set to the exact MuJoCo body pose."""
        from pxr import Gf, UsdGeom, UsdPhysics
        m, mj = self.m, self.mj
        d = mj.MjData(m)
        mj.mj_kinematics(m, d)                     # world-welded poses do not depend on qpos
        xc = UsdGeom.XformCache()
        self.n_anchored = 0
        for prim in list(self.stage.Traverse()):
            if not prim.IsA(UsdPhysics.FixedJoint) or not prim.IsActive():
                continue
            j = UsdPhysics.Joint(prim)
            t0 = j.GetBody0Rel().GetTargets()
            p0 = self.stage.GetPrimAtPath(t0[0]) if t0 else None
            if p0 is not None and p0.IsValid() and p0.HasAPI(UsdPhysics.RigidBodyAPI):
                continue                           # not anchored to the world
            t1 = j.GetBody1Rel().GetTargets()
            b = self.path_body.get(str(t1[0])) if t1 else None
            if b is None or m.body_weldid[b] != 0 or b in self.static_bodies:
                continue
            T = Gf.Matrix4d(1.0)
            if p0 is not None and p0.IsValid():
                T = xc.GetLocalToWorldTransform(p0).GetInverse()     # world -> body0 frame
            Mb = Gf.Matrix4d().SetRotate(Gf.Quatd(*map(float, d.xquat[b])))
            Mb.SetTranslateOnly(Gf.Vec3d(*map(float, d.xpos[b])))
            L = Mb * T
            q = L.ExtractRotationQuat()
            j.CreateLocalPos0Attr().Set(Gf.Vec3f(L.ExtractTranslation()))
            j.CreateLocalRot0Attr().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            body = self.stage.GetPrimAtPath(t1[0])
            if body.HasAPI(UsdPhysics.ArticulationRootAPI):
                body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                UsdPhysics.ArticulationRootAPI.Apply(prim)
            self.n_anchored += 1

    def _map_colliders(self) -> None:
        """PhysX collider prim -> MuJoCo geom id (name, mesh name, then nearest primitive)."""
        from pxr import UsdGeom, UsdPhysics
        m, mj = self.m, self.mj
        T = mj.mjtGeom
        types = {"Cube": T.mjGEOM_BOX, "Sphere": T.mjGEOM_SPHERE, "Capsule": T.mjGEOM_CAPSULE,
                 "Cylinder": T.mjGEOM_CYLINDER, "Mesh": T.mjGEOM_MESH, "Plane": T.mjGEOM_PLANE}
        by_body = defaultdict(list)
        for g in range(m.ngeom):
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                by_body[int(m.geom_bodyid[g])].append(g)
        self.body_geoms = {b: np.array(gs, int) for b, gs in by_body.items()}
        gname = {g: _key(n) for g in range(m.ngeom) if (n := mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g))}
        mname = {g: _key(mj.mj_id2name(m, mj.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) or "")
                 for g in range(m.ngeom) if m.geom_type[g] == T.mjGEOM_MESH and m.geom_dataid[g] >= 0}
        xc = UsdGeom.XformCache()
        self.collider_geom: dict[str, int] = {}
        used, unmatched = set(), []
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            path = str(prim.GetPath())
            b = self._body_of_path(path)
            cands = [g for g in by_body.get(b, []) if g not in used]
            names = {_key(prim.GetMetadata("displayName") or ""), _decode(prim.GetName()), _key(prim.GetName()),
                     _decode(prim.GetParent().GetName())} - {""}
            hit = [g for g in cands if gname.get(g) in names]
            if not hit and prim.GetTypeName() == "Mesh":
                hit = [g for g in cands if mname.get(g) in names]
            if not hit:
                same = [g for g in cands if m.geom_type[g] == types.get(prim.GetTypeName(), -1)]
                if same:
                    pw = xc.GetLocalToWorldTransform(prim).ExtractTranslation()
                    if b in self.body_path:
                        Tb = xc.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.body_path[b]))
                        pw = Tb.GetInverse().Transform(pw)
                    rel = np.array([pw[0], pw[1], pw[2]])
                    hit = [min(same, key=lambda g: float(np.linalg.norm(m.geom_pos[g] - rel)))]
            if hit:
                used.add(hit[0])
                self.collider_geom[path] = int(hit[0])
            else:
                unmatched.append(path)
        self.geom_prim = {g: p for p, g in self.collider_geom.items()}
        self.unmapped_colliders = unmatched

    def _friction_modes(self) -> dict:
        """Per-geom PhysX friction combine mode reproducing MuJoCo's rule exactly:
        equal priority -> max(mu1, mu2); different priority -> mu of the higher-priority geom.
        PhysX uses "max" if either material is "max", "min" only if both are "min": geoms that
        take part in a pair where the lower friction must win are put in "min" mode."""
        m = self.m
        geoms = sorted(set(self.collider_geom.values()))
        mu = {g: (0.0 if m.geom_condim[g] == 1 else float(m.geom_friction[g, 0])) for g in geoms}
        pairs_min, pairs_max = [], []
        for i, ga in enumerate(geoms):
            ba = int(m.geom_bodyid[ga])
            for gb in geoms[i + 1:]:
                bb = int(m.geom_bodyid[gb])
                if ba == bb or (m.body_weldid[ba] == 0 and m.body_weldid[bb] == 0) or abs(mu[ga] - mu[gb]) < 1e-9:
                    continue
                if not ((m.geom_contype[ga] & m.geom_conaffinity[gb]) or (m.geom_contype[gb] & m.geom_conaffinity[ga])):
                    continue
                if self._mj_body_filtered(ba, bb):
                    continue
                pa, pb = int(m.geom_priority[ga]), int(m.geom_priority[gb])
                want = max(mu[ga], mu[gb]) if pa == pb else (mu[ga] if pa > pb else mu[gb])
                (pairs_min if want < max(mu[ga], mu[gb]) else pairs_max).append((ga, gb))
        # pairs with a free object (things that are grasped, pushed, stacked, rest on supports)
        # are satisfied first; the other priority pairs (robot links vs furniture, i.e.
        # collisions) only where that does not break a pair that needs "max"
        jointed_desc = set()
        for b in range(m.nbody - 1, 0, -1):          # children have larger ids than parents
            if m.body_jntnum[b] > 0 and m.jnt_type[m.body_jntadr[b]] != self.mj.mjtJoint.mjJNT_FREE \
                    or b in jointed_desc:
                jointed_desc.add(int(m.body_parentid[b]))
        objects = {b for b in range(1, m.nbody) if m.body_jntnum[b] > 0 and b not in jointed_desc
                   and m.jnt_type[m.body_jntadr[b]] == self.mj.mjtJoint.mjJNT_FREE}
        is_obj = lambda g: int(m.geom_bodyid[g]) in objects
        partners = defaultdict(set)
        for ga, gb in pairs_max:
            partners[ga].add(gb)
            partners[gb].add(ga)
        mode = {g: "max" for g in geoms}
        for ga, gb in sorted(pairs_min, key=lambda pr: not (is_obj(pr[0]) or is_obj(pr[1]))):
            if is_obj(ga) or is_obj(gb):
                mode[ga] = mode[gb] = "min"
                continue
            trial = {ga, gb}
            if not any(mode[q] == "min" or q in trial for g in trial for q in partners[g]):
                mode[ga] = mode[gb] = "min"
        self.friction_conflicts = sum(1 for ga, gb in pairs_max if mode[ga] == "min" and mode[gb] == "min")
        self.friction_unmet_min = sum(1 for ga, gb in pairs_min if mode[ga] != "min" or mode[gb] != "min")
        self.friction_object_pairs_exact = all(mode[ga] == mode[gb] == "min" for ga, gb in pairs_min
                                               if is_obj(ga) or is_obj(gb))
        self.friction_min_geoms = sum(1 for v in mode.values() if v == "min")
        return mode

    def _bind_materials(self) -> None:
        """One PhysX material per (MuJoCo friction, combine mode), bound to every collider."""
        from pxr import PhysxSchema, UsdPhysics, UsdShade, UsdGeom
        m = self.m
        self._excl = set(int(x) for x in m.exclude_signature[:m.nexclude])
        modes = self._friction_modes()
        UsdGeom.Scope.Define(self.stage, "/BistlPhysicsMaterials")
        mats = {}
        for path, g in self.collider_geom.items():
            mu = 0.0 if m.geom_condim[g] == 1 else float(m.geom_friction[g, 0])
            key = (round(mu, 4), modes[g])
            if key not in mats:
                mat = UsdShade.Material.Define(self.stage, f"/BistlPhysicsMaterials/mu_{len(mats):03d}")
                pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                pm.CreateStaticFrictionAttr().Set(key[0])
                pm.CreateDynamicFrictionAttr().Set(key[0])
                pm.CreateRestitutionAttr().Set(0.0)
                px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
                px.CreateFrictionCombineModeAttr().Set(key[1])
                px.CreateRestitutionCombineModeAttr().Set("min")
                mats[key] = mat
            prim = self.stage.GetPrimAtPath(path)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[key], UsdShade.Tokens.strongerThanDescendants, "physics")
            if m.geom_condim[g] >= 4 and m.geom_friction[g, 0] > 0:
                r = float(m.geom_friction[g, 1] / m.geom_friction[g, 0])   # torque <= mu_t N
                pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                pc.CreateTorsionalPatchRadiusAttr().Set(r)
                pc.CreateMinTorsionalPatchRadiusAttr().Set(r)
        self.n_materials = len(mats)

    def _mj_body_filtered(self, b1: int, b2: int) -> bool:
        """MuJoCo's body-pair filter: same weld, parent/child weld (filterparent), <exclude>."""
        m, mj = self.m, self.mj
        w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
        if w1 == w2:
            return True
        if not (m.opt.disableflags & mj.mjtDisableBit.mjDSBL_FILTERPARENT) and w1 and w2:
            if w1 == int(m.body_weldid[m.body_parentid[w2]]) or w2 == int(m.body_weldid[m.body_parentid[w1]]):
                return True
        return ((b1 << 16) + b2) in self._excl or ((b2 << 16) + b1) in self._excl

    def _filter_pairs(self) -> None:
        """Reproduce MuJoCo's collision filtering with PhysX filtered collider pairs."""
        from pxr import UsdPhysics
        m = self.m
        self._excl = set(int(x) for x in m.exclude_signature[:m.nexclude])
        cols = sorted(self.collider_geom.items(), key=lambda kv: kv[1])
        n = 0
        rels = defaultdict(list)
        for i, (pa, ga) in enumerate(cols):
            ba = int(m.geom_bodyid[ga])
            for pb, gb in cols[i + 1:]:
                bb = int(m.geom_bodyid[gb])
                if ba == bb or (m.body_weldid[ba] == 0 and m.body_weldid[bb] == 0):
                    continue                     # same rigid body / static-static: PhysX never collides
                mask_ok = (m.geom_contype[ga] & m.geom_conaffinity[gb]) or (m.geom_contype[gb] & m.geom_conaffinity[ga])
                if not mask_ok or self._mj_body_filtered(ba, bb):
                    rels[pa].append(pb)
                    n += 1
        for pa, targets in rels.items():
            rel = UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath(pa)).CreateFilteredPairsRel()
            for pb in targets:
                rel.AddTarget(pb)
        self.n_filtered_pairs = n

    def _joint_of_prim(self, joint_prim) -> int | None:
        """MuJoCo joint id of a USD joint prim (via its child body + decoded name)."""
        from pxr import UsdPhysics
        m = self.m
        j = UsdPhysics.Joint(joint_prim)
        tgt = j.GetBody1Rel().GetTargets()
        if not tgt:
            return None
        b = self.path_body.get(str(tgt[0]))
        if b is None:
            # joint on an intermediate link of a split multi-joint body (export.py): by name
            cands = [j for j in range(m.njnt)
                     if m.jnt_type[j] in (self.mj.mjtJoint.mjJNT_HINGE, self.mj.mjtJoint.mjJNT_SLIDE)]
        else:
            cands = [m.body_jntadr[b] + i for i in range(m.body_jntnum[b])
                     if m.jnt_type[m.body_jntadr[b] + i] in (self.mj.mjtJoint.mjJNT_HINGE, self.mj.mjtJoint.mjJNT_SLIDE)]
        if len(cands) == 1 and b is not None:
            return int(cands[0])
        k = _decode(joint_prim.GetName())
        for c in cands:
            if _key(self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_JOINT, c) or "") == k:
                return int(c)
        return None

    def _mimic_joints(self, UsdPhysics, PhysxSchema) -> None:
        """MuJoCo joint equalities q1 = c0 + c1 q2 -> PhysX mimic: q1 - c1 q2 - c0 = 0."""
        m, mj = self.m, self.mj
        jprim = {}
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                jid = self._joint_of_prim(prim)
                if jid is not None:
                    jprim[jid] = prim
        self.mimic = []
        for e in range(m.neq):
            if m.eq_type[e] != mj.mjtEq.mjEQ_JOINT or not m.eq_active0[e]:
                continue
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            c = m.eq_data[e, :5]
            if j1 not in jprim or j2 < 0 or j2 not in jprim or np.any(np.abs(c[2:]) > 1e-12):
                continue
            p1 = jprim[j1]
            axis = UsdPhysics.Tokens.rotX if p1.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.Tokens.transX
            api = PhysxSchema.PhysxMimicJointAPI.Apply(p1, axis)
            api.GetReferenceJointRel().SetTargets([jprim[j2].GetPath()])
            api.GetGearingAttr().Set(float(-c[1]))
            api.GetOffsetAttr().Set(float(-c[0]))
            self.mimic.append((j1, j2, float(c[0]), float(c[1])))

    # ------------------------------------------------------------ tensor views
    def _build_views(self) -> None:
        from pxr import UsdPhysics
        m, mj = self.m, self.mj
        self.arts = []           # dicts: view, dof->joint map, root body
        in_art = set()
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI) or str(prim.GetPath()) in self._dropped_roots:
                continue
            view = self.view.create_articulation_view(str(prim.GetPath()))
            if view is None or getattr(view, "_backend", True) is None or view.count != 1:
                continue
            if view.link_paths[0][0] in {a["links"][0] for a in self.arts}:
                continue                       # same articulation reached via body and joint
            dof_paths = view.dof_paths[0]
            jmap = []
            for i, p in enumerate(dof_paths):
                prim_p = self.stage.GetPrimAtPath(p) if p else None
                jid = self._joint_of_prim(prim_p) if prim_p is not None and prim_p.IsValid() else None
                jmap.append(-1 if jid is None else jid)
            # fallback (Isaac Sim pip builds): match unmapped DOFs by their decoded joint name
            try:
                names = list(view.dof_names or [])
            except Exception:
                names = []
            jnames = {_key(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""): j for j in range(m.njnt)
                      if m.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)}
            for i, p in enumerate(dof_paths):
                if jmap[i] >= 0:
                    continue
                for cand in (str(p).split("/")[-1] if p else "", names[i] if i < len(names) else ""):
                    if cand and _decode(cand) in jnames:
                        jmap[i] = jnames[_decode(cand)]
                        break
            links = view.link_paths[0]
            for lp in links:
                if lp in self.path_body:
                    in_art.add(self.path_body[lp])
            root_body = self.path_body.get(links[0])
            free = None
            if root_body is not None and m.body_jntnum[root_body] > 0 and \
                    m.jnt_type[m.body_jntadr[root_body]] == mj.mjtJoint.mjJNT_FREE:
                free = int(m.body_jntadr[root_body])
            self.arts.append({"view": view, "jmap": np.array(jmap), "links": links,
                              "root_body": root_body, "free_joint": free,
                              "link_bodies": [self.path_body.get(lp, -1) for lp in links]})
        # free rigid bodies (objects) outside articulations
        self.free_bodies = []
        for b, path in self.body_path.items():
            if b in in_art or m.body_jntnum[b] == 0:
                continue
            j = int(m.body_jntadr[b])
            if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                self.free_bodies.append((b, j, path))
        self.rb_view = (self.view.create_rigid_body_view([p for _, _, p in self.free_bodies])
                        if self.free_bodies else None)
        self.unmapped_dofs = int(sum((a["jmap"] < 0).sum() for a in self.arts))

    # -------------------------------------------------------------- actuation
    def _classify_actuators(self) -> None:
        m, mj = self.m, self.mj
        self.drive_act = {}      # actuator -> (art index, dof index, gear, joint)
        dof_of_joint = {}
        for ai, a in enumerate(self.arts):
            for di, j in enumerate(a["jmap"]):
                if j >= 0:
                    dof_of_joint[int(j)] = (ai, di)
        self.dof_of_joint = dof_of_joint
        for u in range(m.nu):
            if m.actuator_trntype[u] not in (mj.mjtTrn.mjTRN_JOINT, mj.mjtTrn.mjTRN_JOINTINPARENT):
                continue
            j = int(m.actuator_trnid[u, 0])
            if j not in dof_of_joint or m.actuator_dyntype[u] != mj.mjtDyn.mjDYN_NONE:
                continue
            if m.actuator_gaintype[u] != mj.mjtGain.mjGAIN_FIXED or m.actuator_biastype[u] != mj.mjtBias.mjBIAS_AFFINE:
                continue
            b1, b2 = m.actuator_biasprm[u, 1], m.actuator_biasprm[u, 2]
            if b1 < 0 or (b1 == 0 and b2 < 0):
                self.drive_act[u] = (*dof_of_joint[j], float(m.actuator_gear[u, 0]), j)
        # per-dof drive parameters (stiffness/damping/max force) set once
        for ai, a in enumerate(self.arts):
            v = a["view"]
            n = v.max_dofs
            stiff = np.zeros((1, n)); damp = np.zeros((1, n)); fmax = np.full((1, n), 1e9)
            for di, j in enumerate(a["jmap"]):
                if j >= 0:
                    damp[0, di] = m.dof_damping[m.jnt_dofadr[j]]
            for u, (aj, di, g, j) in self.drive_act.items():
                if aj != ai:
                    continue
                stiff[0, di] = -g * g * m.actuator_biasprm[u, 1]
                damp[0, di] += -g * g * m.actuator_biasprm[u, 2]
                lim = np.inf
                if m.actuator_forcelimited[u]:
                    lim = min(lim, abs(g) * max(abs(m.actuator_forcerange[u, 0]), abs(m.actuator_forcerange[u, 1])))
                if m.jnt_actfrclimited[j]:
                    lim = min(lim, max(abs(m.jnt_actfrcrange[j, 0]), abs(m.jnt_actfrcrange[j, 1])))
                fmax[0, di] = lim if np.isfinite(lim) else 1e9
            idx = np.arange(1, dtype=np.int32)
            v.set_dof_stiffnesses(stiff.astype(np.float32), idx)
            v.set_dof_dampings(damp.astype(np.float32), idx)
            v.set_dof_max_forces(fmax.astype(np.float32), idx)
            a["stiff"], a["damp"] = stiff[0], damp[0]
        # rotor/reflected inertia and joint Coulomb friction exactly as in MuJoCo: PhysX's
        # implicit joint friction "effort" (static = dynamic = frictionloss, no viscous part —
        # viscous damping is already in the drive damping); re-synced when the task changes them
        self._refresh_joint_friction()

    # ----------------------------------------------------------- state sync
    def write_state_from_mirror(self) -> None:
        """PRE-EPISODE ONLY: copy the task's reset/settled state into PhysX."""
        m, d = self.m, self.d
        self.mj.mj_forward(m, d)
        idx = np.arange(1, dtype=np.int32)
        for a in self.arts:
            v, jmap = a["view"], a["jmap"]
            q = np.zeros((1, v.max_dofs), np.float32)
            qd = np.zeros((1, v.max_dofs), np.float32)
            for di, j in enumerate(jmap):
                if j >= 0:
                    q[0, di] = d.qpos[m.jnt_qposadr[j]] - m.qpos0[m.jnt_qposadr[j]]
                    qd[0, di] = d.qvel[m.jnt_dofadr[j]]
            v.set_dof_positions(q, idx)
            v.set_dof_velocities(qd, idx)
            v.set_dof_position_targets(q, idx)
            if a["free_joint"] is not None:
                b = a["root_body"]
                pose, vel = self._free_to_physx(b, a["free_joint"])
                v.set_root_transforms(pose[None].astype(np.float32), idx)
                v.set_root_velocities(vel[None].astype(np.float32), idx)
        if self.rb_view is not None:
            poses, vels = [], []
            for b, j, _ in self.free_bodies:
                p, vv = self._free_to_physx(b, j)
                poses.append(p); vels.append(vv)
            ridx = np.arange(len(self.free_bodies), dtype=np.int32)
            self.rb_view.set_transforms(np.asarray(poses, np.float32), ridx)
            self.rb_view.set_velocities(np.asarray(vels, np.float32), ridx)

    def _free_to_physx(self, b, j):
        """MuJoCo free joint (origin vel world, ang vel LOCAL) -> PhysX (COM lin vel, world ang)."""
        m, d = self.m, self.d
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        pos, quat = d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7]
        R = quat_to_mat(quat)
        w_world = R @ d.qvel[va + 3:va + 6]
        com = R @ m.body_ipos[b]
        v_com = d.qvel[va:va + 3] + np.cross(w_world, com)
        return np.r_[pos, quat_wxyz_to_xyzw(quat)], np.r_[v_com, w_world]

    def _physx_to_free(self, j, b, pose, vel):
        m, d = self.m, self.d
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        quat = quat_xyzw_to_wxyz(pose[3:7])
        quat /= np.linalg.norm(quat)
        R = quat_to_mat(quat)
        d.qpos[qa:qa + 3] = pose[:3]
        d.qpos[qa + 3:qa + 7] = quat
        w_world = vel[3:6]
        d.qvel[va:va + 3] = vel[:3] - np.cross(w_world, R @ m.body_ipos[b])
        d.qvel[va + 3:va + 6] = R.T @ w_world

    def sync_from_physx(self) -> None:
        m, d = self.m, self.d
        for a in self.arts:
            v, jmap = a["view"], a["jmap"]
            q = np.asarray(v.get_dof_positions())[0]
            qd = np.asarray(v.get_dof_velocities())[0]
            for di, j in enumerate(jmap):
                if j >= 0:
                    d.qpos[m.jnt_qposadr[j]] = q[di] + m.qpos0[m.jnt_qposadr[j]]
                    d.qvel[m.jnt_dofadr[j]] = qd[di]
            if a["free_joint"] is not None:
                self._physx_to_free(a["free_joint"], a["root_body"],
                                    np.asarray(v.get_root_transforms())[0], np.asarray(v.get_root_velocities())[0])
        if self.rb_view is not None:
            P = np.asarray(self.rb_view.get_transforms())
            V = np.asarray(self.rb_view.get_velocities())
            for k, (b, j, _) in enumerate(self.free_bodies):
                self._physx_to_free(j, b, P[k], V[k])
        d.time += self.dt
        self.mj.mj_forward(m, d)

    # -------------------------------------------------------------- commands
    def _refresh_joint_friction(self) -> None:
        """Tasks may change joint friction loss / armature at run time (e.g. a task that
        switches dof_frictionloss): keep PhysX's implicit joint friction in sync."""
        m = self.m
        if hasattr(self, "_floss") and np.array_equal(self._floss, m.dof_frictionloss) \
                and np.array_equal(self._farm, m.dof_armature):
            return
        self._floss, self._farm = m.dof_frictionloss.copy(), m.dof_armature.copy()
        idx = np.arange(1, dtype=np.int32)
        for a in self.arts:
            v = a["view"]
            n = v.max_dofs
            arm = np.zeros((1, n), np.float32)
            fr = np.zeros((1, n, 3), np.float32)
            for di, j in enumerate(a["jmap"]):
                if j >= 0:
                    arm[0, di] = m.dof_armature[m.jnt_dofadr[j]]
                    fr[0, di, :2] = m.dof_frictionloss[m.jnt_dofadr[j]]
            v.set_dof_armatures(arm, idx)
            v.set_dof_friction_properties(fr, idx)

    def _refresh_drive_limits(self) -> None:
        self._refresh_joint_friction()
        m = self.m
        if not hasattr(self, "_frange") or not np.array_equal(self._frange, m.actuator_forcerange):
            self._frange = m.actuator_forcerange.copy()
            for ai, a in enumerate(self.arts):
                n = a["view"].max_dofs
                fmax = np.full((1, n), 1e9, np.float32)
                for u, (aj, di, g, j) in self.drive_act.items():
                    if aj != ai:
                        continue
                    lim = np.inf
                    if m.actuator_forcelimited[u]:
                        lim = min(lim, abs(g) * max(abs(m.actuator_forcerange[u, 0]), abs(m.actuator_forcerange[u, 1])))
                    if m.jnt_actfrclimited[j]:
                        lim = min(lim, max(abs(m.jnt_actfrcrange[j, 0]), abs(m.jnt_actfrcrange[j, 1])))
                    fmax[0, di] = lim if np.isfinite(lim) else 1e9
                a["view"].set_dof_max_forces(fmax, np.arange(1, dtype=np.int32))

    def apply_controls(self) -> None:
        """Translate the task's MjData.ctrl / xfrc_applied into PhysX drive targets and forces."""
        m, d, mj = self.m, self.d, self.mj
        self._refresh_drive_limits()
        mj.mj_fwdActuation(m, d)                     # actuator forces for the current ctrl/state
        explicit = d.qfrc_actuator.copy()
        explicit += d.qfrc_passive - d.qfrc_damper   # springs, fluid drag, plain gravcomp
        idx = np.arange(1, dtype=np.int32)
        targets = [np.zeros(a["view"].max_dofs, np.float32) for a in self.arts]
        vtargets = [np.zeros(a["view"].max_dofs, np.float32) for a in self.arts]
        for u, (ai, di, g, j) in self.drive_act.items():
            dof = m.jnt_dofadr[j]
            fd = g * d.actuator_force[u]
            if m.jnt_actfrclimited[j]:      # qfrc_actuator is clamped per joint (actuatorfrcrange)
                fd = float(np.clip(fd, m.jnt_actfrcrange[j, 0], m.jnt_actfrcrange[j, 1]))
            explicit[dof] -= fd              # the PhysX drive (max force = same limit) supplies it
            ctrl = d.ctrl[u]
            if m.actuator_ctrllimited[u]:
                ctrl = np.clip(ctrl, m.actuator_ctrlrange[u, 0], m.actuator_ctrlrange[u, 1])
            f0 = m.actuator_gainprm[u, 0] * ctrl + m.actuator_biasprm[u, 0]
            k, c = -g * g * m.actuator_biasprm[u, 1], -g * g * m.actuator_biasprm[u, 2]
            if k > 0:
                targets[ai][di] = g * f0 / k - m.qpos0[m.jnt_qposadr[j]]
            elif c > 0:
                vtargets[ai][di] = g * f0 / c
        for ai, a in enumerate(self.arts):
            v, jmap = a["view"], a["jmap"]
            eff = np.zeros(v.max_dofs, np.float32)
            for di, j in enumerate(jmap):
                if j >= 0:
                    eff[di] = explicit[m.jnt_dofadr[j]]
            v.set_dof_actuation_forces(eff[None], idx)
            v.set_dof_position_targets(targets[ai][None], idx)
            v.set_dof_velocity_targets(vtargets[ai][None], idx)
            if a["free_joint"] is not None:
                self._apply_free_wrench(v, a["free_joint"], a["root_body"], explicit, link_view=True)
        if self.rb_view is not None:
            F = np.zeros((len(self.free_bodies), 3), np.float32)
            T = np.zeros((len(self.free_bodies), 3), np.float32)
            P = np.zeros((len(self.free_bodies), 3), np.float32)
            for k, (b, j, _) in enumerate(self.free_bodies):
                f, t, p = self._free_wrench(j, b, explicit)
                F[k], T[k], P[k] = f, t, p
            self.rb_view.apply_forces_and_torques_at_position(F, T, P, np.arange(len(F), dtype=np.int32), True)
        self._apply_xfrc()

    def _free_wrench(self, j, b, explicit):
        m, d = self.m, self.d
        va = m.jnt_dofadr[j]
        R = d.xmat[b].reshape(3, 3)
        force = explicit[va:va + 3]
        torque = R @ explicit[va + 3:va + 6]
        return force, torque, d.xpos[b]

    def _apply_free_wrench(self, view, j, b, explicit, link_view):
        f, t, p = self._free_wrench(j, b, explicit)
        if not (np.any(f) or np.any(t)):
            return
        n = view.max_links
        F = np.zeros((1, n, 3), np.float32); T = np.zeros((1, n, 3), np.float32); P = np.zeros((1, n, 3), np.float32)
        F[0, 0], T[0, 0], P[0, 0] = f, t, p
        view.apply_forces_and_torques_at_position(F, T, P, np.arange(1, dtype=np.int32), True)

    def _apply_xfrc(self) -> None:
        d = self.d
        bodies = np.flatnonzero(np.any(d.xfrc_applied != 0, axis=1))
        if not len(bodies):
            return
        for b in bodies:
            f, t, com = d.xfrc_applied[b, :3], d.xfrc_applied[b, 3:], d.xipos[b]
            for a in self.arts:
                if b in a["link_bodies"]:
                    n = a["view"].max_links
                    li = a["link_bodies"].index(b)
                    F = np.zeros((1, n, 3), np.float32); T = np.zeros((1, n, 3), np.float32); P = np.zeros((1, n, 3), np.float32)
                    F[0, li], T[0, li], P[0, li] = f, t, com
                    a["view"].apply_forces_and_torques_at_position(F, T, P, np.arange(1, dtype=np.int32), True)
                    break
            else:
                n = len(self.free_bodies)
                for k, (bb, _, _) in enumerate(self.free_bodies):
                    if bb == b:
                        F = np.zeros((n, 3), np.float32); T = np.zeros((n, 3), np.float32)
                        P = np.zeros((n, 3), np.float32)
                        F[k], T[k], P[k] = f, t, com
                        self.rb_view.apply_forces_and_torques_at_position(F, T, P, np.arange(n, dtype=np.int32), True)

    # --------------------------------------------------------------- contacts
    def read_contacts(self) -> None:
        """PhysX contact report of the last step -> per collider pair records (forces in N)."""
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        headers, data, anchors = get_physx_simulation_interface().get_full_contact_report()
        forces: dict[tuple[int, int], float] = defaultdict(float)
        cache = self._actor_cache = getattr(self, "_actor_cache", {})
        gcache = self._collider_cache = getattr(self, "_collider_cache", {})
        weld = self.m.body_weldid
        recs = []
        inv = 1.0 / self.dt
        for h in headers:
            if h.num_contact_data == 0:
                continue
            pair = []
            for a in (h.actor0, h.actor1):
                if a not in cache:
                    cache[a] = self._body_of_path(str(PhysicsSchemaTools.intToSdfPath(a)))
                pair.append(cache[a])
            # MuJoCo never generates contacts between two bodies welded to the world
            if weld[pair[0]] == 0 and weld[pair[1]] == 0:
                continue
            gs = []
            for c in (h.collider0, h.collider1):
                if c not in gcache:
                    gcache[c] = self.collider_geom.get(str(PhysicsSchemaTools.intToSdfPath(c)), -1)
                gs.append(gcache[c])
            fvec = np.zeros(3)
            fn_tot = 0.0
            pts = []
            for c in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                cd = data[c]
                imp = np.array(cd.impulse, float)
                fn = abs(float(imp @ np.array(cd.normal, float))) * inv
                fvec += imp
                fn_tot += fn
                pts.append((tuple(cd.position), fn))
            for c in range(h.friction_anchors_offset, h.friction_anchors_offset + h.num_friction_anchors_data):
                fvec += np.array(anchors[c].impulse, float)
            recs.append({"b0": pair[0], "b1": pair[1], "g0": gs[0], "g1": gs[1], "fn": fn_tot,
                         "f0": fvec * inv, "n": int(h.num_contact_data), "points": pts})
            key = (min(pair), max(pair))
            forces[key] += fn_tot
        self.contact_records = recs
        self._contacts = dict(forces)

    def _body_of_path(self, path: str):
        while path and path != "/":
            if path in self.path_body:
                return self.path_body[path]
            path = path.rsplit("/", 1)[0]
        return 0          # static world geometry

    def body_pair_force(self, bodies_a: set, bodies_b: set) -> tuple[float, int]:
        tot, n = 0.0, 0
        for (a, b), f in self._contacts.items():
            if (a in bodies_a and b in bodies_b) or (b in bodies_a and a in bodies_b):
                tot += f
                n += 1
        return tot, n

    def _patch_contact_monitor(self) -> None:
        from eddies.isaac import sensing
        sensing.install(self)

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        self.apply_controls()
        self.world.step(render=False)
        self.read_contacts()
        self.sync_from_physx()

    def pose_error(self) -> float:
        """Max |PhysX link position - mirror body position| (state-mapping check)."""
        err = 0.0
        for a in self.arts:
            T = np.asarray(a["view"].get_link_transforms())[0]
            for li, b in enumerate(a["link_bodies"]):
                if b >= 0:
                    err = max(err, float(np.linalg.norm(T[li, :3] - self.d.xpos[b])))
        return err

    def diagnostics(self) -> dict:
        return {"articulations": len(self.arts), "free_bodies": len(self.free_bodies),
                "unmapped_dofs": self.unmapped_dofs, "drive_actuators": len(self.drive_act),
                "explicit_actuators": int(self.m.nu - len(self.drive_act)), "mimic_joints": len(self.mimic),
                "colliders_mapped": len(self.collider_geom), "colliders_unmapped": len(self.unmapped_colliders),
                "collidable_geoms": int(sum(len(v) for v in self.body_geoms.values())),
                "friction_materials": self.n_materials, "filtered_collider_pairs": self.n_filtered_pairs,
                "friction_min_mode_geoms": self.friction_min_geoms, "friction_pairs_max_violated": self.friction_conflicts,
                "friction_pairs_priority_unmet": self.friction_unmet_min,
                "friction_object_pairs_exact": self.friction_object_pairs_exact}
