"""Shared physical settings.

Every scene in the pipeline uses real gravity, finite-friction contact and
physical joints/actuators.  No object is welded, teleported or made kinematic
during an episode; grasps hold only through contact friction.

Friction triples follow MuJoCo's (sliding, torsional [m], rolling [m]) layout.
MuJoCo combines two geoms' coefficients with the element-wise MAXIMUM, so the
pair coefficient is that of the grippier surface.  The values are typical
engineering coefficients for dry contact, not identified measurements.
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.common.config import load

SIM = load().sim

GRAVITY = (0.0, 0.0, -9.81)
AIR_DENSITY = 1.204        # kg/m^3 at 20 C
AIR_VISCOSITY = 1.81e-5    # Pa s

FRICTION = {
    "rubber": (1.1, 0.010, 1e-4),     # finger pads, robot feet, tyres
    "skin": (0.9, 0.008, 1e-4),       # silicone/TPU robot-hand skins
    "wood": (0.5, 0.005, 1e-4),
    "plastic": (0.4, 0.004, 1e-4),
    "metal": (0.35, 0.003, 1e-4),
    "glass": (0.3, 0.003, 1e-4),
    "table": (0.6, 0.005, 1e-4),     # laminate tabletop
    "floor": (0.9, 0.005, 1e-4),     # concrete/rubber floor for legged robots
    "cardboard": (0.55, 0.005, 1e-4),
}

DENSITY = {  # kg/m^3
    "wood": 600.0,
    "plastic": 1050.0,
    "aluminium": 2700.0,
    "steel": 7850.0,
    "foam": 120.0,
    "water_bottle": 1000.0,
}


def configure_options(spec: mujoco.MjSpec, *, timestep: float = SIM.physics_dt,
                      cone: str = SIM.physics.cone, impratio: float = SIM.physics.impratio,
                      integrator: str = SIM.physics.integrator, iterations: int = SIM.physics.iterations,
                      ls_iterations: int = SIM.physics.ls_iterations, air: bool = SIM.physics.air,
                      noslip_iterations: int = SIM.physics.noslip_iterations) -> None:
    """Apply the common physics options (gravity, solver, friction cone).

    Elliptic cones with impratio>1 are MuJoCo's recommended setting for
    slip-resistant frictional grasping; noslip is kept off so grasps are not
    artificially stiffened.
    """
    o = spec.option
    o.timestep = float(timestep)
    o.gravity = np.asarray(GRAVITY, float)
    o.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
              "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[cone]
    o.impratio = float(impratio)
    o.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                    "euler": mujoco.mjtIntegrator.mjINT_EULER,
                    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT}[integrator]
    o.iterations = int(iterations)
    o.ls_iterations = int(ls_iterations)
    o.noslip_iterations = int(noslip_iterations)
    if air:
        o.density = AIR_DENSITY
        o.viscosity = AIR_VISCOSITY


def configure_visual(spec: mujoco.MjSpec, width: int = 1920, height: int = 1080) -> None:
    """High-resolution offscreen buffer, soft shadows and anti-aliasing."""
    g = spec.visual.global_
    g.offwidth = int(width)
    g.offheight = int(height)
    q = spec.visual.quality
    q.shadowsize = 4096
    q.offsamples = 8
    spec.visual.headlight.ambient = np.array([0.35, 0.35, 0.35])
    spec.visual.headlight.diffuse = np.array([0.55, 0.55, 0.55])
    spec.visual.headlight.specular = np.array([0.1, 0.1, 0.1])
    spec.visual.map.znear = 0.005
    spec.visual.map.zfar = 50.0


def physics_summary(model: mujoco.MjModel) -> dict:
    """Recorded in every episode manifest."""
    o = model.opt
    return {
        "timestep": float(o.timestep),
        "gravity": [float(x) for x in o.gravity],
        "cone": "elliptic" if o.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal",
        "impratio": float(o.impratio),
        "integrator": int(o.integrator),
        "iterations": int(o.iterations),
        "noslip_iterations": int(o.noslip_iterations),
        "air_density": float(o.density),
        "air_viscosity": float(o.viscosity),
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "nbody": int(model.nbody), "ngeom": int(model.ngeom),
    }
