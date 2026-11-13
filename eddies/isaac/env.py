"""Paths, interpreter set-up and MuJoCo 3.8 compatibility shims for Isaac Sim."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ISAAC_ROOT = Path(os.environ.get("ISAAC_INSTALL_ROOT", "~/isaacsim")).expanduser()
ISAAC_PY = ISAAC_ROOT / "python.sh"
PKG = Path(__file__).resolve().parent                 # eddies/isaac
ISAAC_DIR = PKG.parent                                 # eddies/
REPO = ISAAC_DIR.parent                                # repository root
CACHE = Path(os.environ.get("EDDIES_ISAAC_CACHE", REPO / ".cache" / "isaac_usd"))   # converted USD scenes
NEWTON_PREBUNDLE = ISAAC_ROOT / "exts" / "isaacsim.pip.newton" / "pip_prebundle"


def setup_python_path() -> None:
    """Make eddies.core (math core), eddies.envs (task descriptions, methods) and the bundled
    mujoco importable inside Isaac's interpreter.  Nothing is installed."""
    for p in (str(REPO),):
        if p not in sys.path:
            sys.path.insert(0, p)
    if str(NEWTON_PREBUNDLE) not in sys.path:
        sys.path.append(str(NEWTON_PREBUNDLE))


_SHIMMED = False


def apply_mujoco_shims() -> None:
    """Runtime adapters for the MuJoCo 3.15 API used by the (unchanged) task code:

    * ``mj_fullM(m, data, dst)`` (3.15) -> ``mj_fullM(m, dst, data.qM)`` (3.8);
    * ``MjSpec.from_file`` makes mesh/texture paths absolute so the generated scene MJCF
      can be converted from any directory.
    """
    global _SHIMMED
    if _SHIMMED:
        return
    import mujoco
    orig_full = mujoco.mj_fullM

    def mj_fullM(m, a, b):
        if isinstance(a, mujoco.MjData):
            return orig_full(m, b, a.qM)
        return orig_full(m, a, b)

    mujoco.mj_fullM = mj_fullM
    orig_from_file = mujoco.MjSpec.from_file

    def from_file(path, *args, **kwargs):
        spec = orig_from_file(path, *args, **kwargs)
        base = os.path.dirname(os.path.abspath(path))
        md, td = spec.meshdir or "", spec.texturedir or ""
        for m in spec.meshes:
            if m.file and not os.path.isabs(m.file):
                m.file = os.path.normpath(os.path.join(base, md, m.file))
        for t in spec.textures:
            if t.file and not os.path.isabs(t.file):
                t.file = os.path.normpath(os.path.join(base, td, t.file))
        spec.meshdir, spec.texturedir = "", ""
        return spec

    mujoco.MjSpec.from_file = staticmethod(from_file)
    _SHIMMED = True
