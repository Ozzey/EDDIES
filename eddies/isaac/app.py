"""Launch Isaac Sim (headless, RTX) — must be imported/called before any omni/pxr import."""
from __future__ import annotations

_APP = None


def launch(width: int = 1920, height: int = 1080, headless: bool = True):
    """Start SimulationApp once per process and enable the extensions we need.

    Keep DISPLAY set: the NVIDIA Vulkan ICD may need it even for headless offscreen rendering.
    """
    global _APP
    if _APP is not None:
        return _APP
    from isaacsim import SimulationApp
    _APP = SimulationApp({"headless": headless, "width": width, "height": height,
                         "anti_aliasing": 3, "renderer": "RaytracedLighting"})
    # enable one by one: a missing extension (the pip build of Isaac Sim 6.0.1 has no
    # isaacsim.pip.newton; Newton/MuJoCo come as pip packages) must not fail the whole batch
    import omni.kit.app
    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in ("isaacsim.asset.importer.utils", "isaacsim.asset.importer.mjcf", "isaacsim.pip.newton",
                "omni.replicator.core"):
        try:
            ok = mgr.set_extension_enabled_immediate(ext, True)
            print(f"[eddies] extension {ext}: {'enabled' if ok else 'not available'}", flush=True)
        except Exception as ex:      # never abort the launch for an optional extension
            print(f"[eddies] extension {ext}: skipped ({ex!r})", flush=True)
    _APP.update()
    from eddies.isaac.env import apply_mujoco_shims, setup_python_path
    setup_python_path()
    apply_mujoco_shims()
    return _APP


def close() -> None:
    """Isaac 6.0.1: stop Kit's loop before close() to avoid the known teardown hang."""
    global _APP
    if _APP is None:
        return
    try:
        import omni.kit.app
        omni.kit.app.get_app().post_quit()
    except Exception:
        pass
    try:
        _APP.close()
    finally:
        _APP = None
