#!/usr/bin/env python3
"""Record a video of one episode (MuJoCo, EGL off-screen rendering).

    MUJOCO_GL=egl python -m eddies.render --task franka_cube_stack --seed 2000 [--method bistl] \
        [--rho-scale 2] [--perturb] [--horizon 20] [--width 1920 --height 1080] [--out DIR]

Writes <out>/<task>_s<seed>[_<label>]/{video.mp4, manifest.json, trajectory.npz}.  The Streamer's
Eddie hand-overs are in manifest["method_info"]["skills"] (name, outcome, start / end step), which the
video overlays of assets/videos use.  One episode takes about a minute; for batches use a multi-core machine
(MUJOCO_GL=egl when headless).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from eddies.common.config import load
from eddies.common.paths import SCRATCH


def main(argv=None):
    cfg = load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--method", default=cfg.method)
    ap.add_argument("--rho-scale", type=float, default=None, help="multiply the task's slew bound (faster commands)")
    ap.add_argument("--budget", type=int, default=None, help="failure budget of the supervisor")
    ap.add_argument("--perturb", action="store_true", help="perturbed condition")
    ap.add_argument("--horizon", type=float, default=None, help="seconds (default: the task's horizon)")
    ap.add_argument("--width", type=int, default=cfg.sim.video.width)
    ap.add_argument("--height", type=int, default=cfg.sim.video.height)
    ap.add_argument("--label", default=None, help="suffix of the output folder")
    ap.add_argument("--out", default=str(SCRATCH / "render"))
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    import eddies.envs.tasks as tasks_mod
    from eddies.envs.framework.runner import run_episode

    orig = tasks_mod.make_task

    def patched(name, seed, **kw):
        t = orig(name, seed, **kw)
        if a.rho_scale is not None:
            t.rho_step = float(type(t).rho_step) * float(a.rho_scale)
        if a.budget is not None:
            t.failure_budget = int(a.budget)
        return t

    tasks_mod.make_task = patched
    label = a.label or (f"rho{a.rho_scale:g}" if a.rho_scale else "") + ("_perturbed" if a.perturb else "")
    out = Path(a.out) / (f"{a.task}_s{a.seed}" + (f"_{label}" if label else ""))
    try:
        man = run_episode(a.task, a.method, a.seed, out, record=True, width=a.width, height=a.height,
                          horizon=a.horizon, save_trajectory=True, verbose=True,
                          task_kwargs={"perturb": True} if a.perturb else None)
    finally:
        tasks_mod.make_task = orig
    man["render_overrides"] = {"rho_scale": a.rho_scale, "budget": a.budget, "perturb": a.perturb}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"{a.task} seed {a.seed}: status={man.get('status')} success={man.get('success')} "
          f"stop={man.get('stop_reason')} -> {out}")
    return 0 if man.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
