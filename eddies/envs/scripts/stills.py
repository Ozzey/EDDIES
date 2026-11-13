#!/usr/bin/env python3
"""Render still images of a task's cameras after reset (camera framing check).

    python -m eddies.envs.scripts.stills --task franka_cube_stack --seed 0 \
        --cams main,detail --out /tmp/stills [--steps N --method bistl]

With --steps N the given method is run for N control steps first, so the
framing can be checked mid-task (e.g. while opening a drawer).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from eddies.common.paths import SCRATCH

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cams", default="main")
    ap.add_argument("--out", default=str(SCRATCH / "stills"))
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--method", default="bistl")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    import mujoco
    from eddies.core.ltlf import Monitor
    from eddies.envs.framework.recorder import render_still
    from eddies.envs.methods import make_method
    from eddies.envs.tasks import make_task
    task = make_task(a.task, a.seed)
    task.reset()
    if a.steps:
        method = make_method(a.method, task)
        mon = Monitor(task.formula)
        labels = frozenset(task.labels())
        mon.step(labels)
        obs = task.observe()
        obs.update(labels=labels, ltl_residual=mon.state, ltl_status=mon.status, t=task.time)
        method.reset(obs)
        dec = max(1, int(round(task.control_dt / task.model.opt.timestep)))
        for k in range(a.steps * dec):
            if k % dec == 0:
                obs = task.observe()
                obs.update(labels=labels, ltl_residual=mon.state, ltl_status=mon.status, t=task.time)
                task.apply_command(method.act(obs))
            task.inner_loop()
            mujoco.mj_step(task.model, task.data)
            labels = frozenset(task.labels())
            mon.step(labels)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cam in a.cams.split(","):
        path = out / f"{a.task}_s{a.seed}_{cam}_step{a.steps}.png"
        render_still(task.model, task.data, cam, path, a.width, a.height)
        print(path)


if __name__ == "__main__":
    main()
