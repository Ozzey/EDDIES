#!/usr/bin/env python3
"""Clean main-camera stills in Isaac Sim (PhysX, RTX), matching eddies.envs.scripts.stills.

    python -m eddies.isaac.stills --tasks franka_fmb:250,franka_peg_insertion:150 --seed 2000 --out DIR
    python -m eddies.isaac.stills --tasks franka_peg_insertion:330 --cam detail --at 250,280,310 --out DIR

Each task is run with the Streamer for the given number of control steps (same seed and step count as the
MuJoCo still) and the last raw frame of the main camera is saved, without the inset camera or the
status overlay of the video recorder: DIR/<task>_s<seed>_main_step<n>.png.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # isaac/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="task:steps,task:steps,...")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam", default="main", choices=("main", "detail"))
    ap.add_argument("--at", default=None, help="also save frames at these control steps (comma list)")
    a = ap.parse_args()
    from eddies.isaac.app import close, launch
    launch(a.width, a.height)
    import numpy as np
    from PIL import Image

    import eddies.isaac.recorder as R
    from eddies.isaac.runner import run_episode
    from eddies.envs.tasks import make_task

    last, marks = {}, {}
    orig, orig_init = R.IsaacRecorder.capture, R.IsaacRecorder.__init__

    def init(self, *args, **kw):                     # render the close-up camera at full size, not as an inset
        if a.cam == "detail":
            kw["inset_scale"] = 1.0
        orig_init(self, *args, **kw)

    def capture(self, title="", lines=None):
        orig(self, title, lines)
        img = self._grab(a.cam)
        if img is None:
            return
        img = np.ascontiguousarray(img, dtype=np.uint8)
        last["img"] = img
        t = float(self.bridge.d.time)
        for k, ts in marks.items():                  # first frame at or after each requested time
            if k not in last and t >= ts:
                last[k] = img

    R.IsaacRecorder.__init__ = init
    R.IsaacRecorder.capture = capture
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.tasks.split(","):
        task, steps = spec.split(":")
        last.clear()
        marks.clear()
        try:
            dt = float(make_task(task, a.seed).control_dt)
            for k in (a.at.split(",") if a.at else []):
                marks[int(k)] = int(k) * dt
            run_episode(task, "bistl", a.seed, out / f"run_{task}", record=True, width=a.width, height=a.height,
                        horizon=int(steps) * dt, save_trajectory=False)
            for k in marks:
                if k in last:
                    path = out / f"{task}_s{a.seed}_{a.cam}_step{k}.png"
                    Image.fromarray(last[k]).save(path)
                    print("[stills] wrote", path, flush=True)
            if "img" in last and not marks:
                path = out / f"{task}_s{a.seed}_{a.cam}_step{steps}.png"
                Image.fromarray(last["img"]).save(path)
                print("[stills] wrote", path, flush=True)
            else:
                print("[stills] no frame for", task, flush=True)
        except Exception:
            print("[stills] FAILED", task, flush=True)
            traceback.print_exc()
    close()


if __name__ == "__main__":
    main()
