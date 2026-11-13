#!/usr/bin/env python3
"""Developer tool: run a task with several methods/seeds and print a compact table.

    python -m eddies.envs.scripts.try_task --task franka_cube_stack \
        --methods bistl --seeds 0-9 --workers 8 [--video] [--out DIR]

Episodes are written under --out (default: a scratch dir, NOT results/episodes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from eddies.common.paths import SCRATCH


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _job(args):
    task, method, seed, out, video, horizon, perturb = args
    import warnings
    warnings.filterwarnings("ignore")
    from eddies.envs.framework.runner import run_episode
    cond = "perturbed" if perturb else "nominal"
    m = run_episode(task, method, seed, Path(out) / cond / task / method / f"seed_{seed:02d}",
                    record=video, horizon=horizon, verbose=False,
                    task_kwargs={"perturb": True} if perturb else None)
    keep = ["task", "method", "seed", "status", "success", "task_success", "property_success",
            "stop_reason", "completion_time_s", "sim_time_s", "bad_steps", "max_command_jump",
            "wall_time_s", "error"]
    r = {k: m.get(k) for k in keep}
    mi = m.get("method_info") or {}
    r["skills"] = [(s["skill"], s["outcome"]) for s in mi.get("skills", [])]
    r["compute_ms"] = (m.get("compute_ms_per_control_step") or {}).get("mean")
    r["traceback"] = m.get("traceback")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--methods", default="bistl")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--out", default=str(SCRATCH / "try"))
    ap.add_argument("--verbose", action="store_true", help="print skill sequences and tracebacks")
    ap.add_argument("--perturb", action="store_true", help="stress-test condition (task.disturb)")
    a = ap.parse_args()
    methods = a.methods.split(",")
    seeds = parse_seeds(a.seeds)
    jobs = [(a.task, m, s, a.out, a.video, a.horizon, a.perturb) for m in methods for s in seeds]
    t0 = time.time()
    results = []
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:  # worker crash
                results.append({"status": "crash", "error": repr(e)})
    results.sort(key=lambda r: (str(r.get("method")), r.get("seed") or 0))
    print(f"\n{a.task}: {len(jobs)} episodes in {time.time() - t0:.1f}s  (out: {a.out})")
    for m in methods:
        rs = [r for r in results if r.get("method") == m]
        ok = sum(bool(r.get("success")) for r in rs)
        times = [r["completion_time_s"] for r in rs if r.get("completion_time_s")]
        errs = sum(r.get("status") != "completed" for r in rs)
        tt = f"{sum(times) / len(times):6.2f}s" if times else "   -  "
        print(f"  {m:14s} success {ok:2d}/{len(rs):2d}   mean completion {tt}   errors {errs}")
        for r in rs:
            flag = "OK " if r.get("success") else "-- "
            print(f"     seed {r.get('seed')}: {flag} {r.get('stop_reason')}  t={r.get('completion_time_s')}"
                  f"  bad={r.get('bad_steps')}  jump={r.get('max_command_jump') and round(r['max_command_jump'], 3)}"
                  f"  ms={r.get('compute_ms') and round(r['compute_ms'], 2)}")
            if a.verbose:
                print("        skills:", r.get("skills"))
                if r.get("traceback"):
                    print(r["traceback"])
    errors = [r for r in results if r.get("status") != "completed"]
    if errors and not a.verbose:
        print("\nFirst error:", errors[0].get("error"))
        print(errors[0].get("traceback") or "")
    json.dump(results, open(Path(a.out) / f"{a.task}_try_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
