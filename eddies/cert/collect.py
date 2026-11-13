#!/usr/bin/env python3
"""Run instrumented Streamer episodes without video.

    python -m eddies.cert.collect --tasks franka_drawer --seeds 1000-1099 \
        --conditions nominal,perturbed --tag train --workers 20

Episodes go to $EDDIES_EPISODES/<tag>/<condition>/<task>/<method>/seed_XXXX/
{manifest.json, trajectory.npz}.  Overrides (all optional, recorded in the manifest):
    --rho-scale s     multiply the task's per-step slew bound rho_step by s
    --budget F        failure budget of the supervisor
    --method m        bistl (default), bistl_noslew, bistl_nosup, ...
    --spec "..."      LTLf specification for a composed task (*_ltl); use with --label
Completed episodes are skipped, so the command can be re-run to resume.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

from eddies.common.config import load
from eddies.common.paths import EPISODES, ROOT, episode_dir, parse_seeds  # noqa: F401  (re-exported)

CFG = load()


def _job(args):
    tag, condition, task, method, seed, rho_scale, budget, label, params, spec = args
    import warnings
    warnings.filterwarnings("ignore")
    import eddies.envs.tasks as tasks_mod
    from eddies.envs.framework.runner import run_episode

    orig = tasks_mod.make_task

    def patched(name, s, **kw):
        t = orig(name, s, **kw)
        if rho_scale is not None:
            t.rho_step = float(type(t).rho_step) * float(rho_scale)
        if budget is not None:
            t.failure_budget = int(budget)
        return t

    tasks_mod.make_task = patched
    mkw = {}
    if params:
        P = json.loads(Path(params).read_text())
        mkw = {k: P[k] for k in ("skill_rho", "skill_delta", "skill_timeout") if k in P}
        if P.get("failure_budget") is not None and budget is None:
            budget = int(P["failure_budget"])
    out = episode_dir(tag, condition, task, label or method, seed)
    t0 = time.time()
    try:
        man = run_episode(task, method, seed, out, record=False, save_trajectory=True,
                          task_kwargs={"perturb": condition == "perturbed", **({"spec": spec} if spec else {})},
                          method_kwargs=mkw)
    finally:
        tasks_mod.make_task = orig
    man["cert_overrides"] = {"rho_scale": rho_scale, "budget": budget, "tag": tag, "params": params, "spec": spec}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    return (task, method, condition, seed, man.get("status"), man.get("success"), time.time() - t0)


def done(path: Path) -> bool:
    m = path / "manifest.json"
    if not m.exists():
        return False
    try:
        return json.loads(m.read_text()).get("status") == "completed"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--conditions", default=",".join(CFG.collect.conditions))
    ap.add_argument("--method", default=CFG.method)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rho-scale", type=float, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--params", default=None, help="JSON with skill_rho / skill_delta / failure_budget")
    ap.add_argument("--label", default=None, help="directory name instead of the method name")
    ap.add_argument("--spec", default=None, help="LTLf specification (composed *_ltl tasks)")
    ap.add_argument("--workers", type=int, default=CFG.collect.workers)
    a = ap.parse_args()
    jobs = []
    for task in a.tasks.split(","):
        for cond in a.conditions.split(","):
            for seed in parse_seeds(a.seeds):
                if not done(episode_dir(a.tag, cond, task, a.label or a.method, seed)):
                    jobs.append((a.tag, cond, task, a.method, seed, a.rho_scale, a.budget, a.label, a.params, a.spec))
    print(f"{len(jobs)} episodes to run -> {EPISODES / a.tag}", flush=True)
    if not jobs:
        return
    t0 = time.time()
    n_ok = 0
    # multiprocessing.Pool recycles workers reliably (ProcessPoolExecutor's max_tasks_per_child
    # can deadlock under 'spawn'); recycling keeps memory flat over thousands of episodes.
    # EDDIES_MAXTASKS=0 disables recycling: on some machines the pool's semaphore disappears
    # from /dev/shm and every re-spawned worker dies on SemLock._rebuild (the pool then hangs).
    mt = int(os.environ.get("EDDIES_MAXTASKS", str(CFG.collect.max_tasks_per_child))) or None
    with get_context("spawn").Pool(processes=a.workers, maxtasksperchild=mt) as pool:
        for i, res in enumerate(pool.imap_unordered(_safe_job, jobs), 1):
            if isinstance(res, str):
                print("job failed:", res, flush=True)
                continue
            task, method, cond, seed, status, succ, dt = res
            n_ok += bool(succ)
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.time() - t0:7.0f}s  last {task}/{cond}/{seed} "
                      f"{status} success={succ} ({dt:.0f}s)  successes so far {n_ok}", flush=True)


def _safe_job(args):
    try:
        return _job(args)
    except Exception as ex:          # recorded, never silently dropped
        return repr(ex)


if __name__ == "__main__":
    main()
