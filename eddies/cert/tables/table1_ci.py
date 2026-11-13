#!/usr/bin/env python3
"""95% bootstrap intervals for Table 1 (table1_eddies / calibration_ablation with --episodes).

    python -m eddies.cert.tables.table1_ci [--dir table1] [--B 2000] [--seed 0]

Two resampling schemes, both stratified by task (each task's 100 test and 200 calibration
episodes are resampled with replacement on their own):
  test        the test episodes only; models, normalisers and the per-task quantile q fixed
  cal+test    also the calibration episodes: q is recomputed on each resample, and the gate's
              completion follows from the per-episode thresholds (table1_eddies.episode_records)
The same resampled episodes are used for every row (both rows certify the same logged episodes), so
the difference between the two Eddie rows gets paired intervals.  For the per-step bound the
resamples count how often every plan Eddie of a task would be certified at all (b1_bootstrap).
Columns: tube p90 (step-weighted 90th percentile of the ultimate tube over the executed steps,
largest over the four tasks) and runs finished at mu = 5 / 7.5 / 10 cm (of 400).  Light: reads the
JSONs only.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import mu_list

CFG = load_config("experiments/table1_ci")
ROWS = [(str(label), str(variant)) for label, variant in CFG.rows]
MUS = tuple(mu_list(CFG.mu_cm))


def cq(s, alpha):
    """conformal_quantile on a sorted finite array."""
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[k - 1]) if 1 <= k <= n else np.inf


def wp90(r, w):
    """radius_stats' step-weighted 90th percentile [cm]."""
    m = w > 0
    if not m.any():
        return np.nan
    r, w = r[m], w[m]
    o = np.argsort(r)
    c = np.cumsum(w[o]) / w.sum()
    return 100 * r[o][np.searchsorted(c, 0.9)]


class Task:
    def __init__(self, rec, mu_all, alpha):
        self.alpha = alpha
        self.mi = [mu_all.index(m) for m in MUS]
        test = rec["test"]
        self.n = len(test)
        self.q = rec["q"]
        self.cal = np.array(rec["cal_scores"], float)            # -inf: no certified token (dropped)
        self.seeds = [ep["seed"] for ep in test]
        self.tok = sorted(rec["coef"])
        a, b = np.array([rec["coef"][s] for s in self.tok]).T.reshape(2, -1)
        self.a, self.b = a, b
        self.steps = np.array([[ep["steps"].get(s, 0) for s in self.tok] for ep in test], float)
        self.comp = np.array([[ep["completed"][m] for m in self.mi] for ep in test], float)
        self.unc = np.array([ep["uncertified"] for ep in test])
        self.qres = np.array([ep["q_res"] for ep in test], float)
        self.qhi = np.array([[min(ep["q_ho"][m], rec["q_plan"][m]) for m in self.mi] for ep in test], float)
        self.recon_ok = all(ep["recon_ok"] for ep in test)

    def completed(self, q):
        ok = (~self.unc) & (self.qres <= q)
        return (ok[:, None] & (q <= self.qhi)).astype(float)

    def stats(self, c, q=None):
        """c: resample counts of the test episodes; q: None = calibrated q (fixed)."""
        comp = self.comp if q is None else self.completed(q)
        q = self.q if q is None else q
        return wp90(self.a + q * self.b, c @ self.steps), c @ comp


def load_variant(path, key=None):
    r = json.loads(path.read_text())
    if key is None:                                     # table1_eddies
        tasks = {t: x["episodes"] for t, x in r["indist"]["tasks"].items()}
        return tasks, r["mu_m"], r["alpha"], r["indist"]
    tasks = {t: x[key]["episodes"] for t, x in r["tasks"].items()}   # calibration_ablation variant
    return tasks, r["mu_m"], r["alpha"], r


def draws(tasks, B, seed):
    """Shared resample indices: {task: (test indices [B, n], calibration indices [B, n_cal])}."""
    rng = np.random.default_rng(seed)
    return {t: (rng.integers(x.n, size=(B, x.n)), rng.integers(len(x.cal), size=(B, len(x.cal))))
            for t, x in sorted(tasks.items())}


def bootstrap(tasks, D, joint):
    T = list(tasks)
    B = next(iter(D.values()))[0].shape[0]
    p90 = np.zeros((B, len(T)))
    fin = np.zeros((B, len(MUS)))
    for i in range(B):
        for j, t in enumerate(T):
            x = tasks[t]
            c = np.bincount(D[t][0][i], minlength=x.n).astype(float)
            q = None
            if joint:
                s = x.cal[D[t][1][i]]
                q = cq(np.sort(s[np.isfinite(s)]), x.alpha)
            p, f = x.stats(c, q)
            p90[i, j] = p
            fin[i] += f
    return p90, fin


def ci(v):
    v = v[np.isfinite(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def summarise(name, tasks, D):
    pt_p90 = [x.stats(np.ones(x.n))[0] for x in tasks.values()]
    pt_fin = sum(x.stats(np.ones(x.n))[1] for x in tasks.values())
    out = {"row": name, "p90_max": float(np.nanmax(pt_p90)), "p90_min": float(np.nanmin(pt_p90)),
           "finished": [int(v) for v in pt_fin], "recon_ok": all(x.recon_ok for x in tasks.values())}
    for joint in (False, True):
        p90, fin = bootstrap(tasks, D, joint)
        k = "cal+test" if joint else "test"
        mx = np.where(np.isnan(p90), -np.inf, p90).max(1)
        mn = np.where(np.isnan(p90), np.inf, p90).min(1)
        out[k] = {"p90_max": ci(np.where(np.isfinite(mx), mx, np.nan)), "p90_min": ci(np.where(np.isfinite(mn), mn, np.nan)),
                  "finished": [ci(fin[:, m]) for m in range(len(MUS))], "fin_samples": fin.tolist(), "p90_samples": mx.tolist()}
    return out


def b1_bootstrap(cal_json, B, seed):
    """Per-step bound: a plan is refused if any plan Eddie has no finite per-step quantile.  Count the
    resamples of the calibration episodes in which every plan Eddie of a task would be certified."""
    r = json.loads(cal_json.read_text())
    alpha = r["alpha"]
    rng = np.random.default_rng(seed)
    hit = {t: 0 for t in r["tasks"]}
    need = {}
    for t, x in r["tasks"].items():
        e = x["b1"]["episodes"]
        L = np.array(e["len"])
        plan = x["b1"]["eps_plan"]
        miss = [s for s in plan if s not in e["n"]]
        n = {s: np.array(e["n"][s]) for s in e["n"]}
        K = L.max()
        need[t] = {"K": int(K), "need_steps": int(math.ceil(K / alpha - 1)),
                   "plan_steps": {s: int(n[s].sum()) for s in n}, "not_fitted": miss}
        if miss:
            continue
        for _ in range(B):
            c = np.bincount(rng.integers(len(L), size=len(L)), minlength=len(L))
            Kb = L[c > 0].max()
            a = alpha / Kb
            if all(math.ceil((int(c @ n[s]) + 1) * (1 - a)) <= int(c @ n[s]) for s in n):
                hit[t] += 1
    return hit, need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=CFG.dirs[0])
    ap.add_argument("--B", type=int, default=CFG.B)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--json", default=None, help="write the summary here")
    a = ap.parse_args()
    d = ROOT / a.dir
    res, D, seeds = [], None, None
    for name, v in ROWS:
        f = d / f"{v}.json"
        if not f.exists():
            print(f"missing {f}")
            continue
        tasks, mus, alpha, ind = load_variant(f)
        tasks = {t: Task(x, mus, alpha) for t, x in sorted(tasks.items())}
        if D is None:
            D, seeds = draws(tasks, a.B, a.seed), {t: x.seeds for t, x in tasks.items()}
        assert {t: x.seeds for t, x in tasks.items()} == seeds and all(len(x.cal) == D[t][1].shape[1] for t, x in tasks.items())
        res.append(summarise(name, tasks, D))
        res[-1]["handovers"] = float(np.mean([x["tokens"]["handovers_per_episode"] for x in ind["tasks"].values()]))
        res[-1]["labels"], res[-1]["certified"] = ind["labels"], ind["certified"]
        res[-1]["shrink"] = ind.get("shrink")
    f = d / "calibration.json"
    if f.exists():
        hit, need = b1_bootstrap(f, a.B, a.seed)
        res.append({"row": "Eddies, per-step bound", "resamples_with_all_plan_eddies_certified": hit, "detail": need})
    for r in res:
        if "finished" not in r:
            print(r["row"], r["resamples_with_all_plan_eddies_certified"], r["detail"])
            continue
        s = f"{r['row']:30s} p90 {r['p90_max']:5.1f} [test {r['test']['p90_max'][0]:.1f}-{r['test']['p90_max'][1]:.1f};" \
            f" cal+test {r['cal+test']['p90_max'][0]:.1f}-{r['cal+test']['p90_max'][1]:.1f}]  fin "
        s += " / ".join(f"{v} [{lo:.0f}-{hi:.0f}; {lo2:.0f}-{hi2:.0f}]" for v, (lo, hi), (lo2, hi2)
                        in zip(r["finished"], r["test"]["finished"], r["cal+test"]["finished"]))
        print(s, "" if r["recon_ok"] else "RECONSTRUCTION MISMATCH")
    ref = res[0]
    print("paired differences, Eddies minus row (95% intervals, test / cal+test):")
    for r in res[1:]:
        if "finished" not in r:
            continue
        s = f"  {r['row']:30s} p90 "
        for k in ("test", "cal+test"):
            dp = np.array(ref[k]["p90_samples"]) - np.array(r[k]["p90_samples"])
            s += f"[{ci(dp)[0]:+.1f},{ci(dp)[1]:+.1f}] "
        s += " fin "
        for m in range(len(MUS)):
            s += " ".join(f"[{ci(df)[0]:+.0f},{ci(df)[1]:+.0f}]" for df in
                          (np.array(ref[k]["fin_samples"])[:, m] - np.array(r[k]["fin_samples"])[:, m] for k in ("test", "cal+test")))
            s += " | "
        print(s)
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
