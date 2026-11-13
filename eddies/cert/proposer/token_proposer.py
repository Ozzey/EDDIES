#!/usr/bin/env python3
"""Learned Eddie proposer: a small MLP that predicts the NEXT Eddie token at every hand-over.

Behaviour cloning on the symbolic supervisor's choices in logged episodes.  A hand-over is every control step at
which the executor asks for a new token (episode start, an Eddie finished / failed / timed out, a re-plan after a
disturbance); the label is the token the supervisor then chose, or "<done>" when it declared the task complete.

Features (all available at run time and logged in trajectory.npz, so training and deployment see the same vector):
    z_k (the task's measured embedding: tcp pose, aperture, force, ...), every task log signal (object / tcp
    positions, aperture, ...; for the stack task: red cube, blue cube, tcp, aperture), pairwise differences of the
    3-vector signals, the LTLf proposition bits, and ("full" model only) the previous Eddie (one-hot, or <start>)
    and its outcome class (<start> / ok / timeout / failure / none).
The "state" model drops the previous Eddie and outcome (it must read the hand-over from the measured state).
The symbolic cell / product state is NOT a feature.

    python -m eddies.cert.proposer.token_proposer --task franka_cube_stack \
        --train sb:nominal:1000-1199,tokprop:perturbed:1000-1199 \
        --val sb:nominal:1200-1399,tokprop:perturbed:1200-1399 \
        --test tokprop:nominal:2000-2099,tokprop:perturbed:2000-2099
Writes $EDDIES_MODELS/<task>_<features>.npz (model) and <task>_<features>_train.json (accuracies).
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

DONE = "<done>"
START = "<start>"
OUTCOMES = (START, "ok", "timeout", "failure", "none")
from eddies.common.config import load

CFG = load("experiments/token_proposer")
MAX_DECISIONS_PER_EPISODE = int(CFG.max_decisions)   # a few degenerate logged episodes re-plan thousands of times


def model_dir() -> Path:
    from eddies.common.paths import MODELS
    return MODELS


def outcome_class(outcome: str | None, handed_over: bool) -> str:
    if outcome is None:
        return START
    if not handed_over:
        return "none"
    if outcome in ("ok", "timeout"):
        return outcome
    return "failure"


# ------------------------------------------------------------------------------------------------ features
class FeatureSpec:
    def __init__(self, vocab, props, sig_dims, z_dim, mode="full"):
        self.vocab = list(vocab)                     # Eddies (sorted) + DONE
        self.props = list(props)                     # LTLf propositions in the runner's bit order
        self.sig_dims = dict(sig_dims)               # signal -> dimension (insertion order = feature order)
        self.z_dim = int(z_dim)
        self.mode = mode
        self.eddies = [v for v in self.vocab if v != DONE]
        self.vec3 = [k for k, d in self.sig_dims.items() if d == 3]

    def to_json(self):
        return {"vocab": self.vocab, "props": self.props, "sig_dims": self.sig_dims, "z_dim": self.z_dim,
                "mode": self.mode}

    @classmethod
    def from_json(cls, d):
        return cls(d["vocab"], d["props"], d["sig_dims"], d["z_dim"], d["mode"])

    def __call__(self, z, sig: dict, label_bits: int, prev: str | None, prev_out: str) -> np.ndarray:
        parts = [np.asarray(z, float).ravel()[: self.z_dim]]
        for k in self.sig_dims:
            parts.append(np.asarray(sig[k], float).ravel())
        for a, b in combinations(self.vec3, 2):
            parts.append(np.asarray(sig[a], float).ravel() - np.asarray(sig[b], float).ravel())
        parts.append(np.array([(int(label_bits) >> i) & 1 for i in range(len(self.props))], float))
        if self.mode == "full":
            p = np.zeros(len(self.eddies) + 1)
            p[self.eddies.index(prev) if prev in self.eddies else len(self.eddies)] = 1.0
            o = np.zeros(len(OUTCOMES))
            o[OUTCOMES.index(prev_out)] = 1.0
            parts += [p, o]
        x = np.concatenate(parts)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def spec_for_task(task_name: str, mode: str) -> FeatureSpec:
    """Vocabulary / propositions / signal layout from a task instance (no physics stepping)."""
    import warnings
    warnings.filterwarnings("ignore")
    from eddies.core.ltlf import Monitor, atoms
    from eddies.envs.tasks import make_task
    t = make_task(task_name, 2000)
    t.reset()
    props = sorted(atoms(Monitor(t.formula).initial))
    sig = {k: int(np.asarray(v, float).size) for k, v in t.log_signals().items()}
    obs = t.observe()
    z_dim = int(np.asarray(t.achieved(obs), float).size)
    return FeatureSpec(sorted(t.skills()) + [DONE], props, sig, z_dim, mode)


def runtime_features(spec: FeatureSpec, task, obs, history) -> np.ndarray:
    """The feature vector at a hand-over during a rollout (history = SkillTracker.history)."""
    bits = sum(1 << i for i, p in enumerate(spec.props) if p in obs["labels"])
    prev, out = None, None
    t = round(float(obs["t"]), 4)
    if history:
        prev, out = history[-1]["skill"], history[-1]["outcome"]
        handed = abs(history[-1]["end"] - t) < 1e-6
    else:
        handed = False
    return spec(task.achieved(obs), task.log_signals(), bits, prev, outcome_class(out, handed))


# ------------------------------------------------------------------------------------- logged hand-overs
def episode_handovers(d: Path, spec: FeatureSpec):
    """(X, y, info) of every hand-over of one logged episode; y indexes spec.vocab."""
    man = json.loads((d / "manifest.json").read_text())
    if man.get("status") != "completed" or not (d / "trajectory.npz").exists():
        return None
    npz = np.load(d / "trajectory.npz", allow_pickle=True)
    if "z" not in npz:
        return None
    status = [str(s) for s in npz["skill"]]
    t = np.round(npz["t"], 4)
    hist = man.get("method_info", {}).get("skills", [])
    starts = {round(h["start"], 4) for h in hist}
    sig = {k: npz["sig_" + k] for k in spec.sig_dims}
    X, y, rows = [], [], []
    hi = 0                                           # history entries that ended at or before t_k
    for k in range(len(status)):
        new = k == 0 or status[k] != status[k - 1] or (t[k] in starts and k > 0)
        if not new:
            continue
        tok = status[k].split()[0] if status[k].split() else ""
        if status[k].startswith("terminal"):
            label = DONE
        elif tok in spec.eddies:
            label = tok
        else:                                        # STOP / idle rows are not proposals
            continue
        while hi < len(hist) and round(hist[hi]["end"], 4) <= t[k] + 1e-9:
            hi += 1
        if hi:
            h = hist[hi - 1]
            prev, out = h["skill"], outcome_class(h["outcome"], abs(round(h["end"], 4) - t[k]) < 1e-6)
        else:
            prev, out = None, START
        X.append(spec(npz["z"][k], {key: v[k] for key, v in sig.items()}, int(npz["labels"][k]), prev, out))
        y.append(spec.vocab.index(label))
        rows.append((int(k), prev, out, label))
        if len(y) >= MAX_DECISIONS_PER_EPISODE:
            break
    if not y:
        return None
    return np.array(X), np.array(y), {"seed": man["seed"], "rows": rows, "success": man.get("success")}


def parse_sets(text):
    from eddies.common.paths import parse_seeds
    out = []
    for part in filter(None, text.split(",")):
        tag, cond, seeds = part.split(":")
        method = "bistl"
        if "@" in tag:
            tag, method = tag.split("@")
        out.append((tag, cond, method, parse_seeds(seeds.replace("..", "-"))))
    return out


def load_handovers(task, sets, spec):
    from eddies.common.paths import episode_dir
    X, y, meta = [], [], []
    for tag, cond, method, seeds in sets:
        for s in seeds:
            r = episode_handovers(episode_dir(tag, cond, task, method, s), spec)
            if r is None:
                continue
            X.append(r[0])
            y.append(r[1])
            meta += [(tag, cond, s, *row) for row in r[2]["rows"]]
    if not X:
        return np.zeros((0, 0)), np.zeros(0, int), []
    return np.vstack(X), np.concatenate(y), meta


# ----------------------------------------------------------------------------------------------- the MLP
class MLP:
    """tanh MLP classifier, Adam, weight decay, early stopping on validation loss (numpy only)."""

    def __init__(self, n_in, n_out, hidden=None, seed=0):
        hidden = tuple(CFG.mlp.hidden) if hidden is None else tuple(hidden)
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def _fwd(self, X):
        H = [(X - self.mu) / self.sd]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            H.append(np.tanh(H[-1] @ W + b))
        return H, H[-1] @ self.W[-1] + self.b[-1]

    def proba(self, X):
        _, L = self._fwd(np.atleast_2d(X))
        L = L - L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)

    @staticmethod
    def _loss(P, y, w):
        return float(-(w * np.log(P[np.arange(len(y)), y] + 1e-12)).sum() / w.sum())

    def fit(self, X, y, Xv, yv, epochs=None, lr=None, wd=None, batch=None, seed=0, balance=None):
        M = CFG.mlp
        epochs, lr, wd = (M.epochs if epochs is None else epochs), (M.lr if lr is None else lr), (M.wd if wd is None else wd)
        batch, balance = (M.batch if batch is None else batch), (M.balance if balance is None else balance)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        rng = np.random.default_rng(seed)
        K = self.W[-1].shape[1]
        cnt = np.bincount(y, minlength=K).astype(float)
        cw = np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0.0) if balance else np.ones(K)
        cw = cw / (cw[y].mean() + 1e-12)
        w_all = cw[y]
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step, best, best_state, hist = 0, np.inf, None, []
        for ep in range(epochs):
            idx = rng.permutation(len(y))
            for i in range(0, len(y), batch):
                j = idx[i:i + batch]
                H, L = self._fwd(X[j])
                L = L - L.max(1, keepdims=True)
                P = np.exp(L)
                P /= P.sum(1, keepdims=True)
                G = P.copy()
                G[np.arange(len(j)), y[j]] -= 1
                G *= (w_all[j] / w_all[j].sum())[:, None]
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                for l in range(len(self.W) - 1, -1, -1):
                    gW[l] = H[l].T @ G + wd * self.W[l]
                    gb[l] = G.sum(0)
                    if l:
                        G = (G @ self.W[l].T) * (1 - H[l] ** 2)
                step += 1
                for n, (p, g) in enumerate(zip(params, gW + gb)):
                    m[n] = 0.9 * m[n] + 0.1 * g
                    v[n] = 0.999 * v[n] + 0.001 * g * g
                    p -= lr * (m[n] / (1 - 0.9 ** step)) / (np.sqrt(v[n] / (1 - 0.999 ** step)) + 1e-8)
            vl = self._loss(self.proba(Xv), yv, np.ones(len(yv))) if len(yv) else 0.0
            hist.append(vl)
            if vl < best - 1e-6:
                best, best_state, stale = vl, ([W.copy() for W in self.W], [b.copy() for b in self.b], ep), 0
            else:
                stale += 1
                if stale >= 80:
                    break
        if best_state is not None:
            self.W, self.b = best_state[0], best_state[1]
            params[:] = self.W + self.b
        return {"epochs_run": ep + 1, "best_epoch": best_state[2] if best_state else ep, "best_val_loss": best}

    def save(self, path, spec: FeatureSpec, extra=None):
        arrs = {f"W{i}": W for i, W in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}
        np.savez(path, mu=self.mu, sd=self.sd, n_layers=len(self.W),
                 spec=json.dumps(spec.to_json()), extra=json.dumps(extra or {}), **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        n = int(z["n_layers"])
        m = cls.__new__(cls)
        m.W = [z[f"W{i}"] for i in range(n)]
        m.b = [z[f"b{i}"] for i in range(n)]
        m.mu, m.sd = z["mu"], z["sd"]
        return m, FeatureSpec.from_json(json.loads(str(z["spec"])))


def accuracy_report(model, X, y, meta, vocab):
    if not len(y):
        return {"n": 0}
    P = model.proba(X)
    pred = P.argmax(1)
    ok = pred == y
    out = {"n": int(len(y)), "acc": float(ok.mean()), "errors": int((~ok).sum())}
    # hand-overs that follow a non-ok outcome (recoveries) and the per-label accuracy
    rec = np.array([m[5] in ("failure", "timeout", "none") for m in meta])
    out["recovery_n"] = int(rec.sum())
    out["recovery_acc"] = float(ok[rec].mean()) if rec.any() else None
    out["per_label"] = {vocab[c]: [int(ok[y == c].sum()), int((y == c).sum())] for c in np.unique(y)}
    out["confusions"] = {}
    for i in np.where(~ok)[0]:
        key = f"{vocab[y[i]]} -> {vocab[pred[i]]}"
        out["confusions"][key] = out["confusions"].get(key, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--train", required=True, help="tag[@method]:condition:seeds,...")
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--features", default="full,state")
    ap.add_argument("--hidden", default=",".join(str(int(h)) for h in CFG.mlp.hidden))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_dir = model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in a.features.split(","):
        spec = spec_for_task(a.task, mode)
        Xtr, ytr, mtr = load_handovers(a.task, parse_sets(a.train), spec)
        Xv, yv, mv = load_handovers(a.task, parse_sets(a.val), spec)
        Xte, yte, mte = load_handovers(a.task, parse_sets(a.test), spec) if a.test else (np.zeros((0, 0)), np.zeros(0, int), [])
        print(f"{a.task} [{mode}] features {Xtr.shape[1]}  hand-overs train {len(ytr)} val {len(yv)} test {len(yte)}",
              flush=True)
        model = MLP(Xtr.shape[1], len(spec.vocab), tuple(int(h) for h in a.hidden.split(",")), seed=a.seed)
        fit = model.fit(Xtr, ytr, Xv, yv, seed=a.seed)
        rep = {"task": a.task, "features": mode, "n_features": int(Xtr.shape[1]), "vocab": spec.vocab,
               "hidden": a.hidden, "fit": fit, "train_sets": a.train, "val_sets": a.val, "test_sets": a.test,
               "episodes": {"train": len({(m[0], m[1], m[2]) for m in mtr}), "val": len({(m[0], m[1], m[2]) for m in mv}),
                            "test": len({(m[0], m[1], m[2]) for m in mte})},
               "label_counts_train": {spec.vocab[c]: int(n) for c, n in enumerate(np.bincount(ytr, minlength=len(spec.vocab)))},
               "train": accuracy_report(model, Xtr, ytr, mtr, spec.vocab),
               "val": accuracy_report(model, Xv, yv, mv, spec.vocab),
               "test": accuracy_report(model, Xte, yte, mte, spec.vocab) if len(yte) else {"n": 0}}
        # split the held-out numbers by condition
        for name, (X_, y_, m_) in (("val", (Xv, yv, mv)), ("test", (Xte, yte, mte))):
            for cond in sorted({m[1] for m in m_}):
                sel = np.array([m[1] == cond for m in m_])
                rep[f"{name}_{cond}"] = accuracy_report(model, X_[sel], y_[sel], [m for m, s in zip(m_, sel) if s], spec.vocab)
        model.save(out_dir / f"{a.task}_{mode}.npz", spec, {"fit": fit})
        (out_dir / f"{a.task}_{mode}_train.json").write_text(json.dumps(rep, indent=1, default=float))
        print(json.dumps({k: (v if not isinstance(v, dict) or k in ("fit", "episodes") else
                              {kk: vv for kk, vv in v.items() if kk in ("n", "acc", "errors", "recovery_n", "recovery_acc", "confusions")})
                          for k, v in rep.items() if k not in ("vocab", "label_counts_train")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
