#!/usr/bin/env python3
"""Learned COMMAND proposer inside an Eddie: behaviour cloning of the per-step command increment.

At every control step of an active Eddie the MLP predicts d_k = e_k - e_{k-1} from measured features; the method
(eddies/envs/methods/learned_tokens.py, bistl_learned_cmd / bistl_cmd) proposes e_{k-1} + d_hat, and the Streamer
projection (box ∩ trust ball of the Eddie ∩ slew ball, StreamerMethod._project, unchanged) commits the command.
The Eddie's own target is still evaluated (its termination / outcome logic is untouched) but not used.

Features: active Eddie (one-hot), z_k, e_{k-1}, e_{k-1} - z_k, every log signal, every 3-vector signal mapped into
the command chart's position block and taken relative to e_{k-1}[:3], LTLf proposition bits, steps since the Eddie
started (/50).  Targets standardised per dimension.

    python -m eddies.cert.proposer.command_proposer --task franka_cube_stack \
        --train sb:nominal:1000-1199,tokprop:perturbed:1000-1199 --val sb:nominal:1200-1399
Writes $EDDIES_RESULTS/tokprop/<task>_cmd.npz and <task>_cmd_train.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.cert.proposer.token_proposer import MLP, model_dir, parse_sets, spec_for_task

from eddies.common.config import load

CFG = load("experiments/token_proposer")
MAX_STEPS_PER_EPISODE = int(CFG.cmd_mlp.max_steps)


class CmdSpec:
    def __init__(self, eddies, props, sig_dims, z_dim, e_dim, phys_lower, scale, norm_lower):
        self.eddies, self.props, self.sig_dims = list(eddies), list(props), dict(sig_dims)
        self.z_dim, self.e_dim = int(z_dim), int(e_dim)
        self.pl, self.sc, self.nl = (np.asarray(x, float)[:3] for x in (phys_lower, scale, norm_lower))
        self.vec3 = [k for k, d in self.sig_dims.items() if d == 3]

    def to_json(self):
        return {"eddies": self.eddies, "props": self.props, "sig_dims": self.sig_dims, "z_dim": self.z_dim,
                "e_dim": self.e_dim, "phys_lower": self.pl.tolist(), "scale": self.sc.tolist(),
                "norm_lower": self.nl.tolist()}

    @classmethod
    def from_json(cls, d):
        return cls(d["eddies"], d["props"], d["sig_dims"], d["z_dim"], d["e_dim"], d["phys_lower"], d["scale"],
                   d["norm_lower"])

    def __call__(self, eddie, z, e_prev, sig, label_bits, steps_in):
        z = np.asarray(z, float).ravel()[: self.z_dim]
        e_prev = np.asarray(e_prev, float).ravel()
        oh = np.zeros(len(self.eddies))
        oh[self.eddies.index(eddie)] = 1.0
        parts = [oh, z, e_prev, e_prev[: self.z_dim] - z[: len(e_prev)]]
        for k in self.sig_dims:
            parts.append(np.asarray(sig[k], float).ravel())
        for k in self.vec3:
            pn = self.nl + (np.asarray(sig[k], float).ravel() - self.pl) / self.sc
            parts.append(pn - e_prev[:3])
        parts.append(np.array([(int(label_bits) >> i) & 1 for i in range(len(self.props))], float))
        parts.append([min(steps_in, 500) / 50.0])
        return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)


def cmd_spec_for_task(task_name):
    from eddies.envs.tasks import make_task
    t = make_task(task_name, 2000)
    s = spec_for_task(task_name, "state")
    sp = t.space
    return CmdSpec(s.eddies, s.props, s.sig_dims, s.z_dim, len(t.initial_command()), sp.phys_lower, sp._scale,
                   sp.norm_lower)


def episode_steps(d, spec):
    from eddies.cert.data import skill_of
    m = json.loads((d / "manifest.json").read_text())
    if m.get("status") != "completed" or not (d / "trajectory.npz").exists():
        return None
    z = np.load(d / "trajectory.npz", allow_pickle=True)
    if "z" not in z:
        return None
    sk = [skill_of(s) for s in z["skill"]]
    status = [str(s) for s in z["skill"]]
    sig = {k: z["sig_" + k] for k in spec.sig_dims}
    X, Y = [], []
    k0 = 0
    for k in range(1, len(sk)):
        if status[k] != status[k - 1]:
            k0 = k
        if sk[k] is None or sk[k] not in spec.eddies:
            continue
        X.append(spec(sk[k], z["z"][k], z["e"][k - 1], {a: v[k] for a, v in sig.items()}, int(z["labels"][k]), k - k0))
        Y.append(z["e"][k] - z["e"][k - 1])
        if len(X) >= MAX_STEPS_PER_EPISODE:
            break
    return (np.array(X), np.array(Y)) if X else None


def load_steps(task, sets, spec):
    from eddies.common.paths import episode_dir
    X, Y = [], []
    for tag, cond, method, seeds in sets:
        for s in seeds:
            r = episode_steps(episode_dir(tag, cond, task, method, s), spec)
            if r is not None:
                X.append(r[0])
                Y.append(r[1])
    return np.vstack(X), np.vstack(Y)


class MLPReg(MLP):
    """Same network, linear output, MSE on standardised targets."""

    def predict(self, X):
        _, L = self._fwd(np.atleast_2d(X))
        return L * self.ysd + self.ymu

    def fit_reg(self, X, Y, Xv, Yv, epochs=None, lr=None, wd=None, batch=None, seed=0):
        M = CFG.cmd_mlp
        epochs, lr, wd = (M.epochs if epochs is None else epochs), (M.lr if lr is None else lr), (M.wd if wd is None else wd)
        batch = M.batch if batch is None else batch
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.ymu, self.ysd = Y.mean(0), Y.std(0) + 1e-6
        Yn, Yvn = (Y - self.ymu) / self.ysd, (Yv - self.ymu) / self.ysd
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step, best, best_state, stale = 0, np.inf, None, 0
        for ep in range(epochs):
            idx = rng.permutation(len(Y))
            for i in range(0, len(Y), batch):
                j = idx[i:i + batch]
                H, L = self._fwd(X[j])
                G = (L - Yn[j]) * (2.0 / L.size)
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
            _, Lv = self._fwd(Xv)
            vl = float(((Lv - Yvn) ** 2).mean())
            if vl < best - 1e-6:
                best, best_state, stale = vl, ([W.copy() for W in self.W], [b.copy() for b in self.b], ep), 0
            else:
                stale += 1
                if stale >= 20:
                    break
        self.W, self.b = best_state[0], best_state[1]
        return {"epochs_run": ep + 1, "best_epoch": best_state[2], "best_val_mse_std": best}

    def save(self, path, spec, extra=None):
        arrs = {f"W{i}": W for i, W in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}
        np.savez(path, mu=self.mu, sd=self.sd, ymu=self.ymu, ysd=self.ysd, n_layers=len(self.W),
                 spec=json.dumps(spec.to_json()), extra=json.dumps(extra or {}), **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        n = int(z["n_layers"])
        m = cls.__new__(cls)
        m.W = [z[f"W{i}"] for i in range(n)]
        m.b = [z[f"b{i}"] for i in range(n)]
        m.mu, m.sd, m.ymu, m.ysd = z["mu"], z["sd"], z["ymu"], z["ysd"]
        return m, CmdSpec.from_json(json.loads(str(z["spec"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--hidden", default=",".join(str(int(h)) for h in CFG.cmd_mlp.hidden))
    a = ap.parse_args()
    spec = cmd_spec_for_task(a.task)
    X, Y = load_steps(a.task, parse_sets(a.train), spec)
    Xv, Yv = load_steps(a.task, parse_sets(a.val), spec)
    print(f"{a.task} command BC: features {X.shape[1]}, steps train {len(Y)} val {len(Yv)}", flush=True)
    model = MLPReg(X.shape[1], Y.shape[1], tuple(int(h) for h in a.hidden.split(",")))
    fit = model.fit_reg(X, Y, Xv, Yv)
    P = model.predict(Xv)
    var = Yv.var(0)
    r2 = [float(1 - ((P[:, i] - Yv[:, i]) ** 2).mean() / var[i]) if var[i] > 1e-12 else None for i in range(Y.shape[1])]
    rep = {"task": a.task, "fit": fit, "n_features": int(X.shape[1]), "steps": {"train": len(Y), "val": len(Yv)},
           "val_r2_per_dim": r2, "val_rmse_per_dim": np.sqrt(((P - Yv) ** 2).mean(0)).tolist(),
           "train_sets": a.train, "val_sets": a.val, "hidden": a.hidden}
    out = model_dir()
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / f"{a.task}_cmd.npz", spec, {"fit": fit})
    (out / f"{a.task}_cmd_train.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
