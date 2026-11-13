"""Certify-or-stop gate for Eddie sequences.

Three checks, all from quantities available before or during execution:

1. Plan check (before the task starts).  Every Eddie i of the plan must be certified and its
   ultimate certified tube at the commanded slew bound must fit its tolerance:
       U_i = rho_i * sum_j ||S A_i^j G_i W_e^-1/2|| + eps_i * sum_j ||S A_i^j W_z^-1/2||  <=  mu_i.
2. Hand-over check (at the start of every Eddie).  From the MEASURED error eta_0 = z - z*_i(e):
       B_i(eta_0) = max_j ||S A_i^j eta_0|| + rho_i cumG_i(j) + eps_i cumW_i(j)  <=  mu_i,
   else stop before the Eddie runs.
3. Residual monitor (every control step).  ||W_z^1/2 (z_{k+1} - A_i z_k - B_i e_k - d_i)|| <= eps_i,
   else stop: the Eddie has left the regime its certificate was calibrated on (a new task, a
   heavier object, a speed beyond the identified range).

Consequence: on every execution that COMPLETES (no stop), all residuals were within eps_i and
the commanded slew within rho_i, so the propagated tube holds deterministically and, by check 2,
the tracking error stays within mu_i at every step.  The conformal level alpha bounds instead how
often an in-distribution execution is stopped by check 3 (episode-level calibration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eddies.cert.tube import ordered_segments


def equilibrium(t, e):
    return np.linalg.solve(np.eye(len(t.J)) - t.A, t.B @ e + t.d)


def ultimate_tube(t, rho, eps):
    return t.slew_radius(rho, eps)


def handover_bound(t, rho, eps, eta0):
    """max over the certificate horizon of the propagated position bound from eta0 [m]."""
    tr = np.linalg.norm(np.einsum("jab,b->ja", t.powers_S, eta0), axis=1)
    cum_g = np.concatenate([[0], np.cumsum(t.cg)])[: len(tr)]
    cum_w = np.concatenate([[0], np.cumsum(t.cw)])[: len(tr)]
    return float(np.max(tr + rho * cum_g + eps * cum_w))


@dataclass
class Decision:
    accepted: bool                 # passed the plan check
    completed: bool                # accepted and never stopped
    cause: str                     # 'completed' | 'plan' | 'uncertified' | 'handover' | 'residual'
    stop_step: int | None = None
    stop_eddie: str | None = None
    max_err_m: float = 0.0         # largest tracking error over the executed (certified) part
    exceeded_executed: bool = False  # tolerance exceeded before the stop (must be False when completed)


class RuntimeGate:
    """Online form (for the robot loop): call plan(), then enter() at every hand-over and step() each tick."""

    def __init__(self, tubes, eps, mu, rho):
        self.tubes, self.eps, self.mu, self.rho = tubes, eps, mu, rho
        self.cur = None

    def plan(self, labels):
        for s in labels:
            if s not in self.tubes:
                return False, f"uncertified:{s}"
            if ultimate_tube(self.tubes[s], self.rho(s), self.eps[s]) > self.mu(s):
                return False, f"tolerance:{s}"
        return True, "ok"

    def enter(self, label, z, e):
        t = self.tubes.get(label)
        if t is None:
            return False, "uncertified"
        self.cur = label
        eta0 = z[t.J] - equilibrium(t, e)
        ok = handover_bound(t, self.rho(label), self.eps[label], eta0) <= self.mu(label)
        return ok, "ok" if ok else "handover"

    def step(self, z_prev, e_prev, z):
        t = self.tubes[self.cur]
        w = z[t.J] - (t.A @ z_prev[t.J] + t.B @ e_prev + t.d)
        ok = float(np.linalg.norm(t.Wz * w)) <= self.eps[self.cur] * (1 + 1e-9)
        return ok, "ok" if ok else "residual"


def evaluate_episode(ep, plan, tubes, eps, mu, rho) -> tuple[Decision, dict]:
    """Replay the gate on a logged episode (decisions use only measured / commanded data up to
    that step).  Also returns counterfactual ground truth over the WHOLE episode."""
    z, e = ep["z"], ep["e"]
    gate = RuntimeGate(tubes, eps, lambda s: mu, lambda s: rho)
    truth = {"exceeds_mu": False, "max_err_m": 0.0}
    for s, k0, k1 in ordered_segments(ep["skill"]):          # counterfactual: full execution
        t = tubes.get(s) if s else None
        if t is None:
            continue
        Zs = np.linalg.solve(np.eye(len(t.J)) - t.A, (e[k0:k1 + 1] @ t.B.T + t.d).T).T
        err = np.linalg.norm((z[k0:k1 + 1][:, t.J] - Zs) @ t.S.T, axis=1)
        truth["max_err_m"] = max(truth["max_err_m"], float(err.max()))
    truth["exceeds_mu"] = truth["max_err_m"] > mu

    ok, why = gate.plan(plan)
    if not ok:
        return Decision(False, False, "plan" if why.startswith("tolerance") else "uncertified",
                        0, why.split(":", 1)[1]), truth
    max_err = 0.0
    for s, k0, k1 in ordered_segments(ep["skill"]):
        if s is None:
            continue
        ok, why = gate.enter(s, z[k0], e[k0])
        if not ok:
            return Decision(True, False, why, k0, s, max_err, max_err > mu), truth
        t = tubes[s]
        for k in range(k0, k1):
            zs = equilibrium(t, e[k])
            max_err = max(max_err, float(np.linalg.norm(t.S @ (z[k, t.J] - zs))))
            ok, why = gate.step(z[k], e[k], z[k + 1])
            if not ok:
                return Decision(True, False, why, k + 1, s, max_err, max_err > mu), truth
        max_err = max(max_err, float(np.linalg.norm(t.S @ (z[k1, t.J] - equilibrium(t, e[k1])))))
    return Decision(True, True, "completed", None, None, max_err, max_err > mu), truth
