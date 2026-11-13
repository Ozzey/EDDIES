"""Contact measurement from the physics engine (not from pose overlap)."""
from __future__ import annotations

from typing import Iterable

import mujoco
import numpy as np


class ContactMonitor:
    """Sum of normal contact forces between registered geom groups.

    ``groups`` maps a group name to geom ids; ``pairs`` maps a pair name to two
    group names.  The special group name ``"*"`` matches every geom.  Forces
    come from ``mj_contactForce`` (constraint impulses converted to forces by
    the solver), so they are the physical contact forces of this step.
    """

    def __init__(self, model: mujoco.MjModel, groups: dict[str, Iterable[int]],
                 pairs: dict[str, tuple[str, str]]):
        self.model = model
        self.masks = {}
        for name, ids in groups.items():
            m = np.zeros(model.ngeom, bool)
            m[list(ids)] = True
            self.masks[name] = m
        self.masks["*"] = np.ones(model.ngeom, bool)
        self.pairs = dict(pairs)
        for a, b in self.pairs.values():
            if a not in self.masks or b not in self.masks:
                raise KeyError(f"unknown contact group in pair ({a}, {b})")
        self._f = np.zeros(6)

    def measure(self, data: mujoco.MjData) -> dict:
        out = {name: 0.0 for name in self.pairs}
        counts = {name: 0 for name in self.pairs}
        n = data.ncon
        if n == 0:
            out.update({name + "#n": 0 for name in self.pairs})
            return out
        g1 = data.contact.geom1[:n]
        g2 = data.contact.geom2[:n]
        for name, (a, b) in self.pairs.items():
            ma, mb = self.masks[a], self.masks[b]
            sel = (ma[g1] & mb[g2]) | (mb[g1] & ma[g2])
            idx = np.flatnonzero(sel)
            total = 0.0
            for c in idx:
                mujoco.mj_contactForce(self.model, data, int(c), self._f)
                total += max(0.0, float(self._f[0]))
            out[name] = total
            counts[name] = int(len(idx))
        out.update({name + "#n": counts[name] for name in self.pairs})
        return out


def geom_contact_force(model, data, geoms_a: np.ndarray, geoms_b: np.ndarray) -> float:
    """One-off total normal force between two boolean geom masks."""
    n = data.ncon
    if n == 0:
        return 0.0
    g1, g2 = data.contact.geom1[:n], data.contact.geom2[:n]
    sel = np.flatnonzero((geoms_a[g1] & geoms_b[g2]) | (geoms_b[g1] & geoms_a[g2]))
    f = np.zeros(6)
    total = 0.0
    for c in sel:
        mujoco.mj_contactForce(model, data, int(c), f)
        total += max(0.0, float(f[0]))
    return total


def mask(model, ids: Iterable[int]) -> np.ndarray:
    m = np.zeros(model.ngeom, bool)
    m[list(ids)] = True
    return m
