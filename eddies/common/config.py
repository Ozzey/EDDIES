"""YAML configuration for Eddies (PyYAML only, no Hydra).

    from eddies.common.config import load, seeds, mu_list
    cfg = load("experiments/gate_fixed", "conformal.alpha=0.05", "mu_cm=[5,10]")
    cfg.conformal.alpha          # attribute access, nested
    cfg.get("paths.results")     # dotted get with a default

Resolution order (later wins):
  1. eddies/config.yaml                       global defaults (splits, conformal, identification, ...)
  2. the ``defaults:`` list of the named file   (e.g. ``defaults: [robots/franka]``), recursively
  3. eddies/configs/<name>.yaml                the named experiment / robot / task file
  4. dotted ``key=value`` overrides            values parsed with yaml.safe_load
  5. ``${ENV:-default}`` interpolation in strings

Every value the paper depends on lives in these files; the defaults reproduce the paper.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]          # eddies/
REPO = PKG.parent                                  # repository root
GLOBAL = PKG / "config.yaml"
CONFIGS = PKG / "configs"

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Config(dict):
    """dict with attribute access; nested dicts are Configs."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as ex:
            raise AttributeError(k) from ex

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, key, default=None):          # dotted get
        cur = self
        for part in str(key).split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, key, value):                 # dotted set
        cur = self
        parts = str(key).split(".")
        for part in parts[:-1]:
            if not isinstance(cur.get(part), dict):
                cur[part] = Config()
            cur = cur[part]
        cur[parts[-1]] = _wrap(value)

    def to_dict(self):
        return _plain(self)

    def dump(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _wrap(x):
    if isinstance(x, dict) and not isinstance(x, Config):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, Config):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _interp(x):
    if isinstance(x, dict):
        return {k: _interp(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_interp(v) for v in x]
    if isinstance(x, str):
        def sub(m):
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, "" if default is None else default)
        return _ENV.sub(sub, x)
    return x


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _named(name: str, seen=()) -> dict:
    """configs/<name>.yaml with its ``defaults:`` merged first (recursively)."""
    path = CONFIGS / f"{name}.yaml"
    raw = _read(path)
    out: dict = {}
    for dep in raw.pop("defaults", []) or []:
        if dep in seen:
            raise ValueError(f"circular defaults: {dep}")
        out = _merge(out, _named(dep, (*seen, name)))
    return _merge(out, raw)


def load(name: str | None = None, *overrides: str, env: bool = True) -> Config:
    """Global defaults, then configs/<name>.yaml (with its defaults), then dotted overrides."""
    cfg = _read(GLOBAL)
    if name:
        cfg = _merge(cfg, _named(name))
        cfg.setdefault("config_name", name)
    cfg = _wrap(cfg)
    for ov in overrides:
        if not ov or ov.startswith("-"):
            continue
        key, _, val = ov.partition("=")
        if not _:
            raise ValueError(f"override must be key=value: {ov!r}")
        cfg.set(key.strip(), yaml.safe_load(val))
    if env:
        cfg = _wrap(_interp(cfg))
    return cfg


def available(kind: str = "experiments") -> list[str]:
    return sorted(p.stem for p in (CONFIGS / kind).glob("*.yaml"))


# ------------------------------------------------------------------------- value helpers
def seeds(spec) -> list[int]:
    """[a, b] -> a..b inclusive; [[a,b],[c,d]] -> both; "1000-1199,2000" -> ints; range/list -> list."""
    if spec is None:
        return []
    if isinstance(spec, range):
        return list(spec)
    if isinstance(spec, str):
        out = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
        return out
    if isinstance(spec, dict):
        return seeds(spec.get("seeds"))
    spec = list(spec)
    if len(spec) == 2 and all(isinstance(x, int) for x in spec) and spec[0] <= spec[1]:
        return list(range(spec[0], spec[1] + 1))
    out = []
    for x in spec:
        out += seeds(x) if isinstance(x, (list, tuple, str)) else [int(x)]
    return out


def span(spec) -> str:
    """Inverse of seeds() for a contiguous [a, b]: "a-b" (the string form the CLIs take)."""
    s = seeds(spec)
    if not s:
        return ""
    return f"{s[0]}-{s[-1]}" if s == list(range(s[0], s[-1] + 1)) else ",".join(map(str, s))


def mu_list(cm) -> list[float]:
    """tolerances in centimetres -> metres."""
    return [float(x) / 100 for x in cm]


def mu_str(cm) -> str:
    return ",".join(f"{float(x):g}" for x in cm)


def parse_mu(text) -> list[float]:
    return [float(x) / 100 for x in str(text).split(",") if x]


def robots(cfg: Config | None = None) -> dict[str, Config]:
    """{name: robot config} in the order of ``robots`` of the global config."""
    names = (cfg or load()).get("robots", ["franka", "g1", "leap", "aero"])
    return {r: load(f"robots/{r}") for r in names}


def tasks(cfg: Config | None = None) -> dict[str, Config]:
    names = (cfg or load()).get("paper_tasks", [])
    return {t: load(f"tasks/{t}") for t in names}


def blocks(cfg: Config | None = None) -> dict[str, tuple[list[int], list[int]]]:
    """family -> (certified pose block, position rows) from configs/robots/*.yaml."""
    out = {}
    for r in robots(cfg).values():
        out[r.family] = (list(r.block.pose), list(r.block.position))
    return out


# ------------------------------------------------------------------------------- CLI
def cli(parser: argparse.ArgumentParser | None = None, argv=None, default: str | None = None):
    """Parse ``--config NAME`` and trailing ``key=value`` overrides next to the script's own flags.

    Returns (cfg, args); ``args.overrides`` holds the raw overrides."""
    parser = parser or argparse.ArgumentParser()
    parser.add_argument("--config", default=default, help="configs/<name>.yaml (e.g. experiments/gate_fixed)")
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = parser.parse_args(argv)
    cfg = load(args.config, *args.overrides)
    return cfg, args
