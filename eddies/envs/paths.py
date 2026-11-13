"""Filesystem locations of the simulation package (see eddies.common.paths for the layout)."""
from __future__ import annotations

from pathlib import Path

from eddies.common.paths import MENAGERIE, MENAGERIE_COMMIT, menagerie  # noqa: F401

PKG = Path(__file__).resolve().parent
REPO = PKG.parents[1]
