"""Eddies: dynamical action tokens with certified composition.

Packages
--------
core    the mathematical core (LTLf monitor, contracts, embedding projection, recursive least squares)
cert    Eddies: identification, propagated tubes, conformal calibration, the fixed-calibration and
        self-calibrating gates, chaining / containment, the Table 1 ablations, learned proposer, tables, plots
envs    MuJoCo tasks of the paper, per-embodiment skillbooks, the Streamer and the episode runner
isaac   optional Isaac Sim (PhysX) port of the same tasks
common  YAML configuration (eddies/config.yaml, eddies/configs/) and the repository layout

Entry points: ``python -m eddies.collect``, ``eddies.certify``, ``eddies.gate``, ``eddies.evaluate``, ``eddies.render``.
"""
__version__ = "0.1.0"
