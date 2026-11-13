"""Eddies: identification, certification and the certify-or-stop gates.

Modules
-------
collect        run instrumented Streamer episodes (train / calibration / test seed splits)
data           load trajectories (z, e, Eddie label) and manifests from episode folders
ident          per-Eddie affine models z+ = A z + B e + d + w (least squares with shrinkage)
tube           propagated tube, episode-level conformal quantile (coverage level q), chaining
gate           fixed-calibration gate (RuntimeGate, evaluate_episode)
experiments/   one module per paper experiment (each keeps its main()); table1_eddies and
               calibration_ablation are the rows of Table 1
tables/        bootstrap intervals (Table 1) and the pooled gate-margin table
plots/         the figures of the paper and the supplement
proposer/      learned Eddie proposer and learned command proposer (Appendix D)
"""
