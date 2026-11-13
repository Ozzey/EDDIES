# results/ — the paper's result logs

Every file here is an input of a table, figure or number in *Eddies: Dynamical Action Tokens with
Certified Composition* (main paper or supplement). `MANIFEST.json` lists each file with its size,
sha256, the paper artefacts that use it (`used_by`, ids as in the manifest's `artefact_ids`) and the
command that produces it. `eddies/evaluate.py` reproduces the tables and figures from this folder; the raw
episode logs behind these files are the Hugging Face dataset described in `datasets/README.md`, and
`scripts/reproduce.sh <experiment>` rewrites each file from them (`scripts/reproduce.sh --list`; the
"reproduce.sh" column below names the experiment).

| folder | what | `reproduce.sh` | paper |
|---|---|---|---|
| `embodiments/embodiments.json` | certificates of the ten tasks (Eddies, violations, tube widths, fixed-calibration gate), cross-robot transfer | `certify` | Tab. 2, B.1, C.2, D.1, D.6; Fig. 5; Sec. 5.1, 5.3 |
| `adaptive/adaptive_<task>_known_<robot>.json` | self-calibrating gate on the known tasks | `adaptive-groups` | Fig. 5, Tab. D.1, Sec. 5.1 |
| `adaptive/adaptive_<task>[_known].json` | each Franka task held out of training (few-shot) | `adaptive` | Fig. 7, E.1; Tab. D.2; Sec. 5.4 |
| `table1/`, `table1_robots/{leap,aero,g1}/` | Eddies (`eddie.json`) and the ablation with one reference model (`eddie_refonly.json`); `calibration.json` = the per-step-bound ablation; `table1_ci_final.json` = bootstrap intervals | `table1`, `table1-robots` | Tab. 1, Sec. 5.2, App. D |
| `physics/physics.json` | MuJoCo -> PhysX certificate transfer and re-identification | `physics` | Tab. 2, Sec. 5.3 |
| `compare/stack_{mujoco,physx_sim,physx_own}.json` | cube stacking in MuJoCo and Isaac Sim, both gates | `stack` | Tab. 3 |
| `real/` | real Franka stacking: `table3_robot_rows.json` (Tab. 3 robot rows, values reported in the paper) and the camera calibration (`calib/T_base_cam.yaml`) | — | Tab. 3, Sec. 5.5, Fig. 8 |
| `gate/gate_eddy{,_recal}.json` | certify-or-stop under 1.5-4x faster commands | `fixed` | Tab. D.5, Sec. 5.1 |
| `gate/containment.json` | funnel containment vs the chained recursion (Theorem 1a) per tolerance | `containment` | Tab. D.4 |
| `gate/ltl.json`, `ltl/compose_check_2000.json` | LTLf product game: four hand-written plans recovered, five new goals | `ltl` | Sec. 5.4, App. A, Tab. C.2 |
| `gate_margin/summary.json` | counts of the gate-margin replay per tolerance, method and task (`eddies.cert.tables.gate_margin --out`); the per-step logs it is computed from (`<task>.npz`, `<task>_a.json`, 68 MB) are in the dataset under `derived/gate_margin/` (`datasets/download.sh --subset gate_margin` copies them here) | `gate-margin` | Sec. 5.1 (1,968 / 387-395 / 1,328), Tab. D.1 one-step column, App. D pooled |
| `pool/final_*.json` | certificates per Eddie vs per primitive; vocabulary growth with the push primitive | `pool` | Tab. D.3, Sec. 5.1, 5.4, App. E |
| `pool/mixture_{eddy,primitive}.json` | the same certificates with one quantile over the mixture of the four tasks (no per-task calibration) | `pool` | App. D text after Tab. D.3 |
| `transfer/transfer_*.json` | conformal certificate with 5/10/20/55 calibration episodes | `transfer` | Sec. 5.4 |
| `tokprop/` | learned Eddie proposer: training curves and held-out evaluation | `tokprop` | Sec. 5.4, App. D learned |
| `dynamics_map/` | dynamics descriptors (PCA) of the Eddies on MuJoCo and PhysX | `dynmap` | Fig. D.1 |
| `figs_data/trace_fmb.json` | one staged-insertion trace with its tube and envelope | `trace` | Fig. 4 |
| `eqoff/eqoff.json` | equilibrium offset from the command per Eddie | `facts` | Sec. 4.1, App. D |
| `facts/gain_all.json`, `shrink_probe.json` | one-step gains a_i, spectral radii, ridge-shrinkage probe | `facts` | Sec. 4.2, Fig. B.1, App. B |
| (producers) | the scripts that wrote `eqoff/` and `facts/` are in `scripts/facts/` (`eq_offset.py`, `gain_check.py`, `shrink_probe.py`; `scripts/reproduce.sh facts`) |  | |

The per-step gate-margin logs (`gate_margin/<task>.npz`, `<task>_a.json`) are in the dataset under
`derived/gate_margin/` (rewritten by `scripts/reproduce.sh gate-margin`). The robot rows of Table 3 are the values
reported in the paper (`real/table3_robot_rows.json`).
