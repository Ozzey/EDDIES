# Paper results: provenance

Every figure, table and headline number of *Eddies: Dynamical Action Tokens with Certified Composition* (main
paper and supplement), with the result file in `results/` it is computed from, the module that produced that file
and the episodes behind it. `results/MANIFEST.json` carries the same mapping per file (sizes, sha256, `used_by`);
[REPRODUCTION.md](REPRODUCTION.md) and `scripts/reproduce.sh` give the commands.

## Main paper

| item | what it shows | computed from | produced by | episodes |
| --- | --- | --- | --- | --- |
| Fig. 1 | Streamer chains certified Eddies into an Eddie stream; each Eddie gates its own commands | schematic (`assets/architectures/eddies_architecture.svg` is the vector redraw) | — | — |
| Fig. 2 | projection onto trust radius and slew bound | schematic; geometry as `eddies/cert/plots/projection.py` | — | — |
| Fig. 3 | how an Eddie is built: model from rollouts + conformal envelope → tube | schematic | — | — |
| Fig. 4 | (a) scores and quantile *q* on staged insertion: 95 % of the first 100 test episodes covered (94 % of 200); (b) the chained bound vs the measured error on one episode | `figs_data/trace_fmb.json` | `eddies.gate trace` (`experiments/trace_figs.py`) → `eddies/cert/plots/cert_pair.py` | `sb` Franka |
| Fig. 5 | finished runs vs tolerance, both gates, per robot | `embodiments/embodiments.json` (fixed), `adaptive/adaptive_<task>_known_<robot>.json` (self-calibrating) | `eddies.certify --all`, `eddies.gate self --group --known` → `plots/results.py: fig_gate` | `sb`, 200 test per task |
| Fig. 6 | the ten tasks in MuJoCo and Isaac Sim | stills (dataset `stills/`) | `eddies/envs/scripts/stills.py`, `eddies/isaac/stills.py` → `plots/env_strip.py` | seed 2000 |
| Fig. 7 | four Franka tasks held out of training: finished runs and running miss rate | `adaptive/adaptive_<task>.json`, `_known.json` | `eddies.gate self --task [--known]` → `plots/results.py: fig_heldout` | `sb` Franka, first 100 test |
| Fig. 8 | the real Franka: setup, camera feed, depth heights, a perturbed run with a certified stop | `results/real/calib/T_base_cam.yaml`, the robot photo and frames (`assets/results/real_setup.png`, `real_stop/`) | `plots/real_setup.py`, `real_perturb.py` | robot session |
| Table 1 | Eddies rows: Eddies and the two ablations (one reference model, per-step bound), per robot; bootstrap intervals ±10 runs | `table1/{eddie,eddie_refonly,calibration}.json`, `table1_robots/{leap,aero,g1}/*.json`, `table1/table1_ci_final.json` | `eddies.gate table1 --episodes`, `eddies.gate calibration`, `tables/table1_ci.py` → `eddies.evaluate table1` | `sb` (Franka test 2000–2099, others 2000–2199) |
| Table 2 | certificates moved LEAP ↔ Aero and MuJoCo → Isaac Sim, borrowed vs re-identified, μ = 15 cm | `embodiments.json` (`cross`), `physics/physics.json` (`refit10`) | `eddies.certify --all`, `eddies.gate physics` | `sb`; Isaac `sb` (10 refit + 50 calibration) |
| Table 3 | cube stacking in MuJoCo / Isaac Sim / robot, both gates, caught at 2–4× | `compare/stack_{mujoco,physx_sim,physx_own}.json`; robot rows `real/table3_robot_rows.json` (reported) | `eddies.gate stack` | `sb` + `gate` (MuJoCo 2000–2099; Isaac 2000–2049) |
| Sec. 1, contributions | 4.3 cm tube, 275 of 400 at 5 cm | `table1/eddie.json` | as Table 1 | |
| Sec. 4.1 | equilibrium a median 0.2 cm from the command; error to the command inside the tube on 99.9 % of test steps | `eqoff/eqoff.json` | `scripts/facts/eq_offset.py` | `sb` test |
| after Theorem 1 | 81 of the 85 certified Eddies have `a_i < 1` | `facts/gain_all.json` | `scripts/facts/gain_check.py` | `sb` |
| Sec. 5.1 | 5 violations in 2,000; envelopes missed in 6–16.5 % (Franka) and 8.5–10.5 % under the self-calibrating gate; 890 / 1,968 / 1,955 of 2,000 at 5 cm; at 2 cm 1,627 would leave μ and 97 % are stopped; at 3 cm 387 of 395, 1,328 finish; 35–100 % leave μ when 1.5–4× faster; 77–100 % vs 69–86 % of hand-overs certified at 5 cm (containment vs recursion) | `embodiments.json`, `adaptive/*_known_*.json`, `pool/final_indist_eddy.json`, `gate_margin/summary.json`, `gate/gate_eddy.json`, `gate/containment.json` | `eddies.certify`, `eddies.gate self / margin / margin-table / fixed / containment` | `sb`, `gate` |
| Sec. 5.2 | per-step envelope certifies no plan; reference model 1.5–7.6× wider | `table1/calibration.json`, `eddie_refonly.json` (+ robots) | `eddies.gate calibration / table1` | `sb` |
| Sec. 5.3 | re-identified certificates finish > 80 %; drawer 3.0 vs 8.3 cm, reach peg 2.9–4.9 cm | `embodiments.json`, `physics.json` | | |
| Sec. 5.4 | new task: nearly every run finished with 1–5 false stops; conformal certificate needs ~55 episodes for 83–93; push primitive; four plans recovered, five new goals 99–100 %, three refused; proposer 92–100 % of hand-overs, 99–100 % success | `adaptive/adaptive_<task>.json`, `transfer/transfer_*.json`, `pool/final_grow_v{0,1}.json`, `gate/ltl.json`, `ltl/compose_check_2000.json`, `tokprop/eval_*.json` | `eddies.gate self / transfer / pool / ltl`, `compose_check`, `token_proposer_eval` | `sb`, `ltl`, `tokprop` |
| Sec. 5.5 | fixed-calibration gate stops 19 / 20 of 50 (9 / 7 re-identified), self-calibrating 2–5; 2–4× caught 100 % / 95–99 %; certified stop holds 15 s | `compare/stack_physx_*.json`, `real/table3_robot_rows.json` | | Isaac `sb`/`gate`; robot |

## Supplement

| item | what it shows | computed from | produced by |
| --- | --- | --- | --- |
| App. A, Theorem A.2 text | four plans recovered, five new goals (99–100 % of 100 episodes), three refused | `ltl/compose_check_2000.json`, `gate/ltl.json` | `compose_check`, `eddies.gate ltl` |
| Fig. B.1 | one-step gain vs spectral radius; run factor per Eddie | `facts/gain_all.json` | `plots/supp.py` (`supp_gains`) |
| Table B.1 | identification per robot: m, n, weights, S, reference Eddie | `configs/robots/*.yaml`, `embodiments.json` | — |
| App. B text | 34 of 35 Franka Eddies certified (backing off: 16 transitions); 25 / 13 / 13; largest spectral radius 0.913; hole search 193 → 2.7 cm with shrinkage | `embodiments.json`, `facts/shrink_probe.json` | `eddies.certify`, `scripts/facts/shrink_probe.py` |
| App. B gains text | gains 0.64–0.85 / 0.86–0.92 / 0.84, four G1 Eddies up to 1.41; run factors ≤ 0.011 / 0.21 / 0.34 / 0.23 | `facts/gain_all.json` | `scripts/facts/gain_check.py` |
| Table C.1 | the eleven primitives | `eddies/envs/skillbook/` | — |
| Table C.2 | the ten tasks, Eddies, LTL_f formulas, horizons | `configs/tasks/*.yaml`, `embodiments.json` | — |
| Fig. C.1 | peg insertion in Isaac Sim | stills (dataset) | `eddies/isaac/stills.py` |
| App. C text | PhysX refit 10 + 50 (stacking 50 + 50); real protocol | `physics.json`, `compare/`, [REAL_ROBOT.md](REAL_ROBOT.md) | |
| Table D.1 | per-task certificates and the three gates; totals 96 Eddies (85 certified), 5 viol., 890 / 1,968 / 1,955, 1581 / 1627 | `embodiments.json`, `adaptive/*_known_<robot>.json`, `gate_margin/summary.json` (one-step column) | `eddies.evaluate tableD1` |
| App. D pooled text | 1,968 / 387 of 395 / 1,328 | `gate_margin/summary.json` (counts of the per-step logs `<task>.npz`, `<task>_a.json` in the dataset, `derived/gate_margin/`) | `eddies.gate margin`, `eddies.gate margin-table` |
| Table D.2 | held-out Franka tasks, no calibration vs own calibration as prior | `adaptive/adaptive_<task>.json`, `_known.json` | `eddies.evaluate tableD2` |
| Table D.3 | per Eddie vs per primitive: coverage, bound, tightness; drawer coverage 0.835 (Beta(181, 20)) | `pool/final_indist_{eddy,primitive}.json` | `eddies.gate pool` |
| App. D text after Table D.3 | one quantile over the mixture of tasks is not valid per task: per primitive the drawer's coverage falls to 0.68, per Eddie the stack's and the drawer's to 0.87 and 0.83 | `pool/mixture_{eddy,primitive}.json` | `eddies.gate pool` without `--mondrian` |
| App. D eq. offset | 0.20 / 1.53 / 1.92 cm (Franka), 0.66 / 2.98 / 3.24 (G1), 0.59 / 1.99 / 2.43 (LEAP), 0.23 / 1.82 / 2.44 (Aero); exceedances 0.09 / 0.88 / 0 / 0.0001 % | `eqoff/eqoff.json` | `scripts/facts/eq_offset.py` |
| App. D learned proposer | 91.7 / 99.9 / 99.97 % of hand-overs; guarded proposer reproduces the supervisor on all 100 seeds; BC command policy 16 / 100 with 27 caught of 27 | `tokprop/eval_*.json`, `tokprop/*_train.json` | `token_proposer`, `command_proposer`, `token_proposer_eval` |
| App. D Table 1 text | Eddies at 10 cm (359 / 174 / 340); per-step bound needs 4,769–22,499 calibration steps per task, Aero wheel certified at 10 cm (192 runs) | `table1/eddie.json`, `table1_robots/*/eddie.json`, `table1/calibration.json` | `eddies.gate table1`, `eddies.gate calibration` |
| Table D.4 | containment vs recursion per tolerance | `gate/containment.json` | `eddies.gate containment` |
| Table D.5 | certify or stop under 1.5–4× faster commands, calibrated at nominal vs commanded speed | `gate/gate_eddy.json`, `gate_eddy_recal.json` | `eddies.gate fixed [--recal]` |
| Table D.6 | tube radius of the Eddies that run on several robots | `embodiments.json` (`shared`) | `eddies.evaluate tableD6` |
| Fig. D.1, App. D dynamics text | PCA of dynamics descriptors; distances 4.31 (same Eddie, two robots) vs 4.91 (two Eddies, one robot), p = 0.33; 0.74 MuJoCo vs Isaac | `dynamics_map/dynamics_map.json`, `dynamics_map_stats_own.json` | `eddies.cert.experiments.dynamics_map`, `plots/dynamics_map.py` |
| Fig. E.1 | coverage level `q_t` on each held-out Franka task | `adaptive/adaptive_<task>.json` | `plots/supp.py` (`supp_qtrace`) |
| Prop. E.4 example | push registered after nine primitives: every staged-insertion Eddie certifies, nine unchanged | `pool/final_grow_v0.json`, `final_grow_v1.json` | `eddies.gate pool --extra franka_fmb:push` |

## Values taken from the paper

Not recomputed by this release: the other token rows of Table 1 (one global model, one model per primitive, BeT,
FAST + k-means, VQ-BeT) with their App. D results, the reference-model one-step error and the second Franka split
of App. D, and the robot rows of Table 3 and Sec. 5.5 (printed from `results/real/table3_robot_rows.json`; the
robot logs are not released).

## Conventions of `eddies.evaluate` (where a printed cell could be read two ways)

| table | convention | why |
| --- | --- | --- |
| Table 1, row "Eddies, per-step bound" | the tube p90 is printed only where the per-step bound lets some run finish at some tolerance of the calibration grid (3–15 cm); otherwise `--` | the paper prints `--` for Franka, LEAP and G1 (nothing finishes at any tolerance) and 8.7 cm for Aero (192 runs finish at 10 cm) |
| Table 1, bootstrap | `B: 2000` draws in `eddies/configs/experiments/evaluate.yaml` (the paper's bold marks rest on the CI); `--B 0` skips it | 1–2 s per robot |
| Table D.1, column "Eddies" | the paper's count: the steps of the nominal plan (`tasks/<t>.yaml: n_eddies`), except that the tightening loop grasp–turn–release–unwind of the two peg-tightening tasks counts once (7 Eddies; the executed plan has 26 steps); total 96. The second column "distinct" counts the distinct Eddies of each plan (11 for staged insertion, total 91); "85 certified" counts distinct Eddies certified per robot | reproduces the paper's column as printed; both counts come from the same files |
| Table 3, robot rows | printed from `real/table3_robot_rows.json` under their own heading (values reported in the paper) | the MuJoCo and Isaac Sim rows are computed from `compare/stack_*.json` |
| Table D.6, Eddie names | `transport (above hole)`, `insert (hole)`, `place (goal)`, `turn peg (unwind)` as in the supplement (`plots/shared_own.py: PAPER_NAMES`) | cosmetic |
