# Architecture: from the paper to the code

Equation numbers, Theorem 1 and Algorithm 1 are those of the main paper (`docs/paper.pdf`); lettered appendices
are in the supplement (`docs/supplementary.pdf`). Every item names the module and function that implements it.
The vocabulary: an **Eddie** is the token, an **Eddie stream** is a composed run of Eddies, **Streamer** is the
composer, the envelope **is part of** the Eddie.

```
eddies/
  core/        certificates, compose (contract graph), contracts (product game), ltlf, embedding (projection), control
  cert/        ident (models), tube (tubes, scores, chaining), gate (fixed-calibration gate),
               experiments/* (the paper's runs), tables/*, plots/*, proposer/* (learned Eddie proposer)
  envs/        framework (Task, Supervisor, runner, recorder), skillbook (the primitives), tasks (the ten tasks),
               controllers (differential IK per robot), methods/streamer.py (Streamer)
  isaac/       optional PhysX bridge driving the same task code
  common/      config loader, path layout
```

## 1. Setting: the embedding chart, the trust radius and the slew bound (Sec. 3, Eq. 1)

Commands `e_k ∈ R^m` and measured hand poses `z_k ∈ R^n` live in a normalized chart of the task space (position
and rotation vector in [−1, 1], hand shape and force in [0, 1]); fixed diagonal weights `M` on `e` and `W` on `z`
define the norms (Table B.1).

| paper | code |
| --- | --- |
| the chart, `M` (`embedding.metric_diag` in every manifest) | `eddies/envs/framework/embedding.py` `EmbeddingSpace` (normalize / denormalize / norm); the weights per robot in `eddies/configs/robots/<robot>.yaml: embedding.weights` and `eddies/envs/controllers/{franka,g1,hand_arm}.py` |
| certified block `J` (pose rows of `z`), position rows `S` | `eddies/configs/robots/*.yaml: block`, `eddies.common.config.blocks()`; `S` is built in `eddies/cert/tube.py: make_tube` (metres per normalized unit = `scale`) |
| projection onto trust radius δ_i and slew bound ρ_i, Eq. (1), Fig. 2; Proposition A.1 (well posed) | `eddies/core/embedding.py` `EmbeddingProjector.project(proposed, previous, rho, anchor, delta)` (dual Newton on box ∩ trust ball ∩ slew ball); called from `eddies/envs/methods/streamer.py` `StreamerMethod._project` (incl. the bridging step of App. A when the two balls are disjoint) |
| ρ per robot (0.012 / 0.0075 / 0.009 per 20 ms step) | `configs/robots/*.yaml: rho_step`, `eddies/envs/tasks/{franka,g1,hand}_base.py` |

## 2. What an Eddie carries (Definition 1, Eq. 2, Fig. 3)

`z_{k+1} = A_i z_k + B_i e_k + d_i + w_k`, `‖W^{1/2} w_k‖ ≤ ε_i`.

| paper | code |
| --- | --- |
| primitive bound to a frame (`primitive:binding`) | `eddies/envs/skillbook/book.py` `SkillBook.eddy(task, primitive, binding, anchor, delta, …)` builds the Eddie from a registered `Primitive` (`eddies/envs/skillbook/primitives.py` `build_reach`, `build_grasp`, … `build_retreat`; G1 and hand variants in `g1.py`, `hand.py`); the names logged in `trajectory.npz["skill"]` |
| least-squares model `(A_i, B_i, d_i)`, shrunk toward the reference Eddie, Eq. (B.1) | `eddies/cert/ident.py` `fit(eps, skill, J, p, weights, prior, shrink)` → `SkillModel` (`Θ = [A B d]`, `λ = shrink · n_i`); the grid `{0, 0.03, 0.3, 3}`, the reference rule (≥ 5,000 transitions, smallest training tube) and the selection are in `eddies/cert/experiments/gate_fixed.py` `build_certificates` (used by every experiment) |
| equilibrium `z*_i(e) = (I − A_i)^{-1}(B_i e + d_i)`, tracking error `η_k` | `eddies/cert/ident.py` `SkillModel.equilibrium`, `eddies/cert/tube.py` `Tube.equilibrium`, `eddies/cert/gate.py` `equilibrium`, `gate_adaptive.py` `eta_of` |
| normalizer `κ_i` = 90th percentile of the per-episode max residual on the training episodes | `eddies/cert/tube.py` `Tube.residual_scores` + `conformal.kappa_quantile` (0.9) in `gate_fixed.build_certificates` |
| envelope `ε_i = q κ_i`, `q = s_(⌈(N+1)(1−α)⌉)`, Eq. (3); Lemma E.1 | `eddies/cert/tube.py` `joint_scores(tubes, kappa, calib_eps)` (score `s = max_k ‖W^{1/2} w_k‖ / κ_{i(k)}`), `conformal_quantile(scores, alpha)`; `conformal.alpha = 0.1` |
| certified only if spectral radius < 0.999 and ≥ 50 transitions (App. B) | `Tube.prepare` (`identification.max_spectral_radius`), `make_tube` (`identification.min_train_transitions`) |
| equilibrium offset from the command (Sec. 4.1, App. D "Equilibrium and command") | `scripts/facts/eq_offset.py` → `results/eqoff/eqoff.json` |

## 3. The tube inside an Eddie (Eq. 4, Proposition E.2)

`‖S η_{k0+j}‖ ≤ ‖S A_i^j W^{-1/2}‖ ‖W^{1/2} η_{k0}‖ + ρ_i Σ_{l<j} ‖S A_i^l G_i M^{-1/2}‖ + ε_i Σ_{l<j} ‖S A_i^l W^{-1/2}‖`

| paper | code |
| --- | --- |
| the three sums `g_i(j)`, `β_i(j)`, `ω_i(j)` of Eq. (E.2) over the horizon | `eddies/cert/tube.py` `Tube.prepare`: arrays `cg` (`‖S A^j G M^{-1/2}‖`), `cb`, `cw` (`‖S A^j W^{-1/2}‖`), `powers_S` (`S A^j`), `cA` (`‖W^{1/2} A^j W^{-1/2}‖`), `cgW`; `identification.horizon = 4000` steps |
| the tube on one execution, violation check | `Tube`-level `check_episode(t, eps, rho, ep)` (`bound = tr + ρ·cum_g + ε·cum_w`, `D_viol`); the trust-radius variant with `anchor`/`delta` |
| ultimate radius `ρ g_i(∞) + ε ω_i(∞)` (the "tube" of Tables D.1, D.6) | `Tube.slew_radius(rho, eps)`; `rho_star` / `delta_star` invert it for a tolerance |
| per-task test: violations, tube median / p90, fixed-calibration gate | `eddies/cert/experiments/embodiments.py` `task_eval` → `results/embodiments/embodiments.json` |

## 4. Chaining through hand-overs (Theorem 1a, Eq. 7; Table D.4)

`b_{k+1} ≤ a_i b_k + ε_i + J_{ii'}`, with `a_i = ‖W^{1/2} A_i W^{-1/2}‖` and `J_{ii'}` the jump between equilibria.

| paper | code |
| --- | --- |
| the chain over an episode, started from the measured error at the first step; restart at an uncertified Eddie | `eddies/cert/tube.py` `compose_episode(tubes, eps, rho_of, ep)` (`b_{n+1}(0) = cA[1]·b_n(end) + ε_n + J_n`); `ordered_segments` splits the stream |
| per-Eddie vs per-primitive certificates, coverage, bound, tightness (Table D.3) | `eddies/cert/experiments/pool.py` → `results/pool/final_indist_{eddy,primitive}.json` |
| funnel containment vs the recursion (Table D.4) | `eddies/cert/experiments/containment.py` → `results/gate/containment.json` |
| one-step gains `a_i`, run factor (App. B, Fig. B.1; "81 of 85 Eddies have a_i < 1") | `scripts/facts/gain_check.py` → `results/facts/gain_all.json`; `eddies/cert/plots/supp.py` |
| the chained bound along one episode, Fig. 4b | `eddies/cert/experiments/trace_figs.py` → `results/figs_data/trace_fmb.json`; `eddies/cert/plots/cert_pair.py` |

## 5. The gate (Eq. 5, Theorem 1b, Proposition E.3a)

`η̂_{k+1} = A_i z_k + B_i e_k + d_i − z*_{i'}(e_{k+1})`; send `e_{k+1}` only if
`‖S η̂_{k+1}‖ + ε_i ‖S W^{-1/2}‖ ≤ μ`; stop if `‖S η_k‖ > μ`.

| paper | code |
| --- | --- |
| **fixed-calibration gate** (plan check before motion, hand-over check from the measured error, residual monitor; the gate of Tables D.1 "fixed", D.5, and the "fixed gate" of Table 3) | `eddies/cert/gate.py` `RuntimeGate` (`plan` → `enter` → `step`), `evaluate_episode(ep, plan, tubes, eps, mu, rho)` with the counterfactual truth of the ungated run; `handover_bound` is the full transient bound Eq. (4) from the hand-over error |
| **one-step check** of Eq. (5) (jump included) | `eddies/cert/experiments/gate_adaptive.py` `safe_next(t, t_next, z, e, e_next, eps, H)` (`H = adaptive_gate.lookahead = 0`) |
| certify-or-stop under faster commands, recalibrated at the commanded speed (Table D.5) | `eddies/cert/experiments/gate_fixed.py` (`--recal`) → `results/gate/gate_eddy{,_recal}.json` |
| one-step check with fixed calibration vs threshold monitors at intermediate tolerances (App. D pooled: 1,968 / 387 of 395 / 1,328) | `eddies/cert/experiments/gate_margin.py`, `eddies/cert/tables/gate_margin.py` → `results/gate_margin/summary.json` |
| online form for a robot loop | `RuntimeGate` is the object the robot driver calls (`plan()` once, `enter()` at every hand-over, `step()` every tick) |

## 6. Self-calibration (Algorithm 1, Eq. 6, Theorem 1c, Proposition E.3b)

`q_{t+1} = q_t + γ (err_t − α)`, `err_t = 1{s_t > q_t}`; `ε_i = max(q_t, s) κ_i` within an episode; refit toward
the prior at hand-overs, kept if stable; `|mean(err) − α| ≤ (s_max + γ)/(γ T)`.

| paper | code |
| --- | --- |
| Algorithm 1 as a whole (one episode stream, known or held-out task) | `eddies/cert/experiments/gate_adaptive.py` `run(H, data, mus, alpha, eta_q, n_test, known, …)` (`--known` starts from the task's own calibration; without it the priors come from the other tasks only) |
| priors: own Eddie, else primitive, else one global model; prior weights 200 / 50 / 10 pseudo-transitions | `gate_adaptive.py` prior selection in `run` (`adaptive_gate.lam0`), `OnlineEddie(name, prior, kappa, horizon, lam)` |
| ridge refit toward the prior at hand-overs, kept if spectral radius < 0.995; frozen during a run | `OnlineEddie.add` / `OnlineEddie.refit` (`tube_from`, `adaptive_gate.max_spectral_radius`) |
| `κ_i` from the Eddie's own runs after 5 runs | `OnlineEddie.kappa` (`adaptive_gate.kappa_min_episodes`) |
| the update of `q` (γ = `adaptive_gate.eta` = 0.5, no clipping) | `gate_adaptive.py` `run`: `err = float(score > q); q = q + eta_q * (err - alpha)` |
| results: Fig. 5 / Table D.1 (`--group <robot> --known`), Fig. 7 / Table D.2 / Fig. E.1 (held-out) | `results/adaptive/adaptive_<task>_known_<robot>.json`, `adaptive_<task>[_known].json` |
| few-shot re-identification of a held-out task for comparison (Sec. 5.4: 5 / 10 / 20 / 55 episodes) | `eddies/cert/experiments/transfer.py` → `results/transfer/` |

## 7. Streamer: contracts, the product game and the stream (Sec. 4.3, App. A, Theorem A.2)

| paper | code |
| --- | --- |
| symbolic contract of a primitive (precondition, outcomes) | `eddies/envs/skillbook/contracts.py` `schema(eddie, primitive, bind, scene, outcomes)`; `eddies/core/compose.py` `Schema`, `Effect`, `compose(schemas, initial)` (the contract graph over label cells) |
| the product state `(ℓ, y)`, the attractor sets `V_n` of Eq. (A.1), rank, finish-or-refuse | `eddies/core/contracts.py` `ContractGame` (`_expand` builds the product with the LTL_f residual, `_solve` computes the ranks by the strong-preimage iteration, `allowed` / `choose` pick a rank-decreasing Eddie); `eddies/envs/framework/supervisor.py` `Supervisor.decide` / `observe_outcome`, `expand_contracts` (failure budget, retries), `ResumableGame` |
| LTL_f formulas, progression, acceptance on finite traces | `eddies/core/ltlf.py` `parse`, `progress`, `nullable`, `dead`, `Monitor`, `evaluate` |
| the Eddie stream at run time: hand-overs, outcomes, re-plans | `eddies/envs/methods/streamer.py` `StreamerMethod.act` (tracker → supervisor decision → target → projection) |
| recovering the four hand-written plans, five new goals, three refusals (Sec. 5.4, App. A) | `eddies/envs/scripts/compose_check.py` → `results/ltl/compose_check_2000.json`; `eddies/cert/experiments/gate_ltl.py` → `results/gate/ltl.json` |
| vocabulary growth, Proposition E.4 (fixed share of α per primitive) | `eddies/envs/skillbook/book.py` `SkillBook.register` / `reserve` (`skillbook.alpha_share = 1/12`); `pool.py --calib per_label --extra franka_fmb:push` → `results/pool/final_grow_v{0,1}.json` |
| a learned Eddie proposer replacing the supervisor (App. D) | `eddies/cert/proposer/token_proposer.py` (MLP 64×64), `command_proposer.py`, `eddies/envs/methods/learned_tokens.py` `LearnedSupervisor`, `eddies/cert/experiments/token_proposer_eval.py` → `results/tokprop/` |

## 8. The two ablations of Table 1 (Sec. 5.2, App. D)

`eddies/cert/experiments/table1_eddies.py` certifies the same logged episodes with the pipeline of Sec. 4 (models,
`κ`, `q`, tubes, gate) for Eddies and for the ablation "Eddies, one reference model" (every Eddie gets the robot's
reference model with its own `κ_i`); `calibration_ablation.py` is the row "Eddies, per-step bound" (per-step
conformal quantile at level `α/K` with a union bound over the horizon); `eddies/cert/tables/table1_ci.py` the paired
bootstrap.

## 9. Simulation, data and the real robot

| | |
| --- | --- |
| episode = `manifest.json` + `trajectory.npz` (`t`, `e`, `z`, `skill`, `labels`, signals) | `eddies/envs/framework/runner.py` `run_episode`; readers `eddies/cert/data.py` `load_episode`, `load_split`, `episode_dirs` ([DATA_FORMAT.md](DATA_FORMAT.md)) |
| tasks (`Task`, `Skill`, labels, LTL_f formula, `nominal_plan`) | `eddies/envs/framework/task.py`; the ten tasks in `eddies/envs/tasks/` ([TASKS.md](TASKS.md)) |
| controllers: damped least-squares IK into joint position servos at 50 Hz, admittance loop (Franka insertion), G1 balance, hand synergies | `eddies/envs/controllers/{ik,franka,g1,hand_arm}.py`, `eddies/core/control.py` |
| PhysX: MJCF → USD, state write-back into the MuJoCo data | `eddies/isaac/` (`run_cert_task.py`, `bridge`, `export`) |
| real Franka | [REAL_ROBOT.md](REAL_ROBOT.md): the same `RuntimeGate` / `gate_adaptive.run` on logs in the episode format |
