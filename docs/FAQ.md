# FAQ

**What exactly is an Eddie?** A primitive bound to a task frame (e.g. `grasp:peg`), together with a linear
closed-loop model `(A_i, B_i, d_i)` of how *this* robot tracks it and a conformal envelope `ε_i` on the model's
residual, all in one token (Definition 1). The envelope is part of the Eddie; it is not a hard bound but holds with
probability at least 1 − α over episodes.

**What is an Eddie stream, and what is Streamer?** The stream is the run of Eddies that performs a task, chosen at
run time by Streamer's product-game supervisor from an LTL_f goal and the primitives' contracts (Sec. 4.3). We say
"stream", never "sequence": the order, the recoveries and the stops are decided as the robot goes.

**Why is the method key `bistl`?** It is the registry key of the Streamer, and every episode directory of the
dataset uses it (`.../<task>/bistl/seed_XXXX`; `eddies/config.yaml: method`). The class is `StreamerMethod`
(`eddies/envs/methods/streamer.py`).

**What does the certificate cover?** Tracking only: the distance of the hand from where the running Eddie's model
says it settles under the current command, in metres (`‖S η_k‖`). Whether a skill *succeeds* (a grasp holds, a peg
seats) is left to the symbolic contracts and the supervisor. The equilibrium lies a median 0.2 cm from the command
on the Franka (App. D), so the tube is also a bound on the error to the command up to that offset.

**Fixed-calibration gate vs self-calibrating gate?** Both check before every command. The fixed-calibration gate
keeps the models and the coverage level *q* from the offline calibration; in the tables it also checks every planned
Eddie's tube before motion and stops at the first residual outside the envelope. The self-calibrating gate
(Algorithm 1) checks only the next command, widens the envelope within an episode when the score rises, refits the
models at hand-overs and moves *q* after every episode by `q_{t+1} = q_t + γ (err_t − α)`. From the same calibration
they finish 890 and 1,955 of 2,000 runs at 5 cm; checking only the next command with the calibration fixed finishes
1,968 (Sec. 5.1, App. D).

**What is a certified stop?** The gate refuses the next command, so the robot holds its last one. In the replayed
simulation runs the run then ends unfinished; on the real robot the arm holds until the check passes again (the cube
pushed out of the safe set and back, Fig. 8).

**What does "coverage level q" mean?** The conformal quantile of the episode scores: with N calibration episodes
and sorted scores, `q = s_(⌈(N+1)(1−α)⌉)` (Eq. 3). One *q* per task covers every Eddie of the stream at once because
each Eddie has its own normalizer `κ_i`; `ε_i = q κ_i`.

**Why are the tubes in Table D.1 called "ultimate"?** The tube of Eq. 4 settles, for a stable model, at
`ρ g_i(∞) + ε_i ω_i(∞)` (Proposition E.2); that radius is what the tables report. The gate itself evaluates the
full transient bound.

**What is "caught"?** Of the runs that *would* leave μ without a gate (known because the gates are replayed on
logged ungated runs, or because the fast robot runs were executed ungated), the number the gate refuses or stops
before they do.

**Can I use the gate on my robot?** Yes, if you can log `z` and `e` in the same normalized chart at 50 Hz with the
active Eddie name: `eddies.cert.gate.RuntimeGate` is the online object (`plan()`, `enter()`, `step()`), and
`eddies.cert.experiments.gate_adaptive` the self-calibrating loop. [REAL_ROBOT.md](REAL_ROBOT.md) describes the
Franka setup and the log format.

**Do I need the episode logs?** Not for the tables and figures: `python -m eddies.evaluate all` works from the
committed `results/` (the pooled gate table and the Table D.1 one-step column from `results/gate_margin/summary.json`). You need them (2.4 GB on Hugging Face, subsets available) to recompute the result files,
and the simulator to recollect them.

**Does it run without a GPU?** Yes. Identification, calibration and the gates are NumPy; the MuJoCo tasks run on
CPU. A GPU (or OSMesa) is only needed for rendering videos and stills, and Isaac Sim needs an RTX GPU.

**How long does the simulation take?** One cube-stacking episode takes about a second of wall time without video
(`try_task`); a task's 600 episodes are hours of CPU, so run them on a multi-core machine (`scripts/reproduce.sh collect-sb --workers N`). Rendering an episode at 1080p takes
about a minute.

**Why do Table C.2 / D.1 say 7 Eddies for peg tightening but 16 for staged insertion?** Peg tightening repeats a
four-Eddie cycle six times (26 steps, 7 distinct Eddies); staged insertion runs five Eddies twice (16 steps,
11 distinct). The paper counts the first as distinct and the second as plan length; `eddies.evaluate tableD1`
prints both.

**Which numbers are taken from the paper rather than recomputed?** Not recomputed by this release: the other token rows of Table 1 (one global model, one model per primitive, BeT,
FAST + k-means, VQ-BeT) with their App. D results, the reference-model one-step error and the second Franka split
of App. D, and the robot rows of Table 3 and Sec. 5.5 (printed from `results/real/table3_robot_rows.json`; the
robot logs are not released). See
[PAPER_RESULTS.md](PAPER_RESULTS.md).

**Where do the colours come from?** The paper's scheme (`eddies/config.yaml: plots.colors`): navy `#221b45` ink,
blue `#2e90ef` Franka / Eddies / tube, orange `#f39a2b` G1 / envelope, violet `#6a1fd6` LEAP hand / certified bound,
`#2e4b6c` Aero hand / arrows, red `#d6452b` stop. The SVGs, the video overlays and the website use the same.

**How do I cite?**

```bibtex
@inproceedings{narendra2027eddies,
  title     = {Eddies: Dynamical Action Tokens with Certified Composition},
  author    = {Narendra, Aditya and Tadevosyan, Grik and Swikir, Abdalla},
  booktitle = {Learning for Dynamics and Control (L4DC)},
  year      = {2027}
}
```
