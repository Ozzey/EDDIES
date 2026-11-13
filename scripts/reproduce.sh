#!/usr/bin/env bash
# Recompute the paper's result files (results/) from the episode logs, one experiment at a time.
#
#   scripts/reproduce.sh --list                     # the experiments, what they write and what they need
#   scripts/reproduce.sh certify table1 pool        # some experiments, in the order given
#   scripts/reproduce.sh all                        # every analysis below, in dependency order, then evaluate
#   scripts/reproduce.sh table1 --dry-run           # print the commands only
#   scripts/reproduce.sh gate-margin --workers 10   # --workers N for the steps that run a process pool
#
# Each experiment calls the same Python module with the same arguments as the authors' runs; nothing here is
# tied to a machine or a scheduler.
# Prerequisites (docs/REPRODUCTION.md): pip install -e ".[test]", scripts/download_menagerie.sh and the episode
# logs, datasets/download.sh --subset <the subsets in the "needs" column>.  The analyses only read episodes;
# collect-* recollect them instead of downloading (hours of CPU per task; Isaac Sim needs a GPU and ISAAC_PY).
# Then `python -m eddies.evaluate all` prints the tables and writes the figures.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"
DRY=""

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# datasets/download.sh writes datasets/episodes/{mujoco,isaac}/episodes/...; use those unless set otherwise
if [[ -z "${EDDIES_EPISODES:-}" && -d datasets/episodes/mujoco/episodes ]]; then export EDDIES_EPISODES="$ROOT/datasets/episodes/mujoco"; fi
if [[ -z "${EDDIES_RESULTS_ISAAC:-}" && -d datasets/episodes/isaac/episodes ]]; then export EDDIES_RESULTS_ISAAC="$ROOT/datasets/episodes/isaac"; fi
RES="${EDDIES_RESULTS:-$ROOT/results}"

FRANKA=(franka_fmb franka_peg_insertion franka_cube_stack franka_drawer)
F4=$(IFS=,; echo "${FRANKA[*]}")
GROUP=(franka franka franka franka g1 g1 leap leap aero aero)
TASK=("${FRANKA[@]}" g1_peg_insertion_sb g1_drawer_sb leap_wheel_sb leap_peg_tighten_sb aero_wheel_sb aero_peg_tighten_sb)
TOKPROP=(franka_cube_stack franka_fmb franka_peg_insertion)
TRAIN=sb:nominal:1000-1199,tokprop:perturbed:1000-1199
VAL=sb:nominal:1200-1399,tokprop:perturbed:1200-1399
TEST=tokprop:nominal:2000-2099,tokprop:perturbed:2000-2099

run() { echo "+ $*"; [[ -n "$DRY" ]] || "$@"; }
py() { run "$PY" "$@"; }
isaac_py() { [[ -n "${ISAAC_PY:-}" ]] || { echo "set ISAAC_PY to Isaac Sim's python (external/README.md)" >&2; exit 1; }
             run "$ISAAC_PY" "$@"; }

# ------------------------------------------------------------------ analyses (read episodes, write results/)
certify() {          # Tab. 2 (LEAP<->Aero), Tab. D.1, D.6, Fig. 5 (fixed), App. B  | 8 cores, 48 GB, hours
  py -m eddies.cert.experiments.embodiments
}
adaptive_groups() {  # Fig. 5 (self-calibrating), Tab. D.1, Sec. 5.1                 | 10 runs, 4 cores, 32 GB, < 4 h each
  for i in "${!TASK[@]}"; do
    py -m eddies.cert.experiments.gate_adaptive --group "${GROUP[$i]}" --task "${TASK[$i]}" --known --mu 2,3,5,7.5,10,15
  done
}
adaptive() {         # Fig. 7, Tab. D.2, Fig. E.1, Sec. 5.4                           | 8 runs, 4 cores, 32 GB, < 4 h each
  for t in "${FRANKA[@]}"; do
    py -m eddies.cert.experiments.gate_adaptive --task "$t" --mu 1,2,3,5,7.5,10,15
    py -m eddies.cert.experiments.gate_adaptive --task "$t" --mu 1,2,3,5,7.5,10,15 --known
  done
}
gate_margin() {      # App. D pooled gates, Tab. D.1 one-step column, Sec. 5.1 (1,968)  | ~16 GB per worker, < 5 h with 10
  py -m eddies.cert.experiments.gate_margin --workers "$WORKERS"
  py -m eddies.cert.tables.gate_margin --out "$RES/gate_margin/summary.json"
}
table1() {           # Tab. 1 Franka column (Eddies rows), bootstrap file             | 3 runs, 16 cores, 50 GB, hours
  for v in eddie eddie_refonly; do py -m eddies.cert.experiments.table1_eddies --variant "$v" --episodes --out table1; done
  py -m eddies.cert.experiments.calibration_ablation --episodes --out table1
  py -m eddies.cert.tables.table1_ci --dir table1 --B 2000 --json "$RES/table1/table1_ci_final.json"
}
table1_robots() {    # Tab. 1 G1 / LEAP / Aero columns                                | 9 runs, 16 cores, 64 GB, hours
  for r in g1 leap aero; do
    for v in eddie eddie_refonly; do py -m eddies.cert.experiments.table1_eddies --robot "$r" --variant "$v" --episodes --out "table1_robots/$r"; done
    py -m eddies.cert.experiments.calibration_ablation --robot "$r" --episodes --out "table1_robots/$r"
  done
}
containment() {      # Tab. D.4                                                        | minutes
  py -m eddies.cert.experiments.containment
}
pool() {             # Tab. D.3, App. D mixture text and Beta(181, 20), Prop. E.4       | minutes; mixture 64 GB
  for p in eddy primitive; do
    py -m eddies.cert.experiments.pool --train "$F4" --test "$F4" --pool "$p" --calib joint --select --mondrian --name "final_indist_$p"
    py -m eddies.cert.experiments.pool --train "$F4" --test "$F4" --pool "$p" --calib joint --select --name "mixture_$p"
  done
  local held=franka_peg_insertion,franka_cube_stack,franka_drawer
  py -m eddies.cert.experiments.pool --train "$held" --test franka_fmb --pool primitive --calib per_label --select --mondrian --n-calib 200 --name final_grow_v0
  py -m eddies.cert.experiments.pool --train "$held" --test franka_fmb --pool primitive --calib per_label --select --mondrian --n-calib 200 \
     --extra franka_fmb:push --name final_grow_v1
  py -m eddies.cert.plots.coverage_bb
}
transfer() {         # Sec. 5.4 (5 / 10 / 20 / 55 calibration episodes)               | 5 runs, 8 cores, 32 GB, < 10 h each
  for t in "${FRANKA[@]}"; do py -m eddies.cert.experiments.transfer --task "$t" --no-indist; done
  py -m eddies.cert.experiments.transfer --only-indist
}
trace() {            # Fig. 4                                                          | 4 cores, 32 GB, < 2 h
  py -m eddies.cert.experiments.trace_figs
}
facts() {            # Sec. 4.1 / App. D offsets, Thm 1 gains (Fig. B.1), App. B shrinkage | < 1 h
  py scripts/facts/eq_offset.py --robots franka,g1,leap,aero --out "$RES/eqoff/eqoff.json"
  py scripts/facts/gain_check.py "$RES/facts/gain_all.json"
  py scripts/facts/shrink_probe.py --out "$RES/facts/shrink_probe.json"
}
fixed() {            # Tab. D.5, Sec. 5.1 faster commands                              | minutes
  py -m eddies.cert.experiments.gate_fixed
  py -m eddies.cert.experiments.gate_fixed --recal
}
ltl() {              # Sec. 5.4, App. A: four plans recovered, five new goals          | minutes
  py -m eddies.cert.experiments.gate_ltl
  py -m eddies.envs.scripts.compose_check --seed 2000 --out "$RES/ltl"
}
physics() {          # Tab. 2 MuJoCo -> PhysX rows, Sec. 5.3                           | 8 cores, 32 GB, < 6 h
  py -m eddies.cert.experiments.physics
}
dynmap() {           # Fig. D.1 and its App. D statistics                              | 8 cores, 48 GB, < 4 h
  py -m eddies.cert.experiments.dynamics_map
  py -m eddies.cert.plots.dynamics_map --model own
}
stack() {            # Tab. 3 MuJoCo / Isaac Sim rows, Sec. 5.5                       | 3 runs, 4 cores, 48 GB, < 6 h each
  for r in mujoco physx_sim physx_own; do py -m eddies.cert.experiments.stack_compare --rows "$r"; done
  py -m eddies.cert.experiments.stack_compare --table
}
tokprop_train() {    # App. D learned proposer: training curves (tokprop/*_train.json) | 8 cores, 32 GB, hours per task
  for t in "${TOKPROP[@]}"; do
    py -m eddies.cert.proposer.token_proposer --task "$t" --train "$TRAIN" --val "$VAL" --test "$TEST"
  done
  py -m eddies.cert.proposer.command_proposer --task franka_cube_stack --train "$TRAIN" --val "$VAL"
}
tokprop_eval() {     # Sec. 5.4 / App. D learned proposer, both gates (tokprop/eval_*.json) | minutes per task
  py -m eddies.cert.experiments.token_proposer_eval --task franka_fmb --gate
  py -m eddies.cert.experiments.token_proposer_eval --task franka_peg_insertion --gate
  py -m eddies.cert.experiments.token_proposer_eval --task franka_cube_stack --gate \
     --methods bistl,bistl_learned,bistl_learned_raw,bistl_learned_state,bistl_learned_state_raw,bistl_cmd,bistl_learned_cmd
}
tokprop() { tokprop_train; tokprop_eval; }

# ------------------------------------------------------------------ episodes (instead of datasets/download.sh)
collect_sb() {       # sb: 10 tasks x 600 (train 1000-1199, calibration 1200-1399, test 2000-2199) | hours per task
  py -m eddies.collect --config experiments/collect_sb --workers "$WORKERS"
}
collect_gate() {     # gate: Franka test seeds at 1.5-4x the slew bound, labels bistl_rho<s>      | 12 cores, < 3 h
  py -m eddies.collect --config experiments/collect_gate --workers "$WORKERS"
}
collect_ltl() {      # ltl: hand-written vs composed plans and five new goals, nominal + perturbed | 12 cores, < 4 h
  local c=(-m eddies.cert.collect --workers "$WORKERS" --seeds 2000-2099 --conditions nominal,perturbed --tag ltl)
  py "${c[@]}" --tasks "$F4"
  py "${c[@]}" --tasks franka_cube_stack_ltl,franka_peg_insertion_ltl,franka_drawer_ltl,franka_fmb_ltl
  py "${c[@]}" --tasks franka_drawer_ltl --spec "G !bad & F(open_drawer)" --label drawer_open
  py "${c[@]}" --tasks franka_drawer_ltl --spec "G !bad & F(open_drawer & F(holding_box))" --label drawer_hold
  py "${c[@]}" --tasks franka_fmb_ltl --spec "G !bad & F(in_peg_fixture)" --label fmb_stage
  py "${c[@]}" --tasks franka_fmb_ltl --spec "G !bad & F(latched_peg)" --label fmb_direct
  py "${c[@]}" --tasks franka_fmb_ltl --spec "G !bad & F(in_peg_hole & F(in_peg_fixture))" --label fmb_hole_fixture
}
collect_tokprop() {  # tokprop supervisor episodes (before tokprop-train)                         | hours per task
  export EDDIES_MAXTASKS=0
  for t in "${TOKPROP[@]}"; do
    local c=(-m eddies.cert.collect --workers "$WORKERS" --tag tokprop --tasks "$t")
    py "${c[@]}" --seeds 1000-1399 --conditions perturbed
    py "${c[@]}" --seeds 2000-2099 --conditions nominal,perturbed
  done
}
collect_tokprop_learned() {  # learned-proposer rollouts (after tokprop-train, before tokprop-eval) | hours per task
  export EDDIES_MAXTASKS=0
  for t in "${TOKPROP[@]}"; do
    for m in bistl_learned bistl_learned_raw bistl_learned_state bistl_learned_state_raw; do
      py -m eddies.cert.collect --workers "$WORKERS" --tag tokprop --tasks "$t" --method "$m" --seeds 2000-2099 --conditions nominal,perturbed
    done
  done
  for m in bistl_cmd bistl_learned_cmd; do
    py -m eddies.cert.collect --workers "$WORKERS" --tag tokprop --tasks franka_cube_stack --method "$m" --seeds 2000-2099 --conditions nominal,perturbed
  done
}
collect_isaac() {    # Isaac Sim (PhysX) sb + stacking gate episodes; needs ISAAC_PY and a GPU   | GPU, hours
  export EDDIES_EPISODES="${EDDIES_RESULTS_ISAAC:-$ROOT/datasets/episodes_isaac}"
  for t in franka_fmb franka_peg_insertion franka_cube_stack; do
    for s in 1000-1049 1200-1249 2000-2049; do
      isaac_py -m eddies.isaac.run_cert_task --task "$t" --seeds "$s" --conditions nominal --tag sb
    done
  done
  for r in 1.5 2 3 4; do
    isaac_py -m eddies.isaac.run_cert_task --task franka_cube_stack --seeds 2000-2049 --conditions nominal --tag gate \
      --rho-scale "$r" --label "bistl_rho$r"
  done
}

# name | function | episodes needed | results written
EXPERIMENTS=(
  "certify|certify|sb|embodiments/embodiments.json"
  "adaptive-groups|adaptive_groups|sb|adaptive/adaptive_<task>_known_<robot>.json"
  "adaptive|adaptive|sb (Franka)|adaptive/adaptive_<task>[_known].json"
  "gate-margin|gate_margin|sb|gate_margin/summary.json (+ per-step <task>.npz, <task>_a.json)"
  "table1|table1|sb (Franka)|table1/*.json"
  "table1-robots|table1_robots|sb (G1, LEAP, Aero)|table1_robots/<robot>/*.json"
  "containment|containment|sb (Franka)|gate/containment.json"
  "pool|pool|sb (Franka)|pool/*.json"
  "transfer|transfer|sb (Franka)|transfer/*.json"
  "trace|trace|sb (Franka)|figs_data/trace_fmb.json"
  "facts|facts|sb|eqoff/eqoff.json, facts/gain_all.json, facts/shrink_probe.json"
  "fixed|fixed|sb + gate (Franka)|gate/gate_eddy{,_recal}.json"
  "ltl|ltl|sb + ltl|gate/ltl.json, ltl/compose_check_2000.json"
  "physics|physics|sb + isaac|physics/physics.json"
  "dynmap|dynmap|sb + isaac|dynamics_map/*.json"
  "stack|stack|sb + gate + isaac|compare/stack_*.json"
  "tokprop|tokprop|sb + tokprop|tokprop/*.json (= tokprop-train, tokprop-eval)"
  "tokprop-train|tokprop_train|sb + tokprop (supervisor)|tokprop/*_train.json, proposer models"
  "tokprop-eval|tokprop_eval|sb + tokprop|tokprop/eval_*.json"
  "collect-sb|collect_sb|-|episodes sb"
  "collect-gate|collect_gate|-|episodes gate"
  "collect-ltl|collect_ltl|-|episodes ltl"
  "collect-tokprop|collect_tokprop|-|episodes tokprop (supervisor)"
  "collect-tokprop-learned|collect_tokprop_learned|tokprop models|episodes tokprop (learned proposers)"
  "collect-isaac|collect_isaac|- (Isaac Sim, GPU)|isaac episodes sb + gate"
)
# `all`: the analyses, those on sb alone first; the collect-* steps are left out (download the dataset instead)
ALL=(certify adaptive-groups adaptive gate-margin table1 table1-robots containment pool transfer trace facts
     fixed ltl physics dynmap stack tokprop)

usage() { sed -n '2,15p' "$0"; }
list() {
  printf '%-24s %-28s %s\n' experiment "needs episodes" writes
  for e in "${EXPERIMENTS[@]}"; do IFS='|' read -r n _ need out <<< "$e"; printf '%-24s %-28s %s\n' "$n" "$need" "$out"; done
  echo; echo "all = ${ALL[*]}  (then python -m eddies.evaluate all)"
}
fn_of() { for e in "${EXPERIMENTS[@]}"; do IFS='|' read -r n f _ _ <<< "$e"; [[ "$n" == "$1" ]] && { echo "$f"; return; }; done; }

STEPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --workers=*) WORKERS="${1#--workers=}"; shift ;;
    --dry-run) DRY=1; shift ;;
    --list) list; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    all) STEPS+=("${ALL[@]}"); STEPS+=(evaluate); shift ;;
    *) STEPS+=("$1"); shift ;;
  esac
done
[[ ${#STEPS[@]} -gt 0 ]] || { usage; echo; list; exit 2; }
for s in "${STEPS[@]}"; do
  [[ "$s" == evaluate || -n "$(fn_of "$s")" ]] || { echo "unknown experiment: $s (scripts/reproduce.sh --list)" >&2; exit 2; }
done

echo "results: $RES"
echo "episodes: ${EDDIES_EPISODES:-datasets/episodes (default)}   isaac: ${EDDIES_RESULTS_ISAAC:-datasets/episodes_isaac (default)}"
for s in "${STEPS[@]}"; do
  echo "=== $s"
  if [[ "$s" == evaluate ]]; then py -m eddies.evaluate all; else "$(fn_of "$s")"; fi
done
