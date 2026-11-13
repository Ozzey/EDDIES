#!/usr/bin/env bash
# Download the Eddies episode logs (Hugging Face dataset Ozzey/eddies) into datasets/episodes/.
#
#   datasets/download.sh                      # everything (~2.4 GB)
#   datasets/download.sh --subset sb          # one or more subsets (repeatable, comma-separated)
#   datasets/download.sh --subset franka,isaac --dry-run
#
# Subsets (by tag or robot, see datasets/README.md):
#   sb        MuJoCo skillbook episodes of the ten tasks (train/calibration/test)      1.0 GB
#   gate      MuJoCo Franka tasks commanded 1.5-4x faster (Tab. D.5, Tab. 3)          147 MB
#   ltl       LTLf composition runs, nominal + perturbed (Sec. 5.4, App. A)            288 MB
#   tokprop   learned Eddie proposer episodes (App. D)                                 914 MB
#   isaac     Isaac Sim (PhysX) episodes: sb + gate (Tab. 2, Tab. 3, Fig. D.1)          62 MB
#   franka | g1 | leap | aero   the MuJoCo sb episodes of that robot only
#   gate_margin   derived per-step margin arrays, also copied to results/gate_margin/    68 MB
#   stills    rendered stills behind Fig. 6 and Fig. C.1                                15 MB
#
# Afterwards scripts/reproduce.sh finds the episodes under datasets/episodes/{mujoco,isaac}; when calling the
# modules directly: export EDDIES_EPISODES=datasets/episodes/mujoco EDDIES_RESULTS_ISAAC=datasets/episodes/isaac
#
# Requires the Hugging Face CLI: curl -LsSf https://hf.co/cli/install.sh | bash -s   (or pip install -U huggingface_hub)
set -euo pipefail
REPO="${EDDIES_DATASET:-Ozzey/eddies}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${EDDIES_EPISODES_DIR:-$HERE/episodes}"
SUBSETS=""
DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset) SUBSETS="$SUBSETS,$2"; shift 2 ;;
    --subset=*) SUBSETS="$SUBSETS,${1#--subset=}"; shift ;;
    --dest) DEST="$2"; shift 2 ;;
    --dry-run) DRY="--dry-run"; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
command -v hf >/dev/null || { echo "hf CLI not found: curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2; exit 1; }

INCLUDES=()
add() { INCLUDES+=(--include "$1"); }
if [[ -z "$SUBSETS" ]]; then
  add "*"
else
  for s in $(echo "$SUBSETS" | tr ',' ' '); do
    case "$s" in
      sb)        add "mujoco/episodes/sb/*" ;;
      gate)      add "mujoco/episodes/gate/*" ;;
      ltl)       add "mujoco/episodes/ltl/*" ;;
      tokprop)   add "mujoco/episodes/tokprop/*" ;;
      mujoco)    add "mujoco/*" ;;
      isaac)     add "isaac/*" ;;
      franka)    add "mujoco/episodes/sb/nominal/franka_*/*" ;;
      g1)        add "mujoco/episodes/sb/nominal/g1_*/*" ;;
      leap)      add "mujoco/episodes/sb/nominal/leap_*/*" ;;
      aero)      add "mujoco/episodes/sb/nominal/aero_*/*" ;;
      gate_margin) add "derived/gate_margin/*" ;;
      stills)    add "stills/*" ;;
      *) echo "unknown subset: $s" >&2; exit 2 ;;
    esac
  done
fi
add "manifest.json"; add "README.md"; add "checksums.sha256"; add "schema_samples.json"

mkdir -p "$DEST"
echo "hf download $REPO --type dataset --local-dir $DEST ${INCLUDES[*]} $DRY"
hf download "$REPO" --type dataset --local-dir "$DEST" "${INCLUDES[@]}" $DRY
echo "done: $DEST  (layout: <simulator>/episodes/<tag>/<condition>/<task>/<method>/seed_XXXX/{manifest.json,trajectory.npz})"
# the per-step gate-margin logs go next to their summary, where eddies.cert.tables.gate_margin reads them
if [[ -z "$DRY" && -d "$DEST/derived/gate_margin" ]]; then
  RES="${EDDIES_RESULTS:-$HERE/../results}"
  mkdir -p "$RES/gate_margin"
  cp "$DEST"/derived/gate_margin/* "$RES/gate_margin/"
  echo "copied derived/gate_margin -> $RES/gate_margin/"
fi
