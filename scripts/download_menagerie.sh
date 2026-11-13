#!/usr/bin/env bash
# Fetch the five mujoco_menagerie robots the ten tasks use, at the commit pinned in eddies/config.yaml
# (paths.menagerie_commit), into external/mujoco_menagerie (gitignored, ~130 MB).  Needs git >= 2.25.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${EDDIES_MENAGERIE:-$HERE/external/mujoco_menagerie}"
COMMIT="$(sed -n 's/^ *menagerie_commit: *//p' "$HERE/eddies/config.yaml" | tr -d '[:space:]')"
DIRS=(franka_emika_panda unitree_g1 universal_robots_ur5e leap_hand tetheria_aero_hand_open)
if [[ -e "$DEST/.git" ]] && git -C "$DEST" rev-parse --verify -q "$COMMIT^{commit}" >/dev/null; then
  git -C "$DEST" checkout -q "$COMMIT"
else
  rm -rf "$DEST"; mkdir -p "$(dirname "$DEST")"
  git clone --quiet --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
  git -C "$DEST" sparse-checkout set "${DIRS[@]}"
  git -C "$DEST" checkout -q "$COMMIT"
fi
for d in "${DIRS[@]}"; do [[ -d "$DEST/$d" ]] || { echo "missing $DEST/$d" >&2; exit 1; }; done
echo "mujoco_menagerie @ $(git -C "$DEST" rev-parse --short HEAD) in $DEST: ${DIRS[*]}"
