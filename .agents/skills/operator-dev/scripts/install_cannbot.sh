#!/bin/bash
# Install/verify cannbot-skills (external operator skill library used by operator-dev).
# Usage: bash install_cannbot.sh            # install to $CANNBOT_SKILLS_DIR or ~/.cannbot-skills
#        CANNBOT_SKILLS_DIR=/opt/cannbot-skills bash install_cannbot.sh
#        CANNBOT_UPDATE=1 bash install_cannbot.sh   # git pull when already cloned
set -e

CANNBOT_REMOTE="${CANNBOT_REMOTE:-https://gitcode.com/cann/cannbot-skills}"
TARGET="${CANNBOT_SKILLS_DIR:-$HOME/.cannbot-skills}"
REQUIRED=(
  "AGENTS.md"
  "ops/triton-latency-optimizer/SKILL.md"
  "ops/triton-op-coding/SKILL.md"
  "ops/triton-op-designer/SKILL.md"
  "ops/ascendc-tiling-design/SKILL.md"
)

echo "== cannbot-skills install =="
echo "target : $TARGET"
echo "remote : $CANNBOT_REMOTE"

if [ -d "$TARGET/.git" ]; then
  echo "already cloned."
  if [ "${CANNBOT_UPDATE:-0}" = "1" ]; then
    echo "updating..."
    git -C "$TARGET" pull --ff-only
  fi
else
  mkdir -p "$(dirname "$TARGET")"
  git clone --depth 1 "$CANNBOT_REMOTE" "$TARGET"
fi

missing=0
for f in "${REQUIRED[@]}"; do
  if [ ! -f "$TARGET/$f" ]; then
    echo "MISSING: $TARGET/$f"; missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo "ERROR: cannbot-skills incomplete at $TARGET" >&2
  exit 1
fi

echo "OK: cannbot-skills ready at $TARGET"
echo "hint: export CANNBOT_SKILLS_DIR=$TARGET  # then reuse in operator-dev tasks"
echo "hint: repo bundles its own plugin installers (install.sh / install.ps1) if needed."
