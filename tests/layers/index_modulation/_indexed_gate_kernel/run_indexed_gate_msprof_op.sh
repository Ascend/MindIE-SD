#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset ASCEND_RT_VISIBLE_DEVICES
ENTRY="${SCRIPT_DIR}/performance.py"
HAS_DEVICE=0
for ARG in "$@"; do
  case "$ARG" in
    --device-id|--device-id=*) HAS_DEVICE=1 ;;
    --help|-h) exec python3 "$ENTRY" "$@" ;;
  esac
done
python3 "$ENTRY" "$@" --validate-only
ARGS=("$@")
if [[ "$HAS_DEVICE" == 0 ]]; then
  DEVICE_ID="$(python3 "${SCRIPT_DIR}/indexed_gate_common.py")"
  ARGS+=(--device-id "$DEVICE_ID")
fi
exec python3 -u "$ENTRY" "${ARGS[@]}"
