#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# Large-shape FIA accuracy vs unquantized npu_fusion_attention.
# Scene: DiT-Prof.xlsx / 0825-eaglefia-tiling512 / row 34.
# Do NOT set ASCEND_RT_VISIBLE_DEVICES.
#
# Usage:
#   bash tests/ops/fused_infer_attention_score/run_fia_dit_tiling512_accuracy.sh
#   bash tests/ops/fused_infer_attention_score/run_fia_dit_tiling512_accuracy.sh --device-id 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/check_fia_dit_tiling512_accuracy.py"
SELECT_SCRIPT="${REPO_ROOT}/tests/tools/select_npu_device.py"
DEVICE_ID="${DEVICE_ID:-}"

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra python args]

Options:
  --device-id <id>       Physical NPU ID from npu-smi info. Default: auto-pick idle card.
  -h, --help             Show this help.

Environment:
  ASCEND_ENV_SH          Explicit Ascend set_env.sh path.
  ASCEND_TOOLKIT_HOME    Ascend toolkit root, used to locate set_env.sh.
  DEVICE_ID              Same as --device-id if the flag is omitted.

Python defaults: --enhance-mode 2.0, DiT tiling512 row34 shapes.
Do not set ASCEND_RT_VISIBLE_DEVICES; this script unsets it.
EOF
}

EXTRA_PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_PY_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_PY_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "warning: unsetting ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    unset ASCEND_RT_VISIBLE_DEVICES
fi

source_ascend_env() {
    local candidates=()
    if [[ -n "${ASCEND_ENV_SH:-}" ]]; then
        candidates+=("${ASCEND_ENV_SH}")
    fi
    if [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
        candidates+=("${ASCEND_TOOLKIT_HOME}/set_env.sh")
    fi
    candidates+=(
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    )
    local env_file
    for env_file in "${candidates[@]}"; do
        if [[ -f "${env_file}" ]]; then
            # shellcheck disable=SC1090
            source "${env_file}"
            echo "ascend_env=${env_file}"
            return 0
        fi
    done
    echo "warning=no Ascend set_env.sh found; continuing with current environment"
}

python3 - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("torch_npu") is None:
    sys.exit("ERROR: torch_npu missing — run inside CANN/torch_npu container")
if importlib.util.find_spec("mindiesd") is None:
    sys.exit("ERROR: mindiesd missing — install the current MindIE-SD tree first")
print("python", sys.executable)
PY

source_ascend_env

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "warning: unsetting ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    unset ASCEND_RT_VISIBLE_DEVICES
fi

if [[ -z "${DEVICE_ID}" ]]; then
    if [[ ! -f "${SELECT_SCRIPT}" ]]; then
        echo "ERROR: missing ${SELECT_SCRIPT}" >&2
        exit 1
    fi
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "ERROR: npu-smi not found in PATH" >&2
        exit 1
    fi
    echo "=== Selecting idle NPU (npu-smi) ==="
    python3 "${SELECT_SCRIPT}" --format=report
    DEVICE_ID="$(python3 "${SELECT_SCRIPT}" --format=id)"
    if [[ -z "${DEVICE_ID}" ]]; then
        echo "ERROR: failed to select NPU id" >&2
        exit 1
    fi
else
    echo "=== Using user-specified device_id=${DEVICE_ID} ==="
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES-}"
echo "device_id=${DEVICE_ID}"

PY_CMD=(python3 "${BENCH_SCRIPT}" --device-id "${DEVICE_ID}")
if [[ ${#EXTRA_PY_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${EXTRA_PY_ARGS[@]}")
fi
"${PY_CMD[@]}"
