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

# Capture EagleFusedInferAttentionScore with msprof op.
# Default case: DiT-Prof.xlsx / 0825-eaglefia-tiling512 / row 34.
# Runs inner_precise=0 first, then inner_precise=4.
#
# Must run inside CANN + torch_npu + NPU.
# Do NOT set ASCEND_RT_VISIBLE_DEVICES (incompatible with msprof / msprof op).
#
# Usage:
#   bash tests/ops/fused_infer_attention_score/run_fia_msprof_op.sh
#   bash tests/ops/fused_infer_attention_score/run_fia_msprof_op.sh --device-id 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/profile_fia_dit_tiling512.py"
BENCH_SCRIPT_INNER4="${SCRIPT_DIR}/profile_fia_dit_tiling512_inner_precise4.py"
SELECT_SCRIPT="${REPO_ROOT}/tests/tools/select_npu_device.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/msprof_fia_tiling512_${STAMP}}"
DEVICE_ID="${DEVICE_ID:-}"
KERNEL_NAME="${KERNEL_NAME:-EagleFusedInferAttentionScore}"
MSPROF_WARMUP="${MSPROF_WARMUP:-10}"
MSPROF_LAUNCH_COUNT="${MSPROF_LAUNCH_COUNT:-5}"

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra python args]

Options:
  --output-dir <dir>     msprof op output directory (absolute path recommended).
  --device-id <id>       Physical NPU ID from npu-smi info. Default: auto-pick idle card.
  -h, --help             Show this help.

Environment:
  ASCEND_ENV_SH          Explicit Ascend set_env.sh path.
  ASCEND_TOOLKIT_HOME    Ascend toolkit root, used to locate set_env.sh.
  DEVICE_ID              Same as --device-id if the flag is omitted.
  OUTPUT_DIR / KERNEL_NAME / MSPROF_WARMUP / MSPROF_LAUNCH_COUNT

Do not set ASCEND_RT_VISIBLE_DEVICES; this script unsets it.
EOF
}

EXTRA_PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
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
    echo "warning: unsetting ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} (unsupported with msprof / msprof op)"
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

if ! command -v msprof >/dev/null 2>&1; then
    echo "ERROR: msprof not found; source CANN set_env.sh first" >&2
    exit 1
fi

python3 - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("torch_npu") is None:
    sys.exit("ERROR: torch_npu missing — run inside CANN/torch_npu container")
if importlib.util.find_spec("mindiesd") is None:
    sys.exit("ERROR: mindiesd missing — install the current MindIE-SD tree first")
print("python", sys.executable)
PY

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
source_ascend_env

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "warning: unsetting ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} (unsupported with msprof / msprof op)"
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
    python3 "${SELECT_SCRIPT}" --format=report | tee "${OUTPUT_DIR}/npu_select.log"
    DEVICE_ID="$(python3 "${SELECT_SCRIPT}" --format=id)"
    if [[ -z "${DEVICE_ID}" ]]; then
        echo "ERROR: failed to select NPU id" >&2
        exit 1
    fi
else
    echo "=== Using user-specified device_id=${DEVICE_ID} ===" | tee "${OUTPUT_DIR}/npu_select.log"
fi

echo "=== FIA msprof op (DiT eaglefia tiling512 row34, inner_precise=0 baseline) ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "KERNEL_NAME=${KERNEL_NAME}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES-}"
echo "device_id=${DEVICE_ID}"

PY_CMD=(python3 "${BENCH_SCRIPT}" --device-id "${DEVICE_ID}" --msprof-mode)
if [[ ${#EXTRA_PY_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${EXTRA_PY_ARGS[@]}")
fi

msprof op \
  --kernel-name="${KERNEL_NAME}" \
  --warm-up="${MSPROF_WARMUP}" \
  --launch-count="${MSPROF_LAUNCH_COUNT}" \
  --kill=on \
  --output="${OUTPUT_DIR}" \
  "${PY_CMD[@]}"

echo "${OUTPUT_DIR}" > "${REPO_ROOT}/logs/last_fia_msprof.path"

INNER4_OUTPUT_DIR="${OUTPUT_DIR}_inner_precise4"
mkdir -p "${INNER4_OUTPUT_DIR}"
INNER4_OUTPUT_DIR="$(cd "${INNER4_OUTPUT_DIR}" && pwd)"

echo "=== FIA msprof op (DiT eaglefia tiling512 row34, inner_precise=4) ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${INNER4_OUTPUT_DIR}"
echo "KERNEL_NAME=${KERNEL_NAME}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES-}"
echo "device_id=${DEVICE_ID}"

PY_CMD_INNER4=(python3 "${BENCH_SCRIPT_INNER4}" --device-id "${DEVICE_ID}" --msprof-mode)
if [[ ${#EXTRA_PY_ARGS[@]} -gt 0 ]]; then
  PY_CMD_INNER4+=("${EXTRA_PY_ARGS[@]}")
fi

msprof op \
  --kernel-name="${KERNEL_NAME}" \
  --warm-up="${MSPROF_WARMUP}" \
  --launch-count="${MSPROF_LAUNCH_COUNT}" \
  --kill=on \
  --output="${INNER4_OUTPUT_DIR}" \
  "${PY_CMD_INNER4[@]}"

echo "${INNER4_OUTPUT_DIR}" > "${REPO_ROOT}/logs/last_fia_msprof_inner_precise4.path"
