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

# Generate operator-ready BNSD tensors; each msprof op capture sees one launch.
# Default: both paths, twenty independent captures per path.
#   fp16 — tiling key 0, kernel out FP32
#   native-bf16 — tiling key 1, kernel out BF16
# Default case: BNSD Q [1, 24, 5376, 128], K/V [1, 24, 8960, 128].
# Q/K are FP16 or BF16; V is FP16.
#
# Must run inside CANN + torch_npu + NPU.
# Do NOT set ASCEND_RT_VISIBLE_DEVICES (incompatible with msprof / msprof op).
#
# Usage:
#   bash tests/ops/laser_attention/run_laser_attention_msprof_op.sh
#   bash tests/ops/laser_attention/run_laser_attention_msprof_op.sh --device-id 3
#   bash tests/ops/laser_attention/run_laser_attention_msprof_op.sh --path fp16
#   bash tests/ops/laser_attention/run_laser_attention_msprof_op.sh --path native-bf16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/profile_laser_attention_prod.py"
SELECT_SCRIPT="${REPO_ROOT}/tests/tools/select_npu_device.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/msprof_laser_attention_${STAMP}}"
DEVICE_ID="${DEVICE_ID:-}"
KERNEL_NAME="${KERNEL_NAME:-LaserAttention}"
MSPROF_WARMUP="${MSPROF_WARMUP:-100}"
MSPROF_LAUNCH_COUNT="${MSPROF_LAUNCH_COUNT:-1}"
TEST_COUNT=20
PROFILE_PATH="${PROFILE_PATH:-both}"

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra python args]

Options:
  --output-dir <dir>     Parent output directory (absolute path recommended).
                         Default run writes <dir>/fp16 and <dir>/native_bf16.
  --device-id <id>       Physical NPU ID from npu-smi info. Default: auto-pick idle card.
  --path <name>          both (default) | fp16 | native-bf16; cast-fp16 is an alias for fp16.
  --native-bf16          Same as --path native-bf16 (twenty captures).
  -h, --help             Show this help.

Environment:
  ASCEND_ENV_SH          Explicit Ascend set_env.sh path.
  ASCEND_TOOLKIT_HOME    Ascend toolkit root, used to locate set_env.sh.
  DEVICE_ID              Same as --device-id if the flag is omitted.
  OUTPUT_DIR / KERNEL_NAME / MSPROF_WARMUP / MSPROF_LAUNCH_COUNT

Each path runs 20 independent captures on the selected device (warm-up default: 100).
Each run_NN directory keeps msprof.log; task_duration_summary.txt lists all results.

KERNEL_NAME default is LaserAttention (OpDef / aclnn prefix). If msprof reports
no matching data, retry with KERNEL_NAME=laser_attention.

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
        --path)
            PROFILE_PATH="$2"
            shift 2
            ;;
        --native-bf16)
            PROFILE_PATH="native-bf16"
            shift
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

source_ascend_env

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

echo "=== LaserAttention msprof op (BNSD; actual shape/dtype are printed by Python) ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "KERNEL_NAME=${KERNEL_NAME}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES-}"
echo "device_id=${DEVICE_ID}"

case "${PROFILE_PATH}" in
    both)
        PATHS=(fp16 native-bf16)
        ;;
    cast-fp16)
        PATHS=(fp16)
        ;;
    fp16|native-bf16)
        PATHS=("${PROFILE_PATH}")
        ;;
    *)
        echo "ERROR: --path must be both, fp16, or native-bf16 (got ${PROFILE_PATH})" >&2
        exit 1
        ;;
esac

FILTERED_PY_ARGS=()
if [[ ${#EXTRA_PY_ARGS[@]} -gt 0 ]]; then
    for arg in "${EXTRA_PY_ARGS[@]}"; do
        if [[ "${arg}" == "--native-bf16" ]]; then
            continue
        fi
        FILTERED_PY_ARGS+=("${arg}")
    done
fi

run_one_capture() {
    local path_name="$1"
    local out_dir="$2"
    mkdir -p "${out_dir}"
    local py_cmd=(
        python3 "${BENCH_SCRIPT}"
        --device-id "${DEVICE_ID}"
        --msprof-mode
    )
    if [[ "${path_name}" == "native-bf16" ]]; then
        py_cmd+=(--native-bf16)
    fi
    if [[ ${#FILTERED_PY_ARGS[@]} -gt 0 ]]; then
        py_cmd+=("${FILTERED_PY_ARGS[@]}")
    fi
    echo "=== capturing path=${path_name} output=${out_dir} ==="
    msprof op \
        --kernel-name="${KERNEL_NAME}" \
        --warm-up="${MSPROF_WARMUP}" \
        --launch-count="${MSPROF_LAUNCH_COUNT}" \
        --kill=on \
        --output="${out_dir}" \
        "${py_cmd[@]}" 2>&1 | tee "${out_dir}/msprof.log"
}

read_task_duration() {
    awk '
        {
            gsub(/\033\[[0-9;]*m/, "")
            sub(/\r$/, "")
        }
        /Task[[:space:]]+Duration[[:space:]]*\(us\)[[:space:]]*:/ {
            count++
            sub(/^.*Task[[:space:]]+Duration[[:space:]]*\(us\)[[:space:]]*:[[:space:]]*/, "")
            sub(/[[:space:]]*$/, "")
            duration = $0
        }
        END {
            if (count != 1 || duration !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ || duration + 0 <= 0) {
                exit 1
            }
            print duration
        }
    ' "$1"
}

CAPTURED_DIRS=()
DURATION_ROWS=()
for path_name in "${PATHS[@]}"; do
    case "${path_name}" in
        fp16) sub_dir="fp16" ;;
        native-bf16) sub_dir="native_bf16" ;;
        *)
            echo "ERROR: unknown path ${path_name}" >&2
            exit 1
            ;;
    esac
    if [[ ${#PATHS[@]} -eq 1 ]]; then
        out_dir="${OUTPUT_DIR}"
    else
        out_dir="${OUTPUT_DIR}/${sub_dir}"
    fi
    for ((run_index = 1; run_index <= TEST_COUNT; run_index++)); do
        printf -v run_dir '%s/run_%02d' "${out_dir}" "${run_index}"
        echo "=== ${path_name}: test ${run_index}/${TEST_COUNT} ==="
        run_one_capture "${path_name}" "${run_dir}"
        if ! duration="$(read_task_duration "${run_dir}/msprof.log")"; then
            echo "ERROR: expected one numeric Task Duration(us); check ${run_dir}/msprof.log" >&2
            exit 1
        fi
        printf -v duration_row '%-14s %4d %20s' "${path_name}" "${run_index}" "${duration}"
        DURATION_ROWS+=("${duration_row}")
        CAPTURED_DIRS+=("${run_dir}")
    done
done

{
    for captured in "${CAPTURED_DIRS[@]}"; do
        echo "${captured}"
    done
} > "${REPO_ROOT}/logs/last_laser_attention_msprof.path"
echo "RESULT captured ${#CAPTURED_DIRS[@]} runs (${TEST_COUNT} per path)"
for captured in "${CAPTURED_DIRS[@]}"; do
    echo "  ${captured}"
done
{
    echo "=== LaserAttention Task Duration summary (device_id=${DEVICE_ID}) ==="
    printf '%-14s %4s %20s\n' 'Path' 'Test' 'Task Duration(us)'
    printf '%s\n' "${DURATION_ROWS[@]}"
    echo "=== Performance results: minimum of ${TEST_COUNT} groups per path (us) ==="
    printf '%s\n' "${DURATION_ROWS[@]}" | awk '
        !($1 in best) { order[++count] = $1; best[$1] = $3 }
        { if ($3 + 0 < best[$1] + 0) best[$1] = $3 }
        END { for (i = 1; i <= count; i++) printf "%-14s %s\n", order[i], best[order[i]] }
    '
    echo "Note: Some hardware models may show increased latency across runs. The minimum across all groups for each scenario is reported as the best performance observed under the current test conditions; this does not mean the variation has been eliminated."
} | tee "${OUTPUT_DIR}/task_duration_summary.txt"
