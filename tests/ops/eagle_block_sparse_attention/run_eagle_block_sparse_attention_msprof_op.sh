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

# Capture CANN-bridged BlockSparseAttention first, then EagleBlockSparseAttention.
# Default case: Q [1, 32, 32768, 128], KV [1, 4, 32768, 128], FP16 then BF16.
#
# Must run inside CANN + torch_npu + NPU.
# Do NOT set ASCEND_RT_VISIBLE_DEVICES (incompatible with msprof / msprof op).
#
# Usage:
#   bash tests/ops/eagle_block_sparse_attention/run_eagle_block_sparse_attention_msprof_op.sh
#   bash tests/ops/eagle_block_sparse_attention/run_eagle_block_sparse_attention_msprof_op.sh --device-id 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/profile_eagle_block_sparse_attention_prod.py"
SELECT_SCRIPT="${REPO_ROOT}/tests/tools/select_npu_device.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/msprof_eagle_block_sparse_attention_${STAMP}}"
DEVICE_ID="${DEVICE_ID:-}"
CANN_BSA_KERNEL_NAME="${CANN_BSA_KERNEL_NAME:-BlockSparseAttention}"
EBSA_KERNEL_NAME="${EBSA_KERNEL_NAME:-${KERNEL_NAME:-EagleBlockSparseAttention}}"
MSPROF_WARMUP="${MSPROF_WARMUP:-10}"
MSPROF_LAUNCH_COUNT="${MSPROF_LAUNCH_COUNT:-5}"
IMPLS="${IMPLS:-cann_bsa,eagle_bsa}"
DTYPES="${DTYPES:-float16,bfloat16}"
BLOCK_KVS="${BLOCK_KVS:-128,64}"

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra python args]

Options:
  --output-dir <dir>     msprof op output directory (absolute path recommended).
  --device-id <id>       Physical NPU ID from npu-smi info. Default: auto-pick idle card.
  --impls <csv>          Capture order. Default: cann_bsa,eagle_bsa (baseline first).
  --dtypes <csv>         Compute dtypes. Default: float16,bfloat16.
  --block-kvs <csv>      BlockK values (block_shape[1]). Default: 128,64.
                         cann_bsa only runs BlockK=128 (stock CANN still 128-aligned).
  -h, --help             Show this help.

Environment:
  ASCEND_ENV_SH          Explicit Ascend set_env.sh path.
  ASCEND_TOOLKIT_HOME    Ascend toolkit root, used to locate set_env.sh.
  DEVICE_ID              Same as --device-id if the flag is omitted.
  OUTPUT_DIR / MSPROF_WARMUP / MSPROF_LAUNCH_COUNT
  BLOCK_KVS              Same as --block-kvs if the flag is omitted.
  CANN_BSA_KERNEL_NAME   Default BlockSparseAttention
  EBSA_KERNEL_NAME / KERNEL_NAME   Default EagleBlockSparseAttention

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
        --dtypes)
            DTYPES="$2"
            shift 2
            ;;
        --impls)
            IMPLS="$2"
            shift 2
            ;;
        --block-kvs)
            BLOCK_KVS="$2"
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

echo "=== block_sparse_attention then eagle_block_sparse_attention msprof op (prod Q 32768 / KV 32768) ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CANN_BSA_KERNEL_NAME=${CANN_BSA_KERNEL_NAME}"
echo "EBSA_KERNEL_NAME=${EBSA_KERNEL_NAME}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES-}"
echo "device_id=${DEVICE_ID}"
echo "impls=${IMPLS}"
echo "dtypes=${DTYPES}"
echo "block_kvs=${BLOCK_KVS}"

IFS=',' read -r -a IMPL_LIST <<< "${IMPLS}"
IFS=',' read -r -a DTYPE_LIST <<< "${DTYPES}"
IFS=',' read -r -a BLOCK_KV_LIST <<< "${BLOCK_KVS}"
for impl_name in "${IMPL_LIST[@]}"; do
    impl_name="${impl_name//[[:space:]]/}"
    if [[ -z "${impl_name}" ]]; then
        continue
    fi
    case "${impl_name}" in
        cann_bsa)
            KERNEL_THIS="${CANN_BSA_KERNEL_NAME}"
            ;;
        eagle_bsa)
            KERNEL_THIS="${EBSA_KERNEL_NAME}"
            ;;
        *)
            echo "ERROR: unknown impl ${impl_name} (use cann_bsa or eagle_bsa)" >&2
            exit 1
            ;;
    esac
    for dtype_name in "${DTYPE_LIST[@]}"; do
        dtype_name="${dtype_name//[[:space:]]/}"
        if [[ -z "${dtype_name}" ]]; then
            continue
        fi
        for block_kv in "${BLOCK_KV_LIST[@]}"; do
            block_kv="${block_kv//[[:space:]]/}"
            if [[ -z "${block_kv}" ]]; then
                continue
            fi
            if [[ "${impl_name}" == "cann_bsa" && "${block_kv}" != "128" ]]; then
                echo "=== skip impl=${impl_name} dtype=${dtype_name} block_kv=${block_kv} (CANN BSA BlockK stays 128) ==="
                continue
            fi
            IMPL_OUTPUT_DIR="${OUTPUT_DIR}/${impl_name}/${dtype_name}/k${block_kv}"
            mkdir -p "${IMPL_OUTPUT_DIR}"
            IMPL_OUTPUT_DIR="$(cd "${IMPL_OUTPUT_DIR}" && pwd)"
            echo "=== impl=${impl_name} dtype=${dtype_name} block_kv=${block_kv} kernel=${KERNEL_THIS} output=${IMPL_OUTPUT_DIR} ==="

            PY_CMD=(
                python3 "${BENCH_SCRIPT}"
                --device-id "${DEVICE_ID}"
                --impl "${impl_name}"
                --dtype "${dtype_name}"
                --block-kv "${block_kv}"
                --msprof-mode
            )
            if [[ ${#EXTRA_PY_ARGS[@]} -gt 0 ]]; then
                PY_CMD+=("${EXTRA_PY_ARGS[@]}")
            fi

            msprof op \
                --kernel-name="${KERNEL_THIS}" \
                --warm-up="${MSPROF_WARMUP}" \
                --launch-count="${MSPROF_LAUNCH_COUNT}" \
                --kill=on \
                --output="${IMPL_OUTPUT_DIR}" \
                "${PY_CMD[@]}"
        done
    done
done

echo "${OUTPUT_DIR}" > "${REPO_ROOT}/logs/last_eagle_block_sparse_attention_msprof.path"
echo "RESULT captured; cann_bsa Duration is the performance baseline"
