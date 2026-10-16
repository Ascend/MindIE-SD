#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

# Large-shape performance. Q stays [1,32,2304,128]. K sequence lengths are the
# production length 30757, 64K (65536) and 416K (425984). Other dimensions stay
# at the production preset. Six runs, all BF16: three small-op concatenations,
# then three fused estimate kernels under msprof op.
# Do NOT set ASCEND_RT_VISIBLE_DEVICES.
#
# Usage:
#   bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_msprof_op.sh
#   bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_msprof_op.sh --device-id 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_sbe_performance.py"
BENCH_SCRIPT="${SCRIPT_DIR}/profile_sparse_block_estimate_a5.py"
SELECT_SCRIPT="${REPO_ROOT}/tests/tools/select_npu_device.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/msprof_sparse_block_estimate_${STAMP}}"
DEVICE_ID="${DEVICE_ID:-}"
KERNEL_NAME="${KERNEL_NAME:-SparseBlockEstimate}"
MSPROF_WARMUP="${MSPROF_WARMUP:-10}"
MSPROF_LAUNCH_COUNT="${MSPROF_LAUNCH_COUNT:-5}"
DTYPE="${DTYPE:-bf16}"
KV_SEQLENS=(30757 65536 425984)

usage() {
    cat <<EOF
Usage: $0 [--device-id <id>] [--output-dir <dir>] [--dtype bf16|fp16]

Six runs: concatenation at K=30757, 65536, 425984, then estimate at the same lengths.
Omit --device-id to auto-select an idle card. Do not set ASCEND_RT_VISIBLE_DEVICES.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${DTYPE}" != "bf16" && "${DTYPE}" != "fp16" ]]; then
    echo "ERROR: --dtype must be bf16 or fp16" >&2
    exit 1
fi

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "warning: unsetting ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} (unsupported by msprof op)"
    unset ASCEND_RT_VISIBLE_DEVICES
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

if [[ -z "${DEVICE_ID}" ]]; then
    [[ -f "${SELECT_SCRIPT}" ]] || { echo "ERROR: missing ${SELECT_SCRIPT}" >&2; exit 1; }
    python3 "${SELECT_SCRIPT}" --format=report | tee "${OUTPUT_DIR}/npu_select.log"
    DEVICE_ID="$(python3 "${SELECT_SCRIPT}" --format=id)"
    [[ "${DEVICE_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: failed to select an NPU id" >&2; exit 1; }
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "RESULT phase=baseline dtype=${DTYPE} device_id=${DEVICE_ID}"
for kv_seq in "${KV_SEQLENS[@]}"; do
    python3 "${BASELINE_SCRIPT}" \
        --device-id "${DEVICE_ID}" \
        --dtype "${DTYPE}" \
        --kv-seq "${kv_seq}" \
        | tee "${OUTPUT_DIR}/baseline_k${kv_seq}.log"
done

for kv_seq in "${KV_SEQLENS[@]}"; do
    PATH_OUTPUT="${OUTPUT_DIR}/estimate_k${kv_seq}"
    mkdir -p "${PATH_OUTPUT}"
    PATH_OUTPUT="$(cd "${PATH_OUTPUT}" && pwd)"
    echo "RESULT phase=estimate dtype=${DTYPE} kv_seq=${kv_seq} output=${PATH_OUTPUT}"
    msprof op \
        --kernel-name="${KERNEL_NAME}" \
        --warm-up="${MSPROF_WARMUP}" \
        --launch-count="${MSPROF_LAUNCH_COUNT}" \
        --kill=on \
        --output="${PATH_OUTPUT}" \
        python3 "${BENCH_SCRIPT}" \
            --device-id "${DEVICE_ID}" \
            --dtype "${DTYPE}" \
            --kv-seq "${kv_seq}" \
            --msprof-mode
done

echo "RESULT sparse_block_estimate six runs captured under ${OUTPUT_DIR}"
