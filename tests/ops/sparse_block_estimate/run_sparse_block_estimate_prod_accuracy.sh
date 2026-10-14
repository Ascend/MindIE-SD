#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

# Large-shape accuracy. Golden is NPU matmul + softmax + block sum.
# Q [1,32,2304,128], K [1,4,30757,128], FP16 and BF16.
# Do NOT set ASCEND_RT_VISIBLE_DEVICES.
#
# Usage:
#   bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_prod_accuracy.sh
#   bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_prod_accuracy.sh --device-id 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/check_sparse_block_estimate_prod_accuracy.py"
SELECT_SCRIPT="${REPO_ROOT}/tests/tools/select_npu_device.py"
DEVICE_ID="${DEVICE_ID:-}"

usage() {
    cat <<EOF
Usage: $0 [--device-id <id>] [-- extra python args]

Omit --device-id to auto-select an idle card via tests/tools/select_npu_device.py.
Do not set ASCEND_RT_VISIBLE_DEVICES.
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

if [[ -z "${DEVICE_ID}" ]]; then
    [[ -f "${SELECT_SCRIPT}" ]] || { echo "ERROR: missing ${SELECT_SCRIPT}" >&2; exit 1; }
    python3 "${SELECT_SCRIPT}" --format=report
    DEVICE_ID="$(python3 "${SELECT_SCRIPT}" --format=id)"
    [[ "${DEVICE_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: failed to select an NPU id" >&2; exit 1; }
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python3 "${BENCH_SCRIPT}" --preset target --device-id "${DEVICE_ID}" --dtype both "${EXTRA_PY_ARGS[@]}"
