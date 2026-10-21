#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2. See LICENSE for details.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEVICE_ID="${DEVICE_ID:-}"
ARGS=()
while (($#)); do
    case "$1" in
        --device-id)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --device-id needs a nonnegative ID" >&2; exit 1; }
            DEVICE_ID="$2"; shift 2 ;;
        --device-id=*)
            DEVICE_ID="${1#*=}"
            [[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: --device-id needs a nonnegative ID" >&2; exit 1; }
            shift ;;
        -h|--help)
            echo "Default: BNSD and BSND, each with uniform and normal; all accuracy cases, then 1000 repeats per case."
            echo "Use --layout BNSD or --layout BSND to select one layout; default --layout all."
            echo "Device: --device-id ID or automatic selection via tests/tools/select_npu_device.py."
            echo "Default outputs: repository log/quant_flash_attn/accuracy_<timestamp>."
            python3 "${SCRIPT_DIR}/check_qfa_prod_accuracy.py" --help
            exit 0 ;;
        --validate-only) echo "ERROR: --validate-only is a Python-only option" >&2; exit 1 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid DEVICE_ID" >&2; exit 1; }
# Reject invalid mode, runs and scene options before selecting a card.
python3 "${SCRIPT_DIR}/check_qfa_prod_accuracy.py" "${ARGS[@]}" --validate-only
unset ASCEND_RT_VISIBLE_DEVICES
if [[ -z "$DEVICE_ID" ]]; then
    DEVICE_ID="$(python3 "${REPO_ROOT}/tests/tools/select_npu_device.py" --format=id)"
fi
[[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid selected NPU ID" >&2; exit 1; }
python3 "${SCRIPT_DIR}/check_qfa_prod_accuracy.py" --device-id "$DEVICE_ID" "${ARGS[@]}"
