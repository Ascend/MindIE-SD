#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# Large-shape accuracy and fixed-input stress for rms_norm_indexed_scale_shift.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_ID="${DEVICE_ID:-}"
MODE="all"
SUITE="default"
STRESS_RUNS=1000
PYTHON="${PYTHON:-python3}"
EXTRA_ARGS=()
usage() {
    echo "Usage: $0 [--suite default|production] [--device-id ID] [--mode all|accuracy|stress] [--stress-runs N] [-- SCENE_ARGS]"
    echo "Production suite: PR-listed BF16 shapes/distributions; cannot combine with shape overrides."
    echo "SCENE_ARGS: --rows N --hidden-size N --num-groups N --eps F --seed N (same for performance)."
    echo "Hidden size: 1..8192 (block_n=hidden_size); larger widths are rejected before device selection."
    echo "Default: accuracy for both distributions, then stress (1000 repeats per distribution)."
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)
            [[ $# -ge 2 ]] || { echo "Missing suite" >&2; exit 1; }
            SUITE="$2"; shift 2 ;;
        --device-id)
            [[ $# -ge 2 ]] || { echo "Missing device ID" >&2; exit 1; }
            DEVICE_ID="$2"; shift 2 ;;
        --mode)
            [[ $# -ge 2 ]] || { echo "Missing mode" >&2; exit 1; }
            MODE="$2"; shift 2 ;;
        --stress-runs)
            [[ $# -ge 2 ]] || { echo "Missing stress run count" >&2; exit 1; }
            STRESS_RUNS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) usage >&2; exit 1 ;;
    esac
done
[[ "${MODE}" =~ ^(all|accuracy|stress)$ ]] || { echo "Invalid mode: ${MODE}" >&2; exit 1; }
[[ "${STRESS_RUNS}" =~ ^[1-9][0-9]*$ ]] || { echo "Stress runs must be a positive integer" >&2; exit 1; }
unset ASCEND_RT_VISIBLE_DEVICES
# Validate scene options before selecting a card or importing the DUT.
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i += 2)); do
    case "${EXTRA_ARGS[i]}" in
        --rows|--hidden-size|--num-groups|--eps|--seed) ;;
        *) echo "Invalid scene option: ${EXTRA_ARGS[i]}" >&2; exit 1 ;;
    esac
    ((i + 1 < ${#EXTRA_ARGS[@]})) || { echo "Missing scene value" >&2; exit 1; }
done
SCENES="$("$PYTHON" "${SCRIPT_DIR}/check_index_modulation_prod_accuracy.py" \
    --device-id "${DEVICE_ID:-0}" --mode "$MODE" --stress-runs "$STRESS_RUNS" \
    --suite "$SUITE" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --list-scenes)"
SCENES="${SCENES//$'\r'/}"
if [[ -z "${DEVICE_ID}" ]]; then
    DEVICE_ID="$("$PYTHON" "${SCRIPT_DIR}/../../../tools/select_npu_device.py" --format=id)"
fi
[[ "${DEVICE_ID}" =~ ^[0-9]+$ ]] || { echo "Invalid device ID" >&2; exit 1; }
RESULTS=()
for stage in accuracy stress; do
    [[ "${MODE}" == all || "${MODE}" == "${stage}" ]] || continue
    while read -r rows hidden groups selected_distribution; do
        SCENE="$("$PYTHON" "${SCRIPT_DIR}/check_index_modulation_prod_accuracy.py" \
            --device-id "$DEVICE_ID" --rows "$rows" --hidden-size "$hidden" --num-groups "$groups" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --validate-only)"
        SCENE="${SCENE//$'\r'/}"
        distributions=(uniform normal)
        if [[ "$SUITE" == production ]]; then
            distributions=("$selected_distribution")
        fi
        for distribution in "${distributions[@]}"; do
            "$PYTHON" "${SCRIPT_DIR}/check_index_modulation_prod_accuracy.py" \
                --device-id "${DEVICE_ID}" --mode "${stage}" --distribution "${distribution}" \
                --stress-runs "${STRESS_RUNS}" --rows "$rows" --hidden-size "$hidden" --num-groups "$groups" \
                ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
            if [[ "${stage}" == stress ]]; then
                RESULTS+=("PASS stress ${SCENE} distribution=${distribution} runs=${STRESS_RUNS} output=output bit_exact")
            else
                RESULTS+=("PASS accuracy ${SCENE} distribution=${distribution} output=output (mixed tolerance diagnostic only)")
            fi
        done
    done <<< "$SCENES"
done
echo "=== Selected tests passed ==="
printf '%s\n' "${RESULTS[@]}"
