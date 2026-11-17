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
# Single-operator msprof op performance entry for rms_norm_indexed_scale_shift.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE_ID="${DEVICE_ID:-}"
RUNS=20
SEED=20261017
MSPROF_WARMUP="${MSPROF_WARMUP:-20}"
DISTRIBUTION_SELECT=all
SUITE=default
OUTPUT_DIR=""
KERNEL_NAME="_rms_norm_indexed_scale_shift_kernel"
EXTRA_ARGS=()
usage() {
    echo "Usage: $0 [--suite default|production] [--distribution all|uniform|normal] [--device-id N] [--runs N] [--seed N] [--output DIR] [-- SCENE_ARGS]"
    echo "Production suite: 8 BF16 shapes x selected distributions; cannot combine with shape overrides."
    echo "Default scene: x [13640,5376] BF16; weight [5376]; shift/scale [3,5376]; indices [13640] int64; eps=1e-6."
    echo "SCENE_ARGS: --rows N --hidden-size N --num-groups N --eps F (match the accuracy configuration)."
    echo "Hidden size: 1..8192 (block_n=hidden_size); larger widths are rejected before device selection."
    echo "Each selected distribution: msprof op, MSPROF_WARMUP=20, one target call, Task Duration(us), 20 captures."
    echo "Default output: <repo_root>/log/rms_norm_indexed_scale_shift/perf-XXXXXXXX/"
    exit 0
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite|--distribution|--device-id|--runs|--seed|--output)
            [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --suite) SUITE="$2" ;;
                --distribution) DISTRIBUTION_SELECT="$2" ;;
                --device-id) DEVICE_ID="$2" ;;
                --runs) RUNS="$2" ;;
                --seed) SEED="$2" ;;
                --output) OUTPUT_DIR="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$DISTRIBUTION_SELECT" in
    all) SELECTED_DISTS=(uniform normal) ;;
    uniform|normal) SELECTED_DISTS=("$DISTRIBUTION_SELECT") ;;
    *) echo "Invalid --distribution: $DISTRIBUTION_SELECT" >&2; exit 2 ;;
esac
[[ "$MSPROF_WARMUP" =~ ^(0|[1-9][0-9]*)$ ]] && ((MSPROF_WARMUP <= 500)) || { echo "Invalid MSPROF_WARMUP (0..500)" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid runs" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed" >&2; exit 2; }
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "Invalid device ID" >&2; exit 2; }
command -v "$PYTHON" >/dev/null
unset ASCEND_RT_VISIBLE_DEVICES
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i += 2)); do
    case "${EXTRA_ARGS[i]}" in
        --rows|--hidden-size|--num-groups|--eps) ;;
        *) echo "Invalid scene option: ${EXTRA_ARGS[i]}" >&2; exit 2 ;;
    esac
    ((i + 1 < ${#EXTRA_ARGS[@]})) || { echo "Missing scene value" >&2; exit 2; }
done
SCENE="$("$PYTHON" "${SCRIPT_DIR}/profile_index_modulation_prod.py" \
    --device-id "${DEVICE_ID:-0}" --seed "$SEED" --suite "$SUITE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --validate-only)"
SCENES="$("$PYTHON" "${SCRIPT_DIR}/profile_index_modulation_prod.py" \
    --device-id "${DEVICE_ID:-0}" --seed "$SEED" --suite "$SUITE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --list-scenes)"
# Native Windows Python writes CRLF even when invoked from Git Bash.
SCENE="${SCENE//$'\r'/}"
SCENES="${SCENES//$'\r'/}"
command -v msprof >/dev/null
if [[ -z "$OUTPUT_DIR" ]]; then
    mkdir -p "${REPO_ROOT}/log/rms_norm_indexed_scale_shift"
    OUTPUT_DIR="$(mktemp -d "${REPO_ROOT}/log/rms_norm_indexed_scale_shift/perf-XXXXXXXX")"
else
    mkdir -p "$OUTPUT_DIR"
    [[ -z "$(ls -A -- "$OUTPUT_DIR")" ]] || { echo "Output directory must be empty" >&2; exit 2; }
fi
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
if [[ -z "$DEVICE_ID" ]]; then
    "$PYTHON" "${SCRIPT_DIR}/../../../tools/select_npu_device.py" --format=report | tee "${OUTPUT_DIR}/npu_select.log"
    DEVICE_ID="$(sed -n 's/^Selected NPU: //p' "${OUTPUT_DIR}/npu_select.log")"
fi
[[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "No valid physical device selected" >&2; exit 1; }
echo "NPU=$DEVICE_ID warmup=$MSPROF_WARMUP (default 20); captures=$RUNS output=$OUTPUT_DIR kernel=$KERNEL_NAME"
if [[ "$SUITE" == production ]]; then
    REPORTS=()
    while read -r rows hidden groups; do
        scene_dir="${OUTPUT_DIR}/n${rows}_h${hidden}_k${groups}"
        MSPROF_WARMUP="$MSPROF_WARMUP" "$BASH" "${SCRIPT_DIR}/run_index_modulation_perf.sh" \
            --device-id "$DEVICE_ID" --runs "$RUNS" --seed "$SEED" --distribution "$DISTRIBUTION_SELECT" \
            --output "$scene_dir" -- --rows "$rows" --hidden-size "$hidden" --num-groups "$groups" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
        REPORTS+=("${scene_dir}/final_summary.txt")
    done <<< "$SCENES"
    cat "${REPORTS[@]}" > "${OUTPUT_DIR}/final_summary.txt"
    echo "Production suite complete; combined report: ${OUTPUT_DIR}/final_summary.txt"
    exit 0
fi
for distribution in "${SELECTED_DISTS[@]}"; do
    mkdir -p "${OUTPUT_DIR}/${distribution}"
    for ((run = 1; run <= RUNS; run++)); do
        printf -v run_dir '%s/%s/run_%02d' "${OUTPUT_DIR}" "${distribution}" "${run}"
        mkdir -p "${run_dir}"
        echo "distribution=${distribution} Test=${run}/${RUNS}"
        msprof op --kernel-name="${KERNEL_NAME}" --warm-up="${MSPROF_WARMUP}" \
            --launch-count=1 --kill=on --output="${run_dir}" \
            "$PYTHON" "${SCRIPT_DIR}/profile_index_modulation_prod.py" \
            --device-id "$DEVICE_ID" --seed "$SEED" --distribution "$distribution" \
            --msprof-mode ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "${run_dir}/msprof.log"
        duration="$(awk '
            { gsub(/\033\[[0-9;]*m/, ""); sub(/\r$/, "") }
            /Task[[:space:]]+Duration[[:space:]]*\(us\)[[:space:]]*:/ {
                count++; sub(/^.*Task[[:space:]]+Duration[[:space:]]*\(us\)[[:space:]]*:[[:space:]]*/, "")
                sub(/[[:space:]]*$/, ""); value=$0
            }
            END {
                if (count != 1 || value !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ || value+0 <= 0) exit 1
                print value
            }' "${run_dir}/msprof.log")" || { echo "Expected one positive Task Duration(us) in $run_dir" >&2; exit 1; }
        printf '%s\n' "$duration" > "${run_dir}/task_duration_us.txt"
        printf '%s\n' "$duration" >> "${OUTPUT_DIR}/${distribution}/durations.txt"
    done
    awk -v warmup="$MSPROF_WARMUP" -v dist="$distribution" -v scene="$SCENE" '
        BEGIN { print scene "/" dist ": msprof op Task Duration(us)"; print "Test Task Duration(us)" }
        NR == 1 { minimum=$1; maximum=$1 }
        { print NR, $1; if ($1 < minimum) minimum=$1; if ($1 > maximum) maximum=$1 }
        END {
            printf "Warmup=%s (default 20); captures=%d (default 20)\n", warmup, NR
            printf "Min Task Duration(us): %s; Max Task Duration(us): %s\n", minimum, maximum
            print "NOTICE: if spread exceeds 10%, consider more warmups and a fresh complete set (default 20) in a new directory on the same device/inputs; retain warnings if variation persists."
            if (NR < 2) {
                print "WARNING: cannot assess spread with fewer than two samples."
            } else {
                spread=(maximum-minimum)/minimum*100
                printf "Spread ((max-min)/min): %.2f%%\n", spread
                if (spread > 10) print "WARNING: spread exceeds 10%. Frequency ramp-up is only a possible cause."
            }
            print "Note: Some hardware models may show increased latency across runs. The minimum across all groups for each scenario is reported as the best performance observed under the current test conditions; this does not mean the variation has been eliminated."
        }' "${OUTPUT_DIR}/${distribution}/durations.txt" > "${OUTPUT_DIR}/${distribution}/task_duration_summary.txt"
done
# Pad each column to a shared tab stop (standard eight-column terminal tabs).
format_table() {
    awk -F '\t' '
        { rows[NR]=$0; if (NR==1) columns=NF
          for (i=1; i<columns; i++) if (length($i)>width[i]) width[i]=length($i) }
        END {
            for (i=1; i<columns; i++) width[i]=int(width[i]/8)*8+7
            for (row=1; row<=NR; row++) {
                split(rows[row], cells, "\t")
                for (i=1; i<columns; i++) printf "%-*s\t", width[i], cells[i]
                print cells[columns]
            }
        }'
}
{
    printf 'Scenario\tTest\tMetric\tDuration(us)\tDataDir\n'
    for distribution in "${SELECTED_DISTS[@]}"; do
        for ((run = 1; run <= RUNS; run++)); do
            printf -v run_dir '%s/%s/run_%02d' "${OUTPUT_DIR}" "${distribution}" "${run}"
            duration="$(cat "${run_dir}/task_duration_us.txt")"
            printf '%s/%s\t%d\tTask Duration\t%s\t%s\n' "$SCENE" "$distribution" "$run" "$duration" "$run_dir"
        done
    done
} | format_table > "${OUTPUT_DIR}/capture_details.txt"
{
    echo "=== Final performance summary ==="
    cat "${OUTPUT_DIR}/capture_details.txt"
    echo "=== Final performance overview (us; minimum per scenario) ==="
    {
        printf 'Scenario\tMetric\tMin Duration(us)\n'
        for distribution in "${SELECTED_DISTS[@]}"; do
            awk -v dist="$distribution" -v scene="$SCENE" '/^Min Task Duration\(us\):/ {
                sub(/;$/, "", $4)
                printf "%s/%s\tTask Duration(us)\t%s\n", scene, dist, $4
            }' "${OUTPUT_DIR}/${distribution}/task_duration_summary.txt"
        done
    } | format_table
    echo "=== Measurement notes ==="
    echo "NPU=$DEVICE_ID; captures=$RUNS; warmup=$MSPROF_WARMUP; kernel=$KERNEL_NAME"
    for distribution in "${SELECTED_DISTS[@]}"; do
        echo "rms_norm_indexed_scale_shift/$distribution:"
        awk '/^Warmup=/ { notes=1 } notes' "${OUTPUT_DIR}/${distribution}/task_duration_summary.txt"
    done
    echo "The minimum is the best observed performance under the current conditions, not evidence that variation has disappeared."
    echo "Results: $OUTPUT_DIR; combined report: $OUTPUT_DIR/final_summary.txt"
} | tee "${OUTPUT_DIR}/final_summary.txt"
