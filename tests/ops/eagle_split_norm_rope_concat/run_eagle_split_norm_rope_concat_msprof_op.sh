#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# Fused: msprof op. Composed: msprof with complete-call NPU Events.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE_ID="${DEVICE_ID:-}"
ITERATIONS=10
RUNS=20
SEED=20261017
MSPROF_WARMUP="${MSPROF_WARMUP:-20}"
OUTPUT_DIR=""
USE_MSPROF=1
PATH_SELECTION=""
ROPE_TYPE=all
CACHE_WIDTH=all
OUTPUT_LAYOUT=all
ROTARY_DIM=all
DRY_RUN=0
SUMMARY_ARGS=()
while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --shape)
            [[ $# -ge 5 && "$2 $3 $4 $5" == '1 5672 168 128' ]] || { echo "Performance supports --shape 1 5672 168 128 only" >&2; exit 2; }
            shift 5 ;;
        --no-msprof) USE_MSPROF=0; shift ;;
        --device-id|--iterations|--runs|--seed|--output|--path|--rope-type|--cache-width|--output-layout|--rotary-dim)
            [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --device-id) DEVICE_ID="$2" ;;
                --iterations) ITERATIONS="$2" ;;
                --runs) RUNS="$2" ;;
                --seed) SEED="$2" ;;
                --output) OUTPUT_DIR="$2" ;;
                --path)
                    [[ -z "$PATH_SELECTION" ]] || { echo "Specify --path only once" >&2; exit 2; }
                    [[ "$2" == fused || "$2" == composed ]] || { echo "Invalid --path: $2; use fused or composed, or omit --path to run both" >&2; exit 2; }
                    PATH_SELECTION="$2" ;;
                --rope-type) ROPE_TYPE="$2" ;;
                --cache-width) CACHE_WIDTH="$2" ;;
                --output-layout) OUTPUT_LAYOUT="$2" ;;
                --rotary-dim) ROTARY_DIM="$2" ;;
            esac
            shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--path fused|composed] [--shape 1 5672 168 128] [--rope-type all|half|interleave] [--cache-width all|full|half] [--output-layout all|bsnd|bnsd] [--rotary-dim all|96|128] [--dry-run] [--device-id N] [--iterations 10] [--runs 20] [--seed N] [--output DIR] [--no-msprof]"
            echo "Omit --path to run both: 8 fused + 8 matching complete-QKV composition scenarios (16 total)."
            echo "Filters select the crossed representative matrix; unmatched combinations fail. --dry-run previews without Python/NPU or files."
            echo "Default output: <repository>/log/eagle_split_norm_rope_concat/perf_YYYYMMDD_HHMMSS/"
            echo "Packed QKV [1,5672,168,128] BF16; heads [56,56,56]; cos/sin [1,5672,1,W], W=R or R/2, R=96/128 BF16; weighted RMSNorm."
            echo "Eight representatives cross RoPE, output layout, cache width and R; no 16-case Cartesian expansion."
            echo "D=128 only; fused computes RMSNorm/RoPE in FP32 with final BF16 rounding."
            echo "Composed uses native BF16 RmsNorm and RoPE, including BF16 rounding after RmsNorm."
            echo "Paths: fused QKV; composed QKV including V Slice (one matching baseline)."
            echo "Each path/case uses normal inputs only, 20 independent captures each."
            echo "Fused: msprof op, MSPROF_WARMUP=20, one target call, Task Duration(us)."
            echo "Composed: fixed warmup=5, Event samples=10; --iterations applies only here."
            echo "--no-msprof requires --path composed."
            echo "All selected paths are summarized together at the end in final_summary.txt."
            echo "Per-capture durations and data directories form one table directly above the overview; also saved as capture_details.txt."
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$PATH_SELECTION" in
    "") SELECTED_PATHS=(fused composed) ;;
    fused|composed) SELECTED_PATHS=("$PATH_SELECTION") ;;
    *) echo "Invalid --path: $PATH_SELECTION" >&2; exit 2 ;;
esac
[[ "$ROPE_TYPE" =~ ^(all|half|interleave)$ && "$CACHE_WIDTH" =~ ^(all|full|half)$ && "$OUTPUT_LAYOUT" =~ ^(all|bsnd|bnsd)$ && "$ROTARY_DIM" =~ ^(all|96|128)$ ]] || {
    echo "Invalid RoPE/cache/layout/R selection; see --help" >&2; exit 2;
}
SELECTED_CASES=()
for scenario in half-bsnd half-bnsd-r128 interleave-bsnd-r128 interleave-bnsd half-bsnd-compact-r128 half-bnsd-compact interleave-bsnd-compact interleave-bnsd-compact-r128; do
    rope="${scenario%%-*}"
    layout="${scenario#*-}"; layout="${layout%%-*}"
    width=full; [[ "$scenario" != *-compact* ]] || width=half
    rotary=96; [[ "$scenario" != *-r128 ]] || rotary=128
    [[ "$ROPE_TYPE" == all || "$ROPE_TYPE" == "$rope" ]] || continue
    [[ "$CACHE_WIDTH" == all || "$CACHE_WIDTH" == "$width" ]] || continue
    [[ "$OUTPUT_LAYOUT" == all || "$OUTPUT_LAYOUT" == "$layout" ]] || continue
    [[ "$ROTARY_DIM" == all || "$ROTARY_DIM" == "$rotary" ]] || continue
    SELECTED_CASES+=("$scenario")
done
(("${#SELECTED_CASES[@]}" > 0)) || { echo "No selected representative scenario; relax RoPE/layout/cache/R filters" >&2; exit 2; }
(("${#SELECTED_CASES[@]}" <= 10 && "${#SELECTED_CASES[@]}" * "${#SELECTED_PATHS[@]}" <= 20)) || { echo "Performance scenario budget exceeded" >&2; exit 2; }
COMPOSED_PATHS=()
for path in "${SELECTED_PATHS[@]}"; do
    if [[ "$path" == fused ]]; then
        ((USE_MSPROF)) || { echo "Fused requires msprof op; --no-msprof is only for composed paths" >&2; exit 2; }
    else
        COMPOSED_PATHS+=("$path")
    fi
done
[[ "$MSPROF_WARMUP" =~ ^(0|[1-9][0-9]*)$ ]] && ((MSPROF_WARMUP <= 500)) || { echo "Invalid MSPROF_WARMUP (0..500)" >&2; exit 2; }
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid counts" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed" >&2; exit 2; }
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "Invalid device ID" >&2; exit 2; }
if ((DRY_RUN)); then
    echo "shape=1 5672 168 128 heads=56 56 56 dtype=bf16 selected=${#SELECTED_CASES[@]} performance_total=$((${#SELECTED_CASES[@]} * ${#SELECTED_PATHS[@]})) paths=${SELECTED_PATHS[*]}"
    for scenario in "${SELECTED_CASES[@]}"; do
        rope="${scenario%%-*}"
        layout="${scenario#*-}"; layout="${layout%%-*}"
        rotary=96; [[ "$scenario" != *-r128 ]] || rotary=128
        width=$rotary; [[ "$scenario" != *-compact* ]] || width=$((rotary / 2))
        echo "rope=$rope output=$layout cos/sin_width=$width R=$rotary"
    done
    exit 0
fi
command -v "$PYTHON" >/dev/null
if ((USE_MSPROF)); then
    command -v msprof >/dev/null
    TIMING_FILE=profiled_timing.json
else
    TIMING_FILE=timing.json
    SUMMARY_ARGS=(--no-msprof)
fi
unset ASCEND_RT_VISIBLE_DEVICES
if [[ -z "$OUTPUT_DIR" ]]; then
    mkdir -p "${REPO_ROOT}/log/eagle_split_norm_rope_concat"
    OUTPUT_DIR="${REPO_ROOT}/log/eagle_split_norm_rope_concat/perf_$(date +%Y%m%d_%H%M%S)"
    mkdir "$OUTPUT_DIR" || { echo "Use a fresh output directory or retry in a new second" >&2; exit 2; }
else
    mkdir -p "$OUTPUT_DIR"
    [[ -z "$(ls -A -- "$OUTPUT_DIR")" ]] || { echo "Output directory must be empty" >&2; exit 2; }
fi
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
if [[ -z "$DEVICE_ID" ]]; then
    "$PYTHON" "${REPO_ROOT}/tests/tools/select_npu_device.py" --format=report | tee "${OUTPUT_DIR}/npu_select.log"
    DEVICE_ID="$(sed -n 's/^Selected NPU: //p' "${OUTPUT_DIR}/npu_select.log")"
fi
[[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "No valid physical device selected" >&2; exit 1; }
COMMON=(--device-id "$DEVICE_ID" --iterations "$ITERATIONS" --seed "$SEED")
echo "NPU=$DEVICE_ID cases=${SELECTED_CASES[*]} distribution=normal; fused warmup=$MSPROF_WARMUP (default 20); composed warmup=5 iterations=$ITERATIONS; captures=$RUNS output=$OUTPUT_DIR"
# Complete all scenarios of one path before starting the next path.
for path in "${SELECTED_PATHS[@]}"; do
    for case_id in "${SELECTED_CASES[@]}"; do
        FUSED_DURATIONS=()
        for ((run = 1; run <= RUNS; run++)); do
            printf -v run_dir '%s/%s/normal/run_%02d/%s' "$OUTPUT_DIR" "$case_id" "$run" "$path"
            mkdir -p "$run_dir"
            echo "Path=$path case=$case_id distribution=normal Test=$run/$RUNS"
            if [[ "$path" == fused ]]; then
                msprof op --kernel-name=EagleSplitNormRopeConcat --warm-up="$MSPROF_WARMUP" \
                    --launch-count=1 --kill=on --output="$run_dir" \
                    "$PYTHON" "${SCRIPT_DIR}/profile_eagle_split_norm_rope_concat_prod.py" \
                    --device-id "$DEVICE_ID" --seed "$SEED" --case "$case_id" \
                    --msprof-mode 2>&1 | tee "${run_dir}/msprof.log"
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
                FUSED_DURATIONS+=("$duration")
                continue
            elif ((USE_MSPROF)); then
                msprof --output="$run_dir" --task-time=on --ascendcl=on --runtime-api=on \
                    "$PYTHON" "${SCRIPT_DIR}/profile_eagle_split_norm_rope_concat_layer.py" "${COMMON[@]}" \
                    --path "$path" --case "$case_id" --msprof-mode \
                    --output "${run_dir}/${TIMING_FILE}" 2>&1 | tee "${run_dir}/msprof.log"
            else
                "$PYTHON" "${SCRIPT_DIR}/profile_eagle_split_norm_rope_concat_layer.py" "${COMMON[@]}" \
                    --path "$path" --case "$case_id" \
                    --output "${run_dir}/${TIMING_FILE}" 2>&1 | tee "${run_dir}/timing.log"
            fi
            [[ -s "${run_dir}/${TIMING_FILE}" && -s "${run_dir}/event_samples.csv" ]] || {
                echo "Missing timing results in $run_dir" >&2; exit 1;
            }
            if ((USE_MSPROF)); then
                mapfile -t profiles < <(find "$run_dir" -maxdepth 1 -type d -name 'PROF_*')
                [[ ${#profiles[@]} -eq 1 ]] || { echo "Expected exactly one PROF directory" >&2; exit 1; }
                for pattern in 'op_summary*.csv' 'msprof_*.json'; do
                    artifact="$(find "${profiles[0]}" -type f -name "$pattern" -size +0c -print -quit)"
                    [[ -n "$artifact" ]] || { echo "Missing $pattern in $run_dir" >&2; exit 1; }
                done
            fi
        done
        if [[ "$path" == fused ]]; then
            printf '%s\n' "${FUSED_DURATIONS[@]}" | awk -v warmup="$MSPROF_WARMUP" -v case_id="$case_id" '
                BEGIN { print "Fused EagleSplitNormRopeConcat " case_id "/normal: msprof op Task Duration(us)"; print "Test Task Duration(us)" }
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
                        if (spread > 10+1e-9) print "WARNING: spread exceeds 10%. Frequency ramp-up is only a possible cause."
                    }
                    print "Note: Some hardware models may show increased latency across runs. The minimum across all groups for each scenario is reported as the best performance observed under the current test conditions; this does not mean the variation has been eliminated."
                }' > "${OUTPUT_DIR}/${case_id}/normal/fused_task_duration_summary.txt"
        fi
    done
done
if ((${#COMPOSED_PATHS[@]})); then
    # Guard optional array expansion for older Bash versions with nounset.
    if "$PYTHON" "${SCRIPT_DIR}/summarize_eagle_split_norm_rope_concat_layer.py" \
        --output "$OUTPUT_DIR" --runs "$RUNS" --paths "${COMPOSED_PATHS[@]}" \
        --cases "${SELECTED_CASES[@]}" ${SUMMARY_ARGS[@]+"${SUMMARY_ARGS[@]}"} > "${OUTPUT_DIR}/summary.log" 2>&1; then
        [[ -s "${OUTPUT_DIR}/summary.txt" ]] || { echo "Missing composed summary" >&2; exit 1; }
    else
        status=$?
        cat "${OUTPUT_DIR}/summary.log" >&2
        exit "$status"
    fi
fi
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
# Build one uninterrupted per-capture table, including each capture's data directory.
{
    printf 'Scenario\tTest\tMetric\tDuration(us)\tDataDir\n'
    for path in "${SELECTED_PATHS[@]}"; do
        if [[ "$path" == fused ]]; then
            for case_id in "${SELECTED_CASES[@]}"; do
                for ((run = 1; run <= RUNS; run++)); do
                    printf -v run_dir '%s/%s/normal/run_%02d/%s' "$OUTPUT_DIR" "$case_id" "$run" "$path"
                    duration="$(cat "${run_dir}/task_duration_us.txt")"
                    printf 'fused/%s/normal\t%d\tTask Duration\t%s\t%s\n' "$case_id" "$run" "$duration" "$run_dir"
                done
            done
        fi
    done
    if ((${#COMPOSED_PATHS[@]})); then
        awk -v expected="$(( ${#COMPOSED_PATHS[@]} * ${#SELECTED_CASES[@]} * RUNS ))" '
            /^Scenario[[:space:]]+Test[[:space:]]/ { rows=1; next }
            /^Performance results:/ { rows=0 }
            rows && NF {
                scenario=$1; run=$2; duration=$3; directory=$0
                sub(/^[^[:space:]]+[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+/, "", directory)
                printf "%s\t%s\tComplete-call Event\t%s\t%s\n", scenario, run, duration, directory
                count++
            }
            END { if (count != expected) { print "Incomplete composed capture table" > "/dev/stderr"; exit 1 } }
        ' "${OUTPUT_DIR}/summary.txt"
    fi
} | format_table > "${OUTPUT_DIR}/capture_details.txt"
# Publish only after every selected capture and summary validation has succeeded.
{
    echo "=== Final performance summary: all selected paths ==="
    cat "${OUTPUT_DIR}/capture_details.txt"
    echo "=== Final performance overview (us; minimum per scenario) ==="
    {
    printf 'Scenario\tMetric\tMin Duration(us)\n'
    for path in "${SELECTED_PATHS[@]}"; do
        if [[ "$path" == fused ]]; then
            for case_id in "${SELECTED_CASES[@]}"; do
                awk -v case_id="$case_id" '/^Min Task Duration\(us\):/ {
                    sub(/;$/, "", $4)
                    printf "fused/%s/normal\tTask Duration(us)\t%s\n", case_id, $4
                }' "${OUTPUT_DIR}/${case_id}/normal/fused_task_duration_summary.txt"
            done
        fi
    done
    if ((${#COMPOSED_PATHS[@]})); then
        awk '
            /^Performance results:/ { results=1; next }
            /^Note:/ { results=0 }
            results && NF { value=$NF; NF--; printf "%s\tComplete-call Event(us)\t%s\n", $0, value }
        ' "${OUTPUT_DIR}/summary.txt"
    fi
    } | format_table
    echo "=== Measurement notes ==="
    echo "NPU=$DEVICE_ID; captures=$RUNS; fused warmup=$MSPROF_WARMUP; composed warmup=5 samples=$ITERATIONS; msprof=$USE_MSPROF"
    for path in "${SELECTED_PATHS[@]}"; do
        if [[ "$path" == fused ]]; then
            for case_id in "${SELECTED_CASES[@]}"; do
                echo "fused/$case_id/normal:"
                awk '/^Warmup=/ { notes=1 } notes' "${OUTPUT_DIR}/${case_id}/normal/fused_task_duration_summary.txt"
            done
        fi
    done
    echo "The minimum is the best observed performance under the current conditions, not evidence that variation has disappeared."
    echo "Task Duration and complete-call Event measure different boundaries; both paths return Q/K/V; compare only corresponding configurations."
    echo "Results: $OUTPUT_DIR; combined report: $OUTPUT_DIR/final_summary.txt"
} | tee "${OUTPUT_DIR}/final_summary.txt"
