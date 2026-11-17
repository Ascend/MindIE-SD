#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEVICE_ID="${DEVICE_ID:-}"
MODE="all"
ROTARY_DIM="all"
ROPE_TYPE="all"
CACHE_WIDTH="all"
OUTPUT_LAYOUT="all"
SHAPE=(1 5672 168 128)
HEADS=(56 56 56)
DRY_RUN=0
CUSTOM_GEOMETRY=0
STRESS_RUNS=1000
OUTPUT_DIR=""
usage() {
    echo "Usage: $0 [--device-id ID] [--mode all|accuracy|stress] [--stress-runs N] [--output DIR]"
    echo "  --shape B S Htotal D       Actual BSND input (default: 1 5672 168 128; D=128 only)"
    echo "  --heads Q K V             Head counts (default: 56 56 56; sum=Htotal; Ascend950 each branch 1..128)"
    echo "  --rope-type all|half|interleave   (default: all)"
    echo "  --cache-width all|full|half       cos/sin width full=R, half=R/2 (default: all)"
    echo "  --rotary-dim all|96|128          (default: all; ten input cases on Ascend950)"
    echo "  --output-layout all|bsnd|bnsd     (default: all)"
    echo "  --dry-run                 Validate and print selected scenarios without Python/NPU or output files"
    echo "Fixed precision configuration: BF16, affine RMSNorm (norm_type=3), eps=1e-5."
    echo "Example: $0 --shape 1 5672 168 128 --heads 56 56 56 --rope-type interleave --cache-width half --output-layout bnsd --mode accuracy"
    echo "Default: original eight cases plus Ascend950 [1,128,128]-head normal and uniform, then stress (1000 repeats each)."
    echo "Default output: <repository>/log/eagle_split_norm_rope_concat/accuracy_YYYYMMDD_HHMMSS/ (accuracy.log and stress mismatch dumps)."
}
set_case_distributions() {
    case "$1" in
        half-bsnd-asymmetric) DISTRIBUTIONS=(normal uniform) ;;
        half-bsnd|interleave-bnsd|half-bnsd-compact|interleave-bsnd-compact) DISTRIBUTIONS=(uniform) ;;
        *) DISTRIBUTIONS=(normal) ;;
    esac
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-id)
            [[ $# -ge 2 ]] || { echo "Missing device ID" >&2; exit 1; }
            DEVICE_ID="$2"; shift 2 ;;
        --mode)
            [[ $# -ge 2 ]] || { echo "Missing mode" >&2; exit 1; }
            MODE="$2"; shift 2 ;;
        --stress-runs)
            [[ $# -ge 2 ]] || { echo "Missing stress run count" >&2; exit 1; }
            STRESS_RUNS="$2"; shift 2 ;;
        --shape)
            [[ $# -ge 5 ]] || { echo "--shape needs B S Htotal D" >&2; exit 1; }
            SHAPE=("$2" "$3" "$4" "$5"); CUSTOM_GEOMETRY=1; shift 5 ;;
        --heads)
            [[ $# -ge 4 ]] || { echo "--heads needs Q K V" >&2; exit 1; }
            HEADS=("$2" "$3" "$4"); CUSTOM_GEOMETRY=1; shift 4 ;;
        --rope-type|--cache-width|--output-layout|--rotary-dim)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 1; }
            case "$1" in
                --rotary-dim) ROTARY_DIM="$2" ;;
                --rope-type) ROPE_TYPE="$2" ;;
                --cache-width) CACHE_WIDTH="$2" ;;
                --output-layout) OUTPUT_LAYOUT="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --case) echo "Use --shape, --heads, --rope-type, --cache-width and --output-layout instead of --case" >&2; exit 1 ;;
        --output)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing output directory" >&2; exit 1; }
            OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done
[[ "${MODE}" =~ ^(all|accuracy|stress)$ ]] || { echo "Invalid mode: ${MODE}" >&2; exit 1; }
[[ "${STRESS_RUNS}" =~ ^[1-9][0-9]*$ ]] || { echo "Stress runs must be a positive integer" >&2; exit 1; }
[[ "$ROTARY_DIM" =~ ^(all|96|128)$ ]] || { echo "Invalid --rotary-dim: $ROTARY_DIM" >&2; exit 1; }
[[ "$ROPE_TYPE" =~ ^(all|half|interleave)$ ]] || { echo "Invalid --rope-type: $ROPE_TYPE" >&2; exit 1; }
[[ "$CACHE_WIDTH" =~ ^(all|full|half)$ ]] || { echo "Invalid --cache-width: $CACHE_WIDTH" >&2; exit 1; }
[[ "$OUTPUT_LAYOUT" =~ ^(all|bsnd|bnsd)$ ]] || { echo "Invalid --output-layout: $OUTPUT_LAYOUT" >&2; exit 1; }
for value in "${SHAPE[@]}" "${HEADS[@]}"; do
    [[ "$value" =~ ^[1-9][0-9]{0,9}$ ]] && (( value <= 2147483647 )) || {
        echo "--shape and --heads values must be positive int32 integers" >&2; exit 1;
    }
done
[[ "${SHAPE[3]}" == 128 ]] || { echo "--shape D must be 128" >&2; exit 1; }
numel=1
for value in "${SHAPE[@]}"; do
    (( numel <= 2147483647 / value )) || { echo "--shape numel must not exceed 2147483647" >&2; exit 1; }
    numel=$((numel * value))
done
(( HEADS[0] + HEADS[1] + HEADS[2] == SHAPE[2] )) || {
    echo "--heads must satisfy Q+K+V=Htotal" >&2; exit 1;
}
CASES=()
for scenario in half-bsnd half-bnsd-r128 interleave-bsnd-r128 interleave-bnsd half-bsnd-compact-r128 half-bnsd-compact interleave-bsnd-compact interleave-bnsd-compact-r128 half-bsnd-asymmetric; do
    if ((CUSTOM_GEOMETRY)); then
        if [[ "${SHAPE[*]}" == '1 5672 257 128' && "${HEADS[*]}" == '1 128 128' ]]; then
            [[ "$scenario" == half-bsnd-asymmetric ]] || continue
        else
            [[ "$scenario" != half-bsnd-asymmetric ]] || continue
        fi
    fi
    rope="${scenario%%-*}"
    layout="${scenario#*-}"; layout="${layout%%-*}"
    width=full; [[ "$scenario" != *-compact* ]] || width=half
    [[ "$ROPE_TYPE" == all || "$ROPE_TYPE" == "$rope" ]] || continue
    [[ "$CACHE_WIDTH" == all || "$CACHE_WIDTH" == "$width" ]] || continue
    [[ "$OUTPUT_LAYOUT" == all || "$OUTPUT_LAYOUT" == "$layout" ]] || continue
    rotary=96; [[ "$scenario" != *-r128 ]] || rotary=128
    [[ "$ROTARY_DIM" == all || "$ROTARY_DIM" == "$rotary" ]] || continue
    CASES+=("$scenario")
done
((${#CASES[@]})) || { echo "No selected representative scenario; relax RoPE/layout/cache/R filters" >&2; exit 1; }
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "Invalid device ID" >&2; exit 1; }
if (( DRY_RUN )); then
    selected=0
    for scenario in "${CASES[@]}"; do
        set_case_distributions "$scenario"
        selected=$((selected + ${#DISTRIBUTIONS[@]}))
    done
    echo "shape=${SHAPE[*]} heads=${HEADS[*]} dtype=bf16 norm_type=3 eps=1e-5 mode=$MODE selected=$selected"
    for scenario in "${CASES[@]}"; do
        rope="${scenario%%-*}"
        layout="${scenario#*-}"; layout="${layout%%-*}"
        rotary=96; [[ "$scenario" != *-r128 ]] || rotary=128
        width=$rotary; [[ "$scenario" != *-compact* ]] || width=$((rotary / 2))
        case_shape="${SHAPE[*]}"; case_heads="${HEADS[*]}"
        if ((!CUSTOM_GEOMETRY)) && [[ "$scenario" == half-bsnd-asymmetric ]]; then case_shape='1 5672 257 128'; case_heads='1 128 128'; fi
        set_case_distributions "$scenario"
        for distribution in "${DISTRIBUTIONS[@]}"; do
            echo "case=$scenario shape=$case_shape heads=$case_heads rope=$rope output=$layout cos/sin_width=$width R=$rotary distribution=$distribution"
        done
    done
    exit 0
fi
[[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "Supply --device-id for an allocated healthy logical NPU" >&2; exit 1; }
soc_version="$(python3 -c 'import sys, torch_npu; torch_npu.npu.set_device(int(sys.argv[1])); print(torch_npu.npu.get_soc_version())' "$DEVICE_ID")"
if [[ "$soc_version" != 260 ]]; then
    FILTERED_CASES=()
    for scenario in "${CASES[@]}"; do [[ "$scenario" == half-bsnd-asymmetric ]] || FILTERED_CASES+=("$scenario"); done
    CASES=("${FILTERED_CASES[@]}")
    ((${#CASES[@]})) || { echo "The asymmetric case requires Ascend950" >&2; exit 1; }
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    mkdir -p "${REPO_ROOT}/log/eagle_split_norm_rope_concat"
    OUTPUT_DIR="${REPO_ROOT}/log/eagle_split_norm_rope_concat/accuracy_$(date +%Y%m%d_%H%M%S)"
    mkdir "$OUTPUT_DIR" || { echo "Use a fresh output directory or retry in a new second" >&2; exit 1; }
else
    mkdir -p "$OUTPUT_DIR"
    [[ -z "$(ls -A -- "$OUTPUT_DIR")" ]] || { echo "Output directory must be empty" >&2; exit 1; }
fi
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
cd "$REPO_ROOT"
{
echo "Logical NPU ID=$DEVICE_ID; log=${OUTPUT_DIR}/accuracy.log"
RESULTS=()
for stage in accuracy stress; do
    [[ "${MODE}" == all || "${MODE}" == "${stage}" ]] || continue
    for scenario in "${CASES[@]}"; do
        rope="${scenario%%-*}"
        layout="${scenario#*-}"; layout="${layout%%-*}"
        width=full; [[ "$scenario" != *-compact* ]] || width=half
        rotary=96; [[ "$scenario" != *-r128 ]] || rotary=128
        CASE_SHAPE=("${SHAPE[@]}"); CASE_HEADS=("${HEADS[@]}")
        if ((!CUSTOM_GEOMETRY)) && [[ "$scenario" == half-bsnd-asymmetric ]]; then CASE_SHAPE=(1 5672 257 128); CASE_HEADS=(1 128 128); fi
        description="shape=${CASE_SHAPE[*]} heads=${CASE_HEADS[*]} rope=$rope output=$layout cache_width=$width R=$rotary dtype=bf16"
        set_case_distributions "$scenario"
        for distribution in "${DISTRIBUTIONS[@]}"; do
            python3 -u -m tests.ops.eagle_split_norm_rope_concat.check_eagle_split_norm_rope_concat_prod_accuracy \
                --device-id "${DEVICE_ID}" --mode "${stage}" --rope-type "$rope" --cache-width "$width" \
                --output-layout "$layout" --rotary-dim "$rotary" --case "$scenario" --shape "${CASE_SHAPE[@]}" --heads "${CASE_HEADS[@]}" --distribution "${distribution}" \
                --stress-runs "${STRESS_RUNS}" --output "$OUTPUT_DIR"
            if [[ "${stage}" == stress ]]; then
                RESULTS+=("PASS stress $description distribution=${distribution} runs=${STRESS_RUNS} outputs=q,k,v bit_exact")
            else
                RESULTS+=("PASS accuracy $description distribution=${distribution} outputs=q,k,v (mixed tolerance diagnostic only)")
            fi
        done
    done
done
echo "=== Selected tests passed ==="
printf '%s\n' "${RESULTS[@]}"
} 2>&1 | tee "${OUTPUT_DIR}/accuracy.log"
