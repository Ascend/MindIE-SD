#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEVICE_ID="${DEVICE_ID:-}"
OUTPUT_DIR=""
KERNEL_NAME="EagleIndexedGateNormModulate"
MSPROF_WARMUP="${MSPROF_WARMUP:-20}"
TEST_COUNT="${TEST_COUNT:-20}"
HIDDEN_SIZES=(2048 5376 6656 7168)
ROW_COUNTS=(13640 5672)
INPUT_DTYPES=(bf16)
OUTPUT_DTYPES=(bf16)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [--path all|fused] [--device-id ID] [--output-dir DIR] [--kernel-name PREFIX]"
            echo "  [--hidden-size 2048|5376|6656|7168] [--input-dtype bf16] [--output-dtype bf16]"
            echo "  [--rows 13640|5672] (indexed rows K=3|12 respectively)"
            echo "All selected fused scenarios: MSPROF_WARMUP=20, TEST_COUNT=20 independent captures."
            echo "Report: final_summary.txt and task_duration_summary.txt under output directory."
            exit 0 ;;
        --path|--device-id|--output-dir|--kernel-name|--hidden-size|--input-dtype|--output-dtype|--rows)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
            case "$1" in
                --path) [[ "$2" == all || "$2" == fused ]] || { echo "ERROR: only all|fused supported" >&2; exit 1; } ;;
                --device-id) DEVICE_ID="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --kernel-name) KERNEL_NAME="$2" ;;
                --rows)
                    [[ "$2" == 13640 || "$2" == 5672 ]] || { echo "ERROR: invalid rows" >&2; exit 1; }
                    ROW_COUNTS=("$2") ;;
                --hidden-size)
                    [[ "$2" == 2048 || "$2" == 5376 || "$2" == 6656 || "$2" == 7168 ]] || { echo "ERROR: invalid hidden size" >&2; exit 1; }
                    HIDDEN_SIZES=("$2") ;;
                --input-dtype|--output-dtype)
                    [[ "$2" == bf16 ]] || { echo "ERROR: performance tests only support bf16 input and output" >&2; exit 1; }
                    if [[ "$1" == --input-dtype ]]; then INPUT_DTYPES=("$2"); else OUTPUT_DTYPES=("$2"); fi ;;
            esac
            shift 2 ;;
        *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done
[[ "$TEST_COUNT" =~ ^[1-9][0-9]*$ && "$MSPROF_WARMUP" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "ERROR: TEST_COUNT positive, MSPROF_WARMUP nonnegative integers required" >&2; exit 1;
}
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid device ID" >&2; exit 1; }
unset ASCEND_RT_VISIBLE_DEVICES
command -v msprof >/dev/null || { echo "ERROR: source CANN set_env.sh first" >&2; exit 1; }
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/log/eagle_indexed_gate_norm_modulate/perf_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
[[ ! -e "${OUTPUT_DIR}/capture_rows.tsv" ]] || { echo "ERROR: use a fresh output directory" >&2; exit 1; }
if [[ -z "$DEVICE_ID" ]]; then
    DEVICE_ID="$(python3 "${REPO_ROOT}/tests/tools/select_npu_device.py" --format=id 2> >(tee "${OUTPUT_DIR}/npu_select.log" >&2))"
fi
[[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid selected device" >&2; exit 1; }
echo "Physical NPU ID=$DEVICE_ID; warm-up=$MSPROF_WARMUP (default 20), independent captures=$TEST_COUNT (default 20)."
for rows in "${ROW_COUNTS[@]}"; do
    if [[ "$rows" == 13640 ]]; then indexed_rows=3; else indexed_rows=12; fi
for hidden in "${HIDDEN_SIZES[@]}"; do
    for input_dtype in "${INPUT_DTYPES[@]}"; do
        for output_dtype in "${OUTPUT_DTYPES[@]}"; do
            scenario="fused_n${rows}_k${indexed_rows}_d${hidden}_${input_dtype}_to_${output_dtype}_normal"
            for ((test = 1; test <= TEST_COUNT; test++)); do
                printf -v run_dir '%s/%s/run_%02d' "$OUTPUT_DIR" "$scenario" "$test"
                mkdir -p "$run_dir"
                msprof op --kernel-name="$KERNEL_NAME" --warm-up="$MSPROF_WARMUP" --launch-count=1 \
                    --kill=on --output="$run_dir" \
                    python3 "${SCRIPT_DIR}/profile_eagle_indexed_gate_norm_modulate_prod.py" \
                    --device-id "$DEVICE_ID" --rows "$rows" --hidden-size "$hidden" --input-dtype "$input_dtype" \
                    --output-dtype "$output_dtype" --msprof-mode 2>&1 | tee "${run_dir}/msprof.log"
                duration="$(python3 "${SCRIPT_DIR}/summarize_eagle_indexed_gate_norm_modulate.py" --read-log "${run_dir}/msprof.log")"
                printf '%s\t%s\t%s\t%s\n' "$scenario" "$test" "$duration" "$run_dir" >> "${OUTPUT_DIR}/capture_rows.tsv"
            done
        done
    done
done
done
python3 "${SCRIPT_DIR}/summarize_eagle_indexed_gate_norm_modulate.py" --output-dir "$OUTPUT_DIR" \
    --test-count "$TEST_COUNT" --warmup "$MSPROF_WARMUP" --device-id "$DEVICE_ID"
