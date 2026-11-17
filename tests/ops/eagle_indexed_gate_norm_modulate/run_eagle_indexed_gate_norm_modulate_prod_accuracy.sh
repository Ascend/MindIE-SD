#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEVICE_ID="${DEVICE_ID:-}"
MODE=all
STRESS_RUNS=1000
OUTPUT_DIR=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [--device-id ID] [--mode all|accuracy|stress] [--stress-runs N]"
            echo "  [--hidden-size 2048|5376|6656|7168] [--input-dtype fp16|bf16] [--output-dtype fp16|bf16] [--output-dir DIR]"
            echo "  [--rows 13640|5672] (indexed rows K=3|12 respectively)"
            echo "Default: accuracy for every selected case, then 1000 fixed-input stress repeats per case."
            echo "10 representative cases including distribution; filters only select from the README matrix."
            exit 0 ;;
        --device-id|--mode|--stress-runs|--hidden-size|--input-dtype|--output-dtype|--output-dir|--rows)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
            case "$1" in
                --device-id) DEVICE_ID="$2" ;;
                --mode) MODE="$2" ;;
                --stress-runs) STRESS_RUNS="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --rows)
                    [[ "$2" == 13640 || "$2" == 5672 ]] || { echo "ERROR: invalid rows" >&2; exit 1; }
                    EXTRA_ARGS+=("$1" "$2") ;;
                --hidden-size)
                    [[ "$2" == 2048 || "$2" == 5376 || "$2" == 6656 || "$2" == 7168 ]] || { echo "ERROR: invalid hidden size" >&2; exit 1; }
                    EXTRA_ARGS+=("$1" "$2") ;;
                *)
                    [[ "$2" == fp16 || "$2" == bf16 ]] || { echo "ERROR: invalid dtype" >&2; exit 1; }
                    EXTRA_ARGS+=("$1" "$2") ;;
            esac
            shift 2 ;;
        *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done
[[ "$MODE" == all || "$MODE" == accuracy || "$MODE" == stress ]] || { echo "ERROR: invalid mode" >&2; exit 1; }
[[ "$STRESS_RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: stress runs must be positive" >&2; exit 1; }
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid device ID" >&2; exit 1; }
unset ASCEND_RT_VISIBLE_DEVICES
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/log/eagle_indexed_gate_norm_modulate/accuracy_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
if [[ -z "$DEVICE_ID" ]]; then
    DEVICE_ID="$(python3 "${REPO_ROOT}/tests/tools/select_npu_device.py" --format=id 2> >(tee "${OUTPUT_DIR}/npu_select.log" >&2))"
fi
[[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid selected device" >&2; exit 1; }
echo "Physical NPU ID=$DEVICE_ID; log=${OUTPUT_DIR}/accuracy.log"
python3 "${SCRIPT_DIR}/check_eagle_indexed_gate_norm_modulate_prod_accuracy.py" \
    --device-id "$DEVICE_ID" --mode "$MODE" --stress-runs "$STRESS_RUNS" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "${OUTPUT_DIR}/accuracy.log"
