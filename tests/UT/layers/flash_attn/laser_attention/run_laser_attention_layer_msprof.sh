#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE_ID="${DEVICE_ID:-}"
WARMUP=5
ITERATIONS=10
RUNS=20
SEED=20260921
OUTPUT_DIR=""
USE_MSPROF=1
SUMMARY_ARGS=()
while (($#)); do
  case "$1" in
    --no-msprof) USE_MSPROF=0; shift ;;
    --device-id|--iterations|--runs|--seed|--output)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --device-id) DEVICE_ID="$2" ;;
        --iterations) ITERATIONS="$2" ;;
        --runs) RUNS="$2" ;;
        --seed) SEED="$2" ;;
        --output) OUTPUT_DIR="$2" ;;
      esac
      shift 2 ;;
    -h|--help)
      echo "Usage: bash $0 [--device-id N] [--iterations 10] [--runs 20] [--seed N] [--output DIR] [--no-msprof]"
      echo "--no-msprof runs Python directly and saves every Event sample in JSON and event_samples.csv."
      echo "BSND BF16 Q=[1,5280,24,128], K/V=[1,8800,24,128]; complete current/legacy layers."
      echo "Fixed warmup=5; minimum of individual Event timings per capture. Run all current captures, then all legacy captures."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid iterations" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid runs" >&2; exit 2; }
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "Invalid device-id" >&2; exit 2; }
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
  mkdir -p "${REPO_ROOT}/logs"
  OUTPUT_DIR="$(mktemp -d "${REPO_ROOT}/logs/la_layer_msprof_XXXXXXXX")"
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
BENCH_SCRIPT="${SCRIPT_DIR}/profile_laser_attention_layer.py"
COMMON=(--device-id "$DEVICE_ID" --iterations "$ITERATIONS" --seed "$SEED")
echo "Physical NPU=$DEVICE_ID; fixed warmup=$WARMUP; captures per path=$RUNS; measured calls per capture=$ITERATIONS; output=$OUTPUT_DIR"

# Snapshots are outside benchmark processes. They do not establish in-kernel frequency.
device_snapshot() {
  local destination="$1"
  {
    date -Is
    if command -v npu-smi >/dev/null 2>&1; then
      npu-smi info || echo "WARNING: npu-smi info failed"
      npu-smi info -t common -i "$DEVICE_ID" || echo "WARNING: common query unavailable"
    else
      echo "WARNING: npu-smi unavailable; no device snapshot"
    fi
  } > "$destination" 2>&1
}

{
  date -Is
  uname -a
  command -v "$PYTHON"
  echo "use_msprof=$USE_MSPROF"
  if ((USE_MSPROF)); then
    command -v msprof
    msprof --version || echo "WARNING: msprof version query unavailable"
  fi
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info -t board -i "$DEVICE_ID" || echo "WARNING: board/version query unavailable"
  fi
} > "${OUTPUT_DIR}/environment.log" 2>&1

for path in current legacy; do
  # msprof captures ALL operators, including casts/padding/copies in each layer.
  # Warmup is Python's for-loop; no msprof op / kernel-replay parameters are used.
  for ((run_index = 1; run_index <= RUNS; run_index++)); do
    printf -v group_dir '%s/run_%02d' "$OUTPUT_DIR" "$run_index"
    run_dir="${group_dir}/${path}"
    mkdir -p "$run_dir"
    echo "=== Path=$path Test=$run_index/$RUNS ==="
    device_snapshot "${run_dir}/device_before.log"
    if ((USE_MSPROF)); then
      msprof --output="$run_dir" --task-time=on --ascendcl=on --runtime-api=on \
        "$PYTHON" "$BENCH_SCRIPT" "${COMMON[@]}" --path "$path" --msprof-mode \
        --output "${run_dir}/${TIMING_FILE}" 2>&1 | tee "${run_dir}/msprof.log"
    else
      "$PYTHON" "$BENCH_SCRIPT" "${COMMON[@]}" --path "$path" \
        --output "${run_dir}/${TIMING_FILE}" 2>&1 | tee "${run_dir}/timing.log"
    fi
    device_snapshot "${run_dir}/device_after.log"
    [[ -s "${run_dir}/${TIMING_FILE}" && -s "${run_dir}/event_samples.csv" ]] || { echo "Missing $path timing result/samples" >&2; exit 1; }
    if ((!USE_MSPROF)); then
      echo "Event samples: ${run_dir}/event_samples.csv"
      continue
    fi
    summary="$(find "$run_dir" -type f -name 'op_summary*.csv' -print -quit)"
    timeline="$(find "$run_dir" -type f -name 'msprof_*.json' -print -quit)"
    [[ -n "$summary" && -s "$summary" && -n "$timeline" && -s "$timeline" ]] || {
      echo "Missing msprof summary/timeline for $path; inspect ${run_dir}/msprof.log" >&2; exit 1;
    }
    echo "$path: $summary"
    echo "Timeline: $timeline; $ITERATIONS measured calls follow $WARMUP warmup calls (no named range)"
  done
done

"$PYTHON" "${SCRIPT_DIR}/summarize_laser_attention_layer.py" --output "$OUTPUT_DIR" --runs "$RUNS" "${SUMMARY_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/summary.log"
echo "Results: ${OUTPUT_DIR}/summary.txt and summary.json; each run_NN retains its raw data."
