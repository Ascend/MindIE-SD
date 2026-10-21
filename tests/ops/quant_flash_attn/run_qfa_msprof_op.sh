#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2. See LICENSE for details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEVICE_ID="${DEVICE_ID:-}"
OUTPUT_DIR=""
MSPROF_WARMUP="${MSPROF_WARMUP:-20}"
TEST_COUNT="${TEST_COUNT:-20}"
PATH_SELECTION=all
EXTRA_ARGS=()
usage() {
    cat <<'EOF'
Usage: bash run_qfa_msprof_op.sh [--device-id ID] [--output-dir DIR]
       [--path all|qfa] [--layout all|BNSD|BSND] [-- extra scene arguments]
Only the direct QuantFlashAttn path is defined. Default: all.
Default scenes: BNSD Q=[1,32,2048,128], K/V=[1,4,32768,128];
                BSND Q=[1,2048,32,128], K/V=[1,32768,4,128]. MXFP4, BF16 output.
Both uniform and normal distributions are captured on the same physical device.
Scene arguments: --batch --q-heads --kv-heads --q-seq --kv-seq --layout --seed.
Environment: MSPROF_WARMUP=20, TEST_COUNT=20, DEVICE_ID optional.
Default output: repository log/quant_flash_attn/perf_<timestamp>.
Each independent capture uses --launch-count=1 and --kill=on.
EOF
}
while (($#)); do
    case "$1" in
        --device-id|--output-dir|--path|--layout)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
            case "$1" in
                --device-id) DEVICE_ID="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --path) PATH_SELECTION="$2" ;;
                --layout) EXTRA_ARGS+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --layout=*) EXTRA_ARGS+=("$1"); shift ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done
[[ "$PATH_SELECTION" == all || "$PATH_SELECTION" == qfa ]] || { echo "ERROR: --path must be all or qfa" >&2; exit 1; }
[[ "$TEST_COUNT" =~ ^[1-9][0-9]*$ && "$MSPROF_WARMUP" =~ ^[0-9]+$ ]] || {
    echo "ERROR: TEST_COUNT must be positive; MSPROF_WARMUP must be nonnegative" >&2; exit 1;
}
[[ -z "$DEVICE_ID" || "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid --device-id" >&2; exit 1; }
unset ASCEND_RT_VISIBLE_DEVICES
# Validate all forwarded scene options before device selection or capture.
SCENES="$(python3 - "${SCRIPT_DIR}" "${EXTRA_ARGS[@]}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from profile_qfa_prod import parser, cases_from_args
p = parser(default_layout="all")
args = p.parse_args(sys.argv[2:])
if any(x.startswith("--device-id") for x in sys.argv[2:]):
    p.error("pass --device-id before --")
for case in cases_from_args(args):
    print(case.layout, case.label)
PY
)"
command -v msprof >/dev/null || { echo "ERROR: msprof not found; source CANN environment" >&2; exit 1; }
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/log/quant_flash_attn/perf_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: use a new output directory: $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
if [[ -z "$DEVICE_ID" ]]; then
    DEVICE_ID="$(python3 "${REPO_ROOT}/tests/tools/select_npu_device.py" --format=id)"
fi
[[ "$DEVICE_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid selected NPU ID" >&2; exit 1; }
echo "device_id=${DEVICE_ID} warm-up=${MSPROF_WARMUP} independent_captures=${TEST_COUNT} per layout/distribution"
SUMMARY_ARGS=()
while read -r layout scene; do
    SUMMARY_ARGS+=(--scene "$scene")
    for distribution in uniform normal; do
        for ((run_index=1; run_index<=TEST_COUNT; run_index++)); do
            printf -v run_dir '%s/%s/%s/run_%02d' "$OUTPUT_DIR" "$scene" "$distribution" "$run_index"
            mkdir -p "$run_dir"
            msprof op --kernel-name=QuantFlashAttn --warm-up="$MSPROF_WARMUP" \
                --launch-count=1 --kill=on --output="$run_dir" \
                python3 "${SCRIPT_DIR}/profile_qfa_prod.py" --device-id "$DEVICE_ID" \
                --msprof-mode --distribution "$distribution" "${EXTRA_ARGS[@]}" --layout "$layout" \
                2>&1 | tee "${run_dir}/msprof.log"
            python3 "${SCRIPT_DIR}/qfa_perf_summary.py" --check-log "${run_dir}/msprof.log"
        done
    done
done <<< "$SCENES"
python3 "${SCRIPT_DIR}/qfa_perf_summary.py" --output-dir "$OUTPUT_DIR" \
    "${SUMMARY_ARGS[@]}" --count "$TEST_COUNT" --warmup "$MSPROF_WARMUP"
