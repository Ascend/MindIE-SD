#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
unset ASCEND_RT_VISIBLE_DEVICES
cd "${REPO_ROOT}"
exec python3 -m tests.ops.eagle_quant_batch_matmul_swiglu.check_eagle_quant_batch_matmul_swiglu_prod_accuracy "$@"
