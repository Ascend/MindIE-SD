#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
unset ASCEND_RT_VISIBLE_DEVICES
exec python3 -m tests.ops.sigmoid_mul_dynamic_mx_quant.performance "$@"
