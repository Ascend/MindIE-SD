#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python3 -m tests.ops.sigmoid_mul_dynamic_mx_quant.prod_accuracy "$@"
