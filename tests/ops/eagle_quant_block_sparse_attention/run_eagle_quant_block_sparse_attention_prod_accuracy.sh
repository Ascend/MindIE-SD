#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Source an explicitly requested environment; otherwise use the caller's CANN environment.
if [[ -n "${ASCEND_ENV_SH:-}" ]]; then
    source "${ASCEND_ENV_SH}"
fi
unset ASCEND_RT_VISIBLE_DEVICES
cd "${REPO_ROOT}"
# The device-free coordinator validates arguments/restores the plan before auto-selection.
exec python3 -u -m tests.ops.eagle_quant_block_sparse_attention.check_eagle_quant_block_sparse_attention_prod_accuracy "$@"
