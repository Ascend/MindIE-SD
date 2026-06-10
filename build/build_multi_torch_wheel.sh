#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
VARIANT_ROOT=${MINDIESD_MULTI_TORCH_PLUGIN_DIR:-"${REPO_ROOT}/build/torch_plugin_variants"}
PYTHON_BIN=${PYTHON_BIN:-python3}
WHEEL_CONTAINER=${MINDIESD_WHEEL_CONTAINER:-mindiesd_torch26}
ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT:-"ascend910;ascend910b;ascend910_93"}
MINDIESD_SKIP_OPS_BUILD=${MINDIESD_SKIP_OPS_BUILD:-0}

VARIANTS=(
    "torch26:mindiesd_torch26"
    "torch27:mindiesd_torch27"
    "torch28:mindiesd_torch28"
    "torch29:mindiesd_torch29"
    "torch210:mindiesd_torch210"
)

mkdir -p "${VARIANT_ROOT}"

copy_plugin_from_container() {
    local container=$1
    local variant=$2
    local source_in_repo="${REPO_ROOT}/build/plugin_build/libPTAExtensionOPS.so"
    local target_dir="${VARIANT_ROOT}/${variant}"

    mkdir -p "${target_dir}"
    if docker cp "${container}:${source_in_repo}" "${target_dir}/libPTAExtensionOPS.so"; then
        return 0
    fi

    if [ -f "${source_in_repo}" ]; then
        cp "${source_in_repo}" "${target_dir}/libPTAExtensionOPS.so"
        return 0
    fi

    echo "Failed to copy ${source_in_repo} from ${container}." >&2
    return 1
}

for item in "${VARIANTS[@]}"; do
    variant=${item%%:*}
    container=${item##*:}

    echo "Building ${variant} plugin in ${container}..."
    docker exec "${container}" bash -lc "cd '${REPO_ROOT}/build' && rm -rf plugin_build && bash build_plugin.sh '${REPO_ROOT}/build'"
    copy_plugin_from_container "${container}" "${variant}"
done

echo "Building multi torch wheel..."
docker exec \
    -e MINDIESD_WHEEL_MODE=multi_torch \
    -e MINDIESD_MULTI_TORCH_PLUGIN_DIR="${VARIANT_ROOT}" \
    -e MINDIESD_SKIP_OPS_BUILD="${MINDIESD_SKIP_OPS_BUILD}" \
    -e ASCEND_COMPUTE_UNIT="${ASCEND_COMPUTE_UNIT}" \
    -e PYTHON_BIN="${PYTHON_BIN}" \
    "${WHEEL_CONTAINER}" \
    bash -lc "cd '${REPO_ROOT}' && \"${PYTHON_BIN}\" setup.py bdist_wheel"
