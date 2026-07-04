#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UT_SOURCE_DIR="${REPO_ROOT}/csrc/ops/fused_infer_attention_score/tests/ut"
TARGET_NAME="mindiesd_fia_arch35_tiling_ut"

BUILD_ONLY=0
BUILD_DIR="${REPO_ROOT}/build/fia_arch35_ut"
LOG_DIR="${REPO_ROOT}/logs"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --build-only           Configure and build the FIA arch35 tiling UT without running it.
  --build-dir <dir>      CMake build directory. Default: ${BUILD_DIR}
  --log-dir <dir>        Log directory. Default: ${LOG_DIR}
  -h, --help             Show this help.

Environment:
  ASCEND_ENV_SH          Explicit Ascend set_env.sh path.
  ASCEND_TOOLKIT_HOME    Ascend toolkit root, used to locate set_env.sh.
  CUSTOM_ASCEND_CANN_PACKAGE_PATH
                         Forwarded to CMake when set.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

source_ascend_env() {
    local candidates=()
    if [[ -n "${ASCEND_ENV_SH:-}" ]]; then
        candidates+=("${ASCEND_ENV_SH}")
    fi
    if [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
        candidates+=("${ASCEND_TOOLKIT_HOME}/set_env.sh")
    fi
    candidates+=(
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    )

    local env_file
    for env_file in "${candidates[@]}"; do
        if [[ -f "${env_file}" ]]; then
            # shellcheck disable=SC1090
            source "${env_file}"
            echo "ascend_env=${env_file}"
            return 0
        fi
    done

    echo "warning=no Ascend set_env.sh found; continuing with current environment"
}

mkdir -p "${BUILD_DIR}" "${LOG_DIR}"
source_ascend_env

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_LOG="${LOG_DIR}/${TIMESTAMP}_fia_arch35_ut_configure.log"
BUILD_LOG="${LOG_DIR}/${TIMESTAMP}_fia_arch35_ut_build.log"
RUN_LOG="${LOG_DIR}/${TIMESTAMP}_fia_arch35_ut_run.log"

print_failure_summary() {
    local log_file="$1"
    if [[ -f "${log_file}" ]]; then
        grep -n -E "error:|ERROR|FAILED|failed|Fatal|fatal|undefined reference|not found|No such file" \
            "${log_file}" | tail -n 80 || true
    fi
}

run_logged() {
    local step="$1"
    local log_file="$2"
    shift 2

    echo "step=${step}"
    echo "log=${log_file}"
    set +e
    "$@" >"${log_file}" 2>&1
    local rc=$?
    set -e
    echo "exit_code=${rc}"
    if [[ ${rc} -ne 0 ]]; then
        print_failure_summary "${log_file}"
        exit "${rc}"
    fi
}

CMAKE_ARGS=(
    -S "${UT_SOURCE_DIR}"
    -B "${BUILD_DIR}"
)
if [[ -n "${CUSTOM_ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    CMAKE_ARGS+=("-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${CUSTOM_ASCEND_CANN_PACKAGE_PATH}")
fi

run_logged "configure" "${CONFIG_LOG}" cmake "${CMAKE_ARGS[@]}"
run_logged "build" "${BUILD_LOG}" cmake --build "${BUILD_DIR}" --target "${TARGET_NAME}" -j

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
    echo "target=${TARGET_NAME}"
    echo "status=build_only_complete"
    exit 0
fi

run_logged "run" "${RUN_LOG}" "${BUILD_DIR}/${TARGET_NAME}"
echo "target=${TARGET_NAME}"
echo "status=complete"
