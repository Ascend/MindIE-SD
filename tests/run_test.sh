#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
export MINDIE_LOG_TO_STDOUT=true

set -e

export MINDIE_TEST_MODE="ALL"

test_files=()
precision_test_files=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu_only)
            export MINDIE_TEST_MODE="CPU"
            echo "Run only CPU-compatible tests."
            shift
            ;;
        --npu_only)
            export MINDIE_TEST_MODE="NPU"
            echo "Run only NPU-dependent tests."
            shift
            ;;
        --all)
            export MINDIE_TEST_MODE="ALL"
            echo "Run all tests (default behavior)."
            shift
            ;;
        --precision_test)
            precision_test_files="$2"
            shift 2
            ;;
        --help)
            echo "Usage:  bash run_test.sh [OPTIONS] [TEST_FILES...]"
            echo ""
            echo "Options:"
            echo "  --cpu_only       Run only CPU-compatible tests."
            echo "  --npu_only       Run only NPU-dependent tests."
            echo "  --all            Run all tests (default behavior)."
            echo "  --precision_test <NAMES>  Run only tests whose file names exactly match one of the |-separated NAMES (no coverage)."
            echo "  --help           Show this help message and exit."
            echo ""
            echo "TEST_FILES:"
            echo "  Optional. One or more test file paths to run individually."
            echo "  When specified, only those files are executed (with PYTHONPATH"
            echo "  set correctly for imports)."
            echo ""
            echo "Examples:"
            echo "  bash run_test.sh"
            echo "  bash run_test.sh --cpu_only"
            echo "  bash run_test.sh layers/test_embedding.py"
            echo "  bash run_test.sh layers/test_embedding.py quantization/test_mode.py"
            echo "  bash run_test.sh --cpu_only layers/test_embedding.py"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
        *)
            test_files+=("$1")
            shift
            ;;
    esac
done

if command -v python3 &> /dev/null; then
    python_command=python3
else
    python_command=python
fi

current_directory=$(dirname "$(readlink -f "$0")")

# --- 运行指定的测试用例 ---
if [[ ${#test_files[@]} -gt 0 ]]; then
    project_root=$(dirname "${current_directory}")
    export PYTHONPATH="${project_root}:${current_directory}${PYTHONPATH:+:$PYTHONPATH}"

    # Set ASCEND_CUSTOM_OPP_PATH (same as run.py)
    custom_op_path1="${project_root}/mindiesd/ops/vendors/aie_ascendc"
    custom_op_path2="${project_root}/mindiesd/ops/vendors/customize"
    old_custom_op_path="${ASCEND_CUSTOM_OPP_PATH:-}"
    export ASCEND_CUSTOM_OPP_PATH="${custom_op_path1}:${custom_op_path2}:${old_custom_op_path}"

    rc=0
    for f in "${test_files[@]}"; do
        echo "=== Running Start: ${f} ==="
        ${python_command} "${f}" || rc=$?
    done
    exit ${rc}
fi

# --- 精准匹配模式：PR流水线中只运行指定文件名的测试用例，不进行覆盖率分析 ---
if [[ -n "$precision_test_files" ]]; then
    echo "=== Running Start: ${precision_test_files} ==="
    set +e
    ${python_command} ${current_directory}/run.py --precision_test "$precision_test_files" \
        2>&1 | tee -a ${current_directory}/run.log
    precision_rc=${PIPESTATUS[0]}
    set -e
    exit ${precision_rc}
fi

# --- 默认模式: 运行全量测试用例，并且独立收集覆盖率 ---
cov_task_dir="/tmp/mindiesd/task"
rm -rf "$cov_task_dir"

rc=0
for test_file in $(find ${current_directory} -name "test_*.py" | grep -v "/UT/"); do
    test_name=$(basename "$test_file" .py)
    mkdir -p "${cov_task_dir}/${test_name}/covdata/"
    export COVERAGE_FILE="${cov_task_dir}/${test_name}/covdata/coverage"

    echo "=== Running Start: ${test_file} ==="

    set +e
    ${python_command} -m coverage run \
        --rcfile ${current_directory}/../.coveragerc \
        ${current_directory}/run.py "$test_file" \
        2>&1 | tee -a ${current_directory}/run.log
    cov_rc=${PIPESTATUS[0]}
    set -e
    if [[ ${cov_rc} -ne 0 ]]; then
        rc=${cov_rc}
    fi
done

# 合并覆盖率数据
export COVERAGE_FILE="${cov_task_dir}/.coverage"
${python_command} -m coverage combine --keep --rcfile ${current_directory}/../.coveragerc $(find ${cov_task_dir} -name "coverage" -path "*/covdata/coverage")

# 生成覆盖率报告
${python_command} -m coverage report --rcfile ${current_directory}/../.coveragerc
${python_command} -m coverage xml --rcfile ${current_directory}/../.coveragerc -o ${current_directory}/coverage.xml
${python_command} -m coverage html --rcfile ${current_directory}/../.coveragerc -d ${current_directory}/htmlcov

${python_command} ${current_directory}/scripts/unittest_summary.py ${current_directory}/run.log 2>&1 | tee -a ${current_directory}/run.log

exit ${rc}
