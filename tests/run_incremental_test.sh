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

# 默认参数
BASE_BRANCH="master"
INCLUDE_STAGED=true
INCLUDE_UNSTAGED=true
DRY_RUN=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --base|-b)
            BASE_BRANCH="$2"
            echo "Set base branch to: $BASE_BRANCH"
            shift 2
            ;;
        --no-staged)
            INCLUDE_STAGED=false
            echo "Exclude staged changes from incremental tests."
            shift
            ;;
        --no-unstaged)
            INCLUDE_UNSTAGED=false
            echo "Exclude unstaged changes from incremental tests."
            shift
            ;;
        --dry-run|-n)
            DRY_RUN=true
            echo "Dry run mode - only show test plan."
            shift
            ;;
        --help|-h)
            echo "Usage: bash run_incremental_test.sh [OPTIONS]"
            echo ""
            echo "增量测试脚本 - 仅运行与本次变更相关的测试用例"
            echo ""
            echo "Options:"
            echo "  --base, -b BRANCH       Set base branch for comparison (default: master)."
            echo "  --no-staged             Exclude staged changes from incremental tests."
            echo "  --no-unstaged           Exclude unstaged changes from incremental tests."
            echo "  --dry-run, -n           Dry run mode - only show test plan without execution."
            echo "  --help, -h              Show this help message and exit."
            echo ""
            echo "Examples:"
            echo "  # Run incremental tests based on master branch"
            echo "  bash run_incremental_test.sh"
            echo ""
            echo "  # Run incremental tests based on dev branch"
            echo "  bash run_incremental_test.sh --base dev"
            echo ""
            echo "  # Show incremental test plan without execution"
            echo "  bash run_incremental_test.sh --dry-run"
            echo ""
            echo "  # Run incremental tests with only committed changes (exclude staged and unstaged)"
            echo "  bash run_incremental_test.sh --no-staged --no-unstaged"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

if command -v python3 &> /dev/null; then
    python_command=python3
else
    python_command=python
fi


# 构建Python命令参数
PYTHON_ARGS="--base-branch $BASE_BRANCH"

if [ "$INCLUDE_STAGED" = false ]; then
    PYTHON_ARGS="$PYTHON_ARGS --no-staged"
fi

if [ "$INCLUDE_UNSTAGED" = false ]; then
    PYTHON_ARGS="$PYTHON_ARGS --no-unstaged"
fi

if [ "$DRY_RUN" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --dry-run"
fi

echo ""
echo "========================================"
echo "🚀 启动增量测试"
echo "========================================"
echo "Base Branch: $BASE_BRANCH"
echo "Include Staged: $INCLUDE_STAGED"
echo "Include Unstaged: $INCLUDE_UNSTAGED"
echo "Dry Run: $DRY_RUN"
echo "========================================"
echo ""

current_directory=$(dirname "$(readlink -f "$0")")
# 运行增量测试
if [ "$DRY_RUN" = true ]; then
    # Dry run模式：只显示测试计划，不生成覆盖率报告
    ${python_command} ${current_directory}/run_incremental.py $PYTHON_ARGS 2>&1 | tee ${current_directory}/incremental_test.log
else
    # 正常模式：运行测试并生成覆盖率报告
    ${python_command} -m coverage run --branch --source=../mindiesd ${current_directory}/run_incremental.py $PYTHON_ARGS 2>&1 | tee ${current_directory}/incremental_test.log
    
    ${python_command} -m coverage report
    ${python_command} -m coverage xml -o ${current_directory}/coverage.xml

    ${python_command} ${current_directory}/scripts/unittest_summary.py ${current_directory}/incremental_test.log

fi