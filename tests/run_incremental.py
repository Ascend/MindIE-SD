#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
import os
import re
import sys
import argparse
from typing import List, Set, Optional
from importlib import import_module
from incremental_test_finder import IncrementalTestFinder

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(script_dir)
sys.path.append(parent_dir)
custom_op_path1 = os.path.join(parent_dir, "mindiesd/ops/vendors/aie_ascendc")
custom_op_path2 = os.path.join(parent_dir, "mindiesd/ops/vendors/customize")
old_custom_op_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
new_custom_op_path = f"{custom_op_path1}:{custom_op_path2}:{old_custom_op_path}"
os.environ["ASCEND_CUSTOM_OPP_PATH"] = new_custom_op_path

def load_tests_from_list(test_files: List[str], base_path: str):
    """
    从文件列表加载测试用例
    
    Args:
        test_files: 测试文件路径列表
        base_path: 基础路径
    
    Returns:
        unittest.TestSuite
    """
    test_suite = unittest.TestSuite()
    
    for test_file in test_files:
        # 处理不同格式的路径
        if test_file.startswith("tests/"):
            test_file = test_file[6:]  # 去掉 "tests/" 前缀
        
        file_path = os.path.join(base_path, test_file)
        
        if not os.path.isfile(file_path):
            print(f"Warning: Test file not found: {file_path}")
            continue
        
        filename = os.path.basename(file_path)
        if not (re.match(r"^test_", filename) and filename.endswith(".py")):
            print(f"Warning: Not a valid test file: {filename}")
            continue
        
        try:
            module_name = os.path.splitext(test_file)[0].replace(os.path.sep, '.')
            module = import_module(f'{module_name}')
            tests = unittest.TestLoader().loadTestsFromModule(module)
            test_suite.addTests(tests)
            print(f"  ✅ Loaded: {test_file}")
        except Exception as e:
            print(f"  ❌ Failed to load {test_file}: {e}")
    
    return test_suite


def run_incremental_tests(base_branch: str = "main", 
                         include_staged: bool = True,
                         include_unstaged: bool = True,
                         dry_run: bool = False):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建增量测试查找器
    finder = IncrementalTestFinder(
        repo_root=os.path.dirname(current_dir),
        base_branch=base_branch
    )
    
    # 获取增量测试列表
    tests, deletion_info, changes = finder.get_incremental_tests(
        since_ref=base_branch,
        include_staged=include_staged,
        include_unstaged=include_unstaged
    )
    
    # 打印测试计划
    finder.print_test_plan(tests, deletion_info, changes)

    if dry_run:
        return None
    
    if not tests:
        print("\n没有需要运行的测试。")
        return unittest.TestSuite()
    
    # 加载并返回测试套件
    print("\n🚀 正在加载测试用例...\n")
    test_files = [t.replace("tests/", "") for t in tests]
    return load_tests_from_list(test_files, current_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MindIE SD 增量UT测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--base-branch", "-b", default="master",
                       help="增量测试的基分支 (默认: master)")
    parser.add_argument("--no-staged", action="store_true",
                       help="增量测试时不包含暂存区变更")
    parser.add_argument("--no-unstaged", action="store_true",
                       help="增量测试时不包含未暂存变更")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出模式")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="仅显示测试计划，不执行测试")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 增量测试模式
    suite = run_incremental_tests(
        base_branch=args.base_branch,
        include_staged=not args.no_staged,
        include_unstaged=not args.no_unstaged,
        dry_run=args.dry_run
    )
    if suite is None:
        return
    if isinstance(suite, unittest.TestSuite) and suite.countTestCases() == 0:
        return
    
    print(f"\n共加载 {suite.countTestCases()} 个测试用例\n")
    
    # 不运行测试
    if args.dry_run:
        return
    
    # 设置测试运行器
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # 运行测试
    runner.run(suite)


if __name__ == "__main__":
    main()
