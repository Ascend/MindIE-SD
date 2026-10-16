#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
from importlib import import_module

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)
custom_op_path1 = os.path.join(parent_dir, "mindiesd/ops/vendors/aie_ascendc")
custom_op_path2 = os.path.join(parent_dir, "mindiesd/ops/vendors/customize")
old_custom_op_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
new_custom_op_path = f"{custom_op_path1}:{custom_op_path2}:{old_custom_op_path}"
os.environ["ASCEND_CUSTOM_OPP_PATH"] = new_custom_op_path

if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
    from utils.utils.torch_npu_mock import mock_torch_npu  # pylint: disable=no-name-in-module

    mock_torch_npu()


def load_tests_from_files(folder_path):
    test_suite = unittest.TestSuite()
    for foldername, _, filenames in os.walk(folder_path):
        if "UT" in foldername:
            continue
        for filename in filenames:
            if re.match(r"^test_", filename) and re.search(r"py$", filename):
                file_path = os.path.join(folder_path, foldername, filename)
                module_name = os.path.splitext(os.path.relpath(file_path))[0].replace(os.path.sep, '.')
                module = import_module(f'{module_name}')
                tests = unittest.TestLoader().loadTestsFromModule(module)
                test_suite.addTests(tests)
    return test_suite


def load_tests_from_names(folder_path, pattern):
    """按 | 分隔的测试用例文件名进行精准匹配（不含 .py 后缀），加载命中的测试用例。

    pattern 形如 "test_a|test_b"；只加载 basename 恰好等于其中某个名字的文件，
    不做模糊/子串匹配。名字末尾的 .py 会被自动忽略。
    """
    names = []
    for part in pattern.split("|"):
        name = part.strip()
        if not name:
            continue
        if name.endswith(".py"):
            name = name[:-3]
        names.append(name)
    if not names:
        print("No test case name provided: {!r}".format(pattern), file=sys.stderr)
        sys.exit(1)

    name_set = set(names)
    test_suite = unittest.TestSuite()
    matched = []
    found = set()
    for foldername, _, filenames in os.walk(folder_path):
        if "UT" in foldername:
            continue
        for filename in filenames:
            if re.match(r"^test_", filename) and re.search(r"py$", filename):
                base = os.path.splitext(filename)[0]
                if base in name_set:
                    found.add(base)
                    matched.append(filename)
                    file_path = os.path.join(folder_path, foldername, filename)
                    module_name = os.path.splitext(os.path.relpath(file_path))[0].replace(os.path.sep, '.')
                    try:
                        module = import_module(module_name)
                    except Exception as e:  # pylint: disable=broad-except
                        print("Failed to import {}: {}: {}".format(filename, type(e).__name__, e), file=sys.stderr)
                        continue
                    tests = unittest.TestLoader().loadTestsFromModule(module)
                    test_suite.addTests(tests)
    not_found = name_set - found
    if not_found:
        print("No matching test files found: {}".format(sorted(not_found)), file=sys.stderr)
    print("Exactly matched {} test file(s): {}".format(len(matched), matched))
    if not found:
        print("Error: none of the specified test names matched any file; aborting.", file=sys.stderr)
        sys.exit(1)
    return test_suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindIE-SD 单元测试运行入口")
    parser.add_argument("test_file", nargs="?", default=None, help="单个测试文件路径")
    parser.add_argument(
        "--precision_test",
        dest="names",
        default=None,
        metavar="NAMES",
        help="精准匹配；按 | 分隔的测试用例文件名（不含 .py），只运行文件名恰好等于其中某个名字的测试用例",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    runner = unittest.TextTestRunner()

    if args.names:
        suite = load_tests_from_names(current_dir, args.names)
    elif args.test_file:
        rel_path = os.path.relpath(args.test_file, current_dir)
        test_module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
        test_module = import_module(test_module_name)
        suite = unittest.TestLoader().loadTestsFromModule(test_module)
    else:
        suite = load_tests_from_files(current_dir)

    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
