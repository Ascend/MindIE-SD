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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    suite = load_tests_from_files(current_dir)
    runner = unittest.TextTestRunner()
    runner.run(suite)
