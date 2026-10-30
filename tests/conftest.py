# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Pytest conftest: configure sys.path so all test imports resolve correctly."""

import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _initial_test_mode():
    for index, arg in enumerate(sys.argv):
        if arg.startswith("--test-mode="):
            return arg.split("=", 1)[1]
        if arg == "--test-mode" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return "NPU" if os.environ.get("MINDIE_TEST_MODE") == "NPU" else "CPU"


# Child conftest modules can import production code before pytest_configure runs.
if _initial_test_mode() == "CPU":
    os.environ["MINDIE_TEST_MODE"] = "CPU"
    from utils.utils.torch_npu_mock import mock_torch_npu

    mock_torch_npu()

# Set ASCEND_CUSTOM_OPP_PATH (same as mindiesd.env)
_custom_op_path1 = os.path.join(PROJECT_ROOT, "mindiesd/ops/vendors/aie_ascendc")
_custom_op_path2 = os.path.join(PROJECT_ROOT, "mindiesd/ops/vendors/customize")
_old_custom_op_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
if "aie_ascendc" not in _old_custom_op_path:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{_custom_op_path1}:{_custom_op_path2}:{_old_custom_op_path}"


def pytest_addoption(parser):
    parser.addoption(
        "--test-mode",
        choices=("CPU", "NPU"),
        default=_initial_test_mode(),
        help="Collect CPU unit tests or real operator tests in separate processes.",
    )


def pytest_configure(config):
    mode = config.getoption("--test-mode")
    os.environ["MINDIE_TEST_MODE"] = mode
    config.addinivalue_line("markers", "operator: executes a real device operator")
    if mode == "CPU":
        mock_torch_npu()
    else:
        try:
            import torch_npu

            available = torch_npu.npu.is_available()
        except (ImportError, RuntimeError, OSError) as error:
            raise pytest.UsageError(f"NPU operator tests require a working device backend: {error}") from error
        if not available:
            raise pytest.UsageError("NPU operator tests require an available device")


def pytest_ignore_collect(collection_path, config):
    try:
        relative = collection_path.relative_to(TESTS_DIR)
    except ValueError:
        return None
    if not relative.parts:
        return None
    is_operator = relative.parts[0] == "ops"
    mode = config.getoption("--test-mode")
    return is_operator if mode == "CPU" else not is_operator


def pytest_collection_modifyitems(config, items):
    mode = config.getoption("--test-mode")
    for item in items:
        relative = item.path.relative_to(TESTS_DIR)
        is_operator = relative.parts[0] == "ops"
        if is_operator != (mode == "NPU"):
            raise pytest.UsageError(f"{item.nodeid} does not belong to the {mode} test suite")
        if is_operator:
            item.add_marker(pytest.mark.operator)
