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

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# Set ASCEND_CUSTOM_OPP_PATH (same as run.py)
_custom_op_path1 = os.path.join(PROJECT_ROOT, "mindiesd/ops/vendors/aie_ascendc")
_custom_op_path2 = os.path.join(PROJECT_ROOT, "mindiesd/ops/vendors/customize")
_old_custom_op_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
if "aie_ascendc" not in _old_custom_op_path:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{_custom_op_path1}:{_custom_op_path2}:{_old_custom_op_path}"
