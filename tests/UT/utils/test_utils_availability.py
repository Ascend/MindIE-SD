#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from unittest.mock import patch

from mindiesd.utils import utils


def test_availability_missing_packages():
    for values in ((None,), (object(), None)):
        with patch.object(utils.importlib.util, "find_spec", side_effect=values):
            assert utils.is_npu_available() is False


def test_availability_driver_failure_and_explicit_devices():
    with patch.object(utils.importlib.util, "find_spec", return_value=object()):
        with patch.object(utils.torch.npu, "device_count", side_effect=RuntimeError("driver missing")):
            assert utils.is_npu_available() is False
        for available in (True, False):
            with (
                patch.object(utils.torch.npu, "device_count", return_value=1),
                patch.object(utils.torch.npu, "is_available", return_value=available),
            ):
                assert utils.is_npu_available() is available
