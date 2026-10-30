#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest import mock

from mindiesd.utils import get_platform


class TestGetPlatform(unittest.TestCase):
    def setUp(self):
        self.original_platform = get_platform.PLATFORM
        get_platform.PLATFORM = None

    def tearDown(self):
        get_platform.PLATFORM = self.original_platform

    def test_no_device_returns_undefined(self):
        with mock.patch.object(get_platform.torch_npu.npu, "device_count", return_value=0):
            self.assertEqual(get_platform.get_npu_device(), get_platform.NPUDevice.UNDEFINED)

    def test_soc_version_maps_to_supported_platforms(self):
        cases = [
            (200, get_platform.NPUDevice.Duo),
            (205, get_platform.NPUDevice.Duo),
            (220, get_platform.NPUDevice.A2),
            (225, get_platform.NPUDevice.A2),
            (250, get_platform.NPUDevice.A3),
            (256, get_platform.NPUDevice.A3),
            (260, get_platform.NPUDevice.A5),
        ]
        for soc_version, expected in cases:
            with self.subTest(soc_version=soc_version):
                get_platform.PLATFORM = None
                with (
                    mock.patch.object(get_platform.torch_npu.npu, "device_count", return_value=1),
                    mock.patch.object(get_platform.torch_npu.npu, "get_soc_version", return_value=soc_version),
                ):
                    self.assertEqual(get_platform.get_npu_device(), expected)

    def test_unknown_soc_version_returns_undefined(self):
        with (
            mock.patch.object(get_platform.torch_npu.npu, "device_count", return_value=1),
            mock.patch.object(get_platform.torch_npu.npu, "get_soc_version", return_value=999),
        ):
            self.assertEqual(get_platform.get_npu_device(), get_platform.NPUDevice.UNDEFINED)

    def test_soc_query_error_returns_undefined(self):
        with (
            mock.patch.object(get_platform.torch_npu.npu, "device_count", return_value=1),
            mock.patch.object(
                get_platform.torch_npu.npu, "get_soc_version", side_effect=RuntimeError("driver unavailable")
            ),
        ):
            self.assertEqual(get_platform.get_npu_device(), get_platform.NPUDevice.UNDEFINED)

    def test_platform_result_is_cached(self):
        with (
            mock.patch.object(get_platform.torch_npu.npu, "device_count", return_value=1),
            mock.patch.object(get_platform.torch_npu.npu, "get_soc_version", return_value=260) as get_soc_version,
        ):
            self.assertEqual(get_platform.get_npu_device(), get_platform.NPUDevice.A5)
            get_soc_version.return_value = 220
            self.assertEqual(get_platform.get_npu_device(), get_platform.NPUDevice.A5)
            self.assertEqual(get_soc_version.call_count, 1)

    def test_is_a5_device(self):
        with (
            mock.patch.object(get_platform.torch_npu.npu, "device_count", return_value=1),
            mock.patch.object(get_platform.torch_npu.npu, "get_soc_version", return_value=260),
        ):
            self.assertTrue(get_platform.is_a5_device())

        get_platform.PLATFORM = get_platform.NPUDevice.A2
        self.assertFalse(get_platform.is_a5_device())


if __name__ == "__main__":
    unittest.main()
