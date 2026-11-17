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
from types import SimpleNamespace
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

    def test_cann_query_returns_original_version_and_forwards_component(self):
        query = mock.Mock(return_value="9.2.0RC1.alpha001")
        with mock.patch.object(get_platform.torch_npu, "utils", SimpleNamespace(get_cann_version=query)):
            self.assertEqual(get_platform.get_cann_version(), "9.2.0RC1.alpha001")
            self.assertEqual(get_platform.get_cann_version("RUNTIME"), "9.2.0RC1.alpha001")
        self.assertEqual(query.call_args_list, [mock.call("CANN"), mock.call("RUNTIME")])

    def test_missing_cann_query_raises_without_using_native_binding(self):
        native_query = mock.Mock(return_value="9.1.0")
        with (
            mock.patch.object(get_platform.torch_npu, "utils", SimpleNamespace()),
            mock.patch.object(get_platform.torch_npu, "_C", SimpleNamespace(_get_cann_version=native_query)),
        ):
            with self.assertRaises(AttributeError):
                get_platform.get_cann_version()
        native_query.assert_not_called()

    def test_cann_query_failure_propagates(self):
        for error in (
            AttributeError("missing API"),
            RuntimeError("unsupported API"),
            OSError("missing library"),
            ValueError("invalid component"),
        ):
            with self.subTest(error=error):
                query = mock.Mock(side_effect=error)
                with mock.patch.object(get_platform.torch_npu, "utils", SimpleNamespace(get_cann_version=query)):
                    with self.assertRaises(type(error)) as raised:
                        get_platform.get_cann_version()
                    self.assertIs(raised.exception, error)

    def test_cann_query_returns_result_unchanged(self):
        for result in ("", "9.2.0\n"):
            with self.subTest(result=result):
                query = mock.Mock(return_value=result)
                with mock.patch.object(get_platform.torch_npu, "utils", SimpleNamespace(get_cann_version=query)):
                    self.assertEqual(get_platform.get_cann_version(), result)

    def test_cann_version_comparison(self):
        cases = (
            ("9.1.0", False),
            ("9.2.0", True),
            ("9.2.1", True),
            ("9.10.0", True),
            ("10.0.0", True),
            ("9.2.RC1", False),
            ("9.2.0RC1", False),
            ("9.2.0.alpha001", False),
            ("9.2.0RC1.alpha001", False),
            ("", False),
            ("unknown", False),
        )
        for version, expected in cases:
            with self.subTest(version=version):
                with mock.patch.object(get_platform, "get_cann_version", return_value=version) as query:
                    self.assertEqual(get_platform.is_cann_version_at_least((9, 2, 0)), expected)
                query.assert_called_once_with("CANN")

    def test_cann_version_comparison_propagates_query_failure(self):
        for error in (AttributeError("missing API"), RuntimeError("query failed")):
            with self.subTest(error=error):
                with mock.patch.object(get_platform, "get_cann_version", side_effect=error):
                    with self.assertRaises(type(error)) as raised:
                        get_platform.is_cann_version_at_least((9, 2, 0))
                    self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
