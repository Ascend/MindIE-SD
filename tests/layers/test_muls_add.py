#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import unittest

import torch

from mindiesd.layers import muls_add


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestMulsAdd(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_basic_result_float32(self):
        x = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        y = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        x_orig = x.clone()
        y_orig = y.clone()

        result = muls_add(x, y, 1.5)
        expected = x_orig * 1.5 + y_orig

        self.assertTrue(
            torch.allclose(result, expected, atol=1e-5),
            "muls_add float32 output mismatch",
        )

    def test_basic_result_float16(self):
        x = torch.randn(4, 4096, dtype=torch.float16, device="npu")
        y = torch.randn(4, 4096, dtype=torch.float16, device="npu")

        result = muls_add(x, y, 2.0)
        expected = x * 2.0 + y

        self.assertTrue(
            torch.allclose(result, expected, atol=1e-2),
            "muls_add float16 output mismatch",
        )

    def test_basic_result_bfloat16(self):
        x = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
        y = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")

        result = muls_add(x, y, 0.5)
        expected = x * 0.5 + y

        self.assertTrue(
            torch.allclose(result, expected, atol=1e-1),
            "muls_add bfloat16 output mismatch",
        )

    def test_scale_variants(self):
        x = torch.randn(2, 2048, dtype=torch.float32, device="npu")
        y = torch.randn(2, 2048, dtype=torch.float32, device="npu")

        for scale in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0]:
            with self.subTest(scale=scale):
                result = muls_add(x, y, scale)
                expected = x * scale + y
                self.assertTrue(
                    torch.allclose(result, expected, atol=1e-5),
                    f"scale={scale} mismatch",
                )

    def test_shapes(self):
        for shape in [
            (1, 1),
            (1, 1024),
            (2, 2048),
            (4, 4096),
            (32, 8192),
        ]:
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float32, device="npu")
                y = torch.randn(shape, dtype=torch.float32, device="npu")
                result = muls_add(x, y, 1.5)
                expected = x * 1.5 + y
                self.assertTrue(
                    torch.allclose(result, expected, atol=1e-5),
                    f"shape={shape} mismatch",
                )

    def test_shape_3d(self):
        shape = (2, 32, 4096)
        x = torch.randn(shape, dtype=torch.float32, device="npu")
        y = torch.randn(shape, dtype=torch.float32, device="npu")

        result = muls_add(x, y, 1.5)
        expected = x * 1.5 + y

        self.assertTrue(
            torch.allclose(result, expected, atol=1e-5),
            f"3D shape={shape} mismatch",
        )

    def test_no_inplace_modification(self):
        x = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        y = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        x_orig = x.clone()
        y_orig = y.clone()

        _ = muls_add(x, y, 1.5)

        self.assertTrue(torch.equal(x, x_orig), "x was modified in-place")
        self.assertTrue(torch.equal(y, y_orig), "y was modified in-place")

    def test_result_device(self):
        x = torch.randn(4, 4096, device="npu")
        y = torch.randn(4, 4096, device="npu")

        result = muls_add(x, y, 1.5)

        self.assertEqual(
            result.device.type,
            "npu",
            "result should be on NPU device",
        )

    def test_result_dtype(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                x = torch.randn(4, 4096, dtype=dtype, device="npu")
                y = torch.randn(4, 4096, dtype=dtype, device="npu")
                result = muls_add(x, y, 1.5)
                self.assertEqual(
                    result.dtype,
                    dtype,
                    f"result dtype mismatch for {dtype}",
                )

    def test_result_shape(self):
        x = torch.randn(4, 4096, device="npu")
        y = torch.randn(4, 4096, device="npu")

        result = muls_add(x, y, 1.5)

        self.assertEqual(result.shape, x.shape, "result shape mismatch")

    def test_scale_zero(self):
        x = torch.randn(4, 4096, device="npu")
        y = torch.randn(4, 4096, device="npu")
        y_orig = y.clone()

        result = muls_add(x, y, 0.0)

        self.assertTrue(
            torch.allclose(result, y_orig, atol=1e-5),
            "scale=0 should return y",
        )

    def test_scale_one(self):
        x = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        y = torch.randn(4, 4096, dtype=torch.float32, device="npu")

        result = muls_add(x, y, 1.0)
        expected = x + y

        self.assertTrue(
            torch.allclose(result, expected, atol=1e-5),
            "scale=1 should return x+y",
        )

    def test_input_unchanged_after_call(self):
        x = torch.randn(4, 4096, device="npu")
        y = torch.randn(4, 4096, device="npu")
        x_orig = x.clone()
        y_orig = y.clone()

        _ = muls_add(x, y, 1.5)
        _ = muls_add(x, y, 0.0)
        _ = muls_add(x, y, -2.0)

        self.assertTrue(torch.equal(x, x_orig), "x modified across multiple calls")
        self.assertTrue(torch.equal(y, y_orig), "y modified across multiple calls")


if __name__ == "__main__":
    unittest.main()
