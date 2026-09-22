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

import os
import unittest

import torch

from mindiesd.layers import apply_rotary_pos_emb
from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import is_a5_device


def rotate(x: torch.Tensor, rotary_mode: str) -> torch.Tensor:
    if rotary_mode == "half":
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    if rotary_mode == "interleave":
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    if rotary_mode == "quarter":
        x1, x2, x3, x4 = x.chunk(4, dim=-1)
        return torch.cat((-x2, x1, -x4, x3), dim=-1)
    raise ValueError(f"unsupported rotary mode: {rotary_mode}")


def make_inputs(layout: str, dtype: torch.dtype, device: torch.device):
    shapes = {
        "BSND": ((2, 8, 6, 64), (2, 8, 2, 64), (2, 8, 1, 64)),
        "SBND": ((8, 2, 6, 64), (8, 2, 2, 64), (8, 2, 1, 64)),
        "BNSD": ((2, 6, 8, 64), (2, 2, 8, 64), (2, 1, 8, 64)),
        "TND": ((16, 6, 64), (16, 2, 64), (16, 1, 64)),
    }
    query_shape, key_shape, angle_shape = shapes[layout]
    query = torch.randn(query_shape, device=device, dtype=dtype)
    key = torch.randn(key_shape, device=device, dtype=dtype)
    angles = torch.randn(angle_shape, device=device, dtype=dtype)
    return query, key, angles.cos(), angles.sin()


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestApplyRotaryPosEmb(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

    def assert_case(self, layout: str, rotary_mode: str, dtype: torch.dtype):
        query, key, cos, sin = make_inputs(layout, dtype, self.device)
        expected_query = query * cos + rotate(query, rotary_mode) * sin
        expected_key = key * cos + rotate(key, rotary_mode) * sin
        query_ptr = query.data_ptr()
        key_ptr = key.data_ptr()

        query_out, key_out = apply_rotary_pos_emb(
            query, key, cos, sin, layout=layout, rotary_mode=rotary_mode
        )

        self.assertEqual(query_out.data_ptr(), query_ptr)
        self.assertEqual(key_out.data_ptr(), key_ptr)
        rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(query_out, expected_query, rtol=rtol, atol=atol)
        torch.testing.assert_close(key_out, expected_key, rtol=rtol, atol=atol)

    def test_half_mode_supported_layouts(self):
        for layout in ("BSND", "TND"):
            with self.subTest(layout=layout):
                self.assert_case(layout, "half", torch.float16)

    def test_float32(self):
        self.assert_case("BSND", "half", torch.float32)

    def test_default_arguments(self):
        query, key, cos, sin = make_inputs("BSND", torch.float16, self.device)
        expected_query = query * cos + rotate(query, "half") * sin
        expected_key = key * cos + rotate(key, "half") * sin

        query_out, key_out = apply_rotary_pos_emb(query, key, cos, sin)

        torch.testing.assert_close(query_out, expected_query, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(key_out, expected_key, rtol=1e-3, atol=1e-3)

    @unittest.skipUnless(is_a5_device(), "SBND and BNSD layouts require an A5 (950) NPU.")
    def test_a5_extended_layouts(self):
        for layout in ("SBND", "BNSD"):
            with self.subTest(layout=layout):
                self.assert_case(layout, "half", torch.float16)

    @unittest.skipUnless(is_a5_device(), "Extended rotary modes require an A5 (950) NPU.")
    def test_a5_extended_rotary_modes(self):
        for rotary_mode in ("interleave", "quarter"):
            with self.subTest(rotary_mode=rotary_mode):
                self.assert_case("BSND", rotary_mode, torch.float16)

    @unittest.skipUnless(is_a5_device(), "BF16 ApplyRotaryPosEmbV2 requires an A5 (950) NPU.")
    def test_a5_bfloat16(self):
        self.assert_case("BSND", "half", torch.bfloat16)

    def test_invalid_layout(self):
        query, key, cos, sin = make_inputs("BSND", torch.float16, self.device)
        with self.assertRaisesRegex(ParametersInvalid, "layout must be"):
            apply_rotary_pos_emb(query, key, cos, sin, layout="INVALID", rotary_mode="half")

    def test_invalid_rotary_mode(self):
        query, key, cos, sin = make_inputs("BSND", torch.float16, self.device)
        with self.assertRaisesRegex(ParametersInvalid, "rotary_mode must be"):
            apply_rotary_pos_emb(query, key, cos, sin, layout="BSND", rotary_mode="invalid")


if __name__ == "__main__":
    unittest.main()
