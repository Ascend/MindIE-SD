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
from mindiesd.utils.get_platform import is_a5_device, get_npu_device, NPUDevice


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


def cpu_rotary_reference(x, cos, sin, rotary_mode, device_family=None):
    """Model the native arithmetic using independent CPU operations.

    CANN ApplyRotaryPosEmbV2 documents cast=0 for A2/A3 FP16 and cast=1
    for BF16. A2 FP16 stage rounding was verified bitwise over ten seeds
    and both supported layouts; A5 retains the verified FP32 reference.
    Copy before dispatch because the native operation overwrites x in place.
    """
    value = x.detach().to(device="cpu", dtype=torch.float32)
    cosine = cos.detach().to(device="cpu", dtype=torch.float32)
    sine = sin.detach().to(device="cpu", dtype=torch.float32)
    family = get_npu_device() if device_family is None else device_family
    first, second = value * cosine, rotate(value, rotary_mode) * sine
    if x.dtype == torch.float16 and family in (NPUDevice.A2, NPUDevice.A3):
        first, second = first.to(x.dtype).float(), second.to(x.dtype).float()
    return (first + second).to(x.dtype)


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
        expected_query = cpu_rotary_reference(query, cos, sin, rotary_mode)
        expected_key = cpu_rotary_reference(key, cos, sin, rotary_mode)
        query_ptr = query.data_ptr()
        key_ptr = key.data_ptr()

        query_out, key_out = apply_rotary_pos_emb(query, key, cos, sin, layout=layout, rotary_mode=rotary_mode)

        self.assertEqual(query_out.data_ptr(), query_ptr)
        self.assertEqual(key_out.data_ptr(), key_ptr)
        rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(query_out.cpu(), expected_query, rtol=rtol, atol=atol)
        torch.testing.assert_close(key_out.cpu(), expected_key, rtol=rtol, atol=atol)

    def test_half_mode_supported_layouts(self):
        for layout in ("BSND", "TND"):
            with self.subTest(layout=layout):
                self.assert_case(layout, "half", torch.float16)

    def test_float32(self):
        self.assert_case("BSND", "half", torch.float32)

    def test_default_arguments(self):
        query, key, cos, sin = make_inputs("BSND", torch.float16, self.device)
        expected_query = cpu_rotary_reference(query, cos, sin, "half")
        expected_key = cpu_rotary_reference(key, cos, sin, "half")

        query_out, key_out = apply_rotary_pos_emb(query, key, cos, sin)

        torch.testing.assert_close(query_out.cpu(), expected_query, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(key_out.cpu(), expected_key, rtol=1e-3, atol=1e-3)

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

    @unittest.skipUnless(
        get_npu_device() in (NPUDevice.A2, NPUDevice.A3, NPUDevice.A5),
        "BF16 ApplyRotaryPosEmbV2 requires A2, A3 or A5.",
    )
    def test_bfloat16_supported_devices(self):
        for layout in ("BSND", "TND"):
            with self.subTest(layout=layout):
                self.assert_case(layout, "half", torch.bfloat16)
