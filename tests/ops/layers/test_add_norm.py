#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
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
import torch.nn.functional as F

from mindiesd import add_layer_norm, add_rms_norm
from mindiesd.utils import ParametersInvalid


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestAddNormNPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_fused_outputs_match_reference(self):
        tolerances = {
            torch.float16: (1e-3, 4e-3),
            torch.bfloat16: (1e-2, 4e-2),
            torch.float32: (1e-5, 1e-5),
        }
        for dtype, (rtol, atol) in tolerances.items():
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 8, 64, device="npu", dtype=dtype)
                residual = torch.randn_like(x)
                weight = torch.randn(64, device="npu", dtype=dtype)
                bias = torch.randn(64, device="npu", dtype=dtype)
                x_before = x.clone()
                residual_before = residual.clone()
                expected_residual = x + residual
                expected_layer = F.layer_norm(
                    expected_residual.float(),
                    (64,),
                    weight.float(),
                    bias.float(),
                    1e-5,
                ).to(dtype)
                expected_fp32 = expected_residual.float()
                variance = expected_fp32.pow(2).mean(dim=-1, keepdim=True)
                expected_rms = (expected_fp32 * torch.rsqrt(variance + 1e-6) * weight.float()).to(dtype)

                layer_output, layer_residual = add_layer_norm(x, residual, weight, bias)
                rms_output, rms_residual = add_rms_norm(x, residual, weight)

                self.assertTrue(torch.equal(layer_residual, expected_residual))
                self.assertTrue(torch.equal(rms_residual, expected_residual))
                self.assertTrue(
                    torch.allclose(layer_output, expected_layer, rtol=rtol, atol=atol),
                    f"Add+LayerNorm output mismatch for {dtype}",
                )
                self.assertTrue(
                    torch.allclose(rms_output, expected_rms, rtol=rtol, atol=atol),
                    f"Add+RMSNorm output mismatch for {dtype}",
                )
                self.assertTrue(torch.equal(x, x_before), "x was modified in-place")
                self.assertTrue(
                    torch.equal(residual, residual_before),
                    "residual was modified in-place",
                )
