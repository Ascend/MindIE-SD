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


class TestAddNormValidation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, 4, dtype=torch.float16)
        self.residual = torch.randn_like(self.x)
        self.weight = torch.randn(4, dtype=self.x.dtype)
        self.bias = torch.randn(4, dtype=self.x.dtype)

    def test_add_layer_norm_reference(self):
        output, residual_output = add_layer_norm(
            self.x,
            self.residual,
            self.weight,
            self.bias,
            fused=False,
        )
        expected_residual = self.x + self.residual
        expected_output = F.layer_norm(
            expected_residual.float(),
            (4,),
            self.weight.float(),
            self.bias.float(),
            1e-5,
        ).to(self.x.dtype)

        self.assertTrue(torch.equal(residual_output, expected_residual))
        self.assertTrue(torch.equal(output, expected_output))

    def test_add_rms_norm_reference(self):
        output, residual_output = add_rms_norm(
            self.x,
            self.residual,
            self.weight,
            fused=False,
        )
        expected_residual = self.x + self.residual
        expected_fp32 = expected_residual.float()
        variance = expected_fp32.pow(2).mean(dim=-1, keepdim=True)
        expected_output = (expected_fp32 * torch.rsqrt(variance + 1e-6) * self.weight.float()).to(self.x.dtype)

        self.assertTrue(torch.equal(residual_output, expected_residual))
        self.assertTrue(torch.equal(output, expected_output))

    def test_rejects_invalid_common_inputs(self):
        cases = [
            (
                "non-tensor input",
                lambda: add_rms_norm("invalid", self.residual, self.weight, fused=False),
                "input x must be torch.Tensor",
            ),
            (
                "shape mismatch",
                lambda: add_rms_norm(self.x, self.residual[:, :2], self.weight, fused=False),
                "shape of x",
            ),
            (
                "dtype mismatch",
                lambda: add_rms_norm(self.x, self.residual.float(), self.weight, fused=False),
                "device and dtype of x and residual",
            ),
            (
                "unsupported dtype",
                lambda: add_rms_norm(self.x.int(), self.residual.int(), self.weight.int(), fused=False),
                "input dtype must be one of",
            ),
            (
                "invalid rank",
                lambda: add_rms_norm(torch.ones(4), torch.ones(4), torch.ones(4), fused=False),
                "input dimension must be between 2 and 8",
            ),
            (
                "weight shape mismatch",
                lambda: add_rms_norm(self.x, self.residual, torch.ones(3), fused=False),
                "shape of weight",
            ),
            (
                "weight dtype mismatch",
                lambda: add_rms_norm(self.x, self.residual, self.weight.float(), fused=False),
                "device and dtype of weight",
            ),
            (
                "bias shape mismatch",
                lambda: add_layer_norm(self.x, self.residual, self.weight, torch.ones(3), fused=False),
                "shape of bias",
            ),
            (
                "non-bool fused",
                lambda: add_rms_norm(self.x, self.residual, self.weight, fused=1),
                "input fused must be bool",
            ),
        ]
        for name, function, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ParametersInvalid, message):
                    function()

    def test_rejects_invalid_epsilon(self):
        for eps in (True, 0.0, -1e-6, float("nan"), float("inf"), -float("inf")):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(ParametersInvalid, "finite positive number"):
                    add_rms_norm(
                        self.x,
                        self.residual,
                        self.weight,
                        eps=eps,
                        fused=False,
                    )

    def test_fused_path_requires_npu(self):
        with self.assertRaisesRegex(ParametersInvalid, "only supports NPU tensors"):
            add_layer_norm(self.x, self.residual, self.weight, self.bias)
        with self.assertRaisesRegex(ParametersInvalid, "only supports NPU tensors"):
            add_rms_norm(self.x, self.residual, self.weight)


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


if __name__ == "__main__":
    unittest.main()
