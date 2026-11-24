#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the License at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import unittest

import torch

# Importing this module registers the torch.ops.mindiesd operators below.
import mindiesd.layers.residual_gate_add


def _reference(input_tensor, tensor1, tensor2):
    """Independent FP32 accumulation reference for the fused residual formula."""
    return (input_tensor.float() + tensor1.float() * tensor2.float()).to(input_tensor.dtype)


def _assert_matches_reference(actual, input_tensor, tensor1, tensor2):
    expected = _reference(input_tensor, tensor1, tensor2)
    if input_tensor.dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-5, 1e-6
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestResidualGateAddNPU(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")
        self.y = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")
        self.gate = torch.randn(2, 1, 64, dtype=torch.bfloat16, device="npu")

    def _assert_matches_reference(self, y, gate):
        actual = torch.ops.mindiesd.residual_gate_add(self.x, y, gate)
        _assert_matches_reference(actual, self.x, y, gate)
        self.assertTrue(actual.is_contiguous())

    def test_wan_operand_order(self):
        self._assert_matches_reference(self.y, self.gate)

    def test_flux_operand_order(self):
        self._assert_matches_reference(self.gate, self.y)

    def test_unsupported_full_sized_gate_raises(self):
        with self.assertRaisesRegex(ValueError, "gate must have shape"):
            torch.ops.mindiesd.residual_gate_add(self.x, self.y, torch.randn_like(self.x))


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestQwenResidualGateAddNPU(unittest.TestCase):
    def test_broadcast_contract_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch_size, sequence_length in ((1, 16), (2, 31)):
                with self.subTest(dtype=dtype, batch_size=batch_size, sequence_length=sequence_length):
                    residual = torch.randn(batch_size, sequence_length, 64, dtype=dtype, device="npu")
                    branch = torch.randn_like(residual)
                    gate = torch.randn(batch_size, 1, 64, dtype=dtype, device="npu")

                    actual = torch.ops.mindiesd.qwen_residual_gate_add(residual, branch, gate)
                    _assert_matches_reference(actual, residual, branch, gate)
                    self.assertEqual(actual.shape, residual.shape)
                    self.assertEqual(actual.dtype, residual.dtype)
                    self.assertTrue(actual.is_contiguous())
