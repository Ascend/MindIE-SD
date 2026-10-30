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

import importlib
import os
import unittest
from unittest.mock import patch

import torch

from mindiesd.layers.residual_gate_add import (
    _normalize_triton_inputs,
    qwen_residual_gate_add,
    residual_gate_add,
)

residual_gate_add_module = importlib.import_module("mindiesd.layers.residual_gate_add")


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestResidualGateAddTriton(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")
        self.y = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")
        self.gate = torch.randn(2, 1, 64, dtype=torch.bfloat16, device="npu")

    def _assert_matches_reference(self, y, gate):
        actual = residual_gate_add(self.x, y, gate)
        expected = (self.x.float() + y.float() * gate.float()).to(self.x.dtype)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-2, atol=1e-2))

    def test_wan_operand_order(self):
        self._assert_matches_reference(self.y, self.gate)

    def test_flux_operand_order(self):
        self._assert_matches_reference(self.gate, self.y)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestQwenResidualGateAddTriton(unittest.TestCase):
    def test_broadcast_contract_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch_size, sequence_length in ((1, 16), (2, 31)):
                with self.subTest(dtype=dtype, batch_size=batch_size, sequence_length=sequence_length):
                    residual = torch.randn(batch_size, sequence_length, 64, dtype=dtype, device="npu")
                    branch = torch.randn_like(residual)
                    gate = torch.randn(batch_size, 1, 64, dtype=dtype, device="npu")

                    actual = qwen_residual_gate_add(residual, branch, gate)
                    expected = (residual.float() + branch.float() * gate.float()).to(dtype)

                    self.assertTrue(torch.allclose(actual, expected, rtol=1e-2, atol=1e-2))
                    self.assertEqual(actual.shape, residual.shape)
                    self.assertEqual(actual.dtype, residual.dtype)
                    self.assertTrue(actual.is_contiguous())
