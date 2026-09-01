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
    residual_gate_add,
)

residual_gate_add_module = importlib.import_module("mindiesd.layers.residual_gate_add")


class TestResidualGateInputNormalization(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.y = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.gate = torch.randn(2, 1, 8, dtype=torch.bfloat16)

    def test_wan_operand_order_is_preserved(self):
        normalized = _normalize_triton_inputs(self.x, self.y, self.gate)
        self.assertIsNotNone(normalized)
        self.assertIs(normalized[1], self.y)
        self.assertIs(normalized[2], self.gate)

    def test_flux_operand_order_is_normalized(self):
        normalized = _normalize_triton_inputs(self.x, self.gate, self.y)
        self.assertIsNotNone(normalized)
        self.assertIs(normalized[1], self.y)
        self.assertIs(normalized[2], self.gate)

    def test_full_sized_gate_raises(self):
        full_gate = torch.randn_like(self.x)
        with self.assertRaisesRegex(ValueError, "gate must have shape"):
            _normalize_triton_inputs(self.x, self.y, full_gate)

    def test_noncontiguous_input_raises(self):
        noncontiguous = self.x.transpose(0, 1)
        y = self.y.transpose(0, 1)
        gate = torch.randn(4, 1, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            _normalize_triton_inputs(noncontiguous, y, gate)

    def test_mismatched_y_shape_raises(self):
        y = torch.randn(2, 2, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "x and y must have the same shape"):
            _normalize_triton_inputs(self.x, y, self.gate)

    def test_empty_input_raises(self):
        x = torch.randn(2, 0, 8, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        gate = torch.randn(2, 1, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "x must not be empty"):
            _normalize_triton_inputs(x, y, gate)

    def test_mismatched_device_raises(self):
        gate = torch.empty(2, 1, 8, dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(ValueError, "must be on the same device"):
            _normalize_triton_inputs(self.x, self.y, gate)

    def test_mismatched_dtype_raises(self):
        gate = self.gate.float()
        with self.assertRaisesRegex(ValueError, "must have the same dtype"):
            _normalize_triton_inputs(self.x, self.y, gate)


class TestResidualGateAddFallback(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.y = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.gate = torch.randn(2, 1, 8, dtype=torch.bfloat16)

    def _assert_matches_reference(self, y, gate):
        with patch.object(residual_gate_add_module, "_TRITON_ON_ASCEND", False):
            actual = residual_gate_add(self.x, y, gate)
        expected = (self.x.float() + y.float() * gate.float()).to(self.x.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_wan_operand_order(self):
        self._assert_matches_reference(self.y, self.gate)

    def test_flux_operand_order(self):
        self._assert_matches_reference(self.gate, self.y)


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

    def test_unsupported_full_sized_gate_raises(self):
        full_gate = torch.randn_like(self.x)
        with self.assertRaisesRegex(ValueError, "gate must have shape"):
            residual_gate_add(self.x, self.y, full_gate)


if __name__ == "__main__":
    unittest.main()
