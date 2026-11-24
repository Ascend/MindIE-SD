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

import unittest

import torch

from mindiesd.layers.residual_gate_add import (
    _normalize_residual_gate_inputs,
    qwen_residual_gate_add,
)


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


class TestResidualGateInputNormalization(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.y = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.gate = torch.randn(2, 1, 8, dtype=torch.bfloat16)

    def test_wan_operand_order_is_preserved(self):
        normalized = _normalize_residual_gate_inputs(self.x, self.y, self.gate)
        self.assertIs(normalized[0], self.x)
        self.assertIs(normalized[1], self.y)
        self.assertIs(normalized[2], self.gate)

    def test_flux_operand_order_is_normalized(self):
        normalized = _normalize_residual_gate_inputs(self.x, self.gate, self.y)
        self.assertIs(normalized[0], self.x)
        self.assertIs(normalized[1], self.y)
        self.assertIs(normalized[2], self.gate)

    def test_noncontiguous_input_returns_contiguous_output(self):
        x = self.x.transpose(0, 1)
        y = self.y.transpose(0, 1)
        gate = torch.randn(4, 1, 8, dtype=torch.bfloat16)
        actual = torch.ops.mindiesd.residual_gate_add(x, y, gate)
        _assert_matches_reference(actual, x, y, gate)
        self.assertTrue(actual.is_contiguous())

    def test_full_sized_gate_raises(self):
        with self.assertRaisesRegex(ValueError, "gate must have shape"):
            _normalize_residual_gate_inputs(self.x, self.y, torch.randn_like(self.x))

    def test_mismatched_y_shape_raises(self):
        y = torch.randn(2, 2, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "x and y must have the same shape"):
            _normalize_residual_gate_inputs(self.x, y, self.gate)

    def test_mismatched_dtype_raises(self):
        with self.assertRaisesRegex(ValueError, "must have the same dtype"):
            _normalize_residual_gate_inputs(self.x, self.y, self.gate.float())

    def test_empty_input_raises(self):
        x = torch.empty(2, 0, 8, dtype=torch.bfloat16)
        y = torch.empty_like(x)
        gate = torch.empty(2, 1, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "x must not be empty"):
            _normalize_residual_gate_inputs(x, y, gate)

    def test_mismatched_device_raises(self):
        gate = torch.empty(2, 1, 8, dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(ValueError, "must be on the same device"):
            _normalize_residual_gate_inputs(self.x, self.y, gate)


class TestResidualGateAdd(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.y = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.gate = torch.randn(2, 1, 8, dtype=torch.bfloat16)

    def _assert_matches_reference(self, y, gate):
        actual = torch.ops.mindiesd.residual_gate_add(self.x, y, gate)
        _assert_matches_reference(actual, self.x, y, gate)
        self.assertTrue(actual.is_contiguous())

    def test_wan_operand_order(self):
        self._assert_matches_reference(self.y, self.gate)

    def test_flux_operand_order(self):
        self._assert_matches_reference(self.gate, self.y)


class TestQwenResidualGateAdd(unittest.TestCase):
    def _assert_matches_reference(self, dtype, batch_size, sequence_length):
        residual = torch.randn(batch_size, sequence_length, 8, dtype=dtype)
        branch = torch.randn_like(residual)
        gate = torch.randn(batch_size, 1, 8, dtype=dtype)

        actual = torch.ops.mindiesd.qwen_residual_gate_add(residual, branch, gate)
        _assert_matches_reference(actual, residual, branch, gate)
        self.assertEqual(actual.shape, residual.shape)
        self.assertEqual(actual.dtype, residual.dtype)
        self.assertTrue(actual.is_contiguous())

    def test_broadcast_contract_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch_size, sequence_length in ((1, 4), (2, 7)):
                with self.subTest(dtype=dtype, batch_size=batch_size, sequence_length=sequence_length):
                    self._assert_matches_reference(dtype, batch_size, sequence_length)

    def test_gate_without_token_axis_raises(self):
        residual = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        branch = torch.randn_like(residual)
        gate = torch.randn(2, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"exactly \[B,1,D\]"):
            qwen_residual_gate_add(residual, branch, gate)

    def test_full_sized_gate_raises(self):
        residual = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        branch = torch.randn_like(residual)
        with self.assertRaisesRegex(ValueError, r"exactly \[B,1,D\]"):
            qwen_residual_gate_add(residual, branch, torch.randn_like(residual))

    def test_non_3d_residual_raises(self):
        residual = torch.randn(2, 8, dtype=torch.bfloat16)
        branch = torch.randn_like(residual)
        gate = torch.randn(2, 1, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"must both be \[B,S,D\]"):
            qwen_residual_gate_add(residual, branch, gate)

    def test_branch_shape_mismatch_raises(self):
        residual = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        branch = torch.randn(2, 5, 8, dtype=torch.bfloat16)
        gate = torch.randn(2, 1, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"identical \[B,S,D\] shapes"):
            qwen_residual_gate_add(residual, branch, gate)

    def test_dtype_mismatch_raises(self):
        residual = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        branch = torch.randn_like(residual)
        gate = torch.randn(2, 1, 8, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "must share a dtype"):
            qwen_residual_gate_add(residual, branch, gate)


if __name__ == "__main__":
    unittest.main()
