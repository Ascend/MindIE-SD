#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""CPU reference tests for mindiesd.mul_add: out = a + b * c."""

import unittest

import torch

from tests.UT.ops_reference.reference_math import torch_mul_add_ref as _torch_mul_add_ref


class TestMulAddReference(unittest.TestCase):
    """Reference tests: pure PyTorch computation, always runnable."""

    def test_ref_basic(self):
        """Reference basic functionality."""
        batch, seq_len, hidden_size = 1, 4, 128
        a = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
        b = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
        c = torch.randn(batch, 1, hidden_size, dtype=torch.bfloat16)

        ref = _torch_mul_add_ref(a, b, c)
        expected = a.float() + b.float() * c.float()
        torch.testing.assert_close(ref, expected, atol=1e-5, rtol=1e-5)

    def test_ref_broadcast(self):
        """Reference verifies c broadcasts along seq_len dimension."""
        batch, seq_len, hidden_size = 2, 8, 64
        a = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)
        b = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)
        c = torch.randn(batch, 1, hidden_size, dtype=torch.float32)

        ref = _torch_mul_add_ref(a, b, c)
        expected = a + b * c
        torch.testing.assert_close(ref, expected, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
