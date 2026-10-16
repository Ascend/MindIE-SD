#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
Unit tests for mindiesd.mul_add operator.

Tests the fused mul-add operator: out = a + b * c
where a/b = [batch, seq_len, hidden_size] and c = [batch, 1, hidden_size].
"""

import os
import unittest

import torch

from device import DEVICE_ID

# ============================================================================
# Helpers
# ============================================================================


def _is_npu_available():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def _has_mindiesd_op():
    try:
        import mindiesd  # noqa: F401
        return hasattr(torch.ops, 'mindiesd') and hasattr(torch.ops.mindiesd, 'mul_add')
    except Exception:
        return False


NPU_AVAILABLE = _is_npu_available() and _has_mindiesd_op()


def _to_npu(*tensors):
    """Move tensors to NPU if available, otherwise keep on CPU."""
    if NPU_AVAILABLE:
        device = torch.device(f"npu:{DEVICE_ID}")
        torch.npu.set_device(device)
        return tuple(t.to(device) for t in tensors)
    return tensors


def _torch_mul_add_ref(a, b, c):
    """PyTorch reference for mul_add: out = a + b * c."""
    return a.float() + b.float() * c.float()


# ============================================================================
# Test Cases
# ============================================================================


@unittest.skipUnless(NPU_AVAILABLE, "NPU or mindiesd operator not available")
@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestMulAddNPU(unittest.TestCase):
    """NPU tests: run operator on real hardware and compare against reference."""

    def _run_and_check(self, a, b, c, atol=1e-2, rtol=1e-2, label=""):
        """Run operator on NPU and compare with reference."""
        ref = _torch_mul_add_ref(a, b, c)

        npu_a, npu_b, npu_c = _to_npu(a, b, c)
        out = torch.ops.mindiesd.mul_add(npu_a, npu_b, npu_c)

        self.assertEqual(out.shape, a.shape, f"output shape mismatch: {out.shape} vs {a.shape}")
        self.assertTrue(
            torch.allclose(out.cpu().float(), ref, atol=atol, rtol=rtol),
            f"{label} output mismatch (max diff: {(out.cpu().float() - ref).abs().max()})"
        )

    def test_mul_add_basic(self):
        """Small shape basic functionality test."""
        batch, seq_len, hidden_size = 1, 4, 128
        a = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
        b = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
        c = torch.randn(batch, 1, hidden_size, dtype=torch.bfloat16)
        self._run_and_check(a, b, c, label=f"[BF16] shape=({batch},{seq_len},{hidden_size})")

    def test_mul_add_typical(self):
        """Typical LLM shape tests (BF16)."""
        typical_shapes = [
            (1, 7200, 4608),
            (1, 256, 4608),
            (1, 7200, 3072),
            (1, 256, 3072),
        ]
        for batch, seq_len, hidden_size in typical_shapes:
            a = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
            b = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
            c = torch.randn(batch, 1, hidden_size, dtype=torch.bfloat16)
            self._run_and_check(a, b, c, label=f"[BF16] shape=({batch},{seq_len},{hidden_size})")

    def test_mul_add_fp16(self):
        """FP16 precision test."""
        batch, seq_len, hidden_size = 1, 256, 3072
        a = torch.randn(batch, seq_len, hidden_size, dtype=torch.float16)
        b = torch.randn(batch, seq_len, hidden_size, dtype=torch.float16)
        c = torch.randn(batch, 1, hidden_size, dtype=torch.float16)
        self._run_and_check(a, b, c, label=f"[FP16] shape=({batch},{seq_len},{hidden_size})")

    def test_mul_add_small_shapes(self):
        """Multiple small shape boundary tests."""
        shapes = [
            (1, 1, 64),
            (1, 2, 128),
            (1, 4, 256),
            (1, 8, 512),
            (1, 16, 768),
            (1, 64, 1024),
        ]
        for batch, seq_len, hidden_size in shapes:
            a = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
            b = torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)
            c = torch.randn(batch, 1, hidden_size, dtype=torch.bfloat16)
            self._run_and_check(a, b, c, label=f"[BF16] shape=({batch},{seq_len},{hidden_size})")


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


# ============================================================================
# Main
# ============================================================================


if __name__ == '__main__':
    unittest.main()
