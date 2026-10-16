#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
Unit tests for mindiesd.eagle_ffn_linear operator.

y = act(x @ W1^T + b1) @ W2^T + b2, act in {gelu, silu, swiglu},
bf16/fp16 with fp16/bf16/fp32 bias, linear/canonical weight layouts,
swiglu uses single-matmul form weight1=[2H, K].
"""

import os
import unittest

import torch
import torch.nn.functional as F

from device import DEVICE_ID


def _is_npu_available():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def _has_mindiesd_op():
    try:
        import mindiesd  # noqa: F401
        return hasattr(torch.ops, 'mindiesd') and hasattr(torch.ops.mindiesd, 'eagle_ffn_linear')
    except Exception:
        return False


NPU_AVAILABLE = _is_npu_available() and _has_mindiesd_op()


def _to_npu(*tensors):
    if NPU_AVAILABLE:
        device = torch.device(f"npu:{DEVICE_ID}")
        torch.npu.set_device(device)
        return tuple(None if t is None else t.to(device) for t in tensors)
    return tensors


def _ref(act, x, w1, w2, b1, b2):
    """CPU fp32 golden: 全程 fp32 计算。"""
    up = F.linear(x.float(), w1.float(), None if b1 is None else b1.float())
    if act == 'swiglu':
        gate, up_half = up.chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up_half, w2.float(), None if b2 is None else b2.float())
    if act == 'silu':
        return F.linear(F.silu(up), w2.float(), None if b2 is None else b2.float())
    return F.linear(F.gelu(up), w2.float(), None if b2 is None else b2.float())


@unittest.skipUnless(NPU_AVAILABLE, "NPU or mindiesd operator not available")
@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU tests when MINDIE_TEST_MODE is CPU.")
class TestFfnLinearNPU(unittest.TestCase):
    M, K, H, N = 97, 1024, 2816, 1024  # 非对齐 M

    def _make(self, dtype, bias_dtype, act, with_bias=True, batch=1):
        shape = (batch, self.M, self.K)
        rows = 2 * self.H if act == 'swiglu' else self.H
        x = torch.randn(shape, dtype=dtype) * 0.3
        w1 = torch.randn((rows, self.K), dtype=dtype) * 0.05
        w2 = torch.randn((self.N, self.H), dtype=dtype) * 0.05
        b1 = torch.randn(rows, dtype=bias_dtype) * 0.1 if with_bias else None
        b2 = torch.randn(self.N, dtype=bias_dtype) * 0.1 if with_bias else None
        return _to_npu(x, w1, w2, b1, b2)

    def _run_and_check(self, act, dtype, bias_dtype, with_bias=True, batch=1, label=""):
        x, w1, w2, b1, b2 = self._make(dtype, bias_dtype, act, with_bias, batch)
        out = torch.ops.mindiesd.eagle_ffn_linear(x, w1, w2, b1, b2, act, 0)
        ref = _ref(act, x, w2=w2, w1=w1, b1=b1, b2=b2).to(dtype)
        self.assertEqual(tuple(out.shape), tuple(ref.shape), f"{label} shape mismatch")
        diff = (out.float() - ref.float()).abs()
        rel = (diff / (ref.float().abs() + 1e-3)).mean().item()
        self.assertLess(rel, 0.05, f"{label} mean_rel={rel:.5f} max_abs={diff.max().item():.4f}")

    # ---- dtype 组合（bias 与 x 同 dtype / fp32 bias）----
    def test_fp16_bias(self):
        for act in ('gelu', 'silu'):
            self._run_and_check(act, torch.float16, torch.float16, label=f"[fp16/bias/{act}]")

    def test_fp16_fp32bias(self):
        for act in ('gelu', 'silu'):
            self._run_and_check(act, torch.float16, torch.float32, label=f"[fp16/fp32bias/{act}]")

    def test_bf16_bias(self):
        for act in ('gelu', 'silu'):
            self._run_and_check(act, torch.bfloat16, torch.bfloat16, label=f"[bf16/bias/{act}]")

    def test_bf16_fp32bias(self):
        for act in ('gelu', 'silu'):
            self._run_and_check(act, torch.bfloat16, torch.float32, label=f"[bf16/fp32bias/{act}]")

    # ---- swiglu（w1=[2H,K]）----
    def test_swiglu_no_bias(self):
        for dtype in (torch.float16, torch.bfloat16):
            self._run_and_check('swiglu', dtype, dtype, with_bias=False, label=f"[swiglu-nobias/{dtype}]")

    def test_swiglu_with_bias(self):
        for dtype in (torch.float16, torch.bfloat16):
            self._run_and_check('swiglu', dtype, dtype, with_bias=True, label=f"[swiglu-bias/{dtype}]")

    # ---- 多维输入（[.., M, K] 前导维）----
    def test_batch_leading_dims(self):
        self._run_and_check('gelu', torch.bfloat16, torch.bfloat16, batch=3, label="[bf16/3D]")

    # ---- canonical 布局与 linear 等价 ----
    def test_canonical_layout(self):
        act, dtype = 'gelu', torch.bfloat16
        x, w1, w2, _, _ = self._make(dtype, dtype, act, with_bias=False)
        out_linear = torch.ops.mindiesd.eagle_ffn_linear(x, w1, w2, None, None, act, 0)
        out_canon = torch.ops.mindiesd.eagle_ffn_linear(x, w1.t().contiguous(), w2.t().contiguous(), None, None, act, 0)
        self.assertEqual(tuple(out_linear.shape), tuple(out_canon.shape))
        self.assertLess((out_linear.float() - out_canon.float()).abs().max().item(), 0.5,
                        "linear/canonical outputs should match")

    # ---- 拒绝路径 ----
    def test_reject_shape_mismatch(self):
        x = torch.randn(8, 512, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        w1 = torch.randn(1023, 512, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        w2 = torch.randn(512, 1024, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        with self.assertRaises(RuntimeError):
            torch.ops.mindiesd.eagle_ffn_linear(x, w1, w2, None, None, 'gelu', 0)

    def test_reject_mixed_bias_dtype(self):
        x = torch.randn(8, 512, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        w1 = torch.randn(1024, 512, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        w2 = torch.randn(512, 1024, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        b1 = torch.randn(1024, dtype=torch.float16, device=f"npu:{DEVICE_ID}")
        b2 = torch.randn(512, dtype=torch.float16, device=f"npu:{DEVICE_ID}")
        with self.assertRaises(RuntimeError):
            torch.ops.mindiesd.eagle_ffn_linear(x, w1, w2, b1, b2, 'gelu', 0)

    def test_reject_bias1_only(self):
        x = torch.randn(8, 512, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        w1 = torch.randn(1024, 512, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        w2 = torch.randn(512, 1024, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        b1 = torch.randn(1024, dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
        with self.assertRaises(RuntimeError):
            torch.ops.mindiesd.eagle_ffn_linear(x, w1, w2, b1, None, 'gelu', 0)

    # ---- 高层 API：fused 与原生链回退一致性 ----
    def test_layers_api_matches_op(self):
        from mindiesd.layers import eagle_ffn_linear as ffn_layer
        act, dtype = 'silu', torch.bfloat16
        x, w1, w2, b1, b2 = self._make(dtype, dtype, act)
        out_op = torch.ops.mindiesd.eagle_ffn_linear(x, w1, w2, b1, b2, act, 0)
        out_layer = ffn_layer(x, w1, w2, b1, b2, act, fused=True)
        self.assertTrue(torch.allclose(out_op.float(), out_layer.float(), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
