#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
Unit tests for mindiesd.layers.quant.transpose_dynamic_quant operator.

Fused Transpose + DynamicQuantV2 + Transpose: per-group symmetric dynamic quant
(scale = max(|x * smooth|) / 448, fp8_e4m3fn output). Both transposes are folded
into index arithmetic: y layout always equals the x layout, any quant axis
(0, middle axes, -1).

Cases cover correctness (golden torch reference, eco fp8/fp32 tolerances) for
2D/3D/4D, fp16/bf16, with/without smooth scales, every axis position, input
guards (axis range, unsupported dtype, smooth device mismatch), empty-tensor
behavior (deterministic zero scale), and a perf smoke vs the composed torch
baseline.

Run: pytest tests/ops/transpose_dynamic_quant/test_transpose_dynamic_quant.py
"""

import os
import sys
import unittest
from unittest import mock

import torch

try:
    from device import DEVICE_ID
except ImportError:
    DEVICE_ID = 0

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

_DTYPE_MAX = 448.0
_FP8_ATOL, _FP8_RTOL, _FP8_CAP = 0.0625, 0.25, 1.0
_F32_ATOL, _F32_RTOL, _F32_CAP = 1.526e-05, 9.766e-04, 1e-2


def _is_npu_available():
    if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
        return False
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def _load_op():
    """Import the operator; fall back to direct file loading when the C++ plugin
    (libPTAExtensionOPS.so) is absent so that ``import mindiesd`` fails.
    """

    def _via_package():
        from mindiesd.layers.quant.transpose_dynamic_quant import (
            transpose_dynamic_quant as _op,
        )

        return _op

    def _via_file():
        import importlib.util
        import types

        up4 = os.path.dirname
        root = up4(up4(up4(up4(os.path.abspath(__file__)))))
        mindiesd_root = os.path.join(root, "mindiesd")
        added = []
        for name in ("mindiesd", "mindiesd.layers", "mindiesd.layers.quant"):
            if name not in sys.modules:
                module = types.ModuleType(name)
                sub = name.split(".")[1:]
                module.__path__ = [os.path.join(mindiesd_root, *sub)] if sub else [mindiesd_root]
                sys.modules[name] = module
                added.append(name)
        module_path = os.path.join(mindiesd_root, "layers", "quant", "transpose_dynamic_quant.py")
        spec = importlib.util.spec_from_file_location("mindiesd.layers.quant.transpose_dynamic_quant", module_path)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        finally:
            # Restore sys.modules so later tests importing the real package are unaffected.
            sys.modules.pop(spec.name, None)
            for name in added:
                sys.modules.pop(name, None)
        return module.transpose_dynamic_quant

    try:
        return _via_package()
    except Exception:
        if not _is_npu_available():
            return None
        return _via_file()


transpose_dynamic_quant = _load_op()
NPU_AVAILABLE = _is_npu_available() and transpose_dynamic_quant is not None


def _to_npu(tensor):
    if NPU_AVAILABLE:
        device = torch.device(f"npu:{DEVICE_ID}")
        torch.npu.set_device(device)
        return tensor.to(device)
    return tensor


def _reference(x, smooth_scales, axis):
    """Golden: moveaxis to dim0, per-column dynamic quant, moveaxis back.

    y always keeps the x layout. smooth_scales follows the official
    aclnnDynamicQuantV2 semantics: 1-D with length == quant axis length,
    multiplied element-wise along that axis.
    """
    ndim = x.dim()
    a = axis if axis >= 0 else ndim + axis
    x2 = x.moveaxis(a, 0)
    m, r = x2.shape[0], x2.numel() // x2.shape[0]
    r2 = x2.reshape(m, r)
    inp32 = r2.to(torch.float32)
    if smooth_scales is not None:
        # Product in fp32, aligned with aclnnDynamicQuantV2 (no low-precision fallback).
        inp32 = inp32 * smooth_scales.reshape(-1).to(torch.float32).unsqueeze(1)
    col_max = inp32.abs().amax(dim=0)
    scale = (col_max / _DTYPE_MAX).reshape(tuple(x.shape[:a]) + tuple(x.shape[a + 1 :]))
    denom = torch.where(scale > 0, scale, torch.ones_like(scale))
    # Single RN: fp32 quotient cast straight to fp8_e4m3fn (no round-then-cast).
    q = (inp32 / denom.reshape(-1).unsqueeze(0)).clamp_(-_DTYPE_MAX, _DTYPE_MAX)
    q = q.to(torch.float8_e4m3fn)
    y = q.reshape(x2.shape).moveaxis(0, a)
    return y, scale


def _ulp(g, mantissa_bits):
    e = torch.floor(torch.log2(g.clamp_min(1e-30)))
    return torch.pow(2.0, e - mantissa_bits)


def _assert_close(testcase, y, s, yr, sr):
    d = (s.float() - sr.float()).abs()
    g = sr.float().abs()
    matched = float((d <= _F32_ATOL + _F32_RTOL * g).float().mean())
    cap = torch.maximum(torch.maximum(torch.full_like(g, _F32_CAP), 32.0 * _ulp(g, 23)), _F32_ATOL + _F32_RTOL * g)
    testcase.assertEqual(int((d > cap).sum()), 0, "scale exceeds eco absolute cap")
    testcase.assertGreaterEqual(matched, 0.99, "scale matched_ratio below eco threshold")

    d = (y.to(torch.float32) - yr.to(torch.float32)).abs()
    g = yr.to(torch.float32).abs()
    matched = float((d <= _FP8_ATOL + _FP8_RTOL * g).float().mean())
    cap = torch.maximum(torch.maximum(torch.full_like(g, _FP8_CAP), 32.0 * _ulp(g, 3)), _FP8_ATOL + _FP8_RTOL * g)
    testcase.assertEqual(int((d > cap).sum()), 0, "y exceeds eco absolute cap")
    testcase.assertGreaterEqual(matched, 0.99, "y matched_ratio below eco threshold")


def _baseline_chain(x):
    """Composed small-op dynamic quant (transpose copy + fp32 math chain +
    transpose back), producing the same layout as the fused operator.
    """
    xt = x.transpose(1, 2).to(torch.float32)
    row_max = xt.abs().amax(dim=-1)
    scale = row_max / _DTYPE_MAX
    denom = torch.where(scale > 0, scale, torch.ones_like(scale))
    # Single RN cast, same math as the reference.
    q = (xt / denom.unsqueeze(-1)).clamp_(-_DTYPE_MAX, _DTYPE_MAX)
    y = q.to(torch.float8_e4m3fn).transpose(1, 2).contiguous()
    return y, scale


def _get_op_module():
    """Return the operator module for monkeypatching; None when only the
    standalone-file loader succeeded (sys.modules entry was restored).
    """
    try:
        import mindiesd.layers.quant.transpose_dynamic_quant as tdq

        return tdq
    except Exception:
        return None


@unittest.skipUnless(NPU_AVAILABLE, "NPU or transpose_dynamic_quant operator not available")
class TestTransposeDynamicQuant(unittest.TestCase):
    def _run_case(self, x_cpu, smooth_cpu, axis):
        x_npu = _to_npu(x_cpu)
        sm_npu = _to_npu(smooth_cpu) if smooth_cpu is not None else None
        with torch.no_grad():
            y, s = transpose_dynamic_quant(x_npu, sm_npu, axis)
        torch.npu.synchronize()
        yr, sr = _reference(x_cpu, smooth_cpu, axis)
        self.assertEqual(tuple(y.shape), tuple(yr.shape))
        self.assertEqual(tuple(y.shape), tuple(x_cpu.shape))
        self.assertEqual(tuple(s.shape), tuple(sr.shape))
        self.assertEqual(y.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.float32)
        _assert_close(self, _to_npu(y).cpu(), _to_npu(s).cpu(), yr, sr)

    def test_axis0_same_layout_fp16(self):
        x = torch.randn(256, 512, dtype=torch.float16)
        sm = torch.randn(256, dtype=torch.float16)  # 1-D along the quant axis
        self._run_case(x, sm, 0)
        self._run_case(x, None, 0)

    def test_axis0_same_layout_3d(self):
        x = torch.randn(4, 128, 256, dtype=torch.float16)
        sm = torch.randn(4, dtype=torch.float16)  # 1-D along the quant axis
        self._run_case(x, sm, 0)

    def test_axis_last_same_layout_bf16(self):
        x = torch.randn(256, 512, dtype=torch.bfloat16)
        sm = torch.randn(512, dtype=torch.bfloat16)  # 1-D along the quant axis
        self._run_case(x, sm, -1)

    def test_axis1_same_layout_bf16(self):
        x = torch.randn(3, 100, 128, dtype=torch.bfloat16)
        self._run_case(x, None, 1)
        sm = torch.randn(100, dtype=torch.bfloat16)  # 1-D along the quant axis
        self._run_case(x, sm, 1)

    def test_axis_neg2_same_layout_fp16(self):
        # Negative middle axis resolves to dim 1.
        x = torch.randn(2, 511, 65, dtype=torch.float16)
        self._run_case(x, None, -2)

    def test_axis2_same_layout_4d_fp16(self):
        x = torch.randn(2, 3, 96, 64, dtype=torch.float16)
        sm = torch.randn(96, dtype=torch.float16)  # 1-D along the quant axis
        self._run_case(x, sm, 2)

    def test_smooth_axis_len_distinct_coeffs(self):
        # Pins the smooth semantics: axis length differs from other dims and
        # every coefficient is distinct, so a wrong indexing direction fails.
        x = torch.randn(3, 8, dtype=torch.float16)
        sm = torch.arange(1, 4, dtype=torch.float16)  # length 3 (axis=0), all distinct
        self._run_case(x, sm, 0)
        x2 = torch.randn(8, 3, dtype=torch.float16)
        sm2 = torch.arange(1, 4, dtype=torch.float16)  # length 3 (axis=-1)
        self._run_case(x2, sm2, -1)

    def test_dtype_not_supported(self):
        x = _to_npu(torch.randn(4, 8, dtype=torch.float32))
        with self.assertRaises(ValueError):
            transpose_dynamic_quant(x, None, 0)

    def test_smooth_device_mismatch(self):
        x = _to_npu(torch.randn(4, 8, dtype=torch.float16))
        sm = torch.randn(8, dtype=torch.float16)  # stays on CPU while x is on NPU
        with self.assertRaises(ValueError):
            transpose_dynamic_quant(x, sm, 0)

    def test_empty_tensor_zero_scale(self):
        # Empty quant axis: y is empty and scale must be deterministic zeros.
        for shape, axis, scale_shape in (
            ((0, 3), 0, (3,)),
            ((3, 0), -1, (3,)),
            ((2, 0, 3), 1, (2, 3)),
        ):
            with self.subTest(shape=shape, axis=axis):
                x = _to_npu(torch.randn(*shape, dtype=torch.float16))
                with torch.no_grad():
                    y, s = transpose_dynamic_quant(x, None, axis)
                torch.npu.synchronize()
                self.assertEqual(tuple(y.shape), shape)
                self.assertEqual(tuple(s.shape), scale_shape)
                self.assertEqual(y.dtype, torch.float8_e4m3fn)
                self.assertEqual(s.dtype, torch.float32)
                self.assertTrue(bool((s.float() == 0).all()), "empty-axis scale must be zeros")

    def test_shape_mismatch_smooth(self):
        x = _to_npu(torch.randn(4, 8, dtype=torch.float16))
        sm = _to_npu(torch.randn(5, dtype=torch.float16))
        with self.assertRaises(ValueError):
            transpose_dynamic_quant(x, sm, 0)

    def test_perf_smoke_vs_baseline_chain(self):
        x = _to_npu(torch.randn(14, 8192, 128, dtype=torch.bfloat16))

        def run_fused():
            y, s = transpose_dynamic_quant(x, None, 1)
            torch.npu.synchronize()
            return y, s

        def run_baseline():
            y, s = _baseline_chain(x)
            torch.npu.synchronize()
            return y, s

        # Result validation: fused output must match the eco golden reference.
        y, s = run_fused()
        yr, sr = _reference(x.cpu(), None, 1)
        _assert_close(self, _to_npu(y).cpu(), _to_npu(s).cpu(), yr, sr)

        for _ in range(3):
            run_fused()
            run_baseline()
        start, end = torch.npu.Event(True), torch.npu.Event(True)
        start.record()
        for _ in range(10):
            run_fused()
        end.record()
        torch.npu.synchronize()
        t_fused = start.elapsed_time(end) / 10
        start.record()
        for _ in range(10):
            run_baseline()
        end.record()
        torch.npu.synchronize()
        t_baseline = start.elapsed_time(end) / 10
        # Fused kernel must not be slower than the composed torch chain.
        msg = f"fused {t_fused:.3f}ms vs baseline {t_baseline:.3f}ms"
        self.assertLess(t_fused, t_baseline, msg)

    def test_axis_last_fullload_tier(self):
        # Elements >= 4M, BLOCK_COL <= 8192 and cdiv(rows, 128) < num_cores:
        # exercises the whole-row full-load kernel (single GM read) including
        # smooth on/off, odd row counts and non-power-of-two col masks.
        self._run_case(torch.randn(1024, 7168, dtype=torch.float16), None, -1)
        sm = torch.randn(7168, dtype=torch.float16)
        self._run_case(torch.randn(1024, 7168, dtype=torch.float16), sm, -1)
        self._run_case(torch.randn(1025, 7168, dtype=torch.bfloat16), None, -1)
        self._run_case(torch.randn(1024, 5000, dtype=torch.bfloat16), None, -1)

    def test_large_shape_tiers(self):
        # axis=0 on a >=4M shape selects the BLOCK_M=128 tier with a row mask
        # (1000 % 128 != 0); axis=-1 on (8192, 3584) falls through the full-load
        # gate (cdiv(rows, 128) == num_cores) back to the chunked kernel.
        sm = torch.randn(1000, dtype=torch.float16)
        self._run_case(torch.randn(1000, 7168, dtype=torch.float16), sm, 0)
        self._run_case(torch.randn(8192, 3584, dtype=torch.bfloat16), None, -1)

    def test_middle_axis_large_shape(self):
        # Middle axis on a >=4M shape exercises the two-kernel path (atomic
        # partial max + same-layout quantize) with non-multiple masks.
        sm = torch.randn(8192, dtype=torch.bfloat16)
        self._run_case(torch.randn(14, 8192, 128, dtype=torch.bfloat16), None, 1)
        self._run_case(torch.randn(14, 8192, 128, dtype=torch.bfloat16), sm, 1)

    def test_thin_quant_axis(self):
        # Tiny quant axis (m_size <= 64) with a huge trailing dim selects the
        # thin-axis tier (BLOCK_M shrunk to the axis, BLOCK_K widened); a 128
        # row tile would be almost fully masked and degenerate the k-loop.
        self._run_case(torch.randn(2, 1024, 2048, 256, dtype=torch.bfloat16), None, 0)
        self._run_case(torch.randn(3, 8, dtype=torch.float16), None, 0)

    def test_small_shape_block_mask(self):
        # Small shape keeps the chunked kernel with (64, 64) tiles; 130 rows and
        # 500 cols are both non-multiples, so the row/col masks are exercised.
        sm = torch.randn(500, dtype=torch.float16)
        self._run_case(torch.randn(130, 500, dtype=torch.float16), sm, -1)
        self._run_case(torch.randn(130, 500, dtype=torch.float16), None, -1)

    def test_non_contiguous_inputs(self):
        # Non-contiguous x and smooth must be materialized via .contiguous().
        x = torch.randn(512, 256, dtype=torch.float16).t()  # (256, 512) view
        sm = torch.randn(512, dtype=torch.float16)[::2]  # (256,) strided view
        self._run_case(x, sm, 0)

    def test_validation_rejections(self):
        x_npu = _to_npu(torch.randn(4, 8, dtype=torch.float16))
        sm32_npu = _to_npu(torch.randn(8, dtype=torch.float32))
        x1d_npu = _to_npu(torch.randn(8, dtype=torch.float16))
        sm2d_npu = _to_npu(torch.randn(1, 8, dtype=torch.float16))
        cases = [
            # 1-D input
            lambda: transpose_dynamic_quant(x1d_npu, None, 0),
            # axis out of range (positive and negative)
            lambda: transpose_dynamic_quant(x_npu, None, 5),
            lambda: transpose_dynamic_quant(x_npu, None, -3),
            # smooth dtype mismatch
            lambda: transpose_dynamic_quant(x_npu, sm32_npu, 0),
            # smooth not 1-D
            lambda: transpose_dynamic_quant(x_npu, sm2d_npu, 0),
            # x on CPU while NPU is available
            lambda: transpose_dynamic_quant(torch.randn(4, 8, dtype=torch.float16), None, 0),
        ]
        for i, call in enumerate(cases):
            with self.subTest(case=i):
                with self.assertRaises(ValueError):
                    call()

    def test_int32_limit_rejection(self):
        tdq = _get_op_module()
        if tdq is None:
            self.skipTest("operator module not importable for monkeypatching")
        x = _to_npu(torch.randn(4, 512, dtype=torch.float16))
        with mock.patch.object(tdq, "_MAX_INT32", 1000):
            with self.assertRaises(ValueError):
                transpose_dynamic_quant(x, None, -1)

    def test_guard_runtimes(self):
        # Missing fp8 dtype support and absent Triton-Ascend both fail fast.
        tdq = _get_op_module()
        if tdq is None:
            self.skipTest("operator module not importable for monkeypatching")
        x = _to_npu(torch.randn(4, 8, dtype=torch.float16))
        with mock.patch.object(tdq, "_FP8_DTYPE", None):
            with self.assertRaises(RuntimeError):
                transpose_dynamic_quant(x, None, 0)
        with mock.patch.object(tdq, "_TRITON_ON_ASCEND", False):
            with self.assertRaises(RuntimeError):
                transpose_dynamic_quant(x, None, 0)

    def test_register_fake_meta(self):
        # The fake impl runs under FakeTensorMode and must shape/dtype both
        # outputs (y keeps the x layout) without touching the device.
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            for axis in (0, -1, 1):
                with self.subTest(axis=axis):
                    x = torch.empty(4, 8, 6, dtype=torch.float16, device="npu")
                    y, s = transpose_dynamic_quant(x, None, axis)
                    a = axis if axis >= 0 else 3 + axis
                    scale_shape = (4, 8, 6)[:a] + (4, 8, 6)[a + 1 :]
                    self.assertEqual(tuple(y.shape), (4, 8, 6))
                    self.assertEqual(y.dtype, torch.float8_e4m3fn)
                    self.assertEqual(tuple(s.shape), scale_shape)
                    self.assertEqual(s.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
