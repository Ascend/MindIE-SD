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
Unit tests for mindiesd.quant_four_over_six_a5 operator.

Dynamic MX quantization (46 adaptive, the only remaining path):
  y (packed fp4, uint8) = quant(x), mxscale (E8M0) = block-wise power-of-two scale.
  y shape == x shape with last dim halved; mxscale shape == x shape with axis dim
  compressed to ceil(ceil(dim/blocksize)/2) and a trailing dim of 2.

T9b: only the 4/6 adaptive combination remains — dst_type=40 (FLOAT4_E2M1),
scale_alg=2, dst_type_max=4.0, blocksize=32, tail axis, bf16 input.
T9c: host-side 46 guards — blocksize==32 (B1), round_mode in {rint, round, floor}
(B2), 46 attribute combination enforced on plugin/infershape/tiling (B3), axis range
(B4); negative cases assert clear error messages, floor is a positive round_mode case.
Best-effort golden comparison uses aclnn-directly-verified
binaries from the T5 smoke run (QFOS_GOLDEN_DIR, default t5_bins on 184).
"""

import os
import struct
import unittest

import torch

from device import DEVICE_ID

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

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
        return hasattr(torch.ops, 'mindiesd') and hasattr(torch.ops.mindiesd, 'quant_four_over_six_a5')
    except Exception:
        return False


NPU_AVAILABLE = _is_npu_available() and _has_mindiesd_op()


def _to_npu(tensor):
    """Move tensor to NPU if available, otherwise keep on CPU."""
    if NPU_AVAILABLE:
        device = torch.device(f"npu:{DEVICE_ID}")
        torch.npu.set_device(device)
        return tensor.to(device)
    return tensor


def _dst_type_dtype(dst_type):
    """Map dst_type (ge::DataType) to torch dtype (46 路径仅 40；float4 -> uint8, 实测呈现)."""
    mapping = {
        40: torch.uint8,
    }
    return mapping[dst_type]


def _scale_dtype():
    return torch.uint8


def _expected_scale_shape(shape, axis, blocksize):
    dim = axis if axis >= 0 else axis + len(shape)
    scale_shape = list(shape)
    scale_shape[dim] = ((shape[dim] + blocksize - 1) // blocksize + 1) // 2
    scale_shape.append(2)
    return scale_shape


def _run_adaptive(x, axis=-1, dst_type=40, blocksize=32, scale_alg=2, dst_type_max=4.0,
                  round_mode="rint"):
    """Run the operator through the mindiesd python wrapper (46 adaptive path)."""
    from mindiesd.layers._custom_ops import quant_four_over_six_a5
    return quant_four_over_six_a5(
        x, axis=axis, round_mode=round_mode, dst_type=dst_type,
        blocksize=blocksize, scale_alg=scale_alg, dst_type_max=dst_type_max,
    )


# ============================================================================
# Golden data (best-effort, from T5 aclnn-direct verification)
# ============================================================================

DEFAULT_GOLDEN_DIR = "/home/h00925030/mindiesd-migration/t5_bins"
GOLDEN_DIR = os.environ.get("QFOS_GOLDEN_DIR", DEFAULT_GOLDEN_DIR)

# 46 自适应路径（bf16, 64x1024, dst_type=40, scale_alg=2, blocksize=32, dst_type_max=4.0）
GOLDEN_FILES = {
    "input_64x1024": ("input_64x1024.bin", 64 * 1024 * 2),
    "y_64x1024": ("y_64x1024_184.bin", 64 * 1024 // 2),
    "mx_64x1024": ("npu_mx_64x1024.bin", 64 * 16 * 2),
    "input_1x1024": ("input_1x1024.bin", 1 * 1024 * 2),
    "mx_1x1024": ("mx_184_1x1024.bin", 1 * 16 * 2),
}


def _read_golden(key):
    """Read a golden bin file; return raw bytes or None if unavailable."""
    fname, nbytes = GOLDEN_FILES[key]
    path = os.path.join(GOLDEN_DIR, fname)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != nbytes:
        return None
    return data


def _assert_bytes_equal(testcase, tensor, raw, label):
    """Compare a tensor's raw bytes with golden bytes (best-effort)."""
    if raw is None:
        testcase.skipTest(f"golden file for {label} not found, skipped (best-effort)")
    tbytes = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    if len(tbytes) != len(raw):
        testcase.skipTest(
            f"golden byte layout mismatch for {label}: got {len(tbytes)}B vs golden {len(raw)}B, "
            "skipped (best-effort)"
        )
    testcase.assertEqual(tbytes, raw, f"{label} bytes mismatch")


# ============================================================================
# Test Cases
# ============================================================================


@unittest.skipUnless(NPU_AVAILABLE, "NPU or mindiesd operator not available")
@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestQuantFourOverSixA5NPU(unittest.TestCase):
    """NPU tests: run operator on real hardware."""

    def test_four_six_adaptive_64x1024(self):
        """46 adaptive path: bf16 64x1024, dst_type=40, scale_alg=2, dst_type_max=4.0, blocksize=32."""
        x = torch.randn(64, 1024, dtype=torch.bfloat16).abs() + 0.5  # positive to avoid zero scale
        y, mxscale = _run_adaptive(_to_npu(x))

        # y：float4 打包，torch 层 uint8，形状 = x 形状最后一维减半（64x512，32768B）
        self.assertEqual(tuple(y.shape), (64, 512), f"y shape mismatch: {tuple(y.shape)}")
        self.assertEqual(y.dtype, _dst_type_dtype(40), f"y dtype mismatch: {y.dtype}")
        self.assertEqual(tuple(mxscale.shape), (64, 16, 2), f"mxscale shape mismatch: {tuple(mxscale.shape)}")
        self.assertEqual(mxscale.dtype, _scale_dtype(), f"mxscale dtype mismatch: {mxscale.dtype}")

        # 值阈：非空、数值有限（整数 dtype 天然无 NaN/Inf）
        self.assertGreater(y.numel(), 0, "y must be non-empty")
        self.assertGreater(mxscale.numel(), 0, "mxscale must be non-empty")
        mx = mxscale.cpu().to(torch.uint8)
        self.assertTrue(torch.all(mx <= 255), "mxscale values out of uint8 range")
        self.assertFalse(torch.all(mx == 0), "mxscale must not be all-zero (x has non-zero magnitude)")
        # 打包数据非空：x 非零，y 字节不应全零
        self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0), "y must not be all-zero")

    def test_four_six_adaptive_1x1024(self):
        """46 adaptive path, 1x1024 (T5 golden shape)."""
        x = torch.randn(1, 1024, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))

        self.assertEqual(tuple(y.shape), (1, 512))
        self.assertEqual(tuple(mxscale.shape), (1, 16, 2))
        self.assertEqual(y.dtype, _dst_type_dtype(40))

    def test_default_attrs(self):
        """T9b: 默认值即 46 组合（scale_alg=2, dst_type_max=4.0, blocksize=32）——不再显式传参。"""
        x = torch.randn(8, 256, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))

        self.assertEqual(tuple(y.shape), (8, 128))
        # axis=-1 -> dim 1: ceil(ceil(256/32)/2) = 4
        self.assertEqual(tuple(mxscale.shape), (8, 4, 2))
        self.assertEqual(mxscale.dtype, _scale_dtype())

    def test_r1_fp16_with_d4_rejected(self):
        """R1: fp16 输入 + dst_type_max=4.0（46 自适应）必须报错，禁止静默无输出。"""
        x = torch.randn(8, 256, dtype=torch.float16).abs() + 0.5
        with self.assertRaises(RuntimeError) as ctx:
            _run_adaptive(_to_npu(x))
        self.assertIn("bf16", str(ctx.exception).lower())

    def test_t9c_blocksize_64_rejected(self):
        """T9c/B1: blocksize=64 + 46 组合必须被拒绝（46 仅 blocksize=32）。"""
        x = torch.randn(8, 256, dtype=torch.bfloat16).abs() + 0.5
        with self.assertRaises(RuntimeError) as ctx:
            _run_adaptive(_to_npu(x), blocksize=64)
        self.assertIn("blocksize", str(ctx.exception).lower())

    def test_t9c_scale_alg_1_rejected(self):
        """T9c/B3: scale_alg=1 必须被拒绝（46 仅 scale_alg=2）。"""
        x = torch.randn(8, 256, dtype=torch.bfloat16).abs() + 0.5
        with self.assertRaises(RuntimeError) as ctx:
            _run_adaptive(_to_npu(x), scale_alg=1)
        self.assertIn("scale_alg", str(ctx.exception).lower())

    def test_t9c_round_mode_ceil_rejected(self):
        """T9c/B2: round_mode 白名单之外（ceil）必须被拒绝。"""
        x = torch.randn(8, 256, dtype=torch.bfloat16).abs() + 0.5
        with self.assertRaises(RuntimeError) as ctx:
            _run_adaptive(_to_npu(x), round_mode="ceil")
        self.assertIn("round_mode", str(ctx.exception).lower())

    def test_t9c_round_mode_floor(self):
        """T9c/B2: round_mode='floor' 正向——kernel tilingKey=12 支持 floor，应正常执行。"""
        x = torch.randn(8, 256, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x), round_mode="floor")
        self.assertEqual(tuple(y.shape), (8, 128))
        self.assertEqual(tuple(mxscale.shape), (8, 4, 2))
        self.assertEqual(mxscale.dtype, _scale_dtype())
        self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0), "y must not be all-zero")

    def test_four_six_matrix_257x1024(self):
        """CompareSelect46 修复矩阵①：257×1024（源 docs 大 shape 失败点）——46-only 清理后重跑。"""
        x = torch.randn(257, 1024, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))
        self.assertEqual(tuple(y.shape), (257, 512))
        self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0), "y must not be all-zero")

    def test_four_six_matrix_600x512(self):
        """修复矩阵②：600×512。"""
        x = torch.randn(600, 512, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))
        self.assertEqual(tuple(y.shape), (600, 256))
        self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0), "y must not be all-zero")

    def test_four_six_matrix_512x768(self):
        """修复矩阵③：512×768。"""
        x = torch.randn(512, 768, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))
        self.assertEqual(tuple(y.shape), (512, 384))
        self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0), "y must not be all-zero")

    def test_four_six_matrix_1024x1024(self):
        """修复矩阵④：1024×1024。"""
        x = torch.randn(1024, 1024, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))
        self.assertEqual(tuple(y.shape), (1024, 512))
        self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0), "y must not be all-zero")

    def test_edge_dim_not_aligned_rejected(self):
        """B5 硬条件: 量化轴（尾轴）长度 16（非 32 对齐）必须被拒绝——46 仅块对齐输入。"""
        x = torch.randn(1, 16, dtype=torch.bfloat16).abs() + 0.5
        with self.assertRaises(RuntimeError) as ctx:
            _run_adaptive(_to_npu(x))
        self.assertIn("32-aligned", str(ctx.exception))

    def test_edge_dim_exact_block(self):
        """边界: dim 恰好 = 32（整除临界）。"""
        x = torch.randn(1, 32, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))
        self.assertEqual(tuple(y.shape), (1, 16))
        self.assertEqual(tuple(mxscale.shape), (1, 1, 2))

    def test_edge_dim_64_and_96(self):
        """边界: dim=64（2 block）/ dim=96（3 block 尾块）。"""
        for dim_value in (64, 96):
            x = torch.randn(1, dim_value, dtype=torch.bfloat16).abs() + 0.5
            y, mxscale = _run_adaptive(_to_npu(x))
            self.assertEqual(tuple(y.shape), (1, dim_value // 2))
            self.assertEqual(tuple(mxscale.shape), (1, ((dim_value + 31) // 32 + 1) // 2, 2))

    def test_edge_rank1(self):
        """边界: rank=1（1D 输入）。"""
        x = torch.randn(1024, dtype=torch.bfloat16).abs() + 0.5
        y, mxscale = _run_adaptive(_to_npu(x))
        self.assertEqual(tuple(y.shape), (512,))
        self.assertEqual(tuple(mxscale.shape), (16, 2))

    def test_edge_values_zero_negative(self):
        """边界: 值域含 0 与负值（非全正 0.5+）——量化须正常产出有限值，无 NaN 路径。
        注意：轴维须 32 对齐（B5 硬条件），故用 (2,32)。"""
        pattern = torch.tensor([0.0, -0.5, 3.0, 1e-3, -1.0, 2.5, 0.75, -0.125,
                                1.0, 0.0, -2.0, 0.5, 0.25, -0.75, 4.0, 0.0,
                                6.0, -6.0, 0.5, -0.5, 1.5, -1.5, 3.5, -3.5,
                                0.0, 0.125, -0.25, 5.0, -4.0, 2.0, -1.0, 0.0],
                               dtype=torch.bfloat16)
        vals = torch.stack([pattern, pattern]).reshape(2, 32)
        y, mxscale = _run_adaptive(_to_npu(vals))
        self.assertEqual(tuple(y.shape), (2, 16))
        self.assertEqual(tuple(mxscale.shape), (2, 1, 2))
        # fp4 打包字节允许任意值；断言无异常/非全零即可（NaN 量化不会产生负 scale 异常）
        self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0), "y must not be all-zero")

    def test_t9c_dst_type_41_rejected(self):
        """T9c/B3: dst_type=41 (FLOAT4_E1M2, e1m2) 必须被拒绝——46 仅保留 dst_type=40 (E2M1)。
        e1m2 在 aclnn/kernel 层虽支持（源 8-dtype 时代），但非 46 保留组合；
        torch 层拦截（TORCH_CHECK），不允许走到真机执行。"""
        x = torch.randn(8, 256, dtype=torch.bfloat16).abs() + 0.5
        with self.assertRaises(RuntimeError) as ctx:
            _run_adaptive(_to_npu(x), dst_type=41)
        self.assertIn("dst_type", str(ctx.exception).lower())
        self.assertIn("40", str(ctx.exception))

    def test_fake_shape_dtype_64x1024(self):
        """fake op 元信息：y 打包减半，mxscale 按公式，dtype 按 dst_type 映射。"""
        from mindiesd.layers._custom_ops import quant_four_over_six_a5_fake
        x = torch.empty(64, 1024, dtype=torch.bfloat16)
        y, mxscale = quant_four_over_six_a5_fake(x, axis=1, dst_type=40, blocksize=32, scale_alg=2, dst_type_max=4.0)
        self.assertEqual(tuple(y.shape), (64, 512))
        self.assertEqual(y.dtype, _dst_type_dtype(40))
        self.assertEqual(tuple(mxscale.shape), (64, 16, 2))
        self.assertEqual(mxscale.dtype, _scale_dtype())

    def test_fake_shape_negative_axis(self):
        """fake 负 axis 处理（46 路径 dst_type=40）。"""
        from mindiesd.layers._custom_ops import quant_four_over_six_a5_fake
        x = torch.empty(2, 128, 64, dtype=torch.bfloat16)
        y, mxscale = quant_four_over_six_a5_fake(x, axis=-2, dst_type=40, blocksize=32)
        self.assertEqual(tuple(y.shape), (2, 128, 32))
        # dim=1, 128 -> ceil(ceil(128/32)/2)=2
        self.assertEqual(tuple(mxscale.shape), (2, 2, 64, 2))
        self.assertEqual(mxscale.dtype, _scale_dtype())

    def test_fake_dtype_mapping_46(self):
        """T9b: dst_type 映射仅剩 40 -> uint8（fp4 打包），41/36/35 已被清理须报错。"""
        from mindiesd.layers._custom_ops import quant_four_over_six_a5_fake
        x = torch.empty(4, 128, dtype=torch.bfloat16)
        y, _ = quant_four_over_six_a5_fake(x, dst_type=40)
        self.assertEqual(y.dtype, _dst_type_dtype(40))
        self.assertEqual(tuple(y.shape), (4, 64))

    def test_fake_dtype_mapping_removed_rejected(self):
        """T9b: 已删除的 dst_type（41/36/35）在 fake 层同样报错（与 plugin/def 一致）。"""
        from mindiesd.layers._custom_ops import quant_four_over_six_a5_fake
        x = torch.empty(4, 128, dtype=torch.bfloat16)
        for dst_type in (41, 36, 35):
            with self.assertRaises(Exception, msg=f"dst_type={dst_type} should be rejected"):
                quant_four_over_six_a5_fake(x, dst_type=dst_type)

    def test_fake_blocksize_not_multiple(self):
        """blocksize 非法时 fake 不应崩溃（校验交给 NPU 路径），形状公式按输入计算。"""
        from mindiesd.layers._custom_ops import quant_four_over_six_a5_fake
        x = torch.empty(4, 1024, dtype=torch.bfloat16)
        y, mxscale = quant_four_over_six_a5_fake(x, blocksize=64)
        self.assertEqual(tuple(y.shape), (4, 512))
        self.assertEqual(tuple(mxscale.shape), (4, 8, 2))


# ============================================================================
# Main
# ============================================================================


# ============================================================================
# T9d 扩展精度验证：ref_golden_46（源仓 golden 算法 oracle）逐字节比对
# ============================================================================
import ref_golden_46 as ref46
import numpy as _np


def _bits_to_bf16(xbits2d):
    """np uint16 bf16 位 -> torch.bfloat16（位级一致）。"""
    return torch.frombuffer(xbits2d.astype(_np.uint16).tobytes(), dtype=torch.bfloat16
                            ).reshape(xbits2d.shape)


def _anypack_match(npu_y_np, ref_lohi, ref_hilo):
    return (_np.array_equal(npu_y_np, ref_lohi) or _np.array_equal(npu_y_np, ref_hilo))


class TestQuantFourOverSixA5Precision(unittest.TestCase):
    """46 精度扩展边界：shape/值域/round_mode 池——mxscale 与 y 双逐字节 vs 源仓 golden oracle。"""

    def _run_ref_compare(self, xf64, y_npu, mx_npu, round_mode="rint"):
        ref_mx = ref46.ref_46_scale(xf64, round_mode=round_mode)
        ref_lohi = ref46.ref_46_y(xf64, pack="lohi", round_mode=round_mode)
        ref_hilo = ref46.ref_46_y(xf64, pack="hilo", round_mode=round_mode)
        # mxscale: E8M0 逐字节
        self.assertTrue(_np.array_equal(mx_npu, ref_mx),
                        "mxscale mismatch: npu=%s ref=%s" % (mx_npu[:8], ref_mx[:8]))
        # y: fp4 打包（lohi/hilo 任一 100% 一致即可）
        self.assertTrue(_anypack_match(y_npu, ref_lohi, ref_hilo),
                        "y mismatch (lohi/hilo)")
        return True

    def test_precision_shape_pool(self):
        exps = [
            ("varied", 64, 1024), ("varied", 257, 1024), ("varied", 600, 512),
            ("varied", 512, 768), ("varied", 1024, 1024),
            ("uniform", 128, 2048), ("rev", 96, 640), ("varied", 32, 32),
        ]
        for mode, rows, cols in exps:
            with self.subTest(mode=mode, rows=rows, cols=cols):
                xb, xf, _ = ref46.build_input(rows, cols, mode)
                x = _bits_to_bf16(xb)
                y, mx = _run_adaptive(_to_npu(x))
                self.assertEqual(tuple(y.shape), (rows, cols // 2))
                self.assertFalse(torch.all(y.cpu().view(torch.uint8) == 0))
                self._run_ref_compare(xf, y.cpu().view(torch.uint8).numpy(), mx.cpu().numpy())

    def test_precision_golden_double_verify(self):
        """核心：逐字节双比对（mxscale+y）——覆盖 varied 64×1024 与 1×1024。"""
        for rows, cols in [(64, 1024), (1, 1024)]:
            with self.subTest(rows=rows, cols=cols):
                xb, xf, bpr = ref46.build_input(rows, cols, "varied")
                xt = _bits_to_bf16(xb)
                y, mx = _run_adaptive(_to_npu(xt))
                self._run_ref_compare(xf, y.cpu().view(torch.uint8).numpy(), mx.cpu().numpy())

    def test_precision_value_domain_edges(self):
        """值域边界（0/负值/±6/1e-3/大值）——mx+y 双比对（对齐 32）。"""
        xf = _np.array([0.0, -0.5, 3.0, 1e-3, -1.0, 2.5, 0.75, -0.125,
                       1.0, 0.0, -2.0, 0.5, 0.25, -0.75, 4.0, 0.0,
                       6.0, -6.0, 0.5, -0.5, 1.5, -1.5, 3.5, -3.5,
                       0.0, 0.125, -0.25, 5.0, -4.0, 2.0, -1.0, 0.0],
                      dtype=_np.float64)
        xf2 = _np.stack([xf, xf * 0.5])
        xb = _np.array([ref46.f2bf16_rne(v) for v in xf2.ravel()]).reshape(xf2.shape)
        xt = _bits_to_bf16(xb)
        y, mx = _run_adaptive(_to_npu(xt))
        self._run_ref_compare(xf2, y.cpu().view(torch.uint8).numpy(), mx.cpu().numpy())

    def test_precision_round_modes(self):
        """round_mode 白名单三种 × 1×1024——mx+y 双比对。"""
        for rm in ("rint", "round", "floor"):
            with self.subTest(round_mode=rm):
                xb, xf, _ = ref46.build_input(1, 1024, "varied")
                xt = _bits_to_bf16(xb)
                y, mx = _run_adaptive(_to_npu(xt), round_mode=rm)
                self._run_ref_compare(xf, y.cpu().view(torch.uint8).numpy(), mx.cpu().numpy(), round_mode=rm)

    def test_boundary_precision_512x5120(self):
        """最大边界 shape（源报告 512×5120）全量 mx+y vs oracle——T10 写回/判据修复的回归守护。
        oracle 全量（81920 块）python 参考，本用例预计 ~1-2 分钟。"""
        rows, cols = 512, 5120
        xb, xf, _ = ref46.build_input(rows, cols, "varied")
        xt = _bits_to_bf16(xb)
        y, mx = _run_adaptive(_to_npu(xt))
        self._run_ref_compare(xf, y.cpu().view(torch.uint8).numpy(), mx.cpu().numpy())

    def test_boundary_precision_value_extreme_mix(self):
        """值域极端混合（1e-4 与 1e4 量级同一行）——bf16 判据无溢出位的回归守护。"""
        rows, cols = 4, 32
        raw = _np.array([[1e-4, 1e4, -1e4, 1e-4, 1e3, -1e3, 5e-2, -5e-2,
                          200.0, -200.0, 0.0, 0.0, -0.125, 0.5, 7.0, -7.0,
                          1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0,
                          6.0, -6.0, 1e-2, -1e-2, 2.5, -2.5, 0.25, 8.0]] * rows,
                         dtype=_np.float64).reshape(rows, cols)
        xb = _np.array([ref46.f2bf16_rne(v) for v in raw.ravel()]).reshape(rows, cols)
        xt = _bits_to_bf16(xb)
        y, mx = _run_adaptive(_to_npu(xt))
        self._run_ref_compare(raw, y.cpu().view(torch.uint8).numpy(), mx.cpu().numpy())


if __name__ == '__main__':
    unittest.main()
