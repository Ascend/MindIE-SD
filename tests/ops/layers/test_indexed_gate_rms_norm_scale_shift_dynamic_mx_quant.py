#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import torch_npu
import triton
import triton.language as tl

from mindiesd.layers import indexed_gate_rms_norm_scale_shift_dynamic_mx_quant
from mindiesd.layers.indexed_gate_rms_norm_scale_shift_dynamic_mx_quant import (
    _use_hybrid,
)
from mindiesd.utils import ParametersInvalid


@triton.jit
def _reference_modulation_kernel(
    out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    row_offset,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_n: tl.constexpr,
    round_bf16: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
    cols = tl.arange(0, block_n)
    mask = cols < hidden_size
    index = tl.load(indices_ptr + row)
    residual = tl.load(residual_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    # 与融合 kernel 一致：门控残差先舍入到 input dtype 再回到 fp32 参与方差/归一化。
    input_t = tl.bfloat16 if round_bf16 else tl.float16
    updated = (residual + gate * branch).to(input_t).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(updated * updated, axis=0) / hidden_size
    normalized = updated * tl.rsqrt(variance + eps) * weight
    shift = tl.load(shift_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    affine_scale = tl.load(scale_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    result = normalized * (1.0 + affine_scale) + shift
    tl.store(out_ptr + row * hidden_size + cols, result, mask=mask)


def _reference(residual, gate, branch, weight, shift, scale, indices, eps, dst_type):
    rows, hidden = residual.shape
    modulated = torch.empty_like(residual)
    if rows:
        # block_n 对齐到 256/32，避免 hidden>=8193 时 next_power_of_2 触发 UB 溢出。
        block_n = (hidden + 255) // 256 * 256 if hidden > 512 else (hidden + 31) // 32 * 32
        # Ascend 限制 launch coreDim 为 uint16，rows>=65536 需按 65535 分块，与融合算子一致。
        for row_offset in range(0, rows, 65535):
            chunk_rows = min(rows - row_offset, 65535)
            _reference_modulation_kernel[(chunk_rows,)](
                modulated,
                residual,
                gate,
                branch,
                weight,
                shift,
                scale,
                indices,
                row_offset,
                hidden_size=hidden,
                eps=eps,
                block_n=block_n,
                round_bf16=residual.dtype == torch.bfloat16,
            )
    torch_npu_dst = {
        "float8_e4m3fn": torch_npu.float8_e4m3fn,
        "float8_e5m2": torch_npu.float8_e5m2,
        "float4_e2m1fn_x2": torch_npu.float4_e2m1fn_x2,
        "float4_e1m2fn_x2": torch_npu.float4_e1m2fn_x2,
    }[dst_type]
    y, mxscale = torch_npu.npu_dynamic_mx_quant(
        modulated,
        dst_type=torch_npu_dst,
        axis=-1,
        block_size=32,
        round_mode="rint",
        scale_alg=0,
    )
    return y, mxscale


def _bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


def _make_inputs(rows, hidden, conditions, dtype, pattern="randn", seed=0):
    torch.manual_seed(seed)
    if pattern == "zeros":
        residual = torch.zeros((rows, hidden), dtype=dtype, device="npu")
        gate = torch.zeros((conditions, hidden), dtype=dtype, device="npu")
        branch = torch.zeros((rows, hidden), dtype=dtype, device="npu")
        weight = torch.zeros(hidden, dtype=dtype, device="npu")
        shift = torch.ones((conditions, hidden), dtype=dtype, device="npu")
        scale = torch.full((conditions, hidden), -1.0, dtype=dtype, device="npu")
    elif pattern == "constant":
        residual = torch.ones((rows, hidden), dtype=dtype, device="npu")
        gate = torch.full((conditions, hidden), 0.5, dtype=dtype, device="npu")
        branch = torch.ones((rows, hidden), dtype=dtype, device="npu")
        weight = torch.full((hidden,), 1.125, dtype=dtype, device="npu")
        shift = torch.zeros((conditions, hidden), dtype=dtype, device="npu")
        scale = torch.full((conditions, hidden), 0.09375, dtype=dtype, device="npu")
    elif pattern == "saturate":
        residual = (torch.randn((rows, hidden), dtype=torch.float32, device="npu") * 1e4).to(dtype)
        gate = torch.randn((conditions, hidden), dtype=dtype, device="npu")
        branch = torch.randn((rows, hidden), dtype=dtype, device="npu")
        weight = torch.full((hidden,), 100.0, dtype=dtype, device="npu")
        shift = torch.full((conditions, hidden), 100.0, dtype=dtype, device="npu")
        scale = torch.full((conditions, hidden), 10.0, dtype=dtype, device="npu")
    else:
        residual = torch.randn((rows, hidden), dtype=dtype, device="npu")
        gate = torch.randn((conditions, hidden), dtype=dtype, device="npu")
        branch = torch.randn((rows, hidden), dtype=dtype, device="npu")
        weight = torch.randn(hidden, dtype=dtype, device="npu")
        shift = torch.randn((conditions, hidden), dtype=dtype, device="npu")
        scale = torch.randn((conditions, hidden), dtype=dtype, device="npu")
    if seed % 2:
        indices = torch.randint(0, conditions, (rows,), dtype=torch.int64, device="npu")
    else:
        indices = (torch.arange(rows, device="npu") % conditions).to(torch.int64)
    return residual, gate, branch, weight, shift, scale, indices


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestIndexedGateRmsNormScaleShiftDynamicMxQuant(unittest.TestCase):
    maxDiff = None

    def _assert_matches(
        self,
        dst_type,
        rows,
        hidden,
        conditions,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        residual, gate, branch, weight, shift, scale, indices = _make_inputs(rows, hidden, conditions, dtype, **kwargs)
        expected = _reference(residual, gate, branch, weight, shift, scale, indices, 1e-6, dst_type)
        actual = indexed_gate_rms_norm_scale_shift_dynamic_mx_quant(
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            1e-6,
            dst_type,
        )
        context = f"dst={dst_type} shape={(rows, hidden)} conditions={conditions} seed={kwargs.get('seed')}"
        self.assertEqual(actual[0].shape, expected[0].shape, context)
        self.assertEqual(actual[1].shape, expected[1].shape, context)
        self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])), context)
        self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])), context)

    def test_precision_matrix_bf16_fp8(self):
        cases = (
            (1, 32, 1, "randn", 0),
            (2, 33, 3, "randn", 1),
            (64, 32, 9, "randn", 15),
            (65, 33, 9, "randn", 16),
            (1024, 32, 9, "randn", 17),
            (1025, 33, 9, "randn", 18),
            (7, 100, 9, "randn", 2),
            (16, 128, 9, "constant", 4),
            (65, 512, 17, "randn", 5),
            (128, 1000, 9, "randn", 6),
            (257, 5376, 9, "randn", 7),
            (513, 5377, 9, "randn", 8),
            (128, 8192, 9, "randn", 9),
            (65, 6145, 9, "randn", 22),
            (9, 8193, 9, "randn", 23),
            (129, 1024, 9, "randn", 19),
            (65536, 160, 9, "randn", 20),
            (65535, 63, 9, "randn", 21),
            (512, 5375, 9, "randn", 11),
            (1023, 5377, 9, "randn", 12),
            (65535, 64, 9, "randn", 13),
            (65537, 96, 9, "randn", 14),
        )
        for rows, hidden, conditions, pattern, seed in cases:
            with self.subTest(rows=rows, hidden=hidden):
                self._assert_matches(
                    "float8_e4m3fn",
                    rows,
                    hidden,
                    conditions,
                    pattern=pattern,
                    seed=seed,
                )

    def test_saturation_and_zero_boundaries(self):
        for pattern in ("saturate", "zeros"):
            with self.subTest(pattern=pattern):
                self._assert_matches(
                    "float8_e4m3fn",
                    33,
                    1000,
                    9,
                    pattern=pattern,
                    seed=11,
                )

    def test_nonfinite_blocks(self):
        rows, hidden, conditions = 2, 64, 1
        residual = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
        residual[0, 0] = float("nan")
        residual[1, 0] = float("inf")
        gate = torch.full((conditions, hidden), 0.5, dtype=torch.bfloat16, device="npu")
        branch = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
        weight = torch.ones(hidden, dtype=torch.bfloat16, device="npu")
        shift = torch.zeros((conditions, hidden), dtype=torch.bfloat16, device="npu")
        scale = torch.zeros_like(shift)
        indices = torch.zeros(rows, dtype=torch.int64, device="npu")
        expected = _reference(
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            1e-6,
            "float8_e4m3fn",
        )
        actual = indexed_gate_rms_norm_scale_shift_dynamic_mx_quant(
            residual, gate, branch, weight, shift, scale, indices, 1e-6
        )
        self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
        self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_supported_dtype_combinations(self):
        dst_types = (
            "float8_e4m3fn",
            "float8_e5m2",
            "float4_e2m1fn_x2",
            "float4_e1m2fn_x2",
        )
        for dst_type in dst_types:
            with self.subTest(dst_type=dst_type):
                self._assert_matches(dst_type, 8, 5376, 9, seed=12)

    def test_fp16_input_matrix(self):
        # fp16 输入走 general 路径（E4M3/E5M2/E2M1/E1M2 四种量化）。
        cases = (
            ("float8_e4m3fn", 8, 5376, 9, 51),
            ("float8_e4m3fn", 1025, 33, 9, 52),
            ("float8_e4m3fn", 65535, 64, 9, 58),
            ("float8_e5m2", 8, 5376, 9, 53),
            ("float8_e5m2", 257, 5376, 9, 54),
            ("float8_e5m2", 65, 33, 9, 55),
            ("float4_e2m1fn_x2", 8, 5376, 9, 56),
            ("float4_e1m2fn_x2", 33, 5376, 9, 57),
        )
        for dst_type, rows, hidden, conditions, seed in cases:
            with self.subTest(dst_type=dst_type, rows=rows, hidden=hidden):
                self._assert_matches(dst_type, rows, hidden, conditions, dtype=torch.float16, seed=seed)

    def test_fp4_tiled_large_hidden(self):
        # H > 3072 触发 tile_n=2048 分块 kernel。
        for dst_type in ("float4_e2m1fn_x2", "float4_e1m2fn_x2"):
            with self.subTest(dst_type=dst_type):
                self._assert_matches(dst_type, 16, 4096, 9, seed=25)
                self._assert_matches(dst_type, 33, 5376, 9, seed=26)

    def test_fp4_negative_zero_and_empty_group(self):
        rows, hidden, conditions = 1, 64, 1
        for dst_type in ("float4_e2m1fn_x2", "float4_e1m2fn_x2"):
            with self.subTest(dst_type=dst_type):
                residual = torch.zeros((rows, hidden), dtype=torch.bfloat16, device="npu")
                residual[0, 0] = 1.0
                residual[0, 1] = -0.0
                gate = torch.full((conditions, hidden), 1.0, dtype=torch.bfloat16, device="npu")
                branch = torch.zeros((rows, hidden), dtype=torch.bfloat16, device="npu")
                weight = torch.ones(hidden, dtype=torch.bfloat16, device="npu")
                shift = torch.zeros((conditions, hidden), dtype=torch.bfloat16, device="npu")
                shift[0, 1] = -0.0
                scale = torch.zeros_like(shift)
                indices = torch.zeros(rows, dtype=torch.int64, device="npu")

                expected = _reference(
                    residual,
                    gate,
                    branch,
                    weight,
                    shift,
                    scale,
                    indices,
                    1e-6,
                    dst_type,
                )
                actual = indexed_gate_rms_norm_scale_shift_dynamic_mx_quant(
                    residual,
                    gate,
                    branch,
                    weight,
                    shift,
                    scale,
                    indices,
                    1e-6,
                    dst_type,
                )
                self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
                self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_fp4_byte254_scale(self):
        rows, hidden, conditions = 2, 64, 1
        for dst_type in ("float4_e2m1fn_x2", "float4_e1m2fn_x2"):
            with self.subTest(dst_type=dst_type):
                residual = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
                gate = torch.full(
                    (conditions, hidden),
                    0.5,
                    dtype=torch.bfloat16,
                    device="npu",
                )
                branch = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
                weight = torch.ones(hidden, dtype=torch.bfloat16, device="npu")
                shift = torch.full(
                    (conditions, hidden),
                    3.0e38,
                    dtype=torch.bfloat16,
                    device="npu",
                )
                scale = torch.zeros_like(shift)
                indices = torch.zeros(rows, dtype=torch.int64, device="npu")

                expected = _reference(
                    residual,
                    gate,
                    branch,
                    weight,
                    shift,
                    scale,
                    indices,
                    1e-6,
                    dst_type,
                )
                actual = indexed_gate_rms_norm_scale_shift_dynamic_mx_quant(
                    residual,
                    gate,
                    branch,
                    weight,
                    shift,
                    scale,
                    indices,
                    1e-6,
                    dst_type,
                )
                self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
                self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_model_shape_bfloat16_fp8(self):
        self._assert_matches("float8_e4m3fn", 56240, 5376, 9, seed=20261025)

    def test_hybrid_routing(self):
        # `_use_hybrid` is the shape-only gate: wide rows (hidden >= 3000) and
        # large launches take the two-segment Triton affine + CANN quant tail.
        self.assertTrue(_use_hybrid(1024, 5376))
        self.assertTrue(_use_hybrid(2047, 8192))
        self.assertFalse(_use_hybrid(65, 32))
        self.assertTrue(_use_hybrid(2048, 4096))
        self.assertTrue(_use_hybrid(56240, 5375))
        self.assertTrue(_use_hybrid(56240, 5376))
        self.assertFalse(_use_hybrid(9, 8193))
        self.assertFalse(_use_hybrid(7, 96))
        # Mid/long hidden dims fall back to hybrid even at small batch sizes,
        # fixing 256×8192 and 512×4096 degradation.
        self.assertTrue(_use_hybrid(256, 8192))
        self.assertTrue(_use_hybrid(512, 4096))
        # hidden=3072 also falls back to hybrid, fixing 256/512×3072 FP4 degradation.
        self.assertTrue(_use_hybrid(256, 3072))
        self.assertTrue(_use_hybrid(512, 3072))
        # Narrow 32-column rows fall back to hybrid, fixing 65535×32 degradation.
        self.assertTrue(_use_hybrid(65535, 32))

    def test_invalid_dst_type(self):
        args = _make_inputs(2, 64, 1, torch.bfloat16, seed=1)
        with self.assertRaises(ParametersInvalid):
            indexed_gate_rms_norm_scale_shift_dynamic_mx_quant(*args, 1e-6, "float8_e4m3fnz")

    def test_dst_type_alias(self):
        # 别名 float4_e2m1 -> float4_e2m1fn_x2 应等价。
        residual, gate, branch, weight, shift, scale, indices = _make_inputs(8, 5376, 9, torch.bfloat16, seed=41)
        a = indexed_gate_rms_norm_scale_shift_dynamic_mx_quant(
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            1e-6,
            "float4_e2m1fn_x2",
        )
        b = indexed_gate_rms_norm_scale_shift_dynamic_mx_quant(
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            1e-6,
            "float4_e2m1",
        )
        self.assertTrue(torch.equal(_bytes(a[0]), _bytes(b[0])))
        self.assertTrue(torch.equal(_bytes(a[1]), _bytes(b[1])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
