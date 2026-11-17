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

import multiprocessing
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch_npu
import triton
import triton.language as tl

from mindiesd.layers import rms_norm_indexed_scale_shift_dynamic_mx_quant
from mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant import (
    _e4m3_rows_per_prog,
    _fused_e4m3,
    _fused_general_mx,
)
from mindiesd.utils import ParametersInvalid

# The package re-exports the operator under the module's name, so the module
# itself is looked up through ``sys.modules`` (used to patch ``_MAX_GRID``).
_fused_module = sys.modules["mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant"]


EPS = 1e-6
FP8 = "float8_e4m3fn"
FP8_E5M2 = "float8_e5m2"
FP4 = "float4_e2m1fn_x2"
FP4_E1M2 = "float4_e1m2fn_x2"

_NPU_DST_TYPES = {
    FP8: torch_npu.float8_e4m3fn,
    FP8_E5M2: torch_npu.float8_e5m2,
    FP4: torch_npu.float4_e2m1fn_x2,
    FP4_E1M2: torch_npu.float4_e1m2fn_x2,
}


class _KernelLaunchSpy:
    """Record grid and row offset before delegating to a Triton kernel."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.grids = []
        self.row_offsets = []

    def __getitem__(self, grid):
        self.grids.append(grid)

        def launch(*args, **kwargs):
            self.row_offsets.append(args[7])
            return self.kernel[grid](*args, **kwargs)

        return launch


class _LaunchCountSpy:
    def __init__(self, kernel):
        self.kernel = kernel
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return self.kernel[grid]


def _invalid_index_worker(rows, hidden, dst_type, bad_index, workdir):
    os.chdir(workdir)
    x = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
    weight = torch.ones(hidden, dtype=torch.bfloat16, device="npu")
    shift = torch.ones((1, hidden), dtype=torch.bfloat16, device="npu")
    scale = torch.zeros((1, hidden), dtype=torch.bfloat16, device="npu")
    indices = torch.full((rows,), bad_index, dtype=torch.int64, device="npu")
    try:
        rms_norm_indexed_scale_shift_dynamic_mx_quant(
            x,
            weight,
            shift,
            scale,
            indices,
            EPS,
            dst_type,
        )
        torch.npu.synchronize()
    except RuntimeError as error:
        message = str(error)
        with open("invalid_index_result.txt", "w", encoding="utf-8") as stream:
            stream.write(message)
        expected = "507035" in message or "device error type 3" in message or "NPU function error" in message
        os._exit(0 if expected else 2)
    os._exit(1)


@triton.jit
def _reference_modulation_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    row_offset,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
    cols = tl.arange(0, block_n)
    mask = cols < hidden_size
    index = tl.load(indices_ptr + row)
    x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    normalized = x * tl.rsqrt(variance + eps) * weight
    shift = tl.load(shift_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    affine_scale = tl.load(scale_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    result = normalized * (1.0 + affine_scale) + shift
    tl.store(out_ptr + row * hidden_size + cols, result, mask=mask)


def _reference(x, weight, shift, scale, indices, eps, dst_type):
    """Reference: one Triton modulation kernel + the native MX quantizer."""
    rows, hidden = x.shape
    modulated = torch.empty_like(x)
    for row_offset in range(0, rows, _fused_module._MAX_GRID):
        chunk_rows = min(rows - row_offset, _fused_module._MAX_GRID)
        _reference_modulation_kernel[(chunk_rows,)](
            modulated,
            x,
            weight,
            shift,
            scale,
            indices,
            row_offset,
            hidden_size=hidden,
            eps=eps,
            block_n=triton.next_power_of_2(hidden),
        )
    y, mxscale = torch_npu.npu_dynamic_mx_quant(
        modulated,
        dst_type=_NPU_DST_TYPES[dst_type],
        axis=-1,
        block_size=32,
        round_mode="rint",
        scale_alg=0,
    )
    return y, mxscale


def _invoke(impl, x, weight, shift, scale, indices, dst_type):
    """Call the public operator or one explicit internal implementation.

    ``fused`` forces the single fused kernel (the hybrid MX tail is switched off),
    ``hybrid`` calls the two-segment path directly, and ``public`` uses the
    shipped shape-only dispatch.
    """
    if impl == "public":
        return rms_norm_indexed_scale_shift_dynamic_mx_quant(x, weight, shift, scale, indices, EPS, dst_type)
    if impl == "hybrid":
        return _fused_module._hybrid_mx(x, weight, shift, scale, indices, EPS, dst_type)
    with mock.patch.object(_fused_module, "_HYBRID_MIN_ROWS", 10**9):
        if x.dtype == torch.bfloat16 and dst_type == FP8:
            return _fused_e4m3(x, weight, shift, scale, indices, EPS, dst_type)
        return _fused_general_mx(x, weight, shift, scale, indices, EPS, dst_type)


def _bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


def _code_delta(got, expected):
    """Per-byte FP8/FP4 code distance (both tensors are compared as bytes)."""
    return (got.to(torch.int16) - expected.to(torch.int16)).abs()


def _make_inputs(rows, hidden, conditions, dtype, pattern="randn", seed=0):
    torch.manual_seed(seed)
    if pattern == "zeros":
        x = torch.zeros((rows, hidden), dtype=dtype, device="npu")
        weight = torch.zeros(hidden, dtype=dtype, device="npu")
        shift = torch.ones((conditions, hidden), dtype=dtype, device="npu")
        affine_scale = torch.full((conditions, hidden), -1.0, dtype=dtype, device="npu")
    elif pattern == "constant":
        x = torch.ones((rows, hidden), dtype=dtype, device="npu")
        weight = torch.full((hidden,), 1.125, dtype=dtype, device="npu")
        shift = torch.zeros((conditions, hidden), dtype=dtype, device="npu")
        affine_scale = torch.full((conditions, hidden), 0.09375, dtype=dtype, device="npu")
    elif pattern == "saturate":
        x = (torch.randn((rows, hidden), dtype=torch.float32, device="npu") * 1e4).to(dtype)
        weight = torch.full((hidden,), 100.0, dtype=dtype, device="npu")
        shift = torch.full((conditions, hidden), 100.0, dtype=dtype, device="npu")
        affine_scale = torch.full((conditions, hidden), 10.0, dtype=dtype, device="npu")
    else:
        x = torch.randn((rows, hidden), dtype=dtype, device="npu")
        weight = torch.randn(hidden, dtype=dtype, device="npu")
        shift = torch.randn((conditions, hidden), dtype=dtype, device="npu")
        affine_scale = torch.randn((conditions, hidden), dtype=dtype, device="npu")
    if seed % 2:
        indices = torch.randint(0, conditions, (rows,), dtype=torch.int64, device="npu")
    else:
        indices = (torch.arange(rows, device="npu") % conditions).to(torch.int64)
    return x, weight, shift, affine_scale, indices


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestRmsNormIndexedScaleShiftDynamicMxQuant(unittest.TestCase):
    maxDiff = None

    def test_dispatch_rows_per_program(self):
        """Lock the row-tile choice of the E4M3 dispatch (no device needed).

        Only tiny blocks (``block_n <= 64``) pack rows; every wider block keeps
        one row per program so that no dispatch path can move a ``y`` byte.
        """
        cases = (
            (1024, 1024, 8192, 0),
            (1024, 1024, 1024, 0),
            (1024, 1024, 256, 0),
            (512, 512, 8192, 0),
            (2048, 2048, 1024, 0),
            (64, 64, 65535, 128),
            (32, 32, 65535, 256),
            (64, 34, 65535, 0),
            (64, 64, 64, 16),
            (32, 32, 64, 0),
        )
        for block_n, hidden, rows, expected in cases:
            with self.subTest(block_n=block_n, hidden=hidden, rows=rows):
                self.assertEqual(_e4m3_rows_per_prog(block_n, hidden, rows), expected)

    def _assert_matches(
        self,
        dtype,
        dst_type,
        rows,
        hidden,
        conditions,
        impl="public",
        **kwargs,
    ):
        """Compare against the reference chain.  Every path must be byte-exact.

        There is no tolerance knob: ``y`` and ``mxscale`` must match the reference
        byte for byte.  A ``y`` mismatch means the kernel changed a rounding or a
        reduction order, which is never accepted (the operator must not need an
        environment switch to make a case "accurate enough").
        """
        x, weight, shift, scale, indices = _make_inputs(rows, hidden, conditions, dtype, **kwargs)
        expected = _reference(x, weight, shift, scale, indices, EPS, dst_type)
        actual = _invoke(impl, x, weight, shift, scale, indices, dst_type)
        context = (
            f"dtype={dtype} dst={dst_type} shape={(rows, hidden)} "
            f"conditions={conditions} impl={impl} seed={kwargs.get('seed')}"
        )
        self.assertEqual(len(actual), 2, context)
        self.assertEqual(actual[0].shape, expected[0].shape, context)
        self.assertEqual(actual[1].shape, expected[1].shape, context)
        expected_y, actual_y = _bytes(expected[0]), _bytes(actual[0])
        delta = _code_delta(actual_y, expected_y)
        mismatches = int((delta > 0).sum())
        self.assertEqual(
            mismatches,
            0,
            f"{context} {mismatches}/{expected_y.numel()} y bytes differ "
            f"(worst code delta {int(delta.max()) if delta.numel() else 0})",
        )
        # ``mxscale`` is an exponent-only byte, so it must always be exact.
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
            # Row blocks wider than a tiny block always keep one row per program,
            # so these used to be the level-2 tile: they must now be byte-exact.
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
                    torch.bfloat16,
                    FP8,
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
                    torch.bfloat16,
                    FP8,
                    33,
                    1000,
                    9,
                    pattern=pattern,
                    seed=11,
                )

    def test_nonfinite_blocks(self):
        rows, hidden, conditions = 2, 64, 1
        x = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
        x[0, 0] = float("nan")
        x[1, 0] = float("inf")
        weight = torch.ones(hidden, dtype=torch.bfloat16, device="npu")
        shift = torch.zeros((conditions, hidden), dtype=torch.bfloat16, device="npu")
        scale = torch.zeros_like(shift)
        indices = torch.zeros(rows, dtype=torch.int64, device="npu")
        expected = _reference(x, weight, shift, scale, indices, EPS, FP8)
        actual = rms_norm_indexed_scale_shift_dynamic_mx_quant(x, weight, shift, scale, indices, EPS)
        self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
        self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_hidden_one(self):
        for rows in (1, 64):
            with self.subTest(rows=rows):
                self._assert_matches(
                    torch.bfloat16,
                    FP8,
                    rows,
                    1,
                    9,
                    seed=7000 + rows,
                )

    def test_supported_dtype_combinations(self):
        for dtype in (torch.bfloat16, torch.float16):
            for dst_type in (FP8, FP8_E5M2, FP4, FP4_E1M2):
                with self.subTest(dtype=dtype, dst_type=dst_type):
                    self._assert_matches(dtype, dst_type, 8, 5376, 9, seed=12)

    def test_fused_tiny_hidden_all_dtype_combinations(self):
        for rows, hidden in ((64, 32), (65, 34)):
            for dtype in (torch.bfloat16, torch.float16):
                for dst_type in (FP8, FP8_E5M2, FP4, FP4_E1M2):
                    with self.subTest(rows=rows, hidden=hidden, dtype=dtype, dst_type=dst_type):
                        self._assert_matches(
                            dtype,
                            dst_type,
                            rows,
                            hidden,
                            9,
                            impl="fused",
                            seed=17,
                        )

    def test_fp16_dynamic_mx_rounding_regressions(self):
        cases = (
            (65, 34, FP8_E5M2, 20269578),
            (56240, 1000, FP8_E5M2, 27629469),
            (2047, 8192, FP4, 20537378),
            (56240, 5376, FP4_E1M2, 29633851),
        )
        for rows, hidden, dst_type, seed in cases:
            with self.subTest(rows=rows, hidden=hidden, dst_type=dst_type):
                self._assert_matches(
                    torch.float16,
                    dst_type,
                    rows,
                    hidden,
                    9,
                    impl="fused",
                    seed=seed,
                )

    def test_fp16_mxfp4_negative_zero_and_empty_group(self):
        rows, hidden, conditions = 1, 64, 1
        for dst_type in (FP4, FP4_E1M2):
            with self.subTest(dst_type=dst_type):
                x = torch.zeros((rows, hidden), dtype=torch.float16, device="npu")
                x[0, 0] = 1.0
                x[0, 1] = -0.0
                weight = torch.ones(hidden, dtype=torch.float16, device="npu")
                shift = torch.zeros((conditions, hidden), dtype=torch.float16, device="npu")
                shift[0, 1] = -0.0
                affine_scale = torch.zeros_like(shift)
                indices = torch.zeros(rows, dtype=torch.int64, device="npu")
                self.assertNotEqual(int(x.view(torch.uint16)[0, 1].item()) & 0x8000, 0)

                expected = _reference(x, weight, shift, affine_scale, indices, EPS, dst_type)
                actual = _invoke("fused", x, weight, shift, affine_scale, indices, dst_type)
                self.assertEqual((int(_bytes(expected[0])[0, 0].item()) >> 4) & 0xF, 8)
                self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
                self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_bf16_mxfp4_byte254_scale(self):
        rows, hidden, conditions = 2, 64, 1
        x = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
        weight = torch.ones(hidden, dtype=torch.bfloat16, device="npu")
        shift = torch.full((conditions, hidden), 3.0e38, dtype=torch.bfloat16, device="npu")
        scale = torch.zeros_like(shift)
        indices = torch.zeros(rows, dtype=torch.int64, device="npu")
        for dst_type in (FP4, FP4_E1M2):
            with self.subTest(dst_type=dst_type):
                expected = _reference(x, weight, shift, scale, indices, EPS, dst_type)
                actual = _invoke("fused", x, weight, shift, scale, indices, dst_type)
                self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
                self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_model_shape_bfloat16_fp8(self):
        self._assert_matches(torch.bfloat16, FP8, 56240, 5376, 9, seed=20261025)

    def test_non_contiguous_inputs(self):
        rows, hidden, conditions = 65, 1000, 9
        base_x, base_w, base_shift, base_scale, indices = _make_inputs(
            rows, 2 * hidden, conditions, torch.bfloat16, seed=41
        )
        x = base_x[:, ::2]
        weight = base_w[::2]
        shift = base_shift[:, ::2]
        scale = base_scale[:, ::2]
        self.assertFalse(x.is_contiguous())
        expected = _reference(
            x.contiguous(),
            weight.contiguous(),
            shift.contiguous(),
            scale.contiguous(),
            indices,
            EPS,
            FP8,
        )
        actual = rms_norm_indexed_scale_shift_dynamic_mx_quant(x, weight, shift, scale, indices, EPS)
        self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
        self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_backend_parameter_is_not_exposed(self):
        # The fixed fused implementation must not leak an implementation-selection
        # parameter into the public op schema.
        schema = torch.ops.mindiesd.rms_norm_indexed_scale_shift_dynamic_mx_quant.default._schema
        names = [argument.name for argument in schema.arguments]
        self.assertEqual(
            names,
            ["x", "weight", "shift", "scale", "indices", "eps", "dst_type"],
        )
        x, weight, shift, scale, indices = _make_inputs(2, 32, 1, torch.bfloat16, seed=1)
        with self.assertRaises((TypeError, RuntimeError)):
            rms_norm_indexed_scale_shift_dynamic_mx_quant(x, weight, shift, scale, indices, backend="fused")

    def test_returns_quantized_outputs_only(self):
        # The AdaLN activation has no consumer in the quantized path, so the
        # fused op materialises only the quantized pair.
        x, weight, shift, scale, indices = _make_inputs(1024, 5375, 3, torch.bfloat16, seed=7)
        y, mxscale = rms_norm_indexed_scale_shift_dynamic_mx_quant(x, weight, shift, scale, indices, EPS)
        self.assertEqual(y.shape, (1024, 5375))
        self.assertEqual(y.dtype, torch.float8_e4m3fn)
        self.assertEqual(mxscale.shape, (1024, 84, 2))
        self.assertEqual(mxscale.dtype, torch.float8_e8m0fnu)

    def test_no_dispatch_path_moves_a_byte(self):
        """Every dispatch path (full-row, tiny multi-row, hybrid tail) is byte-exact.

        These shapes cover the decisions in ``_e4m3_rows_per_prog`` and the wide
        row hybrid, including the shapes whose multi-row tile used to trade a
        few ``y`` bytes for speed: they must all match the reference byte for byte
        without any tolerance switch.
        """
        cases = (
            (256, 1024),
            (1024, 1000),
            (4096, 160),
            (56240, 1000),
            (65536, 160),
            (1024, 1024),
            (8192, 1024),
            (1024, 5375),
            # Padded tiny block on a long batch: the wide 2-D tile moved a group
            # scale by one E8M0 step, so this shape must stay full-row.
            (65536, 63),
        )
        for rows, hidden in cases:
            with self.subTest(rows=rows, hidden=hidden):
                self._assert_matches(torch.bfloat16, FP8, rows, hidden, 9, seed=31)

    def test_hybrid_mxquant_byte_exact(self):
        """The two-segment MX tail must reproduce the reference chain byte for byte.

        All eight dtype combinations go through the CANN quantizer, on both an
        unpadded block (5376) and padded blocks (5374, 3000, 1000).
        """
        shapes = ((1024, 3000), (1024, 5374), (1025, 5376), (56240, 1000))
        for dtype in (torch.bfloat16, torch.float16):
            for dst_type in (FP8, FP8_E5M2, FP4, FP4_E1M2):
                for rows, hidden in shapes:
                    with self.subTest(dtype=dtype, dst_type=dst_type, shape=(rows, hidden)):
                        self._assert_matches(dtype, dst_type, rows, hidden, 9, impl="hybrid", seed=31)

    def test_hybrid_mxquant_wide_rows_byte_exact(self):
        """Wide rows quantize through CANN; the bytes must match the single kernel."""
        for rows, hidden in ((1024, 5376), (2048, 4096), (8192, 3072)):
            with self.subTest(rows=rows, hidden=hidden):
                x, weight, shift, scale, indices = _make_inputs(rows, hidden, 9, torch.bfloat16, seed=31)
                hybrid = _fused_e4m3(x, weight, shift, scale, indices, EPS, FP8)
                # Raise the row floor so the same operator takes the single
                # fused kernel instead of the CANN tail.
                with mock.patch.object(_fused_module, "_HYBRID_MIN_ROWS", 10**9):
                    fused = _fused_e4m3(x, weight, shift, scale, indices, EPS, FP8)
                reference = _reference(x, weight, shift, scale, indices, EPS, FP8)
                for expected, name in ((fused, "single kernel"), (reference, "reference chain")):
                    self.assertTrue(
                        torch.equal(_bytes(hybrid[0]), _bytes(expected[0])),
                        f"y differs from the {name}",
                    )
                    self.assertTrue(
                        torch.equal(_bytes(hybrid[1]), _bytes(expected[1])),
                        f"mxscale differs from the {name}",
                    )
                self.assertEqual(hybrid[1].dtype, torch.float8_e8m0fnu)

    def test_hybrid_mxquant_gate_dispatch(self):
        """The gate decides between the CANN tail and the single fused kernel."""
        calls = []
        real_quant = _fused_module.torch_npu.npu_dynamic_mx_quant

        def spy(*args, **kwargs):
            calls.append(1)
            return real_quant(*args, **kwargs)

        # (rows, hidden) -> the CANN quantizer must be called
        cases = (
            ((1024, 5376), True),  # wide row, unpadded block
            ((1024, 5375), True),  # wide row, padded block
            ((1024, 3000), True),  # wide row, padded block
            ((56240, 1000), True),  # narrow row, padded block, large launch
            ((512, 3072), False),  # below the row floor
            ((256, 5376), False),  # below the row floor
            ((1024, 1024), False),  # narrow row, unpadded block
            ((1024, 1000), False),  # narrow row, launch too small
            ((16384, 1024), False),  # narrow row, unpadded block
        )
        with mock.patch.object(_fused_module.torch_npu, "npu_dynamic_mx_quant", side_effect=spy):
            for (rows, hidden), expect_cann in cases:
                with self.subTest(rows=rows, hidden=hidden):
                    calls.clear()
                    x, weight, shift, scale, indices = _make_inputs(rows, hidden, 9, torch.bfloat16, seed=31)
                    _fused_e4m3(x, weight, shift, scale, indices, EPS, FP8)
                    self.assertEqual(len(calls), 1 if expect_cann else 0)

    def test_multirow_grid_chunks_preserve_output(self):
        """Splitting the launch at ``_MAX_GRID`` must not change the bytes."""
        x, weight, shift, scale, indices = _make_inputs(4096, 64, 9, torch.bfloat16, seed=31)
        # rows=4096 with an unpadded 64-wide block packs 128 rows per program,
        # so a grid limit of 1/4 forces several launches.
        reference = _fused_e4m3(x, weight, shift, scale, indices, EPS, FP8)
        for max_grid in (1, 4):
            with mock.patch.object(_fused_module, "_MAX_GRID", max_grid):
                chunked = _fused_e4m3(x, weight, shift, scale, indices, EPS, FP8)
            with self.subTest(max_grid=max_grid):
                self.assertTrue(torch.equal(_bytes(chunked[0]), _bytes(reference[0])))
                self.assertTrue(torch.equal(_bytes(chunked[1]), _bytes(reference[1])))

    def test_general_fullrow_grid_chunks_preserve_output(self):
        """The general full-row path must honor ``_MAX_GRID`` row chunks."""
        x, weight, shift, scale, indices = _make_inputs(8, 1024, 9, torch.bfloat16, seed=32)
        expected = _reference(x, weight, shift, scale, indices, EPS, FP8_E5M2)
        spy = _KernelLaunchSpy(_fused_module._general_fullrow_kernel)
        with (
            mock.patch.object(_fused_module, "_MAX_GRID", 3),
            mock.patch.object(_fused_module, "_general_fullrow_kernel", spy),
        ):
            actual = _fused_general_mx(x, weight, shift, scale, indices, EPS, FP8_E5M2)
        self.assertEqual(spy.grids, [(3,), (3,), (2,)])
        self.assertEqual(spy.row_offsets, [0, 3, 6])
        self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
        self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_general_tiled_grid_chunks_preserve_output(self):
        """The general tiled path must honor ``_MAX_GRID`` row chunks."""
        x, weight, shift, scale, indices = _make_inputs(8, 8192, 9, torch.bfloat16, seed=33)
        expected = _reference(x, weight, shift, scale, indices, EPS, FP8_E5M2)
        spy = _KernelLaunchSpy(_fused_module._general_tiled_kernel)
        with (
            mock.patch.object(_fused_module, "_MAX_GRID", 3),
            mock.patch.object(_fused_module, "_general_tiled_kernel", spy),
        ):
            actual = _fused_general_mx(x, weight, shift, scale, indices, EPS, FP8_E5M2)
        self.assertEqual(spy.grids, [(3,), (3,), (2,)])
        self.assertEqual(spy.row_offsets, [0, 3, 6])
        self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
        self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_general_fullrow_rows_above_max_grid(self):
        """Cover the reachable general-path failure reported by review."""
        rows, hidden = 65536, 1024
        x, weight, shift, scale, indices = _make_inputs(rows, hidden, 9, torch.bfloat16, seed=34)
        expected = _reference(x, weight, shift, scale, indices, EPS, FP8_E5M2)
        spy = _KernelLaunchSpy(_fused_module._general_fullrow_kernel)
        with mock.patch.object(_fused_module, "_general_fullrow_kernel", spy):
            actual = _fused_general_mx(x, weight, shift, scale, indices, EPS, FP8_E5M2)
        self.assertEqual(spy.grids, [(65535,), (1,)])
        self.assertEqual(spy.row_offsets, [0, 65535])
        self.assertTrue(torch.equal(_bytes(actual[0]), _bytes(expected[0])))
        self.assertTrue(torch.equal(_bytes(actual[1]), _bytes(expected[1])))

    def test_invalid_inputs(self):
        x, weight, shift, scale, indices = _make_inputs(2, 32, 1, torch.bfloat16, seed=1)

        def call(**overrides):
            arguments = {
                "x": x,
                "weight": weight,
                "shift": shift,
                "scale": scale,
                "indices": indices,
                "eps": EPS,
                "dst_type": FP8,
            }
            arguments.update(overrides)
            return rms_norm_indexed_scale_shift_dynamic_mx_quant(**arguments)

        cases = (
            ("unsupported dst_type", {"dst_type": "float8_e8m0"}),
            (
                "odd hidden for MXFP4",
                {
                    "dst_type": FP4,
                    "x": x[:, :-1],
                    "scale": scale[:, :-1],
                    "shift": shift[:, :-1],
                    "weight": weight[:-1],
                },
            ),
            ("dtype mismatch", {"scale": scale.float()}),
            ("shape mismatch", {"weight": weight[:-1]}),
            ("non-positive eps", {"eps": 0.0}),
            ("non-finite eps", {"eps": float("nan")}),
            ("wrong index count", {"indices": indices[:1]}),
            ("wrong index rank", {"indices": indices.reshape(1, 2)}),
            ("empty conditions", {"shift": shift[:0], "scale": scale[:0]}),
        )
        for name, overrides in cases:
            with self.subTest(case=name):
                with self.assertRaises(ParametersInvalid):
                    call(**overrides)

    def test_invalid_index_values_fail_on_device(self):
        cases = (
            ("scalar_h1", 1, 1, FP8, -1),
            ("e4m3_multirow", 64, 32, FP8, 1),
            ("hybrid", 1024, 3000, FP8, -1),
            ("general_tiled_fp4", 1, 4098, FP4, 1),
        )
        context = multiprocessing.get_context("spawn")
        for name, rows, hidden, dst_type, bad_index in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as workdir:
                    process = context.Process(
                        target=_invalid_index_worker,
                        args=(rows, hidden, dst_type, bad_index, workdir),
                    )
                    process.start()
                    process.join(timeout=180)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
                        self.fail(f"{name}: invalid-index worker timed out")
                    result_path = os.path.join(workdir, "invalid_index_result.txt")
                    detail = "worker exited without a device error"
                    if os.path.exists(result_path):
                        with open(result_path, encoding="utf-8") as stream:
                            detail = stream.read()
                    self.assertEqual(process.exitcode, 0, f"{name}: {detail}")

    def test_index_validation_runs_once_per_tensor_version(self):
        indices = torch.tensor([0, 0], dtype=torch.int64, device="npu")
        _fused_module._VALIDATED_INDICES.clear()
        spy = _LaunchCountSpy(_fused_module._validate_indices_kernel)
        with mock.patch.object(_fused_module, "_validate_indices_kernel", spy):
            _fused_module._validate_indices(indices, 1)
            torch.npu.synchronize()
            _fused_module._validate_indices(indices, 1)
            torch.npu.synchronize()
            self.assertEqual(len(spy.grids), 1)

            indices.add_(0)
            torch.npu.synchronize()
            _fused_module._validate_indices(indices, 1)
            torch.npu.synchronize()
            self.assertEqual(len(spy.grids), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
