#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.utils.precision_compare import data_compare

from mindiesd.utils.exception import ParametersInvalid

try:
    from mindiesd.layers.flash_attn.sparse_linear_attn import (
        SparseLinearAttention,
        _ASCENDC_BLKK_ALIGN,
        _ASCENDC_SUPPORTED_HEAD_DIMS,
        _ascendc_shape_supported,
        _SUPPORTED_SPARSE_BACKENDS,
        _TRITON_SUPPORTED_BLOCK_SIZES,
        _TRITON_SUPPORTED_HEAD_DIMS,
        _triton_shape_supported,
        _resolve_sparse_attn_backend,
        _validate_inner_precise,
        get_block_map,
    )

    _IMPORT_OK = True
    _IMPORT_ERR = None
except ImportError as exc:
    SparseLinearAttention = None
    _ASCENDC_BLKK_ALIGN = 128
    _ASCENDC_SUPPORTED_HEAD_DIMS = ()
    _ascendc_shape_supported = None
    _SUPPORTED_SPARSE_BACKENDS = ()
    _TRITON_SUPPORTED_BLOCK_SIZES = (64, 128)
    _TRITON_SUPPORTED_HEAD_DIMS = ()
    _triton_shape_supported = None
    _resolve_sparse_attn_backend = None
    _validate_inner_precise = None
    get_block_map = None
    _IMPORT_OK = False
    _IMPORT_ERR = exc

_SKIP_NO_TRITON = unittest.skipIf(
    not _IMPORT_OK,
    f'sparse_linear_attn import failed (need triton / sparse_linear_attn_triton): {_IMPORT_ERR}',
)
_SKIP_NPU = unittest.skipIf(
    os.environ.get('MINDIE_TEST_MODE', 'ALL') == 'CPU',
    'Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.',
)

BATCH = 1
HEAD = 2
HEADDIM = 64
HEADDIM_128 = 128
SEQ = 256
TOPK = 0.25
BLK = 128
BLK_TRITON = 64
# Neither ascendc nor triton can serve these head_dim values.
_UNSUPPORTED_HEAD_DIMS = (96, 48, 192, 256, 512)
# BLKQ/BLKK combos that neither backend supports (head_dim=64, BLKK not 128-aligned).
_UNSUPPORTED_BLK_COMBOS = ((16, 64), (32, 64), (100, 64), (64, 96), (128, 96))
# All valid triton BLKQ/BLKK combinations (product of supported block sizes).
_TRITON_BLK_COMBOS = (
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)
# ascendc forward smoke: supported BLKQ with BLKK aligned to 128.
_ASCENDC_BLK_COMBOS = (
    (64, 128),
    (128, 128),
    (100, 128),
    (128, 256),
)
_RESOLVE_PATCH = 'mindiesd.layers.flash_attn.sparse_linear_attn._resolve_sparse_attn_backend'
_ATTN_PATCH = 'mindiesd.layers.flash_attn.sparse_linear_attn._attention.apply'
_BSA_PATCH = 'mindiesd.layers.flash_attn.sparse_linear_attn.block_sparse_attention'
# Production-like precision scenario on 910B (see bench_sparse_linear_attn_forward.py).
PRECISION_BATCH = 1
PRECISION_HEAD = 1
PRECISION_SEQ_LEN = 1048576
PRECISION_HEAD_DIM = 128
PRECISION_BLK = 128
PRECISION_INNER_PRECISE = 0
PRECISION_SEED = 42


def _backend_for(mod):
    return _resolve_sparse_attn_backend(mod.proj_l.in_features, mod.BLKQ, mod.BLKK)


def _iter_ascendc_shape_matrix():
    for head_dim in _ASCENDC_SUPPORTED_HEAD_DIMS:
        for blkq, blkk in _ASCENDC_BLK_COMBOS:
            yield head_dim, blkq, blkk


def _iter_triton_shape_matrix():
    for head_dim in _TRITON_SUPPORTED_HEAD_DIMS:
        for blkq in _TRITON_SUPPORTED_BLOCK_SIZES:
            for blkk in _TRITON_SUPPORTED_BLOCK_SIZES:
                yield head_dim, blkq, blkk


def _ascendc_npu_runnable(head_dim, blkq, blkk):
    # get_block_map compress_kernel UB overflows on 910B when head_dim=128, BLKK=256.
    return not (head_dim == 128 and blkk == 256)


def _triton_npu_runnable(head_dim, blkq, blkk):
    # _attention kernel UB overflows on 910B when head_dim=128, BLKQ=64, BLKK=128.
    return not (head_dim == 128 and blkq == 64 and blkk == 128)


def _iter_ascendc_npu_shapes():
    for head_dim, blkq, blkk in _iter_ascendc_shape_matrix():
        if _ascendc_npu_runnable(head_dim, blkq, blkk):
            yield head_dim, blkq, blkk


def _iter_triton_npu_shapes():
    for head_dim, blkq, blkk in _iter_triton_shape_matrix():
        if _triton_npu_runnable(head_dim, blkq, blkk):
            yield head_dim, blkq, blkk


def _mock_block_map_return(seq_len=SEQ, batch=BATCH, head=HEAD, blk=BLK):
    q_blocks = math.ceil(seq_len / blk)
    kv_blocks = math.ceil(seq_len / blk)
    return (
        torch.zeros(batch, head, q_blocks, kv_blocks, dtype=torch.int8),
        torch.zeros(batch, head, q_blocks, 1, dtype=torch.int32),
        1,
    )


# ---------------------------------------------------------------------------
# 1. TestSparseLinearAttentionInit
# ---------------------------------------------------------------------------
@_SKIP_NO_TRITON
class TestSparseLinearAttentionInit(unittest.TestCase):
    def test_default_backend_falls_back_to_triton_when_blkk_64(self):
        mod = SparseLinearAttention(head_dim=HEADDIM, topk=TOPK)
        self.assertEqual(mod.BLKQ, BLK_TRITON)
        self.assertEqual(mod.BLKK, BLK_TRITON)
        self.assertEqual(_backend_for(mod), 'triton')

    def test_auto_selects_ascendc_when_shape_supported(self):
        mod = SparseLinearAttention(head_dim=HEADDIM, topk=TOPK, BLKQ=BLK, BLKK=BLK)
        self.assertEqual(_backend_for(mod), 'ascendc')

    def test_supported_backends(self):
        self.assertEqual(_SUPPORTED_SPARSE_BACKENDS, ('triton', 'ascendc'))

    def test_ascendc_supported_head_dims_constant(self):
        self.assertEqual(_ASCENDC_SUPPORTED_HEAD_DIMS, (64, 128))

    def test_block_size_constants(self):
        self.assertEqual(_ASCENDC_BLKK_ALIGN, 128)
        self.assertEqual(_TRITON_SUPPORTED_BLOCK_SIZES, (64, 128))

    def test_resolve_prefers_ascendc_when_both_supported(self):
        self.assertEqual(_resolve_sparse_attn_backend(64, 64, 128), 'ascendc')
        self.assertEqual(_resolve_sparse_attn_backend(128, 128, 128), 'ascendc')

    def test_resolve_falls_back_to_triton_when_ascendc_unsupported(self):
        for head_dim, blkq, blkk in ((64, 64, 64), (32, 64, 128), (16, 64, 64)):
            with self.subTest(head_dim=head_dim, blkq=blkq, blkk=blkk):
                self.assertEqual(
                    _resolve_sparse_attn_backend(head_dim, blkq, blkk),
                    'triton',
                )
        mod = SparseLinearAttention(head_dim=HEADDIM, topk=TOPK, BLKQ=100, BLKK=BLK)
        self.assertEqual(_backend_for(mod), 'ascendc')

    def test_ascendc_accepts_supported_head_dims_at_init(self):
        for head_dim in (HEADDIM, HEADDIM_128):
            with self.subTest(head_dim=head_dim):
                mod = SparseLinearAttention(
                    head_dim=head_dim,
                    topk=TOPK,
                    BLKQ=BLK,
                    BLKK=BLK,
                )
                self.assertEqual(_backend_for(mod), 'ascendc')
                self.assertEqual(mod.proj_l.in_features, head_dim)

    def test_ascendc_accepts_supported_block_sizes_at_init(self):
        for blkq, blkk in ((BLK, BLK), (64, BLK), (128, 256)):
            with self.subTest(blkq=blkq, blkk=blkk):
                mod = SparseLinearAttention(
                    head_dim=HEADDIM,
                    topk=TOPK,
                    BLKQ=blkq,
                    BLKK=blkk,
                )
                self.assertEqual(_backend_for(mod), 'ascendc')
                self.assertEqual(mod.BLKQ, blkq)
                self.assertEqual(mod.BLKK, blkk)

    def test_triton_fallback_for_small_head_dim(self):
        for head_dim in (16, 32):
            with self.subTest(head_dim=head_dim):
                mod = SparseLinearAttention(
                    head_dim=head_dim,
                    topk=TOPK,
                    BLKQ=BLK_TRITON,
                    BLKK=BLK_TRITON,
                )
                self.assertEqual(_backend_for(mod), 'triton')

    def test_triton_fallback_when_blkk_not_aligned(self):
        mod = SparseLinearAttention(
            head_dim=HEADDIM,
            topk=TOPK,
            BLKQ=BLK_TRITON,
            BLKK=BLK_TRITON,
        )
        self.assertEqual(_backend_for(mod), 'triton')

    def test_rejects_unsupported_head_dim_at_init(self):
        for head_dim in _UNSUPPORTED_HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                with self.assertRaises(ParametersInvalid):
                    SparseLinearAttention(
                        head_dim=head_dim,
                        topk=TOPK,
                        BLKQ=BLK_TRITON,
                        BLKK=BLK_TRITON,
                    )

    def test_rejects_unsupported_block_combo_at_init(self):
        for blkq, blkk in _UNSUPPORTED_BLK_COMBOS:
            with self.subTest(blkq=blkq, blkk=blkk):
                with self.assertRaises(ParametersInvalid):
                    SparseLinearAttention(
                        head_dim=HEADDIM,
                        topk=TOPK,
                        BLKQ=blkq,
                        BLKK=blkk,
                    )

    def test_triton_supported_head_dims_constant(self):
        self.assertEqual(_TRITON_SUPPORTED_HEAD_DIMS, (16, 32, 64, 128))

    def test_triton_accepts_supported_head_dims_via_fallback(self):
        for head_dim in _TRITON_SUPPORTED_HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                blkk = BLK if head_dim in _ASCENDC_SUPPORTED_HEAD_DIMS else BLK_TRITON
                blkq = BLK_TRITON if blkk == BLK_TRITON else BLK
                mod = SparseLinearAttention(
                    head_dim=head_dim,
                    topk=TOPK,
                    BLKQ=blkq,
                    BLKK=blkk,
                )
                expected = 'ascendc' if head_dim in _ASCENDC_SUPPORTED_HEAD_DIMS and blkk == BLK else 'triton'
                self.assertEqual(_backend_for(mod), expected)
                self.assertEqual(mod.proj_l.in_features, head_dim)


# ---------------------------------------------------------------------------
# 1b. TestInnerPreciseValidation — CPU：inner_precise 与 dtype/芯片组合
# ---------------------------------------------------------------------------
@_SKIP_NO_TRITON
class TestInnerPreciseValidation(unittest.TestCase):
    def test_bf16_non_950_only_accepts_inner_precise_0(self):
        _validate_inner_precise(0, is_950=False, use_bf16=True)
        for invalid in (1, 4):
            with self.subTest(inner_precise=invalid):
                with self.assertRaises(ParametersInvalid):
                    _validate_inner_precise(invalid, is_950=False, use_bf16=True)

    def test_fp16_non_950_accepts_inner_precise_0_and_1(self):
        for val in (0, 1):
            with self.subTest(inner_precise=val):
                _validate_inner_precise(val, is_950=False, use_bf16=False)
        with self.assertRaises(ParametersInvalid):
            _validate_inner_precise(4, is_950=False, use_bf16=False)

    def test_950_only_accepts_inner_precise_4(self):
        _validate_inner_precise(4, is_950=True, use_bf16=True)
        _validate_inner_precise(4, is_950=True, use_bf16=False)
        for invalid in (0, 1):
            with self.subTest(inner_precise=invalid):
                with self.assertRaises(ParametersInvalid):
                    _validate_inner_precise(invalid, is_950=True, use_bf16=True)


# ---------------------------------------------------------------------------
# 2. TestGetBlockMap — triton kernel，NPU 实机（@_SKIP_NPU）
# ---------------------------------------------------------------------------
@_SKIP_NPU
@_SKIP_NO_TRITON
class TestGetBlockMap(unittest.TestCase):
    def test_sparse_map_dtype_and_shape(self):
        device = torch.device('npu:0')
        q = torch.randn(BATCH, HEAD, SEQ, HEADDIM, device=device)
        k = torch.randn(BATCH, HEAD, SEQ, HEADDIM, device=device)
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=0.5, BLKQ=BLK, BLKK=BLK)
        q_blocks = math.ceil(SEQ / BLK)
        kv_blocks = math.ceil(SEQ / BLK)
        self.assertEqual(sparse_map.dtype, torch.int8)
        self.assertEqual(tuple(sparse_map.shape), (BATCH, HEAD, q_blocks, kv_blocks))
        self.assertEqual(lut.shape[-1], real_topk)
        self.assertGreater(real_topk, 0)

    def test_sparse_map_dtype_and_shape_head_dim_128(self):
        device = torch.device('npu:0')
        q = torch.randn(BATCH, HEAD, SEQ, HEADDIM_128, dtype=torch.float16, device=device)
        k = torch.randn(BATCH, HEAD, SEQ, HEADDIM_128, dtype=torch.float16, device=device)
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=0.5, BLKQ=BLK, BLKK=BLK)
        q_blocks = math.ceil(SEQ / BLK)
        kv_blocks = math.ceil(SEQ / BLK)
        self.assertEqual(sparse_map.dtype, torch.int8)
        self.assertEqual(tuple(sparse_map.shape), (BATCH, HEAD, q_blocks, kv_blocks))
        self.assertEqual(lut.shape[-1], real_topk)
        self.assertGreater(real_topk, 0)

    def test_sparse_map_dtype_and_shape_blk_64(self):
        device = torch.device('npu:0')
        q = torch.randn(BATCH, HEAD, SEQ, HEADDIM, dtype=torch.float16, device=device)
        k = torch.randn(BATCH, HEAD, SEQ, HEADDIM, dtype=torch.float16, device=device)
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=0.5, BLKQ=BLK_TRITON, BLKK=BLK_TRITON)
        q_blocks = math.ceil(SEQ / BLK_TRITON)
        kv_blocks = math.ceil(SEQ / BLK_TRITON)
        self.assertEqual(sparse_map.dtype, torch.int8)
        self.assertEqual(tuple(sparse_map.shape), (BATCH, HEAD, q_blocks, kv_blocks))
        self.assertEqual(lut.shape[-1], real_topk)
        self.assertGreater(real_topk, 0)


# ---------------------------------------------------------------------------
# 3. TestSparseAttentionForward — CPU：forward 校验（mock get_block_map）
# ---------------------------------------------------------------------------
@_SKIP_NO_TRITON
class TestSparseAttentionForward(unittest.TestCase):
    @patch('mindiesd.layers.flash_attn.sparse_linear_attn.get_block_map')
    def test_rejects_non_npu_device(self, mock_get_block_map):
        mock_get_block_map.return_value = _mock_block_map_return()
        q = torch.randn(BATCH, HEAD, SEQ, HEADDIM)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mod = SparseLinearAttention(
            head_dim=HEADDIM,
            topk=TOPK,
            BLKQ=BLK,
            BLKK=BLK,
        )
        with self.assertRaises(ParametersInvalid):
            mod(q, k, v)
        mock_get_block_map.assert_not_called()

    @patch('mindiesd.layers.flash_attn.sparse_linear_attn.get_block_map')
    def test_rejects_head_dim_mismatch_in_forward(self, mock_get_block_map):
        mock_get_block_map.return_value = _mock_block_map_return()
        q = torch.randn(BATCH, HEAD, SEQ, 32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mod = SparseLinearAttention(
            head_dim=HEADDIM,
            topk=TOPK,
            BLKQ=BLK,
            BLKK=BLK,
        )
        with self.assertRaises(ParametersInvalid):
            mod(q, k, v)
        mock_get_block_map.assert_not_called()


# ---------------------------------------------------------------------------
# 4. TestSparseLinearAttentionNPU — NPU 实机（ascendc 自然路径 + triton 自然/mock 路径）
#    构建: build/build_plugin.sh + pip install -e .；勿保留旧 mindiesd/ops vendor
# ---------------------------------------------------------------------------
@_SKIP_NPU
@_SKIP_NO_TRITON
class TestSparseLinearAttentionNPU(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('npu:0')
        torch.npu.set_device(self.device)
        dev_name = torch.npu.get_device_properties(self.device).name
        self.is_950 = '950' in dev_name
        self.inner_precise = 4 if self.is_950 else 0
        self.seq_len = 1024

    def _run_npu_forward(
        self,
        head_dim,
        blkq,
        blkk,
        *,
        use_bf16=False,
        inner_precise=None,
        input_dtype=None,
        force_backend=None,
    ):
        if input_dtype is None:
            input_dtype = torch.bfloat16 if use_bf16 else torch.float16

        q = torch.randn(
            BATCH,
            HEAD,
            self.seq_len,
            head_dim,
            dtype=input_dtype,
            device=self.device,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        mod = SparseLinearAttention(
            head_dim=head_dim,
            topk=0.5,
            BLKQ=blkq,
            BLKK=blkk,
            use_bf16=use_bf16,
            inner_precise=inner_precise,
        ).to(self.device)

        expected_backend = force_backend or _resolve_sparse_attn_backend(head_dim, blkq, blkk)
        resolve_ctx = patch(_RESOLVE_PATCH, return_value=force_backend) if force_backend is not None else nullcontext()

        with resolve_ctx:
            if expected_backend == 'triton':
                with patch(_BSA_PATCH) as mock_bsa:
                    out = mod(q, k, v)
                    mock_bsa.assert_not_called()
            else:
                with patch(_ATTN_PATCH) as mock_attn:
                    out = mod(q, k, v)
                    mock_attn.assert_not_called()

        self.assertEqual(tuple(out.shape), (BATCH, HEAD, self.seq_len, head_dim))
        self.assertEqual(out.dtype, input_dtype)
        return out

    def test_ascendc_npu_all_supported_shapes_fp16(self):
        """实机：ascendc 自然路径覆盖其支持的全部 shape。"""
        for head_dim, blkq, blkk in _iter_ascendc_npu_shapes():
            with self.subTest(head_dim=head_dim, blkq=blkq, blkk=blkk):
                self.assertTrue(_ascendc_shape_supported(head_dim, blkq, blkk))
                self.assertEqual(_resolve_sparse_attn_backend(head_dim, blkq, blkk), 'ascendc')
                self._run_npu_forward(
                    head_dim,
                    blkq,
                    blkk,
                    inner_precise=self.inner_precise,
                )

    def test_ascendc_npu_all_supported_shapes_bf16(self):
        """实机：ascendc bf16 覆盖其支持的全部 shape。"""
        for head_dim, blkq, blkk in _iter_ascendc_npu_shapes():
            with self.subTest(head_dim=head_dim, blkq=blkq, blkk=blkk):
                self._run_npu_forward(
                    head_dim,
                    blkq,
                    blkk,
                    use_bf16=True,
                    inner_precise=None,
                )

    def test_triton_npu_natural_fallback_shapes_fp16(self):
        """实机：triton 自然 fallback 覆盖 ascendc 不支持的 shape。"""
        for head_dim, blkq, blkk in _iter_triton_npu_shapes():
            if _ascendc_shape_supported(head_dim, blkq, blkk):
                continue
            with self.subTest(head_dim=head_dim, blkq=blkq, blkk=blkk):
                self.assertTrue(_triton_shape_supported(head_dim, blkq, blkk))
                self.assertEqual(_resolve_sparse_attn_backend(head_dim, blkq, blkk), 'triton')
                self._run_npu_forward(head_dim, blkq, blkk)

    def test_triton_npu_natural_fallback_shapes_bf16(self):
        """实机：triton bf16 自然 fallback 覆盖 ascendc 不支持的 shape。"""
        for head_dim, blkq, blkk in _iter_triton_npu_shapes():
            if _ascendc_shape_supported(head_dim, blkq, blkk):
                continue
            with self.subTest(head_dim=head_dim, blkq=blkq, blkk=blkk):
                self._run_npu_forward(head_dim, blkq, blkk, use_bf16=True)

    def test_triton_npu_mock_all_supported_overlap_shapes_fp16(self):
        """实机：mock resolve 强制走 triton，覆盖 ascendc 也支持但默认不会选中的 shape。"""
        for head_dim, blkq, blkk in _iter_triton_npu_shapes():
            if not _ascendc_shape_supported(head_dim, blkq, blkk):
                continue
            with self.subTest(head_dim=head_dim, blkq=blkq, blkk=blkk):
                self.assertTrue(_triton_shape_supported(head_dim, blkq, blkk))
                self._run_npu_forward(
                    head_dim,
                    blkq,
                    blkk,
                    force_backend='triton',
                )

    def test_triton_npu_mock_all_supported_overlap_shapes_bf16(self):
        """实机：mock resolve 强制走 triton（bf16），覆盖 ascendc 也支持的 shape。"""
        for head_dim, blkq, blkk in _iter_triton_npu_shapes():
            if not _ascendc_shape_supported(head_dim, blkq, blkk):
                continue
            with self.subTest(head_dim=head_dim, blkq=blkq, blkk=blkk):
                self._run_npu_forward(
                    head_dim,
                    blkq,
                    blkk,
                    use_bf16=True,
                    force_backend='triton',
                )

    def _expect_triton_compile_error(self):
        try:
            from triton.compiler.errors import MLIRCompilationError

            return (MLIRCompilationError,)
        except ImportError:
            return (Exception,)

    def test_triton_head_dim_256_get_block_map_ub_overflow(self):
        """head_dim=256：init 已拒绝；get_block_map compress_kernel 在 910B 仍 UB 溢出。"""
        q = torch.randn(
            BATCH,
            HEAD,
            self.seq_len,
            256,
            dtype=torch.float16,
            device=self.device,
        )
        k = torch.randn_like(q)
        with self.assertRaises(self._expect_triton_compile_error()):
            get_block_map(q, k, topk_ratio=0.5, BLKQ=BLK, BLKK=BLK)

    def test_ascendc_hd128_blkk256_get_block_map_ub_overflow(self):
        """ascendc 声明 BLKK=256，但 head_dim=128 时 get_block_map 在 910B UB 溢出。"""
        q = torch.randn(
            BATCH,
            HEAD,
            self.seq_len,
            HEADDIM_128,
            dtype=torch.float16,
            device=self.device,
        )
        k = torch.randn_like(q)
        with self.assertRaises(self._expect_triton_compile_error()):
            get_block_map(q, k, topk_ratio=0.5, BLKQ=BLK, BLKK=256)

    def test_triton_hd128_blk64_128_attention_ub_overflow(self):
        """triton 声明支持 (128,64,128)，但 _attention 在 910B UB 溢出；mock 强制走 triton。"""
        q = torch.randn(
            BATCH,
            HEAD,
            self.seq_len,
            HEADDIM_128,
            dtype=torch.float16,
            device=self.device,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mod = SparseLinearAttention(
            head_dim=HEADDIM_128,
            topk=0.5,
            BLKQ=64,
            BLKK=128,
        ).to(self.device)
        with patch(_RESOLVE_PATCH, return_value='triton'):
            with patch(_BSA_PATCH) as mock_bsa:
                with self.assertRaises(self._expect_triton_compile_error()):
                    mod(q, k, v)
                mock_bsa.assert_not_called()

    def test_rejects_wrong_inner_precise_for_device(self):
        if self.is_950:
            invalid_cases = ((0, True), (0, False), (1, True), (1, False))
        else:
            invalid_cases = ((4, True), (4, False), (1, True))
        for val, use_bf16 in invalid_cases:
            with self.subTest(inner_precise=val, use_bf16=use_bf16, is_950=self.is_950):
                input_dtype = torch.bfloat16 if use_bf16 else torch.float16
                q = torch.randn(
                    BATCH,
                    HEAD,
                    self.seq_len,
                    HEADDIM,
                    dtype=input_dtype,
                    device=self.device,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                mod = SparseLinearAttention(
                    head_dim=HEADDIM,
                    topk=TOPK,
                    BLKQ=BLK,
                    BLKK=BLK,
                    inner_precise=val,
                    use_bf16=use_bf16,
                ).to(self.device)
                self.assertEqual(_backend_for(mod), 'ascendc')
                with self.assertRaises(ParametersInvalid):
                    mod(q, k, v)

    def test_accepts_correct_inner_precise_for_device(self):
        q = torch.randn(BATCH, HEAD, self.seq_len, HEADDIM, dtype=torch.float16, device=self.device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mod = SparseLinearAttention(
            head_dim=HEADDIM,
            topk=TOPK,
            BLKQ=BLK,
            BLKK=BLK,
            use_bf16=False,
            inner_precise=self.inner_precise,
        ).to(self.device)
        self.assertEqual(_backend_for(mod), 'ascendc')
        out = mod(q, k, v)
        self.assertEqual(tuple(out.shape), (BATCH, HEAD, self.seq_len, HEADDIM))


# ---------------------------------------------------------------------------
# 5. TestSparseLinearAttentionPrecision — NPU 实机 ascendc vs triton 精度对比
#    shape (1,1,1048576,128), BLKQ=BLKK=128, bf16, inner_precise=0 (910B)
# ---------------------------------------------------------------------------
def _run_precision_forward(mod, q, k, v, *, force_backend=None):
    ctx = patch(_RESOLVE_PATCH, return_value=force_backend) if force_backend is not None else nullcontext()
    with ctx:
        return mod(q, k, v)


@_SKIP_NPU
@_SKIP_NO_TRITON
class TestSparseLinearAttentionPrecision(unittest.TestCase):
    """Compare ascendc (default) and triton (mock-forced) forward outputs on NPU."""

    def setUp(self):
        self.device = torch.device('npu:0')
        torch.npu.set_device(self.device)
        dev_name = torch.npu.get_device_properties(self.device).name
        if '950' in dev_name:
            self.skipTest('inner_precise=0 with bf16 is unsupported on 950 series devices.')

    def test_ascendc_vs_triton_bf16_large_seq(self):
        """ascendc vs triton on (1,1,1048576,128), BLKQ=BLKK=128, bf16, inner_precise=0."""
        torch.manual_seed(PRECISION_SEED)
        q = torch.randn(
            PRECISION_BATCH,
            PRECISION_HEAD,
            PRECISION_SEQ_LEN,
            PRECISION_HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        mod = SparseLinearAttention(
            head_dim=PRECISION_HEAD_DIM,
            topk=TOPK,
            BLKQ=PRECISION_BLK,
            BLKK=PRECISION_BLK,
            use_bf16=True,
            inner_precise=PRECISION_INNER_PRECISE,
        ).to(self.device)
        self.assertEqual(
            _resolve_sparse_attn_backend(PRECISION_HEAD_DIM, PRECISION_BLK, PRECISION_BLK),
            'ascendc',
        )

        torch.npu.synchronize()
        out_ascendc = _run_precision_forward(mod, q, k, v)
        torch.npu.synchronize()

        torch.npu.synchronize()
        out_triton = _run_precision_forward(mod, q, k, v, force_backend='triton')
        torch.npu.synchronize()

        self.assertEqual(
            tuple(out_ascendc.shape),
            (PRECISION_BATCH, PRECISION_HEAD, PRECISION_SEQ_LEN, PRECISION_HEAD_DIM),
        )
        self.assertEqual(tuple(out_triton.shape), tuple(out_ascendc.shape))
        self.assertEqual(out_ascendc.dtype, torch.bfloat16)
        self.assertEqual(out_triton.dtype, torch.bfloat16)

        result, fulfill_pct, max_err = data_compare(
            out_triton.detach().cpu(),
            out_ascendc.detach().cpu(),
        )
        self.assertEqual(
            result,
            'success',
            msg=(f'ascendc vs triton mismatch: fulfill={fulfill_pct:.4f}%, max_relative_err={max_err}'),
        )


if __name__ == '__main__':
    unittest.main()
