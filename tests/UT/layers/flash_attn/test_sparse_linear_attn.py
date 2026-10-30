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
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch

from tests.utils.utils.precision_compare import data_compare

from mindiesd.utils.exception import ParametersInvalid
from mindiesd.utils.get_platform import NPUDevice, get_npu_device

try:
    from mindiesd.layers.flash_attn.sparse_linear_attn import (
        SparseLinearAttention,
        _ASCENDC_A5_BLKK_ALIGN,
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
        is_a5_device,
    )

    _IMPORT_OK = True
    _IMPORT_ERR = None
except ImportError as exc:
    SparseLinearAttention = None
    _ASCENDC_A5_BLKK_ALIGN = 16
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
    is_a5_device = None
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
# These combos are valid on A5 but unsupported on A2/A3.
_NON_A5_UNSUPPORTED_BLK_COMBOS = ((16, 64), (32, 64), (100, 64), (64, 96), (128, 96))
# These combos are unsupported on every device because BLKK is not 16-aligned or Triton-supported.
_ALL_DEVICE_UNSUPPORTED_BLK_COMBOS = ((64, 72), (100, 72), (128, 100))
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
_A5_PATCH = 'mindiesd.layers.flash_attn.sparse_linear_attn.is_a5_device'
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
    return _resolve_sparse_attn_backend(
        mod.proj_l.in_features,
        mod.BLKQ,
        mod.BLKK,
        is_950=getattr(mod, '_is_950', None),
    )


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
    @patch(_A5_PATCH, return_value=True)
    def test_default_block64_prefers_ascendc_on_a5(self, _mock_is_a5_device):
        mod = SparseLinearAttention(head_dim=HEADDIM, topk=TOPK)
        self.assertTrue(mod._is_950)
        self.assertEqual(_backend_for(mod), 'ascendc')

    @patch(_A5_PATCH, return_value=False)
    def test_default_backend_falls_back_to_triton_when_blkk_64(self, _mock_is_a5_device):
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
        self.assertEqual(_ASCENDC_A5_BLKK_ALIGN, 16)
        self.assertEqual(_ASCENDC_BLKK_ALIGN, 128)
        self.assertEqual(_TRITON_SUPPORTED_BLOCK_SIZES, (64, 128))

    def test_resolve_prefers_ascendc_when_both_supported(self):
        self.assertEqual(_resolve_sparse_attn_backend(64, 64, 128), 'ascendc')
        self.assertEqual(_resolve_sparse_attn_backend(128, 128, 128), 'ascendc')

    def test_resolve_prefers_ascendc_for_block64_only_on_a5(self):
        self.assertEqual(_resolve_sparse_attn_backend(64, 64, 64, is_950=True), 'ascendc')
        self.assertEqual(_resolve_sparse_attn_backend(64, 64, 64, is_950=False), 'triton')

    def test_resolve_detects_a5_when_device_flag_is_omitted(self):
        with patch(_A5_PATCH, return_value=True):
            self.assertEqual(_resolve_sparse_attn_backend(64, 64, 64), 'ascendc')

    def test_shape_support_detects_a5_when_device_flag_is_omitted(self):
        with patch(_A5_PATCH, return_value=True):
            self.assertTrue(_ascendc_shape_supported(64, 64, 64))
        with patch(_A5_PATCH, return_value=False):
            self.assertFalse(_ascendc_shape_supported(64, 64, 64))

    def test_resolve_falls_back_to_triton_when_ascendc_unsupported(self):
        with patch(_A5_PATCH, return_value=False):
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

    def test_rejects_non_a5_block_combos_at_init(self):
        with patch(_A5_PATCH, return_value=False):
            for blkq, blkk in _NON_A5_UNSUPPORTED_BLK_COMBOS:
                with self.subTest(blkq=blkq, blkk=blkk):
                    with self.assertRaises(ParametersInvalid):
                        SparseLinearAttention(
                            head_dim=HEADDIM,
                            topk=TOPK,
                            BLKQ=blkq,
                            BLKK=blkk,
                        )

    def test_accepts_same_block_combos_on_a5(self):
        with patch(_A5_PATCH, return_value=True):
            for blkq, blkk in _NON_A5_UNSUPPORTED_BLK_COMBOS:
                with self.subTest(blkq=blkq, blkk=blkk):
                    mod = SparseLinearAttention(
                        head_dim=HEADDIM,
                        topk=TOPK,
                        BLKQ=blkq,
                        BLKK=blkk,
                    )
                    self.assertEqual(_backend_for(mod), 'ascendc')

    def test_rejects_unaligned_block_combos_on_all_devices(self):
        for is_a5 in (False, True):
            with patch(_A5_PATCH, return_value=is_a5):
                for blkq, blkk in _ALL_DEVICE_UNSUPPORTED_BLK_COMBOS:
                    with self.subTest(is_a5=is_a5, blkq=blkq, blkk=blkk):
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


# ---------------------------------------------------------------------------
# 3. TestSparseAttentionForward — CPU：forward 校验（mock get_block_map）
# ---------------------------------------------------------------------------
@_SKIP_NO_TRITON
class TestSparseAttentionForward(unittest.TestCase):
    def test_forward_refreshes_a5_detection_before_backend_resolution(self):
        class ExpectedBsaBackend(Exception):
            pass

        class UnexpectedTritonBackend(Exception):
            pass

        with patch(_A5_PATCH, return_value=False):
            mod = SparseLinearAttention(head_dim=HEADDIM, topk=TOPK)

        def make_fake_npu_tensor():
            tensor = MagicMock()
            tensor.dtype = torch.float16
            tensor.shape = (BATCH, HEAD, SEQ, HEADDIM)
            tensor.device.type = 'npu'
            tensor.contiguous.return_value = tensor
            tensor.to.return_value = tensor
            return tensor

        q = make_fake_npu_tensor()
        k = make_fake_npu_tensor()
        v = make_fake_npu_tensor()
        with (
            patch(_A5_PATCH, return_value=True),
            patch(
                'mindiesd.layers.flash_attn.sparse_linear_attn.get_block_map',
                return_value=(None, None, 1),
            ),
            patch(_BSA_PATCH, side_effect=ExpectedBsaBackend),
            patch(_ATTN_PATCH, side_effect=UnexpectedTritonBackend),
        ):
            with self.assertRaises(ExpectedBsaBackend):
                mod(q, k, v)

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


# ---------------------------------------------------------------------------
# 5. TestSparseLinearAttentionPrecision — NPU 实机 ascendc vs triton 精度对比
#    shape (1,1,1048576,128), BLKQ=BLKK=128, bf16, inner_precise=0 (910B)
# ---------------------------------------------------------------------------
def _run_precision_forward(mod, q, k, v, *, force_backend=None):
    ctx = patch(_RESOLVE_PATCH, return_value=force_backend) if force_backend is not None else nullcontext()
    with ctx:
        return mod(q, k, v)


if __name__ == '__main__':
    unittest.main()
