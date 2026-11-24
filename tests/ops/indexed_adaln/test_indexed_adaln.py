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
"""Tests for the fused Indexed AdaLayerNorm operator."""

import pytest
import torch

from mindiesd import layernorm_scale_shift, layernorm_scale_shift_indexed
from mindiesd.utils import ParametersInvalid


def _npu_available():
    try:
        return torch.npu.is_available() and hasattr(torch.ops.mindiesd, "indexed_adaln")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _npu_available(), reason="NPU or mindiesd::indexed_adaln not available")


def _make_case(batch, seq_len, hidden, table_k, seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    device = "npu"
    x = torch.randn(batch, seq_len, hidden, dtype=dtype, device=device)
    scale_table = torch.randn(batch, table_k, hidden, dtype=dtype, device=device) * 0.05
    shift_table = torch.randn(batch, table_k, hidden, dtype=dtype, device=device) * 0.05
    mod_index = torch.arange(seq_len, dtype=torch.int64, device=device) % table_k
    layernorm = torch.nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
    return x, scale_table, shift_table, mod_index, layernorm


def _baseline(x, scale_table, shift_table, mod_index, layernorm):
    scale = torch.index_select(scale_table, 1, mod_index)
    shift = torch.index_select(shift_table, 1, mod_index)
    return layernorm_scale_shift(layernorm, x, scale, shift, fused=True)


@pytest.mark.parametrize(
    "batch,seq_len,hidden,table_k",
    [
        (1, 4096, 3072, 3),
        (1, 128, 3072, 3),
        (2, 256, 3072, 3),
        (1, 64, 1024, 3),
    ],
)
def test_matches_index_select_adaln(batch, seq_len, hidden, table_k):
    x, scale_table, shift_table, mod_index, layernorm = _make_case(batch, seq_len, hidden, table_k)
    expected = _baseline(x, scale_table, shift_table, mod_index, layernorm)
    actual = layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=True)
    torch.npu.synchronize()
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.bfloat16, 2e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-3, 1e-3),
    ],
)
def test_supported_dtypes(dtype, atol, rtol):
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 4096, 3072, 3, dtype=dtype)
    expected = _baseline(x, scale_table, shift_table, mod_index, layernorm)
    actual = layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=True)
    torch.npu.synchronize()
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


def test_identity_tables_equal_layernorm():
    """scale_table=shift_table=0 must reproduce LayerNorm(x)."""
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 4096, 3072, 3)
    scale_table.zero_()
    shift_table.zero_()
    expected = layernorm(x)
    actual = layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=True)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)


def test_boundary_mod_indices():
    """All table rows must be selectable, including K-1."""
    x, scale_table, shift_table, _, layernorm = _make_case(1, 256, 3072, 3)
    for idx_value in range(3):
        mod_index = torch.full((256,), idx_value, dtype=torch.int64, device="npu")
        expected = _baseline(x, scale_table, shift_table, mod_index, layernorm)
        actual = layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=True)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)


def test_fused_false_matches_eager_reference():
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 256, 3072, 3)
    expected = _baseline(x, scale_table, shift_table, mod_index, layernorm)
    actual = layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=False)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)


def test_rejects_affine_layernorm():
    x, scale_table, shift_table, mod_index, _ = _make_case(1, 64, 1024, 3)
    layernorm = torch.nn.LayerNorm(1024, elementwise_affine=True, eps=1e-6, dtype=torch.bfloat16, device="npu")
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=True)


def test_rejects_int32_mod_index():
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 64, 1024, 3)
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index.to(torch.int32), fused=True)


def test_rejects_wrong_index_length():
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 64, 1024, 3)
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index[:-1], fused=True)


def test_rejects_mismatched_table_dtype():
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 64, 1024, 3)
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table.to(torch.float16), shift_table, mod_index, fused=True)


def test_rejects_unaligned_hidden():
    # H=3080 with 2-byte dtype gives a 6160-byte row, not a multiple of 32B.
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 8, 3080, 3)
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=True)


def test_rejects_wrong_normalized_shape():
    x, scale_table, shift_table, mod_index, _ = _make_case(1, 64, 1024, 3)
    layernorm = torch.nn.LayerNorm((2, 512), elementwise_affine=False, eps=1e-6, dtype=torch.bfloat16, device="npu")
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, mod_index, fused=True)


def test_rejects_out_of_range_mod_index():
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 64, 1024, 3)
    bad_high = torch.zeros(64, dtype=torch.int64, device="npu")
    bad_high[1] = 3
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, bad_high, fused=True)
    bad_negative = torch.zeros(64, dtype=torch.int64, device="npu")
    bad_negative[2] = -1
    with pytest.raises(ParametersInvalid):
        layernorm_scale_shift_indexed(layernorm, x, scale_table, shift_table, bad_negative, fused=True)


def test_validate_mod_index_false_valid_input():
    x, scale_table, shift_table, mod_index, layernorm = _make_case(1, 64, 1024, 3)
    expected = _baseline(x, scale_table, shift_table, mod_index, layernorm)
    actual = layernorm_scale_shift_indexed(
        layernorm, x, scale_table, shift_table, mod_index, fused=True, validate_mod_index=False
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)
