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

"""CPU contracts for the qbmm_swiglu_mxquant golden and code comparator."""

import pytest
import torch

from tests.ops.qbmm_swiglu_mxquant.code_compare import assert_codes_close
from tests.ops.qbmm_swiglu_mxquant.reference import _dynamic_mx_quant
from tests.ops.qbmm_swiglu_mxquant.reference import run as reference_run


def _make_inputs(m, k, n, in_fp4, seed=7):
    gen = torch.Generator().manual_seed(seed)
    if in_fp4:
        a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, generator=gen)
        b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    else:
        a = torch.randint(0, 127, (m, k), dtype=torch.uint8, generator=gen)
        b = torch.randint(0, 127, (k, n), dtype=torch.uint8, generator=gen)
    scale_a = torch.randint(120, 136, (m, k // 64, 2), dtype=torch.uint8, generator=gen)
    scale_b = torch.randint(120, 136, (k // 64, n, 2), dtype=torch.uint8, generator=gen)
    return a, scale_a, b, scale_b


CASES = [
    dict(m=9, k=256, n=384, in_fp4=0, out_fp4=0),
    dict(m=33, k=320, n=640, in_fp4=0, out_fp4=1),
    dict(m=16, k=512, n=1024, in_fp4=1, out_fp4=1),
    dict(m=128, k=768, n=512, in_fp4=1, out_fp4=0),
]


@pytest.mark.parametrize("case", CASES, ids=[str(c) for c in CASES])
def test_reference_contract(case):
    a, sa, b, sb = _make_inputs(case["m"], case["k"], case["n"], case["in_fp4"])
    y, sy = reference_run(a, sa, b, sb, case["in_fp4"], case["out_fp4"], matmul_bf16=False)
    n_out = case["n"] // 2
    if case["out_fp4"]:
        assert y.shape == (case["m"], n_out // 2)
    else:
        assert y.shape == (case["m"], n_out)
    assert y.dtype == torch.uint8
    # One scale group per 64 output columns; N=128 -> n_out=64 -> one group.
    assert sy.shape == (case["m"], n_out // 64, 2)
    # Deterministic golden.
    y2, sy2 = reference_run(a, sa, b, sb, case["in_fp4"], case["out_fp4"], matmul_bf16=False)
    assert torch.equal(y, y2) and torch.equal(sy, sy2)


def test_comparator_accepts_bounded_differences():
    diff = torch.zeros(10000, dtype=torch.int16)
    assert_codes_close("y", diff)
    # A few one-code differences are still within the declared tolerance.
    diff[:2] = 1
    assert_codes_close("y", diff)


def test_comparator_rejects_uniform_off_by_one():
    # Every code is off by one: max diff passes but exact-match ratio is zero.
    diff = torch.ones(64, dtype=torch.int16)
    with pytest.raises(AssertionError, match="exact match ratio"):
        assert_codes_close("y", diff)


def test_comparator_rejects_single_large_outlier():
    # 10559/10560 exact still exceeds the ratio threshold, so max diff is
    # what must catch the bad code.
    diff = torch.zeros(10560, dtype=torch.int16)
    diff[0] = 2
    with pytest.raises(AssertionError, match="max code diff"):
        assert_codes_close("y", diff)


def _make_special_inputs(m, k, n, in_fp4, kind):
    groups = k // 64
    scale = 127 if kind == "nan" else 200
    if in_fp4:
        a = torch.full((m, k // 2), 0x77, dtype=torch.uint8)
        b = torch.full((n, k // 2), 0x77, dtype=torch.uint8)
    else:
        a = torch.full((m, k), 0x7F if kind == "nan" else 0x7E, dtype=torch.uint8)
        b = torch.full((k, n), 0x00 if kind == "nan" else 0x7E, dtype=torch.uint8)
    scale_a = torch.full((m, groups, 2), scale, dtype=torch.uint8)
    scale_b = torch.full((groups, n, 2), scale, dtype=torch.uint8)
    return a, scale_a, b, scale_b


@pytest.mark.parametrize(
    "in_fp4,out_fp4,kind",
    [
        (0, 0, "nan"),
        (0, 1, "nan"),
        (0, 0, "inf"),
        (0, 1, "inf"),
        (1, 0, "inf"),
        (1, 1, "inf"),
    ],
)
def test_reference_special_blocks(in_fp4, out_fp4, kind):
    """Inf/NaN accumulator blocks must use scale byte 255 and the special
    half-scale, matching the official DynamicMxQuant golden.
    """
    m, k, n = 1, 64, 128
    a, sa, b, sb = _make_special_inputs(m, k, n, in_fp4, kind)
    y, sy = reference_run(a, sa, b, sb, in_fp4, out_fp4, matmul_bf16=False)
    assert torch.all(sy == 255)
    assert torch.all(y == (0x7E if out_fp4 == 0 else 0x00))


def test_dynamic_mx_quant_special_and_zero_blocks():
    # First 32-element block: Inf + NaN; second block: all zeros.
    bits = torch.tensor([[0x7F80, 0x7FC0] + [0] * 30 + [0] * 32], dtype=torch.int64)
    sy, y = _dynamic_mx_quant(bits, 8)
    assert sy.tolist() == [[[255, 0]]]
    assert torch.all(y[:, :32] == 0x7E)
    assert torch.all(y[:, 32:] == 0)
    sy4, y4 = _dynamic_mx_quant(bits, 2)
    assert sy4.tolist() == [[[255, 0]]]
    assert torch.all(y4 == 0)
