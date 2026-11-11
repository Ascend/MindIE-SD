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

"""NPU parity tests for QuantBatchMatmulV3 + SwiGLU + DynamicMxQuant.

The cases are skipped when no Ascend device or no CANN CATLASS DSL is
available.  CPU contract checks for the golden and the comparator live in
``tests/UT/ops_reference``.
"""

import pytest
import torch

from tests.ops.qbmm_swiglu_mxquant.code_compare import assert_codes_close, code_diff
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

SPECIAL_CASES = [
    dict(in_fp4=0, out_fp4=0, kind="nan"),
    dict(in_fp4=0, out_fp4=1, kind="nan"),
    dict(in_fp4=0, out_fp4=0, kind="inf"),
    dict(in_fp4=0, out_fp4=1, kind="inf"),
    dict(in_fp4=1, out_fp4=0, kind="inf"),
    dict(in_fp4=1, out_fp4=1, kind="inf"),
]


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


def _npu_ready():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available() and torch.npu.device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _npu_ready(), reason="no Ascend device")
@pytest.mark.parametrize("case", CASES, ids=[str(c) for c in CASES])
def test_npu_matches_reference(case):
    pytest.importorskip("catlass.tla")
    from mindiesd.layers.qbmm_swiglu_mxquant import run

    torch.npu.set_device(0)
    a, sa, b, sb = _make_inputs(case["m"], case["k"], case["n"], case["in_fp4"])
    ref_y, ref_sy = reference_run(a, sa, b, sb, case["in_fp4"], case["out_fp4"], matmul_bf16=False)
    cand_y, cand_sy = run(a.npu(), sa.npu(), b.npu(), sb.npu(), case["in_fp4"], case["out_fp4"])
    torch.npu.synchronize()
    cand_y, cand_sy = cand_y.cpu(), cand_sy.cpu()

    assert_codes_close("y", code_diff(cand_y, ref_y, case["out_fp4"]))
    assert_codes_close("scale", code_diff(cand_sy, ref_sy, False))


@pytest.mark.skipif(not _npu_ready(), reason="no Ascend device")
@pytest.mark.parametrize("case", SPECIAL_CASES, ids=[str(c) for c in SPECIAL_CASES])
def test_npu_matches_reference_special(case):
    """Inf/NaN accumulator blocks must produce scale byte 255 and the special
    half-scale, matching the official DynamicMxQuant golden.
    """
    pytest.importorskip("catlass.tla")
    from mindiesd.layers.qbmm_swiglu_mxquant import run

    torch.npu.set_device(0)
    m, k, n = 1, 64, 128
    a, sa, b, sb = _make_special_inputs(m, k, n, case["in_fp4"], case["kind"])
    ref_y, ref_sy = reference_run(a, sa, b, sb, case["in_fp4"], case["out_fp4"], matmul_bf16=False)
    cand_y, cand_sy = run(a.npu(), sa.npu(), b.npu(), sb.npu(), case["in_fp4"], case["out_fp4"])
    torch.npu.synchronize()
    cand_y, cand_sy = cand_y.cpu(), cand_sy.cpu()

    assert torch.all(ref_sy == 255)
    assert_codes_close("y", code_diff(cand_y, ref_y, case["out_fp4"]))
    assert_codes_close("scale", code_diff(cand_sy, ref_sy, False))
