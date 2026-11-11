#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from types import SimpleNamespace

import pytest
import torch

from mindiesd.compilation.patterns.wan_adalayernorm_pattern import _has_matching_modulation_dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_same_dtype_is_supported(dtype):
    values = {
        name: SimpleNamespace(meta={"val": torch.empty(1, 2, 8, dtype=dtype)}) for name in ("x", "scale", "shift")
    }
    assert _has_matching_modulation_dtype(SimpleNamespace(kwargs=values))


def test_mixed_dtype_keeps_original_adaln_path():
    values = {
        name: SimpleNamespace(meta={"val": torch.empty(1, 2, 8, dtype=dtype)})
        for name, dtype in (("x", torch.float32), ("scale", torch.bfloat16), ("shift", torch.bfloat16))
    }
    assert not _has_matching_modulation_dtype(SimpleNamespace(kwargs=values))


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("has_weight,has_bias", [(False, False), (True, False), (False, True), (True, True)])
def test_add_layer_norm_retrace_accepts_optional_affine(autocast, has_weight, has_bias):
    from mindiesd.compilation.patterns.add_layer_norm_pattern import create

    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones(8, dtype=x.dtype) if has_weight else None
    bias = torch.zeros(8, dtype=x.dtype) if has_bias else None
    pattern = create(x.dtype, autocast=autocast)
    actual, summed = pattern.pattern(x, residual, weight, bias)
    expected = torch.nn.functional.layer_norm(
        (x + residual).float() if autocast else x + residual,
        (8,),
        weight.float() if autocast and weight is not None else weight,
        bias.float() if autocast and bias is not None else bias,
        1e-6,
    ).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(summed, x + residual, rtol=0, atol=0)
