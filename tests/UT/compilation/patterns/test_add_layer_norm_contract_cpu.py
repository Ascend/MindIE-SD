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

"""CPU metadata and dispatch contracts; device numerics live in tests/ops."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mindiesd.compilation.patterns.add_layer_norm_pattern import _is_add_layer_norm_match, create


class NpuMetadataTensor(torch.Tensor):
    """CPU storage with NPU device metadata solely for guard evaluation."""

    @property
    def device(self):
        return SimpleNamespace(type="npu", index=0)


def make_match(values):
    return SimpleNamespace(kwargs={name: SimpleNamespace(meta={"val": value}) for name, value in values.items()})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_valid_metadata_matches(dtype):
    values = {
        name: torch.empty(shape, dtype=dtype).as_subclass(NpuMetadataTensor)
        for name, shape in (("x", (1, 2048, 8)), ("residual", (1, 2048, 8)), ("weight", (8,)), ("bias", (8,)))
    }
    assert _is_add_layer_norm_match(make_match(values))


@pytest.mark.parametrize("invalid", ["short", "dtype", "shape", "affine", "strided", "cpu", "none"])
def test_incompatible_metadata_is_rejected(invalid):
    values = {
        name: torch.empty(shape).as_subclass(NpuMetadataTensor)
        for name, shape in (("x", (1, 2048, 8)), ("residual", (1, 2048, 8)), ("weight", (8,)), ("bias", (8,)))
    }
    if invalid == "short":
        values["x"] = values["x"][:, :16]
        values["residual"] = values["residual"][:, :16]
    elif invalid == "dtype":
        values["bias"] = values["bias"].bfloat16()
    elif invalid == "shape":
        values["residual"] = values["residual"][:, :16]
    elif invalid == "affine":
        values["weight"] = values["weight"][:4]
    elif invalid == "strided":
        values["x"] = torch.empty(1, 2048, 16).as_subclass(NpuMetadataTensor)[..., ::2]
    elif invalid == "cpu":
        values["x"] = torch.empty(1, 2048, 8)
    else:
        values["weight"] = None
    assert not _is_add_layer_norm_match(make_match(values))


@pytest.mark.parametrize("match", [SimpleNamespace(), SimpleNamespace(kwargs={})])
def test_missing_match_metadata_is_rejected(match):
    assert not _is_add_layer_norm_match(match)


def test_replacement_dispatches_and_selects_normalized_and_residual(monkeypatch):
    import torch_npu

    inputs = (torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.ones(8), torch.zeros(8))
    outputs = tuple(torch.tensor(float(index)) for index in range(4))
    fused = Mock(return_value=outputs)
    monkeypatch.setattr(torch_npu, "npu_add_layer_norm", fused, raising=False)
    normalized, residual = create(torch.float32).replacement(*inputs)
    fused.assert_called_once_with(*inputs, 1e-6, True)
    assert normalized is outputs[0]
    assert residual is outputs[3]
