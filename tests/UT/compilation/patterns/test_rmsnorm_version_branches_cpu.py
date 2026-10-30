#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Historical torch dtype-cast branches execute against explicit CPU schemas."""

import importlib
from unittest.mock import patch

import pytest
import torch


@pytest.mark.parametrize("version", ["2.6.0", "2.8.0"])
def test_rms_pattern_legacy_cast_and_rounding(version):
    module = importlib.import_module("mindiesd.compilation.patterns.rms_norm_pattern")
    x = torch.randn(2, 2, 2, 4, dtype=torch.bfloat16)
    weight = torch.randn(4, dtype=torch.bfloat16)
    with patch.object(module.torch, "__version__", version), patch.object(module, "IS_TORCH_GE_29", False):
        pattern = module.create(torch.bfloat16, epsilon=1e-6)
        actual = pattern.pattern(x, weight)
    eps = torch.tensor(1e-6, dtype=torch.bfloat16 if version == "2.6.0" else torch.float32).item()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    expected = normalized.to(x.dtype) * weight if version == "2.6.0" else (normalized * weight).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
