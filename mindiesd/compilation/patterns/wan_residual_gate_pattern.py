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

"""Wan2.2 residual + gate fusion: `x + y * gate` -> mindiesd::residual_gate_add.

安全使用泛化 pattern 的前提是注册顺序在 adaLN/rope 之后:adaLN 先融合
modulation 站点(3D 相似结构),rope 先融合 4D add(mul,mul) 链,避免误匹配。
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

npu_available = (
    hasattr(torch, "npu")
    and hasattr(torch.npu, "is_available")
    and torch.npu.is_available()
)
if npu_available:
    import torch_npu  # noqa: F401

    import mindiesd


def _is_wan_residual_gate_match(match) -> bool:
    try:
        residual = match.kwargs["x"].meta["val"]
        branch = match.kwargs["y"].meta["val"]
        gate = match.kwargs["gate"].meta["val"]
    except (AttributeError, KeyError):
        return False
    if not all(isinstance(value, torch.Tensor) for value in (residual, branch, gate)):
        return False
    if residual.dim() != 3 or branch.dim() != 3 or gate.dim() != 3:
        return False
    batch, _, dim = residual.shape
    return (
        residual.shape == branch.shape
        and gate.shape == (batch, 1, dim)
        and residual.dtype == branch.dtype == gate.dtype
    )


def create(dtype):
    class WanResidualGatePattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 16, 5120, dtype=dtype, device="meta")
            y = torch.empty(1, 16, 5120, dtype=dtype, device="meta")
            gate = torch.empty(1, 1, 5120, dtype=dtype, device="meta")
            return [x, y, gate]

        @staticmethod
        def pattern(x, y, gate):
            def func(x, y, gate):
                return x + y * gate

            return func(x, y, gate)

        @staticmethod
        def replacement(x, y, gate):
            def func(x, y, gate):
                return mindiesd.layers.residual_gate_add(x, y, gate)

            return func(x, y, gate)

        extra_check = staticmethod(_is_wan_residual_gate_match)

    return WanResidualGatePattern


WanResidualGatePatternGroup = [create(torch.bfloat16), create(torch.float32)]
