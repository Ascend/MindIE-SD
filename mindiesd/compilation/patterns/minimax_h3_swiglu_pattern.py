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

"""MiniMax-H3 SwiGLU fusion pattern (register_replacement).

MiniMax-H3 FFN uses diffusers SwiGLU: proj (Linear [S,D]->[S,2F]) then
    hidden, gate = proj.chunk(2, dim=-1)   # graph: aten.split.Tensor(proj, F, -1)
    out = hidden * silu(gate)

Real graph (verified by dump):
    split = aten.split.Tensor(matmul_4, 14336, -1)
    getitem_10 = split[0]   # hidden (first half)
    getitem_11 = split[1]   # gate (second half)
    silu = aten.silu.default(getitem_11)
    mul_9 = aten.mul.Tensor(getitem_10, silu)

Replacement: mindiesd triton `swiglu(proj)` -- a single row kernel reading
proj directly (no concat). History: the earlier replacement used
cat([gate, hidden]) + torch_npu.npu_swiglu (same math, swapped halves), but
the [1, S, 2F] concat cost ~190us/site; triton kernel ~276us vs cat+swiglu
~550us in bench, and avoids the extra cat kernel in the graph.
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()

from mindiesd.layers.scale_shift import swiglu as mindie_swiglu  # noqa: E402


def create(dtype, half_dim=14336):
    class MiniMaxH3SwigluPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            proj = torch.empty(1, 4, 2 * half_dim, dtype=dtype, device="meta")
            return [proj]

        @staticmethod
        def pattern(proj):
            def func(proj):
                hidden, gate = torch.ops.aten.split.Tensor(proj, half_dim, -1)
                silu = torch.ops.aten.silu.default(gate)
                return torch.ops.aten.mul.Tensor(hidden, silu)

            return func(proj)

        @staticmethod
        def replacement(proj):
            def func(proj):
                return mindie_swiglu(proj)

            return func(proj)

    return MiniMaxH3SwigluPattern


MiniMaxH3SwigluPatternGroup = [
    create(dtype=torch.bfloat16),
    create(dtype=torch.float32),
]
