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

"""MiniMax-H3 AdaLN modulation fusion pattern (register_replacement).

MiniMax-H3 block modulation: out = x * (1 + scale) + shift, where scale/shift
are rows of the per-(timestep, modality) modulation tables selected per row of
the packed sequence via index_select (tables are [3, D], indices [S]).

Real graph (verified by dump, before-freezing):
    index_select   = scale_table.index_select(0, indices)   # [S, D]
    add_9          = index_select + 1.0                      # scale+1
    mul_18         = x * add_9                               # x*(scale+1)
    index_select_1 = shift_table.index_select(0, indices)   # [S, D]
    add_10         = mul_18 + index_select_1                 # +shift

Replacement: mindiesd gather_scale_shift(x, scale_table, shift_table, indices)
-> ONE triton kernel doing gather + scale-shift (tables L2-resident). This
beats the plain scale_shift(x, scale, shift) fusion (which left the two
index_select kernels standalone and moved [S, D] scale/shift twice):
bench 910B+CANN9.1: gather kernel 80us warm / 131us cold vs plain 94/127,
plus the two standalone gathers (33us/site) disappear.
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()

from mindiesd.layers.scale_shift import (
    gather_scale_shift as mindie_gather_scale_shift,  # noqa: E402
)


def create(dtype):
    class MiniMaxH3AdaLnPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 4, 5376, dtype=dtype, device="meta")
            scale_table = torch.empty(3, 5376, dtype=dtype, device="meta")
            shift_table = torch.empty(3, 5376, dtype=dtype, device="meta")
            indices = torch.empty(4, dtype=torch.int64, device="meta")
            return [x, scale_table, shift_table, indices]

        @staticmethod
        def pattern(x, scale_table, shift_table, indices):
            def func(x, scale_table, shift_table, indices):
                scale = torch.ops.aten.index_select.default(scale_table, 0, indices)
                shift = torch.ops.aten.index_select.default(shift_table, 0, indices)
                add = torch.ops.aten.add.Tensor(scale, 1.0)
                mul = torch.ops.aten.mul.Tensor(x, add)
                return torch.ops.aten.add.Tensor(mul, shift)

            return func(x, scale_table, shift_table, indices)

        @staticmethod
        def replacement(x, scale_table, shift_table, indices):
            def func(x, scale_table, shift_table, indices):
                return mindie_gather_scale_shift(x, scale_table, shift_table, indices)

            return func(x, scale_table, shift_table, indices)

    return MiniMaxH3AdaLnPattern


MiniMaxH3AdaLnPatternGroup = [
    create(dtype=torch.bfloat16),
    create(dtype=torch.float32),
]
