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

"""MiniMax-H3 RoPE fusion pattern (register_replacement, rotate_half).

MiniMax-H3 applies rotary only to the leading `rotary_dim` channels of every
head (`_apply_rotary_emb`, verified by frozen-graph dump):

    x_rot  = slice(x, -1, 0, 96)                  # rotary part (96 of 128)
    x_pass = slice(x, -1, 96, MAX)                # pass-through part
    cos4   = unsqueeze(unsqueeze(_to_copy(cos), 0), 2)   # [1, S, 1, 96]
    x1, x2 = split(x_rot, 48, -1)                 # chunk(2)
    rotated = cat([neg(x2), x1], -1)              # rotate_half
    out    = add(mul(x_rot, cos4), mul(rotated, sin4))
    result = cat([out, x_pass], -1).contiguous()  # outside the matched subgraph

The pattern matches the rotate_half chain of the rotary part only; the outer
slices/cat stay untouched (npu_rotary_mul cannot do partial rotation, so the
96-channel rotary part is fused and re-catenated outside).

Replacement: mindiesd rotary_position_embedding(x_rot, cos, sin,
    rotated_mode="rotated_half", head_first=False, fused=True) -> npu_rotary_mul.

Registered BEFORE enable_wan_residual_gate so the rope subgraph is consumed
first and the generic residual+gate pattern cannot mis-match it (F2 lesson).
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu  # noqa: F401

    import mindiesd

# config rope_freq_dim=16 -> rotary_dim = 2 * 3 * 16 = 96; half = 48
_ROTARY_DIM = 96
_HALF = _ROTARY_DIM // 2


def create(dtype):
    class MiniMaxH3RopePattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 4, 56, 128, dtype=dtype, device="meta")
            cos = torch.empty(4, _ROTARY_DIM, dtype=torch.float32, device="meta")
            sin = torch.empty(4, _ROTARY_DIM, dtype=torch.float32, device="meta")
            return [x, cos, sin]

        @staticmethod
        def pattern(x, cos, sin):
            def func(x, cos, sin):
                x_rot = torch.ops.aten.slice.Tensor(x, 3, 0, _ROTARY_DIM)
                x1, x2 = torch.ops.aten.split.Tensor(x_rot, _HALF, -1)
                rotated = torch.ops.aten.cat.default([torch.ops.aten.neg.default(x2), x1], -1)
                if dtype == torch.bfloat16:
                    cos = torch.ops.aten._to_copy.default(cos, dtype=torch.bfloat16)
                    sin = torch.ops.aten._to_copy.default(sin, dtype=torch.bfloat16)
                cos4 = torch.ops.aten.unsqueeze.default(torch.ops.aten.unsqueeze.default(cos, 0), 2)
                sin4 = torch.ops.aten.unsqueeze.default(torch.ops.aten.unsqueeze.default(sin, 0), 2)
                return torch.ops.aten.add.Tensor(
                    torch.ops.aten.mul.Tensor(x_rot, cos4),
                    torch.ops.aten.mul.Tensor(rotated, sin4),
                )

            return func(x, cos, sin)

        @staticmethod
        def replacement(x, cos, sin):
            def func(x, cos, sin):
                x_rot = torch.ops.aten.slice.Tensor(x, 3, 0, _ROTARY_DIM)
                if dtype == torch.bfloat16:
                    # 匹配到的 cos/sin 是 cast 前的 fp32 节点;显式对齐 x.dtype,
                    # 避免 rope.py 内 x.to(cos.dtype) 把 bf16 x 提升到 fp32 再降回。
                    cos = torch.ops.aten._to_copy.default(cos, dtype=x.dtype)
                    sin = torch.ops.aten._to_copy.default(sin, dtype=x.dtype)
                return mindiesd.layers.rotary_position_embedding(
                    x_rot, cos, sin,
                    rotated_mode="rotated_half",
                    head_first=False,
                    fused=True,
                )

            return func(x, cos, sin)

    return MiniMaxH3RopePattern


MiniMaxH3RopePatternGroup = [
    create(dtype=torch.bfloat16),
    create(dtype=torch.float32),
]
