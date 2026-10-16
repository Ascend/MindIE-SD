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

"""Wan2.2 RoPE fusion: view/unbind/slice/empty_like+copy+slice_scatter 链
-> mindiesd.rotary_position_embedding(fused=True) -> npu_rotary_mul(interleave)。

关键形态(实测): cos/sin 布局 [B,S,1,D], slice dim=3 step=2, 输出即 4D。
"""

import sys

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu  # noqa: F401

    import mindiesd

_INT64_MAX = sys.maxsize


def create(dtype):
    class WanRopePattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 16, 40, 128, dtype=dtype, device="meta")
            cos = torch.empty(1, 16, 1, 128, dtype=torch.float32, device="meta")
            sin = torch.empty(1, 16, 1, 128, dtype=torch.float32, device="meta")
            return [x, cos, sin]

        @staticmethod
        def pattern(x, cos, sin):
            def func(x, cos, sin):
                x_shape = list(x.shape)
                x_view = torch.ops.aten.view.default(x, x_shape[:-1] + [x_shape[-1] // 2, 2])
                x1, x2 = torch.ops.aten.unbind.int(x_view, -1)
                cos_e = torch.ops.aten.slice.Tensor(cos, 3, 0, _INT64_MAX, 2)
                sin_o = torch.ops.aten.slice.Tensor(sin, 3, 1, _INT64_MAX, 2)
                out = torch.ops.aten.empty_like.default(x)
                sub = x1 * cos_e - x2 * sin_o
                s0 = torch.ops.aten.slice.Tensor(out, 3, 0, _INT64_MAX, 2)
                c0 = torch.ops.aten.copy.default(s0, sub)
                ss0 = torch.ops.aten.slice_scatter.default(out, c0, 3, 0, _INT64_MAX, 2)
                add = x1 * sin_o + x2 * cos_e
                s1 = torch.ops.aten.slice.Tensor(ss0, 3, 1, _INT64_MAX, 2)
                c1 = torch.ops.aten.copy.default(s1, add)
                ss1 = torch.ops.aten.slice_scatter.default(ss0, c1, 3, 1, _INT64_MAX, 2)
                return ss1

            return func(x, cos, sin)

        @staticmethod
        def replacement(x, cos, sin):
            def func(x, cos, sin):
                return mindiesd.layers.rotary_position_embedding(
                    x, cos, sin,
                    rotated_mode="rotated_interleaved",
                    head_first=False,
                    fused=True,
                )

            return func(x, cos, sin)

    return WanRopePattern


WanRopePatternGroup = [create(torch.bfloat16)]
