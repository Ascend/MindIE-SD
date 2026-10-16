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

"""norm_out LayerNorm+modulation 融合 pattern (FLUX/Qwen 共用, Q3/F3)。

与 wan_adalayernorm 的区别: norm_out(FluxLayerNorm0 同构) 的调制形态是
`(1 + scale)[:, None]` —— **unsqueeze 在 add 之后**(图: add(scale,1) ->
unsqueeze -> mul), 而 norm1/norm2 的 modulation 是 `1 + scale[:, None]`
(unsqueeze 在 add 之前/预 unsqueeze 输入)。现有 wan_adalayernorm pattern 只
覆盖后者, 导致 FLUX/Qwen 各 3 处 norm_out 的 native_layer_norm 未融合
(graph dump 实证)。

匹配形态(FLUX/Qwen 同构):
    native_layer_norm(x) -> getitem[0] -> ln
    add(scale_raw, 1) -> unsqueeze(1) -> scale_unsq      # [1,D] -> [1,1,D]
    mul(ln, scale_unsq) -> add(mul, unsqueeze(shift_raw))
Replacement: mindiesd.layernorm_scale_shift(..., fused=True) -> adaln_v2。
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


def create(dtype, epsilon=1e-6):
    class NormOutAdaLayerNormPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 4, 512, dtype=dtype, device="meta")
            scale = torch.empty(1, 512, dtype=dtype, device="meta")
            shift = torch.empty(1, 512, dtype=dtype, device="meta")
            return [x, scale, shift]

        @staticmethod
        def pattern(x, scale, shift):
            def func(x, scale, shift):
                ln_out = torch.ops.aten.native_layer_norm.default(
                    x, [x.shape[-1]], None, None, epsilon)[0]
                return ln_out * (scale + 1).unsqueeze(1) + shift.unsqueeze(1)

            return func(x, scale, shift)

        @staticmethod
        def replacement(x, scale, shift):
            norm = torch.nn.LayerNorm(
                x.shape[-1], eps=epsilon, dtype=x.dtype, device=x.device)

            def func(x, scale, shift):
                return mindiesd.layernorm_scale_shift(
                    layernorm=norm, x=x, scale=scale, shift=shift, fused=True)

            return func(x, scale, shift)

    return NormOutAdaLayerNormPattern


NormOutAdaLayerNormPatternGroup = [
    create(dtype=torch.bfloat16, epsilon=1e-6),
    create(dtype=torch.float32, epsilon=1e-6),
]
