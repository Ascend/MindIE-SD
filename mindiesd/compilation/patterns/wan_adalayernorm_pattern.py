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

"""Wan2.2 adaLN modulation fusion:
native_layer_norm(x, [D], None, None, eps) -> mul(scale+1) -> add(shift) -> ops.adaln.

注意 scale+1 参数顺序(模型图为 add(scale, 1));native_layer_norm 不在分解表保留单节点。
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu  # noqa: F401

    import mindiesd


def create(dtype, epsilon=1e-6):
    class WanAdaLayerNormPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 75600, 5120, dtype=dtype, device="meta")
            scale = torch.empty(1, 1, 5120, dtype=dtype, device="meta")
            shift = torch.empty(1, 1, 5120, dtype=dtype, device="meta")
            return [x, scale, shift]

        @staticmethod
        def pattern(x, scale, shift):
            def func(x, scale, shift):
                ln_out = torch.ops.aten.native_layer_norm(
                    x, [x.shape[-1]], None, None, epsilon)[0]
                return ln_out * (scale + 1) + shift

            return func(x, scale, shift)

        @staticmethod
        def replacement(x, scale, shift):
            norm = torch.nn.LayerNorm(
                x.shape[-1], eps=epsilon, dtype=x.dtype, device=x.device)

            def func(x, scale, shift):
                return mindiesd.layernorm_scale_shift(
                    layernorm=norm, x=x, scale=scale, shift=shift, fused=True)

            return func(x, scale, shift)

    return WanAdaLayerNormPattern


WanAdaLayerNormPatternGroup = [create(dtype=torch.bfloat16), create(dtype=torch.float32)]
