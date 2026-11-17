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

import mindiesd


def _has_matching_modulation_dtype(match):
    """Keep mixed FP32/BF16 inputs out of the same-dtype AdaLN replacement.

    BF16 scale + 1 rounds before FP32 multiplication in the original graph.
    Mixed inputs must retain that rounding through the dedicated modulation pattern.
    """
    try:
        x, scale, shift = (match.kwargs[name].meta["val"] for name in ("x", "scale", "shift"))
    except (AttributeError, KeyError):
        return False
    return (
        all(isinstance(value, torch.Tensor) for value in (x, scale, shift))
        and x.dtype == scale.dtype == shift.dtype
        and x.device == scale.device == shift.device
    )


def create(dtype, epsilon=1e-6):
    class WanAdaLayerNormPattern(PatternBase):
        # Pattern matching requires named tensor arguments.
        # pylint: disable=arguments-differ
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
                ln_out = torch.ops.aten.native_layer_norm(x, [x.shape[-1]], None, None, epsilon)[0]
                return ln_out * (scale + 1) + shift

            return func(x, scale, shift)

        @staticmethod
        def replacement(x, scale, shift):
            norm = torch.nn.LayerNorm(x.shape[-1], eps=epsilon, dtype=x.dtype, device=x.device)

            def func(x, scale, shift):
                return mindiesd.layernorm_scale_shift(layernorm=norm, x=x, scale=scale, shift=shift, fused=True)

            return func(x, scale, shift)

        extra_check = staticmethod(_has_matching_modulation_dtype)

    return WanAdaLayerNormPattern


class WanMixedModulationPattern(PatternBase):
    # Pattern matching requires named tensor arguments.
    # pylint: disable=arguments-differ
    """Fuse modulation while retaining the original FP32 LayerNorm reduction."""

    @staticmethod
    def name():
        return "WanMixedModulationPattern-float32-bfloat16"

    @staticmethod
    def inputs():
        return [
            torch.empty(1, 2048, 3072, dtype=torch.float32, device="meta"),
            torch.empty(1, 1, 3072, dtype=torch.bfloat16, device="meta"),
            torch.empty(1, 1, 3072, dtype=torch.bfloat16, device="meta"),
        ]

    @staticmethod
    def pattern(x, scale, shift):
        return x * (scale + 1) + shift

    @staticmethod
    def replacement(x, scale, shift):
        # Keep scale + 1 in BF16 before promotion, preserving its rounding.
        from mindiesd.ops.triton.mixed_precision_modulation import mixed_precision_modulation

        return mixed_precision_modulation(x, scale, shift)

    @staticmethod
    def extra_check(match):
        try:
            x, scale, shift = (match.kwargs[key].meta["val"] for key in ("x", "scale", "shift"))
        except (AttributeError, KeyError):
            return False
        from mindiesd.layers.triton_utils import _TRITON_ON_ASCEND

        return (
            _TRITON_ON_ASCEND
            and all(isinstance(value, torch.Tensor) for value in (x, scale, shift))
            and x.device.type == "npu"
            and x.dtype == torch.float32
            and scale.dtype == shift.dtype == torch.bfloat16
            and x.device == scale.device == shift.device
            and x.dim() == scale.dim() == shift.dim() == 3
            and scale.shape == shift.shape
            and scale.shape[0] == x.shape[0]
            and x.shape[0] > 0
            # Short sequences did not benefit in the device benchmark.
            and x.shape[1] >= 16384
            and scale.shape[1] == x.shape[1]
            and x.is_contiguous()
            and scale.stride(-1) == shift.stride(-1) == 1
            and 0 < x.shape[-1] <= 4096
            and scale.shape[2] == x.shape[2]
        )


WanAdaLayerNormPatternGroup = [
    WanMixedModulationPattern,
    create(dtype=torch.bfloat16),
    create(dtype=torch.float32),
]
