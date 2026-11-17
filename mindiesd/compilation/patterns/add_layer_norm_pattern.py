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

"""Affine residual LayerNorm, including the Wan TI2V autocast conversion chain.

The CANN kernel normalizes the unrounded sum. BF16 normalization therefore has
numerical tolerance, while its residual output preserves the original rounding.
Match compatible hidden sizes and retain the minimum sequence-length guard.
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase


def _is_add_layer_norm_match(match):
    try:
        tensors = [match.kwargs[name].meta["val"] for name in ("x", "residual", "weight", "bias")]
    except (AttributeError, KeyError):
        return False
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    x, residual, weight, bias = tensors
    return (
        x.device.type == "npu"
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.dim() == 3
        and x.shape[0] > 0
        and x.shape[1] >= 2048
        and x.shape == residual.shape
        and weight.shape == bias.shape == (x.shape[-1],)
        and all(value.device == x.device and value.dtype == x.dtype for value in tensors)
        and all(value.is_contiguous() for value in tensors)
    )


def create(dtype, autocast=False):
    class AddLayerNormPattern(PatternBase):
        # Pattern matching requires named tensor arguments.
        # pylint: disable=arguments-differ
        @staticmethod
        def name():
            return f"AddLayerNormPattern-{dtype}-autocast={autocast}"

        @staticmethod
        def inputs():
            return [
                torch.empty(1, 2048, 3072, dtype=dtype, device="meta"),
                torch.empty(1, 2048, 3072, dtype=dtype, device="meta"),
                torch.empty(3072, dtype=dtype, device="meta"),
                torch.empty(3072, dtype=dtype, device="meta"),
            ]

        @staticmethod
        def pattern(x, residual, weight, bias):
            # Matching retraces candidates before extra_check rejects non-affine inputs.
            summed = x + residual
            if autocast:
                normalized = torch.ops.aten.native_layer_norm.default(
                    summed.float(),
                    [summed.shape[-1]],
                    weight.float() if weight is not None else None,
                    bias.float() if bias is not None else None,
                    1e-6,
                )[0].to(dtype)
            else:
                normalized = torch.ops.aten.native_layer_norm.default(summed, [summed.shape[-1]], weight, bias, 1e-6)[0]
            return normalized, summed

        @staticmethod
        def replacement(x, residual, weight, bias):
            import torch_npu

            outputs = torch_npu.npu_add_layer_norm(x, residual, weight, bias, 1e-6, True)
            return outputs[0], outputs[3]

        extra_check = staticmethod(_is_add_layer_norm_match)

    return AddLayerNormPattern


AddLayerNormPatternGroup = [
    create(torch.bfloat16, autocast=True),
    create(torch.bfloat16),
    create(torch.float32),
]
