#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import re

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu

    import mindiesd

_torch_version_tuple = tuple(int(x) for x in re.match(r"(\d+)\.(\d+)", torch.__version__).groups())
# torch_npu >= 2.9 移除 Tensor.to 的 NPU 过适配(MR 30358)后,
# dtype cast 由 npu._npu_dtype_cast 变为 aten._to_copy。
IS_TORCH_GE_29 = _torch_version_tuple >= (2, 9)


def create(dtype):
    if "2.6.0" in torch.__version__:
        _dtype_cast_func = torch.ops.npu.npu_dtype_cast.default
    else:
        if IS_TORCH_GE_29:
            _dtype_cast_func = lambda x, dtype: torch.ops.aten._to_copy.default(x, dtype=dtype)
        else:
            _dtype_cast_func = torch.ops.npu._npu_dtype_cast.default


    class RopePattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(2, 2, 2, 2, dtype=torch.bfloat16, device="meta")
            cos = torch.empty(1, 2, 1, 2, dtype=dtype, device="meta")
            sin = torch.empty(1, 2, 1, 2, dtype=dtype, device="meta")
            return [x, cos, sin]

        @staticmethod
        def pattern(x, cos, sin):
            def func(x, cos, sin):
                x_real, x_img = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
                x_rotated = torch.stack([-x_img, x_real], dim=-1).flatten(3)
                if dtype == torch.bfloat16:
                    cos_part = x * cos
                    sin_part = x_rotated * sin
                else:
                    cos_part = _dtype_cast_func(x, dtype) * cos
                    sin_part = _dtype_cast_func(x_rotated, dtype) * sin
                x_out = cos_part + sin_part
                return torch.ops.aten._to_copy.default(x_out, dtype=x.dtype)
            return func(x, cos, sin)

        @staticmethod
        def replacement(x, cos, sin):
            def func(x, cos, sin):
                norm_q = mindiesd.rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved",
                                                            head_first=False, fused=True)
                return norm_q
            return func(x, cos, sin)

    return RopePattern

RopePatternGroup = [create(torch.bfloat16), create(torch.float32)]
