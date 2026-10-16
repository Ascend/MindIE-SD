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

"""Wan2.2 qk_norm (RMSNorm) fusion: pow->mean->add->rsqrt->mul->mul -> npu_rms_norm.

注意: 真实图 add(mean, eps) 的 eps 是 float32 舍入常量(9.999999974752427e-07),
pattern 常量按 == 精确匹配, 必须同样舍入否则静默不命中。
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu  # noqa: F401


def create(dtype, epsilon=1e-6):
    _eps_in_fp32 = torch.tensor(epsilon, dtype=torch.float32, device="cpu").item()

    class WanRmsNormPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 75600, 5120, dtype=dtype, device="meta")
            weight = torch.empty(5120, dtype=dtype, device="meta")
            return [x, weight]

        @staticmethod
        def pattern(x, weight):
            def func(x, weight):
                variance = torch.ops.aten.pow.Tensor_Scalar(x, 2)
                mean = torch.ops.aten.mean.dim(variance, [x.dim() - 1], True)
                add = torch.ops.aten.add.Scalar(mean, _eps_in_fp32)
                rsqrt = torch.ops.aten.rsqrt.default(add)
                result = x * rsqrt
                return result * weight

            return func(x, weight)

        @staticmethod
        def replacement(x, weight):
            def func(x, weight):
                # npu_rms_norm 要求 x/gamma 同 dtype
                return torch_npu.npu_rms_norm(x, weight.to(x.dtype), epsilon=_eps_in_fp32)[0]

            return func(x, weight)

    return WanRmsNormPattern


WanRmsNormPatternGroup = [create(dtype=torch.bfloat16), create(dtype=torch.float32)]
