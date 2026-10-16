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

"""MiniMax-H3 RMSNorm fusion pattern (register_replacement, hand-written chain).

MiniMax-H3 uses torch.nn.RMSNorm (norm1/norm2, qk_norm, token refiner, norm_out).
On torch 2.11 Dynamo already lowers torch.rms_norm into a chain BEFORE freezing
(verified by graph dump):

    _to_copy(x, fp32) -> pow(x,2) -> mean(last_dim) -> add.Scalar(eps)
        -> rsqrt -> mul(x_f32, rsqrt) -> mul(weight)          # bf16 variant

so the before-freezing pattern matcher matches it once; no backend change needed.

Why hand-write the chain instead of calling torch.rms_norm? make_fx decomposes
torch.rms_norm via composite op expansion and produces `add_.Scalar` (inplace),
while the real graph produces `add.Scalar` -- target mismatch, 0 replacements.
Hand-writing pins every target (add.Scalar, non-inplace) and the mean dim
([x.dim()-1], 2 for 3-D, 3 for 4-D). The pattern stops at the terminal mul so
the graph-side `_to_copy(bf16)` after it stays outside the matched subgraph.

Replacement: torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu  # noqa: F401


def create(dtype, rank, epsilon=1e-5):
    # freeze 分解链中 add.Scalar(mean, eps) 的 eps 是 float32 舍入后的常量
    # (9.999999747378752e-06), pattern 常量参数按 == 精确匹配, 必须用该值。
    _eps_in_fp32 = torch.tensor(epsilon, dtype=torch.float32, device="cpu").item()

    class MiniMaxH3RmsNormPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}-{rank}d"

        @staticmethod
        def inputs():
            if rank == 3:
                x = torch.empty(1, 4, 5376, dtype=dtype, device="meta")
            else:
                x = torch.empty(1, 4, 56, 128, dtype=dtype, device="meta")
            weight = torch.empty(x.shape[-1], dtype=dtype, device="meta")
            return [x, weight]

        @staticmethod
        def pattern(x, weight):
            def func(x, weight):
                # hand-written freeze-decomposed chain (non-inplace add.Scalar)
                if dtype == torch.bfloat16:
                    x = torch.ops.aten._to_copy.default(x, dtype=torch.float32)
                variance = torch.ops.aten.pow.Tensor_Scalar(x, 2)
                mean = torch.ops.aten.mean.dim(variance, [x.dim() - 1], True)
                add = torch.ops.aten.add.Scalar(mean, _eps_in_fp32)
                rsqrt = torch.ops.aten.rsqrt.default(add)
                result = torch.ops.aten.mul.Tensor(x, rsqrt)
                return torch.ops.aten.mul.Tensor(result, weight)

            return func(x, weight)

        @staticmethod
        def replacement(x, weight):
            def func(x, weight):
                # npu_rms_norm 要求 x/gamma 同 dtype;weight 对齐 x 的 dtype
                return torch_npu.npu_rms_norm(x, weight.to(x.dtype), epsilon=_eps_in_fp32)[0]

            return func(x, weight)

    return MiniMaxH3RmsNormPattern


MiniMaxH3RmsNormPatternGroup = [
    create(dtype=torch.bfloat16, rank=3),
    create(dtype=torch.bfloat16, rank=4),
    create(dtype=torch.float32, rank=3),
    create(dtype=torch.float32, rank=4),
]
