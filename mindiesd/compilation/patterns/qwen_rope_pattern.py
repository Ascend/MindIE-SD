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

"""Qwen-Image RoPE fusion pattern (register_replacement, real-domain complex rotary).

Qwen-Image 的 attention 用 complex 旋转(`apply_rotary_emb_qwen(use_real=False)`):
原实现 `view_as_complex(x.float()) * freqs_cis` 依赖 fp32 复数。dummy run 的
compute_precision 机制已把该函数改写为实数域等价形式(dummy_run/model/
compute_precision.py `_rewrite_apply_rotary_emb_qwen`), 冻结图为:

    x.reshape(..., D/2, 2).unbind(-1) -> xr, xi          # [B,S,H,D/2]
    freqs.real -> unsqueeze(1) -> _to_copy(bf16) -> cos  # [S,1,D/2]
    freqs.imag -> unsqueeze(1) -> _to_copy(bf16) -> sin
    out_real = xr*cos - xi*sin;  out_imag = xr*sin + xi*cos
    stack([out_real, out_imag], -1).flatten(3).type_as(x)

即 pair-interleaved(rotated_interleaved) 旋转, cos/sin 为半维(每对 1 个角度)。
Replacement: npu_rotary_mul 需要全维 cos/sin(每元素 1 个角度), 在 replacement 内
repeat_interleave(2) 扩展 + 显式对齐 x.dtype(R1 教训: 否则 x 被提升回 fp32)。

注册顺序: 必须在 enable_wan_residual_gate 之前 —— residual_gate(`x + y*gate`)
会误匹配 rope 的 `add(mul(x,cos), mul(x_rot,sin))` 子图(实测 qwen 4D fallback),
先注册本 pattern 吃掉 rope 子图(与 minimax F2 教训同源)。
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


def create(dtype):
    class QwenRopePattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            x = torch.empty(1, 4, 24, 128, dtype=dtype, device="meta")
            freqs = torch.empty(4, 64, dtype=torch.complex64, device="meta")
            return [x, freqs]

        @staticmethod
        def pattern(x, freqs):
            def func(x, freqs):
                xr, xi = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
                if dtype == torch.bfloat16:
                    # bf16 图: freqs.real 是 fp32, `.to(x.dtype)` 是真实 cast, 必须保留
                    cos = freqs.real.unsqueeze(1).to(x.dtype)
                    sin = freqs.imag.unsqueeze(1).to(x.dtype)
                else:
                    # fp32 图: no-op cast 可能被 Dynamo 省略, pattern 不带 cast
                    cos = freqs.real.unsqueeze(1)
                    sin = freqs.imag.unsqueeze(1)
                out_real = xr * cos - xi * sin
                out_imag = xr * sin + xi * cos
                x_out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
                return x_out.type_as(x)

            return func(x, freqs)

        @staticmethod
        def replacement(x, freqs):
            def func(x, freqs):
                # 半维 cos/sin([S,D/2]) -> 全维([S,D]) 供 npu_rotary_mul;
                # 显式 cast 到 x.dtype 防止 x 被提升回 fp32(R1)。
                cos = freqs.real.repeat_interleave(2, dim=-1).to(x.dtype)
                sin = freqs.imag.repeat_interleave(2, dim=-1).to(x.dtype)
                return mindiesd.layers.rotary_position_embedding(
                    x, cos, sin,
                    rotated_mode="rotated_interleaved",
                    head_first=False,
                    fused=True,
                )

            return func(x, freqs)

    return QwenRopePattern


QwenRopePatternGroup = [
    create(dtype=torch.bfloat16),
    create(dtype=torch.float32),
]
