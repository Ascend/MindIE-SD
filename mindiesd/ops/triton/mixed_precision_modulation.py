#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""FP32 modulation with explicit BF16 scale rounding and separate FP32 operations."""

import torch
from mindiesd.layers.triton_utils import _HAS_TRITON, _TRITON_ON_ASCEND, get_vectorcore_num

if _HAS_TRITON:
    from mindiesd.layers.triton_utils import triton, tl

    @triton.jit
    def _modulation_kernel(
        X,
        S,
        T,
        Y,
        N: tl.constexpr,
        D: tl.constexpr,
        SEQ: tl.constexpr,
        MS: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        T0: tl.constexpr,
        T1: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cores = tl.num_programs(0)
        col = tl.arange(0, C)[None, :]
        for start in range(pid * R, N, cores * R):
            row = start + tl.arange(0, R)[:, None]
            sr = row % SEQ if MS != 1 else tl.full((R, 1), 0, tl.int32)
            batch = row // SEQ
            x = tl.load(X + row * D + col, (row < N) & (col < D), 0)
            scale = tl.load(S + batch * S0 + sr * S1 + col, (row < N) & (col < D), 0).to(tl.float32)
            shift = tl.load(T + batch * T0 + sr * T1 + col, (row < N) & (col < D), 0).to(tl.float32)
            factor = (scale + 1.0).to(tl.bfloat16).to(tl.float32)
            product = x * factor
            out = product + shift
            tl.store(Y + row * D + col, out, (row < N) & (col < D))


@torch.library.custom_op("mindiesd::mixed_precision_modulation", mutates_args=())
def mixed_precision_modulation(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # The kernel is defined only when Triton is available, matching this guard.
    # pylint: disable=possibly-used-before-assignment
    if not _TRITON_ON_ASCEND:
        raise RuntimeError("Mixed precision modulation requires Triton-Ascend.")
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.numel() // x.shape[-1]
    _modulation_kernel[(min(rows, get_vectorcore_num()),)](
        x,
        scale,
        shift,
        output,
        rows,
        x.shape[-1],
        x.shape[1],
        scale.shape[1],
        scale.stride(0),
        scale.stride(1),
        shift.stride(0),
        shift.stride(1),
        2,
        triton.next_power_of_2(x.shape[-1]),
        enable_fp_fusion=False,
    )
    return output


@mixed_precision_modulation.register_fake
def _mixed_precision_modulation_fake(x, scale, shift):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)
