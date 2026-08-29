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

"""Fused residual + gate: out = x + y * gate (gate [B,1,D] 沿 S 广播, 输出与 x 同 dtype).

Wan residual 分支融合目标;triton kernel 按行分块(门控索引每行算一次,避免逐元素
div/mod 标量化),输出固定连续(否则下游 norm 多一次 transpose)。
"""

import torch

from .triton_utils import _HAS_TRITON, _TRITON_ON_ASCEND

if _HAS_TRITON:
    from .triton_utils import triton, tl


if _HAS_TRITON:

    @triton.jit
    def residual_gate_add_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        output_ptr,
        S,
        D,
        n_rows,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_programs = tl.num_programs(axis=0)
        d = tl.arange(0, BLOCK_SIZE)
        mask = d < D
        for row in range(pid, n_rows, num_programs):
            offs = row * D + d
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            g = tl.load(gate_ptr + (row // S) * D + d, mask=mask)
            out = x + y.to(tl.float32) * g
            tl.store(output_ptr + offs, out, mask=mask)


    def _residual_gate_add_triton(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or y.dim() != 3:
            print(f"[residual_gate_add] fallback (ndim) x={tuple(x.shape)} y={tuple(y.shape)} gate={tuple(gate.shape)}", flush=True)
            return _residual_gate_add_fallback(x, y, gate)
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape; got x={tuple(x.shape)} y={tuple(y.shape)}")
        if gate.dim() < 2 or gate.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"gate must be broadcastable [B,1,D] with D == x.shape[-1]; "
                f"got gate={tuple(gate.shape)} dim={gate.dim()} x={tuple(x.shape)} y={tuple(y.shape)} dtype={gate.dtype}"
            )

        B, S, D = x.shape
        n_rows = x.numel() // D
        # 必须连续: x 可能是非连续 view(如 transpose 节点),empty_like 会继承其 stride,
        # 导致下游 norm 需要一次额外 transpose
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)

        from .triton_utils import get_vectorcore_num

        num_cores = get_vectorcore_num()
        block_size = max(triton.next_power_of_2(D), 1024)

        residual_gate_add_kernel[(min(n_rows, num_cores),)](
            x,
            y,
            gate,
            output,
            S,
            D,
            n_rows,
            BLOCK_SIZE=block_size,
        )
        return output

else:

    def _residual_gate_add_triton(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Triton is not available. Use torch fallback.")


def _residual_gate_add_fallback(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # 保持 x 的 dtype(bf16 图下输出 bf16);fp32 累加更精确
    return (x + y.float() * gate.float()).to(x.dtype)


@torch.library.custom_op("mindiesd::residual_gate_add", mutates_args=())
def residual_gate_add(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fused residual + gate: out = x + y.float() * gate (fp32 in/out).

    Uses a Triton kernel on Ascend NPU when available, otherwise falls back
    to native PyTorch operations (which torch.compile can fuse).
    """
    if _TRITON_ON_ASCEND:
        return _residual_gate_add_triton(x, y, gate)
    return _residual_gate_add_fallback(x, y, gate)


@residual_gate_add.register_fake
def _(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
