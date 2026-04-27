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

import torch

from .triton_utils import _HAS_TRITON, _TRITON_ON_ASCEND

if _HAS_TRITON:
    from .triton_utils import triton, tl


if _HAS_TRITON:

    @triton.jit
    def muls_add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        scale,
        n_elements,
        n_blocks,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_programs = tl.num_programs(axis=0)
        for block_id in range(pid, n_blocks, num_programs):
            block_start = block_id * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x * scale + y
            tl.store(output_ptr + offsets, output, mask=mask)


    def _muls_add_triton(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("Input tensors must have the same shape.")
        hidden_size = x.shape[-1]
        n_elements = x.numel()
        output = torch.empty_like(x)

        from .triton_utils import get_vectorcore_num

        num_cores = get_vectorcore_num()
        block_size = max(hidden_size // 2, 1024)
        n_blocks = (n_elements + block_size - 1) // block_size
        num_programs = min(n_blocks, num_cores)

        muls_add_kernel[(num_programs,)](
            x,
            y,
            output,
            scale,
            n_elements,
            n_blocks,
            BLOCK_SIZE=block_size,
        )
        return output

else:
    def _muls_add_triton(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
        raise RuntimeError("Triton is not available. Use torch fallback.")


@torch.library.custom_op("mindiesd::muls_add", mutates_args=())
def muls_add(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """Fused element-wise x * scale + y.

    Uses Triton kernel on Ascend NPU when available, otherwise falls back
    to native PyTorch operations (x * scale + y) which torch.compile can
    fuse into a single kernel on NPU.
    """
    if _TRITON_ON_ASCEND:
        return _muls_add_triton(x, y, scale)
    return x * scale + y


@muls_add.register_fake
def _(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.empty_like(x)
