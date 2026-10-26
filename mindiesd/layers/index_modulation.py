#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Indexed RMSNorm with an explicit multi-row loop per Triton program."""

from __future__ import annotations

import math
import struct

import torch

from .triton_utils import _HAS_TRITON

if _HAS_TRITON:
    from .triton_utils import tl, triton
    from triton.runtime import driver


def _get_num_vector_cores(device: torch.device) -> int:
    if not _HAS_TRITON:
        raise RuntimeError("triton is required to query the Ascend vector core count")
    device_index = device.index
    if device_index is None:
        device_index = driver.active.get_current_device()  # pylint: disable=possibly-used-before-assignment
    properties = driver.active.utils.get_device_properties(device_index)
    num_cores = int(properties["num_vectorcore"])
    if num_cores <= 0:
        raise RuntimeError(f"Invalid Vector core count for {device}: {num_cores}")
    return num_cores


if _HAS_TRITON:
    # Keep the baseline kernel name for existing profiling filters.
    @triton.jit
    def _rms_norm_indexed_scale_shift_kernel(
        rows: tl.constexpr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
        x_ptr,
        stride_x_row,
        weight_ptr,
        shift_ptr,
        stride_shift_row,
        scale_ptr,
        stride_scale_row,
        indices_ptr,
        stride_indices,
        output_ptr,
    ):
        lane_ids = tl.arange(0, block_n)

        # A program handles pid, pid + grid_size, pid + 2 * grid_size, ...
        # Leave cross-iteration buffering and scheduling to the compiler.
        for row_id in range(tl.program_id(0), rows, tl.num_programs(0)):
            group_id = tl.load(indices_ptr + row_id * stride_indices)

            row_value = tl.load(x_ptr + row_id * stride_x_row + lane_ids).to(tl.float32)
            norm_gain = tl.load(weight_ptr + lane_ids).to(tl.float32)
            row_scale = tl.load(scale_ptr + group_id * stride_scale_row + lane_ids).to(tl.float32)
            row_bias = tl.load(shift_ptr + group_id * stride_shift_row + lane_ids).to(tl.float32)
            # block_n == hidden_size, so the reduction has no padding lanes.
            mean_square = tl.sum(row_value * row_value, axis=0) / hidden_size
            norm_value = row_value * tl.rsqrt(mean_square + eps) * norm_gain
            result = norm_value * (1.0 + row_scale) + row_bias
            tl.store(
                output_ptr + row_id * hidden_size + lane_ids,
                result,
            )


def _check_inputs(x, weight, shift, scale, indices, eps):
    if x.dim() != 2:
        raise ValueError("x must have shape [N, H]")
    rows, hidden_size = x.shape
    if not 0 < hidden_size <= 8192:
        raise ValueError(
            "rms_norm_indexed_scale_shift requires 1 <= block_n = hidden_size <= 8192, "
            f"but got x.shape={tuple(x.shape)} (hidden_size={hidden_size})."
        )
    if weight.shape != (hidden_size,):
        raise ValueError("weight must have shape [H]")
    if shift.dim() != 2 or shift.shape[0] == 0 or shift.shape[1] != hidden_size:
        raise ValueError("shift must have shape [K, H] with K > 0")
    if scale.shape != shift.shape:
        raise ValueError("shift and scale must have the same [K, H] shape")
    if indices.shape != (rows,) or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must have shape [N] and dtype int32 or int64")
    tensors = (x, weight, shift, scale)
    if any(value.dtype != torch.bfloat16 for value in tensors):
        raise ValueError("x, weight, shift, and scale must all be bfloat16")
    if any(value.device != x.device for value in (*tensors, indices)):
        raise ValueError("all inputs must be on the same device")
    # The kernel addresses the innermost dimension as a contiguous run, so the
    # last dim of every participating tensor must be stride-1. The row stride is
    # still taken from the tensor (x.stride(0)), which allows non-contiguous
    # row strides such as the NaN-filled production-width case.
    if x.stride(-1) != 1 or weight.stride(-1) != 1 or shift.stride(-1) != 1 or scale.stride(-1) != 1:
        raise ValueError(
            "x, weight, shift, and scale must be contiguous along the last dimension; "
            f"got x.stride={x.stride()} weight.stride={weight.stride()} "
            f"shift.stride={shift.stride()} scale.stride={scale.stride()}"
        )
    try:
        eps_fp32 = struct.unpack("f", struct.pack("f", eps))[0]
    except OverflowError:
        eps_fp32 = math.inf
    if not math.isfinite(eps_fp32) or eps_fp32 <= 0:
        raise ValueError("eps must be finite and positive in FP32")


def rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm and indexed affine on NPU, with FP32 arithmetic and a final BF16 store.

    x is [N,H], weight is [H], and shift/scale are [K,H], all BF16 on one NPU.
    1 <= H <= 8192, K > 0; N may be zero. Floating inputs have unit last stride;
    row gaps and storage offsets are allowed. indices is int32/int64 [N], with
    arbitrary stride and caller-guaranteed values in [0,K). eps must remain
    finite and positive in FP32. Returns a new contiguous tensor without
    modifying inputs. Requires Triton on Ascend; inference only.
    """
    _check_inputs(x, weight, shift, scale, indices, eps)
    if not _HAS_TRITON:
        raise RuntimeError("rms_norm_indexed_scale_shift requires triton")
    if x.device.type != "npu":
        raise ValueError("rms_norm_indexed_scale_shift requires Ascend NPU inputs")
    rows, hidden_size = x.shape
    # Always produce a contiguous output regardless of x's memory format.
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows:
        # Triton obtains the launch device and stream from the current context.
        with torch.npu.device(x.device):
            grid_size = min(_get_num_vector_cores(x.device), rows)
            _rms_norm_indexed_scale_shift_kernel[(grid_size,)](  # pylint: disable=possibly-used-before-assignment
                rows=rows,
                hidden_size=hidden_size,
                eps=eps,
                block_n=hidden_size,
                x_ptr=x,
                stride_x_row=x.stride(0),
                weight_ptr=weight,
                shift_ptr=shift,
                stride_shift_row=shift.stride(0),
                scale_ptr=scale,
                stride_scale_row=scale.stride(0),
                indices_ptr=indices,
                stride_indices=indices.stride(0),
                output_ptr=output,
                num_warps=8,
            )
    return output
