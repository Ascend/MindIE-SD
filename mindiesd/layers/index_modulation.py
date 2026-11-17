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
from functools import lru_cache
from typing import Final

import torch

from .triton_utils import _HAS_TRITON, get_vectorcore_num

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


@lru_cache(maxsize=None)
def _get_npu_resources(device_index: int) -> tuple[int, int]:
    with torch.npu.device(device_index):
        vector_count = int(get_vectorcore_num())
        device_name = torch.npu.get_device_name(device_index)
    ub_sizes = {"ascend950": 216 * 1024, "ascend910": 183 * 1024}
    ub_size = next((size for series, size in ub_sizes.items() if device_name.lower().startswith(series)), None)
    if ub_size is None:
        raise RuntimeError(f"Unsupported NPU device: {device_name}; expected Ascend 910 or 950")
    if vector_count <= 0:
        raise RuntimeError(f"Invalid NPU resources for {device_name}: vector_count={vector_count}, ub_size={ub_size}")
    return vector_count, ub_size


def _tile_shape(width: int, ub_size: int) -> tuple[int, int]:
    """Choose (block_rows, block_cols) within the UB budget, given in bytes."""
    # Split the UB budget between the kernel's two pipeline stages.
    STAGES: Final[int] = 2
    # Conservative per-element allowance per stage, including compiler temporaries.
    BYTES_PER_ELEMENT: Final[int] = 16
    BYTES_PER_INDEX: Final[int] = 8
    # Column alignment in elements, not bytes: 32 BF16 elements occupy 64 bytes.
    COLUMN_ALIGNMENT: Final[int] = 32
    bytes_per_stage = ub_size // STAGES
    width = triton.cdiv(width, COLUMN_ALIGNMENT) * COLUMN_ALIGNMENT
    # Use a single row if two complete rows and their indices do not fit in one stage.
    if_single_row = ((bytes_per_stage - 2 * BYTES_PER_INDEX) // BYTES_PER_ELEMENT // width) <= 1
    if if_single_row:
        # Reserve one index; split wide rows into column chunks when needed.
        max_cols = (bytes_per_stage - BYTES_PER_INDEX) // (BYTES_PER_ELEMENT * COLUMN_ALIGNMENT) * COLUMN_ALIGNMENT
        block_rows = 1
        block_cols = min(width, max_cols)
    else:
        # Batch as many complete rows as fit, including one index per row.
        block_rows = bytes_per_stage // (BYTES_PER_ELEMENT * width + BYTES_PER_INDEX)
        # After reserving indices, round the column budget down to a multiple of width.
        block_cols = (
            (bytes_per_stage - block_rows * BYTES_PER_INDEX) // block_rows // BYTES_PER_ELEMENT // width * width
        )
    return block_rows, block_cols


if _HAS_TRITON:

    @triton.jit
    def _indexed_gate_kernel(
        output_ptr,
        x_ptr,
        gate_ptr,
        other_ptr,
        indices_ptr,
        stride_output_row: tl.constexpr,
        stride_x_row: tl.constexpr,
        stride_gate_row: tl.constexpr,
        stride_other_row: tl.constexpr,
        stride_indices: tl.constexpr,
        rows: tl.constexpr,
        width: tl.constexpr,
        column_tiles: tl.constexpr,
        tiles_per_program: tl.constexpr,
        extra_tiles: tl.constexpr,
        block_rows: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        # Each output row uses its index to select a gate row: output = x + gate[indices] * other.
        program = tl.program_id(0)
        first_tile = program * tiles_per_program + tl.minimum(program, extra_tiles)
        program_tiles = tiles_per_program + (program < extra_tiles)
        columns = tl.arange(0, block_cols)

        for local_tile in tl.range(0, program_tiles, num_stages=2):
            tile = first_tile + local_tile
            first_row = (tile // column_tiles) * block_rows
            cols = (tile % column_tiles) * block_cols + columns

            if block_rows == 1:
                # Process one row or a column chunk of a wide row, using a scalar gate index.
                index = tl.load(indices_ptr + first_row * stride_indices, care_padding=False)
                # Mask columns beyond the row width in the final chunk.
                mask = cols < width
                # Load x and other from the current row, and gate from the indexed row, all in FP32.
                x = tl.load(x_ptr + first_row * stride_x_row + cols, mask=mask, care_padding=False).to(tl.float32)
                gate = tl.load(gate_ptr + index * stride_gate_row + cols, mask=mask, care_padding=False).to(tl.float32)
                other = tl.load(
                    other_ptr + first_row * stride_other_row + cols,
                    mask=mask,
                    care_padding=False,
                ).to(tl.float32)
                # Store valid lanes only; masked lanes need no padding values for this elementwise operation.
                tl.store(output_ptr + first_row * stride_output_row + cols, x + gate * other, mask=mask)
            else:
                # Batch narrow rows, broadcasting each row's gate index across its columns.
                tile_rows = first_row + tl.arange(0, block_rows)
                indices = tl.load(indices_ptr + tile_rows * stride_indices, mask=tile_rows < rows, care_padding=False)
                mask = (tile_rows[:, None] < rows) & (cols[None, :] < width)
                x = tl.load(
                    x_ptr + tile_rows[:, None] * stride_x_row + cols[None, :],
                    mask=mask,
                    care_padding=False,
                ).to(tl.float32)
                gate = tl.load(
                    gate_ptr + indices[:, None] * stride_gate_row + cols[None, :],
                    mask=mask,
                    care_padding=False,
                ).to(tl.float32)
                other = tl.load(
                    other_ptr + tile_rows[:, None] * stride_other_row + cols[None, :],
                    mask=mask,
                    care_padding=False,
                ).to(tl.float32)
                tl.store(
                    output_ptr + tile_rows[:, None] * stride_output_row + cols[None, :],
                    x + gate * other,
                    mask=mask,
                )


def _check_indexed_gate_inputs(output, x, gate, other, indices):
    if x.dim() != 2:
        raise ValueError("x must have shape [N, H]")
    rows, width = x.shape
    if rows == 0 or width == 0:
        raise ValueError("indexed_gate requires N > 0 and H > 0")
    if output.shape != x.shape or other.shape != x.shape:
        raise ValueError("output, x, and other must have the same [N, H] shape")
    if gate.dim() != 2 or gate.shape[0] == 0 or gate.shape[1] != width:
        raise ValueError("gate must have shape [K, H] with K > 0")
    if indices.shape != (rows,) or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must have shape [N] and dtype int32 or int64")
    tensors = (output, x, gate, other)
    if any(value.dtype != torch.bfloat16 for value in tensors):
        raise ValueError("output, x, gate, and other must all be bfloat16")
    if any(value.device != x.device for value in (*tensors, indices)):
        raise ValueError("all inputs and output must be on the same device")
    if any(value.stride(-1) != 1 for value in tensors):
        raise ValueError("output, x, gate, and other must be contiguous along the last dimension")
    # Row gaps are supported, but output rows must not overlap during parallel stores.
    if rows > 1 and output.stride(0) < width:
        raise ValueError("output rows must not overlap")
    for name, value in (
        ("x", x),
        ("gate", gate),
        ("other", other),
        ("indices", indices),
    ):
        if torch._C._overlaps(output, value):
            raise ValueError(f"output must not share storage with {name}")


def indexed_gate(
    output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Write x + gate[indices] * other into output, using FP32 arithmetic.

    output, x, and other are BF16 [N, H]; gate is BF16 [K, H] with N, H, K > 0.
    All tensors must be on one NPU. Floating tensors have unit last stride;
    output rows must not overlap. indices is int32/int64 [N], with arbitrary
    stride and caller-guaranteed values in [0, K).
    """
    _check_indexed_gate_inputs(output, x, gate, other, indices)
    if not _HAS_TRITON:
        raise RuntimeError("indexed_gate requires triton")
    if x.device.type != "npu":
        raise ValueError("indexed_gate requires Ascend NPU inputs")
    rows, width = x.shape

    vector_count, ub_size = _get_npu_resources(x.device.index)
    block_rows, block_cols = _tile_shape(width, ub_size)

    column_tiles = triton.cdiv(width, block_cols)
    row_tiles = triton.cdiv(rows, block_rows)
    num_tiles = row_tiles * column_tiles

    num_programs = min(vector_count, num_tiles)
    tiles_per_program = num_tiles // num_programs
    extra_tiles = num_tiles % num_programs

    grid = (num_programs,)
    with torch.npu.device(x.device):
        _indexed_gate_kernel[grid](  # pylint: disable=possibly-used-before-assignment
            output,
            x,
            gate,
            other,
            indices,
            stride_output_row=output.stride(0),
            stride_x_row=x.stride(0),
            stride_gate_row=gate.stride(0),
            stride_other_row=other.stride(0),
            stride_indices=indices.stride(0),
            rows=rows,
            width=width,
            column_tiles=column_tiles,
            tiles_per_program=tiles_per_program,
            extra_tiles=extra_tiles,
            block_rows=block_rows,
            block_cols=block_cols,
        )
    return output
