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

"""Fused tensor scale-shift for MiniMax-H3 AdaLN modulation.

Two variants (both Triton kernels on Ascend NPU, torch-op fallbacks otherwise):

1. ``scale_shift(x, scale, shift)``: out = x * (1 + scale) + shift, with
   per-row scale/shift tensors [S, D] broadcast against x of [1, S, D].
   Plain elementwise; kept as the shape-generic fallback path.

2. ``gather_scale_shift(x, scale_table, shift_table, indices)``: out =
   x * (1 + scale_table[indices]) + shift_table[indices]. Absorbs the two
   ``index_select`` gathers (tables are tiny, e.g. [3, D] per-modality rows,
   fully L2-resident) so the whole AdaLN site becomes ONE kernel with half
   the memory traffic (no [S, D] scale/shift materialization). Benchmarked on
   910B+CANN9.1: 80us warm / 131us cold vs plain 94us/127us, and it also
   removes the two standalone gather kernels (33us/site) from the graph.
"""

import torch

from .triton_utils import _HAS_TRITON, _TRITON_ON_ASCEND

if _HAS_TRITON:
    from .triton_utils import tl, triton

# Max block elements the Ascend UB can hold (BS32768 failed to compile).
_MAX_BLOCK = 16384


if _HAS_TRITON:

    @triton.jit
    def scale_shift_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        n_elements,
        n_blocks,
        BLOCK_SIZE: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        pid = tl.program_id(axis=0)
        num_programs = tl.num_programs(axis=0)
        for block_id in range(pid, n_blocks, num_programs):
            offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            s = tl.load(scale_ptr + offs, mask=mask)
            sh = tl.load(shift_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x * (1.0 + s) + sh, mask=mask)

    @triton.jit
    def swiglu_kernel(
        x_ptr,
        out_ptr,
        F,  # noqa: N803  # triton half-dim arg
        n_chunk,
        BLOCK_D: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        # Row kernel: proj [S, 2F] = [hidden | gate]; out = hidden * silu(gate).
        # No concat needed vs npu_swiglu's single-input [gate|hidden] layout;
        # bench 276us vs cat+npu_swiglu 550us ([1,3967,28672], bf16).
        row = tl.program_id(axis=0)
        base = row * 2 * F
        obase = row * F
        for c in tl.range(0, n_chunk):
            col = c * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = col < F
            hv = tl.load(x_ptr + base + col, mask=mask)
            gv = tl.load(x_ptr + base + F + col, mask=mask)
            tl.store(out_ptr + obase + col, hv * (gv * tl.sigmoid(gv)), mask=mask)

    @triton.jit
    def gather_residual_gate_kernel(
        res_ptr,
        val_ptr,
        gate_ptr,
        idx_ptr,
        out_ptr,
        row_base,
        D,  # noqa: N803  # triton row-length arg
        ROWS: tl.constexpr,  # noqa: N803  # rows per program
        BLOCK_D: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        # out = residual + gate_table[idx[row]] * value (MiniMax residual gate).
        # Same tuning as gather_scale_shift: scalar idx gather (not discrete),
        # i32 (avoid i64 scalar lowering), ROWS rows/program.
        row0 = tl.program_id(axis=0) * ROWS + row_base
        col = tl.arange(0, BLOCK_D)
        for r in tl.static_range(ROWS):
            ii = tl.load(idx_ptr + row0 + r).to(tl.int32)
            offs = (row0 + r) * D + col
            rv = tl.load(res_ptr + offs)
            vv = tl.load(val_ptr + offs)
            gv = tl.load(gate_ptr + ii * D + col)
            tl.store(out_ptr + offs, rv + gv * vv)

    @triton.jit
    def gather_scale_shift_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        idx_ptr,
        out_ptr,
        row_base,
        D,  # noqa: N803  # triton row-length arg
        ROWS: tl.constexpr,  # noqa: N803  # rows per program
        BLOCK_D: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        # ROWS rows per program: 3x5376=16128 elements fits the Ascend UB,
        # fewer programs + more contiguous work per program (bench 57us warm
        # vs 80us for 1-row). idx is a scalar gather -> not discrete access;
        # cast to int32 to avoid i64 vector-arithmetic scalar lowering.
        row0 = tl.program_id(axis=0) * ROWS + row_base
        col = tl.arange(0, BLOCK_D)
        for r in tl.static_range(ROWS):
            ii = tl.load(idx_ptr + row0 + r).to(tl.int32)
            offs = (row0 + r) * D + col
            xv = tl.load(x_ptr + offs)
            sv = tl.load(scale_ptr + ii * D + col)
            shv = tl.load(shift_ptr + ii * D + col)
            tl.store(out_ptr + offs, xv * (1.0 + sv) + shv)

    def _scale_shift_triton(x: torch.Tensor, scale: torch.Tensor,
                            shift: torch.Tensor) -> torch.Tensor:
        # x: [1, S, D]; scale/shift: [S, D] -> flatten to 1D elementwise
        s = scale.reshape(-1).contiguous()
        sh = shift.reshape(-1).contiguous()
        xf = x.reshape(-1).contiguous()
        n_elements = xf.numel()
        out = torch.empty_like(xf)

        from .triton_utils import get_vectorcore_num

        num_cores = get_vectorcore_num()
        # BLOCK_SIZE tuning (Ascend 910B, bf16 [3967, 5376]): 1024 -> ~216us/call
        # (slower than the 3-aclnn eager chain it replaces); 8192 -> ~94us/call
        # warm / 127us cold. Keep 8192; gather_scale_shift is preferred anyway.
        block_size = 8192
        n_blocks = (n_elements + block_size - 1) // block_size
        num_programs = min(n_blocks, num_cores)

        scale_shift_kernel[(num_programs,)](
            xf, s, sh, out, n_elements, n_blocks, BLOCK_SIZE=block_size,
        )
        return out.reshape(x.shape)

    def _gather_scale_shift_triton(x: torch.Tensor, scale_table: torch.Tensor,
                                   shift_table: torch.Tensor,
                                   indices: torch.Tensor) -> torch.Tensor:
        # x: [1, S, D]; tables: [K, D]; indices: [S] (int64/int32)
        D = x.shape[-1]  # noqa: N806  # row length
        if D > _MAX_BLOCK or x.numel() % D != 0:
            return _gather_scale_shift_eager(x, scale_table, shift_table, indices)
        xc = x.reshape(-1, D).contiguous()
        st = scale_table.contiguous()
        ht = shift_table.contiguous()
        ii = indices.contiguous()
        out = torch.empty_like(x)
        rows = xc.shape[0]
        n3 = (rows // 3) * 3
        if n3 > 0:
            gather_scale_shift_kernel[(n3 // 3,)](
                xc, st, ht, ii, out, 0, D, ROWS=3, BLOCK_D=D,
            )
        if rows - n3 > 0:
            gather_scale_shift_kernel[(rows - n3,)](
                xc, st, ht, ii, out, n3, D, ROWS=1, BLOCK_D=D,
            )
        return out

    def _gather_scale_shift_eager(x: torch.Tensor, scale_table: torch.Tensor,
                                  shift_table: torch.Tensor,
                                  indices: torch.Tensor) -> torch.Tensor:
        scale = scale_table.index_select(0, indices)
        shift = shift_table.index_select(0, indices)
        return x * (1.0 + scale) + shift

    def _swiglu_triton(x: torch.Tensor) -> torch.Tensor:
        # x: [..., 2F] (proj); out: [..., F] = hidden * silu(gate)
        F = x.shape[-1] // 2  # noqa: N806  # half dim
        xc = x.reshape(-1, 2 * F).contiguous()
        out = torch.empty(xc.shape[0], F, dtype=x.dtype, device=x.device)
        # BLOCK_D=7168 divides F=14336 (2 chunks); generic ceil-chunk w/ mask.
        block_d = 7168 if F % 7168 == 0 else min(F, 8192)
        n_chunk = (F + block_d - 1) // block_d
        swiglu_kernel[(xc.shape[0],)](xc, out, F, n_chunk, BLOCK_D=block_d)
        return out.reshape(x.shape[:-1] + (F,))

    def _gather_residual_gate_triton(residual: torch.Tensor, value: torch.Tensor,
                                     gate_table: torch.Tensor,
                                     indices: torch.Tensor) -> torch.Tensor:
        # out = residual + gate_table[indices] * value, all [1, S, D]
        D = value.shape[-1]  # noqa: N806  # row length
        if D > _MAX_BLOCK or value.numel() % D != 0:
            return _gather_residual_gate_eager(residual, value, gate_table, indices)
        rc = residual.reshape(-1, D).contiguous()
        vc = value.reshape(-1, D).contiguous()
        gt = gate_table.contiguous()
        ii = indices.contiguous()
        out = torch.empty_like(value)
        rows = vc.shape[0]
        n3 = (rows // 3) * 3
        if n3 > 0:
            gather_residual_gate_kernel[(n3 // 3,)](
                rc, vc, gt, ii, out, 0, D, ROWS=3, BLOCK_D=D,
            )
        if rows - n3 > 0:
            gather_residual_gate_kernel[(rows - n3,)](
                rc, vc, gt, ii, out, n3, D, ROWS=1, BLOCK_D=D,
            )
        return out

    def _gather_residual_gate_eager(residual: torch.Tensor, value: torch.Tensor,
                                    gate_table: torch.Tensor,
                                    indices: torch.Tensor) -> torch.Tensor:
        gate = gate_table.index_select(0, indices)
        return residual + gate * value

else:
    def _scale_shift_triton(x: torch.Tensor, scale: torch.Tensor,
                            shift: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Triton is not available. Use torch fallback.")

    def _gather_scale_shift_triton(x: torch.Tensor, scale_table: torch.Tensor,
                                   shift_table: torch.Tensor,
                                   indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Triton is not available. Use torch fallback.")

    def _gather_scale_shift_eager(x: torch.Tensor, scale_table: torch.Tensor,
                                  shift_table: torch.Tensor,
                                  indices: torch.Tensor) -> torch.Tensor:
        scale = scale_table.index_select(0, indices)
        shift = shift_table.index_select(0, indices)
        return x * (1.0 + scale) + shift

    def _swiglu_triton(x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Triton is not available. Use torch fallback.")

    def _gather_residual_gate_triton(residual: torch.Tensor, value: torch.Tensor,
                                     gate_table: torch.Tensor,
                                     indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Triton is not available. Use torch fallback.")

    def _gather_residual_gate_eager(residual: torch.Tensor, value: torch.Tensor,
                                    gate_table: torch.Tensor,
                                    indices: torch.Tensor) -> torch.Tensor:
        gate = gate_table.index_select(0, indices)
        return residual + gate * value


@torch.library.custom_op("mindiesd::scale_shift", mutates_args=())
def scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Fused tensor scale-shift: out = x * (1 + scale) + shift.

    Args:
        x: activation, [1, S, D] (or any shape whose trailing dims match scale).
        scale: per-row scale, [S, D] broadcastable to x.
        shift: per-row shift, [S, D] broadcastable to x.

    Returns:
        Tensor of x's shape.
    """
    if _TRITON_ON_ASCEND:
        return _scale_shift_triton(x, scale, shift)
    return x * (1.0 + scale) + shift


@torch.library.custom_op("mindiesd::gather_scale_shift", mutates_args=())
def gather_scale_shift(x: torch.Tensor, scale_table: torch.Tensor,
                       shift_table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather + scale-shift fused: out = x * (1 + scale_table[indices]) + shift_table[indices].

    Args:
        x: activation, [1, S, D] (contiguous rows of D).
        scale_table: per-modality scale rows, [K, D] (tiny, L2-resident).
        shift_table: per-modality shift rows, [K, D].
        indices: per-row table index, [S] (int64/int32).

    Returns:
        Tensor of x's shape.
    """
    if _TRITON_ON_ASCEND:
        return _gather_scale_shift_triton(x, scale_table, shift_table, indices)
    return _gather_scale_shift_eager(x, scale_table, shift_table, indices)


@scale_shift.register_fake
def _(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@gather_scale_shift.register_fake
def _(x: torch.Tensor, scale_table: torch.Tensor, shift_table: torch.Tensor,
      indices: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("mindiesd::swiglu", mutates_args=())
def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU without concat: out = hidden * silu(gate) for proj [..., 2F].

    Replaces the cat([gate, hidden]) + npu_swiglu pair with one row kernel
    (bench 276us vs 550us on [1,3967,28672] bf16).

    Args:
        x: proj [..., 2F] = [hidden | gate].

    Returns:
        Tensor of shape [..., F].
    """
    if _TRITON_ON_ASCEND:
        return _swiglu_triton(x)
    hidden, gate = x.chunk(2, dim=-1)
    return hidden * torch.nn.functional.silu(gate)


@torch.library.custom_op("mindiesd::gather_residual_gate", mutates_args=())
def gather_residual_gate(residual: torch.Tensor, value: torch.Tensor,
                         gate_table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Residual gate fusion: out = residual + gate_table[indices] * value.

    Absorbs the index_select gather (table [3, D] L2-resident) + mul + add of
    MiniMax-H3's `hidden + gate_msa[adaln_indices] * attn_out` sites into one
    triton kernel (bench 61us vs 108us eager chain).

    Args:
        residual: [1, S, D].
        value: [1, S, D] (attn output or ff output).
        gate_table: per-modality gate rows, [K, D].
        indices: per-row table index, [S].

    Returns:
        Tensor of value's shape.
    """
    if _TRITON_ON_ASCEND:
        return _gather_residual_gate_triton(residual, value, gate_table, indices)
    return _gather_residual_gate_eager(residual, value, gate_table, indices)


@swiglu.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device)


@gather_residual_gate.register_fake
def _(residual: torch.Tensor, value: torch.Tensor, gate_table: torch.Tensor,
      indices: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)
