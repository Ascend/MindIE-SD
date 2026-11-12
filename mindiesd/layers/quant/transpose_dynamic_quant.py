#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Fused Transpose + DynamicQuantV2 + Transpose Triton operator.

Per-group symmetric dynamic quant, bit-aligned with aclnnDynamicQuantV2:
``scale = max(|x * smooth|) / 448`` over the quantization ``axis`` and
``y = fp8_rn(x * smooth / scale)`` in float8_e4m3fn. The smooth product is
evaluated in fp32 and the quotient is cast to fp8 with a single RN rounding.
Both transposes are folded into index arithmetic, so no transpose copy is
launched. Smooth scales are optional, 1-D and indexed along the quantization
axis (length == x.shape[axis]), aligned with aclnnDynamicQuantV2.

Dispatch by quantization axis:
  - axis=0: single kernel, per-column max + quantize.
  - axis=-1: single kernel (full-load tier when cores would sit idle).
  - middle axes: two kernels cooperate — partial column max via atomic_max,
    then quantize with the full scale; y keeps the x layout.
"""

import math

import torch

from ..triton_utils import _HAS_TRITON, _TRITON_ON_ASCEND, get_vectorcore_num

if _HAS_TRITON:
    from ..triton_utils import tl, triton

_DTYPE_MAX = 448.0  # float8_e4m3fn max
_MAX_INT32 = 2**31 - 1
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# Tile candidates: BLOCK_M * BLOCK_K <= 16384 elements (UB budget).
_BLOCK_M = 64
_BLOCK_K = 64
# Middle-axis pass 1 (m-major load, max over axis 0): BLOCK_K=64 keeps the load
# tile at 8192 elements and M_GROUP=4 amortizes per-task overhead; measured
# ~1.7 TB/s vs ~0.7 TB/s for the k-major layout on Ascend950DT.
_TRANS_P1_BLOCK_M = 128
_TRANS_P1_BLOCK_K = 64
_TRANS_P1_M_GROUP = 4

# Axis-last full-load tier: when the regular row-block kernel cannot fill the
# device (cdiv(rows, BLOCK_M) < num_cores) and a whole row fits in the UB
# budget, process FULL_LOAD_BLOCK_ROW rows per program with the whole row
# resident in registers: the max pass and the quant pass share one GM read
# (halves read traffic) and grid utilization improves. Single-row scale div
# must stay a >=2-lane vector op to bit-match the chunked kernel's division
# lowering, hence BLOCK_ROW=2 (BLOCK_ROW=4 x 8192 overflows the UB).
_FULL_LOAD_BLOCK_ROW = 2
_FULL_LOAD_MAX_BLOCK_COL = 8192

# Middle-axis single-kernel tier: for moderate quant-axis lengths one kernel
# owns (batch, k-block) and sweeps the whole axis twice (max, then quantize),
# needing no atomic workspace and only one launch — measurably faster than the
# two-kernel path whenever the serial sweep stays short (m in (64, 8192];
# e.g. (14,8192,128) 1.39x, (4,2048,512) 4.5x). Beyond that the sweep starves
# the device (only batch*cdiv(k,BLOCK_K) programs) and the two-kernel path wins.
_SINGLE_KERNEL_MIN_M = 65
_SINGLE_KERNEL_MAX_M = 8192
_SINGLE_KERNEL_BLOCK_K = 32

# Shape-tiered tiling (mixed-strategy dispatch): once the workload is
# device-bound, BLOCK_M=128 halves the reduction-loop trip count and wins by
# ~10%; small shapes keep BLOCK_M=64 to preserve grid parallelism. Threshold
# picked between the measured small (1M elems, flat) and large (10M elems,
# +10%) tiers on Ascend950DT.
_LARGE_SHAPE_ELEMS = 4_000_000


def _pick_axis_blocks(rows: int, cols: int) -> tuple[int, int]:
    if rows * cols >= _LARGE_SHAPE_ELEMS:
        return 128, 64
    return _BLOCK_M, _BLOCK_K


def _pick_thin_axis_blocks(m_size: int) -> tuple[int, int] | None:
    """Tiling for a tiny quant axis (m_size <= 64) with a huge trailing dim.

    The default (BLOCK_M=128, BLOCK_K=64) tile would be almost entirely masked
    (only m_size of 128 rows are real), leaving each program iteration to move
    a few hundred bytes and the k-loop to degenerate into millions of latency-
    bound iterations. Shrinking BLOCK_M to the axis length and widening
    BLOCK_K keeps every iteration busy with real data.
    """
    if m_size > 64:
        return None
    block_m = max(2, 1 << max(0, m_size - 1).bit_length())
    return block_m, 8192 // block_m


if _HAS_TRITON:

    @triton.jit
    def _dynamic_quant_axis0_kernel(  # pragma: no cover
        x_ptr,
        smooth_ptr,
        y_ptr,
        scale_ptr,
        rows,
        cols,
        HAS_SMOOTH: tl.constexpr,  # noqa: N803  # triton constexpr naming
        DTYPE_MAX: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_M: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_K: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        """axis=0: each program owns column blocks, reduce max over rows."""
        pid = tl.program_id(0).to(tl.int32)
        num_col_blocks = tl.cdiv(cols, BLOCK_K)
        for kb in range(pid, num_col_blocks, tl.num_programs(0).to(tl.int32)):
            k_off = (kb * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int32)
            k_mask = k_off < cols

            max_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for mb in range(0, tl.cdiv(rows, BLOCK_M)):
                m_off = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
                mask2d = (m_off < rows)[:, None] & k_mask[None, :]
                offs = m_off[:, None] * cols + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    # smooth is 1-D along the quant axis (rows): load per row block,
                    # broadcast as [BLOCK_M, 1]. Boundary rows use other=0.0 so
                    # inp=0 never pollutes the max reduce.
                    smooth_vec = tl.load(smooth_ptr + m_off, mask=m_off < rows, other=0.0)
                    # fp32 product, aligned with aclnnDynamicQuantV2.
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[:, None]
                else:
                    inp = x_tile.to(tl.float32)
                # abs >= 0, so other=0.0 is a safe sentinel for the max reduce.
                max_acc = tl.maximum(max_acc, tl.max(tl.abs(inp), axis=0))

            scale = max_acc / DTYPE_MAX
            tl.store(scale_ptr + k_off, scale, mask=k_mask)
            # Zero columns keep scale=0 while dividing through 1.0 (y = 0).
            denom = tl.where(scale > 0.0, scale, 1.0)
            for mb in range(0, tl.cdiv(rows, BLOCK_M)):
                m_off = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
                mask2d = (m_off < rows)[:, None] & k_mask[None, :]
                offs = m_off[:, None] * cols + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    smooth_vec = tl.load(smooth_ptr + m_off, mask=m_off < rows, other=0.0)
                    # fp32 product, aligned with aclnnDynamicQuantV2.
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[:, None]
                else:
                    inp = x_tile.to(tl.float32)
                q = inp / denom[None, :]
                q = tl.minimum(tl.maximum(q, -DTYPE_MAX), DTYPE_MAX)
                # Single RN cast to fp8, aligned with aclnnDynamicQuantV2.
                y8 = q.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y8, mask=mask2d)

    @triton.jit
    def _dynamic_quant_axis_last_kernel(  # pragma: no cover
        x_ptr,
        smooth_ptr,
        y_ptr,
        scale_ptr,
        rows,
        cols,
        HAS_SMOOTH: tl.constexpr,  # noqa: N803  # triton constexpr naming
        DTYPE_MAX: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_M: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_K: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        """axis=-1: each program owns row blocks, reduce max over the last dim."""
        pid = tl.program_id(0).to(tl.int32)
        num_col_chunks = tl.cdiv(cols, BLOCK_K)
        for rb in range(pid, tl.cdiv(rows, BLOCK_M), tl.num_programs(0).to(tl.int32)):
            m_off = (rb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
            m_mask = m_off < rows

            max_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
            for kc in range(0, num_col_chunks):
                k_off = (kc * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int32)
                mask2d = m_mask[:, None] & (k_off < cols)[None, :]
                offs = m_off[:, None] * cols + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    # smooth is 1-D along the quant axis (cols): load per column
                    # block, broadcast as [1, BLOCK_K]. Boundary cols use 0.0.
                    smooth_vec = tl.load(smooth_ptr + k_off, mask=k_off < cols, other=0.0)
                    # fp32 product, aligned with aclnnDynamicQuantV2.
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[None, :]
                else:
                    inp = x_tile.to(tl.float32)
                max_acc = tl.maximum(max_acc, tl.max(tl.abs(inp), axis=1))

            scale = max_acc / DTYPE_MAX
            tl.store(scale_ptr + m_off, scale, mask=m_mask)
            denom = tl.where(scale > 0.0, scale, 1.0)
            for kc in range(0, num_col_chunks):
                k_off = (kc * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int32)
                mask2d = m_mask[:, None] & (k_off < cols)[None, :]
                offs = m_off[:, None] * cols + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    smooth_vec = tl.load(smooth_ptr + k_off, mask=k_off < cols, other=0.0)
                    # fp32 product, aligned with aclnnDynamicQuantV2.
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[None, :]
                else:
                    inp = x_tile.to(tl.float32)
                q = inp / denom[:, None]
                q = tl.minimum(tl.maximum(q, -DTYPE_MAX), DTYPE_MAX)
                # Single RN cast to fp8, aligned with aclnnDynamicQuantV2.
                y8 = q.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y8, mask=mask2d)

    @triton.jit
    def _dynamic_quant_axis_last_fullload_kernel(  # pragma: no cover
        x_ptr,
        smooth_ptr,
        y_ptr,
        scale_ptr,
        rows,
        cols,
        HAS_SMOOTH: tl.constexpr,  # noqa: N803  # triton constexpr naming
        DTYPE_MAX: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_ROW: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_COL: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        """axis=-1 full-load tier: rows resident in registers, one GM read.

        Mirrors the aclnn full-load strategy: each program loads whole rows
        (padded to BLOCK_COL lanes), computes the row max from the resident
        fp32 tile and quantizes the same tile, so x is read from GM once
        instead of twice. Requires cols <= BLOCK_COL and keeps BLOCK_COL a
        power of two; boundary lanes load 0.0 which never wins the max and is
        masked out of the store.
        """
        pid = tl.program_id(0).to(tl.int32)
        c_off = tl.arange(0, BLOCK_COL)
        c_mask = c_off < cols
        num_row_blocks = tl.cdiv(rows, BLOCK_ROW)
        for rb in range(pid, num_row_blocks, tl.num_programs(0).to(tl.int32)):
            r_off = (rb * BLOCK_ROW + tl.arange(0, BLOCK_ROW)).to(tl.int32)
            r_mask = r_off < rows
            mask2d = r_mask[:, None] & c_mask[None, :]
            offs = r_off[:, None] * cols + c_off[None, :]
            x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
            if HAS_SMOOTH:
                # smooth is 1-D along the quant axis (cols), broadcast [1, BLOCK_COL].
                smooth_vec = tl.load(smooth_ptr + c_off, mask=c_mask, other=0.0)
                inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[None, :]
            else:
                inp = x_tile.to(tl.float32)
            # [BLOCK_ROW]-shaped max keeps the scale division on the same vector
            # lowering as the chunked kernel (bit-identical results).
            amax = tl.max(tl.abs(inp), axis=1)
            scale = amax / DTYPE_MAX
            tl.store(scale_ptr + r_off, scale, mask=r_mask)
            denom = tl.where(scale > 0.0, scale, 1.0)
            q = inp / denom[:, None]
            q = tl.minimum(tl.maximum(q, -DTYPE_MAX), DTYPE_MAX)
            # Single RN cast to fp8, aligned with aclnnDynamicQuantV2.
            y8 = q.to(y_ptr.dtype.element_ty)
            tl.store(y_ptr + offs, y8, mask=mask2d)

    @triton.jit
    def _dynamic_quant_max_partial_kernel(  # pragma: no cover
        x_ptr,
        smooth_ptr,
        scale_ptr,
        m_size,
        k_size,
        batch,
        HAS_SMOOTH: tl.constexpr,  # noqa: N803  # triton constexpr naming
        DTYPE_MAX: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_M: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_K: tl.constexpr,  # noqa: N803  # triton constexpr naming
        M_GROUP: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        """Transposed mode pass 1: partial column max reduced by atomic_max.

        Tasks are (batch, k-block, m-group) triples flattened onto a 1D grid;
        each task locally reduces M_GROUP tiles and commits one atomic_max.
        Non-negative fp32 shares the ordering of its int32 bit pattern, so the
        max is exact and deterministic.

        The max is divided by DTYPE_MAX inside pass 1 and accumulated straight
        into the fp32 ``scale`` output buffer (passed here as its int32 bit
        view, pre-zeroed): IEEE division by a positive constant is monotonic, so
        the max of the per-task quotients equals max/DYPE_MAX exactly. Pass 2
        then only reads the finished scale, so the operator needs neither a
        separate atomic workspace tensor nor any in-place rewrite of it.

        Tiles are m-major (``[BLOCK_M, BLOCK_K]`` with the contiguous axis on
        tile axis 1): the NPU vectorizes loads along the last tile axis, and
        this layout measured ~2.4x the read bandwidth of the k-major variant.
        """
        pid = tl.program_id(0).to(tl.int32)
        num_k_blocks = tl.cdiv(k_size, BLOCK_K)
        num_m_groups = tl.cdiv(m_size, M_GROUP * BLOCK_M)
        total = batch * num_k_blocks * num_m_groups
        for t in range(pid, total, tl.num_programs(0).to(tl.int32)):
            b = t // (num_k_blocks * num_m_groups)
            rem = t - b * (num_k_blocks * num_m_groups)
            kb = rem // num_m_groups
            mg = rem - kb * num_m_groups

            k_off = (kb * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int32)
            k_mask = k_off < k_size
            local = tl.zeros((BLOCK_K,), dtype=tl.float32)
            base_m = mg * (M_GROUP * BLOCK_M)
            for i in range(M_GROUP):
                m_off = (base_m + i * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
                m_mask = m_off < m_size
                mask2d = m_mask[:, None] & k_mask[None, :]
                # m-major tile: the contiguous axis (k) stays on tile axis 1.
                offs = b * (m_size * k_size) + m_off[:, None] * k_size + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    # smooth is 1-D along the quant axis (m_size): load per row
                    # block, broadcast as [BLOCK_M, 1]. Boundary rows use 0.0.
                    smooth_vec = tl.load(smooth_ptr + m_off, mask=m_mask, other=0.0)
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[:, None]
                else:
                    inp = x_tile.to(tl.float32)
                local = tl.maximum(local, tl.max(tl.abs(inp), axis=0))
            bits = (local / DTYPE_MAX).to(tl.int32, bitcast=True)
            tl.atomic_max(scale_ptr + b * k_size + k_off, bits, mask=k_mask)

    @triton.jit
    def _dynamic_quant_quant_same_kernel(  # pragma: no cover
        x_ptr,
        smooth_ptr,
        y_ptr,
        scale_ptr,
        m_size,
        k_size,
        batch,
        DTYPE_MAX: tl.constexpr,  # noqa: N803  # triton constexpr naming
        HAS_SMOOTH: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_M: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_K: tl.constexpr,  # noqa: N803  # triton constexpr naming
        M_GROUP: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        """Middle-axis pass 2: quantize and store y in the SAME layout as x.

        ``scale_ptr`` already holds the finished fp32 scale written by pass 1, so
        this pass only reads it — no cross-task write race on the scale buffer.

        Both the load and the store use m-major tiles (the contiguous axis k on
        tile axis 1), so neither side needs a transposed address pattern.
        """
        pid = tl.program_id(0).to(tl.int32)
        num_k_blocks = tl.cdiv(k_size, BLOCK_K)
        num_m_groups = tl.cdiv(m_size, M_GROUP * BLOCK_M)
        total = batch * num_k_blocks * num_m_groups
        for t in range(pid, total, tl.num_programs(0).to(tl.int32)):
            b = t // (num_k_blocks * num_m_groups)
            rem = t - b * (num_k_blocks * num_m_groups)
            kb = rem // num_m_groups
            mg = rem - kb * num_m_groups

            k_off = (kb * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int32)
            k_mask = k_off < k_size
            scale = tl.load(scale_ptr + b * k_size + k_off, mask=k_mask, other=0.0)
            denom = tl.where(scale > 0.0, scale, 1.0)

            base_m = mg * (M_GROUP * BLOCK_M)
            for i in range(M_GROUP):
                m_off = (base_m + i * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
                m_mask = m_off < m_size
                mask2d = m_mask[:, None] & k_mask[None, :]
                # m-major tile: load and store share the same contiguous-last
                # address pattern (y keeps the x layout).
                offs = b * (m_size * k_size) + m_off[:, None] * k_size + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    # Same fp32 product path as pass 1; smooth along the quant axis.
                    smooth_vec = tl.load(smooth_ptr + m_off, mask=m_mask, other=0.0)
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[:, None]
                else:
                    inp = x_tile.to(tl.float32)
                q = inp / denom[None, :]  # true divide, same as aclnn
                q = tl.minimum(tl.maximum(q, -DTYPE_MAX), DTYPE_MAX)
                # Single RN cast to fp8, aligned with aclnnDynamicQuantV2.
                y8 = q.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y8, mask=mask2d)

    @triton.jit
    def _dynamic_quant_quant_single_kernel(  # pragma: no cover
        x_ptr,
        smooth_ptr,
        y_ptr,
        scale_ptr,
        m_size,
        k_size,
        batch,
        DTYPE_MAX: tl.constexpr,  # noqa: N803  # triton constexpr naming
        HAS_SMOOTH: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_M: tl.constexpr,  # noqa: N803  # triton constexpr naming
        BLOCK_K: tl.constexpr,  # noqa: N803  # triton constexpr naming
    ):
        """Middle-axis single-kernel tier: no workspace, one launch.

        Each program owns (batch, k-block) and sweeps the quant axis twice —
        first for the group max, then to quantize — so no cross-program
        reduction (atomic workspace) is needed. Only viable when the sweep is
        short (moderate m_size); the host gates this tier accordingly. Tile
        layout matches pass 1/2: m-major, contiguous axis on tile axis 1.
        """
        pid = tl.program_id(0).to(tl.int32)
        num_k_blocks = tl.cdiv(k_size, BLOCK_K)
        total = batch * num_k_blocks
        for t in range(pid, total, tl.num_programs(0).to(tl.int32)):
            b = t // num_k_blocks
            kb = t - b * num_k_blocks
            k_off = (kb * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int32)
            k_mask = k_off < k_size
            local = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for mb in range(0, tl.cdiv(m_size, BLOCK_M)):
                m_off = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
                m_mask = m_off < m_size
                mask2d = m_mask[:, None] & k_mask[None, :]
                offs = b * (m_size * k_size) + m_off[:, None] * k_size + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    smooth_vec = tl.load(smooth_ptr + m_off, mask=m_mask, other=0.0)
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[:, None]
                else:
                    inp = x_tile.to(tl.float32)
                local = tl.maximum(local, tl.max(tl.abs(inp), axis=0))
            scale = local / DTYPE_MAX
            tl.store(scale_ptr + b * k_size + k_off, scale, mask=k_mask)
            denom = tl.where(scale > 0.0, scale, 1.0)
            for mb in range(0, tl.cdiv(m_size, BLOCK_M)):
                m_off = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
                m_mask = m_off < m_size
                mask2d = m_mask[:, None] & k_mask[None, :]
                offs = b * (m_size * k_size) + m_off[:, None] * k_size + k_off[None, :]
                x_tile = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
                if HAS_SMOOTH:
                    smooth_vec = tl.load(smooth_ptr + m_off, mask=m_mask, other=0.0)
                    inp = x_tile.to(tl.float32) * smooth_vec.to(tl.float32)[:, None]
                else:
                    inp = x_tile.to(tl.float32)
                q = inp / denom[None, :]  # true divide, same as aclnn
                q = tl.minimum(tl.maximum(q, -DTYPE_MAX), DTYPE_MAX)
                # Single RN cast to fp8, aligned with aclnnDynamicQuantV2.
                y8 = q.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y8, mask=mask2d)


def _validate_args(x: torch.Tensor, smooth_scales: torch.Tensor | None, axis: int):
    if _FP8_DTYPE is None:
        raise RuntimeError("transpose_dynamic_quant requires torch with float8_e4m3fn support.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Input tensor dtype must be float16 or bfloat16, got {x.dtype}.")
    if x.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dims, got {x.dim()}.")
    if x.numel() >= _MAX_INT32:
        raise ValueError(f"Input tensor has {x.numel()} elements, exceeding the int32 index limit.")

    ndim = x.dim()
    a = axis if axis >= 0 else ndim + axis
    if not (0 <= a < ndim):
        raise ValueError(f"axis={axis} is out of range for ndim={ndim}.")
    scale_shape = tuple(x.shape[:a]) + tuple(x.shape[a + 1 :])
    if smooth_scales is not None:
        if smooth_scales.device != x.device:
            raise ValueError(f"smooth_scales device {smooth_scales.device} must match x device {x.device}.")
        if smooth_scales.dtype != x.dtype:
            raise ValueError(f"smooth_scales dtype {smooth_scales.dtype} must match x dtype {x.dtype}.")
        if smooth_scales.dim() != 1 or smooth_scales.shape[0] != x.shape[a]:
            raise ValueError(
                f"smooth_scales must be 1-D with length equal to the quant axis length "
                f"{x.shape[a]} (axis={axis}, aligned with the official "
                f"aclnnDynamicQuantV2 smoothScalesOptional), "
                f"got shape {tuple(smooth_scales.shape)}."
            )
    return a, scale_shape


def _transpose_dynamic_quant_triton(
    x: torch.Tensor, smooth_scales: torch.Tensor | None, axis: int
) -> tuple[torch.Tensor, torch.Tensor]:
    a, scale_shape = _validate_args(x, smooth_scales, axis)
    x = x.contiguous()
    if smooth_scales is not None:
        smooth_scales = smooth_scales.contiguous()
    num_cores = get_vectorcore_num()
    has_smooth = smooth_scales is not None

    y = torch.empty(x.shape, device=x.device, dtype=_FP8_DTYPE)
    if x.numel() == 0:
        # Empty quant axis: max is undefined; deterministic zero scale per group.
        return y, torch.zeros(scale_shape, device=x.device, dtype=torch.float32)
    scale = torch.empty(scale_shape, device=x.device, dtype=torch.float32)
    # smooth=None is specialized away by Triton, so the no-smooth variant carries
    # no placeholder tensor in the kernel signature (profiling shows only real
    # operands instead of a duplicate view of x).
    smooth_flat = smooth_scales.view(-1) if has_smooth else None

    if a == 0:
        m_size, k_size = x.shape[0], x.numel() // x.shape[0]
        thin = _pick_thin_axis_blocks(m_size)
        block_m, block_k = thin if thin is not None else _pick_axis_blocks(m_size, k_size)
        grid = (min(triton.cdiv(k_size, block_k), num_cores),)
        _dynamic_quant_axis0_kernel[grid](  # pylint: disable=possibly-used-before-assignment
            x,
            smooth_flat,
            y,
            scale,
            m_size,
            k_size,
            HAS_SMOOTH=has_smooth,
            DTYPE_MAX=_DTYPE_MAX,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
        )
        return y, scale

    if a == x.dim() - 1:
        k_size = x.shape[-1]
        rows = x.numel() // k_size
        block_m, block_k = _pick_axis_blocks(rows, k_size)
        block_col = triton.next_power_of_2(k_size)
        if (
            rows * k_size >= _LARGE_SHAPE_ELEMS
            and block_col <= _FULL_LOAD_MAX_BLOCK_COL
            and triton.cdiv(rows, block_m) < num_cores
        ):
            # Full-load tier: the chunked kernel would leave cores idle and
            # read x twice; whole-row residency fixes both.
            grid = (min(triton.cdiv(rows, _FULL_LOAD_BLOCK_ROW), num_cores),)
            _dynamic_quant_axis_last_fullload_kernel[grid](  # pylint: disable=possibly-used-before-assignment
                x,
                smooth_flat,
                y,
                scale,
                rows,
                k_size,
                HAS_SMOOTH=has_smooth,
                DTYPE_MAX=_DTYPE_MAX,
                BLOCK_ROW=_FULL_LOAD_BLOCK_ROW,
                BLOCK_COL=block_col,
            )
            return y, scale
        grid = (min(triton.cdiv(rows, block_m), num_cores),)
        _dynamic_quant_axis_last_kernel[grid](  # pylint: disable=possibly-used-before-assignment
            x,
            smooth_flat,
            y,
            scale,
            rows,
            k_size,
            HAS_SMOOTH=has_smooth,
            DTYPE_MAX=_DTYPE_MAX,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
        )
        return y, scale

    # Middle axis: view x as (batch, m_size, k_size); the scale is per (batch,
    # k) column max, which no single kernel can finish before quantizing.
    batch = math.prod(x.shape[:a])
    m_size = x.shape[a]
    k_size = math.prod(x.shape[a + 1 :])
    if _SINGLE_KERNEL_MIN_M <= m_size <= _SINGLE_KERNEL_MAX_M:
        # Single-kernel tier: each program sweeps the (moderate) quant axis
        # twice — no atomic workspace, one launch; fastest while the sweep is
        # short. Grid = (batch, k-blocks); BLOCK_M shrinks to the axis when
        # the axis is shorter than 128.
        block_m = min(128, triton.next_power_of_2(m_size))
        grid = (min(batch * triton.cdiv(k_size, _SINGLE_KERNEL_BLOCK_K), num_cores),)
        _dynamic_quant_quant_single_kernel[grid](  # pylint: disable=possibly-used-before-assignment
            x,
            smooth_flat,
            y,
            scale,
            m_size,
            k_size,
            batch,
            DTYPE_MAX=_DTYPE_MAX,
            HAS_SMOOTH=has_smooth,
            BLOCK_M=block_m,
            BLOCK_K=_SINGLE_KERNEL_BLOCK_K,
        )
        return y, scale
    # Two-kernel tier for long quant axes: partial max via atomic_max, then
    # quantize with the full scale and store y in the SAME layout as x.
    thin = _pick_thin_axis_blocks(m_size)
    block_m = thin[0] if thin is not None else _TRANS_P1_BLOCK_M
    block_k = thin[1] if thin is not None else _TRANS_P1_BLOCK_K
    num_k_blocks = triton.cdiv(k_size, block_k)
    num_m_groups = triton.cdiv(m_size, _TRANS_P1_M_GROUP * block_m)
    grid = (min(batch * num_k_blocks * num_m_groups, num_cores),)
    # Zeroed fp32 scale == int32 bit pattern 0, the neutral element of atomic_max.
    # Pass 1 folds the /DTYPE_MAX into the atomic value, so the buffer ends up
    # holding the final scale and pass 2 only reads it.
    scale.zero_()
    _dynamic_quant_max_partial_kernel[grid](  # pylint: disable=possibly-used-before-assignment
        x,
        smooth_flat,
        scale.view(torch.int32),
        m_size,
        k_size,
        batch,
        HAS_SMOOTH=has_smooth,
        DTYPE_MAX=_DTYPE_MAX,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        M_GROUP=_TRANS_P1_M_GROUP,
    )
    _dynamic_quant_quant_same_kernel[grid](  # pylint: disable=possibly-used-before-assignment
        x,
        smooth_flat,
        y,
        scale,
        m_size,
        k_size,
        batch,
        DTYPE_MAX=_DTYPE_MAX,
        HAS_SMOOTH=has_smooth,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        M_GROUP=_TRANS_P1_M_GROUP,
    )
    return y, scale


@torch.library.custom_op("mindiesd::transpose_dynamic_quant", mutates_args=())
def transpose_dynamic_quant(
    x: torch.Tensor, smooth_scales: torch.Tensor | None, axis: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Transpose + DynamicQuantV2 + Transpose with fp8_e4m3fn output.

    Quantizes ``x`` along ``axis`` with per-group symmetric dynamic quant:
    ``scale = max(|x * smooth|) / 448`` and ``y = fp8_rn(x * smooth / scale)``
    (fp32 product, single RN cast), bit-aligned with ``aclnnDynamicQuantV2``.
    y always has the SAME layout as x (the quantized axis is written back in
    place, equivalent to transpose-quantize-transpose-back with both transposes
    folded into index arithmetic). Requires Triton-Ascend; no torch fallback.

    Args:
        x: Input tensor (float16 or bfloat16), ND, contiguous-friendly.
        smooth_scales: Optional smooth scales. Must be 1-D with length equal to
            the quant axis length (x.shape[axis]) and dtype matching ``x``;
            multiplied element-wise along the quant axis, aligned with the
            official ``aclnnDynamicQuantV2`` ``smoothScalesOptional`` semantics.
        axis: Quantization axis (the reduction axis for max); any axis.

    Returns:
        (y, scale): y in float8_e4m3fn with the same shape and layout as x;
        scale in float32 with the shape of x without ``axis``.

    Raises:
        RuntimeError: If Triton-Ascend is not available in this environment.
        ValueError: If arguments are invalid or ``x`` is not on an Ascend NPU.
    """
    if not _TRITON_ON_ASCEND:
        raise RuntimeError("transpose_dynamic_quant requires Triton-Ascend")
    if x.device.type != "npu":
        raise ValueError("transpose_dynamic_quant requires Ascend NPU inputs")
    return _transpose_dynamic_quant_triton(x, smooth_scales, axis)


@transpose_dynamic_quant.register_fake
def _(x, smooth_scales, axis):
    ndim = x.dim()
    a = axis if axis >= 0 else ndim + axis
    scale_shape = tuple(x.shape[:a]) + tuple(x.shape[a + 1 :])
    return (
        torch.empty(x.shape, dtype=_FP8_DTYPE, device=x.device),
        torch.empty(scale_shape, dtype=torch.float32, device=x.device),
    )
