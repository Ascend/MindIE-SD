#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# pylint: disable=too-many-lines

"""Fused RMSNorm, indexed AdaLN affine and DynamicMxQuant.

The operator fuses the three per-block steps of an AdaLN diffusion transformer
into one call:

1. RMSNorm over the hidden dimension (fp32 accumulation);
2. indexed AdaLN affine, ``out = x * (1 + scale[index]) + shift[index]``;
3. per-32-element MX quantization to FP8 (E4M3/E5M2) or FP4 (E2M1/E1M2).

Input dtype: ``bfloat16`` or ``float16``. Output dtype: ``float8_e4m3fn``,
``float8_e5m2``, ``float4_e2m1fn_x2`` or ``float4_e1m2fn_x2``. All eight
combinations return ``(y, mxscale)``; MXFP4 output requires an even hidden size.
The public wrapper owns contract validation, alias resolution and contiguity
normalization so each private launch path can stay focused on kernel selection.

On the single-kernel path, the rounded modulation result stays in registers and
is consumed directly by quantization. On the two-stage path, it is written to a
temporary HBM tensor and read by ``npu_dynamic_mx_quant``. Both paths return only
``y`` and ``mxscale``.

Because the activation is not an output, a multi-row program tile is used for
tiny row blocks (``block_n <= 64``). Those tiles are laid out so that each row's
32-element groups stay in their original order, so ``(y, mxscale)`` remains
byte-exact against the reference chain. Row blocks wider than a tiny block keep
one row per program: packing rows there would change the fp32 reduction order
and move a few ``y`` bytes by one FP8/FP4 code, which this operator does not
allow.

Wide rows keep the affine in Triton but hand the MX tail to CANN. The current
Triton-Ascend version does not expose the underlying DynamicMxQuant instruction;
in addition, the 32-element group amax and FP4 packing path costs more than the
native operator on Ascend950 for these shapes. The hybrid path writes the
activation to HBM in the input dtype before the native quantizer reads it, so
``(y, mxscale)`` stays byte-exact. The path is selected by shape alone
(``rows >= 1024`` and either ``hidden >= 3000`` or a padded narrow block with
``rows * hidden >= 24 Mi``); there is no switch that turns it off.
"""

from __future__ import annotations

import math
import weakref
from numbers import Real

import torch
import torch_npu  # noqa: F401  # registers torch.npu for triton_utils
from ..utils import ParametersInvalid
from .triton_utils import _HAS_TRITON, _TRITON_ON_ASCEND

__all__ = ["rms_norm_indexed_scale_shift_dynamic_mx_quant"]

_MAX_GRID = 65535
# Tiny block width (hidden padded to at most this many columns) and the row
# count from which packing several rows into one program starts to pay off.
_TINY_BLOCK_N = 64
_TINY_ROWS = 64
# Wide-row MX tail: the in-register 32-element group amax (and the FP4 packing)
# of the fused kernel is several times slower than CANN's DynamicMxQuant on
# Ascend950, so for large launches the fused operator keeps the (32% faster)
# Triton affine and hands the tail to CANN.  The round trip through the
# activation costs one `rows * hidden` store plus one read, exactly what the
# two-segment chain pays, so it only pays off from this row count / hidden size
# on (measurements in the design doc); below it the single kernel stays.
_HYBRID_MIN_ROWS = 1024
_HYBRID_MIN_HIDDEN = 3000
# Narrow rows (512 <= hidden < 3000) only take the two-segment path when the
# fused block is padded AND the launch is this large: the masks of a padded
# full-row block are what make the single kernel expensive there.
_HYBRID_MIN_ELEMENTS = 24 << 20
# Below this width the fused multi-row kernel is 3.9x - 5.3x faster than the
# two-segment chain, so tiny rows never use the CANN quantizer.
_HYBRID_MIN_HIDDEN_ANY = 512
# Canonical output types after alias resolution. The two FP8 names are already
# the public spellings, so only the packed FP4 variants need aliases below.
_SUPPORTED_DST_TYPES = {
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1fn_x2",
    "float4_e1m2fn_x2",
}
# Common FP4 spellings that omit the packed-storage ``_x2`` suffix.
_DST_TYPE_ALIASES = {
    "float4_e2m1": "float4_e2m1fn_x2",
    "float4_e2m1fn": "float4_e2m1fn_x2",
    "float4_e1m2": "float4_e1m2fn_x2",
    "float4_e1m2fn": "float4_e1m2fn_x2",
}
# The torch dtypes of the CANN quantizer, one per output type.
_NPU_DST_TYPES = {
    "float8_e4m3fn": torch_npu.float8_e4m3fn,
    "float8_e5m2": torch_npu.float8_e5m2,
    "float4_e2m1fn_x2": torch_npu.float4_e2m1fn_x2,
    "float4_e1m2fn_x2": torch_npu.float4_e1m2fn_x2,
}


def _use_hybrid(rows: int, hidden: int) -> bool:
    """Shape-only selection of the two-segment MX tail.

    Args:
        rows: number of activation rows.
        hidden: row length.

    Returns:
        ``True`` when the Triton affine kernel should be followed by
        ``npu_dynamic_mx_quant``. This Triton-Ascend version does not expose
        the native DynamicMxQuant instruction, and the Triton MX tail is slower
        than CANN at these launch sizes.

    The two-segment path writes the activation once and reads it back once, which
    the fused kernel avoids; it wins only when the per-row work is large enough
    to amortize that traffic and the extra launch.
    """
    if rows < _HYBRID_MIN_ROWS or hidden < _HYBRID_MIN_HIDDEN_ANY:
        return False
    if hidden >= _HYBRID_MIN_HIDDEN:
        return True
    # Narrow rows: only a padded fused block (masked loads on every element) is
    # slow enough for the two-segment path to win, and only at this launch size.
    block_n = (hidden + 255) // 256 * 256
    return block_n != hidden and rows * hidden >= _HYBRID_MIN_ELEMENTS


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
    dst_type: str,
) -> None:
    """Check the public operator contract.

    Args:
        x: activation, ``[rows, hidden]``.
        weight: RMSNorm weight, ``[hidden]``.
        shift: indexed AdaLN shift, ``[conditions, hidden]``.
        scale: indexed AdaLN scale, ``[conditions, hidden]``.
        indices: per-row condition index, ``[rows]``.
        eps: RMSNorm epsilon, must be finite and positive.
        dst_type: MX output type after alias resolution.

    Raises:
        ParametersInvalid: If any input violates the documented constraints.
    """
    for name, value in (
        ("x", x),
        ("weight", weight),
        ("shift", shift),
        ("scale", scale),
        ("indices", indices),
    ):
        if not isinstance(value, torch.Tensor):
            raise ParametersInvalid(f"The data type of input {name} must be torch.Tensor, but got {type(value)}.")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ParametersInvalid(f"The input eps must be a finite positive number, but got {eps}.")
    if x.dim() != 2:
        raise ParametersInvalid(f"The dimensional of input x must be a 2, but got {x.dim()}.")
    rows, hidden = x.shape
    if weight.shape != (hidden,):
        raise ParametersInvalid(f"The shape of weight must be [{hidden}], but got {tuple(weight.shape)}.")
    if shift.dim() != 2 or scale.shape != shift.shape:
        raise ParametersInvalid(
            "The shape of shift and scale must be the same 2-D [conditions, hidden], but got "
            f"{tuple(shift.shape)} and {tuple(scale.shape)}."
        )
    if shift.shape[1] != hidden:
        raise ParametersInvalid(f"The hidden size of shift and scale must be {hidden}, but got {shift.shape[1]}.")
    if shift.shape[0] == 0:
        raise ParametersInvalid("shift and scale must contain at least one condition.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ParametersInvalid(f"The dtype of x must be bfloat16 or float16, but got {x.dtype}.")
    if any(t.dtype != x.dtype for t in (weight, shift, scale)):
        raise ParametersInvalid(
            "The dtype of x, weight, shift and scale must be the same, but got "
            f"{[str(t.dtype) for t in (x, weight, shift, scale)]}."
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise ParametersInvalid(f"The dtype of indices must be int32 or int64, but got {indices.dtype}.")
    if indices.shape != (rows,):
        raise ParametersInvalid(f"indices must have shape [{rows}], but got {tuple(indices.shape)}.")
    if len({x.device, weight.device, shift.device, scale.device, indices.device}) != 1:
        raise ParametersInvalid("The device of all inputs must be the same.")
    if dst_type.lower() not in _SUPPORTED_DST_TYPES:
        raise ParametersInvalid(
            f"Unsupported dst_type {dst_type}, the supported dst_type must be one of {sorted(_SUPPORTED_DST_TYPES)}."
        )
    if dst_type.lower().startswith("float4") and hidden % 2:
        raise ParametersInvalid(f"MXFP4 output requires an even hidden size, but got {hidden}.")
    if hidden == 0:
        raise ParametersInvalid("The hidden size of x must be positive.")


def _mxscale_shape(rows: int, hidden: int) -> tuple[int, int, int]:
    n_groups = (hidden + 31) // 32
    n_pad = (n_groups + 1) // 2 * 2
    return rows, n_pad // 2, 2


def _e4m3_rows_per_prog(block_n: int, hidden: int, rows: int) -> int:
    """Rows packed into one E4M3 program, or 0 for one row per program.

    Args:
        block_n: column block of the kernel (hidden padded to whole MX groups).
        hidden: row length; a padded tiny block stays full-row (see below).
        rows: number of rows in the launch.

    Returns:
        Rows per program for the multi-row kernel, or ``0`` to keep the
        full-row kernel.

    Every value returned here keeps ``(y, mxscale)`` byte-exact against the 2-D
    reference: the tiny-block tiles reduce the RMS sum over a 2-D tile whose
    lanes are whole MX groups of the same row, so the fp32 accumulation order
    used by the reference is reproduced lane by lane.  Row blocks wider than a
    tiny block never pack rows: a wider 2-D tile would move a row's block amax
    across a power-of-two boundary, changing an E8M0 group scale by a whole step.
    """
    if block_n <= _TINY_BLOCK_N and rows >= _TINY_ROWS:
        if block_n <= 32 and rows < 128:
            return 0
        if rows < 4096:
            return 16
        if block_n == hidden:
            # Long batches: 256/128 rows per program on an unpadded 32/64-wide
            # block are byte-exact against the reference reduction.
            return 256 if block_n <= 32 else 128
        # A padded tiny block (H = 33..63) keeps one row per program: the long
        # 2-D tiles move a row's block amax across a power-of-two boundary,
        # which changes one E8M0 group scale by a whole step.
        return 0
    return 0


if _HAS_TRITON:
    from .triton_utils import tl, triton

    @triton.jit
    def _safe_scalar_index(index, n_conds: tl.constexpr):
        """Clamp a condition index before it is used to address shift/scale."""
        return tl.minimum(tl.maximum(index, 0), n_conds - 1)

    @triton.jit
    def _safe_index_vector(index, n_conds: tl.constexpr):
        """Clamp a vector of condition indices before address calculation."""
        return tl.minimum(tl.maximum(index, 0), n_conds - 1)

    @triton.jit
    def _validate_indices_kernel(
        indices_ptr,
        rows,
        n_conds: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < rows
        index = tl.load(indices_ptr + offsets, mask=mask, other=0)
        invalid = mask & ((index < 0) | (index >= n_conds))
        tl.device_assert(
            tl.sum(invalid.to(tl.int32), axis=0) == 0,
            "indices must satisfy 0 <= index < conditions",
        )

    _VALIDATED_INDICES = {}
    _INDEX_VALIDATION_BLOCK = 1024

    def _validate_indices(indices: torch.Tensor, n_conds: int) -> None:
        """Validate indices once per tensor storage/version without host sync."""
        rows = indices.numel()
        if rows == 0:
            return
        key = (
            indices.device.type,
            indices.device.index,
            indices.data_ptr(),
            indices._version,
            n_conds,
            rows,
            indices.dtype,
        )
        cached = _VALIDATED_INDICES.get(key)
        if cached is not None and cached() is not None:
            return

        _validate_indices_kernel[(triton.cdiv(rows, _INDEX_VALIDATION_BLOCK),)](
            indices,
            rows,
            n_conds,
            _INDEX_VALIDATION_BLOCK,
            debug=True,
        )

        def remove(ref):
            if _VALIDATED_INDICES.get(key) is ref:
                _VALIDATED_INDICES.pop(key, None)

        _VALIDATED_INDICES[key] = weakref.ref(indices, remove)

    # BF16 -> E4M3 specialized path.
    @triton.jit
    def _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel(
        out_y_ptr,
        out_scale_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        row_offset,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
        scale_stride: tl.constexpr,
        n_conds: tl.constexpr,
    ):
        """One program handles one row: RMSNorm -> indexed affine -> MXFP8 quant.

        ``block_n`` is ``ceil(hidden_size / 32) * 32``: the smallest vector block
        that contains the whole row and all complete 32-element MX groups.  Keeping
        the full row resident lets the variance reduction and the affine/quant work
        reuse the same ``x`` register tile.
        """
        row = (tl.program_id(0) + row_offset).to(tl.int64)
        cols = tl.arange(0, block_n)
        mask = cols < hidden_size
        index = _safe_scalar_index(tl.load(indices_ptr + row), n_conds)

        # ---- RMSNorm + indexed scale/shift, fp32 until the final bf16 rounding ----
        if block_n == hidden_size:
            x = tl.load(x_ptr + row * hidden_size + cols).to(tl.float32)
            weight = tl.load(weight_ptr + cols).to(tl.float32)
        else:
            x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / hidden_size
        normalized = x * tl.rsqrt(variance + eps) * weight
        if block_n == hidden_size:
            shift = tl.load(shift_ptr + index * hidden_size + cols).to(tl.float32)
            scale = tl.load(scale_ptr + index * hidden_size + cols).to(tl.float32)
        else:
            shift = tl.load(shift_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
        output = normalized * (1.0 + scale) + shift
        # The reference kernel stores fp32 into a bf16 tensor here; DynamicMxQuant
        # then consumes that bf16 tensor, so the rounding must happen before the
        # block amax and the quantization.
        mod = output.to(tl.bfloat16)

        # ---- DynamicMxQuant (e4m3fn, block_size = 32, round_mode = rint) ----
        # bf16 exponent bits: value = 2 ** (exp_bits - 127).  The shared MX
        # exponent byte is exp_bits(max|x|) - 8 (e4m3fn emax = 8); values whose
        # exponent is <= 8 take the all-zero-block path of the CANN operator
        # (scale byte 0 and zero elements).
        groups: tl.constexpr = block_n // 32
        # Reshape once: the group amax and the scaled product both read the same
        # (groups, 32) view, so a single layout conversion covers both.
        mod_groups = tl.reshape(mod, (groups, 32))
        bits = mod_groups.to(tl.uint16, bitcast=True)
        absbits = bits & 0x7FFF
        maxbits = tl.max(absbits, axis=1)
        exp = (maxbits >> 7).to(tl.int32)
        byte = tl.minimum(tl.maximum(exp - 8, 0), 255)
        # Non-finite block (Inf/NaN): DynamicMxQuant emits the e8m0 NaN scale
        # (0xff), so flag it before building the reciprocal.
        special = maxbits >= 0x7F80
        byte = tl.where(special, 255, byte)
        # reciprocal scale 2 ** (127 - byte), as a bf16 bit pattern
        mult_bits = ((254 - byte) << 7).to(tl.uint16)
        mult = tl.reshape(mult_bits, (groups, 1)).to(tl.bfloat16, bitcast=True)
        byte_col = tl.reshape(byte, (groups, 1))
        mult = tl.where(byte_col > 0, mult, 0.0)
        mult = tl.broadcast_to(mult, (groups, 32))
        yv = mod_groups * mult
        yv = tl.where(tl.broadcast_to(byte_col == 255, (groups, 32)), 0.0, yv)

        # Saturate to +-448 in the bf16 bit domain before the fp8 conversion.
        # Triton's fp32 -> e4m3fn conversion encodes an overflowing negative value
        # as +0x7f, while the CANN operator saturates to +-448 (0x7e / 0xfe), so
        # both the magnitude clamp and the sign have to be handled here.
        yb16 = yv.to(tl.uint16, bitcast=True)
        ylim = tl.full(yb16.shape, 0x43E0, tl.int16)
        yb16 = tl.where((yb16 & 0x7FFF) > ylim, (yb16 & 0x8000) | 0x43E0, yb16)
        yv = yb16.to(tl.bfloat16, bitcast=True)
        y8 = yv.to(tl.float32).to(tl.float8e4nv)
        if block_n == hidden_size:
            tl.store(out_y_ptr + row * hidden_size + cols, tl.reshape(y8, (block_n,)))
        else:
            tl.store(
                out_y_ptr + row * hidden_size + cols,
                tl.reshape(y8, (block_n,)),
                mask=mask,
            )

        gid = tl.arange(0, groups)
        n_groups: tl.constexpr = (hidden_size + 31) // 32
        tl.store(
            out_scale_ptr + row * scale_stride + gid,
            byte.to(tl.uint8),
            mask=gid < n_groups,
        )

    @triton.jit
    def _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel_tiled(
        out_y_ptr,
        out_scale_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        row_offset,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        tile_n: tl.constexpr,
        var_block: tl.constexpr,
        chunked_var: tl.constexpr,
        scale_stride: tl.constexpr,
        n_conds: tl.constexpr,
    ):
        """Large-hidden variant of the fused kernel.

        The row is processed in ``tile_n``-element column tiles.  Pass 1 accumulates
        the RMS sum of squares, pass 2 recomputes the affine result tile by tile and
        quantizes it in the same 32-element groups.  ``tile_n`` is a multiple of 32
        so every tile starts on an MX group boundary; keeping the tile at 4096
        elements holds the fp32 working set well inside the 248 KB UB while still
        covering 8 KB / 16 KB contiguous DMA transfers.
        """
        row = (tl.program_id(0) + row_offset).to(tl.int64)
        index = _safe_scalar_index(tl.load(indices_ptr + row), n_conds)
        # Pass 1: for H <= 32768 the variance uses the same power-of-two block as
        # the reference Triton RMSNorm kernel, so the fp32 reduction order (and
        # therefore the bf16 result) is identical.  Beyond that the full block no
        # longer fits UB, so the sum falls back to a tile-wise accumulation.
        if chunked_var:
            acc = 0.0
            for off in range(0, hidden_size, tile_n):
                cols = off + tl.arange(0, tile_n)
                mask = cols < hidden_size
                x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
                acc += tl.sum(x * x, axis=0)
            variance = acc / hidden_size
        else:
            vcols = tl.arange(0, var_block)
            if var_block == hidden_size:
                xv = tl.load(x_ptr + row * hidden_size + vcols).to(tl.float32)
            else:
                vmask = vcols < hidden_size
                xv = tl.load(x_ptr + row * hidden_size + vcols, mask=vmask, other=0.0).to(tl.float32)
            variance = tl.sum(xv * xv, axis=0) / hidden_size
        rstd = tl.rsqrt(variance + eps)

        n_groups: tl.constexpr = (hidden_size + 31) // 32
        groups: tl.constexpr = tile_n // 32
        full_tile: tl.constexpr = (hidden_size % tile_n) == 0
        gid0 = tl.arange(0, groups)
        for off in range(0, hidden_size, tile_n):
            cols = off + tl.arange(0, tile_n)
            if full_tile:
                x = tl.load(x_ptr + row * hidden_size + cols).to(tl.float32)
                weight = tl.load(weight_ptr + cols).to(tl.float32)
                shift = tl.load(shift_ptr + index * hidden_size + cols).to(tl.float32)
                scale = tl.load(scale_ptr + index * hidden_size + cols).to(tl.float32)
            else:
                mask = cols < hidden_size
                x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
                weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                shift = tl.load(shift_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
                scale = tl.load(scale_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
            mod = (x * rstd * weight * (1.0 + scale) + shift).to(tl.bfloat16)

            bits = mod.to(tl.uint16, bitcast=True)
            absbits = bits & 0x7FFF
            maxbits = tl.max(tl.reshape(absbits, (groups, 32)), axis=1)
            exp = (maxbits >> 7).to(tl.int32)
            byte = tl.minimum(tl.maximum(exp - 8, 0), 255)
            special = maxbits >= 0x7F80
            byte = tl.where(special, 255, byte)
            mult_bits = ((254 - byte) << 7).to(tl.uint16)
            mult = tl.reshape(mult_bits, (groups, 1)).to(tl.bfloat16, bitcast=True)
            byte_col = tl.reshape(byte, (groups, 1))
            mult = tl.where(byte_col > 0, mult, 0.0)
            mult = tl.broadcast_to(mult, (groups, 32))
            # The MX scale is a power of two, so the bf16 product is exact; keep it
            # in bf16 (the reference quantizer rounds to bf16 before encoding).
            yv = (tl.reshape(mod, (groups, 32)) * mult).to(tl.bfloat16)
            yb16 = yv.to(tl.uint16, bitcast=True)
            ylim = tl.full(yb16.shape, 0x43E0, tl.int16)
            yb16 = tl.where((yb16 & 0x7FFF) > ylim, (yb16 & 0x8000) | 0x43E0, yb16)
            yv = yb16.to(tl.bfloat16, bitcast=True)
            y8 = yv.to(tl.float32).to(tl.float8e4nv)
            y8 = tl.where(tl.reshape(byte == 255, (groups, 1)), 0, y8.to(tl.uint8, bitcast=True)).to(
                tl.float8e4nv, bitcast=True
            )
            if full_tile:
                tl.store(out_y_ptr + row * hidden_size + cols, tl.reshape(y8, (tile_n,)))
            else:
                tl.store(out_y_ptr + row * hidden_size + cols, tl.reshape(y8, (tile_n,)), mask=mask)
            gid = off // 32 + gid0
            if full_tile:
                tl.store(out_scale_ptr + row * scale_stride + gid, byte.to(tl.uint8))
            else:
                tl.store(
                    out_scale_ptr + row * scale_stride + gid,
                    byte.to(tl.uint8),
                    mask=gid < n_groups,
                )

    @triton.jit
    def _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel_multirow(
        out_y_ptr,
        out_scale_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        rows,
        row_offset,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
        rows_per_prog: tl.constexpr,
        n_conds: tl.constexpr,
        scale_stride: tl.constexpr,
    ):
        """Point-3 partition fix for tiny hidden sizes: R rows per program (2-D)."""
        pid = tl.program_id(0)
        r = tl.arange(0, rows_per_prog)
        row_ids = (pid * rows_per_prog + r).to(tl.int64) + row_offset
        rmask = row_ids < rows
        cols = tl.arange(0, block_n)
        cmask = cols < hidden_size
        mask = rmask[:, None] & cmask[None, :]

        index = _safe_index_vector(tl.load(indices_ptr + row_ids, mask=rmask, other=0), n_conds)
        x = tl.load(
            x_ptr + row_ids[:, None] * hidden_size + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=1) / hidden_size
        rstd = tl.rsqrt(variance + eps)
        if n_conds == 1:
            # One AdaLN condition: load it once as 1-D and broadcast.  A 2-D gather
            # with identical indices is lowered to a slow repeated-address path.
            shift = tl.load(shift_ptr + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            scale = tl.load(scale_ptr + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        else:
            shift = tl.load(
                shift_ptr + index[:, None] * hidden_size + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(
                scale_ptr + index[:, None] * hidden_size + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        mod = ((x * rstd[:, None]) * weight[None, :] * (1.0 + scale) + shift).to(tl.bfloat16)

        groups: tl.constexpr = block_n // 32
        n_groups: tl.constexpr = (hidden_size + 31) // 32
        flat_groups: tl.constexpr = rows_per_prog * groups
        bits = mod.to(tl.uint16, bitcast=True)
        absbits = bits & 0x7FFF
        maxbits = tl.max(tl.reshape(absbits, (flat_groups, 32)), axis=1)
        exp = (maxbits >> 7).to(tl.int32)
        byte = tl.minimum(tl.maximum(exp - 8, 0), 255)
        special = maxbits >= 0x7F80
        byte = tl.where(special, 255, byte)
        mult_bits = ((254 - byte) << 7).to(tl.uint16)
        mult = tl.reshape(mult_bits, (flat_groups, 1)).to(tl.bfloat16, bitcast=True)
        byte_col = tl.reshape(byte, (flat_groups, 1))
        mult = tl.where(byte_col > 0, mult, 0.0)
        mult = tl.broadcast_to(mult, (flat_groups, 32))
        yv = tl.reshape(mod, (flat_groups, 32)) * mult
        yb16 = yv.to(tl.uint16, bitcast=True)
        yabs = yb16 & 0x7FFF
        yb16 = tl.where(yabs > 0x43E0, (yb16 & 0x8000) | 0x43E0, yb16)
        yv = yb16.to(tl.bfloat16, bitcast=True)
        y8 = yv.to(tl.float32).to(tl.float8e4nv)
        y8 = tl.where(tl.reshape(byte == 255, (flat_groups, 1)), 0, y8.to(tl.uint8, bitcast=True)).to(
            tl.float8e4nv, bitcast=True
        )
        tl.store(
            out_y_ptr + row_ids[:, None] * hidden_size + cols[None, :],
            tl.reshape(y8, (rows_per_prog, block_n)),
            mask=mask,
        )

        gid = tl.arange(0, flat_groups)
        grow = gid // groups
        gcol = gid % groups
        grow_ids = (pid * rows_per_prog + grow).to(tl.int64) + row_offset
        smask = (grow_ids < rows) & (gcol < n_groups)
        tl.store(
            out_scale_ptr + grow_ids * scale_stride + gcol,
            byte.to(tl.uint8),
            mask=smask,
        )

    @triton.jit
    def _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel_h1(
        out_y_ptr,
        out_scale_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        row_offset,
        eps: tl.constexpr,
        scale_stride: tl.constexpr,
        n_conds: tl.constexpr,
    ):
        """H == 1 edge path: scalar per row, no 2-D reshape/stride folding."""
        row = (tl.program_id(0) + row_offset).to(tl.int64)
        index = _safe_scalar_index(tl.load(indices_ptr + row), n_conds)
        xv = tl.load(x_ptr + row).to(tl.float32)
        wv = tl.load(weight_ptr).to(tl.float32)
        rstd = tl.rsqrt(xv * xv + eps)
        sv = tl.load(shift_ptr + index).to(tl.float32)
        cv = tl.load(scale_ptr + index).to(tl.float32)
        mod = (xv * rstd * wv * (1.0 + cv) + sv).to(tl.bfloat16)

        bits = mod.to(tl.uint16, bitcast=True)
        exp = ((bits & 0x7FFF) >> 7).to(tl.int32)
        byte = tl.minimum(tl.maximum(exp - 8, 0), 255)
        special = (bits & 0x7FFF) >= 0x7F80
        byte = tl.where(special, 255, byte)
        mult_bits = ((254 - byte) << 7).to(tl.uint16)
        mult = mult_bits.to(tl.bfloat16, bitcast=True)
        mult = tl.where(byte > 0, mult, 0.0)
        yv = mod * mult
        yb = yv.to(tl.uint16, bitcast=True)
        yb = tl.where((yb & 0x7FFF) > 0x43E0, (yb & 0x8000) | 0x43E0, yb)
        yv = yb.to(tl.bfloat16, bitcast=True)
        yv = tl.where(special, 0.0, yv)
        yf = yv.to(tl.float32)
        # Force an fp32 op so the compiler does not fold this into the unsupported
        # scalar bf16 -> fp8 vcast (hivm.hir.vcast rintmode).
        yf = tl.maximum(tl.minimum(yf, 1.0e30), -1.0e30)
        tl.store(out_y_ptr + row, yf.to(tl.float8e4nv))
        tl.store(out_scale_ptr + row * scale_stride, byte.to(tl.uint8))

    @triton.jit
    def _modulation_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        row_offset,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
        out_fp16: tl.constexpr,
        n_conds: tl.constexpr,
    ):
        """RMSNorm + indexed AdaLN affine into ``out``, one row per program.

        The activation is written to HBM in the input dtype, matching the
        two-segment chain before ``npu_dynamic_mx_quant`` reads it back.
        ``block_n == hidden_size`` drops every mask, which is what makes
        this 32% faster than the reference modulation kernel (that one rounds the
        row up to a power of two, 8192 lanes for H = 5376).
        """
        row = (tl.program_id(0) + row_offset).to(tl.int64)
        cols = tl.arange(0, block_n)
        index = _safe_scalar_index(tl.load(indices_ptr + row), n_conds)
        if block_n == hidden_size:
            x = tl.load(x_ptr + row * hidden_size + cols).to(tl.float32)
            weight = tl.load(weight_ptr + cols).to(tl.float32)
        else:
            mask = cols < hidden_size
            x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / hidden_size
        normalized = x * tl.rsqrt(variance + eps) * weight
        if block_n == hidden_size:
            shift = tl.load(shift_ptr + index * hidden_size + cols).to(tl.float32)
            affine_scale = tl.load(scale_ptr + index * hidden_size + cols).to(tl.float32)
        else:
            mask = cols < hidden_size
            shift = tl.load(shift_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
            affine_scale = tl.load(scale_ptr + index * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
        value = normalized * (1.0 + affine_scale) + shift
        if out_fp16:
            value16 = value.to(tl.float16)
        else:
            value16 = value.to(tl.bfloat16)
        if block_n == hidden_size:
            tl.store(out_ptr + row * hidden_size + cols, value16)
        else:
            tl.store(out_ptr + row * hidden_size + cols, value16, mask=cols < hidden_size)

    def _hybrid_mx(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float,
        dst_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Triton affine + CANN ``npu_dynamic_mx_quant`` for large launches.

        The current Triton-Ascend version cannot emit the native DynamicMxQuant
        instruction. Measured on Ascend950 at 56240 x 5376, the single fused kernel costs
        2151 us, of which 1144 us is the affine part and ~1000 us the in-register
        MX tail, while CANN quantizes the same activation in 288 us.  The tail is
        the Triton cost of the per-32-element group amax (a cross-lane reduce) and
        of the FP4 threshold/packing chain.  The two-segment path therefore keeps
        the affine in Triton and calls the CANN operator of the reference chain.
        The activation is written to HBM in the input dtype, exactly as the
        reference chain does, so ``(y, mxscale)`` stay byte-exact.

        Args:
            x: activation, ``[rows, hidden]``, bf16 or fp16, contiguous.
            weight: RMSNorm weight, ``[hidden]``.
            shift: indexed AdaLN shift, ``[conditions, hidden]``.
            scale: indexed AdaLN scale, ``[conditions, hidden]``.
            indices: per-row condition index, ``[rows]``.
            eps: RMSNorm epsilon.
            dst_type: resolved MX output type.

        Returns:
            ``(y, mxscale)`` of the CANN DynamicMxQuant operator.
        """
        rows, hidden = x.shape
        _validate_indices(indices, shift.shape[0])
        groups = (hidden + 31) // 32
        block_n = (hidden + 255) // 256 * 256 if hidden > 512 else groups * 32
        modulated = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
        for row_offset in range(0, rows, _MAX_GRID):
            chunk_rows = min(rows - row_offset, _MAX_GRID)
            _modulation_kernel[(chunk_rows,)](
                modulated,
                x,
                weight,
                shift,
                scale,
                indices,
                row_offset,
                hidden,
                eps,
                block_n,
                x.dtype == torch.float16,
                shift.shape[0],
            )
        y, exponent = torch_npu.npu_dynamic_mx_quant(
            modulated,
            dst_type=_NPU_DST_TYPES[dst_type],
            axis=-1,
            block_size=32,
            round_mode="rint",
            scale_alg=0,
        )
        # ``npu_dynamic_mx_quant`` returns the exponent bytes as ``uint8``; the
        # operator contract (and the fused kernel) return ``float8_e8m0fnu``.
        return y, exponent.view(torch.float8_e8m0fnu)

    def _fused_e4m3(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float = 1e-6,
        dst_type: str = "float8_e4m3fn",
    ):
        """Run the specialized BF16-to-E4M3 fused path.

        The public dispatcher enters here only after validation and normalization.
        The grid is chunked at ``_MAX_GRID`` rows and ``H`` is rounded up to a
        complete MX-group block.
        """
        dst_type = dst_type.lower()
        if dst_type != "float8_e4m3fn":
            raise ParametersInvalid(f"Unsupported dst_type {dst_type}.")

        rows, hidden = x.shape
        if hidden <= 0:
            raise ParametersInvalid("The hidden size of x must be positive.")
        _validate_indices(indices, shift.shape[0])

        n_groups = (hidden + 31) // 32
        n_pad = (n_groups + 1) // 2 * 2
        if _use_hybrid(rows, hidden):
            # Large launches: the in-register MX tail costs several times CANN's,
            # so keep the (unmasked, faster) Triton affine and let CANN quantize.
            return _hybrid_mx(x, weight, shift, scale, indices, eps, dst_type)
        # Smallest block that covers the row and all complete 32-element MX
        # groups.  For the model H=5376 this is exactly 5376.
        block_n = (hidden + 255) // 256 * 256 if hidden > 512 else n_groups * 32
        scale_shape = (rows, n_pad // 2, 2)
        if n_pad == n_groups:
            output_scale = torch.empty(
                scale_shape,
                dtype=torch.uint8,
                device=x.device,
            )
        else:
            # DynamicMxQuant pads an odd group count with one zero exponent byte.
            output_scale = torch.zeros(
                scale_shape,
                dtype=torch.uint8,
                device=x.device,
            )
        y = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=x.device)

        if rows:
            if hidden == 1:
                for row_offset in range(0, rows, _MAX_GRID):
                    chunk_rows = min(rows - row_offset, _MAX_GRID)
                    _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel_h1[(chunk_rows,)](
                        y,
                        output_scale,
                        x,
                        weight,
                        shift,
                        scale,
                        indices,
                        row_offset,
                        eps,
                        n_pad,
                        shift.shape[0],
                    )
                return y, output_scale.view(torch.float8_e8m0fnu)
            rows_per_prog = _e4m3_rows_per_prog(block_n, hidden, rows)
            if rows_per_prog:
                # The launch grid tops out at 65535 programs, so at most
                # ``65535 * rows_per_prog`` rows go out in one launch.
                chunk_rows = _MAX_GRID * rows_per_prog
                for row_offset in range(0, rows, chunk_rows):
                    grid = (triton.cdiv(min(rows - row_offset, chunk_rows), rows_per_prog),)
                    _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel_multirow[grid](
                        y,
                        output_scale,
                        x,
                        weight,
                        shift,
                        scale,
                        indices,
                        rows,
                        row_offset,
                        hidden,
                        eps,
                        block_n,
                        rows_per_prog,
                        shift.shape[0],
                        n_pad,
                    )
                return y, output_scale.view(torch.float8_e8m0fnu)
            # Keep x resident through quantization whenever the exact block fits.
            # On Ascend950 the measured crossover is above 8193 elements; the tiled
            # fallback re-reads x and loses 14%-17% for small-row large-hidden cases.
            use_tiled = hidden > 8193
            var_block = triton.next_power_of_2(hidden)
            chunked_var = var_block > 32768
            if chunked_var:
                var_block = 32768
            for row_offset in range(0, rows, _MAX_GRID):
                chunk_rows = min(rows - row_offset, _MAX_GRID)
                if use_tiled:
                    _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel_tiled[(chunk_rows,)](
                        y,
                        output_scale,
                        x,
                        weight,
                        shift,
                        scale,
                        indices,
                        row_offset,
                        hidden,
                        eps,
                        4096,
                        var_block,
                        chunked_var,
                        n_pad,
                        shift.shape[0],
                        multibuffer=False,
                    )
                else:
                    _rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel[(chunk_rows,)](
                        y,
                        output_scale,
                        x,
                        weight,
                        shift,
                        scale,
                        indices,
                        row_offset,
                        hidden,
                        eps,
                        block_n,
                        n_pad,
                        shift.shape[0],
                        multibuffer=False,
                    )

        return y, output_scale.view(torch.float8_e8m0fnu)

    # FP16/BF16 -> FP8/FP4 generalized path.
    @triton.jit
    def _general_fullrow_kernel(
        out_y_ptr,
        out_scale_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        row_offset,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
        in_fp16: tl.constexpr,
        out_fp4: tl.constexpr,
        fp4_e1m2: tl.constexpr,
        out_e5m2: tl.constexpr,
        scale_stride: tl.constexpr,
        n_conds: tl.constexpr,
    ):
        """Quantize one complete row while keeping the activation in registers."""
        row = (tl.program_id(0) + row_offset).to(tl.int64)
        cols = tl.arange(0, block_n)
        mask = cols < hidden_size
        index = _safe_scalar_index(tl.load(indices_ptr + row), n_conds)
        if block_n == hidden_size:
            x = tl.load(x_ptr + row * hidden_size + cols).to(tl.float32)
            weight = tl.load(weight_ptr + cols).to(tl.float32)
        else:
            x = tl.load(
                x_ptr + row * hidden_size + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / hidden_size
        rstd = tl.rsqrt(variance + eps)
        if block_n == hidden_size:
            shift = tl.load(shift_ptr + index * hidden_size + cols).to(tl.float32)
            affine_scale = tl.load(scale_ptr + index * hidden_size + cols).to(tl.float32)
        else:
            shift = tl.load(
                shift_ptr + index * hidden_size + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            affine_scale = tl.load(
                scale_ptr + index * hidden_size + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        out = x * rstd * weight * (1.0 + affine_scale) + shift
        if in_fp16:
            mod = out.to(tl.float16)
        else:
            mod = out.to(tl.bfloat16)

        groups: tl.constexpr = block_n // 32
        n_groups: tl.constexpr = (hidden_size + 31) // 32
        bits = mod.to(tl.uint16, bitcast=True)
        absbits = bits & 0x7FFF
        maxbits = tl.max(tl.reshape(absbits, (groups, 32)), axis=1)
        if in_fp16:
            special = maxbits >= 0x7C00
            mex = ((maxbits >> 10) & 0x1F).to(tl.int32)
            if out_fp4:
                byte = mex + (112 if fp4_e1m2 else 110)
            elif out_e5m2:
                byte = mex + 97
            else:
                byte = mex + 104
        else:
            special = maxbits >= 0x7F80
            mex = ((maxbits >> 7) & 0xFF).to(tl.int32)
            if out_fp4:
                byte = mex - (0 if fp4_e1m2 else 2)
            elif out_e5m2:
                byte = mex - 15
            else:
                byte = mex - 8
        byte = tl.where(maxbits == 0, 0, byte)
        byte = tl.minimum(tl.maximum(byte, 0), 255)
        byte = tl.where(special, 255, byte)
        if out_fp4:
            quant_byte = tl.where((byte > 0) & (byte < 255), byte, 127)
            scale_bits = quant_byte.to(tl.uint32) << 23
            divider = scale_bits.to(tl.float32, bitcast=True)
            divider = tl.broadcast_to(
                tl.reshape(divider, (groups, 1)),
                (groups, 32),
            )
            yv = tl.fdiv(
                tl.reshape(mod, (groups, 32)).to(tl.float32),
                divider,
                ieee_rounding=True,
            )
        else:
            # e8m0 scaling is an exact power of two. Build the reciprocal in the
            # fp32 exponent and multiply once instead of issuing per-element fdiv.
            mult_byte = tl.where((byte != 0) & (byte != 255), byte, 127)
            mult_bits = ((254 - mult_byte) << 23).to(tl.uint32)
            mult = mult_bits.to(tl.float32, bitcast=True)
            mult = tl.broadcast_to(
                tl.reshape(mult, (groups, 1)),
                (groups, 32),
            )
            yv = tl.reshape(mod, (groups, 32)).to(tl.float32) * mult

        if out_fp4:
            abs_values = tl.abs(yv.to(tl.float32))
            if fp4_e1m2:
                code = tl.where(
                    abs_values <= 0.125,
                    0,
                    tl.where(
                        abs_values < 0.375,
                        1,
                        tl.where(
                            abs_values <= 0.625,
                            2,
                            tl.where(
                                abs_values < 0.875,
                                3,
                                tl.where(
                                    abs_values <= 1.125,
                                    4,
                                    tl.where(
                                        abs_values < 1.375,
                                        5,
                                        tl.where(abs_values <= 1.625, 6, 7),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            else:
                code = tl.where(
                    abs_values <= 0.25,
                    0,
                    tl.where(
                        abs_values < 0.75,
                        1,
                        tl.where(
                            abs_values <= 1.25,
                            2,
                            tl.where(
                                abs_values < 1.75,
                                3,
                                tl.where(
                                    abs_values <= 2.5,
                                    4,
                                    tl.where(
                                        abs_values < 3.5,
                                        5,
                                        tl.where(abs_values <= 5.0, 6, 7),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            # Preserve the sign bit for negative zero.  Comparing ``yv < 0``
            # loses that bit because IEEE comparisons treat -0.0 == 0.0.
            sign = (yv.to(tl.uint32, bitcast=True) >> 31).to(tl.int32)
            code = code | (sign << 3)
            invalid_group = (byte == 0) | (byte == 255)
            code = tl.where(
                tl.broadcast_to(
                    tl.reshape(invalid_group, (groups, 1)),
                    (groups, 32),
                ),
                0,
                code,
            )
            pairs = tl.reshape(code, (groups * 16, 2))
            lo, hi = tl.split(pairs)
            packed = (lo | (hi << 4)).to(tl.uint8)
            packed_offsets = tl.arange(0, groups * 16)
            tl.store(
                out_y_ptr + row * (hidden_size // 2) + packed_offsets,
                packed,
                mask=packed_offsets < (hidden_size // 2),
            )
        elif out_e5m2:
            yv = tl.minimum(tl.maximum(yv, -57344.0), 57344.0)
            y8 = yv.to(tl.float8e5)
            yb8 = y8.to(tl.uint8, bitcast=True)
            yb8 = tl.where(
                tl.reshape((byte == 0) | (byte == 255), (groups, 1)),
                tl.zeros_like(yb8),
                yb8,
            )
            y8 = yb8.to(tl.float8e5, bitcast=True)
            if block_n == hidden_size:
                tl.store(
                    out_y_ptr + row * hidden_size + cols,
                    tl.reshape(y8, (block_n,)),
                )
            else:
                tl.store(
                    out_y_ptr + row * hidden_size + cols,
                    tl.reshape(y8, (block_n,)),
                    mask=mask,
                )
        else:
            yv = tl.minimum(tl.maximum(yv, -448.0), 448.0)
            y8 = yv.to(tl.float8e4nv)
            y8 = tl.where(
                tl.reshape((byte == 0) | (byte == 255), (groups, 1)),
                0,
                y8.to(tl.uint8, bitcast=True),
            ).to(tl.float8e4nv, bitcast=True)
            if block_n == hidden_size:
                tl.store(
                    out_y_ptr + row * hidden_size + cols,
                    tl.reshape(y8, (block_n,)),
                )
            else:
                tl.store(
                    out_y_ptr + row * hidden_size + cols,
                    tl.reshape(y8, (block_n,)),
                    mask=mask,
                )
        group_ids = tl.arange(0, groups)
        if block_n == hidden_size:
            tl.store(
                out_scale_ptr + row * scale_stride + group_ids,
                byte.to(tl.uint8),
            )
        else:
            tl.store(
                out_scale_ptr + row * scale_stride + group_ids,
                byte.to(tl.uint8),
                mask=group_ids < n_groups,
            )

    @triton.jit
    def _general_multirow_kernel(
        out_y_ptr,
        out_scale_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        rows,
        row_offset,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
        rows_per_prog: tl.constexpr,
        in_fp16: tl.constexpr,
        out_fp4: tl.constexpr,
        fp4_e1m2: tl.constexpr,
        out_e5m2: tl.constexpr,
        scale_stride: tl.constexpr,
        n_conds: tl.constexpr,
    ):
        """Pack several tiny-hidden rows into one program."""
        pid = tl.program_id(0)
        row_ids = (pid * rows_per_prog + tl.arange(0, rows_per_prog)).to(tl.int32) + row_offset
        rmask = row_ids.to(tl.float32) < rows
        cols = tl.arange(0, block_n)
        cmask = cols.to(tl.float32) < hidden_size
        mask = rmask[:, None] & cmask[None, :]

        indices = _safe_index_vector(
            tl.load(indices_ptr + row_ids, mask=rmask, other=0).to(tl.int32),
            n_conds,
        )
        x = tl.load(
            x_ptr + row_ids[:, None] * hidden_size + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=1) / hidden_size
        rstd = tl.rsqrt(variance + eps)
        shift = tl.load(
            shift_ptr + indices[:, None] * hidden_size + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        affine_scale = tl.load(
            scale_ptr + indices[:, None] * hidden_size + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out = (x * rstd[:, None]) * weight[None, :] * (1.0 + affine_scale) + shift
        if in_fp16:
            mod = out.to(tl.float16)
        else:
            mod = out.to(tl.bfloat16)

        groups: tl.constexpr = block_n // 32
        n_groups: tl.constexpr = (hidden_size + 31) // 32
        flat_groups: tl.constexpr = rows_per_prog * groups
        bits = mod.to(tl.uint16, bitcast=True)
        absbits = bits & 0x7FFF
        maxbits = tl.max(tl.reshape(absbits, (flat_groups, 32)), axis=1)
        if in_fp16:
            special = maxbits.to(tl.float32) >= 0x7C00
            mex = ((maxbits >> 10) & 0x1F).to(tl.int32)
            if out_fp4:
                byte = mex + (112 if fp4_e1m2 else 110)
            elif out_e5m2:
                byte = mex + 97
            else:
                byte = mex + 104
        else:
            special = maxbits.to(tl.float32) >= 0x7F80
            mex = ((maxbits >> 7) & 0xFF).to(tl.int32)
            if out_fp4:
                byte = mex - (0 if fp4_e1m2 else 2)
            elif out_e5m2:
                byte = mex - 15
            else:
                byte = mex - 8
        byte = tl.where(maxbits == 0, 0, byte)
        byte = tl.minimum(tl.maximum(byte, 0), 255)
        byte = tl.where(special, 255, byte)

        if out_fp4:
            quant_byte = tl.where((byte != 0) & (byte != 255), byte, 127)
            scale_bits = quant_byte.to(tl.uint32) << 23
            divider = scale_bits.to(tl.float32, bitcast=True)
            divider = tl.broadcast_to(
                tl.reshape(divider, (flat_groups, 1)),
                (flat_groups, 32),
            )
            yv = tl.fdiv(
                tl.reshape(mod, (flat_groups, 32)).to(tl.float32),
                divider,
                ieee_rounding=True,
            )
        else:
            mult_byte = tl.where((byte != 0) & (byte != 255), byte, 127)
            mult_bits = ((254 - mult_byte) << 23).to(tl.uint32)
            mult = mult_bits.to(tl.float32, bitcast=True)
            mult = tl.broadcast_to(
                tl.reshape(mult, (flat_groups, 1)),
                (flat_groups, 32),
            )
            yv = tl.reshape(mod, (flat_groups, 32)).to(tl.float32) * mult

        if out_fp4:
            abs_values = tl.abs(yv)
            if fp4_e1m2:
                code = tl.where(abs_values <= 0.125, 0, 7)
                code = tl.where((abs_values > 0.125) & (abs_values < 0.375), 1, code)
                code = tl.where((abs_values >= 0.375) & (abs_values <= 0.625), 2, code)
                code = tl.where((abs_values > 0.625) & (abs_values < 0.875), 3, code)
                code = tl.where((abs_values >= 0.875) & (abs_values <= 1.125), 4, code)
                code = tl.where((abs_values > 1.125) & (abs_values < 1.375), 5, code)
                code = tl.where((abs_values >= 1.375) & (abs_values <= 1.625), 6, code)
            else:
                code = tl.where(abs_values <= 0.25, 0, 7)
                code = tl.where((abs_values > 0.25) & (abs_values < 0.75), 1, code)
                code = tl.where((abs_values >= 0.75) & (abs_values <= 1.25), 2, code)
                code = tl.where((abs_values > 1.25) & (abs_values < 1.75), 3, code)
                code = tl.where((abs_values >= 1.75) & (abs_values <= 2.5), 4, code)
                code = tl.where((abs_values > 2.5) & (abs_values < 3.5), 5, code)
                code = tl.where((abs_values >= 3.5) & (abs_values <= 5.0), 6, code)
            sign = (yv.to(tl.uint32, bitcast=True) >> 31).to(tl.int32)
            code = code | (sign << 3)
            invalid_group = (byte == 0) | (byte == 255)
            code = tl.where(
                tl.broadcast_to(
                    tl.reshape(invalid_group, (flat_groups, 1)),
                    (flat_groups, 32),
                ),
                0,
                code,
            )
            pairs = tl.reshape(code, (flat_groups * 16, 2))
            lo, hi = tl.split(pairs)
            packed = (lo | (hi << 4)).to(tl.uint8)
            packed_cols = tl.arange(0, block_n // 2)
            tl.store(
                out_y_ptr + row_ids[:, None] * (hidden_size // 2) + packed_cols[None, :],
                tl.reshape(packed, (rows_per_prog, block_n // 2)),
                mask=rmask[:, None] & (packed_cols.to(tl.float32)[None, :] < hidden_size // 2),
            )
        elif out_e5m2:
            yv = tl.minimum(tl.maximum(yv, -57344.0), 57344.0)
            y8 = yv.to(tl.float8e5)
            yb8 = y8.to(tl.uint8, bitcast=True)
            yb8 = tl.where(
                tl.reshape((byte == 0) | (byte == 255), (flat_groups, 1)),
                tl.zeros_like(yb8),
                yb8,
            )
            y8 = yb8.to(tl.float8e5, bitcast=True)
            tl.store(
                out_y_ptr + row_ids[:, None] * hidden_size + cols[None, :],
                tl.reshape(y8, (rows_per_prog, block_n)),
                mask=mask,
            )
        else:
            yv = tl.minimum(tl.maximum(yv, -448.0), 448.0)
            y8 = yv.to(tl.float8e4nv)
            y8 = tl.where(
                tl.reshape((byte == 0) | (byte == 255), (flat_groups, 1)),
                0,
                y8.to(tl.uint8, bitcast=True),
            ).to(tl.float8e4nv, bitcast=True)
            tl.store(
                out_y_ptr + row_ids[:, None] * hidden_size + cols[None, :],
                tl.reshape(y8, (rows_per_prog, block_n)),
                mask=mask,
            )

        gid = tl.arange(0, flat_groups)
        grow = gid // groups
        gcol = gid - grow * groups
        grow_ids = (pid * rows_per_prog + grow).to(tl.int32) + row_offset
        smask = (grow_ids.to(tl.float32) < rows) & (gcol.to(tl.float32) < n_groups)
        tl.store(
            out_scale_ptr + grow_ids * scale_stride + gcol,
            byte.to(tl.uint8),
            mask=smask,
        )

    @triton.jit
    def _general_tiled_kernel(
        out_y_ptr,
        out_scale_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        row_offset,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        tile_n: tl.constexpr,
        var_block: tl.constexpr,
        chunked_var: tl.constexpr,
        in_fp16: tl.constexpr,
        out_fp4: tl.constexpr,
        fp4_e1m2: tl.constexpr,
        out_e5m2: tl.constexpr,
        scale_stride: tl.constexpr,
        n_conds: tl.constexpr,
    ):
        """Two-pass RMSNorm and quantized affine for rows larger than one tile."""
        row = (tl.program_id(0) + row_offset).to(tl.int64)
        index = _safe_scalar_index(tl.load(indices_ptr + row), n_conds)
        n_groups: tl.constexpr = (hidden_size + 31) // 32

        if chunked_var:
            acc = 0.0
            for off in range(0, hidden_size, tile_n):
                cols = off + tl.arange(0, tile_n)
                mask = cols < hidden_size
                xx = tl.load(
                    x_ptr + row * hidden_size + cols,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(xx * xx, axis=0)
            variance = acc / hidden_size
        else:
            vcols = tl.arange(0, var_block)
            vmask = vcols < hidden_size
            xv = tl.load(
                x_ptr + row * hidden_size + vcols,
                mask=vmask,
                other=0.0,
            ).to(tl.float32)
            variance = tl.sum(xv * xv, axis=0) / hidden_size
        rstd = tl.rsqrt(variance + eps)

        groups: tl.constexpr = tile_n // 32
        gid0 = tl.arange(0, groups)
        for off in range(0, hidden_size, tile_n):
            cols = off + tl.arange(0, tile_n)
            mask = cols < hidden_size
            x = tl.load(
                x_ptr + row * hidden_size + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            shift = tl.load(
                shift_ptr + index * hidden_size + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            affine_scale = tl.load(
                scale_ptr + index * hidden_size + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            out = x * rstd * weight * (1.0 + affine_scale) + shift
            if in_fp16:
                mod = out.to(tl.float16)
            else:
                mod = out.to(tl.bfloat16)

            bits = mod.to(tl.uint16, bitcast=True)
            absbits = bits & 0x7FFF
            exp_mask: tl.constexpr = 0x7C00 if in_fp16 else 0x7F80
            maxbits = tl.max(tl.reshape(absbits, (groups, 32)), axis=1)
            special = maxbits >= exp_mask
            if in_fp16:
                mex = ((maxbits >> 10) & 0x1F).to(tl.int32)
                if out_fp4:
                    byte = mex + (112 if fp4_e1m2 else 110)
                elif out_e5m2:
                    byte = mex + 97
                else:
                    byte = mex + 104
            else:
                mex = ((maxbits >> 7) & 0xFF).to(tl.int32)
                if out_fp4:
                    byte = mex - (0 if fp4_e1m2 else 2)
                elif out_e5m2:
                    byte = mex - 15
                else:
                    byte = mex - 8
            byte = tl.where(maxbits == 0, 0, byte)
            byte = tl.minimum(tl.maximum(byte, 0), 255)
            byte = tl.where(special, 255, byte)

            if out_fp4:
                quant_byte = tl.where((byte > 0) & (byte < 255), byte, 127)
                scale_bits = quant_byte.to(tl.uint32) << 23
                divider = scale_bits.to(tl.float32, bitcast=True)
                divider = tl.broadcast_to(
                    tl.reshape(divider, (groups, 1)),
                    (groups, 32),
                )
                yv = tl.fdiv(
                    tl.reshape(mod, (groups, 32)).to(tl.float32),
                    divider,
                    ieee_rounding=True,
                )
            else:
                mult_byte = tl.where((byte != 0) & (byte != 255), byte, 127)
                mult_bits = ((254 - mult_byte) << 23).to(tl.uint32)
                mult = mult_bits.to(tl.float32, bitcast=True)
                mult = tl.broadcast_to(
                    tl.reshape(mult, (groups, 1)),
                    (groups, 32),
                )
                yv = tl.reshape(mod, (groups, 32)).to(tl.float32) * mult

            if out_fp4:
                abs_values = tl.abs(yv.to(tl.float32))
                if fp4_e1m2:
                    code = tl.where(
                        abs_values <= 0.125,
                        0,
                        tl.where(
                            abs_values < 0.375,
                            1,
                            tl.where(
                                abs_values <= 0.625,
                                2,
                                tl.where(
                                    abs_values < 0.875,
                                    3,
                                    tl.where(
                                        abs_values <= 1.125,
                                        4,
                                        tl.where(
                                            abs_values < 1.375,
                                            5,
                                            tl.where(abs_values <= 1.625, 6, 7),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                else:
                    code = tl.where(
                        abs_values <= 0.25,
                        0,
                        tl.where(
                            abs_values < 0.75,
                            1,
                            tl.where(
                                abs_values <= 1.25,
                                2,
                                tl.where(
                                    abs_values < 1.75,
                                    3,
                                    tl.where(
                                        abs_values <= 2.5,
                                        4,
                                        tl.where(
                                            abs_values < 3.5,
                                            5,
                                            tl.where(abs_values <= 5.0, 6, 7),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                # Keep the FP4 sign bit for negative zero.
                sign = (yv.to(tl.uint32, bitcast=True) >> 31).to(tl.int32)
                code = code | (sign << 3)
                invalid_group = (byte == 0) | (byte == 255)
                code = tl.where(
                    tl.broadcast_to(
                        tl.reshape(invalid_group, (groups, 1)),
                        (groups, 32),
                    ),
                    0,
                    code,
                )
                pairs = tl.reshape(code, (groups * 16, 2))
                lo, hi = tl.split(pairs)
                packed = (lo | (hi << 4)).to(tl.uint8)
                packed_offsets = off // 2 + tl.arange(0, groups * 16)
                tl.store(
                    out_y_ptr + row * (hidden_size // 2) + packed_offsets,
                    packed,
                    mask=packed_offsets < (hidden_size // 2),
                )
            elif out_e5m2:
                yv = tl.minimum(tl.maximum(yv, -57344.0), 57344.0)
                y8 = yv.to(tl.float8e5)
                yb8 = y8.to(tl.uint8, bitcast=True)
                yb8 = tl.where(
                    tl.reshape((byte == 0) | (byte == 255), (groups, 1)),
                    tl.zeros_like(yb8),
                    yb8,
                )
                y8 = yb8.to(tl.float8e5, bitcast=True)
                tl.store(
                    out_y_ptr + row * hidden_size + cols,
                    tl.reshape(y8, (tile_n,)),
                    mask=mask,
                )
            else:
                yv = tl.minimum(tl.maximum(yv, -448.0), 448.0)
                y8 = yv.to(tl.float8e4nv)
                y8 = tl.where(
                    tl.reshape((byte == 0) | (byte == 255), (groups, 1)),
                    0,
                    y8.to(tl.uint8, bitcast=True),
                ).to(tl.float8e4nv, bitcast=True)
                tl.store(
                    out_y_ptr + row * hidden_size + cols,
                    tl.reshape(y8, (tile_n,)),
                    mask=mask,
                )

            group_ids = off // 32 + gid0
            tl.store(
                out_scale_ptr + row * scale_stride + group_ids,
                byte.to(tl.uint8),
                mask=group_ids < n_groups,
            )

    def _fused_general_mx(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float = 1e-6,
        dst_type: str = "float8_e4m3fn",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the generalized FP8/FP4 path.

        This path supports ``bfloat16`` or ``float16`` input and
        ``float8_e4m3fn``, ``float8_e5m2``, ``float4_e2m1fn_x2`` or
        ``float4_e1m2fn_x2`` output.
        """
        dst_type = dst_type.lower()
        if dst_type not in _SUPPORTED_DST_TYPES:
            raise ParametersInvalid(f"Unsupported dst_type {dst_type}.")
        in_fp16 = x.dtype == torch.float16
        if not in_fp16 and x.dtype != torch.bfloat16:
            raise ParametersInvalid("The dtype of x must be float16 or bfloat16.")
        if weight.dtype != x.dtype or shift.dtype != x.dtype or scale.dtype != x.dtype:
            raise ParametersInvalid("The dtype of x, weight, shift and scale must be the same.")
        rows, hidden = x.shape
        out_fp4 = dst_type in ("float4_e2m1fn_x2", "float4_e1m2fn_x2")
        fp4_e1m2 = dst_type == "float4_e1m2fn_x2"
        out_e5m2 = dst_type == "float8_e5m2"
        if out_fp4 and hidden % 2:
            raise ParametersInvalid(f"MXFP4 output requires an even hidden size, but got {hidden}.")
        for t in (x, weight, shift, scale, indices):
            if not t.is_contiguous():
                raise ParametersInvalid("The input tensors must be contiguous.")
        _validate_indices(indices, shift.shape[0])
        if _use_hybrid(rows, hidden):
            return _hybrid_mx(x, weight, shift, scale, indices, eps, dst_type)
        n_groups = (hidden + 31) // 32
        n_pad = (n_groups + 1) // 2 * 2
        scale_shape = (rows, n_pad // 2, 2)
        output_scale = (
            torch.empty(scale_shape, dtype=torch.uint8, device=x.device)
            if n_pad == n_groups
            else torch.zeros(scale_shape, dtype=torch.uint8, device=x.device)
        )
        if out_fp4:
            ybuf = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=x.device)
        elif out_e5m2:
            ybuf = torch.empty((rows, hidden), dtype=torch.float8_e5m2, device=x.device)
        else:
            ybuf = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=x.device)
        var_block = triton.next_power_of_2(hidden)
        chunked = var_block > 32768
        if chunked:
            var_block = 32768
        tile_n = 2048 if out_fp4 else (8192 if hidden == 8192 else 4096)
        if rows:
            tiny_block_n = (hidden + 31) // 32 * 32
            if tiny_block_n <= 64 and rows >= 32:
                if rows < 4096:
                    rows_per_prog = 8
                elif tiny_block_n <= 32:
                    rows_per_prog = 256
                else:
                    rows_per_prog = 128
                if out_fp4:
                    # The MXFP4 tail (e8m0 divide, threshold chain, nibble packing)
                    # needs more UB than the FP8 tail: the FP8 row counts (128/256)
                    # overflow UB and fail to compile for these shapes.  64 rows per
                    # program is the largest value that still compiles and the
                    # fastest point probed (4/8/16/32/64/96/128) once the grid holds
                    # enough programs.  Below that a fixed 64 leaves the cores idle
                    # (32 rows become one program, 65 rows become two), so the row
                    # count shrinks with ``rows``: the measured optimum keeps about
                    # 64 programs in flight (1.5-2.0x over a fixed 64 at 32..512
                    # rows, identical from 4096 rows up).
                    rows_per_prog = 8
                    for candidate in (16, 32, 64):
                        if rows // candidate >= 64:
                            rows_per_prog = candidate
                rows_per_launch = rows_per_prog * _MAX_GRID
                for row_offset in range(0, rows, rows_per_launch):
                    chunk_rows = min(rows - row_offset, rows_per_launch)
                    _general_multirow_kernel[(triton.cdiv(chunk_rows, rows_per_prog),)](
                        ybuf,
                        output_scale,
                        x,
                        weight,
                        shift,
                        scale,
                        indices,
                        rows,
                        row_offset,
                        hidden,
                        eps,
                        tiny_block_n,
                        rows_per_prog,
                        in_fp16,
                        out_fp4,
                        fp4_e1m2,
                        out_e5m2,
                        n_pad,
                        shift.shape[0],
                        multibuffer=False,
                    )
            elif hidden <= 6144 and (not out_fp4 or hidden <= 4096):
                # MXFP4 keeps the whole row in one program up to a 4096-wide block;
                # at 5632 (hidden 5376) the packed tail spills and measures ~1.6x
                # slower than the tiled path on Ascend950DT.
                block_n = (hidden + 255) // 256 * 256 if hidden > 512 else n_groups * 32
                for row_offset in range(0, rows, _MAX_GRID):
                    chunk_rows = min(rows - row_offset, _MAX_GRID)
                    _general_fullrow_kernel[(chunk_rows,)](
                        ybuf,
                        output_scale,
                        x,
                        weight,
                        shift,
                        scale,
                        indices,
                        row_offset,
                        hidden,
                        eps,
                        block_n,
                        in_fp16,
                        out_fp4,
                        fp4_e1m2,
                        out_e5m2,
                        n_pad,
                        shift.shape[0],
                        multibuffer=False,
                    )
            else:
                for row_offset in range(0, rows, _MAX_GRID):
                    chunk_rows = min(rows - row_offset, _MAX_GRID)
                    _general_tiled_kernel[(chunk_rows,)](
                        ybuf,
                        output_scale,
                        x,
                        weight,
                        shift,
                        scale,
                        indices,
                        row_offset,
                        hidden,
                        eps,
                        tile_n,
                        var_block,
                        chunked,
                        in_fp16,
                        out_fp4,
                        fp4_e1m2,
                        out_e5m2,
                        n_pad,
                        shift.shape[0],
                        multibuffer=False,
                    )
        y = ybuf
        return y, output_scale.view(torch.float8_e8m0fnu)

else:

    def _fused_e4m3(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float = 1e-6,
        dst_type: str = "float8_e4m3fn",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Triton is not available. Use the fused operator on Ascend NPU.")

    def _fused_general_mx(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float = 1e-6,
        dst_type: str = "float8_e4m3fn",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Triton is not available. Use the fused operator on Ascend NPU.")


@torch.library.custom_op(
    "mindiesd::rms_norm_indexed_scale_shift_dynamic_mx_quant",
    mutates_args=(),
)
def rms_norm_indexed_scale_shift_dynamic_mx_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-6,
    dst_type: str = "float8_e4m3fn",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse RMSNorm, indexed AdaLN affine and DynamicMxQuant.

    Args:
        x: activation, 2-D ``[rows, hidden]``, ``bfloat16`` or ``float16``.
        weight: RMSNorm weight, 1-D ``[hidden]``, same dtype as ``x``.
        shift: indexed AdaLN shift, 2-D ``[conditions, hidden]``.
        scale: indexed AdaLN scale, 2-D ``[conditions, hidden]``. The affine is
            ``x * (1 + scale[index]) + shift[index]``.
        eps: finite positive value added to the RMSNorm variance.
        dst_type: MX output type, ``"float8_e4m3fn"`` (default),
            ``"float8_e5m2"``, ``"float4_e2m1fn_x2"`` or ``"float4_e1m2fn_x2"``.
            ``"float4_e2m1"`` and ``"float4_e1m2"`` are accepted aliases.

    Returns:
        A tuple of ``(y, mxscale)``. ``y`` is the quantized activation:
        ``[rows, hidden]`` for FP8 output and ``[rows, hidden // 2]`` (packed
        ``uint8``) for FP4 output. ``mxscale`` is the ``float8_e8m0fnu`` block
        scale of shape ``[rows, ceil(groups / 2), 2]``, one byte per 32-element
        MX group. The AdaLN activation itself is not returned: the fused norm
        exists for the quantized consumer. The single-kernel path keeps the
        rounded activation in registers; the two-stage path stores it in a
        temporary HBM tensor before quantization. Callers that need the floating
        activation should use the unfused normalization operator instead.

    Raises:
        ParametersInvalid: If any input violates the documented shape, dtype,
            device or value constraints.
        RuntimeError: If Triton is not available on the Ascend NPU.
    """
    dst_type = _DST_TYPE_ALIASES.get(dst_type.lower(), dst_type.lower())
    _validate_inputs(x, weight, shift, scale, indices, eps, dst_type)
    if not _TRITON_ON_ASCEND:
        raise RuntimeError("RMSNorm indexed DynamicMxQuant requires Triton on Ascend NPU.")

    x = x.contiguous()
    weight = weight.contiguous()
    shift = shift.contiguous()
    scale = scale.contiguous()
    indices = indices.contiguous()
    if x.dtype == torch.bfloat16 and dst_type == "float8_e4m3fn":
        return _fused_e4m3(x, weight, shift, scale, indices, eps, dst_type)
    return _fused_general_mx(x, weight, shift, scale, indices, eps, dst_type)


@rms_norm_indexed_scale_shift_dynamic_mx_quant.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-6,
    dst_type: str = "float8_e4m3fn",
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, shift, scale, indices, eps
    rows, hidden = x.shape
    is_fp4 = dst_type.lower().startswith("float4")
    y_shape = (rows, hidden // 2) if is_fp4 else (rows, hidden)
    y_dtype = (
        torch.uint8 if is_fp4 else (torch.float8_e5m2 if dst_type.lower() == "float8_e5m2" else torch.float8_e4m3fn)
    )
    return (
        torch.empty(y_shape, dtype=y_dtype, device=x.device),
        torch.empty(
            _mxscale_shape(rows, hidden),
            dtype=torch.float8_e8m0fnu,
            device=x.device,
        ),
    )
