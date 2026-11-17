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

"""Fused split, RMSNorm, partial NeoX RoPE, concat, and optional V copy.

The input is packed QKV ``[T, 3*N*D]``. A single Ascend Triton kernel owns one
token per program and processes compile-time head tiles. The V branch is either
left as a caller-owned view of the packed input (``v_out=None``) or copied into
the supplied contiguous output.
"""

import math

import torch

from ..utils import ParametersInvalid
from .triton_utils import _HAS_TRITON, _TRITON_ON_ASCEND

if _HAS_TRITON:  # pragma: no cover - Imported only when the Triton-Ascend runtime is available.
    from .triton_utils import tl, triton


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_MAX_TOKENS = 262144
_MAX_HEADS = 128
_MIN_HEAD_DIM = 96
_MAX_HEAD_DIM = 256
_MAX_GRID = 65535


if _HAS_TRITON:  # pragma: no cover - Kernel body executes only through an Ascend Triton launch.

    @triton.jit
    def _split_qkv_norm_rope_kernel(
        qkv_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cache_ptr,
        positions_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        num_tokens,
        token_offset,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        ROW_WIDTH: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_ROPE_HALF: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
        NUM_HEAD_BLOCKS: tl.constexpr,
        HAS_TAIL: tl.constexpr,
        WRITE_V: tl.constexpr,
        EPS: tl.constexpr,
    ):
        token = (tl.program_id(0).to(tl.int32) + token_offset).to(tl.int32)
        valid_token = token < num_tokens
        input_token_base = token.to(tl.int64) * ROW_WIDTH
        output_token_base = token.to(tl.int64) * NUM_HEADS * HEAD_DIM
        head_lane = tl.arange(0, BLOCK_HEADS).to(tl.int32)
        half_lane = tl.arange(0, BLOCK_ROPE_HALF).to(tl.int32)
        rope_half = ROPE_DIM // 2
        second_lane = rope_half + half_lane
        position = tl.load(positions_ptr + token, mask=valid_token, other=0).to(tl.int32)
        cache_base = position.to(tl.int64) * ROPE_DIM
        half_valid = half_lane < rope_half
        second_valid = second_lane < ROPE_DIM
        cos = tl.load(cache_ptr + cache_base + half_lane, mask=half_valid, other=0.0).to(tl.float32)
        sin = tl.load(
            cache_ptr + cache_base + rope_half + half_lane,
            mask=half_valid,
            other=0.0,
        ).to(tl.float32)
        qw_first = tl.load(q_weight_ptr + half_lane, mask=half_valid, other=0.0).to(tl.float32)
        qw_second = tl.load(q_weight_ptr + second_lane, mask=second_valid, other=0.0).to(tl.float32)
        kw_first = tl.load(k_weight_ptr + half_lane, mask=half_valid, other=0.0).to(tl.float32)
        kw_second = tl.load(k_weight_ptr + second_lane, mask=second_valid, other=0.0).to(tl.float32)

        for head_block in range(NUM_HEAD_BLOCKS):
            head = head_block * BLOCK_HEADS + head_lane
            valid_head = valid_token & (head < NUM_HEADS)
            q_base = input_token_base + head.to(tl.int64) * HEAD_DIM
            out_base = output_token_base + head.to(tl.int64) * HEAD_DIM
            first_offsets = q_base[:, None] + half_lane[None, :]
            second_offsets = q_base[:, None] + second_lane[None, :]
            out_first_offsets = out_base[:, None] + half_lane[None, :]
            out_second_offsets = out_base[:, None] + second_lane[None, :]
            first_mask = valid_head[:, None] & half_valid[None, :]
            second_mask = valid_head[:, None] & second_valid[None, :]

            q_first = tl.load(qkv_ptr + first_offsets, mask=first_mask, other=0.0).to(tl.float32)
            q_second = tl.load(qkv_ptr + second_offsets, mask=second_mask, other=0.0).to(tl.float32)
            k_first = tl.load(
                qkv_ptr + first_offsets + NUM_HEADS * HEAD_DIM,
                mask=first_mask,
                other=0.0,
            ).to(tl.float32)
            k_second = tl.load(
                qkv_ptr + second_offsets + NUM_HEADS * HEAD_DIM,
                mask=second_mask,
                other=0.0,
            ).to(tl.float32)

            q_sq_sum = tl.sum(q_first * q_first, axis=1) + tl.sum(q_second * q_second, axis=1)
            k_sq_sum = tl.sum(k_first * k_first, axis=1) + tl.sum(k_second * k_second, axis=1)

            if HAS_TAIL:
                tail_lane = tl.arange(0, BLOCK_TAIL).to(tl.int32)
                tail_dim = HEAD_DIM - ROPE_DIM
                tail_valid = tail_lane < tail_dim
                tail_mask = valid_head[:, None] & tail_valid[None, :]
                tail_offsets = q_base[:, None] + ROPE_DIM + tail_lane[None, :]
                q_tail = tl.load(qkv_ptr + tail_offsets, mask=tail_mask, other=0.0).to(tl.float32)
                k_tail = tl.load(
                    qkv_ptr + tail_offsets + NUM_HEADS * HEAD_DIM,
                    mask=tail_mask,
                    other=0.0,
                ).to(tl.float32)
                q_sq_sum += tl.sum(q_tail * q_tail, axis=1)
                k_sq_sum += tl.sum(k_tail * k_tail, axis=1)

            q_rstd = tl.rsqrt(q_sq_sum / HEAD_DIM + EPS)
            k_rstd = tl.rsqrt(k_sq_sum / HEAD_DIM + EPS)
            qn_first = q_first * q_rstd[:, None] * qw_first[None, :]
            qn_second = q_second * q_rstd[:, None] * qw_second[None, :]
            kn_first = k_first * k_rstd[:, None] * kw_first[None, :]
            kn_second = k_second * k_rstd[:, None] * kw_second[None, :]
            tl.store(
                q_out_ptr + out_first_offsets,
                qn_first * cos[None, :] - qn_second * sin[None, :],
                mask=first_mask,
            )
            tl.store(
                q_out_ptr + out_second_offsets,
                qn_second * cos[None, :] + qn_first * sin[None, :],
                mask=second_mask,
            )
            tl.store(
                k_out_ptr + out_first_offsets,
                kn_first * cos[None, :] - kn_second * sin[None, :],
                mask=first_mask,
            )
            tl.store(
                k_out_ptr + out_second_offsets,
                kn_second * cos[None, :] + kn_first * sin[None, :],
                mask=second_mask,
            )

            if HAS_TAIL:
                qw_tail = tl.load(
                    q_weight_ptr + ROPE_DIM + tail_lane,
                    mask=tail_lane < tail_dim,
                    other=0.0,
                ).to(tl.float32)
                kw_tail = tl.load(
                    k_weight_ptr + ROPE_DIM + tail_lane,
                    mask=tail_lane < tail_dim,
                    other=0.0,
                ).to(tl.float32)
                out_tail_offsets = out_base[:, None] + ROPE_DIM + tail_lane[None, :]
                tl.store(
                    q_out_ptr + out_tail_offsets,
                    q_tail * q_rstd[:, None] * qw_tail[None, :],
                    mask=tail_mask,
                )
                tl.store(
                    k_out_ptr + out_tail_offsets,
                    k_tail * k_rstd[:, None] * kw_tail[None, :],
                    mask=tail_mask,
                )

            if WRITE_V:
                full_lane = tl.arange(0, BLOCK_D).to(tl.int32)
                full_valid = full_lane < HEAD_DIM
                full_mask = valid_head[:, None] & full_valid[None, :]
                value = tl.load(
                    qkv_ptr + q_base[:, None] + 2 * NUM_HEADS * HEAD_DIM + full_lane[None, :],
                    mask=full_mask,
                    other=0.0,
                )
                tl.store(v_out_ptr + out_base[:, None] + full_lane[None, :], value, mask=full_mask)


def _metadata_error(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    v_out: torch.Tensor | None,
    *,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    out_dtype: torch.dtype | None,
) -> str | None:
    if not isinstance(qkv, torch.Tensor):
        return "qkv must be torch.Tensor"
    if qkv.dim() != 2:
        return "qkv must be 2-D [T, 3*N*D]"
    if qkv.dtype not in _SUPPORTED_DTYPES:
        return "qkv dtype must be bfloat16 or float16"
    if not qkv.is_contiguous():
        return "qkv must be contiguous"
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or not 1 <= num_heads <= _MAX_HEADS:
        return "num_heads must be in [1, 128]"
    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        return "head_dim must be an integer"
    if not _MIN_HEAD_DIM <= head_dim <= _MAX_HEAD_DIM:
        return "head_dim must be in [96, 256]"
    if head_dim % 8 != 0:
        return "head_dim must be divisible by 8"
    if isinstance(rope_dim, bool) or not isinstance(rope_dim, int):
        return "rope_dim must be an integer"
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2 != 0:
        return "rope_dim must be positive, even and no greater than head_dim"
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        return "eps must be a finite positive number"
    if qkv.shape[0] > _MAX_TOKENS:
        return "num_tokens exceeds 262144"
    if qkv.shape[1] != 3 * num_heads * head_dim:
        return "qkv width does not match 3*num_heads*head_dim"
    if q_weight is None:
        return "q_weight is required and must have shape [head_dim]"
    if k_weight is None:
        return "k_weight is required and must have shape [head_dim]"
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if not isinstance(weight, torch.Tensor):
            return f"{name} must be torch.Tensor"
        if weight.dim() != 1 or weight.shape[0] != head_dim:
            return f"{name} must have shape [head_dim]"
        if weight.dtype != qkv.dtype:
            return "weight dtype must equal qkv dtype"
        if not weight.is_contiguous():
            return "weights must be contiguous"
        if weight.device != qkv.device:
            return "weights must be on the same device as qkv"
    if not isinstance(cos_sin_cache, torch.Tensor):
        return "cos_sin_cache must be torch.Tensor"
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] != rope_dim:
        return "cos_sin_cache must have shape [maxPos, rope_dim]"
    if cos_sin_cache.dtype != qkv.dtype or not cos_sin_cache.is_contiguous():
        return "cos_sin_cache dtype must equal qkv dtype and be contiguous"
    if cos_sin_cache.device != qkv.device:
        return "cos_sin_cache must be on the same device as qkv"
    if not isinstance(positions, torch.Tensor):
        return "positions must be torch.Tensor"
    if positions.dim() != 1 or positions.shape[0] != qkv.shape[0]:
        return "positions must have shape [T]"
    if positions.dtype != torch.int32 or not positions.is_contiguous():
        return "positions must be contiguous int32"
    if positions.device != qkv.device:
        return "positions must be on the same device as qkv"
    if out_dtype is not None and out_dtype != qkv.dtype:
        return "out_dtype must be None or equal to qkv dtype"
    if v_out is not None:
        if not isinstance(v_out, torch.Tensor):
            return "v_out must be torch.Tensor or None"
        expected = (qkv.shape[0], num_heads, head_dim)
        if tuple(v_out.shape) != expected:
            return "v_out must have shape [T, N, D]"
        if v_out.dtype != qkv.dtype:
            return "v_out dtype must equal qkv dtype"
        if not v_out.is_contiguous():
            return "v_out must be contiguous"
        if v_out.device != qkv.device:
            return "v_out device must equal qkv device"
    return None


def validate_inputs(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float = 1e-5,
    v_out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    check_values: bool = True,
) -> None:
    """Validate the eager contract, including optional value-domain checks."""

    error = _metadata_error(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        v_out,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        out_dtype=out_dtype,
    )
    if error is not None:
        raise ParametersInvalid(error)
    if not check_values:
        return
    for name, tensor in (
        ("qkv", qkv),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos_sin_cache", cos_sin_cache),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ParametersInvalid(f"{name} contains NaN or Inf")
    if positions.numel() != 0:
        if bool((positions < 0).any()) or bool((positions >= cos_sin_cache.shape[0]).any()):
            raise ParametersInvalid("positions must satisfy 0 <= p < maxPos")


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


validate_split_qkv_norm_rope_concat_inputs = validate_inputs


if _HAS_TRITON:

    def _launch(
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        *,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        eps: float,
        v_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        num_tokens = qkv.shape[0]
        output_shape = (num_tokens, num_heads, head_dim)
        q_out = torch.empty(output_shape, dtype=qkv.dtype, device=qkv.device)
        k_out = torch.empty(output_shape, dtype=qkv.dtype, device=qkv.device)
        if num_tokens == 0:
            return q_out, k_out, v_out

        block_d = _next_power_of_two(head_dim)
        block_rope_half = _next_power_of_two(rope_dim // 2)
        tail_dim = head_dim - rope_dim
        block_tail = _next_power_of_two(max(1, tail_dim))
        block_heads = 16 if head_dim <= 128 else 8
        num_head_blocks = triton.cdiv(num_heads, block_heads)
        write_v = v_out is not None
        v_kernel_out = v_out if write_v else q_out

        token_offset = 0
        while token_offset < num_tokens:
            grid_tokens = min(_MAX_GRID, num_tokens - token_offset)
            _split_qkv_norm_rope_kernel[(grid_tokens,)](
                qkv,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                v_kernel_out,
                num_tokens,
                token_offset,
                NUM_HEADS=num_heads,
                HEAD_DIM=head_dim,
                ROPE_DIM=rope_dim,
                ROW_WIDTH=3 * num_heads * head_dim,
                BLOCK_D=block_d,
                BLOCK_ROPE_HALF=block_rope_half,
                BLOCK_TAIL=block_tail,
                BLOCK_HEADS=block_heads,
                NUM_HEAD_BLOCKS=num_head_blocks,
                HAS_TAIL=tail_dim != 0,
                WRITE_V=write_v,
                EPS=eps,
            )
            token_offset += grid_tokens
        return q_out, k_out, v_out

else:

    def _launch(
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        *,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        eps: float,
        v_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:  # pragma: no cover
        raise RuntimeError("split_qkv_norm_rope_concat requires Triton-Ascend.")


def _require_triton_ascend() -> None:
    if not _TRITON_ON_ASCEND:
        raise RuntimeError("split_qkv_norm_rope_concat requires an Ascend NPU with Triton-Ascend.")


@torch.library.custom_op("mindiesd::split_qkv_norm_rope_concat", mutates_args=())
def _split_qkv_norm_rope_concat_op(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    validate_values: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_inputs(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        check_values=validate_values,
    )
    _require_triton_ascend()
    v_out = torch.empty(
        (qkv.shape[0], num_heads, head_dim),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    return _launch(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        v_out=v_out,
    )


@torch.library.custom_op("mindiesd::split_qkv_norm_rope_concat_no_v", mutates_args=())
def _split_qkv_norm_rope_concat_no_v_op(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    validate_values: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_inputs(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        check_values=validate_values,
    )
    _require_triton_ascend()
    q_out, k_out, _ = _launch(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        v_out=None,
    )
    return q_out, k_out


@torch.library.custom_op("mindiesd::split_qkv_norm_rope_concat_out", mutates_args=("v_out",))
def _split_qkv_norm_rope_concat_out_op(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    validate_values: bool,
    v_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_inputs(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        v_out=v_out,
        check_values=validate_values,
    )
    _require_triton_ascend()
    q_out, k_out, _ = _launch(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        v_out=v_out,
    )
    return q_out, k_out


def split_qkv_norm_rope_concat(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float = 1e-5,
    validate_values: bool = False,
    v_out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the fused SplitQkvNormRopeConcat operator.

    Args:
        qkv: Packed QKV input ``[T, 3*N*D]`` in bf16 or fp16.
        q_weight: Per-dimension Q RMSNorm weight ``[D]``.
        k_weight: Per-dimension K RMSNorm weight ``[D]``.
        cos_sin_cache: ``[maxPos, R]`` cache containing ``concat(cos, sin)``.
        positions: int32 positions ``[T]``; each value must satisfy ``0 <= p < maxPos``.
        num_heads: Number of Q/K/V heads ``N``.
        head_dim: Head dimension ``D``.
        rope_dim: Partial NeoX RoPE dimension ``R``.
        eps: RMSNorm epsilon.
        v_out: Optional contiguous ``[T, N, D]`` V destination. When omitted,
            the result returns ``None`` for V and the caller can consume the
            corresponding packed-QKV view without an extra copy.
        validate_values: Check finite tensor values and position bounds before
            launch. This is a debug option and is disabled by default to avoid
            scanning the full input and cache in hot paths.
        out_dtype: Optional output dtype. Only ``None`` or the input dtype is
            accepted.

    Returns:
        ``(q_out, k_out, v_out_or_none)``.
    """

    validate_inputs(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=eps,
        v_out=v_out,
        out_dtype=out_dtype,
        check_values=False,
    )
    if v_out is None:
        q_out, k_out = torch.ops.mindiesd.split_qkv_norm_rope_concat_no_v.default(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            num_heads,
            head_dim,
            rope_dim,
            eps,
            validate_values,
        )
        return q_out, k_out, None

    q_out, k_out = torch.ops.mindiesd.split_qkv_norm_rope_concat_out.default(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads,
        head_dim,
        rope_dim,
        eps,
        validate_values,
        v_out,
    )
    return q_out, k_out, v_out


triton_split_qkv_norm_rope_concat = split_qkv_norm_rope_concat


@_split_qkv_norm_rope_concat_op.register_fake
def _split_qkv_norm_rope_concat_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    validate_values: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos_sin_cache, positions, rope_dim, eps, validate_values
    shape = (qkv.shape[0], num_heads, head_dim)
    return (
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
    )


@_split_qkv_norm_rope_concat_no_v_op.register_fake
def _split_qkv_norm_rope_concat_no_v_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    validate_values: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos_sin_cache, positions, rope_dim, eps, validate_values
    shape = (qkv.shape[0], num_heads, head_dim)
    return (
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
    )


@_split_qkv_norm_rope_concat_out_op.register_fake
def _split_qkv_norm_rope_concat_out_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    validate_values: bool,
    v_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos_sin_cache, positions, rope_dim, eps, validate_values, v_out
    shape = (qkv.shape[0], num_heads, head_dim)
    return (
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
    )


__all__ = [
    "split_qkv_norm_rope_concat",
    "triton_split_qkv_norm_rope_concat",
    "validate_inputs",
    "validate_split_qkv_norm_rope_concat_inputs",
]
