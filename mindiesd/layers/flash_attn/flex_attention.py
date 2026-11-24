#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Local CATLASS FlexAttention and block-causal mask construction."""

from collections.abc import Callable, Sequence
import importlib
import math
from types import ModuleType

import torch

from ...utils.get_platform import is_a5_device


_REQUIRED_HEAD_DIM = 128


def check_flex_attention_device(device: torch.device) -> None:
    """Check the selected NPU before loading optional DSL modules or communicating."""
    if device.type != "npu":
        raise ValueError("FlexAttention requires NPU tensors.")
    if not is_a5_device():
        name = torch.npu.get_device_name()
        raise NotImplementedError(f"CATLASS DSL FlexAttention requires Ascend 950; got {name}.")


def _load_dsl(device: torch.device) -> ModuleType:
    check_flex_attention_device(device)
    try:
        return importlib.import_module(f"{__package__}.flex_attention_dsl")
    except ModuleNotFoundError as exc:
        if exc.name not in ("catlass", "flash_attn_npu_4", "flash_attn_npu_4_950"):
            raise
        raise ImportError(
            "CATLASS DSL FlexAttention requires ascend-catlass-dsl and a flash-attn-npu build "
            "with the FA4 CATLASS DSL backend for Ascend 950."
        ) from exc


def _map_identity(tensor: torch.Tensor | None) -> tuple | None:
    if tensor is None:
        return None
    return id(tensor), None if tensor.is_inference() else tensor._version


class FlexAttentionMask:
    """Logical mask and optional score rule, with sparse metadata cached per request.

    Native @tla.jit callbacks receive global positions and heads. The mask
    signature is (b, h, q, kv, info, aux, scalars), and the score signature
    prepends the scaled score. Auxiliary inputs and explicit position maps
    must stay immutable while the mask is reused. Clear it on all ranks or
    construct a new mask when request geometry or auxiliary contents change.
    """

    def __init__(
        self,
        mask_mod: Callable | None = None,
        score_mod: Callable | None = None,
        *,
        aux_tensors: Sequence[torch.Tensor] = (),
        aux_scalars: Sequence[bool | int | float] = (),
        batch_shared: bool = False,
        head_shared: bool = False,
    ) -> None:
        if any(rule is not None and not callable(rule) for rule in (mask_mod, score_mod)):
            raise TypeError("mask_mod and score_mod must be callable or None.")
        if not isinstance(batch_shared, bool) or not isinstance(head_shared, bool):
            raise TypeError("batch_shared and head_shared must be booleans.")
        self.mask_mod = mask_mod
        self.score_mod = score_mod
        self.aux_tensors = tuple(aux_tensors)
        self.aux_scalars = tuple(aux_scalars)
        self.batch_shared = batch_shared
        self.head_shared = head_shared
        self._block_causal = False
        self._entries: dict[tuple, tuple] = {}
        self._validated_plans: set[tuple] = set()

    def clear(self) -> None:
        """Release prepared mask tensors and validated parallel plans."""
        self._entries.clear()
        self._validated_plans.clear()

    def _validate(self, batch_size: int) -> None:
        """Reject host-visible callback errors before distributed exchanges."""
        for callback in (self.mask_mod, self.score_mod):
            if callback is not None and not getattr(callback, "_tla_jit", False):
                raise TypeError("CATLASS DSL mask_mod and score_mod must use @tla.jit.")
        if self._block_causal and not self.batch_shared and self.aux_tensors[1].shape[0] != batch_size:
            raise ValueError("valid_tokens batch size must match the query batch size.")

    def _prepare(self, dsl, q_map, kv_map, heads, *, batch_size, cache_key, owners):
        key = (cache_key, batch_size, heads.numel(), q_map.numel(), kv_map.numel(), q_map.device)
        if key not in self._entries:
            if torch.any(heads < 0).item():
                raise ValueError("head_indices must contain non-negative logical query-head IDs.")
            prepared = dsl.prepare(self, q_map, kv_map, heads, batch_size)
            # Keep the source maps alive while their identities are cache keys.
            self._entries[key] = prepared, owners
        return self._entries[key][0]


def make_block_causal_mask(
    spans: Sequence[tuple[int, int]],
    seq_len: int,
    *,
    device: torch.device | str,
    valid_tokens: torch.Tensor | None = None,
) -> FlexAttentionMask:
    """Allow causal attention plus bidirectional attention within each span.

    Spans are ordered, non-overlapping global half-open intervals. Optional
    boolean validity has shape [S] or [B, S] and masks both queries and keys.
    """
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer.")
    block_ids = torch.full((seq_len,), -1, dtype=torch.int32, device="cpu")
    cursor = 0
    for block_id, (start, end) in enumerate(spans):
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not cursor <= start < end <= seq_len
        ):
            raise ValueError("spans must be sorted, non-overlapping and within seq_len.")
        block_ids[start:end] = block_id
        cursor = end

    if valid_tokens is None:
        valid = torch.ones((1, seq_len), dtype=torch.int32, device=device)
    else:
        if (
            not isinstance(valid_tokens, torch.Tensor)
            or valid_tokens.dtype != torch.bool
            or valid_tokens.ndim not in (1, 2)
            or valid_tokens.shape[-1] != seq_len
            or not valid_tokens.numel()
        ):
            raise ValueError("valid_tokens must be a non-empty boolean tensor of shape [S] or [B, S].")
        valid = valid_tokens.reshape(-1, seq_len).to(device=device, dtype=torch.int32)

    mask = FlexAttentionMask(
        aux_tensors=(block_ids.to(device), valid),
        batch_shared=valid.shape[0] == 1,
        head_shared=True,
    )
    mask._block_causal = True
    return mask


def _validate_query(query: torch.Tensor) -> None:
    """Check the query device and eager execution before backend work."""
    if not isinstance(query, torch.Tensor):
        raise ValueError("query must be a four-dimensional Tensor.")
    check_flex_attention_device(query.device)
    if torch.compiler.is_compiling():
        raise ValueError("FlexAttention must run outside an outer torch.compile region.")


def _validate_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layout: str) -> None:
    """Check common local tensor and execution contracts before communication."""
    if layout not in ("BSND", "BNSD"):
        raise ValueError("layout must be 'BSND' or 'BNSD'.")
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(f"{name} must be a four-dimensional Tensor.")
        if min(tensor.shape) <= 0:
            raise ValueError(f"{name} dimensions must be non-empty.")
        if tensor.device != query.device or tensor.dtype != query.dtype:
            raise ValueError("Q/K/V must have the same device and dtype.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlexAttention supports FP16 and BF16 inputs.")
    if torch.is_grad_enabled():
        raise ValueError("FlexAttention is inference-only; use torch.no_grad() or torch.inference_mode().")


def _validate_map(name: str, tensor: torch.Tensor | None, length: int, device: torch.device) -> None:
    """Check coordinate metadata without copying or scanning its contents."""
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tensor.numel() != length:
        raise ValueError(f"{name} must be a one-dimensional Tensor of length {length}.")
    if tensor.dtype not in (torch.int32, torch.int64) or tensor.device != device:
        raise ValueError(f"{name} must be an int32/int64 Tensor on {device}.")


def _validate_options(mask, query, scale, enable_gqa, return_lse):
    """Validate options shared by the local and distributed entry points."""
    if not isinstance(mask, FlexAttentionMask):
        raise TypeError("mask must be a FlexAttentionMask.")
    mask._validate(query.shape[0])
    if not isinstance(enable_gqa, bool) or not isinstance(return_lse, bool):
        raise TypeError("enable_gqa and return_lse must be booleans.")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (float, int))):
        raise TypeError("scale must be a real number or None.")
    if scale is not None and (abs(scale) > torch.finfo(torch.float32).max or not math.isfinite(scale)):
        raise ValueError("scale must be finite and within the FP32 range.")


def _validate_qkv_shapes(query, key, value, enable_gqa):
    """Validate Q/K/V in BSND layout and return their head counts."""
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V batch sizes, K/V sequence/head counts and Q/K head dimensions must match.")
    if any(tensor.shape[-1] != _REQUIRED_HEAD_DIM for tensor in (query, key, value)):
        raise ValueError(f"CATLASS DSL FlexAttention requires head_dim={_REQUIRED_HEAD_DIM} for Q/K/V.")
    q_heads, kv_heads = query.shape[2], key.shape[2]
    if q_heads != kv_heads and (not enable_gqa or q_heads % kv_heads):
        raise ValueError("Different Q/KV head counts require enable_gqa=True and an integral Q/KV head ratio.")
    return q_heads, kv_heads


def _validate_positions(query, key, query_positions, key_positions, head_indices):
    """Check maps in the caller's layout-normalized coordinate space."""
    _validate_map("query_positions", query_positions, query.shape[1], query.device)
    _validate_map("key_positions", key_positions, key.shape[1], query.device)
    _validate_map("head_indices", head_indices, query.shape[2], query.device)
    if head_indices is not None and torch.any(head_indices < 0).item():
        raise ValueError("head_indices must contain non-negative logical query-head IDs.")


def _format_output(output, lse, layout):
    """Restore the caller's output layout and return contiguous FP32 LSE."""
    if layout == "BNSD":
        output = output.transpose(1, 2)
    output = output.contiguous()
    return (output, lse.contiguous().float()) if lse is not None else output


def flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: FlexAttentionMask | None = None,
    scale: float | None = None,
    layout: str = "BSND",
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    head_indices: torch.Tensor | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run local inference attention with an optional logical CATLASS mask."""
    _validate_query(query)
    _validate_inputs(query, key, value, layout)
    mask = FlexAttentionMask() if mask is None else mask
    _validate_options(mask, query, scale, enable_gqa, return_lse)
    if layout == "BNSD":
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    _validate_qkv_shapes(query, key, value, enable_gqa)
    dsl = _load_dsl(query.device)
    _validate_positions(query, key, query_positions, key_positions, head_indices)
    maps = (query_positions, key_positions, head_indices)
    owners = tuple(tensor for tensor in maps if tensor is not None)
    cache_key = ("local", tuple(query.shape), tuple(key.shape), *(_map_identity(tensor) for tensor in maps))
    if query_positions is None:
        query_positions = torch.arange(query.shape[1], device=query.device)
    if key_positions is None:
        key_positions = torch.arange(key.shape[1], device=query.device)
    if head_indices is None:
        head_indices = torch.arange(query.shape[2], device=query.device)
    prepared = mask._prepare(
        dsl,
        query_positions.to(torch.int64),
        key_positions.to(torch.int64),
        head_indices,
        batch_size=query.shape[0],
        cache_key=cache_key,
        owners=owners,
    )
    result = dsl.attention(query, key, value, prepared, scale, return_lse)
    output, lse = result if return_lse else (result, None)
    return _format_output(output, lse, layout)
