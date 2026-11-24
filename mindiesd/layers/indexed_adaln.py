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

"""Indexed AdaLayerNorm: fuse scale/shift table gather with AdaLayerNorm.

out = LayerNorm(x) * (1 + scale_table[:, mod_index]) + shift_table[:, mod_index]

The gather is not materialized: the Ascend C kernel selects the K-row table
inside the LayerNorm epilogue. Gate modulation is intentionally out of scope.
"""

import weakref

import torch

from ..utils import ParametersInvalid
from . import _custom_ops as ops

# Cache validated mod_index tensors by identity/version. Reading min/max needs
# a device sync, and FLUX.3 passes the same tensor to all blocks of one
# transformer invocation, so validation should happen once per tensor, not
# once per block.
_INDEX_VALIDATION_CACHE = {}


def _validate_mod_index_range(mod_index, table_k):
    version = getattr(mod_index, "_version", 0)
    key = id(mod_index)
    entry = _INDEX_VALIDATION_CACHE.get(key)
    if entry is not None:
        tensor_ref, cached_version, cached_k = entry
        if tensor_ref() is mod_index and cached_version == version and cached_k == table_k:
            return

    if mod_index.numel() != 0:
        min_index = int(mod_index.min().item())
        max_index = int(mod_index.max().item())
        if min_index < 0 or max_index >= table_k:
            raise ParametersInvalid(
                f"mod_index values must be in [0, {table_k}), but got min={min_index}, max={max_index}."
            )

    if len(_INDEX_VALIDATION_CACHE) > 256:
        _INDEX_VALIDATION_CACHE.clear()
    try:
        tensor_ref = weakref.ref(mod_index)
    except TypeError:
        return
    _INDEX_VALIDATION_CACHE[key] = (tensor_ref, version, table_k)
    weakref.finalize(mod_index, _INDEX_VALIDATION_CACHE.pop, key, None)


def _check_inputs(layernorm, x, scale_table, shift_table, mod_index, validate_mod_index):
    if not isinstance(layernorm, torch.nn.LayerNorm):
        raise ParametersInvalid(f"The type of input layernorm must be torch.nn.LayerNorm, but got {type(layernorm)}.")
    for name, tensor in (
        ("x", x),
        ("scale_table", scale_table),
        ("shift_table", shift_table),
        ("mod_index", mod_index),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ParametersInvalid(f"The data type of input {name} must be torch.Tensor.")
    if x.dim() != 3:
        raise ParametersInvalid(f"x must be 3D [B, T, H], but got {x.dim()}D.")
    if scale_table.dim() != 3 or shift_table.dim() != 3:
        raise ParametersInvalid("scale_table/shift_table must be 3D [B, K, H].")
    if mod_index.dim() != 1:
        raise ParametersInvalid(f"mod_index must be 1D [T], but got {mod_index.dim()}D.")
    if x.shape[0] != scale_table.shape[0] or x.shape[0] != shift_table.shape[0]:
        raise ParametersInvalid("Batch size of x and scale/shift tables must match.")
    if x.shape[2] != scale_table.shape[2] or x.shape[2] != shift_table.shape[2]:
        raise ParametersInvalid("Hidden size of x and scale/shift tables must match.")
    if scale_table.shape[1] != shift_table.shape[1]:
        raise ParametersInvalid("scale_table and shift_table must have the same K.")
    if scale_table.shape[1] <= 0:
        raise ParametersInvalid(f"K must be positive, but got {scale_table.shape[1]}.")
    if mod_index.numel() != x.shape[1]:
        raise ParametersInvalid(f"mod_index length must equal T, but got {mod_index.numel()} vs {x.shape[1]}.")

    normalized_shape = tuple(layernorm.normalized_shape)
    if len(normalized_shape) != 1 or normalized_shape[0] != x.shape[-1]:
        raise ParametersInvalid(
            f"layernorm.normalized_shape must be ({x.shape[-1]},) to match the fused "
            f"last-dim LayerNorm, but got {normalized_shape}."
        )

    supported_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if x.dtype not in supported_dtypes:
        raise ParametersInvalid(f"x must be BF16, FP16 or FP32, but got {x.dtype}.")
    if scale_table.dtype != x.dtype or shift_table.dtype != x.dtype:
        raise ParametersInvalid("scale_table/shift_table dtype must match x.")
    if mod_index.dtype != torch.int64:
        raise ParametersInvalid(f"mod_index must be INT64, but got {mod_index.dtype}.")

    # The K-row table DataCopyPad packs one row per 32B-aligned block. For
    # 2-byte dtypes this requires H % 16 == 0; for FP32 H % 8 == 0.
    row_bytes = x.shape[-1] * x.element_size()
    if row_bytes % 32 != 0:
        raise ParametersInvalid(
            f"H * dtype_size must be a multiple of 32 bytes for the K-row table copy, "
            f"got H={x.shape[-1]}, dtype_size={x.element_size()}."
        )

    if layernorm.elementwise_affine:
        raise ParametersInvalid(
            "indexed_adaln currently supports elementwise_affine=False only. "
            "Apply a separate affine transform if needed."
        )

    if validate_mod_index:
        _validate_mod_index_range(mod_index, scale_table.shape[1])


def layernorm_scale_shift_indexed(
    layernorm, x, scale_table, shift_table, mod_index, fused=True, validate_mod_index=True
):
    """Fused ``LayerNorm(x) * (1 + scale_table[idx]) + shift_table[idx]``.

    Args:
        layernorm: ``torch.nn.LayerNorm`` with ``elementwise_affine=False``.
        x: activation, ``[B, T, H]``, BF16/FP16/FP32.
        scale_table: per-modality scale rows, ``[B, K, H]``, same dtype as x.
        shift_table: per-modality shift rows, ``[B, K, H]``, same dtype as x.
        mod_index: per-token table row, ``[T]``, INT64.
        fused: use the Ascend C kernel on NPU; ``False`` falls back to eager.
        validate_mod_index: validate ``mod_index`` is in ``[0, K)``. The result
            is cached per tensor/version, so reuse across blocks is cheap.

    Returns:
        Tensor of x's shape and dtype.
    """
    _check_inputs(layernorm, x, scale_table, shift_table, mod_index, validate_mod_index)

    if not fused or x.device.type != "npu" or not hasattr(torch.ops.mindiesd, "indexed_adaln"):
        scale = torch.index_select(scale_table, 1, mod_index)
        shift = torch.index_select(shift_table, 1, mod_index)
        return layernorm(x) * (1 + scale) + shift

    return ops.indexed_adaln(
        x=x,
        scale_table=scale_table,
        shift_table=shift_table,
        mod_index=mod_index,
        epsilon=layernorm.eps,
    )
