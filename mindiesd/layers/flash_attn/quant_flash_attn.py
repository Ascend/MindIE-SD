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


"""Public quantized attention entry, shared validation and precision routing."""

import torch
from .common import _get_bnsd_shape
from .fused_infer_attention_score import _fp8_attention_forward, _mxfp8_attention_forward


_SUPPORTED_PRECISIONS = ("fp8", "mxfp8")


def quant_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    precision: str = "fp8",
    *,
    layout: str = "BNSD",
    scale: float | None = None,
    q_rot: torch.Tensor | None = None,
    k_rot: torch.Tensor | None = None,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    **kwargs,
) -> torch.Tensor:
    """Compute quantized attention from floating-point Q/K/V on NPU.

    Args:
        query (torch.Tensor):
            Unquantized query, shaped [B, Nq, Sq, D] for BNSD or
            [B, Sq, Nq, D] for BSND. All dimensions must be non-empty and
            B must be 1 for FP8. Nq must be divisible by the number of KV heads.
            Accepted input dtypes are float16, bfloat16 and float32; execution
            requires an NPU/operator version supporting the input dtype.
        key (torch.Tensor):
            Unquantized key, shaped [B, Nkv, Skv, D] for BNSD or
            [B, Skv, Nkv, D] for BSND. Must share query's batch size,
            head dimension, device and dtype. FP8 and MXFP8 permit Skv to
            differ from Sq.
        value (torch.Tensor):
            Unquantized value with the same shape, device and dtype as key.
        precision (str, optional, defaults to "fp8"):
            Quantization precision. Supported values are "fp8" and "mxfp8".
            Other values, including "float", raise ValueError. For
            non-quantized attention use mindiesd.attention_forward instead.
        layout (str, optional, defaults to "BNSD"):
            Shared Q/K/V and output layout: "BNSD" or "BSND". B is batch,
            N is head count, S is sequence length and D is head dimension.
        scale (float, optional, defaults to None):
            Multiplier applied to Q @ K.transpose(-1, -2) before softmax.
            Defaults to D ** -0.5 when scale is None.
        q_rot (torch.Tensor, optional, defaults to None):
            Optional [D, D] matrix applied as query @ q_rot before quantization.
            Must share query's device and dtype. None leaves query unchanged.
        k_rot (torch.Tensor, optional, defaults to None):
            Optional [D, D] matrix applied as key @ k_rot before quantization.
            Must share key's device and dtype. Independent of q_rot: either
            may be omitted. The caller supplies rotations preserving the
            desired attention semantics; this function does not generate them.
        pre_tokens (int, optional, defaults to 2147483647):
            Maximum number of preceding tokens visible to a query, passed
            unchanged to FIA. The default leaves the preceding window unlimited.
        next_tokens (int, optional, defaults to 2147483647):
            Maximum number of following tokens visible to a query, passed
            unchanged to FIA. The default leaves the following window unlimited.
            With equal Q/KV lengths, 0 selects a causal right boundary.
        kwargs:
            Precision-specific options, parsed at this entry point. Unknown
            options raise TypeError instead of being forwarded or ignored.
            fp8_fa_mode (str or FP8FAMode, optional, defaults to None):
                Block-FP8 mode. None selects HIGH_PRECISION: Q/K/V sequence
                blocks are 128/256/256 and V column blocks are 128.
                C8V16_TILING512 uses sequence blocks 128/256/512, V column
                blocks 64 and FIA inner_precise=4. The corresponding FP8FAMode
                enum values are also accepted; hardware support is required.

    Returns:
        torch.Tensor: Floating-point attention output with query's layout,
        shape and dtype. Internal sequence padding is cropped from the output.

    Raises:
        ValueError: Unsupported precision/layout, invalid
            Q/K/V shapes/dtypes/devices, or incompatible rotation matrices.
        TypeError: An unknown keyword argument is supplied.
        ParametersInvalid: The FP8 mode is unsupported.

    Native execution errors propagate to the caller without a floating-point
    retry. Non-quantized BNSD callers should use attention_forward(q, k, v,
    scale=scale, head_first=True); BSND callers should set head_first=False.
    """
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported quantized attention precision: {precision!r}; supported precisions: {_SUPPORTED_PRECISIONS}. "
            "Use mindiesd.attention_forward for non-quantized attention."
        )
    fp8_fa_mode = kwargs.pop("fp8_fa_mode", None) if precision == "fp8" else None
    if kwargs:
        raise TypeError(f"Unexpected options for {precision} quantized attention: {', '.join(sorted(kwargs))}.")
    _, _, _, head_dim = _validate_quant_attention_inputs(query, key, value, layout=layout)
    _validate_rotation(query, q_rot, "q_rot")
    _validate_rotation(key, k_rot, "k_rot")
    scale = head_dim**-0.5 if scale is None else scale
    if precision == "mxfp8":
        return _mxfp8_attention_forward(
            query,
            key,
            value,
            layout=layout,
            scale=scale,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            q_rot=q_rot,
            k_rot=k_rot,
        )
    return _fp8_attention_forward(
        query,
        key,
        value,
        layout=layout,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        mode=fp8_fa_mode,
        q_rot=q_rot,
        k_rot=k_rot,
    )


def _validate_quant_attention_inputs(query, key, value, *, layout):
    """Validate floating-point Q/K/V before rotation and block quantization."""
    batch, heads, sequence, dim = _get_bnsd_shape(query, layout)
    kv_batch, kv_heads, _, kv_dim = _get_bnsd_shape(key, layout)
    _get_bnsd_shape(value, layout)
    for tensor in (query, key, value):
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Q/K/V must be floating-point inputs, not pre-quantized tensors.")
    if query.device != key.device or key.device != value.device:
        raise ValueError("Q/K/V must be on the same device.")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("Q/K/V must have the same dtype.")
    if key.shape != value.shape or batch != kv_batch or dim != kv_dim:
        raise ValueError("Q/K/V batch and head dimensions must match; K/V shapes must match.")
    if heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count.")
    return batch, heads, sequence, dim


def _validate_rotation(tensor, rotation, name):
    if rotation is None:
        return
    expected = (tensor.shape[-1], tensor.shape[-1])
    if not isinstance(rotation, torch.Tensor) or rotation.shape != expected:
        raise ValueError(f"{name} must be a tensor with shape {expected}.")
    if rotation.device != tensor.device or rotation.dtype != tensor.dtype:
        raise ValueError(f"{name} must match its input device and dtype.")
