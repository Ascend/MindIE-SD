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
from .quant_attention_mxfp4 import _mxfp4_attention_forward
from .fused_infer_attention_score import _fp8_attention_forward, _mxfp8_attention_forward


_SUPPORTED_PRECISIONS = ("fp8", "mxfp8", "mxfp4")


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
            head dimension, device and dtype. Skv may differ from Sq.
            MXFP4 interprets K/V using layout_kv when provided.
        value (torch.Tensor):
            Unquantized value with the same shape, device and dtype as key.
        precision (str, optional, defaults to "fp8"):
            Quantization precision. Supported values are "fp8", "mxfp8" and "mxfp4".
            Other values, including "float", raise ValueError. For
            non-quantized attention use mindiesd.attention_forward instead.
        layout (str, optional, defaults to "BNSD"):
            Q layout; also the default K/V and output layout: "BNSD" or "BSND". B is batch,
            N is head count, S is sequence length and D is head dimension.
        scale (float, optional, defaults to None):
            Multiplier applied to Q @ K.transpose(-1, -2) before softmax.
            Defaults to D ** -0.5 when scale is None.
        q_rot (torch.Tensor, optional, defaults to None):
            Optional [D, D] matrix applied as query @ q_rot before quantization.
            Must share query's device; cast to query's dtype before use.
            None leaves query unchanged.
        k_rot (torch.Tensor, optional, defaults to None):
            Optional [D, D] matrix applied as key @ k_rot before quantization.
            Must share key's device; cast to key's dtype before use. Independent of q_rot: either
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

            MXFP4 options (only with precision="mxfp4"):
                Q/K/V sequence lengths are padded to multiples of 512 before
                    quantization. Native seqused_q/seqused_kv use these padded
                    lengths, including zero-padded KV positions in softmax.
                    The returned output is cropped to the original Q length.
                layout_kv/layout_out: BNSD or BSND; None uses layout.
                mxfp4_scale_alg/mxfp4_dst_type_max: explicit MX quantization
                    parameters; None retains the installed operator defaults.
                metadata: precomputed QFA metadata; None generates it.
                num_key_value_heads: must match K/V heads if provided.
                max_seqlen_q/max_seqlen_kv: native length bounds, default -1.
                mask_mode: native mask mode, default 0.
                win_left/win_right: None uses pre_tokens/next_tokens.
                block_table/sinks/attn_mask: optional native inputs, default None.
                return_softmax_lse: native LSE flag, default 0. This function
                    returns only the attention output; LSE is discarded.
                Optional inputs must satisfy the installed QFA contract.

    Returns:
        torch.Tensor: Floating-point output with query's dimensions and dtype,
        using layout_out for MXFP4 or query's layout otherwise. Internal sequence padding is cropped from the output.

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
    layout_kv = layout
    if precision == "mxfp4":
        layout_kv = kwargs.pop("layout_kv", None)
        layout_kv = layout if layout_kv is None else layout_kv
        layout_out = kwargs.pop("layout_out", None)
        layout_out = layout if layout_out is None else layout_out
        if layout_out not in ("BNSD", "BSND"):
            raise ValueError("layout_out must be BNSD or BSND.")
        mxfp4_scale_alg = kwargs.pop("mxfp4_scale_alg", None)
        mxfp4_dst_type_max = kwargs.pop("mxfp4_dst_type_max", None)
        metadata = kwargs.pop("metadata", None)
        num_key_value_heads = kwargs.pop("num_key_value_heads", None)
        max_seqlen_q = kwargs.pop("max_seqlen_q", -1)
        max_seqlen_kv = kwargs.pop("max_seqlen_kv", -1)
        mask_mode = kwargs.pop("mask_mode", 0)
        win_left = kwargs.pop("win_left", None)
        win_right = kwargs.pop("win_right", None)
        block_table = kwargs.pop("block_table", None)
        sinks = kwargs.pop("sinks", None)
        attn_mask = kwargs.pop("attn_mask", None)
        return_softmax_lse = kwargs.pop("return_softmax_lse", 0)
    if kwargs:
        raise TypeError(f"Unexpected options for {precision} quantized attention: {', '.join(sorted(kwargs))}.")
    _, _, _, head_dim = _validate_quant_attention_inputs(query, key, value, layout=layout, layout_kv=layout_kv)
    q_rot = _validate_rotation(query, q_rot, "q_rot")
    k_rot = _validate_rotation(key, k_rot, "k_rot")
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
    if precision == "mxfp4":
        actual_kv_heads = key.shape[1 if layout_kv == "BNSD" else 2]
        if num_key_value_heads is not None and (
            isinstance(num_key_value_heads, bool)
            or not isinstance(num_key_value_heads, int)
            or num_key_value_heads != actual_kv_heads
        ):
            raise ValueError("num_key_value_heads must be an integer matching K/V tensor head count.")
        return _mxfp4_attention_forward(
            query,
            key,
            value,
            layout=layout,
            layout_kv=layout_kv,
            layout_out=layout_out,
            scale=scale,
            q_rot=q_rot,
            k_rot=k_rot,
            win_left=pre_tokens if win_left is None else win_left,
            win_right=next_tokens if win_right is None else win_right,
            mxfp4_scale_alg=mxfp4_scale_alg,
            mxfp4_dst_type_max=mxfp4_dst_type_max,
            metadata=metadata,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            mask_mode=mask_mode,
            block_table=block_table,
            sinks=sinks,
            attn_mask=attn_mask,
            return_softmax_lse=return_softmax_lse,
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


def _validate_quant_attention_inputs(query, key, value, *, layout, layout_kv=None):
    """Validate floating-point Q/K/V before rotation and block quantization."""
    layout_kv = layout if layout_kv is None else layout_kv
    batch, heads, sequence, dim = _get_bnsd_shape(query, layout)
    kv_batch, kv_heads, _, kv_dim = _get_bnsd_shape(key, layout_kv)
    _get_bnsd_shape(value, layout_kv)
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
    """Validate an optional rotation and return it cast to the input dtype."""
    if rotation is None:
        return None
    if not isinstance(rotation, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, but got {type(rotation).__name__}.")
    expected = (tensor.shape[-1], tensor.shape[-1])
    if rotation.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, but got {tuple(rotation.shape)}.")
    if rotation.device != tensor.device:
        raise ValueError(
            f"{name} must match its input device, "
            f"but got {name}.device={rotation.device} and input.device={tensor.device}."
        )
    return rotation.to(dtype=tensor.dtype)
