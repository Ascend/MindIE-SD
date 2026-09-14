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

import torch
import torch_npu

from .. import register_ops as _mindiesd_register_ops  # noqa: F401
from ...utils.exception import ParametersInvalid
from ..quant.block_quant import _fa_block_quant_preprocess


_SUPPORTED_LAYOUTS = ("BNSD", "BSND", "BSH")
_DTYPE_KWARGS = (
    "query_dtype",
    "key_dtype",
    "value_dtype",
    "query_rope_dtype",
    "key_rope_dtype",
    "key_shared_prefix_dtype",
    "value_shared_prefix_dtype",
    "dequant_scale_query_dtype",
    "dequant_scale_key_dtype",
    "dequant_scale_value_dtype",
    "dequant_scale_key_rope_dtype",
)
_IGNORED_COMPAT_KWARGS = {"quant_scale_p"}
_TORCH_NPU_CANN_DTYPE_NAMES = (
    ("hifloat8", 290),
    ("float8_e8m0fnu", 293),
    ("float4_e2m1fn_x2", 296),
)


FA_PER_BLOCK_QUANT_MODE = 7
FA_V512_D64_QUANT_MODE = 12
FA_C8V16_INNER_PRECISE = 4
_FP8_FA_MODE_SPEC = {
    "HIGH_PRECISION": {
        "q_block": 128,
        "k_block": 256,
        "v_block": 256,
        "v_col_block": 128,
        "query_quant_mode": FA_PER_BLOCK_QUANT_MODE,
        "key_quant_mode": FA_PER_BLOCK_QUANT_MODE,
        "value_quant_mode": FA_PER_BLOCK_QUANT_MODE,
        "inner_precise": None,
    },
    "C8V16_TILING512": {
        "q_block": 128,
        "k_block": 256,
        "v_block": 512,
        "v_col_block": 64,
        "query_quant_mode": FA_PER_BLOCK_QUANT_MODE,
        "key_quant_mode": FA_PER_BLOCK_QUANT_MODE,
        "value_quant_mode": FA_V512_D64_QUANT_MODE,
        "inner_precise": FA_C8V16_INNER_PRECISE,
    },
}


def _get_int_kwarg(kwargs, name, default, positive=True):
    value = kwargs.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParametersInvalid(f"The data type of input {name} must be int, but got {type(value)}.")
    if positive and value <= 0:
        raise ParametersInvalid(f"The input {name} must be greater than 0, but got {value}.")
    if not positive and value < 0:
        raise ParametersInvalid(f"The input {name} must be greater than or equal to 0, but got {value}.")
    return value


def _check_heads(name, tensor, input_layout, expected_heads):
    if input_layout == "BNSD":
        actual_heads = tensor.shape[1]
    elif input_layout == "BSND":
        actual_heads = tensor.shape[2]
    else:
        if tensor.shape[2] % expected_heads != 0:
            raise ParametersInvalid(
                f"The hidden size of input {name} must be divisible by head num {expected_heads}, "
                f"but got {tensor.shape[2]}."
            )
        return
    if actual_heads != expected_heads:
        raise ParametersInvalid(
            f"The head num of input {name} must be {expected_heads} for {input_layout}, but got {actual_heads}."
        )


def _validate_fia_inputs(query, key, value, kwargs):
    """Check FIA layout/head attributes; common tensor checks belong to the caller."""
    input_layout = kwargs.get("input_layout", "BSH")
    if input_layout not in _SUPPORTED_LAYOUTS:
        raise ParametersInvalid(f"The input_layout must in {_SUPPORTED_LAYOUTS}, but got {input_layout}.")
    num_query_heads = _get_int_kwarg(kwargs, "num_query_heads", 1)
    num_key_value_heads = _get_int_kwarg(kwargs, "num_key_value_heads", 0, positive=False)
    effective_kv_heads = num_key_value_heads or num_query_heads
    _check_heads("query", query, input_layout, num_query_heads)
    _check_heads("key", key, input_layout, effective_kv_heads)
    _check_heads("value", value, input_layout, effective_kv_heads)


def _normalize_torch_npu_dtype(value):
    try:
        import torch_npu
    except ImportError:
        return value

    for dtype_name, cann_dtype in _TORCH_NPU_CANN_DTYPE_NAMES:
        torch_npu_dtype = getattr(torch_npu, dtype_name, None)
        if torch_npu_dtype is None:
            continue
        if value is torch_npu_dtype:
            return cann_dtype
        try:
            if value == torch_npu_dtype:
                return cann_dtype
        except TypeError:
            continue
    return value


def _normalize_dtype_arg(value):
    value = _normalize_torch_npu_dtype(value)
    if isinstance(value, torch.dtype):
        return None
    return value


def fused_infer_attention_score_v2(query, key, value, **kwargs):
    """Validate FIA attributes and adapt them to the native operator."""
    _validate_fia_inputs(query, key, value, kwargs)
    normalized_kwargs = {
        name: _normalize_dtype_arg(value) if name in _DTYPE_KWARGS else value
        for name, value in kwargs.items()
        if name not in _IGNORED_COMPAT_KWARGS
    }
    return torch.ops.mindiesd.fused_infer_attention_score_v2(query, key, value, **normalized_kwargs)


def _fp8_attention_forward(query, key, value, *, layout, scale, pre_tokens, next_tokens, mode, q_rot=None, k_rot=None):
    """Block-quantize in BNSD, execute FIA, then restore the caller's layout."""
    # Common Q/K/V and rotation validation is performed by quant_attention.
    # Keep only block-FP8 restrictions and mode resolution here.
    from ...quantization.mode import FP8FAMode, normalize_fp8_fa_mode

    if query.shape[0] != 1:
        raise ValueError("Block-FP8 attention requires batch size 1.")
    mode = normalize_fp8_fa_mode(mode) or FP8FAMode.HIGH_PRECISION
    if layout == "BNSD":
        _, num_heads, sequence_length, _ = query.shape
        num_kv_heads = key.shape[1]
    else:
        _, sequence_length, num_heads, _ = query.shape
        num_kv_heads = key.shape[2]
    spec = _FP8_FA_MODE_SPEC[mode]

    if q_rot is not None:
        query = torch.matmul(query, q_rot)
    if k_rot is not None:
        key = torch.matmul(key, k_rot)

    q, q_scale = _fa_block_quant_preprocess(
        query, block_size=spec["q_block"], dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    k, k_scale = _fa_block_quant_preprocess(
        key, block_size=spec["k_block"], dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    v, v_scale = _fa_block_quant_preprocess(
        value,
        block_size=spec["v_block"],
        col_block_size=spec["v_col_block"],
        dst_type=torch_npu.float8_e4m3fn,
        layout=layout,
    )

    fa_kwargs = {
        "input_layout": "BNSD",
        "num_query_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "softmax_scale": scale,
        "pre_tokens": pre_tokens,
        "next_tokens": next_tokens,
        "query_quant_mode": spec["query_quant_mode"],
        "key_quant_mode": spec["key_quant_mode"],
        "value_quant_mode": spec["value_quant_mode"],
        "dequant_scale_query": q_scale,
        "dequant_scale_key": k_scale,
        "dequant_scale_value": v_scale,
        "out_dtype": query.dtype,
    }
    if spec["inner_precise"] is not None:
        fa_kwargs["inner_precise"] = spec["inner_precise"]

    output = fused_infer_attention_score_v2(q, k, v, **fa_kwargs)[0]
    output = output[:, :, :sequence_length, :]
    return output.transpose(1, 2) if layout == "BSND" else output
