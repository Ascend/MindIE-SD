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

from .. import register_ops as _mindiesd_register_ops  # noqa: F401
from ...utils.exception import ParametersInvalid


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


def _check_tensor_arg(name, value):
    if not isinstance(value, torch.Tensor):
        raise ParametersInvalid(f"The data type of input {name} must be torch.Tensor, but got {type(value)}.")


def _get_int_kwarg(kwargs, name, default, positive=True):
    value = kwargs.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParametersInvalid(f"The data type of input {name} must be int, but got {type(value)}.")
    if positive and value <= 0:
        raise ParametersInvalid(f"The input {name} must be greater than 0, but got {value}.")
    if not positive and value < 0:
        raise ParametersInvalid(f"The input {name} must be greater than or equal to 0, but got {value}.")
    return value


def _check_layout_dim(name, tensor, input_layout):
    expected_dim = 3 if input_layout == "BSH" else 4
    if tensor.dim() != expected_dim:
        raise ParametersInvalid(
            f"The dimensional of input {name} must be {expected_dim} for {input_layout}, but got {tensor.dim()}."
        )


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


def _check_qkv_shape(query, key, value, input_layout, head_counts):
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _check_layout_dim(name, tensor, input_layout)

    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ParametersInvalid(
            f"The batch size of query/key/value must be same, but got {query.shape[0]}/{key.shape[0]}/{value.shape[0]}."
        )

    num_query_heads, num_key_value_heads = head_counts
    effective_kv_heads = num_query_heads if num_key_value_heads == 0 else num_key_value_heads
    if num_query_heads % effective_kv_heads != 0:
        raise ParametersInvalid(
            f"The num_query_heads must be divisible by num_key_value_heads, "
            f"but got {num_query_heads}/{effective_kv_heads}."
        )

    _check_heads("query", query, input_layout, num_query_heads)
    _check_heads("key", key, input_layout, effective_kv_heads)
    _check_heads("value", value, input_layout, effective_kv_heads)

    if input_layout == "BNSD":
        key_seq_dim = value_seq_dim = 2
    else:
        key_seq_dim = value_seq_dim = 1
    if key.shape[key_seq_dim] != value.shape[value_seq_dim]:
        raise ParametersInvalid(
            f"The sequence length of key/value must be same, but got "
            f"{key.shape[key_seq_dim]}/{value.shape[value_seq_dim]}."
        )


def _validate_fia_inputs(query, key, value, kwargs):
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _check_tensor_arg(name, tensor)

    input_layout = kwargs.get("input_layout", "BSH")
    if input_layout not in _SUPPORTED_LAYOUTS:
        raise ParametersInvalid(f"The input_layout must in {_SUPPORTED_LAYOUTS}, but got {input_layout}.")

    num_query_heads = _get_int_kwarg(kwargs, "num_query_heads", 1)
    num_key_value_heads = _get_int_kwarg(kwargs, "num_key_value_heads", 0, positive=False)
    _check_qkv_shape(query, key, value, input_layout, (num_query_heads, num_key_value_heads))

    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ParametersInvalid(
            f"The dtype of query/key/value must be same, but got {query.dtype}/{key.dtype}/{value.dtype}."
        )


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
    _validate_fia_inputs(query, key, value, kwargs)
    normalized_kwargs = {
        name: _normalize_dtype_arg(value) if name in _DTYPE_KWARGS else value
        for name, value in kwargs.items()
        if name not in _IGNORED_COMPAT_KWARGS
    }
    return torch.ops.mindiesd.fused_infer_attention_score_v2(query, key, value, **normalized_kwargs)
