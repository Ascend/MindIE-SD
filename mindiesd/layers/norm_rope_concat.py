#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Validated Python entry point for the existing norm_rope_concat NPU operator."""

import math

import torch

from ..utils import ParametersInvalid
from . import register_ops  # noqa: F401  Register the existing C++ operator.


def norm_rope_concat(
    query,
    key,
    value,
    *,
    input_layout="BSND",
    output_layout="BSND",
    encoder_query=None,
    encoder_key=None,
    encoder_value=None,
    norm_query_weight=None,
    norm_query_bias=None,
    norm_key_weight=None,
    norm_key_bias=None,
    norm_added_query_weight=None,
    norm_added_query_bias=None,
    norm_added_key_weight=None,
    norm_added_key_bias=None,
    rope_sin=None,
    rope_cos=None,
    norm_type=0,
    norm_added_type=0,
    rope_type=0,
    concat_order=0,
    eps=1e-5,
    is_training=False,
):
    """Fuse Q/K normalization, RoPE and optional encoder concatenation.

    Args:
        query, key, value (torch.Tensor): 4D NPU FP16/BF16/FP32 tensors sharing
            batch, heads, head_dim and dtype. K/V sequence lengths must match.
        input_layout (str): Input Q/K/V and encoder layout: "BSND" (default) or "BNSD".
        output_layout (str): Layout of the first three outputs; same choices, default "BSND".
        encoder_query, encoder_key, encoder_value (torch.Tensor, optional): Encoder
            tensors with matching batch/heads/head_dim. Encoder K/V must appear together.
        norm_query_weight, norm_key_weight (torch.Tensor, optional): Main Q/K weights [D].
        norm_query_bias, norm_key_bias (torch.Tensor, optional): Main Q/K biases [D].
        norm_added_query_weight, norm_added_key_weight (torch.Tensor, optional): Encoder weights [D].
        norm_added_query_bias, norm_added_key_bias (torch.Tensor, optional): Encoder biases [D].
            Affine modes require weights; affine LayerNorm also requires biases.
        norm_type (int): 0=none, 1=LayerNorm, 2=affine LayerNorm, 3=RMSNorm,
            4=affine RMSNorm. Default: 0.
        norm_added_type (int): Encoder norm type; same values as norm_type. Default: 0.
        rope_sin, rope_cos (torch.Tensor, optional): Paired [S_rope, D] tables.
            Only the first S_rope positions of the concatenated sequence are rotated.
        rope_type (int): 0=none, 1=interleaved pairs, 2=half-split pairs. Default: 0.
        concat_order (int): 0=main before encoder, 1=encoder before main. Default: 0.
        eps (float): Positive finite normalization epsilon. Default: 1e-5.
        is_training (bool): Whether to write applicable norm statistics. Default: False.

    Returns:
        tuple: Q/K/V in output_layout, followed by the original eight norm statistics.
        BSND outputs are views and may be noncontiguous. Statistics are only valid
        for the corresponding training/norm modes.

    Raises:
        ParametersInvalid: Invalid inputs. Operator execution errors propagate unchanged.
    """

    _validate_inputs(
        query,
        key,
        value,
        input_layout=input_layout,
        output_layout=output_layout,
        encoder_query=encoder_query,
        encoder_key=encoder_key,
        encoder_value=encoder_value,
        norm_query_weight=norm_query_weight,
        norm_query_bias=norm_query_bias,
        norm_key_weight=norm_key_weight,
        norm_key_bias=norm_key_bias,
        norm_added_query_weight=norm_added_query_weight,
        norm_added_query_bias=norm_added_query_bias,
        norm_added_key_weight=norm_added_key_weight,
        norm_added_key_bias=norm_added_key_bias,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_type=norm_type,
        norm_added_type=norm_added_type,
        rope_type=rope_type,
        concat_order=concat_order,
        eps=eps,
        is_training=is_training,
    )

    (
        query,
        key,
        value,
        encoder_query,
        encoder_key,
        encoder_value,
    ) = _convert_input_layout(
        query,
        key,
        value,
        encoder_query,
        encoder_key,
        encoder_value,
        input_layout,
    )
    outputs = torch.ops.mindiesd.norm_rope_concat(
        query,
        key,
        value,
        encoder_query=encoder_query,
        encoder_key=encoder_key,
        encoder_value=encoder_value,
        norm_query_weight=norm_query_weight,
        norm_query_bias=norm_query_bias,
        norm_key_weight=norm_key_weight,
        norm_key_bias=norm_key_bias,
        norm_added_query_weight=norm_added_query_weight,
        norm_added_query_bias=norm_added_query_bias,
        norm_added_key_weight=norm_added_key_weight,
        norm_added_key_bias=norm_added_key_bias,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_type=norm_type,
        norm_added_type=norm_added_type,
        rope_type=rope_type,
        concat_order=concat_order,
        eps=float(eps),
        is_training=is_training,
    )
    return _convert_output_layout(outputs, output_layout)


def _validate_inputs(
    query,
    key,
    value,
    *,
    input_layout,
    output_layout,
    encoder_query,
    encoder_key,
    encoder_value,
    norm_query_weight,
    norm_query_bias,
    norm_key_weight,
    norm_key_bias,
    norm_added_query_weight,
    norm_added_query_bias,
    norm_added_key_weight,
    norm_added_key_bias,
    rope_sin,
    rope_cos,
    norm_type,
    norm_added_type,
    rope_type,
    concat_order,
    eps,
    is_training,
):
    for name, layout in (("input_layout", input_layout), ("output_layout", output_layout)):
        if not isinstance(layout, str) or layout not in ("BSND", "BNSD"):
            raise ParametersInvalid(f"norm_rope_concat: {name} must be 'BSND' or 'BNSD'")
    _validate_qkv_inputs(query, key, value, encoder_query, encoder_key, encoder_value, input_layout)
    _, _, _, dim = query.shape

    _validate_op_attributes(norm_type, norm_added_type, rope_type, concat_order, eps, is_training)

    _validate_norm_params(
        query,
        encoder_query,
        encoder_key,
        norm_query_weight,
        norm_query_bias,
        norm_key_weight,
        norm_key_bias,
        norm_added_query_weight,
        norm_added_query_bias,
        norm_added_key_weight,
        norm_added_key_bias,
        rope_sin,
        rope_cos,
        norm_type,
        norm_added_type,
        rope_type,
        dim,
    )

    _validate_rope_tables(query, key, encoder_query, encoder_key, rope_sin, rope_cos, dim, input_layout)

    if query.device.type != "npu" or query.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ParametersInvalid("norm_rope_concat: query must be NPU FP16, BF16 or FP32")


def _validate_qkv_inputs(query, key, value, encoder_query, encoder_key, encoder_value, input_layout):
    if not isinstance(query, torch.Tensor):
        raise ParametersInvalid("norm_rope_concat: query must be a torch.Tensor")
    if query.ndim != 4 or min(query.shape) <= 0:
        raise ParametersInvalid(f"norm_rope_concat: query must be nonempty {input_layout}")
    seq_axis, head_axis = (1, 2) if input_layout == "BSND" else (2, 1)
    batch, heads, dim = query.shape[0], query.shape[head_axis], query.shape[3]
    if dim % 2 or dim > 1024:
        raise ParametersInvalid("norm_rope_concat: head_dim must be even and <= 1024")

    data = {
        "key": key,
        "value": value,
        "encoder_query": encoder_query,
        "encoder_key": encoder_key,
        "encoder_value": encoder_value,
    }
    for name, tensor in data.items():
        _validate_tensor(name, tensor, required=name in ("key", "value"), device=query.device, dtype=query.dtype)
        sequence_length = None
        if name == "value":
            sequence_length = key.shape[seq_axis]
        elif name == "encoder_value" and encoder_key is not None:
            sequence_length = encoder_key.shape[seq_axis]
        _validate_qkv_shape(name, tensor, batch, heads, dim, input_layout, sequence_length=sequence_length)
    if (encoder_key is None) != (encoder_value is None):
        raise ParametersInvalid("norm_rope_concat: encoder_key and encoder_value must appear together")


def _validate_op_attributes(norm_type, norm_added_type, rope_type, concat_order, eps, is_training):
    for name, number, choices in (
        ("norm_type", norm_type, range(5)),
        ("norm_added_type", norm_added_type, range(5)),
        ("rope_type", rope_type, range(3)),
        ("concat_order", concat_order, range(2)),
    ):
        if not isinstance(number, int) or isinstance(number, bool) or number not in choices:
            raise ParametersInvalid(f"norm_rope_concat: {name} must be one of {tuple(choices)}")
    if not isinstance(is_training, bool):
        raise ParametersInvalid("norm_rope_concat: is_training must be bool")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ParametersInvalid("norm_rope_concat: eps must be finite and positive")


def _validate_norm_params(
    query,
    encoder_query,
    encoder_key,
    norm_query_weight,
    norm_query_bias,
    norm_key_weight,
    norm_key_bias,
    norm_added_query_weight,
    norm_added_query_bias,
    norm_added_key_weight,
    norm_added_key_bias,
    rope_sin,
    rope_cos,
    norm_type,
    norm_added_type,
    rope_type,
    dim,
):
    weights = {
        "norm_query_weight": norm_query_weight,
        "norm_query_bias": norm_query_bias,
        "norm_key_weight": norm_key_weight,
        "norm_key_bias": norm_key_bias,
        "norm_added_query_weight": norm_added_query_weight,
        "norm_added_query_bias": norm_added_query_bias,
        "norm_added_key_weight": norm_added_key_weight,
        "norm_added_key_bias": norm_added_key_bias,
    }
    for name, tensor in weights.items():
        _validate_tensor(name, tensor, required=False, device=query.device, dtype=query.dtype)
        if tensor is not None and tuple(tensor.shape) != (dim,):
            raise ParametersInvalid(f"norm_rope_concat: {name} must have shape ({dim},)")
    if norm_type in (2, 4) and (norm_query_weight is None or norm_key_weight is None):
        raise ParametersInvalid("norm_rope_concat: affine norm requires query/key weights")
    if norm_added_type in (2, 4) and encoder_query is not None and norm_added_query_weight is None:
        raise ParametersInvalid("norm_rope_concat: affine encoder norm requires query weight")
    if norm_added_type in (2, 4) and encoder_key is not None and norm_added_key_weight is None:
        raise ParametersInvalid("norm_rope_concat: affine encoder norm requires key weight")
    for name, bias, required in (
        ("norm_query_bias", norm_query_bias, norm_type == 2),
        ("norm_key_bias", norm_key_bias, norm_type == 2),
        ("norm_added_query_bias", norm_added_query_bias, norm_added_type == 2 and encoder_query is not None),
        ("norm_added_key_bias", norm_added_key_bias, norm_added_type == 2 and encoder_key is not None),
    ):
        if required and bias is None:
            raise ParametersInvalid(f"norm_rope_concat: affine LayerNorm requires {name}")
    if (rope_sin is None) != (rope_cos is None) or (rope_type and rope_sin is None):
        raise ParametersInvalid("norm_rope_concat: RoPE requires both sin and cos")


def _validate_rope_tables(query, key, encoder_query, encoder_key, rope_sin, rope_cos, dim, input_layout):
    for name, tensor in (("rope_sin", rope_sin), ("rope_cos", rope_cos)):
        _validate_tensor(name, tensor, required=False, device=query.device, dtype=query.dtype)
        if tensor is not None and (tensor.ndim != 2 or tensor.shape[1] != dim):
            raise ParametersInvalid(f"norm_rope_concat: {name} must be 2D with head_dim columns")
    if rope_sin is not None:
        seq_axis = 1 if input_layout == "BSND" else 2
        total_q = query.shape[seq_axis] + (0 if encoder_query is None else encoder_query.shape[seq_axis])
        total_k = key.shape[seq_axis] + (0 if encoder_key is None else encoder_key.shape[seq_axis])
        if rope_sin.shape != rope_cos.shape or rope_sin.shape[0] > max(total_q, total_k):
            raise ParametersInvalid("norm_rope_concat: sin/cos shapes must match and fit output sequence")


def _convert_input_layout(query, key, value, encoder_query, encoder_key, encoder_value, input_layout):
    # The existing kernel accepts contiguous BSND inputs.
    if input_layout == "BNSD":
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if encoder_query is not None:
            encoder_query = encoder_query.transpose(1, 2)
        if encoder_key is not None:
            encoder_key = encoder_key.transpose(1, 2)
        if encoder_value is not None:
            encoder_value = encoder_value.transpose(1, 2)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if encoder_query is not None:
        encoder_query = encoder_query.contiguous()
    if encoder_key is not None:
        encoder_key = encoder_key.contiguous()
    if encoder_value is not None:
        encoder_value = encoder_value.contiguous()
    return query, key, value, encoder_query, encoder_key, encoder_value


def _convert_output_layout(outputs, output_layout):
    # The kernel returns BNSD; transpose only the Q/K/V outputs when requested.
    if output_layout == "BSND":
        query_output, key_output, value_output = outputs[:3]
        return (
            query_output.transpose(1, 2),
            key_output.transpose(1, 2),
            value_output.transpose(1, 2),
            *outputs[3:],
        )
    return outputs


def _validate_tensor(name, value, *, required, device, dtype):
    if value is None:
        if required:
            raise ParametersInvalid(f"norm_rope_concat: {name} is required")
        return
    if not isinstance(value, torch.Tensor):
        raise ParametersInvalid(f"norm_rope_concat: {name} must be a torch.Tensor")
    if value.device != device or value.dtype != dtype:
        raise ParametersInvalid(f"norm_rope_concat: {name} must use {device} and {dtype}")


def _validate_qkv_shape(name, value, batch, heads, dim, layout, *, sequence_length=None):
    if value is None:
        return
    if value.ndim != 4 or min(value.shape) <= 0:
        raise ParametersInvalid(f"norm_rope_concat: {name} must be nonempty {layout}")
    head_axis = 2 if layout == "BSND" else 1
    if (value.shape[0], value.shape[head_axis], value.shape[3]) != (batch, heads, dim):
        raise ParametersInvalid(f"norm_rope_concat: {name} must match query batch, heads and head_dim")
    seq_axis = 1 if layout == "BSND" else 2
    if sequence_length is not None and value.shape[seq_axis] != sequence_length:
        raise ParametersInvalid(f"norm_rope_concat: {name} and its paired key sequence lengths must match")
