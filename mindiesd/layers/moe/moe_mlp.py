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

import torch
import torch_npu

from ...quantization.mode import QuantAlgorithm
from .moe_context import get_moe_quant_algo
from .moe_dataclass import MoEMlpComputeInput

torch_npu.npu.config.allow_internal_format = True


def _ensure_nz_weight(weight: torch.Tensor) -> torch.Tensor:
    if torch_npu.get_npu_format(weight) == 29:
        return weight
    return torch_npu.npu_format_cast(weight, 29)


def _normalize_mxfp_scale_layout(scale: torch.Tensor | None) -> torch.Tensor | None:
    if scale is None or scale.dim() != 2:
        return scale
    if scale.shape[-1] % 2 != 0:
        raise ValueError(f"Invalid MXFP scale shape: {tuple(scale.shape)}")
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


def unquant_apply_mlp(mlp_input: MoEMlpComputeInput) -> torch.Tensor:
    hidden_states = mlp_input.hidden_states
    weights = mlp_input.weights
    w13_weight = weights.w13_weight
    w2_weight = weights.w2_weight
    w13_bias = weights.w13_bias
    w2_bias = weights.w2_bias
    group_list = mlp_input.group_list
    group_list_type = mlp_input.group_list_type
    bias_dtype = torch.float32 if hidden_states.dtype == torch.bfloat16 else hidden_states.dtype

    gate_up = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13_weight],
        bias=[w13_bias.to(dtype=bias_dtype)] if w13_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    act_out = torch_npu.npu_swiglu(gate_up)
    return torch_npu.npu_grouped_matmul(
        x=[act_out],
        weight=[w2_weight],
        bias=[w2_bias.to(dtype=bias_dtype)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]


def w8a8_dynamic_apply_mlp(mlp_input: MoEMlpComputeInput) -> torch.Tensor:
    """W8A8 dynamic quantized grouped expert MLP."""
    hidden_states = mlp_input.hidden_states
    weights = mlp_input.weights
    w13_weight = _ensure_nz_weight(weights.w13_weight)
    w2_weight = _ensure_nz_weight(weights.w2_weight)
    w13_weight_scale = weights.w13_weight_scale
    w2_weight_scale = weights.w2_weight_scale
    group_list = mlp_input.group_list
    group_list_type = mlp_input.group_list_type
    per_token_scale = mlp_input.dynamic_scale
    mlp_output_dtype = mlp_input.mlp_output_dtype

    if per_token_scale is None:
        quant_hidden, per_token_scale = torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
    else:
        quant_hidden = hidden_states

    swiglu_out, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
        x=quant_hidden,
        weight=w13_weight,
        weight_scale=w13_weight_scale,
        x_scale=per_token_scale,
        group_list=group_list if group_list_type == 0 else group_list.cumsum(dim=0),
    )

    return torch_npu.npu_grouped_matmul(
        x=[swiglu_out],
        weight=[w2_weight],
        scale=[w2_weight_scale],
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=mlp_output_dtype,
    )[0]


def w8a8_mxfp8_apply_mlp(mlp_input: MoEMlpComputeInput) -> torch.Tensor:
    """W8A8 MXFP8 grouped expert MLP."""
    hidden_states = mlp_input.hidden_states
    weights = mlp_input.weights
    w13_weight = weights.w13_weight
    w2_weight = weights.w2_weight
    w13_weight_scale = weights.w13_weight_scale
    w2_weight_scale = weights.w2_weight_scale
    group_list = mlp_input.group_list
    group_list_type = mlp_input.group_list_type
    per_token_scale = mlp_input.dynamic_scale
    mlp_output_dtype = mlp_input.mlp_output_dtype

    if per_token_scale is None:
        quant_hidden, per_token_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
    else:
        quant_hidden = hidden_states
        per_token_scale = _normalize_mxfp_scale_layout(per_token_scale)

    swiglu_out, swiglu_out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=quant_hidden,
        weight=[w13_weight],
        weight_scale=[w13_weight_scale],
        x_scale=per_token_scale,
        group_list=group_list if group_list_type == 0 else group_list.cumsum(dim=0),
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        x_dtype=None,
        weight_dtype=None,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
    )

    return torch_npu.npu_grouped_matmul(
        x=[swiglu_out],
        weight=[w2_weight],
        scale=[w2_weight_scale],
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=mlp_output_dtype,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        x_dtype=None,
        weight_dtype=None,
    )[0]


def unified_apply_mlp(mlp_input: MoEMlpComputeInput) -> torch.Tensor:
    quant_algo = get_moe_quant_algo()
    if quant_algo == QuantAlgorithm.NO_QUANT:
        return unquant_apply_mlp(mlp_input)
    if quant_algo == QuantAlgorithm.W8A8_DYNAMIC:
        return w8a8_dynamic_apply_mlp(mlp_input)
    if quant_algo == QuantAlgorithm.W8A8_MXFP8:
        return w8a8_mxfp8_apply_mlp(mlp_input)
    raise ValueError(f"Unsupported MoE quantization algorithm: {quant_algo}")
