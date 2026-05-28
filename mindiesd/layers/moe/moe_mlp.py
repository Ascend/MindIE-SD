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

from .moe_dataclass import MoEMlpComputeInput


def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    gate_up = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13_weight],
        bias=[w13_bias.to(dtype=torch.float32)] if w13_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    act_out = torch_npu.npu_swiglu(gate_up)
    return torch_npu.npu_grouped_matmul(
        x=[act_out],
        weight=[w2_weight],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]


def apply_mlp(mlp_input: MoEMlpComputeInput) -> torch.Tensor:
    return unquant_apply_mlp(
        hidden_states=mlp_input.hidden_states,
        w13_weight=mlp_input.w13_weight,
        w2_weight=mlp_input.w2_weight,
        group_list=mlp_input.group_list,
        group_list_type=mlp_input.group_list_type,
        w13_bias=mlp_input.w13_bias,
        w2_bias=mlp_input.w2_bias,
    )
