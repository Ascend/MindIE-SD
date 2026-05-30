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

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MoEPrepareInput:
    """Input consumed by the dispatcher prepare stage."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    tokens_full: bool


@dataclass(frozen=True)
class MoERoutingInput:
    """Input consumed by expert selection."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    top_k: int
    renormalize: bool = False
    custom_routing_function: Any = None


@dataclass(frozen=True)
class MoETokenDispatchInput:
    """Input consumed by the dispatcher token-routing stage."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    num_experts: int
    top_k: int
    local_num_experts: int


@dataclass(frozen=True)
class MoEStaticCombineMetadata:
    """Metadata required to restore token order in static MoE."""

    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor
    restore_shape: torch.Size


@dataclass(frozen=True)
class MoEDynamicCombineMetadata:
    """Metadata required to restore token order after dynamic MoE exchange."""

    input_splits: Any
    output_splits: Any
    topk_weights: torch.Tensor
    local_unpermute_indices: torch.Tensor
    global_unpermute_indices: torch.Tensor | None
    hidden_shape: torch.Size


@dataclass(frozen=True)
class MoEMlpComputeInput:
    """Input consumed by grouped expert MLP computation."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None
