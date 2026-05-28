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

from collections.abc import Callable

import torch
import torch.distributed as dist

from ..utils import ParametersInvalid
from .moe import moe


def fused_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    tokens_full: bool = True,
    reduce_results: bool = True,
    dispatcher_type: str | None = None,
    tp_group: dist.ProcessGroup | None = None,
    ep_group: dist.ProcessGroup | None = None,
    renormalize: bool = False,
    custom_routing_function: Callable | None = None,
    use_fused_op: bool = False,
) -> torch.Tensor:
    """Run MoE through the public fused-MoE entry.

    The current version exposes the fused-op switch for forward compatibility.
    Unsupported fused-op scenarios fall back to the staged MoE path.

    Args:
        hidden_states (torch.Tensor):
            Input activations with shape ``[..., hidden_size]``.
        w13_weight (torch.Tensor):
            Fused gate/up projection weights with shape
            ``[local_experts, hidden_size, 2 * intermediate_size]``.
        w2_weight (torch.Tensor):
            Down projection weights with shape
            ``[local_experts, intermediate_size, hidden_size]``.
        router_logits (torch.Tensor):
            Router logits with shape ``[..., num_experts]``. The leading token
            dimensions must match ``hidden_states``.
        num_experts (int):
            Total number of global experts.
        top_k (int):
            Number of experts selected per token.
        w13_bias (torch.Tensor, optional):
            Optional fused gate/up projection bias with shape
            ``[local_experts, 2 * intermediate_size]``.
        w2_bias (torch.Tensor, optional):
            Optional down projection bias with shape
            ``[local_experts, hidden_size]``.
        tokens_full (bool, optional):
            Token layout across the resolved MoE communication group (TP or EP).
            ``True`` means ``hidden_states`` and ``router_logits`` contain the
            full token set on each rank. ``False`` means each rank receives the
            token shard evenly split by the communication group. Other token
            layouts are not supported.
        reduce_results (bool, optional):
            Whether static MoE reduces full-token routed outputs across the
            resolved MoE communication group. This only applies when static MoE
            is used with ``tokens_full=True``.
        dispatcher_type (str, optional):
            Manual MoE dispatcher override. Supported values are ``"static"`` and
            ``"dynamic"``. ``None`` uses the default device and communication routing.
        tp_group (optional):
            Tensor-parallel process group used for MoE TP communication.
        ep_group (optional):
            Expert-parallel process group used for MoE EP communication.
        renormalize (bool, optional):
            Whether to renormalize the selected top-k routing weights.
        custom_routing_function (optional):
            Optional routing callback. It must return ``(topk_weights, topk_ids)``.
        use_fused_op (bool, optional):
            Whether to use the real fused MoE op. The current version does not
            support this path and falls back to staged MoE.

    Returns:
        torch.Tensor: Output activations with the same shape as ``hidden_states``.
    """
    if not isinstance(use_fused_op, bool):
        raise ParametersInvalid(f"use_fused_op must be a bool, but got {type(use_fused_op)}.")

    moe_kwargs = {
        "hidden_states": hidden_states,
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "router_logits": router_logits,
        "num_experts": num_experts,
        "top_k": top_k,
        "w13_bias": w13_bias,
        "w2_bias": w2_bias,
        "tokens_full": tokens_full,
        "reduce_results": reduce_results,
        "dispatcher_type": dispatcher_type,
        "tp_group": tp_group,
        "ep_group": ep_group,
        "renormalize": renormalize,
        "custom_routing_function": custom_routing_function,
    }
    return moe(**moe_kwargs)
