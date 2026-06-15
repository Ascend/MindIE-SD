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

from ..quantization.config import QuantConfig
from ..utils import ParametersInvalid
from .moe import moe


def fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    quant_config: QuantConfig | None = None,
    w13_weight_scale: torch.Tensor | None = None,
    w2_weight_scale: torch.Tensor | None = None,
    tp_group: dist.ProcessGroup | None = None,
    ep_group: dist.ProcessGroup | None = None,
    dispatcher_type: str | None = None,
    tokens_full: bool = True,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,
    routing_method: str = "softmax",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    custom_routing_function: Callable | None = None,
    reduce_results: bool = True,
    use_fused_op: bool = False,
) -> torch.Tensor:
    """Run MoE through the public fused-MoE entry.

    The current version exposes the fused-op switch for forward compatibility,
    while all calls fall back to the non-fused MoE implementation.

    Args:
        hidden_states (torch.Tensor):
            Input activations with shape ``[..., hidden_size]``.
        router_logits (torch.Tensor):
            Router logits with shape ``[..., num_experts]``. The leading token
            dimensions must match ``hidden_states``.
        num_experts (int):
            Total number of global experts.
        top_k (int):
            Number of experts selected per token.
        w13_weight (torch.Tensor):
            Fused gate/up projection weights with shape
            ``[local_experts, hidden_size, 2 * intermediate_size]``.
        w2_weight (torch.Tensor):
            Down projection weights with shape
            ``[local_experts, intermediate_size, hidden_size]``.
        w13_bias (torch.Tensor, optional):
            Optional fused gate/up projection bias with shape
            ``[local_experts, 2 * intermediate_size]``.
        w2_bias (torch.Tensor, optional):
            Optional down projection bias with shape
            ``[local_experts, hidden_size]``.
        quant_config (QuantConfig, optional):
            MindIE-SD quantization config.
        w13_weight_scale (torch.Tensor, optional):
            Quantization scale for w13_weight.
        w2_weight_scale (torch.Tensor, optional):
            Quantization scale for w2_weight.
        tp_group (optional):
            Tensor-parallel process group used for MoE TP communication.
        ep_group (optional):
            Expert-parallel process group used for MoE EP communication.
        dispatcher_type (str, optional):
            Manual MoE dispatcher override. Supported values are ``"static"`` and
            ``"dynamic"``. ``None`` uses the default device and communication routing.
        tokens_full (bool, optional):
            Token layout across the resolved MoE communication group (TP or EP).
            ``True`` means ``hidden_states`` and ``router_logits`` contain the
            full token set on each rank. ``False`` means each rank receives the
            token shard evenly split by the communication group. Other token
            layouts are not supported.
        k_group (int, optional):
            Number of expert groups selected per token during grouped routing.
        group_count (int, optional):
            Number of expert groups used by grouped routing.
        group_select_mode (int, optional):
            Expert-group scoring mode. ``0`` uses max score in each group and
            ``1`` uses the sum of top-2 scores in each group.
        routing_method (str, optional):
            Router score function. Supported values are ``"softmax"`` and ``"sigmoid"``.
        renormalize (bool, optional):
            Whether to renormalize selected top-k routing weights for softmax routing.
            Sigmoid routing follows the NPU gating top-k op semantics.
        routed_scaling_factor (float, optional):
            Scaling factor applied to routing weights during expert selection.
        custom_routing_function (optional):
            Optional routing callback. It must return ``(topk_weights, topk_ids)``.
        reduce_results (bool, optional):
            Whether static MoE reduces full-token routed outputs across the
            resolved MoE communication group. This only applies when static MoE
            is used with ``tokens_full=True``.
        use_fused_op (bool, optional):
            Whether to use the real fused MoE op. The current version does not
            support this path and falls back to the non-fused MoE implementation.

    Returns:
        torch.Tensor: Output activations with the same shape as ``hidden_states``.
    """
    if not isinstance(use_fused_op, bool):
        raise ParametersInvalid(f"use_fused_op must be a bool, but got {type(use_fused_op)}.")

    moe_kwargs = {
        "hidden_states": hidden_states,
        "router_logits": router_logits,
        "num_experts": num_experts,
        "top_k": top_k,
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_bias": w13_bias,
        "w2_bias": w2_bias,
        "quant_config": quant_config,
        "w13_weight_scale": w13_weight_scale,
        "w2_weight_scale": w2_weight_scale,
        "tp_group": tp_group,
        "ep_group": ep_group,
        "dispatcher_type": dispatcher_type,
        "tokens_full": tokens_full,
        "k_group": k_group,
        "group_count": group_count,
        "group_select_mode": group_select_mode,
        "routing_method": routing_method,
        "renormalize": renormalize,
        "routed_scaling_factor": routed_scaling_factor,
        "custom_routing_function": custom_routing_function,
        "reduce_results": reduce_results,
    }
    return moe(**moe_kwargs)
