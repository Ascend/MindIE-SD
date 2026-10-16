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

from ...quantization.config import QuantConfig
from ...utils import ParametersInvalid
from ...utils.logs.logging import logger
from .experts_selector import select_experts
from .moe_mlp import unified_apply_mlp
from .moe_context import (
    MoECommType,
    build_mlp_compute_input,
    build_moe_weights,
    build_prepare_input,
    build_routing_input,
    build_token_dispatch_input,
    get_moe_comm_type,
    get_moe_group,
    get_moe_quant_algo,
    set_moe_context,
    should_use_gmm_finalize_routing,
    validate_moe_inputs,
)
from .token_dispatcher import DynamicDispatcher, StaticDispatcher

_MOE_CONFIG_LOGGED = False


def _resolve_default_dispatcher(top_k):
    """Select the default dispatcher from communication mode and routing fan-out."""
    if get_moe_comm_type() != MoECommType.EP:
        return StaticDispatcher

    ep_group = get_moe_group()
    ep_size = dist.get_world_size(ep_group)
    if top_k < ep_size:
        return DynamicDispatcher
    return StaticDispatcher


def _resolve_dispatcher(dispatcher_type):
    """Resolve an explicitly requested dispatcher type."""
    if dispatcher_type == "static":
        return StaticDispatcher
    if dispatcher_type == "dynamic":
        if get_moe_comm_type() != MoECommType.EP:
            raise ParametersInvalid("Dynamic MoE dispatcher requires EP communication.")
        return DynamicDispatcher
    raise ParametersInvalid(
        f"Unsupported dispatcher_type: {dispatcher_type}. Supported types are 'static' and 'dynamic'."
    )


def resolve_dispatcher_class(top_k: int, dispatcher_type=None):
    """Resolve the dispatcher class for the current MoE invocation."""
    if dispatcher_type is not None:
        return _resolve_dispatcher(dispatcher_type)
    return _resolve_default_dispatcher(top_k=top_k)


def _log_moe_config_once(dispatcher_cls, inputs_sharded, reduce_routed_out):
    global _MOE_CONFIG_LOGGED
    if _MOE_CONFIG_LOGGED:
        return
    dispatcher_name = _get_dispatcher_type_name(dispatcher_cls)
    comm_type = get_moe_comm_type()
    moe_group = get_moe_group()
    comm_world_size = dist.get_world_size(moe_group) if moe_group is not None else 1
    quant_algo = get_moe_quant_algo()
    logger.debug(
        "[MindIE-SD/moe] MoE config resolved. dispatcher=%s, comm=%s, quant_algo=%s, "
        "inputs_sharded=%s, reduce_routed_out=%s.",
        dispatcher_name,
        f"{comm_type.value}-{comm_world_size}",
        quant_algo.value,
        inputs_sharded,
        reduce_routed_out,
    )
    _MOE_CONFIG_LOGGED = True


def _get_dispatcher_type_name(dispatcher_cls) -> str:
    return "dynamic" if dispatcher_cls is DynamicDispatcher else "static"


def moe(
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
    inputs_sharded: bool = False,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,
    routing_method: str = "softmax",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    custom_routing_function: Callable | None = None,
    reduce_routed_out: bool = True,
    return_dispatcher_type: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    """Run the MoE forward pass on NPU."""

    quant_algo = validate_moe_inputs(
        hidden_states=hidden_states,
        router_logits=router_logits,
        num_experts=num_experts,
        top_k=top_k,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        quant_config=quant_config,
        w13_weight_scale=w13_weight_scale,
        w2_weight_scale=w2_weight_scale,
        tp_group=tp_group,
        ep_group=ep_group,
        dispatcher_type=dispatcher_type,
        inputs_sharded=inputs_sharded,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        routing_method=routing_method,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        custom_routing_function=custom_routing_function,
        reduce_routed_out=reduce_routed_out,
        return_dispatcher_type=return_dispatcher_type,
    )

    set_moe_context(tp_group=tp_group, ep_group=ep_group, quant_algo=quant_algo)
    dispatcher_cls = resolve_dispatcher_class(
        top_k=top_k,
        dispatcher_type=dispatcher_type,
    )
    use_gmm_finalize_routing = should_use_gmm_finalize_routing(_get_dispatcher_type_name(dispatcher_cls))
    moe_weights = build_moe_weights(
        w13_weight,
        w2_weight,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        w13_weight_scale=w13_weight_scale,
        w2_weight_scale=w2_weight_scale,
    )

    prepare_input = build_prepare_input(
        hidden_states=hidden_states,
        router_logits=router_logits,
        inputs_sharded=inputs_sharded,
    )
    _log_moe_config_once(
        dispatcher_cls,
        prepare_input.inputs_sharded,
        reduce_routed_out,
    )
    prepare_output = dispatcher_cls.prepare(prepare_input)

    routing_input = build_routing_input(
        hidden_states=prepare_output.hidden_states,
        router_logits=prepare_output.router_logits,
        top_k=top_k,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        routing_method=routing_method,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        custom_routing_function=custom_routing_function,
    )
    topk_weights, topk_ids = select_experts(routing_input)

    token_dispatch_input = build_token_dispatch_input(
        hidden_states=prepare_output.hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        weights=moe_weights,
        dynamic_scale=prepare_output.dynamic_scale,
        use_gmm_finalize_routing=use_gmm_finalize_routing,
    )
    dispatch_output = dispatcher_cls.dispatch(token_dispatch_input)

    mlp_input = build_mlp_compute_input(
        dispatch_output=dispatch_output,
        weights=moe_weights,
        mlp_output_dtype=prepare_output.mlp_output_dtype,
    )
    expert_output = unified_apply_mlp(
        mlp_input,
        dispatch_output.combine_metadata if use_gmm_finalize_routing else None,
    )
    if use_gmm_finalize_routing:
        routed_out = expert_output
    else:
        routed_out = dispatcher_cls.combine(
            hidden_states=expert_output,
            combine_metadata=dispatch_output.combine_metadata,
        )

    output = dispatcher_cls.finalize(
        routed_out=routed_out,
        original_shape=prepare_output.original_shape,
        inputs_sharded=prepare_input.inputs_sharded,
        reduce_routed_out=reduce_routed_out,
    )
    if return_dispatcher_type:
        return output, _get_dispatcher_type_name(dispatcher_cls)
    return output
