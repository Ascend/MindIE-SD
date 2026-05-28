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

from ...utils import ParametersInvalid
from ...utils.get_platform import NPUDevice, get_npu_device
from ...utils.logs.logging import logger
from .experts_selector import select_experts
from .moe_mlp import apply_mlp
from .moe_context import (
    MoECommType,
    build_mlp_compute_input,
    build_prepare_input,
    build_routing_input,
    build_token_dispatch_input,
    get_moe_comm_type,
    set_moe_comm_context,
    validate_moe_inputs,
)
from .token_dispatcher import DynamicDispatcher, StaticDispatcher

_MOE_CONFIG_LOGGED = False


def _resolve_default_dispatcher():
    """Select the default dispatcher from communication mode and NPU generation."""
    if get_moe_comm_type() == MoECommType.EP:
        npu_device = get_npu_device()
        if npu_device in (NPUDevice.A3, NPUDevice.A5):
            return DynamicDispatcher
        if npu_device in (NPUDevice.A2,):
            return StaticDispatcher
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


def resolve_dispatcher_class(dispatcher_type=None):
    """Resolve the dispatcher class for the current MoE invocation."""
    if dispatcher_type is not None:
        return _resolve_dispatcher(dispatcher_type)
    return _resolve_default_dispatcher()


def _log_moe_config_once(dispatcher_cls, tokens_full, reduce_results):
    global _MOE_CONFIG_LOGGED
    if _MOE_CONFIG_LOGGED:
        return
    dispatcher_name = "dynamic" if dispatcher_cls.__name__ == "DynamicDispatcher" else "static"
    comm_type = get_moe_comm_type().value
    logger.info(
        "MindIE-SD moe config: dispatcher=%s, comm_type=%s, tokens_full=%s, reduce_results=%s.",
        dispatcher_name,
        comm_type,
        tokens_full,
        reduce_results,
    )
    _MOE_CONFIG_LOGGED = True


def moe(
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
) -> torch.Tensor:
    """Run the staged MoE forward pass on NPU."""

    validate_moe_inputs(
        hidden_states=hidden_states,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        router_logits=router_logits,
        num_experts=num_experts,
        top_k=top_k,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        tokens_full=tokens_full,
        reduce_results=reduce_results,
        dispatcher_type=dispatcher_type,
        tp_group=tp_group,
        ep_group=ep_group,
        renormalize=renormalize,
        custom_routing_function=custom_routing_function,
    )

    set_moe_comm_context(tp_group=tp_group, ep_group=ep_group)
    dispatcher_cls = resolve_dispatcher_class(dispatcher_type=dispatcher_type)

    prepare_input = build_prepare_input(
        hidden_states=hidden_states,
        router_logits=router_logits,
        tokens_full=tokens_full,
    )
    _log_moe_config_once(dispatcher_cls, prepare_input.tokens_full, reduce_results)
    prepared_hidden_states, prepared_router_logits, original_shape = dispatcher_cls.prepare(prepare_input)

    routing_input = build_routing_input(
        hidden_states=prepared_hidden_states,
        router_logits=prepared_router_logits,
        top_k=top_k,
        renormalize=renormalize,
        custom_routing_function=custom_routing_function,
    )
    topk_weights, topk_ids = select_experts(routing_input)

    token_dispatch_input = build_token_dispatch_input(
        hidden_states=prepared_hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        w13_weight=w13_weight,
    )
    dispatched_hidden_states, group_list, group_list_type, combine_metadata = dispatcher_cls.dispatch(
        token_dispatch_input
    )

    mlp_input = build_mlp_compute_input(
        dispatched_hidden_states,
        group_list,
        group_list_type,
        w13_weight,
        w2_weight,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )
    expert_output = apply_mlp(mlp_input)

    routed_out = dispatcher_cls.combine(
        hidden_states=expert_output,
        combine_metadata=combine_metadata,
    )

    return dispatcher_cls.finalize(
        routed_out=routed_out,
        original_shape=original_shape,
        tokens_full=prepare_input.tokens_full,
        reduce_results=reduce_results,
    )
