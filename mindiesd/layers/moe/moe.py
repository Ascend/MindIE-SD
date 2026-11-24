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
from .eplb_hook import MoEEplbHook
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


def _resolve_default_dispatcher(top_k, eplb=False):
    """Select the default dispatcher from communication mode and routing fan-out.

    With an EPLB hook the dynamic dispatcher is mandatory: it is the only one that
    consumes the published expert layout. The static path ignores
    `device_indices_map`, so a re-placement there would swap the weights and keep
    the old routing, which silently produces wrong results -- and `top_k >= ep_size`
    (the usual shape of a real MoE model) used to select exactly that path.
    """
    if get_moe_comm_type() != MoECommType.EP:
        return StaticDispatcher

    ep_group = get_moe_group()
    ep_size = dist.get_world_size(ep_group)
    if eplb:
        return DynamicDispatcher
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


def resolve_dispatcher_class(top_k: int, dispatcher_type=None, eplb: bool = False):
    """Resolve the dispatcher class for the current MoE invocation."""
    if dispatcher_type is not None:
        resolved = _resolve_dispatcher(dispatcher_type)
        if eplb and resolved is StaticDispatcher and get_moe_comm_type() == MoECommType.EP:
            raise ParametersInvalid(
                "eplb_hook requires the dynamic MoE dispatcher: the static dispatcher ignores the published "
                "expert layout, so a re-placement would change the weights while the routing stays on the old "
                "layout. Pass dispatcher_type='dynamic' or leave it unset."
            )
        return resolved
    return _resolve_default_dispatcher(top_k=top_k, eplb=eplb)


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


# Methods the forward requires from an `eplb_hook`. Kept as a tuple so the error
# message can name exactly what is missing instead of letting the forward fail
# somewhere in the middle.
EPLB_HOOK_PROTOCOL = (
    "on_routing",
    "poll_weight_update",
    "published_device_map",
    "clear_published_layout",
    # The layout stays in force after a re-placement, so the forward needs a way to
    # read back the weights that layout belongs to; a hook without these two would
    # combine the new layout with the caller's stale rows on the next call.
    "current_weights",
    "remember_weights",
)


def _apply_re_placement(eplb_hook, weight_update, w13_bias, w2_bias, w13_weight_scale, w2_weight_scale):
    """Return the re-placed MoE weights, or `None` when the layer cannot use them.

    A re-placement permutes experts, so per-expert bias and quantization scales must
    be permuted with the weights. The payload only carries w13/w2 today, so a layer
    with either keeps its old weights instead of silently mixing a new weight order
    with an old bias/scale order.
    """
    per_expert_extras = (w13_bias, w2_bias, w13_weight_scale, w2_weight_scale)
    if any(extra is not None for extra in per_expert_extras):
        logger.warning(
            "[MindIE-SD/eplb] EPLB weight re-placement skipped: bias or quantization scale is set "
            "and cannot be permuted with the weights yet. issue=unsupported re-placement scope, "
            "w13_bias=%s, w2_bias=%s, w13_weight_scale=%s, w2_weight_scale=%s. "
            "Troubleshooting=disable EPLB for biased or quantized MoE layers; the current weights are kept.",
            w13_bias is not None,
            w2_bias is not None,
            w13_weight_scale is not None,
            w2_weight_scale is not None,
        )
        # The published layout must not stay in force either: keeping it would drive
        # the exchange with the old weights still loaded.
        eplb_hook.clear_published_layout()
        return None
    return build_moe_weights(weight_update.w13_weight, weight_update.w2_weight)


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
    eplb_hook: MoEEplbHook | None = None,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    """Run the MoE forward pass on NPU.

    Args:
        eplb_hook (MoEEplbHook, optional):
            Dynamic expert load balancing hook. ``None`` (default) leaves the
            legacy path untouched. When set, the per-expert load is collected
            right after routing and the expert weights are re-placed before the
            grouped matrix multiplication.
            A re-placement is applied when every rank keeps the same expert
            count; the set of owned experts may change, because the published
            `device_indices_map` drives the token exchange (see `MoEEplbHook`).
            Biased or quantized layers are skipped because their per-expert
            bias/scale cannot be permuted yet.
    """

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

    if eplb_hook is not None:
        missing = [name for name in EPLB_HOOK_PROTOCOL if not hasattr(eplb_hook, name)]
        if missing:
            raise ParametersInvalid(
                f"eplb_hook must expose {', '.join(EPLB_HOOK_PROTOCOL)}, but got {type(eplb_hook).__name__} "
                f"without {', '.join(missing)}."
            )

    set_moe_context(tp_group=tp_group, ep_group=ep_group, quant_algo=quant_algo)
    # A layer with a per-expert bias or quantization scale cannot take a re-placement
    # yet (`_apply_re_placement` refuses it and clears the layout), so forcing the
    # map-aware dispatcher there would only change the numeric path and cost.
    layer_can_take_update = not any(
        extra is not None for extra in (w13_bias, w2_bias, w13_weight_scale, w2_weight_scale)
    )
    dispatcher_cls = resolve_dispatcher_class(
        top_k=top_k,
        dispatcher_type=dispatcher_type,
        eplb=eplb_hook is not None and layer_can_take_update,
    )
    use_gmm_finalize_routing = should_use_gmm_finalize_routing(_get_dispatcher_type_name(dispatcher_cls))
    # The hook keeps its layout in force across calls, so the caller's tensors go
    # stale after the first re-placement; prefer the weights that were applied.
    if eplb_hook is not None and hasattr(eplb_hook, "current_weights"):
        w13_weight, w2_weight = eplb_hook.current_weights(w13_weight, w2_weight)
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

    if eplb_hook is not None:
        eplb_hook.on_routing(topk_ids)
        weight_update = eplb_hook.poll_weight_update()
        if weight_update is not None:
            re_placed = _apply_re_placement(
                eplb_hook,
                weight_update,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                w13_weight_scale=w13_weight_scale,
                w2_weight_scale=w2_weight_scale,
            )
            if re_placed is not None:
                moe_weights = re_placed
                if hasattr(eplb_hook, "remember_weights"):
                    eplb_hook.remember_weights(weight_update)

    # The hook keeps the effective expert layout (global expert id -> rank) in force
    # on every call, not only on the call that received the re-placement, and it also
    # keeps the weight rows that belong to that layout (see `current_weights` above).
    # Weight rows are ordered by ascending expert id, so routing ids need no remapping.
    eplb_device_map = eplb_hook.published_device_map() if eplb_hook is not None else None
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1

    token_dispatch_input = build_token_dispatch_input(
        hidden_states=prepare_output.hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        weights=moe_weights,
        dynamic_scale=prepare_output.dynamic_scale,
        use_gmm_finalize_routing=use_gmm_finalize_routing,
        device_indices_map=eplb_device_map if ep_size > 1 else None,
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
