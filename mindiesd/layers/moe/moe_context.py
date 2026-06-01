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
from enum import Enum

import torch
import torch.distributed as dist

from ...utils import ParametersInvalid
from .moe_dataclass import (
    MoEMlpComputeInput,
    MoEPrepareInput,
    MoERoutingInput,
    MoETokenDispatchInput,
)


def validate_moe_inputs(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
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
) -> None:
    """Validate public MoE inputs at the beginning of the MoE invocation."""
    if not isinstance(hidden_states, torch.Tensor):
        raise ParametersInvalid(f"hidden_states must be a torch.Tensor, but got {type(hidden_states)}.")
    if not isinstance(router_logits, torch.Tensor):
        raise ParametersInvalid(f"router_logits must be a torch.Tensor, but got {type(router_logits)}.")
    if not isinstance(w13_weight, torch.Tensor):
        raise ParametersInvalid(f"w13_weight must be a torch.Tensor, but got {type(w13_weight)}.")
    if not isinstance(w2_weight, torch.Tensor):
        raise ParametersInvalid(f"w2_weight must be a torch.Tensor, but got {type(w2_weight)}.")
    if w13_bias is not None and not isinstance(w13_bias, torch.Tensor):
        raise ParametersInvalid(f"w13_bias must be a torch.Tensor or None, but got {type(w13_bias)}.")
    if w2_bias is not None and not isinstance(w2_bias, torch.Tensor):
        raise ParametersInvalid(f"w2_bias must be a torch.Tensor or None, but got {type(w2_bias)}.")

    if not isinstance(num_experts, int) or isinstance(num_experts, bool):
        raise ParametersInvalid(f"num_experts must be an integer, but got {type(num_experts)}.")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ParametersInvalid(f"top_k must be an integer, but got {type(top_k)}.")
    if not isinstance(k_group, int) or isinstance(k_group, bool):
        raise ParametersInvalid(f"k_group must be an integer, but got {type(k_group)}.")
    if not isinstance(group_count, int) or isinstance(group_count, bool):
        raise ParametersInvalid(f"group_count must be an integer, but got {type(group_count)}.")
    if not isinstance(routed_scaling_factor, float):
        raise ParametersInvalid(f"routed_scaling_factor must be a float, but got {type(routed_scaling_factor)}.")

    if not isinstance(tokens_full, bool):
        raise ParametersInvalid(f"tokens_full must be a bool, but got {type(tokens_full)}.")
    if not isinstance(renormalize, bool):
        raise ParametersInvalid(f"renormalize must be a bool, but got {type(renormalize)}.")
    if not isinstance(reduce_results, bool):
        raise ParametersInvalid(f"reduce_results must be a bool, but got {type(reduce_results)}.")
    if dispatcher_type not in (None, "static", "dynamic"):
        raise ParametersInvalid(f"dispatcher_type must be None, 'static', or 'dynamic', but got {dispatcher_type}.")
    if group_select_mode not in (0, 1):
        raise ParametersInvalid(f"group_select_mode must be 0 or 1, but got {group_select_mode}.")
    if routing_method not in ("softmax", "sigmoid"):
        raise ParametersInvalid(f"routing_method must be 'softmax' or 'sigmoid', but got {routing_method}.")
    if custom_routing_function is not None and not callable(custom_routing_function):
        raise ParametersInvalid("custom_routing_function must be callable if provided.")

    if not 1 <= top_k <= num_experts:
        raise ParametersInvalid(f"top_k must be in [1, num_experts], but got top_k={top_k}, num_experts={num_experts}.")
    if k_group < 1:
        raise ParametersInvalid(f"k_group must be positive, but got {k_group}.")
    if group_count < 1:
        raise ParametersInvalid(f"group_count must be positive, but got {group_count}.")
    if k_group > group_count:
        raise ParametersInvalid(f"k_group={k_group} exceeds group_count={group_count}.")

    if tp_group is not None and not isinstance(tp_group, dist.ProcessGroup):
        raise ParametersInvalid(f"tp_group must be a dist.ProcessGroup or None, but got {type(tp_group)}.")
    if ep_group is not None and not isinstance(ep_group, dist.ProcessGroup):
        raise ParametersInvalid(f"ep_group must be a dist.ProcessGroup or None, but got {type(ep_group)}.")

    if hidden_states.dim() < 2:
        raise ParametersInvalid(f"hidden_states must be at least 2D, but got shape {tuple(hidden_states.shape)}.")
    if router_logits.dim() < 2:
        raise ParametersInvalid(f"router_logits must be at least 2D, but got shape {tuple(router_logits.shape)}.")
    if w13_weight.dim() != 3:
        raise ParametersInvalid(f"w13_weight must be a 3D tensor, but got shape {tuple(w13_weight.shape)}.")
    if w2_weight.dim() != 3:
        raise ParametersInvalid(
            "w2_weight must be a 3D tensor with shape "
            "[local_experts, intermediate_size, hidden_size], "
            f"but got shape {tuple(w2_weight.shape)}."
        )

    if router_logits.shape[-1] != num_experts:
        raise ParametersInvalid(
            f"The last dim of router_logits must match num_experts={num_experts}, "
            f"but got shape {tuple(router_logits.shape)}."
        )
    if hidden_states.shape[:-1] != router_logits.shape[:-1]:
        raise ParametersInvalid(
            "hidden_states and router_logits must have the same leading dimensions, "
            f"but got hidden_states={tuple(hidden_states.shape)}, "
            f"router_logits={tuple(router_logits.shape)}."
        )
    if w13_weight.shape[0] < 1:
        raise ParametersInvalid(f"local_num_experts must be positive, but got {w13_weight.shape[0]}.")
    if w13_weight.shape[0] != w2_weight.shape[0]:
        raise ParametersInvalid(
            "w13_weight and w2_weight must have the same number of local experts, "
            f"but got w13_weight={tuple(w13_weight.shape)}, w2_weight={tuple(w2_weight.shape)}."
        )
    if hidden_states.shape[-1] != w13_weight.shape[1]:
        raise ParametersInvalid(
            "hidden size must match w13_weight input dimension, "
            f"but got hidden_states={hidden_states.shape[-1]}, "
            f"w13_weight={w13_weight.shape[1]}."
        )
    if w13_weight.shape[1] != w2_weight.shape[2]:
        raise ParametersInvalid(
            "w13_weight input dimension must match w2_weight output dimension, "
            f"but got w13_weight={w13_weight.shape[1]}, w2_weight={w2_weight.shape[2]}."
        )
    if w13_weight.shape[2] != 2 * w2_weight.shape[1]:
        raise ParametersInvalid(
            "w13_weight output dimension must be twice w2_weight input dimension, "
            f"but got w13_weight={w13_weight.shape[2]}, w2_weight={w2_weight.shape[1]}."
        )
    if w13_bias is not None and w13_bias.shape != w13_weight.shape[:1] + w13_weight.shape[2:]:
        raise ParametersInvalid(
            "w13_bias shape must match w13_weight expert and output dimensions, "
            f"but got bias={tuple(w13_bias.shape)}, weight={tuple(w13_weight.shape)}."
        )
    if w2_bias is not None and w2_bias.shape != w2_weight.shape[:1] + w2_weight.shape[2:]:
        raise ParametersInvalid(
            "w2_bias shape must match w2_weight expert and output dimensions, "
            f"but got bias={tuple(w2_bias.shape)}, weight={tuple(w2_weight.shape)}."
        )

    if ep_group is not None:
        ep_size = dist.get_world_size(ep_group)
        if num_experts % ep_size != 0:
            raise ParametersInvalid(
                "num_experts must be evenly divisible by ep_size, "
                f"but got num_experts={num_experts}, ep_size={ep_size}."
            )
    if num_experts % group_count != 0:
        raise ParametersInvalid(
            "num_experts must be evenly divisible by group_count, "
            f"but got num_experts={num_experts}, group_count={group_count}."
        )
    experts_per_group = num_experts // group_count
    if group_select_mode == 1 and experts_per_group < 2:
        raise ParametersInvalid(
            "group_select_mode=1 requires at least two experts per group, "
            f"but got experts_per_group={experts_per_group}."
        )
    if top_k > k_group * experts_per_group:
        raise ParametersInvalid(
            "top_k cannot exceed the number of experts in selected groups, "
            f"but got top_k={top_k}, k_group={k_group}, experts_per_group={experts_per_group}."
        )


class MoECommType(Enum):
    """Resolved MoE communication mode."""

    NONE = "none"
    TP = "tp"
    EP = "ep"


_TP_GROUP = None
_EP_GROUP = None
_MOE_COMM_TYPE = MoECommType.NONE


def set_moe_comm_context(tp_group=None, ep_group=None) -> None:
    """Set process-group context for the current MoE invocation."""
    global _TP_GROUP, _EP_GROUP, _MOE_COMM_TYPE
    _TP_GROUP = tp_group
    _EP_GROUP = ep_group
    if ep_group is not None and dist.get_world_size(ep_group) > 1:
        _MOE_COMM_TYPE = MoECommType.EP
    elif tp_group is not None and dist.get_world_size(tp_group) > 1:
        _MOE_COMM_TYPE = MoECommType.TP
    else:
        _MOE_COMM_TYPE = MoECommType.NONE


def get_moe_comm_type() -> MoECommType:
    """Return the resolved MoE communication mode."""
    return _MOE_COMM_TYPE


def get_moe_group():
    """Return the process group selected for MoE communication."""
    if _MOE_COMM_TYPE == MoECommType.EP:
        return _EP_GROUP
    if _MOE_COMM_TYPE == MoECommType.TP:
        return _TP_GROUP
    return None


def build_prepare_input(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    tokens_full: bool = True,
) -> MoEPrepareInput:
    """Build the prepare-stage input wrapper."""
    return MoEPrepareInput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        tokens_full=tokens_full,
    )


def build_routing_input(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,
    routing_method: str = "softmax",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    custom_routing_function: Callable | None = None,
) -> MoERoutingInput:
    """Build the expert-selection input wrapper."""
    return MoERoutingInput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        renormalize=renormalize,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        norm_type=0 if routing_method == "softmax" else 1,
        routed_scaling_factor=routed_scaling_factor,
        custom_routing_function=custom_routing_function,
    )


def build_token_dispatch_input(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    top_k: int,
    w13_weight: torch.Tensor,
) -> MoETokenDispatchInput:
    """Build the token-dispatch input wrapper."""
    return MoETokenDispatchInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_num_experts=w13_weight.shape[0],
    )


def build_mlp_compute_input(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> MoEMlpComputeInput:
    """Build the grouped-MLP input wrapper."""
    return MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=group_list_type,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )
