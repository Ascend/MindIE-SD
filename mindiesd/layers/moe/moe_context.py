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
import torch_npu

from ...quantization.config import QuantConfig
from ...quantization.mode import QuantAlgorithm
from ...utils import ParametersInvalid
from ...utils.get_platform import NPUDevice, get_npu_device
from .moe_dataclass import (
    MoEMlpComputeInput,
    MoEPrepareInput,
    MoERoutingInput,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    MoEWeights,
)

MOE_INT_QUANT_ALGOS = (QuantAlgorithm.W8A8_DYNAMIC,)
MOE_MXFP_QUANT_ALGOS = (QuantAlgorithm.W8A8_MXFP8,)
MOE_SUPPORTED_QUANT_ALGOS = (QuantAlgorithm.NO_QUANT, *MOE_INT_QUANT_ALGOS, *MOE_MXFP_QUANT_ALGOS)


def validate_moe_inputs(
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
) -> QuantAlgorithm:
    """Validate public MoE inputs and return the normalized quantization algorithm."""
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

    if quant_config is not None and not isinstance(quant_config, QuantConfig):
        raise ParametersInvalid(f"quant_config must be a QuantConfig or None, but got {type(quant_config)}.")
    quant_algo = QuantAlgorithm.NO_QUANT if quant_config is None else quant_config.quant_algo or QuantAlgorithm.NO_QUANT
    if quant_algo not in MOE_SUPPORTED_QUANT_ALGOS:
        raise ParametersInvalid(f"Unsupported MoE quantization algorithm: {quant_algo}.")
    if hidden_states.device.type == "npu":
        npu_device = get_npu_device()
        if quant_algo in MOE_INT_QUANT_ALGOS and npu_device not in (NPUDevice.A2, NPUDevice.A3):
            raise ParametersInvalid("MoE integer quantization is only supported on A2 and A3.")
        if quant_algo in MOE_MXFP_QUANT_ALGOS and npu_device != NPUDevice.A5:
            raise ParametersInvalid("MoE MXFP quantization is only supported on A5.")
    if quant_algo == QuantAlgorithm.NO_QUANT and (w13_weight_scale is not None or w2_weight_scale is not None):
        raise ParametersInvalid("w13_weight_scale and w2_weight_scale must be None.")
    if quant_algo != QuantAlgorithm.NO_QUANT and (w13_weight_scale is None or w2_weight_scale is None):
        raise ParametersInvalid("w13_weight_scale and w2_weight_scale must be provided.")

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

    if not isinstance(inputs_sharded, bool):
        raise ParametersInvalid(f"inputs_sharded must be a bool, but got {type(inputs_sharded)}.")
    if not isinstance(renormalize, bool):
        raise ParametersInvalid(f"renormalize must be a bool, but got {type(renormalize)}.")
    if not isinstance(reduce_routed_out, bool):
        raise ParametersInvalid(f"reduce_routed_out must be a bool, but got {type(reduce_routed_out)}.")
    if not isinstance(return_dispatcher_type, bool):
        raise ParametersInvalid(f"return_dispatcher_type must be a bool, but got {type(return_dispatcher_type)}.")
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
    return quant_algo


class MoECommType(Enum):
    """Resolved MoE communication mode."""

    NONE = "none"
    TP = "tp"
    EP = "ep"


_TP_GROUP = None
_EP_GROUP = None
_MOE_COMM_TYPE = MoECommType.NONE
_MOE_QUANT_ALGO = QuantAlgorithm.NO_QUANT


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


def set_moe_quant_algo(quant_algo: QuantAlgorithm = QuantAlgorithm.NO_QUANT) -> None:
    """Store the current MoE quantization algorithm."""
    global _MOE_QUANT_ALGO
    _MOE_QUANT_ALGO = quant_algo


def get_moe_quant_algo() -> QuantAlgorithm:
    """Return the current MoE quantization algorithm."""
    return _MOE_QUANT_ALGO


def should_use_gmm_finalize_routing(dispatcher_type: str) -> bool:
    """Return whether the current MoE invocation can use GMM finalize routing."""
    return dispatcher_type == "static" and get_moe_quant_algo() == QuantAlgorithm.W8A8_MXFP8


def dynamic_quant(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply dynamic quantization according to the current MoE quantization algorithm."""
    if _MOE_QUANT_ALGO == QuantAlgorithm.W8A8_DYNAMIC:
        return torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
    if _MOE_QUANT_ALGO == QuantAlgorithm.W8A8_MXFP8:
        return torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
    raise ParametersInvalid(f"Unsupported MoE quantization algorithm: {_MOE_QUANT_ALGO}.")


def get_init_routing_quant_mode(dynamic_scale: torch.Tensor | None) -> int:
    """Return npu_moe_init_routing_v2 quant mode: -1 no quant, 1 dynamic INT8, 3 MXFP8."""
    if not is_moe_quant() or dynamic_scale is not None:
        return -1
    if _MOE_QUANT_ALGO == QuantAlgorithm.W8A8_DYNAMIC:
        return 1
    if _MOE_QUANT_ALGO == QuantAlgorithm.W8A8_MXFP8:
        return 3
    raise ParametersInvalid(f"Unsupported MoE quantization algorithm: {_MOE_QUANT_ALGO}.")


def is_moe_quant() -> bool:
    """Return whether the current MoE path uses quantization."""
    return _MOE_QUANT_ALGO != QuantAlgorithm.NO_QUANT


def is_moe_int_quant() -> bool:
    """Return whether the current MoE path uses integer quantization."""
    return _MOE_QUANT_ALGO in MOE_INT_QUANT_ALGOS


def is_moe_mxfp_quant() -> bool:
    """Return whether the current MoE path uses MXFP quantization."""
    return _MOE_QUANT_ALGO in MOE_MXFP_QUANT_ALGOS


def set_moe_context(tp_group=None, ep_group=None, quant_algo: QuantAlgorithm = QuantAlgorithm.NO_QUANT) -> None:
    """Set MoE context."""
    set_moe_comm_context(tp_group=tp_group, ep_group=ep_group)
    set_moe_quant_algo(quant_algo)


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
    inputs_sharded: bool = False,
) -> MoEPrepareInput:
    """Build the prepare-stage input wrapper."""
    return MoEPrepareInput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        inputs_sharded=inputs_sharded,
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


def build_moe_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    w13_weight_scale: torch.Tensor | None = None,
    w2_weight_scale: torch.Tensor | None = None,
) -> MoEWeights:
    """Build the expert weight payload."""
    return MoEWeights(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        w13_weight_scale=w13_weight_scale,
        w2_weight_scale=w2_weight_scale,
    )


def build_token_dispatch_input(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    top_k: int,
    weights: MoEWeights,
    dynamic_scale: torch.Tensor | None = None,
    use_gmm_finalize_routing: bool = False,
) -> MoETokenDispatchInput:
    """Build the token-dispatch input wrapper."""
    return MoETokenDispatchInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_num_experts=weights.w13_weight.shape[0],
        dynamic_scale=dynamic_scale,
        use_gmm_finalize_routing=use_gmm_finalize_routing,
    )


def build_mlp_compute_input(
    dispatch_output: MoETokenDispatchOutput,
    weights: MoEWeights,
    mlp_output_dtype: torch.dtype,
) -> MoEMlpComputeInput:
    """Build the grouped-MLP input wrapper."""
    return MoEMlpComputeInput(
        hidden_states=dispatch_output.hidden_states,
        dynamic_scale=dispatch_output.dynamic_scale,
        group_list=dispatch_output.group_list,
        group_list_type=dispatch_output.group_list_type,
        weights=weights,
        mlp_output_dtype=mlp_output_dtype,
    )
