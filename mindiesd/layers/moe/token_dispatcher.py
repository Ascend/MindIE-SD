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

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from ...utils import ParametersInvalid
from .comm_ops import (
    all_gather,
    all_reduce,
    all_to_all_single,
    reduce_scatter,
)
from .moe_dataclass import (
    MoEDynamicCombineMetadata,
    MoEPrepareInput,
    MoEPrepareOutput,
    MoEStaticCombineMetadata,
    MoETokenDispatchOutput,
)
from .moe_context import (
    MoECommType,
    dynamic_quant,
    get_init_routing_quant_mode,
    get_moe_comm_type,
    get_moe_group,
    is_moe_int_quant,
    is_moe_quant,
)


class TokenDispatcher(ABC):
    @classmethod
    @abstractmethod
    def prepare(cls, prepare_input: MoEPrepareInput):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dispatch(cls, token_dispatch_input):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def combine(cls, hidden_states, combine_metadata):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def finalize(
        cls,
        routed_out,
        original_shape,
        inputs_sharded=False,
        reduce_routed_out=True,
    ):
        raise NotImplementedError


class StaticDispatcher(TokenDispatcher):
    @classmethod
    def prepare(cls, prepare_input: MoEPrepareInput):
        moe_group = get_moe_group()
        hidden_states = prepare_input.hidden_states
        router_logits = prepare_input.router_logits
        inputs_sharded = prepare_input.inputs_sharded
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_router = router_logits.reshape(-1, router_logits.shape[-1])

        if inputs_sharded and moe_group is not None:
            # Static dispatch gathers inputs sharded along the token dimension
            # across the current MoE group into the unsharded token view it expects.
            dynamic_scale = None
            if is_moe_quant():
                flat_hidden, dynamic_scale = dynamic_quant(flat_hidden)
            flat_hidden = all_gather(flat_hidden, moe_group)
            flat_router = all_gather(flat_router, moe_group)
            if dynamic_scale is not None:
                dynamic_scale = all_gather(dynamic_scale, moe_group)
            return MoEPrepareOutput(
                hidden_states=flat_hidden,
                router_logits=flat_router,
                original_shape=hidden_states.shape,
                mlp_output_dtype=hidden_states.dtype,
                dynamic_scale=dynamic_scale,
            )

        return MoEPrepareOutput(
            hidden_states=flat_hidden,
            router_logits=flat_router,
            original_shape=hidden_states.shape,
            mlp_output_dtype=hidden_states.dtype,
        )

    @classmethod
    def dispatch(cls, token_dispatch_input):
        hidden_states = token_dispatch_input.hidden_states
        topk_ids = token_dispatch_input.topk_ids
        topk_weights = token_dispatch_input.topk_weights
        dynamic_scale = token_dispatch_input.dynamic_scale
        restore_shape = hidden_states.shape
        num_tokens = hidden_states.shape[0]
        active_expert_range = cls._get_active_expert_range(
            num_experts=token_dispatch_input.num_experts,
            ep_group=get_moe_group() if get_moe_comm_type() == MoECommType.EP else None,
        )
        if active_expert_range[0] != 0 or active_expert_range[1] != token_dispatch_input.num_experts:
            local_topk_weights = cls._mask_nonlocal_topk_weights(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                active_expert_range=active_expert_range,
            )
        else:
            local_topk_weights = topk_weights
        output = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=dynamic_scale,
            active_num=num_tokens * token_dispatch_input.top_k,
            expert_num=token_dispatch_input.num_experts,
            expert_tokens_num_type=0,
            expert_tokens_num_flag=True,
            active_expert_range=active_expert_range,
            quant_mode=get_init_routing_quant_mode(dynamic_scale),
            row_idx_type=1 if token_dispatch_input.use_gmm_finalize_routing else 0,
        )
        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = output[:4]
        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            dynamic_scale=dynamic_scale if is_moe_quant() else None,
            group_list=expert_tokens.to(torch.int64),
            group_list_type=0,
            combine_metadata=MoEStaticCombineMetadata(
                topk_weights=local_topk_weights,
                restore_shape=restore_shape,
                expanded_row_idx=expanded_row_idx,
            ),
        )

    @classmethod
    def combine(cls, hidden_states, combine_metadata):
        metadata = combine_metadata
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=torch.abs(metadata.expanded_row_idx),
            probs=metadata.topk_weights.to(hidden_states.dtype),
        )
        return output.view(metadata.restore_shape)

    @classmethod
    def finalize(
        cls,
        routed_out,
        original_shape,
        inputs_sharded=False,
        reduce_routed_out=True,
    ):
        moe_group = get_moe_group()
        if moe_group is None:
            return routed_out.reshape(original_shape)

        if inputs_sharded:
            # Static dispatch produced unsharded routed_out; return this rank's
            # token shard in the current MoE group.
            return reduce_scatter(routed_out, moe_group).reshape(original_shape)

        if reduce_routed_out:
            all_reduce(routed_out, moe_group)
        return routed_out.reshape(original_shape)

    @classmethod
    def _get_active_expert_range(cls, num_experts, ep_group=None):
        ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        if ep_size <= 1:
            return [0, num_experts]
        ep_rank = dist.get_rank(ep_group)
        experts_per_rank = num_experts // ep_size
        start = ep_rank * experts_per_rank
        return [start, start + experts_per_rank]

    @classmethod
    def _mask_nonlocal_topk_weights(cls, topk_ids, topk_weights, active_expert_range):
        start, end = active_expert_range
        local_mask = (topk_ids >= start) & (topk_ids < end)
        return topk_weights * local_mask


class DynamicDispatcher(TokenDispatcher):
    _split_cpu_buffers = {}
    _split_copy_events = {}

    @classmethod
    def prepare(cls, prepare_input: MoEPrepareInput):
        moe_group = get_moe_group()
        hidden_states = prepare_input.hidden_states
        router_logits = prepare_input.router_logits
        inputs_sharded = prepare_input.inputs_sharded
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_router = router_logits.reshape(-1, router_logits.shape[-1])
        if inputs_sharded:
            # Inputs sharded along the token dimension across the current MoE
            # group already match dynamic dispatch's rank-local layout.
            return MoEPrepareOutput(
                hidden_states=flat_hidden,
                router_logits=flat_router,
                original_shape=hidden_states.shape,
                mlp_output_dtype=hidden_states.dtype,
            )

        rank = dist.get_rank(moe_group)
        world_size = dist.get_world_size(moe_group)
        original_num_tokens = flat_hidden.shape[0]
        pad_size = (world_size - original_num_tokens % world_size) % world_size
        if pad_size > 0:
            flat_hidden = F.pad(flat_hidden, (0, 0, 0, pad_size))
            flat_router = F.pad(flat_router, (0, 0, 0, pad_size))
        hidden_shards = torch.tensor_split(flat_hidden, world_size, dim=0)
        router_shards = torch.tensor_split(flat_router, world_size, dim=0)
        return MoEPrepareOutput(
            hidden_states=hidden_shards[rank].contiguous(),
            router_logits=router_shards[rank].contiguous(),
            original_shape=(hidden_states.shape, original_num_tokens),
            mlp_output_dtype=hidden_states.dtype,
        )

    @classmethod
    def dispatch(cls, token_dispatch_input):
        hidden_states = token_dispatch_input.hidden_states
        topk_ids = token_dispatch_input.topk_ids
        topk_weights = token_dispatch_input.topk_weights
        moe_group = get_moe_group()

        (
            permuted_local_tokens,
            dynamic_scale,
            reversed_local_mapping,
            input_splits,
            output_splits,
            global_token_counts_per_local_expert,
            split_copy_event,
            hidden_shape,
        ) = cls._dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            num_experts=token_dispatch_input.num_experts,
            local_num_experts=token_dispatch_input.local_num_experts,
            ep_group=moe_group,
        )

        tokens_per_expert = global_token_counts_per_local_expert.sum(dim=0)
        expert_ids_per_ep_rank = None
        if token_dispatch_input.local_num_experts > 1:
            expert_ids_per_ep_rank = torch.arange(
                token_dispatch_input.num_experts,
                dtype=torch.int32,
                device=topk_ids.device,
            )
            expert_ids_per_ep_rank = expert_ids_per_ep_rank % token_dispatch_input.local_num_experts

        split_copy_event.synchronize()
        input_splits = input_splits.tolist()
        output_splits = output_splits.tolist()
        if dynamic_scale is not None:
            dynamic_scale = all_to_all_single(dynamic_scale, output_splits, input_splits, moe_group)
        global_tokens = all_to_all_single(
            permuted_local_tokens,
            output_splits,
            input_splits,
            moe_group,
        )
        reversed_global_mapping = None
        if expert_ids_per_ep_rank is not None:
            global_local_expert_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank,
                global_token_counts_per_local_expert.reshape(-1).to(torch.int32),
                output_size=global_tokens.shape[0],
            )
            if dynamic_scale is not None:
                scale_was_1d = dynamic_scale.dim() == 1
                scale_for_permute = dynamic_scale.unsqueeze(-1) if scale_was_1d else dynamic_scale
                dynamic_scale, _ = torch_npu.npu_moe_token_permute(
                    scale_for_permute,
                    global_local_expert_indices,
                )
                if scale_was_1d:
                    dynamic_scale = dynamic_scale.squeeze(-1)
            global_tokens, reversed_global_mapping = torch_npu.npu_moe_token_permute(
                global_tokens,
                global_local_expert_indices,
            )

        return MoETokenDispatchOutput(
            hidden_states=global_tokens,
            dynamic_scale=dynamic_scale,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEDynamicCombineMetadata(
                topk_weights=topk_weights,
                hidden_shape=hidden_shape,
                input_splits=input_splits,
                output_splits=output_splits,
                local_unpermute_indices=reversed_local_mapping,
                global_unpermute_indices=reversed_global_mapping,
            ),
        )

    @classmethod
    def combine(cls, hidden_states, combine_metadata):
        metadata = combine_metadata
        if metadata.global_unpermute_indices is not None:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states,
                metadata.global_unpermute_indices,
            )
        moe_group = get_moe_group()
        local_tokens = all_to_all_single(hidden_states, metadata.input_splits, metadata.output_splits, moe_group)
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=local_tokens,
            sorted_indices=metadata.local_unpermute_indices.to(torch.int32),
            probs=metadata.topk_weights.to(local_tokens.dtype),
            restore_shape=metadata.hidden_shape,
        )
        return output.view(metadata.hidden_shape)

    @classmethod
    def _dispatch_preprocess(cls, hidden_states, topk_ids, num_experts, local_num_experts, ep_group):
        hidden_shape = hidden_states.shape
        (
            input_splits,
            output_splits,
            global_token_counts_per_local_expert,
            split_copy_event,
            num_out_tokens,
        ) = cls._preprocess(
            topk_ids=topk_ids,
            num_experts=num_experts,
            local_num_experts=local_num_experts,
            ep_group=ep_group,
        )
        permuted_tokens, reversed_local_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=num_out_tokens,
        )
        dynamic_scale = None
        if is_moe_int_quant():
            permuted_tokens, dynamic_scale = dynamic_quant(permuted_tokens)
        return (
            permuted_tokens,
            dynamic_scale,
            reversed_local_mapping,
            input_splits,
            output_splits,
            global_token_counts_per_local_expert,
            split_copy_event,
            hidden_shape,
        )

    @classmethod
    def _preprocess(cls, topk_ids, num_experts, local_num_experts, ep_group):
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
        if local_num_experts * ep_size != num_experts:
            raise ParametersInvalid(
                "Dynamic MoE currently requires evenly partitioned experts, "
                f"but got num_experts={num_experts}, local_num_experts={local_num_experts}, ep_size={ep_size}."
            )

        num_local_tokens_per_expert = torch.histc(
            topk_ids,
            bins=num_experts,
            min=0,
            max=num_experts,
        )
        input_splits = num_local_tokens_per_expert.reshape(ep_size, local_num_experts).sum(dim=1)
        input_splits_cpu = cls._copy_split_sizes_to_cpu(input_splits, ep_size, "input")
        num_global_tokens_per_expert = all_gather(
            num_local_tokens_per_expert,
            ep_group,
        ).reshape(ep_size, num_experts)

        local_start = ep_rank * local_num_experts
        local_end = local_start + local_num_experts
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, local_start:local_end]
        output_splits = num_global_tokens_per_local_expert.sum(dim=-1)
        output_splits_cpu = cls._copy_split_sizes_to_cpu(output_splits, ep_size, "output")
        split_copy_event = cls._get_split_copy_event(topk_ids.device)
        split_copy_event.record(torch.npu.current_stream())

        return (
            input_splits_cpu,
            output_splits_cpu,
            num_global_tokens_per_local_expert,
            split_copy_event,
            topk_ids.numel(),
        )

    @classmethod
    def _copy_split_sizes_to_cpu(cls, split_sizes, ep_size, name):
        split_sizes = split_sizes.to(dtype=torch.int32)
        buffer = cls._get_split_cpu_buffer(ep_size, name)
        buffer.copy_(split_sizes, non_blocking=True)
        return buffer

    @classmethod
    def _get_split_cpu_buffer(cls, ep_size, name):
        key = (name, ep_size)
        buffer = cls._split_cpu_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(ep_size, dtype=torch.int32, device=torch.device("cpu"), pin_memory=True)
            cls._split_cpu_buffers[key] = buffer
        return buffer

    @classmethod
    def _get_split_copy_event(cls, device):
        event = cls._split_copy_events.get(device)
        if event is None:
            event = torch.npu.Event()
            cls._split_copy_events[device] = event
        return event

    @classmethod
    def finalize(
        cls,
        routed_out,
        original_shape,
        inputs_sharded=False,
        reduce_routed_out=True,
    ):
        moe_group = get_moe_group()
        if inputs_sharded:
            return routed_out.reshape(original_shape)

        original_shape, original_num_tokens = original_shape
        routed_out = all_gather(routed_out, moe_group)
        routed_out = routed_out[:original_num_tokens]
        return routed_out.reshape(original_shape)
