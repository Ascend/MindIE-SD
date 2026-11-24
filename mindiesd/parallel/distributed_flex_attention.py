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
"""Distributed FlexAttention with explicit Ulysses and Ring process groups.

Only the local Flex kernel is compiled. Communication and coordinate planning
remain eager. FlexAttentionMask owns request-local BlockMasks; no request
tensors or process groups are retained by a process-global mask cache.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from mindiesd.layers.flash_attn.flex_attention import (
    FlexAttentionMask,
    _format_output,
    _load_dsl,
    _map_identity,
    _validate_inputs,
    _validate_map,
    _validate_options,
    _validate_positions,
    _validate_qkv_shapes,
    _validate_query,
)


def _group_size(group: dist.ProcessGroup | None) -> int:
    """Treat an omitted group as local execution, never as the default group."""
    return 1 if group is None else dist.get_world_size(group)


def _group_rank(group: dist.ProcessGroup | None) -> int:
    """Return a rank local to the explicitly supplied group."""
    return 0 if group is None else dist.get_rank(group)


def _validate_peer_contracts(
    mask: FlexAttentionMask,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix: int,
    ulysses_group: dist.ProcessGroup | None,
    ring_group: dist.ProcessGroup | None,
    explicit_query_positions: bool = False,
    local_error: Exception | None = None,
) -> None:
    """Check fixed-buffer peer contracts once per request execution plan.

    Only a fixed-size shape/dtype descriptor is gathered. Error flags cross
    both orthogonal groups before raising so a mismatch in one Ulysses row
    does not leave another row blocked in Ring communication. The initial
    descriptor synchronization is skipped for matching layer shapes. A small
    local-validation/cache-hit agreement is still required on every distributed call:
    a rank-local fast return could strand a peer entering validation.
    """
    u_size, r_size = _group_size(ulysses_group), _group_size(ring_group)
    if u_size == r_size == 1:
        if local_error is not None:
            raise local_error
        return
    plan = (
        None
        if local_error is not None
        else (
            ulysses_group,
            ring_group,
            tuple(query.shape),
            tuple(key.shape),
            tuple(value.shape),
            query.dtype,
            str(query.device),
            prefix,
            explicit_query_positions,
        )
    )
    agreement = torch.tensor(
        [int(local_error is None), int(local_error is None and plan in mask._validated_plans)],
        dtype=torch.int32,
        device=query.device,
    )
    for group in (ulysses_group, ring_group):
        if _group_size(group) > 1:
            dist.all_reduce(agreement, op=dist.ReduceOp.MIN, group=group)
    valid, cache_hit = agreement.cpu().tolist()
    if not valid:
        if local_error is not None:
            raise local_error
        raise ValueError("FlexAttention input validation failed on another rank.")
    if cache_hit:
        return
    descriptor = torch.tensor(
        [
            *query.shape,
            *key.shape,
            *value.shape,
            (torch.float16, torch.bfloat16).index(query.dtype),
            prefix,
            u_size,
            _group_rank(ulysses_group),
            r_size,
            _group_rank(ring_group),
            int(explicit_query_positions),
        ],
        dtype=torch.int64,
        device=query.device,
    )
    mismatch = False
    for group, columns in (
        (ulysses_group, (*range(15), 16, 17)),
        (ring_group, (0, 2, 3, *range(4, 17))),
    ):
        if _group_size(group) > 1:
            gathered = [torch.empty_like(descriptor) for _ in range(_group_size(group))]
            dist.all_gather(gathered, descriptor, group=group)
            peers = [peer.cpu().tolist() for peer in gathered]
            reference = [peers[0][index] for index in columns]
            mismatch |= any([peer[index] for index in columns] != reference for peer in peers[1:])
            if group is ring_group and len({peer[1] for peer in peers}) > 1:
                # Default rank * local_length coordinates only describe equal
                # query shards. Unequal regions need caller-supplied maps.
                mismatch |= any(not peer[18] for peer in peers)
    failure = torch.tensor(int(mismatch), dtype=torch.int32, device=query.device)
    for group in (ulysses_group, ring_group):
        if _group_size(group) > 1:
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group)
    if failure.item():
        raise ValueError(
            "FlexAttention peer shape/dtype or parallel-grid contracts differ. "
            "Strict Ulysses requires equal peer tensor shapes; Ring requires equal K/V shapes "
            "and compatible Q heads. Unequal Ring Q lengths require explicit query_positions on all peers. "
            "All peers must agree on joint KV and orthogonal group sizes."
        )
    mask._validated_plans.add(plan)


def _ulysses_forward(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Exchange local sequence shards for local head shards in strict mode."""
    size = _group_size(group)
    if size == 1:
        return tensor
    batch, sequence, heads, width = tensor.shape
    packed = tensor.reshape(batch, sequence, size, heads // size, width).permute(2, 0, 1, 3, 4).contiguous()
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, group=group)
    return received.permute(1, 0, 2, 3, 4).reshape(batch, size * sequence, heads // size, width)


def _ulysses_reverse(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Restore original local sequence shards after local attention."""
    size = _group_size(group)
    if size == 1:
        return tensor
    batch, sequence, heads, width = tensor.shape
    if sequence % size:
        raise ValueError("Post-Ulysses query length must be divisible by the Ulysses group size.")
    packed = tensor.reshape(batch, size, sequence // size, heads, width).permute(1, 0, 2, 3, 4).contiguous()
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, group=group)
    return received.permute(1, 2, 0, 3, 4).reshape(batch, sequence // size, size * heads, width)


def _gather_positions(positions: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Exchange logical positions in the same source order as Ulysses Q/KV."""
    if _group_size(group) == 1:
        return positions
    gathered = [torch.empty_like(positions) for _ in range(_group_size(group))]
    dist.all_gather(gathered, positions.contiguous(), group=group)
    return torch.cat(gathered)


def _rotate_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Receive the preceding Ring rank's KV block, retaining buffers until done."""
    size, rank = _group_size(group), _group_rank(group)
    next_rank = dist.get_global_rank(group, (rank + 1) % size)
    previous_rank = dist.get_global_rank(group, (rank - 1) % size)
    sends = (key.contiguous(), value.contiguous(), positions.contiguous())
    received = tuple(torch.empty_like(tensor) for tensor in sends)
    operations = []
    for send, receive in zip(sends, received):
        operations.append(dist.P2POp(dist.isend, send, next_rank, group))
        operations.append(dist.P2POp(dist.irecv, receive, previous_rank, group))
    for request in dist.batch_isend_irecv(operations):
        request.wait()
    return received


def _merge_attention(
    previous_out: torch.Tensor,
    previous_lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized attention blocks using FP32 natural-log LSE.

    Empty rows contribute zero weight even when both blocks have LSE=-inf.
    Valid-row NaNs propagate rather than being silently repaired.
    """
    previous_lse, block_lse = previous_lse.float(), block_lse.float()
    merged_lse = torch.logaddexp(previous_lse, block_lse)
    safe_lse = torch.where(torch.isneginf(merged_lse), 0.0, merged_lse)
    previous_weight = torch.exp(previous_lse - safe_lse).transpose(1, 2).unsqueeze(-1)
    block_weight = torch.exp(block_lse - safe_lse).transpose(1, 2).unsqueeze(-1)
    previous_out = torch.where(torch.isneginf(previous_lse).transpose(1, 2).unsqueeze(-1), 0.0, previous_out.float())
    block_out = torch.where(torch.isneginf(block_lse).transpose(1, 2).unsqueeze(-1), 0.0, block_out.float())
    return previous_out * previous_weight + block_out * block_weight, merged_lse


def _restore_output(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    ulysses_group: dist.ProcessGroup | None,
    layout: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reverse only explicitly requested Ulysses communication and layout."""
    output = _ulysses_reverse(output, ulysses_group)
    if lse is not None:
        lse = _ulysses_reverse(lse.transpose(1, 2).unsqueeze(-1), ulysses_group)
        lse = lse.squeeze(-1).transpose(1, 2)
    return _format_output(output, lse, layout)


def distributed_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: FlexAttentionMask,
    ulysses_group: dist.ProcessGroup | None = None,
    ring_group: dist.ProcessGroup | None = None,
    scale: float | None = None,
    layout: str = "BSND",
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    head_indices: torch.Tensor | None = None,
    joint_key: torch.Tensor | None = None,
    joint_value: torch.Tensor | None = None,
    joint_key_positions: torch.Tensor | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute local, Ulysses, Ring or hybrid inference attention.

    Args:
        query, key, value: Local sequence shards in ``layout``. Strict Ulysses
            requires identical peer sequence lengths and divisible Q/KV head
            counts. Ring peers must have identical K/V shapes and dtype.
        mask: Request-owned logical/global mask and optional score transform.
        ulysses_group, ring_group: Explicit orthogonal PyTorch process groups;
            None disables that communication. This function creates no groups.
            A framework that already exchanged Q/K/V must omit ulysses_group.
        scale: Finite FP32-range QK multiplier, defaulting to query head_dim**-0.5.
        layout: BSND or BNSD; output uses the same layout and query dtype.
        query_positions, key_positions: Optional int32/int64 input sequence
            maps on the tensor device. -1 marks padding. Defaults assume
            contiguous shards ordered by Ring rank, then Ulysses rank, with
            replicated joint KV as a prefix. Maps are required for other
            orders, disjoint sequences, or caller-managed prefix query rows.
        head_indices: Optional non-negative logical/global query-head IDs before Ulysses;
            default is arange(query_heads). Ulysses slices this map by heads.
        joint_key, joint_value: Replicated KV prefix in input layout, included
            exactly once, outside the Ring. Q is never implicitly prefixed.
        joint_key_positions: Optional explicit logical positions of joint KV.
        enable_gqa: Permit query_heads to be an integer multiple of KV heads.
        return_lse: Return (output, FP32 natural-log LSE [B,Hq,Sq]).

    Captured tensors and explicit maps must not change during mask reuse.
    All ranks must call with consistent groups/contracts and in the same
    order. Communications are synchronous; no overlap/quantization is implied.
    With valid NPU devices and groups, local validation/import failures are
    agreed before payload exchange, including on cached plans. Other ranks
    raise ValueError if a peer failed. Device, compiler and collective failures
    still require caller-managed group termination.
    """
    _validate_query(query)
    u_size, u_rank = _group_size(ulysses_group), _group_rank(ulysses_group)
    r_size, r_rank = _group_size(ring_group), _group_rank(ring_group)
    if u_size > 1 and r_size > 1 and ulysses_group is ring_group:
        raise ValueError("Ulysses and Ring require separate orthogonal process groups.")
    # A valid query device and matching process groups are required to report
    # errors collectively. Host-visible validation and optional imports finish
    # before any tensor payload is exchanged, including on cached plans.
    local_error = None
    prefix = q_heads = kv_heads = 0
    dsl = None
    try:
        _validate_inputs(query, key, value, layout)
        _validate_options(mask, query, scale, enable_gqa, return_lse)
        if (joint_key is None) != (joint_value is None):
            raise ValueError("joint_key and joint_value must be provided together.")
        if joint_key is None and joint_key_positions is not None:
            raise ValueError("joint_key_positions requires joint_key and joint_value.")
        if joint_key is not None and (
            not isinstance(joint_key, torch.Tensor)
            or not isinstance(joint_value, torch.Tensor)
            or joint_key.ndim != 4
            or joint_value.ndim != 4
        ):
            raise ValueError("joint_key and joint_value must be four-dimensional tensors.")
        if layout == "BNSD":
            query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
            if joint_key is not None:
                joint_key, joint_value = joint_key.transpose(1, 2), joint_value.transpose(1, 2)
        q_heads, kv_heads = _validate_qkv_shapes(query, key, value, enable_gqa)
        dsl = _load_dsl(query.device)
        if q_heads % u_size or kv_heads % u_size:
            raise ValueError("Strict Ulysses requires both Q and KV heads divisible by its group size.")
        if joint_key is not None:
            if (
                joint_key.shape[:3] != joint_value.shape[:3]
                or joint_key.shape[0] != query.shape[0]
                or joint_key.shape[2:] != key.shape[2:]
                or joint_value.shape[2:] != value.shape[2:]
            ):
                raise ValueError("Joint K/V batch, heads and head dimensions must match the main K/V.")
            if any(tensor.device != query.device or tensor.dtype != query.dtype for tensor in (joint_key, joint_value)):
                raise ValueError("Joint K/V must have the same device and dtype as Q/K/V.")
        prefix = 0 if joint_key is None else joint_key.shape[1]
        _validate_positions(query, key, query_positions, key_positions, head_indices)
        _validate_map("joint_key_positions", joint_key_positions, prefix, query.device)
    except (TypeError, ValueError, ImportError) as error:
        local_error = error
    _validate_peer_contracts(
        mask,
        query,
        key,
        value,
        prefix,
        ulysses_group,
        ring_group,
        query_positions is not None,
        local_error=local_error,
    )
    source_maps = tuple(
        tensor for tensor in (query_positions, key_positions, head_indices, joint_key_positions) if tensor is not None
    )
    # Group objects are retained in request-owned keys to prevent identity reuse
    # if a caller tears down and recreates groups before clearing this mask.
    cache_key = (
        ulysses_group,
        ring_group,
        u_size,
        u_rank,
        r_size,
        r_rank,
        tuple(query.shape),
        tuple(key.shape),
        prefix,
        *(_map_identity(tensor) for tensor in (query_positions, key_positions, head_indices, joint_key_positions)),
    )
    if query_positions is None:
        start = prefix + (r_rank * u_size + u_rank) * query.shape[1]
        query_positions = torch.arange(query.shape[1], device=query.device) + start
    if key_positions is None:
        start = prefix + (r_rank * u_size + u_rank) * key.shape[1]
        key_positions = torch.arange(key.shape[1], device=query.device) + start
    if head_indices is None:
        head_indices = torch.arange(q_heads, device=query.device)
    query, key, value = (_ulysses_forward(tensor, ulysses_group) for tensor in (query, key, value))
    query_positions = _gather_positions(query_positions.to(torch.int64), ulysses_group)
    key_positions = _gather_positions(key_positions.to(torch.int64), ulysses_group)
    head_indices = head_indices[u_rank * (q_heads // u_size) : (u_rank + 1) * (q_heads // u_size)]

    if r_size == 1 and not prefix:
        prepared = mask._prepare(
            dsl,
            query_positions,
            key_positions,
            head_indices,
            batch_size=query.shape[0],
            cache_key=(cache_key, "local"),
            owners=source_maps,
        )
        result = dsl.attention(query, key, value, prepared, scale, return_lse)
        output, lse = result if return_lse else (result, None)
        return _restore_output(output, lse, ulysses_group, layout)

    accumulated_out = accumulated_lse = None
    if joint_key is not None and prefix:
        heads_per_rank = kv_heads // u_size
        joint_key = joint_key[:, :, u_rank * heads_per_rank : (u_rank + 1) * heads_per_rank]
        joint_value = joint_value[:, :, u_rank * heads_per_rank : (u_rank + 1) * heads_per_rank]
        if joint_key_positions is None:
            joint_key_positions = torch.arange(prefix, device=query.device)
        prepared = mask._prepare(
            dsl,
            query_positions,
            joint_key_positions,
            head_indices,
            batch_size=query.shape[0],
            cache_key=(cache_key, "joint"),
            owners=source_maps,
        )
        accumulated_out, accumulated_lse = dsl.attention(query, joint_key, joint_value, prepared, scale, True)

    for step in range(r_size):
        owner = (r_rank - step) % r_size
        prepared = mask._prepare(
            dsl,
            query_positions,
            key_positions,
            head_indices,
            batch_size=query.shape[0],
            cache_key=(cache_key, "ring", owner),
            owners=source_maps,
        )
        block_out, block_lse = dsl.attention(query, key, value, prepared, scale, True)
        if accumulated_out is None:
            accumulated_out, accumulated_lse = block_out, block_lse
        else:
            accumulated_out, accumulated_lse = _merge_attention(accumulated_out, accumulated_lse, block_out, block_lse)
        if step + 1 < r_size:
            key, value, key_positions = _rotate_kv(key, value, key_positions, ring_group)

    accumulated_out = torch.where(
        torch.isneginf(accumulated_lse).transpose(1, 2).unsqueeze(-1), 0.0, accumulated_out
    ).to(query.dtype)
    return _restore_output(accumulated_out, accumulated_lse if return_lse else None, ulysses_group, layout)
