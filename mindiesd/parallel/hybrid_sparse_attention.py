# SPDX-License-Identifier: Apache-2.0
"""CP x Ulysses with global RF blocks and bounded query edge exchange."""

from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .config import POOL_BLOCK

_SUPPORTED_SEQUENCE_WORLDS = (8, 16, 32)


def derive_sparse_parallel_groups(total_ranks, ulysses_size=8):
    total_ranks, ulysses_size = int(total_ranks), int(ulysses_size)
    if total_ranks not in _SUPPORTED_SEQUENCE_WORLDS:
        raise ValueError("sparse sequence world must be one of %r" % (_SUPPORTED_SEQUENCE_WORLDS,))
    if ulysses_size not in (4, 8) or total_ranks % ulysses_size:
        raise ValueError("Ulysses size must be 4 or 8 and divide the sequence world")
    cp_size = total_ranks // ulysses_size
    ulysses_groups = [list(range(begin, begin + ulysses_size)) for begin in range(0, total_ranks, ulysses_size)]
    kv_groups = [list(range(owner, total_ranks, ulysses_size)) for owner in range(ulysses_size)]
    return dict(
        total_ranks=total_ranks,
        ulysses_size=ulysses_size,
        cp_size=cp_size,
        ulysses_groups=ulysses_groups,
        kv_groups=kv_groups,
    )


def resolve_cp_rank(ulysses_group, kv_group, world, rank):
    total = dist.get_world_size()
    plan = derive_sparse_parallel_groups(total, world)
    if dist.get_world_size(ulysses_group) != world:
        raise ValueError("Ulysses group size does not match the requested sparse parallel plan")
    if dist.get_world_size(kv_group) != plan["cp_size"]:
        raise ValueError("KV gather group size does not match the sparse CP size")
    phase = dist.get_rank(kv_group)
    global_rank = dist.get_rank()
    if phase not in range(plan["cp_size"]) or global_rank != phase * world + rank:
        raise ValueError('KV gather rank order does not match contiguous Ulysses sequence ownership')
    if dist.get_rank(ulysses_group) != rank:
        raise ValueError('Ulysses rank does not match the live process group')
    if dist.get_process_group_ranks(ulysses_group) != plan["ulysses_groups"][phase]:
        raise ValueError('Ulysses group does not match its contiguous sequence phase')
    if dist.get_process_group_ranks(kv_group) != plan["kv_groups"][rank]:
        raise ValueError('KV groups must pair matching Ulysses head owners')
    return phase


def derive_query_window(phase, group_length, used_len, phase_count=2):
    if phase not in range(phase_count) or group_length <= 0 or not 0 < used_len <= phase_count * group_length:
        raise ValueError('invalid sparse CP x Ulysses query geometry')
    owned_start = min(phase * group_length, used_len)
    owned_end = min((phase + 1) * group_length, used_len)
    if owned_start == owned_end:
        start = end = owned_start
    else:
        start = owned_start // 128 * 128
        end = min((owned_end + 127) // 128 * 128, used_len)
    return dict(
        owned_start=owned_start,
        owned_end=owned_end,
        start=start,
        end=end,
        output_start=owned_start - start,
        boundary_exchange=bool(start < owned_start or end > owned_end),
        # Collective participation is global, including phases containing only padding.
        boundary_collective=bool(group_length % 128 and group_length < used_len),
        group_length=group_length,
        used_len=used_len,
        phase_count=phase_count,
    )


def expand_query_window(query, boundary, phase, window, phase_count=2):
    owned = window['owned_end'] - window['owned_start']
    query = query[:, :owned]
    if not owned or not window['boundary_exchange']:
        return query
    edge = boundary.shape[1] // 2
    if phase_count == 2 and boundary.shape[1] <= 2 * 128:
        if phase == 0:
            extra = window['end'] - window['owned_end']
            return torch.cat((query, boundary[:, edge : edge + extra]), dim=1)
        prefix = window['output_start']
        return torch.cat((boundary[:, edge - prefix : edge], query), dim=1)
    edge = boundary.shape[1] // (2 * phase_count)
    if window["group_length"] <= edge:
        phases = []
        for owner in range(phase_count):
            width = min(window["group_length"], max(0, window["used_len"] - owner * window["group_length"]))
            phases.append(boundary[:, owner * 2 * edge : owner * 2 * edge + width])
        whole = torch.cat(phases, dim=1)
        return whole[:, window["start"] : window["end"]]
    prefix = window['output_start']
    suffix = window['end'] - window['owned_end']
    pieces = []
    if prefix:
        previous_suffix = (phase - 1) * 2 * edge + 2 * edge
        pieces.append(boundary[:, previous_suffix - prefix : previous_suffix])
    pieces.append(query)
    if suffix:
        next_prefix = (phase + 1) * 2 * edge
        pieces.append(boundary[:, next_prefix : next_prefix + suffix])
    return torch.cat(pieces, dim=1)


def rectangular_rf_mask(query_pool, key_pool, retained, *, scale, sparsity):
    scores = torch.einsum('blnd,bsnd->bnls', query_pool, key_pool) * scale
    probabilities = torch.softmax(scores, dim=-1)
    count = max(1, math.ceil(probabilities.shape[-1] * (1 - sparsity)))
    threshold = torch.topk(probabilities, k=count, dim=-1).values[..., -1:]
    selected = probabilities >= threshold
    selected.logical_or_(retained)
    return selected.to(torch.int8)


def _gather_into(output, value, group):
    dist.all_gather_into_tensor(output.view(-1), value.contiguous().view(-1), group=group)


def _prepare_sends(chain, q, k, v):
    from .pipeline import _head_major_qkv_decision  # pylint: disable=cyclic-import

    mapping = chain.reorder_mapping
    if q.ndim == 6:
        expected = (mapping.chunking, mapping.world, q.shape[2], 1, 1, mapping.hs)
        if (
            getattr(chain, 'precision', None) != 'mix'
            or not chain.optimize_attention_pipeline
            or any(
                t.shape != expected or t.dtype != torch.int8 or t.device != q.device or not t.is_contiguous()
                for t in (q, k, v)
            )
        ):
            raise ValueError('packed CP Q/K/V must match the canonical complete byte layout')
        chain._head_major_qkv_active = False
        chain.last_qkv_layout_fallback_reason = None
        return list(zip(q, k, v))
    local = q.shape[1]
    descriptor = SimpleNamespace(
        s_local=local, w=1, **{name: getattr(mapping, name) for name in ("hc", "hs", "bs", "world", "chunking")}
    )
    head_major, reason = _head_major_qkv_decision(
        q,
        k,
        v,
        descriptor,
        state=chain.state,
        precision=getattr(chain, 'precision', 'bf16'),
        sparse_type='rf_v3',
        optimize_attention_pipeline=chain.optimize_attention_pipeline,
        pre_rearranged=True,
    )
    chain._head_major_qkv_active = head_major
    chain.last_qkv_layout_fallback_reason = reason
    if head_major:
        prepared = [
            t[0].transpose(0, 1).view(mapping.chunking, mapping.world, local, 1, 1, mapping.hs) for t in (q, k, v)
        ]
        chain.last_head_major_qkv_calls = 1
        chain.last_head_major_qkv_view_calls = 3
    else:
        prepared = [
            t.reshape(1, local, mapping.world, mapping.chunking, 1, t.shape[-1]).permute(3, 2, 1, 0, 4, 5).contiguous()
            for t in (q, k, v)
        ]
    if not all(t.is_contiguous() for t in prepared):
        raise RuntimeError('hybrid forward packets must be contiguous before communication')
    return list(zip(*prepared))


def _receive_buffers(chain, sends, kv_group, window):
    cp_size = dist.get_world_size(kv_group) if kv_group is not None else 1
    signature = (
        'kv_gather',
        id(kv_group),
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for triple in sends for t in triple),
        window['boundary_collective'],
        cp_size,
    )
    cached = chain._qkv_receive_buffer_cache
    if chain.optimize_attention_pipeline and cached is not None and cached[0] == signature:
        chain.last_qkv_receive_reuses = len(sends) * 3
        return cached[1], chain._kv_gather_buffer_cache[1]
    incoming = [tuple(torch.empty_like(t) for t in triple) for triple in sends]
    local = sends[0][0].shape[1]
    length = cp_size * chain.reorder_mapping.world * local
    gathered = []
    for triple in sends:
        edge_width = min(128, chain.reorder_mapping.world * local)
        edge_parts = cp_size if cp_size == 2 else 2 * cp_size
        edge = (
            triple[0].new_empty(1, edge_parts * edge_width, 1, triple[0].shape[-1])
            if window['boundary_collective']
            else None
        )
        full_kv = tuple(t.new_empty(1, length, 1, t.shape[-1]) for t in triple[1:]) if kv_group is not None else ()
        gathered.append((*full_kv, edge) if kv_group is not None else None)
    chain.last_qkv_receive_allocations = len(sends) * 3
    if chain.optimize_attention_pipeline:
        chain._qkv_receive_buffer_cache = (signature, incoming)
        chain._kv_gather_buffer_cache = (signature, gathered)
    return incoming, gathered


def _protected_mask(chain, window, device):
    mapping = chain.reorder_mapping
    begin, end = window['start'] // 128, math.ceil(window['end'] / 128)
    whole = chain.retained_blocks.get('bf16_protected_mask')
    if whole is not None:
        return whole[:, :, begin:end]
    cells = math.ceil(mapping.used_len / 128)
    signature = (begin, end, cells, str(device))
    cached = chain.retained_blocks.get('hybrid_protected_mask')
    if cached is not None and cached[0] == signature:
        return cached[1]
    protected = torch.zeros(1, 1, end - begin, cells, dtype=torch.bool, device=device)
    for block in set(mapping.dense_blocks) | set(mapping.first_frame_blocks):
        if begin <= block < end:
            protected[:, :, block - begin, :] = True
        protected[:, :, :, block] = True
    chain.retained_blocks['hybrid_protected_mask'] = (signature, protected)
    return protected


def run_ulysses_kv_gather(chain, q, k, v, *, kv_group, kv_rank, scale, sparsity, inner_precise, counts):
    from . import _layers, distributed_quant as wire
    from .pipeline import _record_stream  # pylint: disable=cyclic-import
    from .quantized_attention_wire import prepare_quantized_wire, run_quantized_bsa
    from .topology import merge_reverse_receive_chunks, merge_reverse_receive_chunks_round_robin

    mapping, state = chain.reorder_mapping, chain.state
    sparse_type = getattr(chain, "sparse_type", "rf_v3")
    precision = getattr(chain, "precision", "bf16")
    quantized = precision in ("mix", "mxfp8")
    quantized_wire = quantized and state.quantized_comm
    local = q.shape[1]
    cp_size = dist.get_world_size(kv_group) if kv_group is not None else 1
    phase = kv_rank if kv_group is not None else 0
    group_length = mapping.world * local
    window = derive_query_window(phase, group_length, mapping.used_len, cp_size)
    packets = (q, k, v)
    if quantized_wire:
        packets, pools, scales = prepare_quantized_wire(
            chain,
            q,
            k,
            v,
            kv_group=kv_group,
            phase=phase,
            counts=counts,
        )
    sends = _prepare_sends(chain, *packets)
    incoming, gathered = _receive_buffers(chain, sends, kv_group, window)
    counts.update(kv_gather_calls=0, query_boundary_gather_calls=0)
    counts.update(quantized_collective_calls=0, quantized_bsa_after_comm_calls=0, mxfp8_bsa_calls=0)
    compute = torch.npu.current_stream(q.device) if state.overlap_comm else None
    comm = state._comm_stream if state.overlap_comm else None
    prepared = None
    if state.overlap_comm:
        prepared = torch.npu.Event()
        prepared.record(compute)
        _record_stream(sends, comm)
    result = q.new_empty(1, local, mapping.hc, mapping.hs)
    result_view = result.view(1, local, mapping.world, mapping.head_groups, mapping.hs)
    reverse = result.new_empty(mapping.chunking, mapping.world, 1, local, 1, mapping.hs)
    reverse_outputs = result.new_empty(reverse.shape)
    if state.overlap_comm:
        _record_stream(reverse, compute)
        _record_stream(reverse, comm)
        _record_stream(reverse_outputs, compute)
        _record_stream(reverse_outputs, comm)
    protected = None
    owned = window['owned_end'] - window['owned_start']

    def gather_chunk(receives, values):
        if kv_group is not None:
            full_k, full_v, edge_output = values
            query, key, value = (wire.from_send(t) for t in receives)
            _gather_into(full_k, key, kv_group)
            _gather_into(full_v, value, kv_group)
            counts['kv_gather_calls'] += 2
            if edge_output is not None:
                edge = edge_output.shape[1] // 2
                if cp_size == 2:
                    piece = query[:, -edge:] if phase == 0 else query[:, :edge]
                else:
                    edge = edge_output.shape[1] // (2 * cp_size)
                    piece = torch.cat((query[:, :edge], query[:, -edge:]), dim=1)
                _gather_into(edge_output, piece, kv_group)
                counts['query_boundary_gather_calls'] += 1
            if quantized_wire:
                counts["quantized_collective_calls"] += 2 + int(edge_output is not None)
            if state.overlap_comm:
                _record_stream((full_k, full_v), compute)
                if edge_output is not None:
                    _record_stream(edge_output, compute)
        if state.overlap_comm:
            event = torch.npu.Event()
            event.record(comm)
            return event
        return None

    def issue_forward(triple, receives):
        if state.overlap_comm:
            _record_stream(receives, compute)
        for recv, send in zip(receives, triple):
            chain._a2a(recv, send, counts)
        if quantized_wire:
            counts["quantized_collective_calls"] = counts.get("quantized_collective_calls", 0) + 3

    def compute_chunk(index, receives, values, event):
        nonlocal protected
        if state.overlap_comm:
            compute.wait_event(event)
        query = wire.from_send(receives[0])
        if values is None:
            full_k, full_v = (wire.from_send(t) for t in receives[1:])
            boundary = None
        else:
            full_k, full_v, boundary = values
        output = reverse_outputs[index].view(1, mapping.world * local, 1, mapping.hs)
        output.zero_()
        if owned:
            if protected is None:
                protected = _protected_mask(chain, window, q.device)
            query = expand_query_window(query, boundary, phase, window, cp_size)
            key, value = full_k[:, : mapping.used_len], full_v[:, : mapping.used_len]
            if quantized_wire:
                head = mapping.rank * mapping.chunking + index
                if not chain.pre_rearranged:
                    positions = mapping.reordered_position_to_token
                    query, key, value = (t.index_select(1, positions) for t in (query, key, value))
                qp, kp = (t[None, :, head : head + 1] for t in pools)
                qp = qp[:, window['start'] // POOL_BLOCK : math.ceil(window['end'] / POOL_BLOCK)]
            else:
                qp, kp = (_layers.avgpool(t, POOL_BLOCK, 'BSND') for t in (query, key))
                chain.last_separate_qk_pool_calls += 1
            mask = rectangular_rf_mask(qp, kp, protected, scale=scale, sparsity=sparsity)
            options = dict(
                block_sparse_mask=mask,
                scale_value=scale,
                inner_precise=inner_precise,
                actual_seq_lengths=[window['end'] - window['start']],
                actual_seq_lengths_kv=[mapping.used_len],
            )
            if quantized:
                if quantized_wire:
                    decoded = run_quantized_bsa(precision, (query, key, value), scales, head, window, options)
                else:
                    decoded = _layers.run_bf16_bsa(precision, (query, key, value), options)
                    counts["quantized_bsa_after_comm_calls"] = counts.get("quantized_bsa_after_comm_calls", 0) + 1
                if precision == "mxfp8":
                    counts["mxfp8_bsa_calls"] = counts.get("mxfp8_bsa_calls", 0) + 1
                if not chain.pre_rearranged:
                    decoded = decoded.index_select(1, mapping.token_to_reordered_position)
            elif sparse_type == "rf_v2":
                rf_v2 = getattr(_layers, "rf_v2_block_sparse_attention")
                decoded = rf_v2(query=query, key=key, value=value, **options)
            else:
                decoded = _layers.bf16_block_sparse_attention(
                    bsa_backend=chain.state.bsa_backend or "eagle",
                    query=query,
                    key=key,
                    value=value,
                    **options,
                )
            begin = window['output_start']
            output[:, :owned].copy_(decoded[:, begin : begin + owned])
        if state.overlap_comm:
            done = torch.npu.Event()
            done.record(compute)
            return done
        return None

    def return_chunk(index, done):
        if state.overlap_comm:
            comm.wait_event(done)
        chain._a2a(reverse[index], reverse_outputs[index], counts)
        if state.overlap_comm:
            event = torch.npu.Event()
            event.record(comm)
            return event
        return None

    returned = []
    ready = []
    with torch.npu.stream(comm) if state.overlap_comm else nullcontext():
        if prepared is not None:
            comm.wait_event(prepared)
            _record_stream(incoming, comm)
            _record_stream([t for triple in gathered if triple is not None for t in triple if t is not None], comm)
        for triple, receives in zip(sends, incoming):
            issue_forward(triple, receives)
        for receives, values in zip(incoming, gathered):
            ready.append(gather_chunk(receives, values))
    for index, (receives, values, event) in enumerate(zip(incoming, gathered, ready)):
        done = compute_chunk(index, receives, values, event)
        with torch.npu.stream(comm) if state.overlap_comm else nullcontext():
            returned.append(return_chunk(index, done))
    if state.overlap_comm:
        for event in returned:
            compute.wait_event(event)
    counts['reverse_a2a_calls'] = mapping.chunking
    merger = merge_reverse_receive_chunks_round_robin if chain._head_major_qkv_active else merge_reverse_receive_chunks
    result_view.copy_(merger(reverse))
    chain.last_common_reverse_buffer_merge_calls = 1
    chain.last_round_robin_merge_calls = int(chain._head_major_qkv_active)
    counts['fused_stages'] = 2
    return result, counts


def create_sparse_parallel_groups(*, ulysses_size=8, backend=None, ulysses_pg_options=None, cp_pg_options=None):
    """Create orthogonal logical-rank groups; callers own devices and HCCL options."""

    if not dist.is_initialized():
        raise RuntimeError("sparse CP x Ulysses requires an initialized distributed world")
    plan = derive_sparse_parallel_groups(dist.get_world_size(), ulysses_size)
    rank = dist.get_rank()
    backend = dist.get_backend() if backend is None else backend
    selected = [None, None]
    for index, name in enumerate(("ulysses", "kv")):
        options = (ulysses_pg_options, cp_pg_options)[index]
        for members in plan[name + "_groups"]:
            group = dist.new_group(members, backend=backend, pg_options=options)
            if rank in members:
                selected[index] = group
    return tuple(selected)


def create_ulysses_kv_gather_groups(*, backend=None, ulysses_pg_options=None, kv_gather_pg_options=None):
    return create_sparse_parallel_groups(
        ulysses_size=8,
        backend=backend,
        ulysses_pg_options=ulysses_pg_options,
        cp_pg_options=kv_gather_pg_options,
    )
