# SPDX-License-Identifier: Apache-2.0
"""Prepare RF-ordered BSA scales and transmit the quantized Q/K/V rows once."""

import math

import torch
import torch.distributed as dist

from .config import POOL_BLOCK, QUANT_BLOCK
from .distributed_quant import _C_INV127, FP8_E4M3_MAX, _fp8_payload


def _reduce(value, group, op, counts):
    dist.all_reduce(value, op=op, group=group)
    counts["allreduce"] += 1


def _rf_tiles(value, positions, block_size, geometry=None):
    if geometry is None:
        owned, inverse = torch.unique(positions // block_size, sorted=True, return_inverse=True)
        rows = inverse * block_size + positions.remainder(block_size)
    else:
        owned, rows = geometry["owned"], geometry["rows"]
    tiles = value.new_zeros((owned.numel(), block_size, *value.shape[1:]))
    tiles.flatten(0, 1).index_copy_(0, rows, value)
    return owned, rows, tiles


def _pool(q, k, blocks, used, group, counts, lengths=None):
    pooled = q.new_zeros((2, math.ceil(used / POOL_BLOCK), *q.shape[1:]), dtype=torch.float32)
    if blocks.numel():
        for partial, value in zip(pooled, (q, k)):
            partial.index_add_(0, blocks, value.float())
    _reduce(pooled, group, dist.ReduceOp.SUM, counts)
    if lengths is None:
        lengths = (used - POOL_BLOCK * torch.arange(pooled.shape[1], device=q.device)).clamp_max(POOL_BLOCK)
    pooled = (pooled / lengths.view(-1, 1, 1)).to(q.dtype)
    return pooled.unbind(0)


def _eagle_packets(q, k, v, positions, used, group, counts, geometry=None, wire_writer=None, chain=None, local=None):
    blocks = positions // QUANT_BLOCK if geometry is None else geometry["blocks"]
    size = math.ceil(used / QUANT_BLOCK)
    qk_max = q.new_zeros((size, 2 * q.shape[1]), dtype=torch.float32)
    if positions.numel():
        row_max = torch.cat((q.abs().amax(dim=-1), k.abs().amax(dim=-1)), dim=1)
        owned, _, tiles = _rf_tiles(row_max, positions, QUANT_BLOCK, geometry)
        qk_max.index_copy_(0, owned, tiles.amax(dim=1).float())
    # An entirely padded shard still participates in the scale collective.
    v_max = v.new_zeros(v.shape[1:], dtype=torch.float32)
    if positions.numel():
        v_max = v.abs().amax(dim=0).float()
    maxima = torch.cat((qk_max.flatten(), v_max.flatten()))
    _reduce(maxima, group, dist.ReduceOp.MAX, counts)
    qk, vs = maxima.split((qk_max.numel(), v_max.numel()))
    qs, ks = (qk.view_as(qk_max) * _C_INV127).chunk(2, dim=1)
    vs = vs.reshape_as(v_max) / FP8_E4M3_MAX
    if wire_writer is not None:
        payload = wire_writer(chain, q, k, v, (qs, ks, vs), geometry, local)
        if payload is not None:
            return payload, (qs, ks, vs)

    def quantize(value, scales):
        scale = scales.index_select(0, blocks).unsqueeze(-1)
        scaled = value.float() / torch.where(scale == 0, 1, scale)
        return scaled.round().clamp(-127, 127).to(torch.int8)

    payload = (quantize(q, qs), quantize(k, ks), _fp8_payload(v, vs).view(torch.int8))
    return payload, (qs, ks, vs)


def _mix_geometry(chain, q, phase, local, start, valid, kv_group):
    mapping = chain.reorder_mapping
    quant_block = QUANT_BLOCK
    cache = chain.retained_blocks.setdefault("mix_wire_geometry", {})
    key = (
        (phase, local, valid, mapping.used_len),
        (mapping.world, mapping.rank, mapping.hc, mapping.hs),
        (q.device, id(chain.group), id(kv_group)),
        (chain.precision, chain.pre_rearranged, quant_block, POOL_BLOCK),
    )
    if key not in cache:
        positions = torch.arange(start, start + valid, device=q.device)
        blocks = positions // quant_block
        owned, inverse = torch.unique(blocks, sorted=True, return_inverse=True)
        rows = inverse * quant_block + positions.remainder(quant_block)
        lengths = (
            mapping.used_len - POOL_BLOCK * torch.arange(math.ceil(mapping.used_len / POOL_BLOCK), device=q.device)
        ).clamp_max(POOL_BLOCK)
        if len(cache) >= 4:
            cache.clear()
        cache[key] = {
            "positions": positions,
            "blocks": blocks,
            "owned": owned,
            "rows": rows,
            "pool_blocks": positions // POOL_BLOCK,
            "lengths": lengths,
        }
    return cache[key]


def prepare_quantized_wire(chain, q, k, v, *, kv_group, phase, counts):
    mapping = chain.reorder_mapping
    local = q.shape[1]
    start = (phase * mapping.world + mapping.rank) * local
    valid = max(0, min(local, mapping.used_len - start))
    geometry = None
    if chain.precision == "mix" and chain.optimize_attention_pipeline and chain.pre_rearranged:
        geometry = _mix_geometry(chain, q, phase, local, start, valid, kv_group)
        positions = geometry["positions"]
    elif chain.pre_rearranged:
        positions = torch.arange(start, start + valid, device=q.device)
    else:
        positions = mapping.token_to_reordered_position[start : start + valid]
    # CP topology validation establishes that the default group spans exactly
    # this sequence. Pure Ulysses can also be a subgroup of the default group.
    group = None if kv_group is not None else chain.group
    from . import _layers

    wire_writer = getattr(_layers, "cp_mix_wire", None)
    q, k, v = (t[0, :valid] for t in (q, k, v))
    pool_blocks = positions // POOL_BLOCK if geometry is None else geometry["pool_blocks"]
    pools = _pool(
        q,
        k,
        pool_blocks,
        mapping.used_len,
        group,
        counts,
        lengths=None if geometry is None else geometry["lengths"],
    )
    prepare = _eagle_packets
    options = {"geometry": geometry, "wire_writer": wire_writer, "chain": chain, "local": local}
    packets, scales = prepare(q, k, v, positions, mapping.used_len, group, counts, **options)
    if packets[0].ndim == 6:
        return packets, pools, scales
    padding = (0, 0, 0, 0, 0, local - valid)
    packets = tuple(torch.nn.functional.pad(t, padding).unsqueeze(0) for t in packets)
    return packets, pools, scales


def run_quantized_bsa(precision, tensors, scales, head, window, options):
    query, key, value = tensors
    if precision == "mix":
        from ._layers import eagle_quant_block_sparse_attention

        qs, ks, vs = scales
        first, last = window["start"] // QUANT_BLOCK, math.ceil(window["end"] / QUANT_BLOCK)
        return eagle_quant_block_sparse_attention(
            query=query.transpose(1, 2).contiguous(),
            key=key.transpose(1, 2).contiguous(),
            value=value.transpose(1, 2).contiguous(),
            query_scale=qs[first:last, head].contiguous().view(1, 1, -1),
            key_scale=ks[:, head].contiguous().view(1, 1, -1),
            value_scale=vs[head].view(1, 1, -1),
            **options,
        ).transpose(1, 2)

    raise ValueError("native MXFP8 consumer is not installed")
