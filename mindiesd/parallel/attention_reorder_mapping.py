# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Cache RF token permutation and per-rank quantization/pool ownership."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from . import _layers
from . import topology
from .config import POOL_BLOCK, QUANT_BLOCK, ParallelConfig

__all__ = [
    "AttentionReorderMapping",
    "get_attention_reorder_mapping",
]

# Mapping caches are caller-owned. A full key includes immutable geometry and device;
# validated hits skip expensive construction assertions, while mismatches are evicted.
_REORDER_MAPPING_CACHE_MAX = 4


def _config_property(name):
    return property(lambda self: getattr(self.config, name))


def _mapping_property(name, *, as_int=False):
    def get(self):
        value = self.token_block_mapping[name]
        return int(value) if as_int else value

    return property(get)


def _get_cached_reorder_mapping(cache: dict, key: tuple, config: ParallelConfig):
    """Return only a full config/device match; evict any colliding or stale entry."""
    reorder_mapping = cache.get(key)
    if reorder_mapping is None:
        return None
    token_block_mapping = getattr(reorder_mapping, "token_block_mapping", None)
    reordered_position_to_token = (
        token_block_mapping.get("reordered_position_to_token") if isinstance(token_block_mapping, dict) else None
    )
    config_ok = (
        isinstance(getattr(reorder_mapping, "config", None), ParallelConfig) and reorder_mapping.config.key == key[0]
    )
    mapping_ok = isinstance(token_block_mapping, dict) and token_block_mapping.get("used_len") == int(config.used_len)
    device_ok = (
        isinstance(reordered_position_to_token, torch.Tensor) and str(reordered_position_to_token.device) == key[1]
    )
    if config_ok and mapping_ok and device_ok:
        return reorder_mapping
    cache.pop(key, None)  # a colliding or foreign entry: evict rather than serve it
    return None


# Leave room for the unpacker's dummy cell beyond every rank's live pool rows.
NPOOL_SLACK = 2


def _assert_owned_order(rank_block_mapping):
    """Require a unique ascending (block, row_in_block) gather order."""
    blk = rank_block_mapping["blocks"].to(torch.int64)[rank_block_mapping["blk_idx"].to(torch.int64)]
    row_in_block = rank_block_mapping["row_in_block"].to(torch.int64)
    # This bound makes the compound key below a bijection.
    row_in_block_max = int(row_in_block.max()) if int(row_in_block.numel()) else -1
    if row_in_block_max >= QUANT_BLOCK:
        raise RuntimeError(
            "a row_in_block index reaches %d, which is not < QUANT_BLOCK=%d: the "
            "(block, row_in_block) compound key is ambiguous" % (row_in_block_max, QUANT_BLOCK),
        )
    key = blk * QUANT_BLOCK + row_in_block
    if key.numel() > 1 and not bool((key[1:] > key[:-1]).all()):
        raise RuntimeError("block-ordered gather is not in ascending (block, row_in_block)")


def _assert_pool_cover(token_to_reordered_position, pool_cell_token_counts, rank_mappings, ncell):
    """Every pool cell this rank reports must be fully accounted for across the ranks that touch it."""
    P = POOL_BLOCK
    dev = token_to_reordered_position.device
    total = torch.zeros(ncell, dtype=torch.int64, device=dev)
    for p in rank_mappings:
        lo, hi = int(p["tok_lo"]), int(p["tok_hi"])
        if hi <= lo:
            continue
        toks = torch.arange(lo, hi, device=dev, dtype=torch.int64)
        c = torch.div(token_to_reordered_position[toks], P, rounding_mode="floor")
        total += torch.bincount(c, minlength=ncell)
        touched = torch.unique(c)
        if int(touched.numel()) != int(p["cells_unique"].numel()) or not bool(torch.equal(touched, p["cells_unique"])):
            raise RuntimeError(
                "rank %d: cells_unique is not exactly the set of pool cells its own tokens land in" % p["rank"],
            )
    counts = pool_cell_token_counts.to(torch.int64)
    if not bool(torch.equal(total, counts)):
        diff = int((total != counts).sum())
        raise RuntimeError(
            "the per-rank token counts per pool cell do not sum to "
            "pool_cell_token_counts (%d of %d cells differ; total %d vs %d): the "
            "consumer's rank-wise accumulation would not equal the pooled "
            "sum of the assembled sequence" % (diff, ncell, int(total.sum()), int(counts.sum())),
        )


def _build_rank_block_mapping(used_len: int, spans, spans_key, s_local: int, world: int, device, cache=None):
    """Build or reuse token reorder and per-rank block/cell ownership."""
    key = (int(used_len), spans_key, int(s_local), int(world), str(device))
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    idx = torch.arange(used_len, device=device, dtype=torch.float32).reshape(1, used_len, 1, 1)
    # Use the stock rearranger as the single authority for permutation and protected blocks.
    q_rf, _k_rf, _v_rf, _qkv_pool, _inverse, dense_blocks, first_frame_blocks = (
        _layers.do_multi_span_tensor_rearrange_pooling(idx, idx, idx, list(spans), POOL_BLOCK, "BSND")
    )
    reordered_position_to_token = q_rf[0, :, 0, 0].to(torch.int64)  # position -> original token
    if reordered_position_to_token.unique().numel() != used_len:
        raise RuntimeError("rearrange is not a bijection")
    token_to_reordered_position = torch.empty_like(reordered_position_to_token)
    token_to_reordered_position[reordered_position_to_token] = torch.arange(
        used_len, device=device, dtype=torch.int64
    )  # token -> position

    nblk = math.ceil(used_len / QUANT_BLOCK)
    blk_len = torch.full((nblk,), QUANT_BLOCK, device=device, dtype=torch.int64)
    if used_len % QUANT_BLOCK:
        blk_len[-1] = used_len % QUANT_BLOCK
    # A boundary block has more than one rank owner and needs a shared scale.
    owner_of_pos = torch.div(reordered_position_to_token, s_local, rounding_mode="floor")
    n_pad = nblk * QUANT_BLOCK - used_len
    ob = torch.cat([owner_of_pos, owner_of_pos.new_full((n_pad,), int(owner_of_pos[-1]))])
    ob = ob.reshape(nblk, QUANT_BLOCK)
    boundary_mask = ob.max(dim=1).values != ob.min(dim=1).values
    boundary_blocks = torch.nonzero(boundary_mask).reshape(-1)  # global, ascending
    # compact amax entry of each boundary block (for the all_reduce vector)
    amax_entry_of_block = torch.full((nblk,), -1, device=device, dtype=torch.int64)
    amax_entry_of_block[boundary_blocks] = torch.arange(boundary_blocks.numel(), device=device)

    # The last pool cell may be shorter than POOL_BLOCK.
    ncell = math.ceil(used_len / POOL_BLOCK)
    pool_cell_token_counts = torch.full((ncell,), float(POOL_BLOCK), device=device, dtype=torch.float32)
    tail = used_len % POOL_BLOCK
    if tail:
        pool_cell_token_counts[-1] = float(tail)

    token_block_mapping = {
        "used_len": int(used_len),
        "reordered_position_to_token": reordered_position_to_token,
        "token_to_reordered_position": token_to_reordered_position,
        "nblk": nblk,
        "num_cross_rank_quant_blocks": int(boundary_blocks.numel()),
        "dense_blocks": set(dense_blocks),
        "first_frame_blocks": set(first_frame_blocks),
        "ncell": ncell,
        "pool_cell_token_counts": pool_cell_token_counts,
    }

    rank_mappings = []
    for r in range(world):
        lo, hi, n_tok = topology.derive_rank_window(r, s_local, used_len)
        toks = torch.arange(lo, hi, device=device, dtype=torch.int64)
        pos = token_to_reordered_position[toks]
        b = torch.div(pos, QUANT_BLOCK, rounding_mode="floor")
        c = torch.div(pos, POOL_BLOCK, rounding_mode="floor")
        order = torch.argsort(b, stable=True)
        b_sorted = b[order]
        ub, grp, cnt = torch.unique_consecutive(b_sorted, return_inverse=True, return_counts=True)
        starts = torch.cumsum(cnt, 0) - cnt
        row_in_block = torch.arange(b_sorted.numel(), device=device) - starts[grp]
        rank_mappings.append(
            {
                "rank": r,
                "tok_lo": lo,
                "tok_hi": hi,
                "n_tok": int(n_tok),
                "blocks": ub,  # blocks this rank touches (ascending)
                "owned_per_block": cnt,  # owned tokens in each of them
                "full": cnt >= blk_len[ub],
                "src_local": order,  # local token index, in block order
                "blk_idx": grp,  # block-order index -> index into `blocks`
                "row_in_block": row_in_block,  # block-order index -> the token's row inside its block
                "amax_entry": amax_entry_of_block[ub],
                "cells_unique": torch.unique(c),  # pool cells this rank touches
            }
        )
    token_block_mapping["rank_mappings"] = rank_mappings
    # The packet needs every live row plus dummy-cell headroom.
    token_block_mapping["npool"] = max(int(p["cells_unique"].numel()) for p in rank_mappings) + NPOOL_SLACK
    for p in rank_mappings:
        _assert_owned_order(p)
    _assert_pool_cover(
        token_to_reordered_position,
        token_block_mapping["pool_cell_token_counts"],
        rank_mappings,
        token_block_mapping["ncell"],
    )
    if cache is not None:
        cache[key] = token_block_mapping
    return token_block_mapping


@dataclass
class AttentionReorderMapping:
    """Parallel configuration with cached attention reorder indices and rank/block ownership."""

    config: ParallelConfig
    token_block_mapping: dict

    world = _config_property("world")
    rank = _config_property("rank")
    hc = _config_property("hc")
    hs = _config_property("hs")
    bs = _config_property("bs")
    s_local = _config_property("s_local")
    s_global = _config_property("s_global")
    used_len = _config_property("used_len")
    chunking = _config_property("chunking")
    head_groups = _config_property("head_groups")
    w = _config_property("w")

    @property
    def spans(self):
        return list(self.config.spans)

    reordered_position_to_token = _mapping_property("reordered_position_to_token")
    token_to_reordered_position = _mapping_property("token_to_reordered_position")
    nblk = _mapping_property("nblk", as_int=True)
    ncell = _mapping_property("ncell", as_int=True)
    num_cross_rank_quant_blocks = _mapping_property("num_cross_rank_quant_blocks", as_int=True)
    rank_mappings = _mapping_property("rank_mappings")

    @property
    def local_rank_mapping(self):
        return self.token_block_mapping["rank_mappings"][self.config.rank]

    dense_blocks = _mapping_property("dense_blocks")
    first_frame_blocks = _mapping_property("first_frame_blocks")
    pool_cell_token_counts = _mapping_property("pool_cell_token_counts")
    npool = _mapping_property("npool", as_int=True)


def _canonical_spans(spans_norm):
    """Build library-facing metadata from the immutable geometry used by the cache key."""
    return [{"start": start, "latent_shape": shape} for start, shape, _length in spans_norm]


def get_attention_reorder_mapping(config, device, token_block_mapping_cache=None, reorder_mapping_cache=None):
    """Get a mapping keyed by the full configuration and device."""
    # The complete mapping key includes the parallel configuration and device.  The underlying
    # token/block mapping has a separate key that depends only on the sequence partition and spans.
    key = None if reorder_mapping_cache is None else (config.key, str(device))
    if key is not None:
        hit = _get_cached_reorder_mapping(reorder_mapping_cache, key, config)
        if hit is not None:
            return AttentionReorderMapping(config=config, token_block_mapping=hit.token_block_mapping)

    spans_key = tuple(sorted((int(start), tuple(int(v) for v in shape)) for start, shape, _len in config.spans_norm))
    token_block_mapping = _build_rank_block_mapping(
        config.used_len,
        _canonical_spans(config.spans_norm),
        spans_key,
        config.s_local,
        config.world,
        device,
        cache=token_block_mapping_cache,
    )

    reorder_mapping = AttentionReorderMapping(config=config, token_block_mapping=token_block_mapping)
    if key is not None:
        # Insert in place of the (evicted) entry, with a small bounded FIFO so a process serving
        # many geometries cannot grow.
        reorder_mapping_cache[key] = reorder_mapping
        while len(reorder_mapping_cache) > _REORDER_MAPPING_CACHE_MAX:
            reorder_mapping_cache.pop(next(iter(reorder_mapping_cache)))
    return reorder_mapping
