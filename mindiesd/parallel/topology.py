# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Pure rank/head/sequence ownership and inverse layout mappings."""

from __future__ import annotations

__all__ = [
    "resolve_world_rank",
    "head_to_row",
    "row_to_head",
    "derive_rank_window",
    "split_into_ranks",
    "merge_ranks",
]


def resolve_world_rank(group, world=None, rank=None):
    """Resolve world/rank and require a live process group at every world size."""
    import torch.distributed as dist

    if group is None:
        raise RuntimeError(
            "no process group was given: the chain issues collectives at any world size "
            "(exactly one all_reduce plus 4*chunking all_to_all_single), so the caller's own SP "
            "group is required even at world=1 -- initialize torch.distributed (e.g. torchrun "
            "--nproc_per_node=1) and pass its group",
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized: the chain issues collectives at any world "
            "size, so the process group has to be live before the first call",
        )
    if world is None:
        world = dist.get_world_size(group)
    if rank is None:
        rank = dist.get_rank(group)
    return int(world), int(rank)


def derive_rank_window(rank, s_local, used_len):
    """Return this rank's clipped ``(lo, hi, token_count)`` sequence window."""
    lo = int(rank) * int(s_local)
    hi = min(lo + int(s_local), int(used_len))
    return lo, hi, max(0, hi - lo)


def head_to_row(chunks: int, world: int):
    """``head_row[head]``: head ``j*chunks + c`` -> head_row ``c*world + j`` (``QuantState.head_row``)."""
    head_row = [0] * (chunks * world)
    for c in range(chunks):
        for j in range(world):
            head_row[j * chunks + c] = c * world + j
    return head_row


def row_to_head(chunks: int, world: int):
    """Inverse of :func:`head_to_row`: ``head_of_row[head_row] = head`` (row order of scales/pool sums)."""
    head_of_row = [0] * (chunks * world)
    for c in range(chunks):
        for j in range(world):
            head_of_row[c * world + j] = j * chunks + c
    return head_of_row


def split_into_ranks(whole, world):
    """(bs, s_global, w, hs) -> (world, w, shard, bs, hs): the reverse-a2a send layout."""
    bs, s_global, w, hs = whole.shape
    shard = s_global // world
    if bs == 1 and w == 1 and whole.is_contiguous():
        # only SIZE-1 axes are transposed here, so the view IS the copied form
        return whole.view(world, w, shard, bs, hs)
    return (
        whole.reshape(bs, world, shard, w, hs)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(world, w, shard, bs, hs)
    )


def merge_ranks(raw):
    """(world, w, shard, bs, hs) -> (bs, shard, world, w, hs)."""
    world, w, shard, bs, hs = raw.shape
    return raw.reshape(world * w, shard, bs, hs).transpose(0, 2).contiguous().reshape(bs, shard, world, w, hs)


def merge_reverse_receive_chunks(raw):
    """View common reverse storage in final ``[B,S,W,chunks*w,D]`` order."""
    chunks, world, width, shard, batch, dim = raw.shape
    if batch != 1 or width != 1:
        raise ValueError("reverse receive chunk merge requires batch=1 and width=1")
    return raw.permute(4, 3, 1, 0, 2, 5).view(batch, shard, world, chunks * width, dim)


def _head_major_qkv_geometry_supported(chunks, world, width, batch, dim):
    """One narrow geometry authority for the contiguous send views and their reverse inverse."""
    return (chunks, world, width, batch, dim) in ((7, 8, 1, 1, 128), (14, 4, 1, 1, 128))


def merge_reverse_receive_chunks_round_robin(raw):
    """Restore global head ``h=chunk*world+rank`` from round-robin receive storage."""
    chunks, world, width, shard, batch, dim = raw.shape
    if not _head_major_qkv_geometry_supported(chunks, world, width, batch, dim):
        raise ValueError("round-robin merge requires (chunks,world)=(7,8) or (14,4), width=1, batch=1, dim=128")
    return raw.permute(4, 3, 0, 1, 2, 5).reshape(batch, shard, world, chunks * width, dim)
