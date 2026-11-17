# SPDX-License-Identifier: Apache-2.0
"""Flat per-destination mixed-precision packet layout, packer and unpacker."""

from __future__ import annotations


from dataclasses import dataclass

import torch


# Align the scale region so following pool rows start on a cache line.
PACKET_ALIGN_BYTES = 512


@dataclass(eq=False, repr=False, slots=True)
class PacketLayout:
    """Offsets for INT8 payload, aligned FP32 scales and FP32 pool partials."""

    world: int
    s_local: int
    hs: int
    nblk: int
    npool: int
    payload_bytes: int
    scale_bytes: int
    pool_off: int
    pool_bytes: int
    per_dest: int


def derive_packet_layout(
    *,
    world: int,
    s_local: int,
    hs: int,
    nblk: int,
    npool: int,
) -> PacketLayout:
    """Derive all offsets once; the scale region is padded to PACKET_ALIGN_BYTES."""
    payload_bytes = s_local * hs
    if payload_bytes % 4:
        raise ValueError("s_local*hs payload bytes must align the following FP32 scale region")
    scale_bytes = nblk * 4
    scale_region_end = (scale_bytes + PACKET_ALIGN_BYTES - 1) // PACKET_ALIGN_BYTES * PACKET_ALIGN_BYTES
    pool_off = payload_bytes + scale_region_end
    pool_bytes = npool * hs * 4
    return PacketLayout(
        world=int(world),
        s_local=int(s_local),
        hs=int(hs),
        nblk=int(nblk),
        npool=int(npool),
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        pool_off=pool_off,
        pool_bytes=pool_bytes,
        per_dest=pool_off + pool_bytes,
    )


__all__ = [
    "derive_packet_layout",
    "PacketLayout",
    "pack_qk_packet",
    "Unpacker",
]


def _require_pool_headroom(nc, npool):
    """Require every live pool row plus the unpacker's discarded dummy row."""
    worst = max(int(n) for n in nc)
    if int(npool) <= worst:
        raise RuntimeError(
            "npool=%d must exceed the largest per-rank live cell count %d: the consumer's "
            "dummy pool row would alias a live row" % (int(npool), worst),
        )


def pack_qk_packet(payload, scales, pool_sums, chunk, *, layout: PacketLayout, chunks: int):
    """Pack one chunk's payload, scales and live pool rows by destination."""
    W, s_local, hs = layout.world, layout.s_local, layout.hs
    payload_bytes, scale_bytes, pool_bytes = layout.payload_bytes, layout.scale_bytes, layout.pool_bytes
    per_dest, npool = layout.per_dest, layout.npool
    src_dm = payload.view(chunks, W, s_local, hs)[chunk]

    buf = torch.empty(W, per_dest, dtype=torch.int8, device=payload.device)
    bu = buf.view(torch.uint8)

    bu[:, :payload_bytes].copy_(src_dm.contiguous().view(torch.uint8).reshape(W, payload_bytes))
    bu[:, payload_bytes : payload_bytes + scale_bytes].view(torch.float32).copy_(scales[0])
    poolv = bu[:, layout.pool_off : layout.pool_off + pool_bytes].view(torch.float32).reshape(W, npool, hs)
    poolv[:, : int(pool_sums.shape[1])].copy_(pool_sums)
    return buf


class Unpacker:
    """Batch scale MAX and rank-ordered pool accumulation while unpacking Q/K."""

    def __init__(self, layout: PacketLayout, *, rank_mappings, pool_cell_token_counts):
        self.layout = layout
        npool = layout.npool
        _require_pool_headroom([int(p["cells_unique"].numel()) for p in rank_mappings], npool)
        ncell = int(pool_cell_token_counts.numel())
        W = layout.world
        dev = rank_mappings[0]["cells_unique"].device

        cells = []
        for r in range(W):
            ncr = int(rank_mappings[r]["cells_unique"].numel())
            c = torch.full((npool,), ncell, device=dev, dtype=torch.int64)
            c[:ncr] = rank_mappings[r]["cells_unique"]
            cells.append(c)
        self.cells_pad = torch.cat(cells)
        self.pool_cell_token_counts = pool_cell_token_counts.unsqueeze(1)  # [ncell, 1] fp32
        self.ncell = ncell
        self.npool = npool
        self.s_global = W * layout.s_local

    def unpack_pair_batched(self, rq, rk):
        """Return Q/K payloads, assembled scales and pooled rows from a packet pair."""
        layout = self.layout
        W, hs = layout.world, layout.hs
        payload_bytes, scale_bytes = layout.payload_bytes, layout.scale_bytes
        ruq, ruk = rq.view(torch.uint8), rk.view(torch.uint8)
        q8 = ruq[:, :payload_bytes].view(torch.int8).reshape(1, self.s_global, 1, hs)
        k8 = ruk[:, :payload_bytes].view(torch.int8).reshape(1, self.s_global, 1, hs)
        sq = ruq[:, payload_bytes : payload_bytes + scale_bytes].view(torch.float32)
        sk = ruk[:, payload_bytes : payload_bytes + scale_bytes].view(torch.float32)
        scales = torch.stack((sq, sk), 0).amax(1).unsqueeze(0)  # (1, 2, nblk)
        q_scale, k_scale = scales[:, 0:1], scales[:, 1:2]
        lo, n = layout.pool_off, layout.pool_bytes
        srcq = ruq[:, lo : lo + n].view(torch.float32).reshape(W * self.npool, hs)
        srck = ruk[:, lo : lo + n].view(torch.float32).reshape(W * self.npool, hs)
        src = torch.stack((srcq, srck), 0)  # (2, W*npool, hs)
        acc = src.new_zeros((2, self.ncell + 1, hs))
        acc.index_add_(1, self.cells_pad, src)
        pool = torch.empty((2, self.ncell, hs), dtype=torch.bfloat16, device=rq.device)
        torch.div(acc[:, : self.ncell], self.pool_cell_token_counts, out=pool)
        return (
            q8,
            k8,
            q_scale,
            k_scale,
            pool[0].reshape(1, self.ncell, 1, -1),
            pool[1].reshape(1, self.ncell, 1, -1),
        )
