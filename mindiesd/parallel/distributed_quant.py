# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Distributed Q/K INT8 and V FP8 quantization.

One global MAX reduction repairs scales for 64-token blocks split across ranks and also carries
V's per-(batch, head, dim) inf-norm. Payload functions remain collective-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import POOL_BLOCK, QUANT_BLOCK

__all__ = [
    "QuantState",
    "make_quant_state",
    "quantize_amax_for_allreduce",
    "quantize_and_pool_for_amax",
    "repair_boundary_and_scatter",
    "FP8_E4M3_MAX",
    "check_sendable",
    "from_send",
    "to_send",
]


BLOCK = QUANT_BLOCK
CELL = POOL_BLOCK
CELLS_PER_BLOCK = CELL // BLOCK  # a pool cell holds CELLS_PER_BLOCK quantizer blocks
_C_INV127 = 0.0078740157186985015869140625  # fl32(1/127), the constant the operator's quantizer uses


@dataclass(eq=False, repr=False, slots=True)
class Maps:
    """Per-rank, layer-invariant index maps (pure geometry); see :func:`build_maps`."""

    src: torch.Tensor  # [n_owned] -- local token index in (block, row_in_block) order
    padded_row: torch.Tensor  # [n_owned] -- that row's position in the padded block layout
    gblk: torch.Tensor  # [nblk_local] -- global block ids this rank touches, ascending
    own: torch.Tensor  # [nblk_local] -- rows this rank owns in each of them
    amax_entry: torch.Tensor  # [nblk_local] -- the block's amax entry, -1 when it is not a boundary
    boundary_blk: torch.Tensor  # [np_local] -- the local index of each boundary block, ascending
    boundary_amax_entries: torch.Tensor  # [np_local] -- its GLOBAL amax entry (DIM 1 of boundary_amax)
    cellb: torch.Tensor  # [nblk_local] -- pool cell of each block, as an index into `cells`
    nblk_local: int
    nc: int  # pool cells this rank touches
    n_owned: int
    np_local: int

    def __repr__(self):
        return "Maps(nblk_local=%d, nc=%d, n_owned=%d, np_local=%d)" % (
            self.nblk_local,
            self.nc,
            self.n_owned,
            self.np_local,
        )


@dataclass(eq=False, repr=False, slots=True)
class QuantState:
    """Every buffer the quantizer owns, allocated ONCE by :func:`make_quant_state`."""

    maps: Maps
    hc: int
    hs: int
    nblk: int
    num_cross_rank_quant_blocks: int
    n_owned: int
    head_row: torch.Tensor
    rank_block_mapping: dict
    payload_padded: torch.Tensor
    block_amax: torch.Tensor  # [nblk_local, hc] fp32
    payload: torch.Tensor  # [hc, n_tok, hs] int8, token order
    scales: torch.Tensor  # [hc, nblk] fp32, head order
    pool_sum: torch.Tensor  # [hc, nc, hs] fp32, head order
    boundary_amax: torch.Tensor  # [hc, num_cross_rank_quant_blocks] fp32
    C: float

    def __repr__(self):
        return "QuantState(hc=%d, hs=%d, nblk=%d, num_cross_rank_quant_blocks=%d, n_owned=%d)" % (
            self.hc,
            self.hs,
            self.nblk,
            self.num_cross_rank_quant_blocks,
            self.n_owned,
        )


def build_maps(rank_block_mapping, device):
    """Build and validate this rank's payload index maps."""
    src = rank_block_mapping["src_local"].to(device=device, dtype=torch.int64)
    gblk = rank_block_mapping["blocks"].to(device=device, dtype=torch.int64)
    own = rank_block_mapping["owned_per_block"].to(device=device, dtype=torch.int64)
    amax_entry = rank_block_mapping["amax_entry"].to(device=device, dtype=torch.int64)
    row_in_block = rank_block_mapping["row_in_block"].to(device=device, dtype=torch.int64)
    index = torch.arange(gblk.numel(), device=device, dtype=torch.int64)
    blk_of_row = torch.repeat_interleave(index, own, output_size=int(src.numel()))
    n_owned = int(src.numel())
    nblk_local = int(gblk.numel())

    cells = rank_block_mapping["cells_unique"].to(device=device, dtype=torch.int64)
    gcell = torch.div(gblk * BLOCK, CELL, rounding_mode="floor")
    cellb = torch.searchsorted(cells, gcell)
    if not bool(torch.equal(cells[cellb], gcell)):
        raise RuntimeError("block -> pool cell map is not exact")
    nc = int(cells.numel())

    padded_row = blk_of_row * BLOCK + row_in_block
    boundary_blk = torch.nonzero(amax_entry >= 0).reshape(-1)
    maps = Maps(
        src=src,
        padded_row=padded_row,
        gblk=gblk,
        own=own,
        amax_entry=amax_entry,
        boundary_blk=boundary_blk,
        boundary_amax_entries=amax_entry[boundary_blk],
        cellb=cellb,
        nblk_local=nblk_local,
        nc=nc,
        n_owned=n_owned,
        np_local=int(boundary_blk.numel()),
    )
    _assert_maps(rank_block_mapping, maps)
    return maps


def _assert_maps(rank_block_mapping, maps):
    """The maps must describe this rank exactly once: every owned row, no row twice."""
    padded_row = maps.padded_row
    if padded_row.numel() and int(padded_row[-1]) >= maps.nblk_local * BLOCK:
        raise RuntimeError(
            "the padded layout addresses row %d, outside %d rows (nblk_local=%d blocks)"
            % (int(padded_row[-1]), maps.nblk_local * BLOCK, maps.nblk_local),
        )
    amax_entry_all = rank_block_mapping["amax_entry"].to(torch.int64)
    if not bool(torch.equal(maps.amax_entry, amax_entry_all)):
        raise RuntimeError("maps.amax_entry is not the unfiltered per-block amax entry")
    nonfull = ~rank_block_mapping["full"]
    bad_entry = int((amax_entry_all[nonfull] < 0).sum()) if bool(nonfull.any()) else 0
    if bad_entry:
        raise RuntimeError(
            "%d non-full block(s) have no amax entry; its owners would derive different scales" % bad_entry,
        )
    distinct = int(torch.unique(maps.src).numel())
    if distinct != int(maps.src.numel()):
        raise RuntimeError(
            "the block-order gather index is not bijective "
            "(len(unique(src))=%d != numel=%d)" % (distinct, int(maps.src.numel())),
        )


def make_quant_state(
    rank_block_mapping,
    hc,
    hs,
    nblk_total,
    n_tok,
    device,
    num_cross_rank_quant_blocks,
    dtype=torch.bfloat16,
    head_row=None,
):
    """Allocate geometry-stable buffers and install the destination-major head map."""
    m = build_maps(rank_block_mapping, device)
    if head_row is None:
        head_row = list(range(hc))
    if not torch.is_tensor(head_row):
        head_row = torch.as_tensor(head_row, device=device, dtype=torch.int64)
    hc, hs = int(hc), int(hs)
    return QuantState(
        maps=m,
        hc=hc,
        hs=hs,
        nblk=int(nblk_total),
        num_cross_rank_quant_blocks=int(num_cross_rank_quant_blocks),
        n_owned=m.n_owned,
        head_row=head_row.to(torch.int64),
        rank_block_mapping=rank_block_mapping,
        payload_padded=torch.zeros((m.nblk_local * BLOCK, hc, hs), dtype=dtype, device=device),
        block_amax=torch.zeros((m.nblk_local, hc), dtype=torch.float32, device=device),
        payload=torch.zeros((hc, int(n_tok), hs), dtype=torch.int8, device=device),
        scales=torch.zeros((hc, int(nblk_total)), dtype=torch.float32, device=device),
        pool_sum=torch.zeros((hc, m.nc, hs), dtype=torch.float32, device=device),
        boundary_amax=torch.zeros((hc, int(num_cross_rank_quant_blocks)), dtype=torch.float32, device=device),
        C=_C_INV127,
    )


def _gather_into_block_order(x, st):
    """``x``: [1, n_tok, hc, hs] bf16 in the rank's token order -> ``st.payload_padded``, zero-padded."""
    m = st.maps
    st.payload_padded.zero_()
    st.payload_padded.index_copy_(0, m.padded_row, x[0].index_select(0, m.src))


def _block_amax(st):
    """Per-block scalar amax over token rows and head width."""
    m = st.maps
    slab = st.payload_padded.view(m.nblk_local, BLOCK, st.hc, st.hs)
    st.block_amax.copy_(slab.abs().amax(dim=1).amax(dim=-1).float())
    return st.block_amax


def _derive_scales(st, forced_amax=None):
    """Derive scales, applying the shared amax to boundary blocks when supplied."""
    m = st.maps
    amax = st.block_amax  # [nblk_local, hc]
    if forced_amax is not None and m.np_local:
        # `boundary_amax` is indexed by the GLOBAL amax entry, not by this rank's block index: only
        # the boundary blocks' entries are filled, the rest stay zero (which is what the caller's
        # single all_reduce sees).
        forced = forced_amax.index_select(1, m.boundary_amax_entries)  # [hc, np_local] -- by GLOBAL entry
        forced_bf16 = forced.to(torch.bfloat16).to(torch.float32)
        amax = amax.clone()
        pos = m.boundary_blk
        amax[pos] = torch.maximum(amax[pos], forced_bf16.t()).contiguous()
    st.scales.zero_()
    st.scales.index_copy_(1, m.gblk, (amax.t() * st.C).contiguous())
    return st.scales


def _quantize_into(st):
    """Quantize this rank's rows with RNE and scatter them back to token order."""
    m = st.maps
    block_scale = st.scales.index_select(1, m.gblk)  # [hc, nblk_local] -- the repaired per-block scale
    # A row's scale is its OWN block's scale: index by the padded position, never by the row's index
    # inside this rank (a boundary block's second owner starts at row_in_block > 0, so the two
    # differ).
    scale_pad = block_scale.t().repeat_interleave(BLOCK, dim=0)  # [nblk_local*BLOCK, hc]
    row_scale = scale_pad.index_select(0, m.padded_row)  # [n_owned, hc]
    divisor = torch.where(row_scale == 0, torch.ones_like(row_scale), row_scale)
    rows = st.payload_padded.index_select(0, m.padded_row)  # [n_owned, hc, hs], (block, row_in_block) order
    q = torch.round(rows.float() / divisor.unsqueeze(-1)).to(torch.int8)
    q_head = q.permute(1, 0, 2)  # [hc, n_owned, hs], head order
    # `head_row[h]` is the head_row head h is stored at, so this is a SCATTER
    # (`q_row[head_row[h]] = head h`), not an `index_select`: selecting would install the INVERSE
    # permutation, and the two only coincide when every head happens to sit on its own head_row --
    # e.g. world=1, where chunking == hc.  On a two-rank geometry the heads at head_row == head
    # stayed right and every other head was wrong.
    q_row = torch.zeros_like(q_head)
    q_row.index_copy_(0, st.head_row, q_head)
    st.payload.zero_()
    st.payload.index_copy_(1, m.src, q_row)  # payload[:, src[i]] = the i-th row, so payload is in TOKEN order
    return st.payload


def _emit_amax(st, x=None):
    """Emit local amax values only at globally assigned boundary entries."""
    if x is not None:
        _gather_into_block_order(x, st)
    m = st.maps
    _block_amax(st)
    if m.np_local:
        st.boundary_amax.index_copy_(1, m.boundary_amax_entries, st.block_amax[m.boundary_blk].t().contiguous())
    return st.boundary_amax


def quantize_amax_for_allreduce(x, st):
    """Gather rows and produce this rank's boundary amax reduction operand."""
    return _emit_amax(st, x)


def quantize_and_pool_for_amax(x, st, emit_amax=True, pool_sums=True, gather=True):
    """Produce provisional payload/scales and optional per-cell pool sums."""
    if emit_amax:
        _emit_amax(st, x if gather else None)
    elif gather:
        # the amax emit already ran: only the gather is still owed to this pass
        _gather_into_block_order(x, st)
    _derive_scales(st)
    _quantize_into(st)
    if pool_sums:
        _pool_sums(st)
    return st.boundary_amax


def _pool_sums(st):
    """Compute mix pooling sums from the retained original BF16 rows."""
    m = st.maps
    blocks = st.payload_padded.view(m.nblk_local, BLOCK, st.hc, st.hs).float().sum(dim=1)
    cells = torch.zeros((m.nc, st.hc, st.hs), dtype=torch.float32, device=blocks.device)
    cells.index_add_(0, m.cellb, blocks)
    st.pool_sum.copy_(cells.permute(1, 0, 2))


def repair_boundary_and_scatter(st, forced_amax):
    """Re-derive the boundary blocks with the all_reduce'd amax and rewrite the payload."""
    if st.maps.np_local:
        _derive_scales(st, forced_amax=forced_amax)
        _quantize_into(st)
    return st.payload, st.scales


FP8_E4M3_MAX = 448.0
_FP32_MIN_NORMAL = 1.1754943508222875e-38


def check_sendable(value_shape, chunks, world):
    """Validate the B=1, one-head-per-(chunk,destination) send geometry."""
    chunks, world = int(chunks), int(world)
    bs, _s_local, hc, _hs = (int(x) for x in value_shape)
    per_slot = chunks * world
    if bs != 1 or per_slot != hc:
        raise ValueError(
            "V's payload needs [1, s_local, hc, hs] and a [1, hc, hs] divisor, with hc == "
            "chunks*world: the packet carries exactly ONE head per (chunk, destination) row, so "
            "w = hc // (chunks*world) must be 1; got value=%s, chunks*world=%d*%d, hc=%d"
            % (tuple(value_shape), chunks, world, hc),
        )


def _fp8_payload(value, divisor):
    """``clamp(v / divisor)`` with the subnormal flush, cast to fp8.  Leading layout is kept."""
    exact = value.float()
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    y = exact / divisor
    y = y.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    y = torch.where(exact.abs() < _FP32_MIN_NORMAL, torch.zeros_like(y), y)
    return y.to(torch.float8_e4m3fn)


@torch.no_grad()
def to_send(value: torch.Tensor, scale_div: torch.Tensor, chunks: int, world: int) -> torch.Tensor:
    """Write FP8 values directly in chunk/destination send order as int8 bytes."""
    bs, s_local, hc, hs = value.shape
    w = hc // (int(chunks) * int(world))
    # head h = (j*chunks + c)*w + wl: split hc into (world, chunks, w) and move (c, j) to the front,
    # so the fp8 cast happens on the payload already in send order (no post-hoc permute).
    v = value.view(bs, s_local, world, chunks, w, hs).permute(3, 2, 1, 0, 4, 5)
    sc = scale_div.view(bs, world, chunks, w, hs).permute(2, 1, 0, 3, 4)
    v8 = _fp8_payload(v, sc.unsqueeze(2)).contiguous()  # [chunks, world, s_local, bs, w, hs]
    return v8.view(torch.int8)


def from_send(rv):
    """`v_g` construction from the received V, layout ``[world, s_local, bs, w, hs]``."""
    world, s_local, bs, _w, hs = rv.shape
    s_global = world * s_local
    return rv.reshape(s_global, bs, hs).transpose(0, 1).contiguous().reshape(bs, s_global, 1, hs)
