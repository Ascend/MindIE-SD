# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision packet producer and consumer preparation."""

# pylint: disable=no-name-in-module  # provided by the preceding mapping/quant PR

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import distributed_quant as _q
from . import packet as _pack
from .attention_reorder_mapping import AttentionReorderMapping
from .config import _validate_attention_mode
from .state import ParallelState
from .topology import head_to_row, row_to_head

__all__ = ["MixedProducer", "CONSUMER", "CHAIN_IMPL"]

CONSUMER = "quantized"
CHAIN_IMPL = "quantized_comm"


@dataclass(slots=True, eq=False)
class MixedProducer:
    """Geometry-stable mixed packet preparation cached by ParallelState."""

    reorder_mapping: AttentionReorderMapping
    st_q: _q.QuantState | None
    st_k: _q.QuantState | None
    layout: _pack.PacketLayout | None
    unpacker: _pack.Unpacker | None
    group: object
    retained_blocks: dict
    precision: str
    sparse_type: str
    pre_rearranged: bool
    optimize_attention_pipeline: bool
    state: ParallelState
    _qkv_receive_buffer_cache: object | None
    last_separate_qk_pool_calls: int
    last_common_reverse_buffer_merge_calls: int
    _head_major_qkv_active: bool
    last_head_major_qkv_calls: int
    last_head_major_qkv_view_calls: int
    last_round_robin_merge_calls: int
    last_qkv_receive_allocations: int
    last_qkv_receive_reuses: int
    last_qkv_layout_fallback_reason: str | None

    def __init__(self, reorder_mapping, *, state, precision="mix", sparse_type="rf_v3", pre_rearranged=False):
        """Allocate mixed-precision producer buffers and validate geometry once."""
        _validate_attention_mode(precision, sparse_type, pre_rearranged)
        if precision != "mix":
            raise ValueError("mixed producer requires precision='mix'")
        self._init_common(reorder_mapping, state, precision, sparse_type, pre_rearranged)
        self._init_mixed()

    def _init_common(self, reorder_mapping, state, precision, sparse_type, pre_rearranged):
        """Initialize fields shared with the distributed execution chain."""
        self.reorder_mapping = reorder_mapping
        self.precision = precision
        self.sparse_type = sparse_type
        self.pre_rearranged = pre_rearranged
        self.optimize_attention_pipeline = state.optimize_attention_pipeline
        self.state = state
        self.group = state.group
        self.retained_blocks = {}
        self._qkv_receive_buffer_cache = None
        self._reset_call_stats()
        self.st_q = self.st_k = self.layout = self.unpacker = None

    def _init_mixed(self):
        """Allocate Q/K quantization buffers and the mixed packet consumer."""
        reorder_mapping = self.reorder_mapping
        device = reorder_mapping.reordered_position_to_token.device
        hc, hs, world, chunking = (
            reorder_mapping.hc,
            reorder_mapping.hs,
            reorder_mapping.world,
            reorder_mapping.chunking,
        )

        head_row = head_to_row(chunking, world)
        n_tok = int(reorder_mapping.local_rank_mapping["n_tok"])
        for name in ("st_q", "st_k"):
            state_buffer = _q.make_quant_state(
                reorder_mapping.local_rank_mapping,
                hc,
                hs,
                reorder_mapping.nblk,
                n_tok,
                device,
                reorder_mapping.num_cross_rank_quant_blocks,
                dtype=torch.bfloat16,
                head_row=head_row,
            )
            setattr(self, name, state_buffer)
        for st in (self.st_q, self.st_k):
            if st.payload.shape[1] != reorder_mapping.s_local:
                st.payload = torch.zeros(
                    (hc, reorder_mapping.s_local, hs), dtype=st.payload.dtype, device=st.payload.device
                )

        _q.check_sendable((reorder_mapping.bs, reorder_mapping.s_local, hc, hs), chunking, world)

        self.layout = _pack.derive_packet_layout(
            world=world,
            s_local=reorder_mapping.s_local,
            hs=hs,
            nblk=reorder_mapping.nblk,
            npool=reorder_mapping.npool,
        )
        self.unpacker = _pack.Unpacker(
            self.layout,
            rank_mappings=reorder_mapping.rank_mappings,
            pool_cell_token_counts=reorder_mapping.pool_cell_token_counts,
        )

    def compute_v_amax(self, value) -> torch.Tensor:
        """Return this rank's V inf-norm partial, excluding rows beyond used_len."""
        reorder_mapping = self.reorder_mapping
        rows = int(reorder_mapping.local_rank_mapping["n_tok"])
        if rows == 0:
            return torch.zeros(
                reorder_mapping.bs, reorder_mapping.hc, reorder_mapping.hs, dtype=torch.float32, device=value.device
            )
        window = value if rows == reorder_mapping.s_local else value[:, :rows]
        return torch.linalg.vector_norm(window, ord=float("inf"), dim=1).float()  # pylint: disable=not-callable  # PyTorch native callable.

    def produce(self, query, key, value) -> torch.Tensor:
        """Build the single flat Q/K-boundary and V-scale MAX-reduction operand."""
        boundary_amax_q = _q.quantize_amax_for_allreduce(query, self.st_q)
        boundary_amax_k = _q.quantize_amax_for_allreduce(key, self.st_k)
        _q.quantize_and_pool_for_amax(query, self.st_q, emit_amax=False, gather=False)
        _q.quantize_and_pool_for_amax(key, self.st_k, emit_amax=False, gather=False)
        qk = torch.stack((boundary_amax_q.t(), boundary_amax_k.t()), dim=-1).contiguous()
        v_amax = self.compute_v_amax(value)
        return torch.cat((qk.reshape(-1), v_amax.reshape(-1)))

    def pack(self, vec) -> tuple[list, torch.Tensor]:
        """Repair boundary scales and eagerly build every chunk's Q/K packet."""
        reorder_mapping = self.reorder_mapping
        qk_numel = reorder_mapping.num_cross_rank_quant_blocks * reorder_mapping.hc * 2
        vec_numel = (
            reorder_mapping.num_cross_rank_quant_blocks * reorder_mapping.hc * 2
            + reorder_mapping.bs * reorder_mapping.hc * reorder_mapping.hs
        )
        if vec.numel() != vec_numel:
            raise ValueError("the all_reduce operand must carry %d fp32 values, got %d" % (vec_numel, vec.numel()))
        qk = vec[:qk_numel].view(reorder_mapping.num_cross_rank_quant_blocks, reorder_mapping.hc, 2)
        forced_q = qk[:, :, 0].t().contiguous()
        forced_k = qk[:, :, 1].t().contiguous()
        _q.repair_boundary_and_scatter(self.st_q, forced_q)
        _q.repair_boundary_and_scatter(self.st_k, forced_k)
        v_scales = (
            vec[qk_numel:].view(reorder_mapping.bs, reorder_mapping.hc, reorder_mapping.hs) / _q.FP8_E4M3_MAX
        ).contiguous()

        W, D, C = reorder_mapping.world, reorder_mapping.hs, reorder_mapping.chunking
        head_of_row = torch.as_tensor(
            row_to_head(C, W), dtype=torch.int64, device=reorder_mapping.reordered_position_to_token.device
        )
        sc_q = self.st_q.scales.index_select(0, head_of_row)
        sc_k = self.st_k.scales.index_select(0, head_of_row)
        pool_sums_q = self.st_q.pool_sum.index_select(0, head_of_row)  # [hc, nc, hs]
        pool_sums_k = self.st_k.pool_sum.index_select(0, head_of_row)
        nc = int(pool_sums_q.shape[1])
        packs = []
        for c in range(C):
            qsc = sc_q.view(C, W, reorder_mapping.nblk)[c].unsqueeze(0)
            ksc = sc_k.view(C, W, reorder_mapping.nblk)[c].unsqueeze(0)
            pool_sum_q = pool_sums_q.view(C, W, nc, D)[c]
            pool_sum_k = pool_sums_k.view(C, W, nc, D)[c]
            packs.append(
                (
                    _pack.pack_qk_packet(self.st_q.payload, qsc, pool_sum_q, c, layout=self.layout, chunks=C),
                    _pack.pack_qk_packet(self.st_k.payload, ksc, pool_sum_k, c, layout=self.layout, chunks=C),
                )
            )
        return packs, v_scales

    def pack_v(self, value, v_scales) -> torch.Tensor:
        """Quantize V directly into its all-to-all send layout."""
        reorder_mapping = self.reorder_mapping
        bs, _s_local, _hc, hs = (int(x) for x in value.shape)
        sc_div = torch.where(v_scales == 0, torch.ones_like(v_scales), v_scales)
        return _q.to_send(value, sc_div, reorder_mapping.chunking, reorder_mapping.world).view(
            reorder_mapping.chunking, reorder_mapping.world, reorder_mapping.s_local, bs, reorder_mapping.w, hs
        )

    def take_v_scale(self, v_scales, chunk):
        """Select the current rank/chunk head's assembled V scale."""
        reorder_mapping = self.reorder_mapping
        head = reorder_mapping.rank * reorder_mapping.chunking + int(chunk)
        return v_scales[:, head, :].unsqueeze(1)

    def make_mask(self, pool_q, pool_k, *, scale, sparsity):
        """Build a mixed-path binary mask and reuse immutable protected block sets."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import get_multi_span_blockwise_mask

        reorder_mapping = self.reorder_mapping
        retained = self.retained_blocks.get("retained")
        if retained is None:
            retained = self.retained_blocks["retained"] = (
                set(reorder_mapping.dense_blocks),
                set(reorder_mapping.first_frame_blocks),
            )
        dense_blocks, first_frame_blocks = retained
        pooled = torch.cat((pool_q, pool_k, pool_k), dim=0)
        mask = get_multi_span_blockwise_mask(
            pooled,
            sparsity,
            scale,
            dense_blocks,
            first_frame_blocks,
            "BSND",
            return_binary=True,
        )
        return mask

    def prepare_chunk(self, chunk, rq, rk, rv, v_scales, *, scale, sparsity):
        """Unpack mixed Q/K/V, select V scale and build the chunk mask."""
        q8, k8, qsc, ksc, pool_q, pool_k = self.unpacker.unpack_pair_batched(rq, rk)
        return (
            q8,
            k8,
            qsc,
            ksc,
            _q.from_send(rv),
            self.take_v_scale(v_scales, chunk),
            self.make_mask(pool_q, pool_k, scale=scale, sparsity=sparsity),
        )

    def _reset_call_stats(self):
        """Initialize diagnostics now; execution PRs update them later."""
        self.last_separate_qk_pool_calls = self.last_common_reverse_buffer_merge_calls = 0
        self._head_major_qkv_active = False
        self.last_head_major_qkv_calls = self.last_head_major_qkv_view_calls = self.last_round_robin_merge_calls = 0
        self.last_qkv_receive_allocations = self.last_qkv_receive_reuses = 0
        self.last_qkv_layout_fallback_reason = None
