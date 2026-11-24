# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision packet producer and consumer preparation."""

# pylint: disable=no-name-in-module  # provided by the preceding mapping/quant PR

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch

from . import _layers
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
    _head_of_row: torch.Tensor | None
    _qk_packet_buffers: object | None
    _direct_qkv_send: torch.Tensor | None
    _direct_v_send: torch.Tensor | None
    _source_rows_identity: bool

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
        self._head_of_row = self._qk_packet_buffers = None
        self._direct_qkv_send = self._direct_v_send = None
        self._source_rows_identity = False

    def _init_mixed(self):
        """Allocate Q/K quantization buffers and the mixed packet consumer."""
        if self.optimize_attention_pipeline:
            params = inspect.signature(_layers.get_multi_span_blockwise_mask).parameters
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()) and not {
                "retained_mask",
                "retained_idx",
                "pool_parts",
            }.issubset(params):
                raise RuntimeError("mixed attention requires a matching RF mask helper")
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
        if self.optimize_attention_pipeline and self.pre_rearranged and device.type != 'meta':
            # Geometry maps are immutable for this cached producer. Confirm the
            # prefix once; arbitrary RF permutations retain the indexed path.
            self._source_rows_identity = all(
                torch.equal(st.maps.src, torch.arange(st.maps.n_owned, dtype=st.maps.src.dtype, device=device))
                for st in (self.st_q, self.st_k)
            )

        _q.check_sendable((reorder_mapping.bs, reorder_mapping.s_local, hc, hs), chunking, world)

        self.layout = _pack.derive_packet_layout(
            world=world,
            s_local=reorder_mapping.s_local,
            hs=hs,
            nblk=reorder_mapping.nblk,
            npool=reorder_mapping.npool,
        )
        if self.optimize_attention_pipeline:
            self._head_of_row = torch.tensor(row_to_head(chunking, world), dtype=torch.int64, device=device)
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

    def _can_use_identity_gather(self):
        return self.optimize_attention_pipeline and self.pre_rearranged and self._source_rows_identity

    def produce(self, query, key, value, *, defer_pool=False) -> torch.Tensor:
        """Build the single flat Q/K-boundary and V-scale MAX-reduction operand."""
        source_identity = self._can_use_identity_gather()
        boundary_amax_q = _q.quantize_amax_for_allreduce(query, self.st_q, source_identity=source_identity)
        boundary_amax_k = _q.quantize_amax_for_allreduce(key, self.st_k, source_identity=source_identity)
        if self.optimize_attention_pipeline:
            # MAX may repair the scales; defer all payload writes until it returns.
            if not defer_pool:
                _q._pool_sums(self.st_q)
                _q._pool_sums(self.st_k)
        else:
            _q.quantize_and_pool_for_amax(query, self.st_q, emit_amax=False, gather=False)
            _q.quantize_and_pool_for_amax(key, self.st_k, emit_amax=False, gather=False)
        qk = torch.stack((boundary_amax_q.t(), boundary_amax_k.t()), dim=-1).contiguous()
        v_amax = self.compute_v_amax(value)
        return torch.cat((qk.reshape(-1), v_amax.reshape(-1)))

    def _can_write_direct_packet(self):
        """Keep pitched output limited to the pure mixed A5 overlap path."""
        return (
            self.optimize_attention_pipeline
            and self.state.overlap_comm
            and self.st_q.payload.device.type == 'npu'
            and _layers.is_a5_device()
        )

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
        if self._can_write_direct_packet() and self._direct_qkv_send is None:
            C, W, S, D = (reorder_mapping.chunking, reorder_mapping.world, reorder_mapping.s_local, reorder_mapping.hs)
            P, L = self.layout.per_dest, self.layout.payload_bytes
            wire = torch.zeros(C * W * (2 * P + L), dtype=torch.int8, device=self.st_q.payload.device)
            self._direct_qkv_send = wire
            q_packet, k_packet, v_packet = wire.split((C * W * P, C * W * P, C * W * L))
            self._qk_packet_buffers = (q_packet.view(C, W, P), k_packet.view(C, W, P))
            self.st_q.payload = q_packet.view(C * W, P)[:, :L].view(C * W, S, D)
            self.st_k.payload = k_packet.view(C * W, P)[:, :L].view(C * W, S, D)
            self._direct_v_send = v_packet.view(C, W, S, 1, 1, D)
        qk = vec[:qk_numel].view(reorder_mapping.num_cross_rank_quant_blocks, reorder_mapping.hc, 2)
        forced_q = qk[:, :, 0].t().contiguous()
        forced_k = qk[:, :, 1].t().contiguous()
        if self.optimize_attention_pipeline:
            for state_buffer, forced in ((self.st_q, forced_q), (self.st_k, forced_k)):
                _q._derive_scales(state_buffer, forced_amax=forced)
                _q._quantize_into_fast(state_buffer)
        else:
            _q.repair_boundary_and_scatter(self.st_q, forced_q)
            _q.repair_boundary_and_scatter(self.st_k, forced_k)
        v_scales = (
            vec[qk_numel:].view(reorder_mapping.bs, reorder_mapping.hc, reorder_mapping.hs) / _q.FP8_E4M3_MAX
        ).contiguous()

        W, D, C = reorder_mapping.world, reorder_mapping.hs, reorder_mapping.chunking
        head_of_row = self._head_of_row
        if head_of_row is None:
            head_of_row = torch.as_tensor(
                row_to_head(C, W), dtype=torch.int64, device=reorder_mapping.reordered_position_to_token.device
            )
        sc_q = self.st_q.scales.index_select(0, head_of_row)
        sc_k = self.st_k.scales.index_select(0, head_of_row)
        pool_sums_q = self.st_q.pool_sum.index_select(0, head_of_row)  # [hc, nc, hs]
        pool_sums_k = self.st_k.pool_sum.index_select(0, head_of_row)
        nc = int(pool_sums_q.shape[1])
        if self._direct_qkv_send is not None:
            # The payload kernels already wrote the wire regions. Submit scale
            # and pool fields across all heads together, retaining each row's
            # original packet layout and rank-local pooling order.
            for buffer, scales, pools in zip(self._qk_packet_buffers, (sc_q, sc_k), (pool_sums_q, pool_sums_k)):
                bytes_view = buffer.view(torch.uint8)
                scale_end = self.layout.payload_bytes + self.layout.scale_bytes
                bytes_view[:, :, self.layout.payload_bytes : scale_end].view(torch.float32).copy_(
                    scales.view(C, W, reorder_mapping.nblk)
                )
                pool_view = (
                    bytes_view[:, :, self.layout.pool_off :].view(torch.float32).view(C, W, self.layout.npool, D)
                )
                pool_view[:, :, :nc].copy_(pools.view(C, W, nc, D))
            return list(zip(*self._qk_packet_buffers)), v_scales
        if self.optimize_attention_pipeline and self._qk_packet_buffers is None:
            self._qk_packet_buffers = tuple(
                torch.zeros(C, W, self.layout.per_dest, dtype=torch.int8, device=self.st_q.payload.device)
                for _ in range(2)
            )
        packs = []
        for c in range(C):
            qsc = sc_q.view(C, W, reorder_mapping.nblk)[c].unsqueeze(0)
            ksc = sc_k.view(C, W, reorder_mapping.nblk)[c].unsqueeze(0)
            pool_sum_q = pool_sums_q.view(C, W, nc, D)[c]
            pool_sum_k = pool_sums_k.view(C, W, nc, D)[c]
            packs.append(
                (
                    _pack.pack_qk_packet(
                        self.st_q.payload,
                        qsc,
                        pool_sum_q,
                        c,
                        layout=self.layout,
                        chunks=C,
                        buffer=self._qk_packet_buffers[0][c] if self._qk_packet_buffers is not None else None,
                    ),
                    _pack.pack_qk_packet(
                        self.st_k.payload,
                        ksc,
                        pool_sum_k,
                        c,
                        layout=self.layout,
                        chunks=C,
                        buffer=self._qk_packet_buffers[1][c] if self._qk_packet_buffers is not None else None,
                    ),
                )
            )
        return packs, v_scales

    def pack_v(self, value, v_scales) -> torch.Tensor:
        """Quantize V directly into its all-to-all send layout."""
        reorder_mapping = self.reorder_mapping
        bs, _s_local, _hc, hs = (int(x) for x in value.shape)
        sc_div = torch.where(v_scales == 0, torch.ones_like(v_scales), v_scales)
        write = _q._to_send_fast if self.optimize_attention_pipeline else _q.to_send
        options = {'out': self._direct_v_send} if self._direct_v_send is not None else {}
        return write(value, sc_div, reorder_mapping.chunking, reorder_mapping.world, **options).view(
            reorder_mapping.chunking, reorder_mapping.world, reorder_mapping.s_local, bs, reorder_mapping.w, hs
        )

    def take_v_scale(self, v_scales, chunk):
        """Select the current rank/chunk head's assembled V scale."""
        reorder_mapping = self.reorder_mapping
        head = reorder_mapping.rank * reorder_mapping.chunking + int(chunk)
        return v_scales[:, head, :].unsqueeze(1)

    def make_mask(self, pool_q, pool_k, *, scale, sparsity):
        """Build a mixed binary mask with cached protected device regions."""
        reorder_mapping = self.reorder_mapping
        retained = self.retained_blocks.get("retained")
        if retained is None:
            retained = self.retained_blocks["retained"] = (
                set(reorder_mapping.dense_blocks),
                set(reorder_mapping.first_frame_blocks),
            )
        dense_blocks, first_frame_blocks = retained
        options = {"return_binary": True}
        if self.optimize_attention_pipeline:
            signature = (pool_q.shape[1], pool_k.shape[1], pool_q.device)
            cached = self.retained_blocks.get("device_retained")
            if cached is None or cached[0] != signature:
                protected = torch.zeros(1, 1, pool_q.shape[1], pool_k.shape[1], dtype=torch.bool, device=pool_q.device)
                indices = torch.tensor(
                    sorted(dense_blocks | first_frame_blocks), dtype=torch.long, device=pool_q.device
                )
                if indices.numel():
                    protected[:, :, indices, :] = True
                    protected[:, :, :, indices] = True
                empty = torch.empty(0, dtype=torch.long, device=pool_q.device)
                cached = (signature, protected, empty)
                self.retained_blocks["device_retained"] = cached
            options.update(retained_mask=cached[1], retained_idx=cached[2], pool_parts=2)
            pooled = torch.cat((pool_q, pool_k), dim=0)
        else:
            pooled = torch.cat((pool_q, pool_k, pool_k), dim=0)
        mask = _layers.get_multi_span_blockwise_mask(
            pooled, sparsity, scale, dense_blocks, first_frame_blocks, "BSND", **options
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
