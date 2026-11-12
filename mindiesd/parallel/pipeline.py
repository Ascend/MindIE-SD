# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Distributed mixed/BF16 attention execution and stream scheduling."""

# pylint: disable=attribute-defined-outside-init,no-member,no-name-in-module

import torch
import torch.distributed as dist

from . import distributed_quant as _q
from . import _layers
from .bf16_primitives import BF16Primitives
from .config import _validate_attention_mode
from .mixed_producer import CHAIN_IMPL, CONSUMER, MixedProducer
from .qkv_layout import plan_qkv_layout
from .topology import (
    _head_major_qkv_geometry_supported,
    merge_ranks,
    merge_reverse_receive_chunks,
    merge_reverse_receive_chunks_round_robin,
    split_into_ranks,
)

__all__ = ["Chain", "CONSUMER", "CHAIN_IMPL"]


def _create_comm_stream(device):
    """Create a device resource; its reuse and lifetime belong to ParallelState."""
    return torch.npu.Stream(device=device)


def _record_stream(tensors, stream):
    """Keep cross-stream tensor storage alive after this Python call returns."""
    if isinstance(tensors, torch.Tensor):
        tensors.record_stream(stream)
    else:
        for tensor in tensors:
            _record_stream(tensor, stream)


def _can_use_common_reverse_receive_buffer(
    result_view, mapping, precision, sparse_type, pipeline_optimizations_enabled
):
    return (
        pipeline_optimizations_enabled
        and precision == "bf16"
        and sparse_type == "rf_v3"
        and mapping.bs == mapping.w == 1
        and mapping.hs == 128
        and result_view.device.type == "npu"
    )


def _head_major_qkv_decision(
    q, k, v, mapping, *, state, precision, sparse_type, optimize_attention_pipeline, pre_rearranged
):
    """Accept alias sends only for the measured BF16 round-robin storage contract."""
    local, dim = mapping.s_local, mapping.hs
    shared = getattr(state, "qkv_layout_plan", None)
    plan = plan_qkv_layout(
        head_count=mapping.hc,
        head_dim=dim,
        batch_size=mapping.bs,
        world=mapping.world,
        chunking=mapping.chunking,
        width=mapping.w,
        precision=precision,
        sparse_type=sparse_type,
        pre_rearranged=pre_rearranged,
        overlap_comm=state.overlap_comm,
        optimize_attention_pipeline=optimize_attention_pipeline,
        requested=shared.requested if shared is not None else getattr(state, "head_major_qkv", False),
    )
    if shared is not None and shared != plan:
        return False, "shared_plan_mismatch"
    if not plan.supported:
        return False, plan.fallback_reason
    supported = all(
        tuple(tensor.shape) == (1, local, mapping.hc, dim)
        and tensor.device.type == "npu"
        and tensor.dtype == torch.bfloat16
        and tensor.stride(2) == local * dim
        and tensor.stride(-1) == 1
        and (local == 1 or tensor.stride(1) == dim)
        for tensor in (q, k, v)
    )
    return supported, None if supported else "tensor_layout_unsupported"


def _can_use_head_major_qkv(
    q, k, v, mapping, *, state, precision, sparse_type, optimize_attention_pipeline, pre_rearranged
):
    """Preserve the existing bool helper, including final physical stride gates."""
    return _head_major_qkv_decision(**locals())[0]


class Chain(MixedProducer, BF16Primitives):
    """Combine producer, native BF16 primitives and collective scheduling."""

    __slots__ = ()

    def __init__(self, reorder_mapping, *, state, precision="mix", sparse_type="rf_v3", pre_rearranged=False):
        _validate_attention_mode(precision, sparse_type, pre_rearranged)
        self._init_common(reorder_mapping, state, precision, sparse_type, pre_rearranged)
        if precision == "bf16":
            self._init_bf16()
        else:
            self._init_mixed()

    def _all_reduce_max(self, vec, counts):
        """Reduce the global scale operand and count the actual collective call."""
        dist.all_reduce(vec, op=dist.ReduceOp.MAX, group=self.group)
        counts["allreduce"] += 1

    def _a2a(self, recv, send, counts):
        """Exchange one payload on the current stream and count the actual collective call."""
        dist.all_to_all_single(recv, send, group=self.group)
        counts["a2a_calls"] += 1

    def run(self, q, k, v, *, scale, sparsity, inner_precise) -> tuple[torch.Tensor, dict]:
        """Manage execution state and dispatch precision; return output and per-call counts."""
        counts = {"allreduce": 0, "a2a_calls": 0, "fused_stages": 0}
        self._reset_call_stats()
        self.state._last_chain = None
        self.state._last_call_stats = None
        overlap = self.state.overlap_comm  # pylint: disable=no-member  # Caller supplies ParallelState.
        if overlap:
            compute = torch.npu.current_stream(q.device)
            self.state._begin_execution(compute, build=lambda: _create_comm_stream(q.device))  # pylint: disable=no-member  # Caller supplies ParallelState.
        try:
            execute = self._run_bf16 if self.precision == "bf16" else self._run_mix
            result = execute(q, k, v, scale=scale, sparsity=sparsity, inner_precise=inner_precise, counts=counts)
            if overlap:
                tail = torch.npu.Event()
                tail.record(compute)
                self.state._finish_execution(tail)  # pylint: disable=no-member  # Caller supplies ParallelState.
            self.state._last_chain = self
            return result
        except Exception as error:
            error.parallel_call_counts = dict(counts)
            if overlap:
                self.state._fail_execution()  # pylint: disable=no-member  # Caller supplies ParallelState.
            raise

    def _run_mix(self, q, k, v, *, scale, sparsity, inner_precise, counts) -> tuple[torch.Tensor, dict]:
        """Quantize/pool, reduce once, repair/pack, then schedule EQBSA chunks."""
        reorder_mapping = self.reorder_mapping

        vec = self.produce(q, k, v)
        counts["fused_stages"] += 1

        self._all_reduce_max(vec, counts)

        packs, v_scales = self.pack(vec)
        counts["fused_stages"] += 1

        v8 = self.pack_v(v, v_scales)
        sends = [(packs[gi][0], packs[gi][1], v8[gi]) for gi in range(reorder_mapping.chunking)]

        def _run_fa(gi, rq, rk, rv, out_dst):
            """Compute one chunk into its reverse-a2a buffer; temporaries stay local to this call."""
            q8, k8, qsc, ksc, v8_g, v_scale, mask = self.prepare_chunk(
                gi, rq, rk, rv, v_scales, scale=scale, sparsity=sparsity
            )
            used = reorder_mapping.used_len
            qq = q8[:, :used].index_select(1, reorder_mapping.reordered_position_to_token)  # (bs, used, 1, hs) int8
            kk = k8[:, :used].index_select(1, reorder_mapping.reordered_position_to_token)
            vv8 = v8_g[:, :used].index_select(1, reorder_mapping.reordered_position_to_token)
            out = _layers.eagle_quant_block_sparse_attention(
                query=qq.transpose(1, 2).contiguous(),
                key=kk.transpose(1, 2).contiguous(),
                value=vv8.transpose(1, 2).contiguous(),
                block_sparse_mask=mask,
                scale_value=scale,
                inner_precise=inner_precise,
                actual_seq_lengths=[used],
                actual_seq_lengths_kv=[used],
                query_scale=qsc,
                key_scale=ksc,
                value_scale=v_scale,
            )
            out = out.permute(0, 2, 1, 3).contiguous()  # BSND (metadata-only at w == 1, re-verified)
            torch.index_select(out, 1, reorder_mapping.token_to_reordered_position, out=out_dst)
            return out_dst

        bs, world, hs = reorder_mapping.bs, reorder_mapping.world, reorder_mapping.hs
        s_local = reorder_mapping.s_local
        result = torch.empty(bs, s_local, reorder_mapping.hc, hs, dtype=q.dtype, device=q.device)
        result_view = result.view(bs, s_local, world, reorder_mapping.head_groups, hs)
        if self.state.overlap_comm:  # pylint: disable=no-member  # Caller supplies ParallelState.
            self._run_overlap(sends, result_view, _run_fa, counts)
        else:
            self._run_serial(sends, result_view, _run_fa, counts)
        return result, counts

    def _run_bf16(self, q, k, v, *, scale, sparsity, inner_precise, counts):
        """Exchange BF16 heads; eligible head-major inputs alias round-robin packets."""
        mapping = self.reorder_mapping
        bs, local, world, chunks, width, dim = (
            mapping.bs,
            mapping.s_local,
            mapping.world,
            mapping.chunking,
            mapping.w,
            mapping.hs,
        )
        self._head_major_qkv_active = False
        head_major, self.last_qkv_layout_fallback_reason = _head_major_qkv_decision(
            q,
            k,
            v,
            mapping,
            state=self.state,
            precision=self.precision,
            sparse_type=self.sparse_type,
            optimize_attention_pipeline=self.optimize_attention_pipeline,
            pre_rearranged=getattr(self, "pre_rearranged", False),
        )
        if self.optimize_attention_pipeline:
            if head_major:
                prepared = [
                    tensor[0].transpose(0, 1).view(chunks, world, local, bs, width, dim) for tensor in (q, k, v)
                ]
                if not all(tensor.is_contiguous() for tensor in prepared):
                    raise RuntimeError("head-major packets must be contiguous views before communication")
                self._head_major_qkv_active = True
                self.last_head_major_qkv_calls += 1
                self.last_head_major_qkv_view_calls += 3
            else:
                prepared = [
                    tensor.reshape(bs, local, world, chunks, width, dim).permute(3, 2, 1, 0, 4, 5).contiguous()
                    for tensor in (q, k, v)
                ]
            sends = list(zip(*prepared))
        else:
            views = [tensor.reshape(bs, local, world, chunks, width, dim) for tensor in (q, k, v)]
            sends = [
                tuple(value[:, :, :, chunk].permute(2, 1, 0, 3, 4).contiguous() for value in views)
                for chunk in range(chunks)
            ]
        counts["fused_stages"] += 2

        sparse_inputs = []
        sparse_mask = []

        def prepare_all(inputs):
            tensors = [
                torch.cat([_q.from_send(received[i])[:, : mapping.used_len] for received in inputs], dim=2)
                for i in range(3)
            ]
            reordered, pooled = self._prepare_bf16_tensors(tensors)
            sparse_inputs.extend(reordered)
            sparse_mask.append(self._make_bf16_sparse_mask(pooled, scale=scale, sparsity=sparsity))

        def compute_chunk(chunk, rq, rk, rv, out_dst):
            start = chunk * width
            values = [tensor[:, :, start : start + width].contiguous() for tensor in sparse_inputs]
            mask = self._slice_bf16_sparse_mask(sparse_mask[0], start, width)
            return self._compute_bf16_head(
                values,
                mask,
                out_dst,
                scale=scale,
                inner_precise=inner_precise,
            )

        result = torch.empty(bs, local, mapping.hc, dim, dtype=q.dtype, device=q.device)
        result_view = result.view(bs, local, world, mapping.head_groups, dim)
        if self.state.overlap_comm:  # pylint: disable=no-member  # Caller supplies ParallelState.

            def compute_ready_head(chunk, rq, rk, rv, out_dst):
                tensors, pooled = self._prepare_bf16_head(rq, rk, rv)
                mask = self._make_bf16_sparse_mask(pooled, scale=scale, sparsity=sparsity)
                return self._compute_bf16_head(tensors, mask, out_dst, scale=scale, inner_precise=inner_precise)

            self._run_overlap(sends, result_view, compute_ready_head, counts)
        else:
            self._run_serial(sends, result_view, compute_chunk, counts, prepare_all=prepare_all)
        return result, counts

    def _run_serial(self, sends, result_view, compute_chunk, counts, *, prepare_all=None):
        """Retain the original chunk-by-chunk collective order on the caller stream."""
        mapping = self.reorder_mapping
        bs, world, w, hs = mapping.bs, mapping.world, mapping.w, mapping.hs
        s_global, used = mapping.s_global, mapping.used_len
        chunk_outputs = [None] * mapping.chunking
        inputs = []
        if prepare_all is not None:
            for sq, sk, sv in sends:
                values = [torch.empty_like(t) for t in (sq, sk, sv)]
                self.last_qkv_receive_allocations += 3
                for recv, send in zip(values, (sq, sk, sv)):
                    self._a2a(recv, send, counts)
                inputs.append((*values, None))
            prepare_all(inputs)
        for gi, (sq, sk, sv) in enumerate(sends):
            if prepare_all is not None:
                rq, rk, rv, _ = inputs[gi]
            else:
                rq, rk, rv = torch.empty_like(sq), torch.empty_like(sk), torch.empty_like(sv)
                self.last_qkv_receive_allocations += 3
                self._a2a(rq, sq, counts)
                self._a2a(rk, sk, counts)
                self._a2a(rv, sv, counts)

            out_pad = result_view.new_empty(bs, s_global, w, hs)
            out_pad[:, used:].zero_()
            chunk_outputs[gi] = compute_chunk(gi, rq, rk, rv, out_pad[:, :used])

            send = split_into_ranks(out_pad, world)
            recv = torch.empty_like(send)
            self._a2a(recv, send, counts)
            result_view[:, :, :, gi * w : (gi + 1) * w, :].copy_(merge_ranks(recv))

    def _run_overlap(self, sends, result_view, compute_chunk, counts):
        """Queue forward transfers, compute ready chunks, then join reverse transfers."""
        mapping = self.reorder_mapping
        use_common_reverse_receive_buffer = _can_use_common_reverse_receive_buffer(
            result_view, mapping, self.precision, self.sparse_type, self.optimize_attention_pipeline
        )
        if getattr(self, "_head_major_qkv_active", False):
            if not use_common_reverse_receive_buffer or not _head_major_qkv_geometry_supported(
                mapping.chunking, mapping.world, mapping.w, mapping.bs, mapping.hs
            ):
                raise RuntimeError("round-robin ownership requires common reverse before any A2A")
        compute = torch.npu.current_stream(result_view.device)
        comm = self.state._comm_stream  # pylint: disable=no-member  # Caller supplies ParallelState.
        prepared = torch.npu.Event()
        prepared.record(compute)
        _record_stream(sends, comm)
        inputs = []
        with torch.npu.stream(comm):
            comm.wait_event(prepared)
            # Calls sharing this state are serialized. The prepared
            # event follows all prior chunk reads and returned-output joins,
            # so comm can safely overwrite the prior call's receive storage.
            if self.optimize_attention_pipeline:
                signature = tuple(
                    (tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for triple in sends for t in triple
                )
                cached = getattr(self, "_qkv_receive_buffer_cache", None)
                if cached is None or cached[0] != signature:
                    buffers = [tuple(torch.empty_like(t) for t in triple) for triple in sends]
                    self._qkv_receive_buffer_cache = (signature, buffers)
                    self.last_qkv_receive_allocations += 3 * len(sends)
                else:
                    buffers = cached[1]
                    self.last_qkv_receive_reuses += 3 * len(sends)
            else:
                buffers = (tuple(torch.empty_like(t) for t in triple) for triple in sends)
                self.last_qkv_receive_allocations += 3 * len(sends)
            for (sq, sk, sv), (rq, rk, rv) in zip(sends, buffers):
                _record_stream((rq, rk, rv), compute)
                self._a2a(rq, sq, counts)
                self._a2a(rk, sk, counts)
                self._a2a(rv, sv, counts)
                ready = torch.npu.Event()
                ready.record(comm)
                inputs.append((rq, rk, rv, ready))

        outputs = []
        reverse_receive_chunks = None
        if use_common_reverse_receive_buffer:
            with torch.npu.stream(comm):
                reverse_receive_chunks = result_view.new_empty(
                    len(sends), mapping.world, mapping.w, mapping.s_local, mapping.bs, mapping.hs
                )
            _record_stream(reverse_receive_chunks, compute)
        for gi, (rq, rk, rv, ready) in enumerate(inputs):
            compute.wait_event(ready)
            out_pad = result_view.new_empty(mapping.bs, mapping.s_global, mapping.w, mapping.hs)
            out_pad[:, mapping.used_len :].zero_()
            compute_chunk(gi, rq, rk, rv, out_pad[:, : mapping.used_len])
            computed = torch.npu.Event()
            computed.record(compute)
            _record_stream(out_pad, comm)
            with torch.npu.stream(comm):
                comm.wait_event(computed)
                send = split_into_ranks(out_pad, mapping.world)
                recv = reverse_receive_chunks[gi] if use_common_reverse_receive_buffer else torch.empty_like(send)
                _record_stream(recv, compute)
                self._a2a(recv, send, counts)
                returned = torch.npu.Event()
                returned.record(comm)
            outputs.append((recv, returned))

        for gi, (recv, returned) in enumerate(outputs):
            compute.wait_event(returned)
            if not use_common_reverse_receive_buffer:
                result_view[:, :, :, gi * mapping.w : (gi + 1) * mapping.w, :].copy_(merge_ranks(recv))
        if use_common_reverse_receive_buffer:
            head_major = getattr(self, "_head_major_qkv_active", False)
            merger = merge_reverse_receive_chunks_round_robin if head_major else merge_reverse_receive_chunks
            result_view.copy_(merger(reverse_receive_chunks))
            self.last_common_reverse_buffer_merge_calls += 1
            self.last_round_robin_merge_calls += int(head_major)
