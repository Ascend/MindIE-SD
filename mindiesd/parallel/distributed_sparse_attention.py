# SPDX-License-Identifier: Apache-2.0
"""Public distributed sparse-attention entry."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from . import _layers, topology
from . import pipeline as _pipeline
from .attention_reorder_mapping import get_attention_reorder_mapping
from .capabilities import resolve_bsa_consumer
from .config import (
    MASK_BLOCK,
    POOL_BLOCK,
    QUANT_BLOCK,
    derive_parallel_config,
    require_inner_precise,
    _validate_attention_mode,
)
from .state import ParallelState, _copy_stats

__all__ = ["distributed_sparse_attention"]

logger = logging.getLogger("mindie-sd")

_CONFIG_STATS_FIELDS = ("used_len", "s_global", "s_local", "hc", "hs", "bs", "world", "chunking", "head_groups", "w")
_SERIAL_MAPPING_CACHE_MAX = 4


def distributed_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spans,
    chunking: int | None = None,
    head_groups: int | None = None,
    *,
    # Caller-owned state and optional topology checks.
    state: ParallelState,
    world: int | None = None,
    rank: int | None = None,
    # Wire precision and numerical operator policy.
    precision: str = "mix",
    inner_precise: int | None = None,
    # Sparse attention selection and score scaling.
    sparse_type: str = "rf_v3",
    sparsity: float = 0.8,
    scale: float | None = None,
    block_size_kv: int = 128,
    # Input geometry and RF row order.
    used_len: int | None = None,
    bs: int | None = None,
    pre_rearranged: bool = False,
    # Host-only per-call diagnostics.
    return_stats: bool = False,
):
    """Preserve [B,S_local,H,D] dtype/shape. RF order is accepted; CP requires pre_rearranged=True.

    Mix uses INT8 Q/K, FP8 V; MXFP8 needs both native extensions on every rank; float modes send native values.
    State owns groups/caches; world/rank check Ulysses; used_len excludes padding; return_stats enables diagnostics.
    """
    if not isinstance(state, ParallelState):
        raise TypeError("state must be a ParallelState owning the process groups")
    state._last_call_stats = None
    if sparse_type == "rf_v2" and getattr(state, "cp_group", None) is not None:
        if not hasattr(_layers, "rf_v2_block_sparse_attention"):
            raise ValueError("RF-v2 CP requires the CP consumer integration")
    _validate_attention_mode(precision, sparse_type, pre_rearranged)
    bsa_backend, bsa_operator = resolve_bsa_consumer(
        precision,
        state.bsa_backend,
        sparse_type,
    )
    if sparse_type == "rf_v2" and _layers.is_a5_device():
        raise ValueError("A5 devices require sparse_type='rf_v3'")
    if inner_precise is None:
        inner_precise = _layers.A5_INNER_PRECISE if precision in ("mix", "mxfp8") else 0
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        raise ValueError("q, k, v must be tensors")
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "q/k/v must be [bs, s_local, hc, hs] and equal; got %s / %s / %s"
            % (tuple(q.shape), tuple(k.shape), tuple(v.shape)),
        )
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if any(x.dtype != dtype for x in (q, k, v)):
        raise ValueError(f"precision={precision!r} requires q/k/v dtype {dtype}")
    if precision in ("bf16", "fp16") and q.device.type == "npu" and not _layers.is_a2_or_a3_device():  # pylint: disable=no-member
        raise RuntimeError(f"{sparse_type} with precision={precision!r} requires A2/A3")
    bs_q, s_local, hc, hs = (int(x) for x in q.shape)
    bs = bs_q if bs is None else int(bs)
    if bs != bs_q:
        raise ValueError("bs=%d but q.shape[0]=%d" % (bs, bs_q))
    group = state.group
    world, rank = topology.resolve_world_rank(group, world, rank)
    kv_group = state.cp_group
    kv_rank = None
    if kv_group is not None:
        from .hybrid_sparse_attention import resolve_cp_rank

        if not pre_rearranged:
            raise ValueError('cp_group requires globally pre-rearranged token shards')
        if bs != 1 or hs != 128:
            raise ValueError('CP x Ulysses currently requires batch=1 and head_dim=128')
        kv_rank = resolve_cp_rank(group, kv_group, world, rank)
    if precision == "mix":
        if not _layers.is_a5_device():
            raise RuntimeError("precision='mix' requires the EQBSA path on Ascend 950PR/950DT (A5)")
        require_inner_precise(inner_precise, _layers.A5_INNER_PRECISE)
    elif precision == "mxfp8":
        if bs != 1 or hs != 128:
            raise ValueError("native MXFP8 BSA currently requires batch=1 and head_dim=128")
        if not _layers.is_a5_device():
            raise RuntimeError("precision='mxfp8' BSA requires Ascend 950PR/950DT (A5)")
        try:
            from ._native_mxfp8 import require_native_mxfp8_runtime  # pylint: disable=import-error,no-name-in-module
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}._native_mxfp8":
                raise
            raise ValueError("MXFP8 public routing is supplied by the native consumer PR") from exc

        require_native_mxfp8_runtime()
        require_inner_precise(inner_precise, _layers.A5_INNER_PRECISE)
    elif inner_precise != 0:
        raise ValueError("BF16/FP16 on A2/A3 requires inner_precise=0")
    op_block = (int(_layers.DEVICE_OP_BLOCK_SHAPE[0]), int(_layers.DEVICE_OP_BLOCK_SHAPE[1]))
    if (int(MASK_BLOCK), int(POOL_BLOCK)) != op_block or int(block_size_kv) != op_block[0]:
        raise ValueError(
            "the chain's mask/pool block (%d, %d), the device operator's block shape (%d, %d) and "
            "block_size_kv=%d must agree"
            % (int(MASK_BLOCK), int(POOL_BLOCK), op_block[0], op_block[1], int(block_size_kv)),
        )

    cp_size = dist.get_world_size(kv_group) if kv_group is not None else 1
    config = derive_parallel_config(
        world=int(world),
        hc=hc,
        s_local=s_local * cp_size,
        used_len=used_len,
        spans=spans,
        chunking=chunking,
        head_groups=head_groups,
        bs=bs,
        hs=hs,
        rank=int(rank),
    )
    scale = float(hs) ** -0.5 if scale is None else float(scale)
    query_window = None
    if kv_group is not None:
        from .hybrid_sparse_attention import derive_query_window

        query_window = derive_query_window(kv_rank, int(world) * s_local, config.used_len, cp_size)

    pstate = state
    pstate.bind_or_check(group=group, world=int(world), rank=int(rank), device=q.device)
    if kv_group is not None:
        pstate.bind_cp_group(kv_group, kv_rank)
        pstate._check_execution()
    if pstate.overlap_comm or q.device.type == "npu" or precision == "mxfp8":
        pstate._check_execution()

    serial_mapping_cache = (
        precision in ("bf16", "fp16")
        and sparse_type in ("rf_v2", "rf_v3")
        and kv_group is None
        and not pstate.overlap_comm
        and not pstate.optimize_attention_pipeline
    )
    if not pstate.optimize_attention_pipeline and precision in ("bf16", "fp16"):
        # Float serial caches immutable mappings; overlap and optimized RF preparation stay disabled.
        if not serial_mapping_cache:
            pstate.token_block_mapping_cache.clear()
            pstate.reorder_mapping_cache.clear()
        pstate.chains.clear()
    pstate._serial_mapping_cache_active = serial_mapping_cache

    # Shared foundation extends this helper with RF input-order selection.
    # pylint: disable=unexpected-keyword-arg
    reorder_mapping = get_attention_reorder_mapping(
        config,
        q.device,
        token_block_mapping_cache=pstate.token_block_mapping_cache,
        reorder_mapping_cache=pstate.reorder_mapping_cache,
        pre_rearranged=pre_rearranged,
    )
    # pylint: enable=unexpected-keyword-arg
    if serial_mapping_cache:
        for cache in (pstate.token_block_mapping_cache, pstate.reorder_mapping_cache):
            while len(cache) > _SERIAL_MAPPING_CACHE_MAX:
                cache.pop(next(iter(cache)))

    chain = pstate.chain(
        reorder_mapping,
        build=_pipeline.Chain,
        precision=precision,
        sparse_type=sparse_type,
        pre_rearranged=pre_rearranged,
    )
    consumer = "mxfp8" if precision == "mxfp8" else _pipeline.CONSUMER if precision == "mix" else sparse_type

    run_options = dict(scale=scale, sparsity=sparsity, inner_precise=inner_precise)
    if kv_group is not None:
        run_options.update(cp_group=kv_group, kv_gather_rank=kv_rank)
    out, call_counts = chain.run(q, k, v, **run_options)

    try:
        _require_collective_counts(
            call_counts, precision, config.chunking, hybrid=kv_group is not None, quantized_comm=pstate.quantized_comm
        )
        if kv_group is not None:
            boundary = query_window['boundary_collective']
            if call_counts['kv_gather_calls'] != 2 * config.chunking:
                raise RuntimeError('incomplete KV gather protocol')
            if call_counts['query_boundary_gather_calls'] != int(boundary) * config.chunking:
                raise RuntimeError('incomplete global query-block boundary protocol')
    except RuntimeError:
        pstate._last_chain = None
        if pstate.overlap_comm or q.device.type == "npu" or kv_group is not None or precision == "mxfp8":
            pstate._fail_execution()
        raise

    pstate._record_call(config.key, precision, sparse_type, call_counts)
    if not pstate.marker_printed:
        pstate.marker_printed = True
        logger.debug("[parallel] rank=%d world=%d mode=%s/%s", config.rank, config.world, sparse_type, precision)

    # Commit host-only diagnostics lazily; never read tensor values or synchronize.
    counts_snapshot = dict(call_counts)
    process_snapshot = pstate.stats()
    engine_snapshot = dict(pstate._chain_totals[(config.key, precision, sparse_type)])
    actual = _build_actual_stats(pstate, precision, bsa_operator)

    def build_stats():
        stats = _build_stats(
            config,
            reorder_mapping,
            pstate,
            precision,
            sparse_type,
            consumer,
            counts_snapshot,
            bsa_backend=bsa_backend,
            pre_rearranged=pre_rearranged,
            process_snapshot=process_snapshot,
            engine_snapshot=engine_snapshot,
        )
        if kv_group is not None:
            stats['plan']['s_local'] = s_local
            stats['plan']['query_s_global'] = int(world) * s_local
            stats['plan']['sequence_world'] = int(world) * cp_size
            stats['hybrid'] = dict(
                ulysses_size=int(world),
                kv_gather_size=cp_size,
                kv_gather_rank=kv_rank,
                global_rank=kv_rank * int(world) + rank,
                kv_gather_calls=counts_snapshot['kv_gather_calls'],
                query_boundary_gather_calls=counts_snapshot['query_boundary_gather_calls'],
                full_query_gather=bool(boundary and int(world) * s_local <= POOL_BLOCK),
                query_window=query_window,
                reverse_a2a_calls=counts_snapshot.get('reverse_a2a_calls', config.chunking),
                forward_precision="bf16" if not pstate.quantized_comm and precision in ("mix", "mxfp8") else precision,
                quantized_collective_calls=counts_snapshot.get('quantized_collective_calls', 0),
            )
        return stats

    if return_stats:
        stats = build_stats()
        pstate._last_call_stats = _copy_stats({**stats, "actual": actual})
        return out, stats

    def build_last_call_stats():
        return {**build_stats(), "actual": actual}

    pstate._last_call_stats = build_last_call_stats
    return out


def _require_collective_counts(counts, precision, chunks, hybrid=False, quantized_comm=True):
    """Accept the complete wire protocol before committing successful-call diagnostics."""
    allreduce, a2a = counts["allreduce"], counts["a2a_calls"]
    expected_allreduce = {"bf16": 0, "fp16": 0, "mix": 2 if hybrid else 1, "mxfp8": 2}[precision]
    if not quantized_comm:
        expected_allreduce = 0
        if precision in ("mix", "mxfp8") and not hybrid and counts.get("quantized_bsa_after_comm_calls") != chunks:
            raise RuntimeError("BF16 communication requires every quantized BSA head after exchange")
    expected_a2a = 4 * chunks
    if allreduce != expected_allreduce:
        message = f"{precision} requires {expected_allreduce} all_reduce calls; issued {allreduce}"
    elif a2a != expected_a2a:
        message = "exactly %d all_to_all_single must be issued per call; issued %d" % (expected_a2a, a2a)
    else:
        return
    error = RuntimeError(message)
    error.parallel_call_counts = dict(counts)
    raise error


def _build_stats(
    config,
    reorder_mapping,
    pstate,
    precision,
    sparse_type,
    consumer,
    call_counts,
    *,
    bsa_backend,
    pre_rearranged=False,
    process_snapshot=None,
    engine_snapshot=None,
):
    """Build requested diagnostics while preserving caller-visible keys and counters."""
    allreduce = call_counts["allreduce"]
    a2a_calls = call_counts["a2a_calls"]
    fused_stages = call_counts["fused_stages"]
    chain_stats = {
        "a2a_calls": a2a_calls,
        "ar_calls": allreduce,
        **(pstate._chain_totals[(config.key, precision, sparse_type)] if engine_snapshot is None else engine_snapshot),
    }

    stats = {
        "tally": {
            "allreduce": allreduce,
            "fused_stages": fused_stages,
            "consumer": consumer,
            "a2a_calls": a2a_calls,
        },
        "plan": {
            "pre_rearranged": pre_rearranged,
            "quantized_comm": pstate.quantized_comm,
            "optimize_attention_pipeline": pstate.optimize_attention_pipeline,
            "bsa_backend": bsa_backend,
            "mask_algorithm": "rainfusion",
            **{name: getattr(config, name) for name in _CONFIG_STATS_FIELDS},
            "nblk": reorder_mapping.nblk,
            "ncell": reorder_mapping.ncell,
            "npartial": reorder_mapping.num_cross_rank_quant_blocks,
            "npool": reorder_mapping.npool,
            "quant_block": QUANT_BLOCK,
            "mask_block": MASK_BLOCK,
            "n_dense_blocks": len(reorder_mapping.dense_blocks),
            "n_first_frame_blocks": len(reorder_mapping.first_frame_blocks),
        },
        "engine": chain_stats,
        "state": pstate.stats() if process_snapshot is None else process_snapshot,
        # Preserve the historical execution-path label for existing diagnostic consumers.
        "mask": _pipeline.CHAIN_IMPL
        if precision == "mix"
        else "%s_comm" % precision
        if precision == "mxfp8"
        else "%s_comm" % sparse_type,
    }
    return stats


def _build_actual_stats(pstate, precision, bsa_operator):
    """Read only host counters from the completed execution, without device synchronization."""
    chain = pstate._last_chain
    fallback_reason = getattr(chain, "last_qkv_layout_fallback_reason", None)
    shared = pstate.qkv_layout_plan
    if precision == "mix" and (shared.requested if shared is not None else pstate.head_major_qkv):
        fallback_reason = "attention_mode_unsupported"
    return {
        "bsa_backend": bsa_operator,
        "serial_mapping_cache": getattr(pstate, "_serial_mapping_cache_active", False),
        "head_major_qkv": bool(getattr(chain, "_head_major_qkv_active", False)),
        "head_major_qkv_calls": getattr(chain, "last_head_major_qkv_calls", 0),
        "head_major_qkv_view_calls": getattr(chain, "last_head_major_qkv_view_calls", 0),
        "round_robin_merge_calls": getattr(chain, "last_round_robin_merge_calls", 0),
        "common_reverse_buffer_merge_calls": getattr(chain, "last_common_reverse_buffer_merge_calls", 0),
        "separate_qk_pool_calls": getattr(chain, "last_separate_qk_pool_calls", 0),
        "qkv_receive_allocations": getattr(chain, "last_qkv_receive_allocations", 0),
        "qkv_receive_reuses": getattr(chain, "last_qkv_receive_reuses", 0),
        "qkv_layout_fallback_reason": fallback_reason,
    }
