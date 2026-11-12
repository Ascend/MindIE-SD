# SPDX-License-Identifier: Apache-2.0
"""Public BF16-in/BF16-out distributed sparse-attention entry."""

# pylint: disable=no-name-in-module  # prerequisites provide the native layer boundary

from __future__ import annotations

import sys

import torch

from . import _layers, topology
from . import pipeline as _pipeline
from .attention_reorder_mapping import get_attention_reorder_mapping
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

_CONFIG_STATS_FIELDS = ("used_len", "s_global", "s_local", "hc", "hs", "bs", "world", "chunking", "head_groups", "w")


def distributed_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spans,
    chunking: int | None = None,
    head_groups: int | None = None,
    *,
    group=None,
    world: int | None = None,
    rank: int | None = None,
    scale: float | None = None,
    sparsity: float = 0.8,
    inner_precise: int | None = None,
    block_size_kv: int = 128,
    used_len: int | None = None,
    bs: int | None = None,
    state: ParallelState | None = None,
    return_stats: bool = False,
    precision: str = "mix",
    sparse_type: str = "rf_v3",
    pre_rearranged: bool = False,
):
    """Process one local ``[B,S_local,H,D]`` BF16 shard and preserve its shape.

    ``mix`` uses Q/K INT8 plus V FP8 on A5; ``bf16`` sends native values. RF-v2 and RF-v3 share
    the communication flow. ``pre_rearranged`` is restricted to BF16 RF-v3 and preserves RF row order.
    ``used_len`` excludes the global padding tail. Set ``return_stats`` for per-call diagnostics.
    """
    if isinstance(state, ParallelState):
        state._last_call_stats = None
    _validate_attention_mode(precision, sparse_type, pre_rearranged)
    a5 = _layers.is_a5_device()
    if a5 and sparse_type == "rf_v2":
        raise ValueError("A5 devices require sparse_type='rf_v3'")
    if inner_precise is None:
        inner_precise = _layers.A5_INNER_PRECISE if a5 else 0
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        raise ValueError("q, k, v must be tensors")
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "q/k/v must be [bs, s_local, hc, hs] and equal; got %s / %s / %s"
            % (tuple(q.shape), tuple(k.shape), tuple(v.shape)),
        )
    if any(x.dtype is not torch.bfloat16 for x in (q, k, v)):
        raise ValueError("q/k/v must be bf16, got %s" % q.dtype)
    bs_q, s_local, hc, hs = (int(x) for x in q.shape)
    bs = bs_q if bs is None else int(bs)
    if bs != bs_q:
        raise ValueError("bs=%d but q.shape[0]=%d" % (bs, bs_q))
    world, rank = topology.resolve_world_rank(group, world, rank)
    if precision == "mix":
        if not a5:
            raise RuntimeError("precision='mix' requires the EQBSA path on Ascend 950PR/950DT (A5)")
        require_inner_precise(inner_precise, _layers.A5_INNER_PRECISE)
    elif a5:
        require_inner_precise(inner_precise, _layers.A5_INNER_PRECISE)
    elif inner_precise not in (0, _layers.A5_INNER_PRECISE):
        raise ValueError("bf16 inner_precise must be 0 or %d" % _layers.A5_INNER_PRECISE)
    elif inner_precise != 0:
        raise ValueError("bf16 on non-A5 devices requires inner_precise=0")
    op_block = (int(_layers.DEVICE_OP_BLOCK_SHAPE[0]), int(_layers.DEVICE_OP_BLOCK_SHAPE[1]))
    if (int(MASK_BLOCK), int(POOL_BLOCK)) != op_block or int(block_size_kv) != op_block[0]:
        raise ValueError(
            "the chain's mask/pool block (%d, %d), the device operator's block shape (%d, %d) and "
            "block_size_kv=%d must agree"
            % (int(MASK_BLOCK), int(POOL_BLOCK), op_block[0], op_block[1], int(block_size_kv)),
        )

    config = derive_parallel_config(
        world=int(world),
        hc=hc,
        s_local=s_local,
        used_len=used_len,
        spans=spans,
        chunking=chunking,
        head_groups=head_groups,
        bs=bs,
        hs=hs,
        rank=int(rank),
    )
    scale = float(hs) ** -0.5 if scale is None else float(scale)

    pstate = state if state is not None else ParallelState(group=group)
    pstate.bind_or_check(group=group, world=int(world), rank=int(rank), device=q.device)
    if pstate.overlap_comm:
        pstate._check_execution()

    if pstate.bsa_backend == "eagle" and sparse_type != "rf_v3":
        raise ValueError("bsa_backend='eagle' requires sparse_type='rf_v3'")
    if not pstate.optimize_attention_pipeline:
        pstate.token_block_mapping_cache.clear()
        pstate.reorder_mapping_cache.clear()
        pstate.chains.clear()

    reorder_mapping = get_attention_reorder_mapping(
        config,
        q.device,
        token_block_mapping_cache=pstate.token_block_mapping_cache,
        reorder_mapping_cache=pstate.reorder_mapping_cache,
    )

    chain = pstate.chain(
        reorder_mapping,
        build=_pipeline.Chain,
        precision=precision,
        sparse_type=sparse_type,
        pre_rearranged=pre_rearranged,
    )
    consumer = _pipeline.CONSUMER if precision == "mix" else sparse_type

    out, call_counts = chain.run(
        q,
        k,
        v,
        scale=scale,
        sparsity=sparsity,
        inner_precise=inner_precise,
    )

    try:
        _require_collective_counts(call_counts, precision, config.chunking)
    except RuntimeError:
        pstate._last_chain = None
        if pstate.overlap_comm:
            pstate._fail_execution()
        raise
    allreduce = call_counts["allreduce"]

    pstate._record_call(config.key, precision, sparse_type, call_counts)
    if not pstate.marker_printed:
        pstate.marker_printed = True
        _say(
            "rank=%d FUSED PRODUCER ENGAGED chunks=%d world=%d s_local=%d "
            "used_len=%d nboundary=%d consumer=%s allreduce=%d"
            % (
                config.rank,
                config.chunking,
                config.world,
                config.s_local,
                config.used_len,
                reorder_mapping.num_cross_rank_quant_blocks,
                consumer,
                allreduce,
            )
        )

    stats = _build_stats(
        config,
        reorder_mapping,
        pstate,
        precision,
        sparse_type,
        consumer,
        call_counts,
        pre_rearranged=pre_rearranged,
    )
    pstate._last_call_stats = _copy_stats({**stats, "actual": _build_actual_stats(pstate, precision, sparse_type)})
    return (out, stats) if return_stats else out


def _require_collective_counts(counts, precision, chunks):
    """Accept the complete wire protocol before committing successful-call diagnostics."""
    allreduce, a2a = counts["allreduce"], counts["a2a_calls"]
    if allreduce != (1 if precision == "mix" else 0):
        message = (
            "exactly ONE all_reduce must be issued per call (the globally assembled per-block "
            "amax, which also carries V's inf-norm partials); issued %d"
            if precision == "mix"
            else "bf16 must not issue a scale all_reduce; issued %d"
        ) % allreduce
    elif a2a != 4 * chunks:
        message = "exactly %d all_to_all_single must be issued per call; issued %d" % (4 * chunks, a2a)
    else:
        return
    error = RuntimeError(message)
    error.parallel_call_counts = dict(counts)
    raise error


def _build_stats(
    config, reorder_mapping, pstate, precision, sparse_type, consumer, call_counts, *, pre_rearranged=False
):
    """Build requested diagnostics while preserving caller-visible keys and counters."""
    allreduce = call_counts["allreduce"]
    a2a_calls = call_counts["a2a_calls"]
    fused_stages = call_counts["fused_stages"]
    chain_stats = {
        "a2a_calls": a2a_calls,
        "ar_calls": allreduce,
        **pstate._chain_totals[(config.key, precision, sparse_type)],
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
            "optimize_attention_pipeline": pstate.optimize_attention_pipeline,
            "bsa_backend": pstate.bsa_backend,
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
        "state": pstate.stats(),
        "mask": _pipeline.CHAIN_IMPL if precision == "mix" else "%s_comm" % sparse_type,
    }
    return stats


def _build_actual_stats(pstate, precision, sparse_type):
    """Read only host counters from the completed execution, without device synchronization."""
    chain = pstate._last_chain
    fallback_reason = getattr(chain, "last_qkv_layout_fallback_reason", None)
    shared = pstate.qkv_layout_plan
    if precision == "mix" and (shared.requested if shared is not None else pstate.head_major_qkv):
        fallback_reason = "attention_mode_unsupported"
    return {
        "bsa_backend": (
            "eagle_quant" if precision == "mix" else "rain_fusion" if sparse_type == "rf_v2" else pstate.bsa_backend
        ),
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


def _say(msg):
    """One diagnostic line on stderr, prefixed with the package name."""
    sys.stderr.write("[mindiesd.parallel] %s\n" % (msg,))
    sys.stderr.flush()
