# SPDX-License-Identifier: Apache-2.0
"""Public RF row plans built by the existing multi-span rearranger.

Spans describe original sequence coordinates, including any dense padding rows.
The caller owns index tensors and must keep their contents unchanged while a
plan is in use. Building a plan does not rearrange the caller's Q/K/V storage.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, eq=False)
class RFMultiSpanPlan:
    """Immutable metadata and caller-owned indices for one complete sequence."""

    permutation: object
    inverse: object
    dense_blocks: frozenset
    first_frame_blocks: frozenset
    video_spans: tuple
    sequence_len: int
    pool_size: int
    input_layout: str


def build_rf_attention_plan(video_spans, sequence_len, pool_size=128, input_layout="BSND", device="cpu"):
    """Wrap the stock RF permutation/protection helpers without changing their order.

    ``permutation[position]`` is the original token at an RF position;
    ``inverse[token]`` is its RF position. Sequence length is the complete
    physical row count; any padding outside video spans remains dense context.
    Unsupported clip boundary layouts raise the original validation errors.
    """
    from .layers.flash_attn.sparse_flash_attn_rf_v2 import _normalize_video_spans, _multi_span_permutation

    if input_layout not in ("BSND", "BNSD"):
        raise ValueError("input_layout must be BSND or BNSD")
    if not isinstance(sequence_len, int) or sequence_len <= 0:
        raise ValueError("sequence_len must be a positive integer")
    if not isinstance(pool_size, int) or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    spans = _normalize_video_spans(video_spans, sequence_len)
    permutation, inverse, dense, first = _multi_span_permutation(spans, sequence_len, pool_size, input_layout, device)
    return RFMultiSpanPlan(
        permutation,
        inverse,
        frozenset(dense),
        frozenset(first),
        tuple((start, shape) for start, shape, _ in spans),
        sequence_len,
        pool_size,
        input_layout,
    )


def rf_multi_span_permutation(video_spans, sequence_len, pool_size=128, input_layout="BSND", device="cpu"):
    """Return the stock ``(permutation, inverse, dense_blocks, first_frame_blocks)`` tuple."""
    plan = build_rf_attention_plan(video_spans, sequence_len, pool_size, input_layout, device)
    return plan.permutation, plan.inverse, set(plan.dense_blocks), set(plan.first_frame_blocks)


__all__ = ["RFMultiSpanPlan", "build_rf_attention_plan", "rf_multi_span_permutation"]
