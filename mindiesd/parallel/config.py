# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Derive and validate the complete parallel split before device work."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "QUANT_BLOCK",
    "MASK_BLOCK",
    "POOL_BLOCK",
    "ParallelConfig",
    "derive_parallel_config",
    "require_inner_precise",
]


QUANT_BLOCK = 64
MASK_BLOCK = 128
POOL_BLOCK = 128


def _validate_attention_mode(precision, sparse_type, pre_rearranged):
    """Share legal wire/consumer/order combinations across entry and cached construction."""
    if not isinstance(pre_rearranged, bool):
        raise ValueError("pre_rearranged must be a bool and agree across ranks")
    if precision not in ("mix", "bf16"):
        raise ValueError("precision must be 'mix' or 'bf16', got %r" % precision)
    if sparse_type not in ("rf_v2", "rf_v3"):
        raise ValueError("sparse_type must be 'rf_v2' or 'rf_v3', got %r" % sparse_type)
    if pre_rearranged and (precision != "bf16" or sparse_type != "rf_v3"):
        raise ValueError("pre_rearranged=True requires precision='bf16' and sparse_type='rf_v3'")
    if precision == "mix" and sparse_type != "rf_v3":
        raise ValueError("precision='mix' requires sparse_type='rf_v3'")


@dataclass(frozen=True)
class ParallelConfig:
    """Everything derivable without touching a device."""

    world: int
    rank: int
    hc: int
    hs: int
    bs: int
    s_local: int
    s_global: int
    used_len: int
    chunking: int
    head_groups: int  # = hc // world, the heads one rank owns
    w: int  # = head_groups // chunking, heads per (chunk, destination)
    spans: tuple  # raw spans, as the caller passed them
    spans_norm: tuple  # ((start, shape, length), ...) -- the library's own form

    @property
    def key(self):
        return (
            self.rank,
            self.world,
            self.hc,
            self.hs,
            self.bs,
            self.s_local,
            self.used_len,
            self.chunking,
            self.head_groups,
            self.w,
            tuple(sorted(self.spans_norm)),
        )


def _span_start_shape(span, index):
    """Accept both the deployed dict form and a plain object (``.latent_grid``)."""
    if isinstance(span, dict):
        try:
            start = int(span["start"])
            shape = tuple(int(v) for v in span["latent_shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("spans[%d] needs int start and latent_shape=[T,H,W]" % index) from exc
    else:
        try:
            start = int(span.start)
            shape = tuple(int(v) for v in span.latent_grid)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("spans[%d] needs int start and latent_grid=[T,H,W]" % index) from exc
    if len(shape) != 3 or any(d <= 0 for d in shape):
        raise ValueError("spans[%d].latent_shape must be three positive ints, got %r" % (index, shape))
    return start, shape


def _assert_span_covers_used_len(spans_norm, used_len, s_global):
    """Require a single span to end at used_len, never at the padded global length."""
    if len(spans_norm) != 1:
        return
    start, _, length = spans_norm[0]
    if start + length != used_len:
        raise ValueError(
            "prod(latent_grid)+span.start must equal used_len: %d + %d = %d, "
            "but used_len=%d (s_global=%d).  Comparing against s_global is "
            "the documented cause of a silent stock fallback; the %d pad rows "
            "are NOT part of the rearrange" % (length, start, start + length, used_len, s_global, s_global - used_len),
        )


def derive_parallel_config(
    *, world, hc, s_local, used_len, spans, chunking=None, head_groups=None, bs=1, hs=None, rank=0
):
    """Normalize the caller's arguments and assert the split (see the module docstring)."""
    if spans is None or len(spans) == 0:
        raise ValueError("spans must be a non-empty sequence")
    if used_len is None:
        raise ValueError(
            "used_len is required: it is the FA's own sequence length and it is "
            "NOT s_global (the delivered geometry has used_len=109088 vs "
            "s_global=109120; the 32-row tail is padding)",
        )
    world = int(world)
    if world < 1:
        raise RuntimeError("world must be >= 1, got %d" % world)
    s_local = int(s_local)
    hc = int(hc)
    used_len = int(used_len)
    bs = int(bs)
    rank = int(rank)
    s_global = world * s_local
    if not 0 <= rank < world:
        raise ValueError("rank=%d out of range [0, world=%d)" % (rank, world))

    derived_hg = hc // world
    hg = derived_hg if head_groups is None else int(head_groups)
    chunking = hg if chunking is None else int(chunking)
    if hc % world or hg != derived_hg or chunking < 1 or hg % chunking:
        raise RuntimeError(
            "the head split is not exact: hc=%d world=%d -> head_groups must be %d (got %r) and "
            "chunking=%r must be >= 1 and divide it" % (hc, world, derived_hg, head_groups, chunking),
        )
    w = hg // chunking
    if w != 1:
        raise RuntimeError(
            "heads per (chunk, destination) slot w = head_groups // chunking = %d // %d = %d, "
            "but the packet carries exactly ONE head per (chunk, destination) slot: chunking "
            "must be head_groups=%d (hc=%d / world=%d), which makes the head sets a bijection "
            "over hc -- one head per slot, none dropped" % (hg, chunking, w, hg, hc, world),
        )

    if used_len <= 0 or used_len > s_global:
        raise ValueError("used_len=%d out of range (0, world*s_local=%d]" % (used_len, s_global))

    norm = []
    total = 0
    previous_end = 0
    for i, span in enumerate(spans):
        start, shape = _span_start_shape(span, i)
        length = math.prod(shape)
        if start < 0 or start + length > used_len:
            raise ValueError(
                "spans[%d]=[%d,%d) is outside the used sequence length %d" % (i, start, start + length, used_len),
            )
        if start < previous_end:
            raise ValueError("spans[%d] starts at %d, before the previous span ends at %d" % (i, start, previous_end))
        previous_end = start + length
        total += length
        norm.append((start, shape, length))
    if total > used_len:
        raise ValueError("the spans cover %d rows but used_len=%d" % (total, used_len))
    _assert_span_covers_used_len(norm, used_len, s_global)

    if QUANT_BLOCK * 2 != MASK_BLOCK or MASK_BLOCK != POOL_BLOCK:
        raise RuntimeError("quant block %d must be mask/pool block %d // 2" % (QUANT_BLOCK, MASK_BLOCK))

    if hs is None:
        hs = 0
    config = ParallelConfig(
        world=world,
        rank=rank,
        hc=hc,
        hs=int(hs),
        bs=bs,
        s_local=s_local,
        s_global=s_global,
        used_len=used_len,
        chunking=chunking,
        head_groups=hg,
        w=w,
        spans=tuple(spans),
        spans_norm=tuple(norm),
    )
    return config


def require_inner_precise(inner_precise, expected):
    """Require the library-owned A5 precision constant instead of silently overriding it."""
    want = int(expected)
    if int(inner_precise) != want:
        raise ValueError(
            "inner_precise must be %d on A5 (got %r): mindiesd's sparse_attention "
            "forces it when it routes rf_v2 -> rf_v3, and passing 0 made the "
            "kernel fail with ERR00100" % (want, inner_precise),
        )
