#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""CATLASS DSL block metadata and local attention dispatch."""

from functools import lru_cache
from typing import NamedTuple

from catlass import tla
from flash_attn_npu_4.catlass_dsl import compute_block_sparsity, flash_attn_func
from flash_attn_npu_4.catlass_dsl.mods import MaskMod, Mod

# The FA4 sparse ABI accepts only this block shape.
_SPARSE_BLOCK_SHAPE = (128, 128)


class _PreparedMask(NamedTuple):
    mask_mod: MaskMod
    score_mod: Mod | None
    blocks: object
    aux_tensors: tuple
    aux_scalars: tuple


@tla.jit
def _block_causal(b, h, q, kv, info, aux, scalars):
    visible = tla.Bool(False)
    length = tla.Int64(aux[0].shape[0])
    if (q < length) & (kv < length):
        q_block = aux[0][q]
        kv_block = aux[0][kv]
        same_block = (q_block >= 0) & (q_block == kv_block)
        visible = ((q >= kv) | same_block) & (aux[1][b, q] != 0) & (aux[1][b, kv] != 0)
    return visible


@lru_cache(maxsize=128)
def _map_mask(rule, batch_shared, head_shared):
    @tla.jit
    def mapped(b, h, q, kv, info, aux, scalars):
        # Input maps may use either integer width; retain global positions in Int64.
        q_global = aux[0][q].to(tla.Int64)
        kv_global = aux[1][kv].to(tla.Int64)
        visible = (q_global >= tla.Int64(0)) & (kv_global >= tla.Int64(0))
        if tla.const_expr(rule is not None):
            if visible:
                batch = b
                if tla.const_expr(batch_shared):
                    batch = tla.Int32(0)
                visible = rule(batch, aux[2][h], q_global, kv_global, info, aux[3:], scalars)
        return visible

    return MaskMod(mapped, broadcast_batch=batch_shared, broadcast_head=head_shared)


@lru_cache(maxsize=128)
def _map_score(rule):
    @tla.jit
    def mapped(score, b, h, q, kv, info, aux, scalars):
        q_global = aux[0][q].to(tla.Int64)
        kv_global = aux[1][kv].to(tla.Int64)
        return rule(score, b, aux[2][h], q_global, kv_global, info, aux[3:], scalars)

    return Mod(mapped)


def prepare(mask, q_map, kv_map, heads, batch_size):
    mask._validate(batch_size)
    rule = _block_causal if mask._block_causal else mask.mask_mod
    mask_mod = _map_mask(rule, mask.batch_shared, mask.head_shared)
    score_mod = None if mask.score_mod is None else _map_score(mask.score_mod)
    aux_tensors = (q_map, kv_map, heads, *mask.aux_tensors)
    blocks = compute_block_sparsity(
        *_SPARSE_BLOCK_SHAPE,
        None if mask.batch_shared else batch_size,
        None if mask.head_shared else heads.numel(),
        q_map.numel(),
        kv_map.numel(),
        mask_mod,
        aux_tensors,
        q_map.device,
        aux_scalars=mask.aux_scalars,
    )
    return _PreparedMask(mask_mod, score_mod, blocks, aux_tensors, mask.aux_scalars)


def attention(query, key, value, prepared, scale, return_lse):
    out, lse = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        mask_mod=prepared.mask_mod,
        score_mod=prepared.score_mod,
        block_sparse_tensors=prepared.blocks,
        aux_tensors=prepared.aux_tensors,
        aux_scalars=prepared.aux_scalars,
        return_lse=return_lse,
    )
    return (out, lse) if return_lse else out
