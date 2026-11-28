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
"""FA4 execution of a single-device block-causal prefill in BSND layout."""

from functools import lru_cache
import importlib

import torch

from ._prebuilt import _backend_prefix

_prefix = _backend_prefix()
tla = importlib.import_module(f"{_prefix}.utils").get_tla()
_backend = importlib.import_module(_prefix)
MaskMod = importlib.import_module(f"{_prefix}.mods").MaskMod


@tla.jit
def _visible(b, h, q, kv, info, aux, scalars):
    visible = tla.Bool(False)
    length = tla.Int64(aux[0].shape[0])
    q_index = q.to(tla.Int64)
    kv_index = kv.to(tla.Int64)
    if (q_index < length) & (kv_index < length):
        q_span = aux[0][q_index]
        kv_span = aux[0][kv_index]
        visible = (q_index >= kv_index) | ((q_span >= 0) & (q_span == kv_span))
    return visible


_MASK = MaskMod(_visible, broadcast_batch=True, broadcast_head=True)


@lru_cache(maxsize=1)
def _prepare(spans, length, device, stream):
    # Keep the block metadata and its backing tensor on the stream that prepared them.
    span_ids = torch.full((length,), -1, dtype=torch.int32, device="cpu")
    for index, (start, end) in enumerate(spans):
        span_ids[start:end] = index
    aux = (span_ids.to(device),)
    blocks = _backend.compute_block_sparsity(128, 128, None, None, length, length, _MASK, aux, device, aux_scalars=())
    return blocks, aux


def run(query, key, value, spans, scale):
    blocks, aux = _prepare(spans, query.size(1), query.device, torch.npu.current_stream(query.device))
    output, _ = _backend.flash_attn_func(
        query, key, value, softmax_scale=scale, mask_mod=_MASK, block_sparse_tensors=blocks, aux_tensors=aux
    )
    return output.contiguous()
