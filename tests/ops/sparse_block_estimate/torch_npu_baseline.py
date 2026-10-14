#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""A5 SBE composition using ordinary PyTorch operations on NPU tensors.

Import torch_npu before passing NPU tensors. No MindIE library is required.
CPU execution is supported for algorithm tests. ``input`` matmul returns the
input dtype before conversion to FP32; it is NOT the kernel's FP32 QK output.
The default ``fp32`` casts operands before bmm to avoid low-precision QK rounding.
Neither mode promises bitwise agreement with the fused online-softmax kernel.
"""

import math

import torch
from torch.nn import functional


def _select_mask(scores, threshold, row_sparse, keep_sink):
    """Batched equivalent of sbe_reference._select_mask, without host reads."""
    if threshold == 1.0:
        return torch.ones_like(scores, dtype=torch.bool)
    candidates = scores.clone()
    if keep_sink:
        candidates[..., 0] = 0
    # Non-causal A5 protects the last block regardless of keep_recent.
    candidates[..., -1] = 0
    ordered = torch.sort(candidates, dim=-1, descending=True).values
    target = candidates.sum(dim=-1, keepdim=True) * threshold
    selected = (ordered.cumsum(dim=-1) < target).sum(dim=-1, keepdim=True) + 1
    cap = scores.shape[-1]
    if row_sparse < 1.0:
        cap = max(1, math.floor(cap * row_sparse))
    selected = selected.clamp(max=cap)
    guard = ordered.gather(-1, selected - 1) - 1.0e-6
    mask = scores >= guard
    if keep_sink:
        mask[..., 0] = True
    mask[..., -1] = True
    return mask


def _validate(query, key, threshold, row_sparse, scale_value, matmul_mode, q_chunk_rows):
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must be BNSD tensors")
    if min(*query.shape, *key.shape) <= 0:
        raise ValueError("all dimensions must be positive")
    if query.shape[0] != key.shape[0] or query.shape[-1] != 128 or key.shape[-1] != 128:
        raise ValueError("batch sizes must match and head_dim must be 128")
    if query.shape[1] % key.shape[1] or key.shape[2] > 425984:
        raise ValueError("Q heads must be divisible by KV heads; K must not exceed 425984")
    if query.dtype != key.dtype or query.device != key.device:
        raise ValueError("query/key dtype and device must match")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16 and float32 (reference testing)")
    if not 0 < threshold <= 1 or not 0 < row_sparse <= 1:
        raise ValueError("threshold and row_sparse must be in (0, 1]")
    if not math.isfinite(scale_value):
        raise ValueError("scale_value must be finite")
    if matmul_mode not in ("input", "fp32"):
        raise ValueError("matmul_mode must be input or fp32")
    if not isinstance(q_chunk_rows, int) or q_chunk_rows < 0 or q_chunk_rows % 16:
        raise ValueError("q_chunk_rows must be zero (all rows) or a positive multiple of 16")


@torch.no_grad()
def sparse_block_estimate_baseline(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale_value: float = 128**-0.5 / 8,
    threshold: float = 0.85,
    row_sparse: float = 1.0,
    keep_sink: bool = True,
    keep_recent: bool = True,
    matmul_mode: str = "fp32",
    q_chunk_rows: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int8 mask [B,Hq,ceil(Q/128),aligned_K_blocks] and int32 counts.

    Fixed A5 contract: BNSD, stride=8, square sparse_size=128, non-causal,
    no actual_seq_lengths. q_chunk_rows counts packed rows within each KV
    head's concatenated Q heads. Chunking preserves the complete K softmax.
    Packing, casts, temporary allocations and output generation occur here,
    so timing this function includes the entire composition.
    On NPU, disable torch.npu.matmul.allow_hf32 before calling the FP32 path;
    the benchmark runner does this once, outside timing.
    """
    _validate(query, key, threshold, row_sparse, scale_value, matmul_mode, q_chunk_rows)
    _ = keep_recent
    batch, q_heads, q_seq, _ = query.shape
    kv_heads, k_seq = key.shape[1:3]
    ratio = q_heads // kv_heads
    q_blocks, k_blocks = (q_seq + 127) // 128, (k_seq + 127) // 128
    q_rows, k_cols = q_blocks * 16, (k_seq + 7) // 8
    packed_q = functional.pad(query, (0, 0, 0, q_blocks * 128 - q_seq))
    packed_q = packed_q.reshape(batch, q_heads, q_rows, 8, 128).flip(-2)
    packed_q = packed_q.reshape(batch * kv_heads, ratio * q_rows, 1024)
    packed_k = functional.pad(key, (0, 0, 0, k_cols * 8 - k_seq))
    packed_k = packed_k.reshape(batch * kv_heads, k_cols, 1024)
    if matmul_mode == "fp32":
        packed_q, packed_k = packed_q.float(), packed_k.float()
    key_transposed = packed_k.transpose(1, 2)
    weights = (q_seq - torch.arange(q_rows, device=query.device) * 8).clamp(0, 8)
    weights = (weights.float() / 8).repeat(ratio)
    rows_per_group = ratio * q_rows
    chunk = q_chunk_rows or rows_per_group
    mask = torch.empty((batch * kv_heads, ratio * q_blocks, k_blocks), device=query.device, dtype=torch.int8)
    for start in range(0, rows_per_group, chunk):
        end = min(start + chunk, rows_per_group)
        logits = torch.bmm(packed_q[:, start:end], key_transposed).float()
        logits.mul_(scale_value)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        del logits
        probabilities.mul_(weights[None, start:end, None])
        probabilities = functional.pad(probabilities, (0, k_blocks * 16 - k_cols))
        scores = probabilities.reshape(batch * kv_heads, (end - start) // 16, 16, k_blocks, 16)
        scores = scores.sum(dim=(2, 4))
        del probabilities
        mask[:, start // 16 : end // 16] = _select_mask(scores, threshold, row_sparse, keep_sink).to(torch.int8)
    mask = mask.reshape(batch, q_heads, q_blocks, k_blocks)
    count = mask.sum(dim=-1, dtype=torch.int32)
    mask = functional.pad(mask, (0, (-k_blocks) % 32))
    return mask, count
