#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared BSA accuracy helpers: opbase 50/50 fill, CPU/NPU compose golden, gates."""

from __future__ import annotations

import math

import torch

SEED = 20260811
BLOCK_SHAPE = (128, 128)
BLOCK_SHAPE_K64 = (128, 64)
INNER_PRECISE = 0
SOFTMAX_LSE_FLAG = 0
LAYOUT = "BNSD"

SMALL_BATCH = 1
SMALL_NUM_Q_HEADS = 2
SMALL_NUM_KV_HEADS = 2
SMALL_HEAD_DIM = 128
SMALL_ALIGNED_SEQ = 256
SMALL_TAIL_SEQ = 257

PROD_BATCH = 1
PROD_NUM_Q_HEADS = 32
PROD_NUM_KV_HEADS = 4
PROD_SEQ = 32768
PROD_HEAD_DIM = 128

COSINE_MIN = 0.99
NORM_RATIO_LO = 0.9
NORM_RATIO_HI = 1.1

# experimental_standard mixed tolerance: (rtol, atol, required_matched_ratio, fixed_max_abs).
# https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
_MIXED_TOLERANCE = {
    "float16": (2**-9, 2**-9, 0.99, 1e-1),
    "bfloat16": (2**-6, 2**-6, 0.99, 1e-0),
    "float32": (2**-10, 2**-16, 0.99, 1e-2),
}
_ULP_AT_ONE = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-23,
}
_ULP_FACTOR = 32


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def softmax_scale(head_dim):
    return head_dim**-0.5


def generate_opbase_tensor(shape, dtype, generator):
    """50% U[-5, 5] + 50% Normal(mu in [-5, 5], sigma in [0.1, 2]), seed-controlled."""
    numel = 1
    for dim in shape:
        numel *= dim
    uniform = torch.empty(numel, dtype=torch.float32)
    uniform.uniform_(-5.0, 5.0, generator=generator)
    mu = torch.empty((), dtype=torch.float32)
    mu.uniform_(-5.0, 5.0, generator=generator)
    sigma = torch.empty((), dtype=torch.float32)
    sigma.uniform_(0.1, 2.0, generator=generator)
    normal = torch.empty(numel, dtype=torch.float32)
    normal.normal_(float(mu.item()), float(sigma.item()), generator=generator)
    pick = torch.empty(numel, dtype=torch.float32)
    pick.uniform_(0.0, 1.0, generator=generator)
    mixed = torch.where(pick < 0.5, uniform, normal)
    return mixed.reshape(shape).to(dtype)


def make_block_sparse_mask(batch, num_heads, query_seq_len, kv_seq_len, block_shape, generator):
    """About 50% True blocks; first/last KV block and last Q block stay connected."""
    block_q, block_kv = block_shape
    nq_blocks = ceil_div(query_seq_len, block_q)
    nkv_blocks = ceil_div(kv_seq_len, block_kv)
    rand = torch.rand((batch, num_heads, nq_blocks, nkv_blocks), generator=generator)
    mask = (rand > 0.5).to(torch.int8)
    mask[:, :, :, 0] = 1
    mask[:, :, :, nkv_blocks - 1] = 1
    mask[:, :, nq_blocks - 1, :] = 1
    return mask


def compose_block_sparse_attention(query, key, value, block_sparse_mask, block_shape, scale_value, out_dtype=None):
    """Block-sparse attention via matmul + masked softmax + matmul on query.device.

    Q is tiled by block_shape[0] so a production-length SxS score tensor is never
    materialized. Fully masked Q rows write zeros (softmax of all -inf).
    """
    if out_dtype is None:
        out_dtype = query.dtype
    device = query.device
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    batch, num_q_heads, query_seq_len, head_dim = query_f.shape
    num_kv_heads = key_f.shape[1]
    kv_seq_len = key_f.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num query heads must be divisible by num kv heads")
    group_size = num_q_heads // num_kv_heads
    if group_size != 1:
        key_f = key_f.repeat_interleave(group_size, dim=1)
        value_f = value_f.repeat_interleave(group_size, dim=1)
    block_q, block_kv = int(block_shape[0]), int(block_shape[1])
    nq_blocks = ceil_div(query_seq_len, block_q)
    nkv_blocks = ceil_div(kv_seq_len, block_kv)
    output = torch.zeros(batch, num_q_heads, query_seq_len, head_dim, dtype=torch.float32, device=device)
    for q_block in range(nq_blocks):
        q_start = q_block * block_q
        q_end = min(q_start + block_q, query_seq_len)
        q_tile = query_f[:, :, q_start:q_end, :]
        block_row = block_sparse_mask[:, :, q_block, :nkv_blocks]
        token_keep = block_row.repeat_interleave(block_kv, dim=-1)[:, :, :kv_seq_len].bool()
        scores = torch.matmul(q_tile, key_f.transpose(-1, -2)) * scale_value
        scores = scores.masked_fill(~token_keep.unsqueeze(2), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        output[:, :, q_start:q_end, :] = torch.matmul(probs, value_f)
    return output.to(out_dtype)


def cpu_block_sparse_attention_golden(query, key, value, block_sparse_mask, block_shape, scale_value, out_dtype=None):
    """Small-shape golden: same compose on CPU in FP32, last cast matches DUT dtype."""
    return compose_block_sparse_attention(
        query.detach().cpu(),
        key.detach().cpu(),
        value.detach().cpu(),
        block_sparse_mask.detach().cpu(),
        block_shape,
        scale_value,
        out_dtype=query.dtype if out_dtype is None else out_dtype,
    )


def npu_block_sparse_attention_golden(query, key, value, block_sparse_mask, block_shape, scale_value, out_dtype=None):
    """Large-shape golden: compose on the same NPU device as the DUT. Never CPU."""
    return compose_block_sparse_attention(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        scale_value,
        out_dtype=query.dtype if out_dtype is None else out_dtype,
    )


def build_bsa_inputs(
    batch,
    num_q_heads,
    num_kv_heads,
    query_seq_len,
    kv_seq_len,
    head_dim,
    dtype,
    generator,
    block_shape=BLOCK_SHAPE,
):
    query = generate_opbase_tensor((batch, num_q_heads, query_seq_len, head_dim), dtype, generator)
    key = generate_opbase_tensor((batch, num_kv_heads, kv_seq_len, head_dim), dtype, generator)
    value = generate_opbase_tensor((batch, num_kv_heads, kv_seq_len, head_dim), dtype, generator)
    mask = make_block_sparse_mask(batch, num_q_heads, query_seq_len, kv_seq_len, block_shape, generator)
    return query, key, value, mask


def call_block_sparse_attention(query, key, value, block_sparse_mask, block_shape, scale_value):
    """CANN-bridged torch.ops.mindiesd.block_sparse_attention. No dequant scales."""
    from mindiesd.layers._custom_ops import block_sparse_attention

    batch = query.shape[0]
    query_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]
    num_kv_heads = key.shape[1]
    return block_sparse_attention(
        query=query,
        key=key,
        value=value,
        block_sparse_mask=block_sparse_mask,
        block_shape=list(block_shape),
        q_input_layout=LAYOUT,
        kv_input_layout=LAYOUT,
        num_key_value_heads=num_kv_heads,
        scale_value=scale_value,
        inner_precise=INNER_PRECISE,
        actual_seq_lengths=[query_seq_len] * batch,
        actual_seq_lengths_kv=[kv_seq_len] * batch,
        softmax_lse_flag=SOFTMAX_LSE_FLAG,
    )


def call_eagle_block_sparse_attention(query, key, value, block_sparse_mask, block_shape, scale_value):
    from mindiesd.layers._custom_ops import eagle_block_sparse_attention

    batch = query.shape[0]
    query_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]
    num_kv_heads = key.shape[1]
    return eagle_block_sparse_attention(
        query=query,
        key=key,
        value=value,
        block_sparse_mask=block_sparse_mask,
        block_shape=list(block_shape),
        q_input_layout=LAYOUT,
        kv_input_layout=LAYOUT,
        num_key_value_heads=num_kv_heads,
        scale_value=scale_value,
        inner_precise=INNER_PRECISE,
        actual_seq_lengths=[query_seq_len] * batch,
        actual_seq_lengths_kv=[kv_seq_len] * batch,
        softmax_lse_flag=SOFTMAX_LSE_FLAG,
    )


def cosine_metrics(reference, actual, chunk_elements=1_048_576):
    """Chunked FP64 cosine / max_abs / norm_ratio."""
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}")
    reference_flat = reference.detach().cpu().reshape(-1)
    actual_flat = actual.detach().cpu().reshape(-1)
    dot = reference_squared = actual_squared = 0.0
    max_abs_error = 0.0
    for start in range(0, reference_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, reference_flat.numel())
        reference_chunk = reference_flat[start:end].double()
        actual_chunk = actual_flat[start:end].double()
        dot += torch.dot(reference_chunk, actual_chunk).item()
        reference_squared += torch.dot(reference_chunk, reference_chunk).item()
        actual_squared += torch.dot(actual_chunk, actual_chunk).item()
        max_abs_error = max(max_abs_error, (actual_chunk - reference_chunk).abs().max().item())
    denominator = math.sqrt(reference_squared * actual_squared)
    if not math.isfinite(dot) or not math.isfinite(denominator) or denominator == 0.0:
        raise ValueError("cosine inputs produced a non-finite or zero norm")
    cosine = max(-1.0, min(1.0, dot / denominator))
    return {
        "cosine": cosine,
        "max_abs_error": max_abs_error,
        "norm_ratio": math.sqrt(actual_squared / reference_squared),
    }


def _mixed_tolerance_dtype_key(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    text = str(dtype).lower().replace(" ", "").replace("_", "").replace("torch.", "")
    if text in _MIXED_TOLERANCE:
        return text
    raise ValueError(f"unsupported mixed-tolerance dtype: {dtype}")


def check_mixed_tolerance(actual, golden, dtype):
    """opbase mixed tolerance: matched_ratio and max_abs_error.

    Element pass: |actual - golden| <= atol + rtol * |golden|.
    Case pass: matched_ratio >= 0.99 and max_abs_error <= max(fixed_limit, 32 * ULP@1).
    dtype selects the compute-dtype table row.
    """
    key = _mixed_tolerance_dtype_key(dtype)
    rtol, atol, required_ratio, fixed_limit = _MIXED_TOLERANCE[key]
    max_abs_limit = max(fixed_limit, _ULP_FACTOR * _ULP_AT_ONE[key])
    if actual.shape != golden.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(golden.shape)}")
    actual_flat = actual.detach().cpu().float().reshape(-1)
    golden_flat = golden.detach().cpu().float().reshape(-1)
    if actual_flat.numel() == 0:
        return {
            "result": "Pass",
            "dtype_key": key,
            "matched_ratio": 1.0,
            "max_abs_error": 0.0,
            "rtol": rtol,
            "atol": atol,
            "required_matched_ratio": required_ratio,
            "max_abs_error_limit": max_abs_limit,
        }
    abs_error = (actual_flat - golden_flat).abs()
    threshold = atol + rtol * golden_flat.abs()
    matched_ratio = float((abs_error <= threshold).sum().item()) / float(actual_flat.numel())
    max_abs_error = float(abs_error.max().item())
    passed = matched_ratio >= required_ratio and max_abs_error <= max_abs_limit
    return {
        "result": "Pass" if passed else "Failed",
        "dtype_key": key,
        "matched_ratio": matched_ratio,
        "max_abs_error": max_abs_error,
        "rtol": rtol,
        "atol": atol,
        "required_matched_ratio": required_ratio,
        "max_abs_error_limit": max_abs_limit,
    }


def print_accuracy_report(golden_name, dut_name, metrics, gate):
    print(f"golden={golden_name}  dut={dut_name}")
    print(
        f"cosine={metrics['cosine']:.8f}  "
        f"max_abs_error={metrics['max_abs_error']:.6f}  "
        f"norm_ratio={metrics['norm_ratio']:.6f}"
    )
    print(
        f"mixed_tolerance={gate['result']}  dtype={gate['dtype_key']}  "
        f"matched_ratio={gate['matched_ratio'] * 100.0:.4f}%  "
        f"max_abs={gate['max_abs_error']:.6f}  "
        f"rtol={gate['rtol']} atol={gate['atol']}  "
        f"required_ratio={gate['required_matched_ratio']}  "
        f"max_abs_limit={gate['max_abs_error_limit']}"
    )


def collect_gate_failures(metrics, gate):
    failures = []
    if gate["result"] != "Pass":
        failures.append(
            f"mixed tolerance {gate['result']}: dtype={gate['dtype_key']} "
            f"matched_ratio={gate['matched_ratio'] * 100.0:.4f}% "
            f"max_abs={gate['max_abs_error']:.6f} limit={gate['max_abs_error_limit']} "
            f"rtol={gate['rtol']} atol={gate['atol']}"
        )
    if metrics["cosine"] < COSINE_MIN:
        failures.append(
            f"cosine {metrics['cosine']:.8f} < {COSINE_MIN}  "
            f"max_abs={metrics['max_abs_error']:.6f}  "
            f"norm_ratio={metrics['norm_ratio']:.6f}"
        )
    if not (NORM_RATIO_LO <= metrics["norm_ratio"] <= NORM_RATIO_HI):
        failures.append(
            f"norm_ratio {metrics['norm_ratio']:.6f} is outside "
            f"[{NORM_RATIO_LO}, {NORM_RATIO_HI}]  "
            f"cosine={metrics['cosine']:.8f}  "
            f"max_abs={metrics['max_abs_error']:.6f}"
        )
    return failures
