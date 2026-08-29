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

"""Shared FIA accuracy helpers: snapshot synthesizer, C8V16 CPU golden, metrics.

Snapshot scalars come from dump_fia_snapshot_features.py on data.pt
(query_bf16 / key_bf16 / value_bf16). Generation is N(mean, std), clamp to
p0.01/p99.99, then multiply by enhance_mode (default 2.0).
"""

from __future__ import annotations

import math

import torch

DEFAULT_ENHANCE_MODE = 2.0
MAX_TOKENS = 2_147_483_647
Q_BLOCK = 128
K_BLOCK = 256
V_BLOCK = 256
COL_BLOCK = 128
QUANT_MODE = 7
P_SCALE_C8V16 = 448.0
S2_BASE_C8V16 = 256
FP16_SOFTMAX_MIN = -65504.0
FP16_TO_FP8_MIN_BITS = 8256
FP16_TO_FP8_BIAS_OFFSET = 8128
FP16_TO_FP8_MANTISSA_SHIFT = 7

# Compact stats from intranet data.pt (2026-08-27 dump).
SNAPSHOT_FEATURES = {
    "query": {
        "mean": -0.011658942126734304,
        "std": 0.7169337224409994,
        "clamp_low": -10.5625,
        "clamp_high": 6.8125,
    },
    "key": {
        "mean": 0.0026953150009640468,
        "std": 0.6181585301036071,
        "clamp_low": -6.28125,
        "clamp_high": 5.9375,
    },
    "value": {
        "mean": 0.00017389531159397624,
        "std": 0.12759240340568168,
        "clamp_low": -0.73828125,
        "clamp_high": 0.68359375,
    },
}


def synthesize_bf16(role, shape, generator, enhance_mode=DEFAULT_ENHANCE_MODE):
    """Build a BF16 BNSD tensor from snapshot mean/std/clamp, then scale."""
    feat = SNAPSHOT_FEATURES[role]
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
    tensor = tensor * feat["std"] + feat["mean"]
    tensor = tensor.clamp(feat["clamp_low"], feat["clamp_high"])
    if enhance_mode != 1.0:
        tensor = tensor * enhance_mode
    return tensor.to(torch.bfloat16)


def dequant_fp8_per_block(fp8_tensor, scale, row_block_size, col_block_size=COL_BLOCK):
    expanded = scale.float().repeat_interleave(row_block_size, dim=2)
    expanded = expanded[:, :, : fp8_tensor.shape[2], :]
    expanded = expanded.repeat_interleave(col_block_size, dim=3)
    expanded = expanded[:, :, :, : fp8_tensor.shape[3]]
    return fp8_tensor.float() * expanded


def _expand_row_scale(scale, row_block_size, seq_len):
    expanded = scale.float().repeat_interleave(row_block_size, dim=2)
    return expanded[:, :, :seq_len, :]


def _fp16_to_fp8_e4m3_rna(values_fp16):
    """C8V16 P quant: (max(I, 8256) - 8128) >> 7, low 8 bits are e4m3.

    Same integer ALU path as vf_mul_sel_softmaxflashv2_cast_nz_dn.h
    (Maxs 8256, Adds -8128, ShiftRights 7). Input must already be *448.
    """
    fp16 = values_fp16.to(torch.float16).contiguous()
    bits = fp16.view(torch.int16).to(torch.int32)
    bits = torch.clamp(bits, min=FP16_TO_FP8_MIN_BITS)
    bits = (bits - FP16_TO_FP8_BIAS_OFFSET) >> FP16_TO_FP8_MANTISSA_SHIFT
    packed = bits.to(torch.uint8).reshape(fp16.shape)
    fp8 = torch.empty(fp16.shape, dtype=torch.float8_e4m3fn)
    fp8.view(torch.uint8).copy_(packed)
    return fp8.float()


def _fp16_pairwise_sum(values_fp16):
    """Reduce last dim with fp16 pairwise adds, then cast to fp32.

    Kernel ProcessVec1DnFp16SoftmaxVF accumulates exp in half, then
    Cast<float, half> before LastDiv multiplies the compact sum by pScale.
    """
    reduced = values_fp16.to(torch.float16)
    while reduced.shape[-1] > 1:
        length = reduced.shape[-1]
        if length % 2 == 1:
            reduced = torch.cat(
                [
                    reduced,
                    torch.zeros(
                        *reduced.shape[:-1],
                        1,
                        dtype=torch.float16,
                        device=reduced.device,
                    ),
                ],
                dim=-1,
            )
        reduced = reduced[..., 0::2] + reduced[..., 1::2]
    return reduced.float()


def cpu_c8v16_fp8_fia_golden(
    query_fp8,
    key_fp8,
    value_fp8,
    query_scale,
    key_scale,
    value_scale,
    softmax_scale,
    out_dtype=torch.bfloat16,
    s2_tile=S2_BASE_C8V16,
    p_scale=P_SCALE_C8V16,
    q_row_block=Q_BLOCK,
    k_row_block=K_BLOCK,
    v_row_block=V_BLOCK,
    col_block=COL_BLOCK,
):
    """C8V16 FP8 FullQuant CPU golden (mode 7/7/7, D=128, S2 tile=256).

    Matches kernel numerical stages, not textbook FA:
    Cube FP8 Q@K^T, fixpipe * (softmax_scale * sQ * sK) to FP16,
    FP16 online softmax, P = RNA(exp * 448), FP8 P@V, LastDiv * sV / (sum * 448).
    """
    query = query_fp8.float().cpu()
    key = key_fp8.float().cpu()
    value = value_fp8.float().cpu()
    query_scale = query_scale.float().cpu()
    key_scale = key_scale.float().cpu()
    value_scale = value_scale.float().cpu()
    batch, num_query_heads, query_seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    kv_seq_len = key.shape[2]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    group_size = num_query_heads // num_kv_heads
    grouped_query = query.reshape(batch, num_kv_heads, group_size, query_seq_len, head_dim)
    grouped_key = key.unsqueeze(2)
    grouped_value = value.unsqueeze(2)
    query_scale = _expand_row_scale(query_scale, q_row_block, query_seq_len)
    key_scale = _expand_row_scale(key_scale, k_row_block, kv_seq_len)
    value_scale = _expand_row_scale(value_scale, v_row_block, kv_seq_len)
    query_scale = query_scale.reshape(batch, num_kv_heads, group_size, query_seq_len, -1)
    d_blocks = query_scale.shape[-1]
    score = None
    for d_block in range(d_blocks):
        start_d = d_block * col_block
        end_d = min(start_d + col_block, head_dim)
        block_score = torch.matmul(
            grouped_query[..., start_d:end_d],
            grouped_key[..., start_d:end_d].transpose(-1, -2),
        )
        key_scale_block = key_scale[..., d_block : d_block + 1].squeeze(-1)
        key_scale_block = key_scale_block[:, :, None, None, :]
        block_score = block_score * query_scale[..., d_block : d_block + 1] * key_scale_block
        score = block_score if score is None else score + block_score
    score = (score * softmax_scale).to(torch.float16)

    output = torch.zeros(
        batch, num_kv_heads, group_size, query_seq_len, head_dim, dtype=torch.float32
    )
    output_sum = torch.zeros(
        batch, num_kv_heads, group_size, query_seq_len, 1, dtype=torch.float32
    )
    output_max = torch.full(
        (batch, num_kv_heads, group_size, query_seq_len, 1),
        FP16_SOFTMAX_MIN,
        dtype=torch.float16,
    )
    for start_s2 in range(0, kv_seq_len, s2_tile):
        end_s2 = min(start_s2 + s2_tile, kv_seq_len)
        tile_score = score[..., start_s2:end_s2]
        tile_max = tile_score.amax(dim=-1, keepdim=True)
        if start_s2 == 0:
            new_max = tile_max
            alpha = torch.ones_like(tile_max)
        else:
            new_max = torch.maximum(tile_max, output_max)
            alpha = torch.exp((output_max - new_max).float()).to(torch.float16)
        prob = torch.exp((tile_score - new_max).float()).to(torch.float16)
        tile_sum = _fp16_pairwise_sum(prob)
        prob_fp8 = _fp16_to_fp8_e4m3_rna(prob.float() * p_scale)
        tile_value = grouped_value[:, :, :, start_s2:end_s2, :]
        tile_out = torch.zeros(
            batch, num_kv_heads, group_size, query_seq_len, head_dim, dtype=torch.float32
        )
        for d_block in range(d_blocks):
            start_d = d_block * col_block
            end_d = min(start_d + col_block, head_dim)
            value_scale_tile = value_scale[:, :, start_s2:end_s2, d_block : d_block + 1]
            tile_v = tile_value[..., start_d:end_d] * value_scale_tile[:, :, None, :, :]
            tile_out[..., start_d:end_d] = torch.matmul(prob_fp8, tile_v)
        alpha_f = alpha.float()
        if start_s2 == 0:
            output = tile_out
            output_sum = tile_sum
        else:
            output = output * alpha_f + tile_out
            output_sum = output_sum * alpha_f + tile_sum
        output_max = new_max
    output = output / (output_sum * p_scale)
    return output.reshape(batch, num_query_heads, query_seq_len, head_dim).to(out_dtype)


def cpu_four_stage_fia(
    query,
    key,
    value,
    softmax_scale,
    out_dtype=torch.bfloat16,
    query_chunk_size=128,
):
    """GQA attention in four stages: QK^T, scale, softmax, PV.

    Runs on CPU in FP32. query/key/value are BNSD float tensors.
    """
    query = query.float().cpu()
    key = key.float().cpu()
    value = value.float().cpu()
    batch, num_query_heads, query_seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    group_size = num_query_heads // num_kv_heads
    grouped_query = query.reshape(batch, num_kv_heads, group_size, query_seq_len, head_dim)
    grouped_key_t = key.unsqueeze(2).transpose(-1, -2)
    grouped_value = value.unsqueeze(2)
    chunks = []
    for start in range(0, query_seq_len, query_chunk_size):
        end = min(start + query_chunk_size, query_seq_len)
        score = torch.matmul(grouped_query[:, :, :, start:end, :], grouped_key_t)
        score = score * softmax_scale
        prob = torch.softmax(score, dim=-1)
        chunks.append(torch.matmul(prob, grouped_value))
    grouped_out = torch.cat(chunks, dim=3)
    return grouped_out.reshape(batch, num_query_heads, query_seq_len, head_dim).to(out_dtype)


def cosine_metrics(reference, actual, chunk_elements=1_048_576):
    """Chunked FP64 cosine / max_abs / norm_ratio (FIA super_test)."""
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


# experimental_standard mixed tolerance: (rtol, atol, required_matched_ratio, fixed_max_abs).
# https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
_MIXED_TOLERANCE = {
    "float16": (2**-9, 2**-9, 0.99, 1e-1),
    "bfloat16": (2**-6, 2**-6, 0.99, 1e-0),
    "float32": (2**-10, 2**-16, 0.99, 1e-2),
    "float8e4m3fn": (2**-2, 2**-4, 0.99, 1e-0),
    "float8e5m2": (2**-1, 2**-3, 0.99, 1e-1),
}
_ULP_AT_ONE = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-23,
    "float8e4m3fn": 2**-3,
    "float8e5m2": 2**-2,
}
_ULP_FACTOR = 32


def _mixed_tolerance_dtype_key(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float8_e4m3fn:
        return "float8e4m3fn"
    if dtype == torch.float8_e5m2:
        return "float8e5m2"
    text = str(dtype).lower().replace(" ", "").replace("_", "").replace("torch.", "")
    if text in _MIXED_TOLERANCE:
        return text
    if "float8e4m3" in text:
        return "float8e4m3fn"
    if "float8e5m2" in text:
        return "float8e5m2"
    raise ValueError(f"unsupported mixed-tolerance dtype: {dtype}")


def check_mixed_tolerance(actual, golden, dtype=torch.float8_e4m3fn):
    """opbase experimental mixed tolerance: matched_ratio and max_abs_error.

    Element pass: |actual - golden| <= atol + rtol * |golden|.
    Case pass: matched_ratio >= 0.99 and max_abs_error <= max(fixed_limit, 32 * ULP@1).
    dtype selects the table row; FIA FullQuant looks up FLOAT8 E4M3, not BF16 output.
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
