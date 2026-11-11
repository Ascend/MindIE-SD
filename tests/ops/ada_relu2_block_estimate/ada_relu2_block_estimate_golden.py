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

# 61200-token KV validation:
#   python tests/ops/ada_relu2_block_estimate/ada_relu2_block_estimate_golden.py \
#       --device-id 0 --batch 1 --query-heads 1 --key-heads 1 \
#       --query-seq-len 1024 --key-seq-len 61200 --head-dim 128 \
#       --stride 8 --sparse-size 128 --force-sparse 0.5
#
# CPU comparison of a saved cannsim result:
#   python tests/ops/ada_relu2_block_estimate/ada_relu2_block_estimate_golden.py \
#       --sim-result-dir /path/to/q128_kv1024_result \
#       --query-seq-len 128 --key-seq-len 1024 --head-dim 128

"""Validate BF16 (--qmode 0) or INT8 (--qmode 1) against the CPU reference.

INT8 uses 64-token block quantization. Saved INT8 simulation inputs are
query.int8.bin, key.int8.bin, q_scale.fp32.bin and k_scale.fp32.bin;
score.fp32.bin and row_sum.fp32.bin are shared by both modes.
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
import mindiesd


QUANT_BLOCK_SIZE = 64
DEVICE_ID = 0


@torch.no_grad()
def perblock_quant_int8(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match eagle's BNSD perblock_quant with block_size=64 and smooth=False."""
    batch, heads, seq_len, head_dim = input_tensor.shape
    padded = F.pad(input_tensor, (0, 0, 0, (-seq_len) % QUANT_BLOCK_SIZE))
    blocks = padded.reshape(batch, heads, math.ceil(seq_len / QUANT_BLOCK_SIZE), -1)
    quantized, scales = torch_npu.npu_dynamic_quant(blocks, dst_type=torch.int8)
    quantized = quantized.reshape(batch, heads, -1, head_dim)[:, :, :seq_len, :]
    return quantized.contiguous(), scales.float().contiguous()


def compressed_scales_cpu(scales: torch.Tensor, tensor: torch.Tensor, stride: int) -> torch.Tensor:
    """Map each 64-token quantization scale to its stride-compressed rows."""
    if scales is None:
        raise ValueError("INT8 input requires q_scale and k_scale")
    expected_shape = (*tensor.shape[:2], math.ceil(tensor.shape[2] / QUANT_BLOCK_SIZE))
    if tuple(scales.shape) != expected_shape:
        raise ValueError(f"scale shape must be {expected_shape}, got {tuple(scales.shape)}")
    if QUANT_BLOCK_SIZE % stride:
        raise ValueError("INT8 stride must divide the 64-token quantization block")
    rows = math.ceil(tensor.shape[2] / stride)
    return scales.float().cpu().repeat_interleave(QUANT_BLOCK_SIZE // stride, dim=-1)[..., :rows]


def squared_relu_intermediates_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    block_size: int,
    stride: int,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use FP16/BF16 or dequantized INT8 values, then apply ReLU(scaled QK + 1)^2."""
    is_int8 = query.dtype == torch.int8
    if is_int8 != (key.dtype == torch.int8):
        raise ValueError("query and key must both be INT8 or both be floating point")
    query_scales = compressed_scales_cpu(q_scale, query, stride) if is_int8 else None
    key_scales = compressed_scales_cpu(k_scale, key, stride) if is_int8 else None
    compute_dtype = torch.int32 if is_int8 else torch.float32
    query = query.to(device="cpu", dtype=compute_dtype)
    key = key.to(device="cpu", dtype=compute_dtype)
    batch_size, query_heads, query_seq_len, head_dim = query.shape
    key_heads = key.shape[1]
    key_seq_len = key.shape[2]
    if query_heads % key_heads != 0:
        raise ValueError("query_heads must be divisible by key_heads")

    gqa = query_heads // key_heads
    reduce_size = block_size // stride
    query_blocks = math.ceil(query_seq_len / block_size)
    key_blocks = math.ceil(key_seq_len / block_size)
    scale = 1.0 / (math.sqrt(head_dim) * stride)

    query_pad = (-query_seq_len) % stride
    query_strided = F.pad(query, (0, 0, 0, query_pad))
    query_strided = query_strided.view(batch_size, query_heads, -1, stride, head_dim)
    query_strided = query_strided.flip(dims=[3]).reshape(
        batch_size,
        query_heads,
        -1,
        stride * head_dim,
    )

    key_pad = (-key_seq_len) % stride
    key_strided = F.pad(key, (0, 0, 0, key_pad))
    key_strided = key_strided.view(batch_size, key_heads, -1, stride * head_dim)
    if gqa > 1:
        key_strided = key_strided.repeat_interleave(gqa, dim=1)
        if is_int8:
            key_scales = key_scales.repeat_interleave(gqa, dim=1)

    valid_key_rows = min((key_blocks - 1) * reduce_size, key_strided.shape[2])
    if valid_key_rows <= 0:
        return (
            torch.empty((batch_size, query_heads, query_blocks, 0, reduce_size), dtype=torch.float32),
            torch.zeros((batch_size, query_heads, query_blocks, reduce_size), dtype=torch.float32),
        )
    key_chunk = key_strided[:, :, :valid_key_rows, :]
    scored_key_blocks = valid_key_rows // reduce_size
    score_out = torch.zeros(
        (batch_size, query_heads, query_blocks, scored_key_blocks, reduce_size), dtype=torch.float32
    )
    row_sum_out = torch.zeros((batch_size, query_heads, query_blocks, reduce_size), dtype=torch.float32)

    compressed_rows_per_tile = 128
    query_tile_count = math.ceil(query_strided.shape[2] / compressed_rows_per_tile)
    for tile_index in range(query_tile_count):
        row_start = tile_index * compressed_rows_per_tile
        row_end = min(row_start + compressed_rows_per_tile, query_strided.shape[2])
        query_chunk = query_strided[:, :, row_start:row_end, :]
        scores = torch.matmul(query_chunk, key_chunk.transpose(-1, -2)).float()
        if is_int8:
            dequant_scale = query_scales[:, :, row_start:row_end, None] * key_scales[:, :, None, :valid_key_rows]
            scores = scores * dequant_scale
        scores = scores * scale
        scores = F.relu(scores + 1).square()
        scores = scores.view(batch_size, query_heads, row_end - row_start, -1, reduce_size).sum(dim=-1)
        query_row_pad = (-scores.shape[2]) % reduce_size
        if query_row_pad:
            scores = F.pad(scores, (0, 0, 0, query_row_pad))
        output_row_start = row_start // reduce_size
        score_blocks = scores.view(batch_size, query_heads, -1, reduce_size, scored_key_blocks)
        valid_output_rows = min(score_blocks.shape[2], query_blocks - output_row_start)
        score_out[:, :, output_row_start : output_row_start + valid_output_rows] = score_blocks[
            :, :, :valid_output_rows
        ].permute(0, 1, 2, 4, 3)
        row_sum_out[:, :, output_row_start : output_row_start + valid_output_rows] = score_blocks[
            :, :, :valid_output_rows
        ].sum(dim=-1)
    return score_out, row_sum_out


def sparse_mask_from_intermediates(
    score: torch.Tensor,
    row_sum: torch.Tensor,
    key_blocks: int,
    force_sparse: float,
) -> torch.Tensor:
    """Normalize and reduce the last dimension: 16 samples for stride=8, 8 for stride=16."""
    mask = torch.zeros((*score.shape[:3], key_blocks), dtype=torch.bool, device=score.device)
    if score.shape[-2] != 0:
        row_scale = row_sum.clamp_min(1e-9).reciprocal()
        row_scores = torch.matmul(score, row_scale.unsqueeze(-1)).squeeze(-1)
        keep_count = max(1, int(row_scores.shape[-1] * (1.0 - force_sparse)))
        keep_count = min(keep_count, row_scores.shape[-1])
        topk_indices = torch.topk(row_scores, k=keep_count, dim=-1, sorted=False).indices
        mask[..., : row_scores.shape[-1]].scatter_(dim=-1, index=topk_indices, value=True)
    if key_blocks != 0:
        mask[..., 0] = True
        mask[..., key_blocks - 1] = True
    return mask


def ada_relu2_block_estimate_full(
    query: torch.Tensor,
    key: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    stride: int,
    sparse_size: int,
    num_heads: int,
    num_key_value_heads: int,
    scale_value: float,
    force_sparse: float,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run block estimation and all following normalization/TopK/mask operators."""
    scale_kwargs = {} if q_scale is None and k_scale is None else {"q_scale": q_scale, "k_scale": k_scale}
    score, row_sum = torch.ops.mindiesd.ada_relu2_block_estimate(
        query=query,
        key=key,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        input_layout="BNSD",
        stride=stride,
        sparse_size=sparse_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale_value,
        threshold=force_sparse,
        **scale_kwargs,
    )
    score = score.cpu()
    row_sum = row_sum.cpu()
    print("POSTPROCESS_DEVICE=cpu")
    key_blocks = math.ceil(key.shape[2] / sparse_size)
    mask = sparse_mask_from_intermediates(score, row_sum, key_blocks, force_sparse)
    return mask, score, row_sum


def run_once(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    query = torch.randn(
        args.batch,
        args.query_heads,
        args.query_seq_len,
        args.head_dim,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        args.batch,
        args.key_heads,
        args.key_seq_len,
        args.head_dim,
        dtype=torch.bfloat16,
    )

    query_npu, key_npu = query.npu(), key.npu()
    q_scale = k_scale = q_scale_npu = k_scale_npu = None
    if args.qmode == 1:
        query_npu, q_scale_npu = perblock_quant_int8(query_npu)
        key_npu, k_scale_npu = perblock_quant_int8(key_npu)
        # Both references consume exactly the same quantized values and scales.
        query, key = query_npu.cpu(), key_npu.cpu()
        q_scale, k_scale = q_scale_npu.cpu(), k_scale_npu.cpu()
    print(f"QMODE={args.qmode} INPUT_DTYPE={query.dtype}")

    actual_mask, actual_score, actual_row_sum = ada_relu2_block_estimate_full(
        query=query_npu,
        key=key_npu,
        actual_seq_lengths=[args.query_seq_len] * args.batch,
        actual_seq_lengths_kv=[args.key_seq_len] * args.batch,
        stride=args.stride,
        sparse_size=args.sparse_size,
        num_heads=args.query_heads,
        num_key_value_heads=args.key_heads,
        scale_value=args.head_dim**-0.5 / args.stride,
        force_sparse=args.force_sparse,
        q_scale=q_scale_npu,
        k_scale=k_scale_npu,
    )
    torch.npu.synchronize()
    print("CALL=PASS")

    golden_score, golden_row_sum = squared_relu_intermediates_cpu(
        query,
        key,
        block_size=args.sparse_size,
        stride=args.stride,
        q_scale=q_scale,
        k_scale=k_scale,
    )
    golden_mask = sparse_mask_from_intermediates(
        golden_score,
        golden_row_sum,
        math.ceil(args.key_seq_len / args.sparse_size),
        args.force_sparse,
    )

    check_intermediates(actual_score.cpu(), actual_row_sum.cpu(), golden_score, golden_row_sum)
    check_mask(actual_mask.cpu(), golden_mask)


def check_mask(actual_mask: torch.Tensor, golden_mask: torch.Tensor) -> None:
    matched = actual_mask == golden_mask
    match_rate = matched.float().mean().item() if matched.numel() != 0 else 1.0
    print(f"mask_shape={tuple(actual_mask.shape)}")
    print(f"mask_match_rate={match_rate:.8%}")
    if actual_mask.numel() != 0:
        mask_rows = actual_mask.reshape(-1, actual_mask.shape[-1])
        selected_per_row = mask_rows.sum(dim=-1)
        unique_row_count = torch.unique(mask_rows, dim=0).shape[0]
        print(f"mask_density={actual_mask.float().mean().item():.8%}")
        print(
            f"mask_selected_per_row_min={int(selected_per_row.min().item())} "
            f"mask_selected_per_row_max={int(selected_per_row.max().item())}"
        )
        print(f"mask_unique_row_count={unique_row_count}")
    if not torch.equal(actual_mask, golden_mask):
        mismatch_count = int((~matched).sum().item())
        raise AssertionError(f"mask mismatch: {mismatch_count}/{matched.numel()} elements differ")
    print("MASK_PRECISION=PASS")


def check_intermediates(
    actual_score: torch.Tensor,
    actual_row_sum: torch.Tensor,
    golden_score: torch.Tensor,
    golden_row_sum: torch.Tensor,
) -> None:
    score_diff = (actual_score - golden_score).abs()
    row_sum_diff = (actual_row_sum - golden_row_sum).abs()
    torch.testing.assert_close(actual_score, golden_score, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_row_sum, golden_row_sum, rtol=2e-3, atol=2e-3)
    print(f"score_shape={tuple(actual_score.shape)}")
    print(f"row_sum_shape={tuple(actual_row_sum.shape)}")
    print(f"score_max_abs_diff={(score_diff.max().item() if score_diff.numel() else 0.0):.8e}")
    print(f"row_sum_max_abs_diff={row_sum_diff.max().item():.8e}")
    print("PRECISION=PASS")


def run_saved_simulation(args: argparse.Namespace) -> None:
    result_dir = Path(args.sim_result_dir)
    input_dtype = torch.int8 if args.qmode == 1 else torch.bfloat16
    input_suffix = "int8" if args.qmode == 1 else "bf16"
    query = torch.frombuffer((result_dir / f"query.{input_suffix}.bin").read_bytes(), dtype=input_dtype).clone()
    key = torch.frombuffer((result_dir / f"key.{input_suffix}.bin").read_bytes(), dtype=input_dtype).clone()
    query = query.view(args.batch, args.query_heads, args.query_seq_len, args.head_dim)
    key = key.view(args.batch, args.key_heads, args.key_seq_len, args.head_dim)
    q_scale = k_scale = None
    if args.qmode == 1:
        q_scale = torch.frombuffer((result_dir / "q_scale.fp32.bin").read_bytes(), dtype=torch.float32).clone()
        k_scale = torch.frombuffer((result_dir / "k_scale.fp32.bin").read_bytes(), dtype=torch.float32).clone()
        q_scale = q_scale.view(args.batch, args.query_heads, math.ceil(args.query_seq_len / QUANT_BLOCK_SIZE))
        k_scale = k_scale.view(args.batch, args.key_heads, math.ceil(args.key_seq_len / QUANT_BLOCK_SIZE))
    query_blocks = math.ceil(args.query_seq_len / args.sparse_size)
    scored_key_blocks = max(math.ceil(args.key_seq_len / args.sparse_size) - 1, 0)
    reduce_size = args.sparse_size // args.stride
    score_bytes = (result_dir / "score.fp32.bin").read_bytes()
    actual_score = (
        torch.frombuffer(score_bytes, dtype=torch.float32).clone()
        if score_bytes
        else torch.empty(0, dtype=torch.float32)
    )
    actual_row_sum = torch.frombuffer((result_dir / "row_sum.fp32.bin").read_bytes(), dtype=torch.float32).clone()
    actual_score = actual_score.view(args.batch, args.query_heads, query_blocks, scored_key_blocks, reduce_size)
    actual_row_sum = actual_row_sum.view(args.batch, args.query_heads, query_blocks, reduce_size)
    golden_score, golden_row_sum = squared_relu_intermediates_cpu(
        query, key, args.sparse_size, args.stride, q_scale, k_scale
    )
    actual_mask = sparse_mask_from_intermediates(
        actual_score, actual_row_sum, math.ceil(args.key_seq_len / args.sparse_size), args.force_sparse
    )
    golden_mask = sparse_mask_from_intermediates(
        golden_score, golden_row_sum, math.ceil(args.key_seq_len / args.sparse_size), args.force_sparse
    )
    check_mask(actual_mask, golden_mask)
    check_intermediates(actual_score, actual_row_sum, golden_score, golden_row_sum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=1)
    parser.add_argument("--query-seq-len", type=int, default=61200)
    parser.add_argument("--key-seq-len", type=int, default=61200)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--qmode",
        type=int,
        choices=(0, 1),
        default=0,
        help="0: BF16 inputs; 1: block-quantized INT8 inputs for both CPU and NPU",
    )
    parser.add_argument("--stride", type=int, choices=(8, 16), default=8)
    parser.add_argument("--sparse-size", type=int, default=128)
    parser.add_argument("--force-sparse", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-result-dir")
    parser.add_argument("--skip-set-device", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.force_sparse <= 1.0:
        raise ValueError("force_sparse must be in [0, 1]")
    if args.sparse_size % args.stride != 0:
        raise ValueError("sparse_size must be divisible by stride")
    if args.query_heads % args.key_heads != 0:
        raise ValueError("query_heads must be divisible by key_heads")
    if args.sim_result_dir:
        run_saved_simulation(args)
        return
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for live NPU validation")

    if not args.skip_set_device:
        torch_npu.npu.set_device(args.device_id)
    run_once(args)


if __name__ == "__main__":
    main()
