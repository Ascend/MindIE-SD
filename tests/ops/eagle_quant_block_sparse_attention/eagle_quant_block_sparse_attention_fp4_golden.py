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

import argparse
import math

import torch
import torch.nn.functional as F
import torch_npu

from mindiesd.layers.register_ops import _load_mindie_ops_library

from mxfp4_fa_npu_input_align import prepare_quantized_qkv, simulate_mxfp4_flash_attention

DEVICE_ID = 0
SPARSE_BLOCK_SIZE = 128
FP4_DTYPE = getattr(torch_npu, "float4_e2m1fn_x2", None)


def make_sparse_mask(batch, query_heads, q_seq_len, kv_seq_len, sparsity):
    q_blocks = math.ceil(q_seq_len / SPARSE_BLOCK_SIZE)
    kv_blocks = math.ceil(kv_seq_len / SPARSE_BLOCK_SIZE)
    mask = torch.rand(batch, query_heads, q_blocks, kv_blocks) > sparsity
    mask[..., 0] = True
    mask[..., -1] = True
    return mask


def compare_tensors(actual, golden, atol, rtol, max_bad_ratio):
    actual = actual.float().cpu()
    golden = golden.float().cpu()
    if not torch.isfinite(actual).all():
        raise RuntimeError("Operator output contains NaN or Inf.")
    if not torch.isfinite(golden).all():
        raise RuntimeError("Golden output contains NaN or Inf.")

    absolute_error = (actual - golden).abs()
    threshold = atol + rtol * golden.abs()
    bad_ratio = (absolute_error > threshold).float().mean().item()
    normalized_error = absolute_error / golden.abs().clamp_min(1.0)
    cosine = F.cosine_similarity(actual.flatten(), golden.flatten(), dim=0).item()
    metrics = {
        "max_abs": absolute_error.max().item(),
        "mean_abs": absolute_error.mean().item(),
        "eb": normalized_error.mean().item(),
        "cosine": cosine,
        "bad_ratio": bad_ratio,
    }
    return bad_ratio <= max_bad_ratio, metrics


def run_once(args):
    torch.manual_seed(args.seed)
    direction = torch.where(
        torch.randn(args.head_dim) >= 0,
        1.0,
        -1.0,
    ).to(torch.bfloat16)
    query = direction.view(1, 1, 1, -1) + 0.05 * torch.randn(
        args.batch,
        args.query_heads,
        args.q_seq_len,
        args.head_dim,
        dtype=torch.bfloat16,
    )
    key = 0.25 * torch.randn(
        args.batch,
        args.kv_heads,
        args.kv_seq_len,
        args.head_dim,
        dtype=torch.bfloat16,
    )
    num_kv_tiles = (args.kv_seq_len + 255) // 256
    candidate_tile_count = max(num_kv_tiles - 2, 0)
    num_updates = min(6, candidate_tile_count)
    update_tiles = (torch.randperm(candidate_tile_count)[:num_updates] + 2).sort().values.tolist()
    update_levels = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    updates = dict(zip(update_tiles, update_levels))

    level = 0.0
    for tile_idx, start in enumerate(range(0, args.kv_seq_len, 256)):
        end = min(start + 256, args.kv_seq_len)
        level = updates.get(tile_idx, level)
        key[:, :, start:end, :] += level * direction

    value = torch.randn(
        args.batch,
        args.kv_heads,
        args.kv_seq_len,
        args.head_dim,
        dtype=torch.bfloat16,
    )
    sparse_mask = make_sparse_mask(
        args.batch,
        args.query_heads,
        args.q_seq_len,
        args.kv_seq_len,
        args.sparsity,
    )

    (
        query_fp4,
        query_scale,
        key_fp4,
        key_scale,
        value_fp4,
        value_scale,
        value_level0_scale,
    ) = prepare_quantized_qkv(
        query.float(),
        key.float(),
        value.float(),
    )
    scale_value = args.head_dim**-0.5
    kv_seq_len_aligned64 = (args.kv_seq_len + 63) // 64 * 64
    value_scale_rows = kv_seq_len_aligned64 // 32
    value_scale_pad_rows = value_scale_rows - value_scale.shape[-2]

    value_scale_npu = (
        F.pad(
            value_scale,
            (0, 0, 0, value_scale_pad_rows),
            value=127,  # E8M0 scale=1
        )
        .reshape(
            args.batch,
            args.kv_heads,
            kv_seq_len_aligned64 // 64,
            2,
            args.head_dim,
        )
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )

    actual, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
        query=query_fp4.npu(),
        key=key_fp4.npu(),
        value=value_fp4.npu(),
        block_sparse_mask=sparse_mask.npu().view(torch.int8),
        block_shape=[SPARSE_BLOCK_SIZE, SPARSE_BLOCK_SIZE],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=args.kv_heads,
        scale_value=scale_value,
        inner_precise=args.inner_precise,
        actual_seq_lengths=[args.q_seq_len] * args.batch,
        actual_seq_lengths_kv=[args.kv_seq_len] * args.batch,
        softmax_lse_flag=0,
        value_scale=value_level0_scale.squeeze(-2).npu(),
        query_dtype_scale=query_scale.npu(),
        key_dtype_scale=key_scale.npu(),
        value_dtype_scale=value_scale_npu.npu(),
        query_dtype=FP4_DTYPE,
        key_dtype=FP4_DTYPE,
        value_dtype=FP4_DTYPE,
        output_dtype=torch.float16,
    )
    torch.npu.synchronize()

    golden = simulate_mxfp4_flash_attention(
        query_fp4,
        query_scale,
        key_fp4,
        key_scale,
        value_fp4,
        value_scale,
        value_level0_scale,
        block_sparse_mask=sparse_mask,
        always_update=False,
        useQMM=False,
    )

    passed, metrics = compare_tensors(
        actual,
        golden,
        args.atol,
        args.rtol,
        args.max_bad_ratio,
    )
    print(
        f"shape: B={args.batch}, Nq={args.query_heads}, Nkv={args.kv_heads}, "
        f"Sq={args.q_seq_len}, Skv={args.kv_seq_len}, D={args.head_dim}"
    )
    for name, value_metric in metrics.items():
        print(f"{name:>10}: {value_metric:.6e}")
    print(f"RESULT={'PASS' if passed else 'FAIL'}")
    return passed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-precise", type=int, choices=(4, 5), default=4, help="qmode2 softmax: 4=APPROX, 5=EXP")
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--q-seq-len", type=int, default=4068)
    parser.add_argument("--kv-seq-len", type=int, default=4068)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-bad-ratio", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.head_dim != 128:
        raise ValueError("qmode2 currently supports head_dim=128 only.")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads.")
    _load_mindie_ops_library()
    torch_npu.npu.set_device(args.device_id)
    passed = run_once(args)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
