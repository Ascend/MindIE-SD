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

"""Single-op FIA profile case for DiT-Prof.xlsx sheet 0825-eaglefia-tiling512 row 34.

QKV are FP8 E4M3FN per-block (query/key/value_quant_mode=7). Scale S-dim is
ceil(seq/block) with Q block 128 and KV block 256. Default shapes:

  Q:   [1, 32, 2304, 128]
  K/V: [1,  4, 30757, 128]
  out: [1, 32, 2304, 128] bfloat16

Excel Duration baseline: 2314.363 us (OpBasicInfo / aicore_time).
"""

from __future__ import annotations

import argparse
import math
import os

import torch

SCENARIO_NAME = "DiT_0825_eaglefia_tiling512_row34"
EXCEL_DURATION_US = 2314.363
EXCEL_CUBE_UTIL = 78.746
DEFAULT_BATCH = 1
DEFAULT_NUM_Q_HEADS = 32
DEFAULT_NUM_KV_HEADS = 4
DEFAULT_SEQ_Q = 2304
DEFAULT_SEQ_KV = 30757
DEFAULT_HEAD_DIM = 128
Q_BLOCK = 128
K_BLOCK = 256
V_BLOCK = 256
QUANT_MODE = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile MindIE-SD fused_infer_attention_score_v2 (DiT eaglefia tiling512 row34)."
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Physical NPU ID from npu-smi info. Do not use ASCEND_RT_VISIBLE_DEVICES.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--num-query-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=DEFAULT_SEQ_Q)
    parser.add_argument("--kv-seq-len", type=int, default=DEFAULT_SEQ_KV)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--msprof-mode",
        action="store_true",
        help="Run a single forward for msprof op capture.",
    )
    return parser.parse_args()


def _validate_args(args):
    if args.device_id < 0:
        raise ValueError("--device-id must be greater than or equal to 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_query_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("head counts must be greater than 0")
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("--num-query-heads must be divisible by --num-kv-heads")
    if args.query_seq_len <= 0 or args.kv_seq_len <= 0:
        raise ValueError("sequence lengths must be greater than 0")
    if args.head_dim not in (64, 128):
        raise ValueError("--head-dim must be 64 or 128")


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _scale_shape(batch, heads, seq_len, block_size):
    return (batch, heads, _ceil_div(seq_len, block_size), 1)


def build_inputs(args, device):
    import torch_npu
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

    query = torch.randn(
        (args.batch_size, args.num_query_heads, args.query_seq_len, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn(
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    q, q_scale = fa_block_quant_preprocess(
        query, block_size=Q_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
    )
    k, k_scale = fa_block_quant_preprocess(
        key, block_size=K_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
    )
    v, v_scale = fa_block_quant_preprocess(
        value, block_size=V_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
    )
    return q, k, v, q_scale, k_scale, v_scale


def run_fia(q, k, v, q_scale, k_scale, v_scale, args):
    from mindiesd.layers.flash_attn.fused_infer_attention_score import (
        fused_infer_attention_score_v2,
    )

    return fused_infer_attention_score_v2(
        q,
        k,
        v,
        num_query_heads=args.num_query_heads,
        num_key_value_heads=args.num_kv_heads,
        softmax_scale=1.0 / math.sqrt(args.head_dim),
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BNSD",
        query_quant_mode=QUANT_MODE,
        key_quant_mode=QUANT_MODE,
        value_quant_mode=QUANT_MODE,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=torch.bfloat16,
    )


def check_shapes(q, k, v, q_scale, k_scale, v_scale, out, args):
    expected = [
        ("Q", tuple(q.shape), (args.batch_size, args.num_query_heads, args.query_seq_len, args.head_dim)),
        ("K", tuple(k.shape), (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim)),
        ("V", tuple(v.shape), (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim)),
        (
            "dequant_scale_query",
            tuple(q_scale.shape),
            _scale_shape(args.batch_size, args.num_query_heads, args.query_seq_len, Q_BLOCK),
        ),
        (
            "dequant_scale_key",
            tuple(k_scale.shape),
            _scale_shape(args.batch_size, args.num_kv_heads, args.kv_seq_len, K_BLOCK),
        ),
        (
            "dequant_scale_value",
            tuple(v_scale.shape),
            _scale_shape(args.batch_size, args.num_kv_heads, args.kv_seq_len, V_BLOCK),
        ),
        (
            "attention_out",
            tuple(out.shape),
            (args.batch_size, args.num_query_heads, args.query_seq_len, args.head_dim),
        ),
    ]
    print("=== shape/dtype check (Excel row34, per-block 7/7/7) ===")
    print(f"  q_block={Q_BLOCK} k_block={K_BLOCK} v_block={V_BLOCK}")
    ok = True
    for name, shape, exp in expected:
        match = shape == exp
        ok = ok and match
        flag = "OK" if match else "MISMATCH"
        print(f"  [{flag}] {name}: shape={shape} expected={exp}")
    if "float8_e4m3fn" not in str(q.dtype):
        ok = False
        print(f"  [MISMATCH] Q dtype should be float8_e4m3fn, got {q.dtype}")
    if out.dtype != torch.bfloat16:
        ok = False
        print(f"  [MISMATCH] attention_out.dtype={out.dtype} expected=torch.bfloat16")
    if not ok:
        raise RuntimeError("shape/dtype 与 per-block 粒度推导不一致")
    print("  all checks passed")


def main():
    args = parse_args()
    _validate_args(args)

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("ERROR: torch_npu missing — run inside CANN + torch_npu") from exc

    if not torch_npu.npu.is_available():
        raise SystemExit("ERROR: NPU is not available")

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        raise SystemExit(
            "ERROR: ASCEND_RT_VISIBLE_DEVICES is set "
            f"({visible!r}). Unset it and pass --device-id with the npu-smi NPU ID. "
            "msprof / msprof op cannot use this env var."
        )

    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device_id)
    device = f"npu:{args.device_id}"

    import mindiesd

    print(f"scenario={SCENARIO_NAME}")
    print(f"msprof_mode={args.msprof_mode}")
    print(f"excel_duration_us={EXCEL_DURATION_US}")
    print(f"excel_cube_utilization={EXCEL_CUBE_UTIL}")
    print(f"device_id={args.device_id}")
    print(f"NPU={torch.npu.get_device_properties(args.device_id).name}")
    print(f"mindiesd={mindiesd.__file__}")
    print(
        f"BNSD Q=({args.batch_size},{args.num_query_heads},{args.query_seq_len},{args.head_dim}) "
        f"KV=({args.batch_size},{args.num_kv_heads},{args.kv_seq_len},{args.head_dim}) "
        f"quant_mode={QUANT_MODE}/{QUANT_MODE}/{QUANT_MODE} "
        f"q_block={Q_BLOCK} k_block={K_BLOCK} v_block={V_BLOCK} out=bf16"
    )

    q, k, v, q_scale, k_scale, v_scale = build_inputs(args, device)
    torch_npu.npu.synchronize()
    with torch.inference_mode():
        out, lse = run_fia(q, k, v, q_scale, k_scale, v_scale, args)
        torch_npu.npu.synchronize()

    if not torch.isfinite(out.float()).all().item():
        raise RuntimeError("attention_out contains NaN or Inf")
    if lse is not None and lse.numel() != 0:
        raise RuntimeError("mode-17 path should return an empty softmax_lse")

    check_shapes(q, k, v, q_scale, k_scale, v_scale, out, args)

    print("attention_out.shape:", tuple(out.shape))
    print("attention_out.dtype:", out.dtype)
    print("softmax_lse.numel:", 0 if lse is None else lse.numel())
    print(
        "RESULT msprof-mode forward done "
        f"(align Excel {EXCEL_DURATION_US} us via msprof op Duration, not wall-clock)"
    )
    if os.environ.get("FIA_BENCH_TAG"):
        print(f"tag={os.environ['FIA_BENCH_TAG']}")


if __name__ == "__main__":
    main()
