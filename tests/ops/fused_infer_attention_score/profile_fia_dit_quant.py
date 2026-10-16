#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Single-forward FIA V-quant workload for ``msprof op``.

The shape and dtype match the large-shape accuracy case. Select exactly one
kernel path per process so profiling records cannot be mixed across paths.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from fia_accuracy_common import DEFAULT_ENHANCE_MODE, MAX_TOKENS, synthesize_bf16  # noqa: E402
from fia_quant_common import (  # noqa: E402
    D128_CHANNEL_BLOCK,
    FIA_QUANT_CASE_BY_NAME,
    K_TOKEN_BLOCK,
    Q_TOKEN_BLOCK,
)

SCENARIO_NAME = "DiT_0825_eaglefia_tiling512_row34_v_quant"
DEFAULT_BATCH = 1
DEFAULT_NUM_Q_HEADS = 32
DEFAULT_NUM_KV_HEADS = 4
DEFAULT_SEQ_Q = 2304
DEFAULT_SEQ_KV = 30757
DEFAULT_HEAD_DIM = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Profile one FIA FP8 V-quant path with msprof op.")
    parser.add_argument("--path", choices=FIA_QUANT_CASE_BY_NAME, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--num-query-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=DEFAULT_SEQ_Q)
    parser.add_argument("--kv-seq-len", type=int, default=DEFAULT_SEQ_KV)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--enhance-mode", type=float, default=DEFAULT_ENHANCE_MODE)
    parser.add_argument("--msprof-mode", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.device_id < 0:
        raise ValueError("--device-id must be greater than or equal to 0")
    if args.batch_size <= 0 or args.query_seq_len <= 0 or args.kv_seq_len <= 0:
        raise ValueError("batch size and sequence lengths must be greater than 0")
    if args.num_query_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("head counts must be greater than 0")
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("--num-query-heads must be divisible by --num-kv-heads")
    if args.head_dim != 128:
        raise ValueError("FIA V512 V-quant profiling supports --head-dim 128 only")
    if not math.isfinite(args.enhance_mode) or args.enhance_mode <= 0:
        raise ValueError("--enhance-mode must be finite and greater than 0")


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def expected_scale_shape(tensor, token_block, channel_block):
    return (
        tensor.shape[0],
        tensor.shape[1],
        ceil_div(tensor.shape[2], token_block),
        ceil_div(tensor.shape[3], channel_block),
    )


def quantize(tensor, token_block, channel_block, torch_npu):
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

    quantized, scale = fa_block_quant_preprocess(
        tensor,
        block_size=token_block,
        col_block_size=channel_block,
        dst_type=torch_npu.float8_e4m3fn,
        layout="BNSD",
    )
    expected = expected_scale_shape(tensor, token_block, channel_block)
    if tuple(scale.shape) != expected:
        raise RuntimeError(f"scale shape mismatch: actual={tuple(scale.shape)} expected={expected}")
    return quantized, scale


def main():
    args = parse_args()
    validate_args(args)
    case = FIA_QUANT_CASE_BY_NAME[args.path]

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("ERROR: torch_npu missing; run inside CANN + torch_npu") from exc

    if not torch_npu.npu.is_available():
        raise SystemExit("ERROR: NPU is not available")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        raise SystemExit(
            "ERROR: ASCEND_RT_VISIBLE_DEVICES is set "
            f"({visible!r}); unset it and select the physical card with --device-id"
        )

    from mindiesd.layers.flash_attn.fused_infer_attention_score import (
        fused_infer_attention_score_v2,
    )

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    query = synthesize_bf16(
        "query",
        (args.batch_size, args.num_query_heads, args.query_seq_len, args.head_dim),
        generator,
        args.enhance_mode,
    ).to(device)
    key = synthesize_bf16(
        "key",
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
        generator,
        args.enhance_mode,
    ).to(device)
    value = synthesize_bf16(
        "value",
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
        generator,
        args.enhance_mode,
    ).to(device)

    q, q_scale = quantize(query, Q_TOKEN_BLOCK, D128_CHANNEL_BLOCK, torch_npu)
    k, k_scale = quantize(key, K_TOKEN_BLOCK, D128_CHANNEL_BLOCK, torch_npu)
    v, v_scale = quantize(value, case.value_token_block, case.value_channel_block, torch_npu)
    torch_npu.npu.synchronize()

    print(f"scenario={SCENARIO_NAME}")
    print(f"path={case.name} description={case.description}")
    print(f"device_id={args.device_id} msprof_mode={args.msprof_mode}")
    print(
        f"BNSD Q={tuple(q.shape)} KV={tuple(k.shape)} qkv={case.quant_modes} "
        f"inner_precise={case.inner_precise} V_block="
        f"{case.value_token_block}x{case.value_channel_block} out=bf16"
    )

    with torch.inference_mode():
        output, softmax_lse = fused_infer_attention_score_v2(
            q,
            k,
            v,
            num_query_heads=args.num_query_heads,
            num_key_value_heads=args.num_kv_heads,
            softmax_scale=args.head_dim**-0.5,
            pre_tokens=MAX_TOKENS,
            next_tokens=MAX_TOKENS,
            input_layout="BNSD",
            query_quant_mode=case.quant_modes[0],
            key_quant_mode=case.quant_modes[1],
            value_quant_mode=case.quant_modes[2],
            inner_precise=case.inner_precise,
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            out_dtype=torch.bfloat16,
        )
        torch_npu.npu.synchronize()

    expected_output_shape = (
        args.batch_size,
        args.num_query_heads,
        args.query_seq_len,
        args.head_dim,
    )
    if tuple(output.shape) != expected_output_shape or output.dtype != torch.bfloat16:
        raise RuntimeError(
            f"output mismatch: shape={tuple(output.shape)} dtype={output.dtype}, "
            f"expected={expected_output_shape}/torch.bfloat16"
        )
    if not torch.isfinite(output.float()).all().item():
        raise RuntimeError("attention_out contains NaN or Inf")
    if softmax_lse is not None and softmax_lse.numel() != 0:
        raise RuntimeError("mode-17 path should return an empty softmax_lse")
    print("RESULT msprof-mode forward done; use msprof op Duration, not wall-clock")


if __name__ == "__main__":
    main()
