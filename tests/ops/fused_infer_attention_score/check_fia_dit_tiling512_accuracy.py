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

"""Large-shape FIA accuracy: FP8 FIA vs BF16 torch_npu.npu_fusion_attention.

Scene matches profile_fia_dit_tiling512.py (DiT eaglefia tiling512 row 34):
  Q  [1, 32, 2304, 128]   K/V [1, 4, 30757, 128]   BNSD  quant 7/7/7
Inputs follow intranet data.pt snapshot scalars, enhance_mode default 2.0.
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

from fia_accuracy_common import (  # noqa: E402
    DEFAULT_ENHANCE_MODE,
    K_BLOCK,
    MAX_TOKENS,
    Q_BLOCK,
    QUANT_MODE,
    V_BLOCK,
    cosine_metrics,
    synthesize_bf16,
)

SCENARIO_NAME = "DiT_0825_eaglefia_tiling512_row34_accuracy"
DEFAULT_BATCH = 1
DEFAULT_NUM_Q_HEADS = 32
DEFAULT_NUM_KV_HEADS = 4
DEFAULT_SEQ_Q = 2304
DEFAULT_SEQ_KV = 30757
DEFAULT_HEAD_DIM = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare quantized FIA against unquantized npu_fusion_attention."
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
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--enhance-mode", type=float, default=DEFAULT_ENHANCE_MODE)
    parser.add_argument(
        "--cosine-min",
        type=float,
        default=0.99,
        help="Fail if cosine(FIA, FusionAttention) is below this value.",
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
    if not math.isfinite(args.enhance_mode) or args.enhance_mode <= 0:
        raise ValueError("--enhance-mode must be finite and greater than 0")


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
            f"({visible!r}). Unset it and pass --device-id with the npu-smi NPU ID."
        )

    from mindiesd.layers.flash_attn.fused_infer_attention_score import (
        fused_infer_attention_score_v2,
    )
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)

    query = synthesize_bf16(
        "query",
        (args.batch_size, args.num_query_heads, args.query_seq_len, args.head_dim),
        generator,
        enhance_mode=args.enhance_mode,
    ).to(device)
    key = synthesize_bf16(
        "key",
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
        generator,
        enhance_mode=args.enhance_mode,
    ).to(device)
    value = synthesize_bf16(
        "value",
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
        generator,
        enhance_mode=args.enhance_mode,
    ).to(device)

    print(f"scenario={SCENARIO_NAME}")
    print(f"device_id={args.device_id}")
    print(f"enhance_mode={args.enhance_mode}")
    print(
        f"BNSD Q={tuple(query.shape)} KV={tuple(key.shape)} "
        f"quant_mode={QUANT_MODE}/{QUANT_MODE}/{QUANT_MODE} out=bf16"
    )

    with torch.inference_mode():
        reference = torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            input_layout="BNSD",
            scale=softmax_scale,
            pre_tockens=MAX_TOKENS,
            next_tockens=MAX_TOKENS,
            head_num=args.num_query_heads,
        )[0]
        torch_npu.npu.synchronize()

        q, q_scale = fa_block_quant_preprocess(
            query, block_size=Q_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
        )
        k, k_scale = fa_block_quant_preprocess(
            key, block_size=K_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
        )
        v, v_scale = fa_block_quant_preprocess(
            value, block_size=V_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
        )
        torch_npu.npu.synchronize()

        fia_cases = (
            (None, "inner_precise=0 (default)"),
            (4, "inner_precise=4"),
        )
        for inner_precise, label in fia_cases:
            fia_kwargs = {
                "num_query_heads": args.num_query_heads,
                "num_key_value_heads": args.num_kv_heads,
                "softmax_scale": softmax_scale,
                "pre_tokens": MAX_TOKENS,
                "next_tokens": MAX_TOKENS,
                "input_layout": "BNSD",
                "query_quant_mode": QUANT_MODE,
                "key_quant_mode": QUANT_MODE,
                "value_quant_mode": QUANT_MODE,
                "dequant_scale_query": q_scale,
                "dequant_scale_key": k_scale,
                "dequant_scale_value": v_scale,
                "out_dtype": torch.bfloat16,
            }
            if inner_precise is not None:
                fia_kwargs["inner_precise"] = inner_precise
            fia_out, lse = fused_infer_attention_score_v2(q, k, v, **fia_kwargs)
            torch_npu.npu.synchronize()

            if not torch.isfinite(reference.float()).all().item():
                raise SystemExit("ERROR: npu_fusion_attention output contains NaN or Inf")
            if not torch.isfinite(fia_out.float()).all().item():
                raise SystemExit(f"ERROR: FIA output contains NaN or Inf ({label})")
            if lse is not None and lse.numel() != 0:
                raise SystemExit(f"ERROR: mode-17 path should return an empty softmax_lse ({label})")
            if tuple(fia_out.shape) != tuple(reference.shape):
                raise SystemExit(
                    f"ERROR: shape mismatch FIA={tuple(fia_out.shape)} ref={tuple(reference.shape)} ({label})"
                )

            metrics = cosine_metrics(reference, fia_out)
            print(
                f"golden=torch_npu.npu_fusion_attention (unquantized BF16)  "
                f"dut=fused_infer_attention_score_v2 (FP8 7/7/7 {label})"
            )
            print(
                f"cosine={metrics['cosine']:.8f}  "
                f"max_abs_error={metrics['max_abs_error']:.6f}  "
                f"norm_ratio={metrics['norm_ratio']:.6f}"
            )
            if metrics["cosine"] < args.cosine_min:
                raise SystemExit(
                    f"ERROR: cosine {metrics['cosine']:.8f} < --cosine-min {args.cosine_min} ({label})"
                )
            if not (0.9 <= metrics["norm_ratio"] <= 1.1):
                raise SystemExit(
                    f"ERROR: norm_ratio {metrics['norm_ratio']:.6f} is outside [0.9, 1.1] ({label})"
                )
            print(f"RESULT accuracy check passed ({label})")


if __name__ == "__main__":
    main()
