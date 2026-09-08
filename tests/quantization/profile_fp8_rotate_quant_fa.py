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

"""Single-forward FP8RotateQuantFA workload for ``msprof op``.

Shape and dtype match the FIA large-shape DiT eaglefia tiling512 row 34 case.
``--mode`` selects HIGH_PRECISION or C8V16_TILING512. One process runs one mode
so msprof records are not mixed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_FIA_TEST_DIR = os.path.abspath(os.path.join(_TEST_DIR, "..", "ops", "fused_infer_attention_score"))
if _FIA_TEST_DIR not in sys.path:
    sys.path.insert(0, _FIA_TEST_DIR)

from fia_accuracy_common import DEFAULT_ENHANCE_MODE, synthesize_bf16  # noqa: E402

from mindiesd.quantization.layer import FP8RotateQuantFA
from mindiesd.quantization.mode import FP8FAMode

SCENARIO_NAME = "DiT_0825_eaglefia_tiling512_row34_fp8_rotate_quant_fa"
DEFAULT_BATCH = 1
DEFAULT_NUM_HEADS = 32
DEFAULT_NUM_KV_HEADS = 4
DEFAULT_SEQ_Q = 2304
DEFAULT_SEQ_KV = 30757
DEFAULT_HEAD_DIM = 128
_MODE_CHOICES = tuple(item.value for item in FP8FAMode)


class _EmptyQuantWeights:
    def keys(self):
        return ()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile one FP8RotateQuantFA mode with msprof op.")
    parser.add_argument("--mode", choices=_MODE_CHOICES, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
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
    if args.num_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("head counts must be greater than 0")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")
    if args.head_dim != 128:
        raise ValueError("FP8RotateQuantFA C8V16_TILING512 profiling supports --head-dim 128 only")
    if not math.isfinite(args.enhance_mode) or args.enhance_mode <= 0:
        raise ValueError("--enhance-mode must be finite and greater than 0")


def main():
    args = parse_args()
    validate_args(args)

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

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    query = synthesize_bf16(
        "query",
        (args.batch_size, args.num_heads, args.query_seq_len, args.head_dim),
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

    fa_quant = FP8RotateQuantFA(prefix="attn", weights=_EmptyQuantWeights(), mode=args.mode)
    fa_quant.to(device)
    torch_npu.npu.synchronize()

    print(f"scenario={SCENARIO_NAME}")
    print(f"mode={fa_quant.mode.value}")
    print(f"device_id={args.device_id} msprof_mode={args.msprof_mode}")
    print(
        f"BNSD Q={tuple(query.shape)} KV={tuple(key.shape)} layout=BNSD "
        f"in=bf16 out=bf16 q_heads={args.num_heads} kv_heads={args.num_kv_heads}"
    )

    with torch.inference_mode():
        output = fa_quant(query, key, value, layout="BNSD")
        torch_npu.npu.synchronize()

    expected_output_shape = (
        args.batch_size,
        args.num_heads,
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
    print("RESULT msprof-mode forward done; use msprof op Duration, not wall-clock")


if __name__ == "__main__":
    main()
