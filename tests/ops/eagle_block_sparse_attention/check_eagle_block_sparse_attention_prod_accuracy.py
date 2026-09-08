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

"""Large-shape BSA accuracy: DUT vs NPU matmul+softmax+matmul compose. Never CPU."""

from __future__ import annotations

import argparse
import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from eagle_block_sparse_attention_accuracy_common import (  # noqa: E402
    BLOCK_SHAPE,
    BLOCK_SHAPE_K64,
    PROD_BATCH,
    PROD_HEAD_DIM,
    PROD_NUM_KV_HEADS,
    PROD_NUM_Q_HEADS,
    PROD_SEQ,
    SEED,
    build_bsa_inputs,
    call_eagle_block_sparse_attention,
    check_mixed_tolerance,
    collect_gate_failures,
    cosine_metrics,
    npu_block_sparse_attention_golden,
    print_accuracy_report,
    softmax_scale,
)

SCENARIO_NAME = "eagle_bsa_prod_q1n32s32768_kv1n4s32768_d128"
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_BLOCK_KV_MAP = {
    "128": BLOCK_SHAPE,
    "64": BLOCK_SHAPE_K64,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare eagle_block_sparse_attention against NPU composed golden."
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Physical NPU ID from npu-smi info. Do not use ASCEND_RT_VISIBLE_DEVICES.",
    )
    parser.add_argument("--batch-size", type=int, default=PROD_BATCH)
    parser.add_argument("--num-query-heads", type=int, default=PROD_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=PROD_NUM_KV_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=PROD_SEQ)
    parser.add_argument("--kv-seq-len", type=int, default=PROD_SEQ)
    parser.add_argument("--head-dim", type=int, default=PROD_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "all"),
        default="all",
        help="Compute/storage dtype. Default all runs FP16 then BF16.",
    )
    parser.add_argument(
        "--block-kv",
        choices=("128", "64", "all"),
        default="all",
        help="block_shape[1] (BlockK). Same Q/KV tensor shape as the 128 case. Default all runs 128 then 64.",
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
        raise ValueError("eagle_block_sparse_attention supports --head-dim 64 or 128")


def _run_one(args, dtype, device, block_shape):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    scale = softmax_scale(args.head_dim)
    query, key, value, mask = build_bsa_inputs(
        args.batch_size,
        args.num_query_heads,
        args.num_kv_heads,
        args.query_seq_len,
        args.kv_seq_len,
        args.head_dim,
        dtype,
        generator,
        block_shape,
    )
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    mask = mask.to(device)
    print(f"scenario={SCENARIO_NAME}")
    print(f"device_id={args.device_id}")
    print(
        f"BNSD Q={tuple(query.shape)} KV={tuple(key.shape)} dtype={dtype} "
        f"block_shape={list(block_shape)} inner_precise=0"
    )

    with torch.inference_mode():
        reference = npu_block_sparse_attention_golden(query, key, value, mask, block_shape, scale)
        torch.npu.synchronize()
        dut_out, softmax_lse = call_eagle_block_sparse_attention(
            query, key, value, mask, block_shape, scale
        )
        torch.npu.synchronize()

    if not torch.isfinite(reference.float()).all().item():
        raise SystemExit("ERROR: NPU composed golden contains NaN or Inf")
    if not torch.isfinite(dut_out.float()).all().item():
        raise SystemExit("ERROR: eagle_block_sparse_attention output contains NaN or Inf")
    if tuple(dut_out.shape) != tuple(reference.shape):
        raise SystemExit(
            f"ERROR: shape mismatch DUT={tuple(dut_out.shape)} ref={tuple(reference.shape)}"
        )
    if dut_out.dtype != dtype or reference.dtype != dtype:
        raise SystemExit(
            f"ERROR: dtype mismatch DUT={dut_out.dtype} ref={reference.dtype} expected={dtype}"
        )

    metrics = cosine_metrics(reference, dut_out)
    gate = check_mixed_tolerance(dut_out, reference, dtype=dtype)
    print_accuracy_report(
        golden_name="npu_block_sparse_attention_golden (NPU matmul+softmax+matmul, tiled Q)",
        dut_name=f"eagle_block_sparse_attention (prod, {dtype}, block_shape={list(block_shape)})",
        metrics=metrics,
        gate=gate,
    )
    failures = collect_gate_failures(metrics, gate)
    if failures:
        raise SystemExit("ERROR: " + "; ".join(failures))
    print(f"RESULT accuracy check passed ({dtype}, block_shape={list(block_shape)})")


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

    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    npu = get_npu_device()
    if npu not in (NPUDevice.A2, NPUDevice.A3):
        raise SystemExit(f"ERROR: eagle_block_sparse_attention is a 910B/910_93 operator, got {npu}")

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    dtypes = tuple(_DTYPE_MAP.values()) if args.dtype == "all" else (_DTYPE_MAP[args.dtype],)
    block_shapes = (
        tuple(_BLOCK_KV_MAP.values()) if args.block_kv == "all" else (_BLOCK_KV_MAP[args.block_kv],)
    )
    for block_shape in block_shapes:
        for dtype in dtypes:
            _run_one(args, dtype, device, block_shape)


if __name__ == "__main__":
    main()
