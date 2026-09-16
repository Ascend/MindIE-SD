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

"""Large-shape EQBSA mix accuracy: DUT vs unquantized NPU compose. Never CPU."""

from __future__ import annotations

import argparse
import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from eagle_quant_block_sparse_attention_accuracy_common import (  # noqa: E402
    BLOCK_SHAPE,
    INNER_PRECISE,
    OUTPUT_DTYPE,
    PROD_BATCH,
    PROD_HEAD_DIM,
    PROD_KV_SEQ,
    PROD_NUM_KV_HEADS,
    PROD_NUM_Q_HEADS,
    PROD_QUERY_SEQ,
    SEED,
    build_eqbsa_prequant_inputs,
    call_eagle_quant_block_sparse_attention,
    collect_gate_failures,
    cosine_metrics,
    npu_eqbsa_unquant_golden,
    print_accuracy_report,
    quantize_eqbsa_qkv,
    softmax_scale,
)

SCENARIO_NAME = "eqbsa_mix_q1n32s2304_kv1n4s30757_d128"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare eagle_quant_block_sparse_attention against unquantized NPU composed golden."
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
    parser.add_argument("--query-seq-len", type=int, default=PROD_QUERY_SEQ)
    parser.add_argument("--kv-seq-len", type=int, default=PROD_KV_SEQ)
    parser.add_argument("--head-dim", type=int, default=PROD_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
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
        raise ValueError("eagle_quant_block_sparse_attention supports --head-dim 64 or 128")


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
    if npu != NPUDevice.A5:
        raise SystemExit(f"ERROR: eagle_quant_block_sparse_attention is an Ascend 950 operator, got {npu}")

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    scale = softmax_scale(args.head_dim)
    query, key, value, mask = build_eqbsa_prequant_inputs(
        args.batch_size,
        args.num_query_heads,
        args.num_kv_heads,
        args.query_seq_len,
        args.kv_seq_len,
        args.head_dim,
        generator,
        BLOCK_SHAPE,
    )
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    mask = mask.to(device)
    print(f"scenario={SCENARIO_NAME}")
    print(f"device_id={args.device_id}")
    print(
        f"BNSD Q={tuple(query.shape)} KV={tuple(key.shape)} pre_quant={query.dtype} "
        f"block_shape={list(BLOCK_SHAPE)} inner_precise={INNER_PRECISE}"
    )

    with torch.inference_mode():
        reference = npu_eqbsa_unquant_golden(query, key, value, mask, BLOCK_SHAPE, scale)
        torch.npu.synchronize()
        query_q, key_q, value_q, query_scale, key_scale, value_scale = quantize_eqbsa_qkv(query, key, value)
        torch.npu.synchronize()
        dut_out, softmax_lse = call_eagle_quant_block_sparse_attention(
            query_q, key_q, value_q, mask, query_scale, key_scale, value_scale, BLOCK_SHAPE, scale
        )
        torch.npu.synchronize()

    if not torch.isfinite(reference.float()).all().item():
        raise SystemExit("ERROR: NPU composed golden contains NaN or Inf")
    if not torch.isfinite(dut_out.float()).all().item():
        raise SystemExit("ERROR: eagle_quant_block_sparse_attention output contains NaN or Inf")
    if tuple(dut_out.shape) != tuple(reference.shape):
        raise SystemExit(
            f"ERROR: shape mismatch DUT={tuple(dut_out.shape)} ref={tuple(reference.shape)}"
        )
    if dut_out.dtype != OUTPUT_DTYPE or reference.dtype != OUTPUT_DTYPE:
        raise SystemExit(
            f"ERROR: dtype mismatch DUT={dut_out.dtype} ref={reference.dtype} expected={OUTPUT_DTYPE}"
        )
    if softmax_lse is not None and softmax_lse.numel() != 0:
        if not torch.isfinite(softmax_lse.float()).all().item():
            raise SystemExit("ERROR: softmax_lse contains NaN or Inf")

    metrics = cosine_metrics(reference, dut_out)
    print_accuracy_report(
        golden_name="npu_eqbsa_unquant_golden (NPU matmul+softmax+matmul on BF16, tiled Q)",
        dut_name=f"eagle_quant_block_sparse_attention (mix INT8 Q/K + FP8 V, inner_precise={INNER_PRECISE})",
        metrics=metrics,
        gate=None,
    )
    failures = collect_gate_failures(metrics, gate=None)
    if failures:
        raise SystemExit("ERROR: " + "; ".join(failures))
    print("RESULT accuracy check passed (mix, quant DUT vs unquant NPU compose)")


if __name__ == "__main__":
    main()
