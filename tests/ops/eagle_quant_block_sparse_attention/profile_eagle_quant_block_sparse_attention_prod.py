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

"""Single-forward large-shape EQBSA mix or CANN FP8 BSA workload for ``msprof op``.

Shape matches the large-shape accuracy case. One impl per process so
cann_fp8_bsa and eqbsa records are not mixed.
"""

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
    CANN_FP8_BLOCK_SHAPE,
    INNER_PRECISE,
    OUTPUT_DTYPE,
    PROD_BATCH,
    PROD_HEAD_DIM,
    PROD_KV_SEQ,
    PROD_NUM_KV_HEADS,
    PROD_NUM_Q_HEADS,
    PROD_QUERY_SEQ,
    SEED,
    adapt_eqbsa_mask_for_cann_fp8,
    build_eqbsa_prequant_inputs,
    call_cann_fp8_block_sparse_attention,
    call_eagle_quant_block_sparse_attention,
    quantize_cann_fp8_qkv,
    quantize_eqbsa_qkv,
    softmax_scale,
)

SCENARIO_NAME = "eqbsa_mix_q1n32s2304_kv1n4s30757_d128"
_IMPLS = ("cann_fp8_bsa", "eqbsa")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CANN FP8 block_sparse_attention or eagle_quant_block_sparse_attention mix."
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=PROD_BATCH)
    parser.add_argument("--num-query-heads", type=int, default=PROD_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=PROD_NUM_KV_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=PROD_QUERY_SEQ)
    parser.add_argument("--kv-seq-len", type=int, default=PROD_KV_SEQ)
    parser.add_argument("--head-dim", type=int, default=PROD_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--impl",
        choices=_IMPLS,
        default="eqbsa",
        help="cann_fp8_bsa: CANN-bridged FP8 block_sparse_attention; eqbsa: mix EagleQuantBlockSparseAttention.",
    )
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
    if args.head_dim not in (64, 128):
        raise ValueError("eagle_quant_block_sparse_attention supports --head-dim 64 or 128")
    if args.impl == "cann_fp8_bsa" and args.batch_size != 1:
        raise ValueError("cann_fp8_bsa uses fa_block_quant_preprocess, which supports --batch-size 1 only")


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

    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    npu = get_npu_device()
    if npu != NPUDevice.A5:
        raise SystemExit(f"ERROR: this profile scene is Ascend 950, got {npu}")

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
    print(f"device_id={args.device_id} msprof_mode={args.msprof_mode} impl={args.impl}")

    if args.impl == "cann_fp8_bsa":
        query_q, key_q, value_q, query_scale, key_scale, value_scale = quantize_cann_fp8_qkv(query, key, value)
        mask_run = adapt_eqbsa_mask_for_cann_fp8(mask)
        block_shape = CANN_FP8_BLOCK_SHAPE
        torch_npu.npu.synchronize()
        print(
            f"BNSD Q={tuple(query.shape)} KV={tuple(key.shape)} "
            f"block_shape={list(block_shape)} inner_precise={INNER_PRECISE} "
            f"quant=cann_fp8_hadamard"
        )
        with torch.inference_mode():
            output = call_cann_fp8_block_sparse_attention(
                query_q, key_q, value_q, mask_run, query_scale, key_scale, value_scale, block_shape, scale
            )
            torch_npu.npu.synchronize()
    else:
        query_q, key_q, value_q, query_scale, key_scale, value_scale = quantize_eqbsa_qkv(query, key, value)
        torch_npu.npu.synchronize()
        print(
            f"BNSD Q={tuple(query.shape)} KV={tuple(key.shape)} "
            f"block_shape={list(BLOCK_SHAPE)} inner_precise={INNER_PRECISE} quant=eqbsa_mix"
        )
        with torch.inference_mode():
            output, softmax_lse = call_eagle_quant_block_sparse_attention(
                query_q, key_q, value_q, mask, query_scale, key_scale, value_scale, BLOCK_SHAPE, scale
            )
            torch_npu.npu.synchronize()
        if softmax_lse is not None and softmax_lse.numel() != 0:
            if not torch.isfinite(softmax_lse.float()).all().item():
                raise RuntimeError("softmax_lse contains NaN or Inf")

    expected_output_shape = (
        args.batch_size,
        args.num_query_heads,
        args.query_seq_len,
        args.head_dim,
    )
    if tuple(output.shape) != expected_output_shape or output.dtype != OUTPUT_DTYPE:
        raise RuntimeError(
            f"output mismatch: shape={tuple(output.shape)} dtype={output.dtype}, "
            f"expected={expected_output_shape}/{OUTPUT_DTYPE}"
        )
    if not torch.isfinite(output.float()).all().item():
        raise RuntimeError("attention_out contains NaN or Inf")
    print("RESULT msprof-mode forward done; use msprof op Duration, not wall-clock")


if __name__ == "__main__":
    main()
