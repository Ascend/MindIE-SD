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

"""Single-forward large-shape BSA workload for ``msprof op``.

Shape and dtype match the large-shape accuracy case. One impl and one compute
dtype per process so cann_bsa and eagle_bsa records are not mixed.
"""

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
    PROD_BATCH,
    PROD_HEAD_DIM,
    PROD_NUM_KV_HEADS,
    PROD_NUM_Q_HEADS,
    PROD_SEQ,
    SEED,
    build_bsa_inputs,
    call_block_sparse_attention,
    call_eagle_block_sparse_attention,
    softmax_scale,
)

SCENARIO_NAME = "eagle_bsa_prod_q1n32s32768_kv1n4s32768_d128"
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_IMPLS = ("cann_bsa", "eagle_bsa")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CANN-bridged block_sparse_attention or eagle_block_sparse_attention."
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=PROD_BATCH)
    parser.add_argument("--num-query-heads", type=int, default=PROD_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=PROD_NUM_KV_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=PROD_SEQ)
    parser.add_argument("--kv-seq-len", type=int, default=PROD_SEQ)
    parser.add_argument("--head-dim", type=int, default=PROD_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dtype", choices=_DTYPE_MAP, default="float16")
    parser.add_argument(
        "--block-kv",
        type=int,
        default=BLOCK_SHAPE[1],
        help="block_shape[1] (BlockK). Tensor shape stays the prod Q/KV case. Default 128; 64 is the 16-aligned path.",
    )
    parser.add_argument(
        "--impl",
        choices=_IMPLS,
        default="eagle_bsa",
        help="cann_bsa: torch.ops.mindiesd.block_sparse_attention; eagle_bsa: local Eagle op.",
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
        raise ValueError("block sparse attention supports --head-dim 64 or 128")
    if args.block_kv <= 0 or args.block_kv % 16 != 0:
        raise ValueError("--block-kv must be a positive multiple of 16")


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
    if npu not in (NPUDevice.A2, NPUDevice.A3):
        raise SystemExit(f"ERROR: this profile scene is 910B/910_93, got {npu}")

    dtype = _DTYPE_MAP[args.dtype]
    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    scale = softmax_scale(args.head_dim)
    block_shape = (BLOCK_SHAPE[0], args.block_kv)
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
    torch_npu.npu.synchronize()

    print(f"scenario={SCENARIO_NAME}")
    print(f"device_id={args.device_id} msprof_mode={args.msprof_mode} impl={args.impl} dtype={dtype}")
    print(
        f"BNSD Q={tuple(query.shape)} KV={tuple(key.shape)} "
        f"block_shape={list(block_shape)} inner_precise=0"
    )

    call_op = call_block_sparse_attention if args.impl == "cann_bsa" else call_eagle_block_sparse_attention
    with torch.inference_mode():
        output, softmax_lse = call_op(query, key, value, mask, block_shape, scale)
        torch_npu.npu.synchronize()

    expected_output_shape = (
        args.batch_size,
        args.num_query_heads,
        args.query_seq_len,
        args.head_dim,
    )
    if tuple(output.shape) != expected_output_shape or output.dtype != dtype:
        raise RuntimeError(
            f"output mismatch: shape={tuple(output.shape)} dtype={output.dtype}, "
            f"expected={expected_output_shape}/{dtype}"
        )
    if not torch.isfinite(output.float()).all().item():
        raise RuntimeError("attention_out contains NaN or Inf")
    print("RESULT msprof-mode forward done; use msprof op Duration, not wall-clock")


if __name__ == "__main__":
    main()
