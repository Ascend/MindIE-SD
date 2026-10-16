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

"""Single-forward large-shape Laser Attention workload for ``msprof op``.

Shape and dtype match the large-shape accuracy case. Inputs are generated directly
in contiguous BNSD with their operator dtypes. One process is one path: default FP16
(tiling key 0, kernel out FP32); ``--native-bf16`` generates BF16 Q/K and FP16 V
(tiling key 1, kernel out BF16). ``run_laser_attention_msprof_op.sh`` runs both
paths by default as two separate captures.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from laser_attention_accuracy_common import (  # noqa: E402
    LAYOUT,
    COMPUTE_DTYPE,
    MAX_TOKEN,
    PROD_BATCH,
    PROD_DTYPE,
    PROD_HEAD_DIM,
    PROD_KV_SEQ,
    PROD_NUM_HEADS,
    PROD_Q_SEQ,
    SEED,
    build_laser_attention_inputs,
    kernel_output_dtype,
    print_operator_inputs,
    softmax_scale,
    validate_shape,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile laser_attention with operator-ready BNSD inputs.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=PROD_BATCH)
    parser.add_argument("--num-heads", type=int, default=PROD_NUM_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=PROD_Q_SEQ)
    parser.add_argument("--kv-seq-len", type=int, default=PROD_KV_SEQ)
    parser.add_argument("--head-dim", type=int, default=PROD_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--msprof-mode", action="store_true")
    parser.add_argument(
        "--native-bf16",
        action="store_true",
        help="Generate BF16 Q/K and FP16 V (tiling key 1). Default generates FP16 Q/K/V (tiling key 0).",
    )
    return parser.parse_args()


def validate_args(args):
    if args.device_id < 0:
        raise ValueError("--device-id must be greater than or equal to 0")
    if args.batch_size <= 0 or args.query_seq_len <= 0 or args.kv_seq_len <= 0:
        raise ValueError("batch size and sequence lengths must be greater than 0")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be greater than 0")
    if args.head_dim <= 0:
        raise ValueError("--head-dim must be greater than 0")
    validate_shape(args.batch_size, args.num_heads, args.query_seq_len, args.kv_seq_len, args.head_dim)


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

    # MindIE-SD imports query the SoC and may lazily initialize the current NPU.
    torch.npu.set_device(args.device_id)

    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    npu = get_npu_device()
    if npu == NPUDevice.A5:
        raise SystemExit(f"ERROR: this profile scene is not supported on A5, got {npu}")

    dtype = PROD_DTYPE if args.native_bf16 else COMPUTE_DTYPE
    device = torch.device(f"npu:{args.device_id}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    scale = softmax_scale(args.head_dim)
    query, key, value = build_laser_attention_inputs(
        args.batch_size,
        args.num_heads,
        args.query_seq_len,
        args.kv_seq_len,
        args.head_dim,
        dtype,
        generator,
    )
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    expected_dtype = kernel_output_dtype(dtype)
    path_name = "native-bf16" if args.native_bf16 else "fp16"
    pre_tokens = MAX_TOKEN
    torch_npu.npu.synchronize()

    import mindiesd  # noqa: F401 - register torch.ops.mindiesd.la

    print("scenario=la_prod_bnsd")
    print(f"device_id={args.device_id} msprof_mode={args.msprof_mode} path={path_name}")
    print_operator_inputs(query, key, value, scale, pre_tokens)

    with torch.inference_mode():
        _, output = torch.ops.mindiesd.la(
            query,
            key,
            value,
            None,
            None,
            None,
            scale,
            args.num_heads,
            LAYOUT,
            1.0,
            pre_tokens,
            1,
            True,
        )
        torch_npu.npu.synchronize()

    expected_kernel_shape = tuple(query.shape)
    if tuple(output.shape) != expected_kernel_shape or output.dtype != expected_dtype:
        raise RuntimeError(
            f"kernel output mismatch: shape={tuple(output.shape)} dtype={output.dtype}, "
            f"expected={expected_kernel_shape}/{expected_dtype}"
        )
    if not torch.isfinite(output.float()).all().item():
        raise RuntimeError("attention_out contains NaN or Inf")
    print("RESULT msprof-mode forward done; Duration is laser_attention only")


if __name__ == "__main__":
    main()
