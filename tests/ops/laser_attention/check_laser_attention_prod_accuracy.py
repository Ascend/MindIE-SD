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

"""Large-shape Laser Attention accuracy: raw BNSD DUT vs NPU compose. Never CPU."""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from laser_attention_accuracy_common import (  # noqa: E402
    ACCURACY_DISTRIBUTIONS,
    COMPUTE_DTYPE,
    PROD_BATCH,
    PROD_DTYPE,
    PROD_HEAD_DIM,
    PROD_KV_SEQ,
    PROD_NUM_HEADS,
    PROD_Q_SEQ,
    SEED,
    build_laser_attention_inputs,
    call_laser_attention,
    check_mixed_tolerance,
    collect_gate_failures,
    cosine_metrics,
    kernel_output_dtype,
    npu_laser_attention_golden,
    print_operator_inputs,
    print_accuracy_report,
    softmax_scale,
    validate_shape,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare raw BNSD LaserAttention output against NPU FP32 compose.")
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Physical NPU ID from npu-smi info. Do not use ASCEND_RT_VISIBLE_DEVICES.",
    )
    parser.add_argument("--batch-size", type=int, default=PROD_BATCH)
    parser.add_argument("--num-heads", type=int, default=PROD_NUM_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=PROD_Q_SEQ)
    parser.add_argument("--kv-seq-len", type=int, default=PROD_KV_SEQ)
    parser.add_argument("--head-dim", type=int, default=PROD_HEAD_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _validate_args(args):
    if args.device_id < 0:
        raise ValueError("--device-id must be greater than or equal to 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be greater than 0")
    if args.query_seq_len <= 0 or args.kv_seq_len <= 0:
        raise ValueError("sequence lengths must be greater than 0")
    if args.head_dim <= 0:
        raise ValueError("--head-dim must be greater than 0")
    validate_shape(args.batch_size, args.num_heads, args.query_seq_len, args.kv_seq_len, args.head_dim)


def _check_dut(dut_out, reference, expected_dtype, mixed_dtype, golden_name, dut_name, mixed_note):
    if not torch.isfinite(dut_out.float()).all().item():
        raise SystemExit(f"ERROR: {dut_name} output contains NaN or Inf")
    if tuple(dut_out.shape) != tuple(reference.shape):
        raise SystemExit(f"ERROR: shape mismatch {dut_name} DUT={tuple(dut_out.shape)} ref={tuple(reference.shape)}")
    if dut_out.dtype != expected_dtype:
        raise SystemExit(f"ERROR: dtype mismatch {dut_name} DUT={dut_out.dtype} expected={expected_dtype}")
    metrics = cosine_metrics(reference, dut_out)
    gate = check_mixed_tolerance(dut_out, reference, dtype=mixed_dtype)
    print_accuracy_report(
        golden_name=golden_name,
        dut_name=dut_name,
        metrics=metrics,
        gate=gate,
    )
    print(mixed_note)
    return collect_gate_failures(metrics, gate, require_mixed=False)


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
    if npu == NPUDevice.A5:
        raise SystemExit(f"ERROR: ascend_laser_attention is not supported on A5, got {npu}")

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    scale = softmax_scale(args.head_dim)
    print(f"device_id={args.device_id}")
    failures = []
    for (path_name, dtype), distribution in product(
        (("fp16", COMPUTE_DTYPE), ("native-bf16", PROD_DTYPE)), ACCURACY_DISTRIBUTIONS
    ):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        print(f"scenario=la_prod_bnsd path={path_name} distribution={distribution} seed={args.seed}")
        query, key, value = build_laser_attention_inputs(
            args.batch_size,
            args.num_heads,
            args.query_seq_len,
            args.kv_seq_len,
            args.head_dim,
            dtype,
            generator,
            distribution=distribution,
            log_distribution=True,
        )
        query, key, value = (tensor.to(device) for tensor in (query, key, value))
        print_operator_inputs(query, key, value, scale)
        with torch.inference_mode():
            reference = npu_laser_attention_golden(query, key, value, scale)
            torch.npu.synchronize()
            dut_out = call_laser_attention(query, key, value, scale)
            torch.npu.synchronize()
        if not torch.isfinite(reference.float()).all().item():
            raise SystemExit("ERROR: NPU compose golden contains NaN or Inf")
        case_failures = _check_dut(
            dut_out=dut_out,
            reference=reference,
            expected_dtype=kernel_output_dtype(dtype),
            mixed_dtype=dtype,
            golden_name="NPU FP32 matmul + softmax + matmul (same BNSD input values)",
            dut_name=f"torch.ops.mindiesd.la ({path_name}, {distribution}, raw BNSD output)",
            mixed_note=(
                "mixed_tolerance matched_ratio and max_abs_error are informational for large shapes; "
                "only cosine and norm_ratio are accuracy hard gates"
            ),
        )
        failures.extend(f"{path_name}/{distribution}: {failure}" for failure in case_failures)
        del query, key, value, reference, dut_out
    if failures:
        raise SystemExit("ERROR: " + "; ".join(failures))
    print("RESULT accuracy check passed (fp16 + native-bf16, uniform + normal, BNSD)")


if __name__ == "__main__":
    main()
