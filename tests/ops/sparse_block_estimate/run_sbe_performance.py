#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Time one large-shape small-op concatenation (torch_npu_baseline).

Q stays [1, 32, 2304, 128]. --kv-seq is the K sequence length.
The large-shape shell runs this once per K length, then profiles the fused estimate kernel.
"""

import argparse
import importlib
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from empirical_inputs import load_snapshot_profile, make_query_key  # noqa: E402
from run_a5_correctness import HEAD_DIM, PRESETS, Case, _make_inputs, _torch_dtype  # noqa: E402
from torch_npu_baseline import sparse_block_estimate_baseline  # noqa: E402

Q_BLOCK = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--kv-seq", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.device_id < 0 or args.seed < 0 or args.kv_seq <= 0:
        parser.error("--device-id and --seed must be non-negative, and --kv-seq must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    return args


def _load_inputs(args, case: Case, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if args.data is None:
        return _make_inputs(case, dtype, args.seed)
    profile, _source = load_snapshot_profile(args.data, case.q_heads, case.kv_heads, HEAD_DIM, args.seed)
    return make_query_key(
        profile,
        case.q_heads,
        case.kv_heads,
        case.q_seq,
        case.kv_seq,
        HEAD_DIM,
        "realistic",
        1.0,
        args.seed,
        dtype,
    )


def main() -> None:
    args = _parse_args()
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        raise SystemExit(
            f"ERROR: ASCEND_RT_VISIBLE_DEVICES is set ({visible!r}). Unset it and pass --device-id (npu-smi NPU ID)."
        )
    torch_npu = importlib.import_module("torch_npu")
    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU is not available")
    torch.npu.set_device(args.device_id)
    soc_version = torch_npu.npu.get_soc_version()
    if soc_version != 260:
        raise RuntimeError(f"device {args.device_id} is SoC version {soc_version}, expected A5 (260)")
    torch.npu.matmul.allow_hf32 = False

    base = PRESETS["target"]
    case = Case(base.batch, base.q_heads, base.kv_heads, base.q_seq, args.kv_seq)
    dtype = _torch_dtype(args.dtype)
    query, key = _load_inputs(args, case, dtype)
    device = f"npu:{args.device_id}"
    query = query.to(device)
    key = key.to(device)
    scale_value = HEAD_DIM**-0.5 / 8
    torch.npu.synchronize()
    print(
        f"scenario=sparse_block_estimate_baseline device_id={args.device_id} "
        f"dtype={args.dtype} Q={tuple(query.shape)} K={tuple(key.shape)}"
    )

    wall_us: list[float] = []
    with torch.no_grad():
        for index in range(args.warmup + args.repeats):
            torch.npu.synchronize()
            begin = time.perf_counter_ns()
            mask, count = sparse_block_estimate_baseline(
                query,
                key,
                scale_value=scale_value,
                threshold=0.85,
                row_sparse=1.0,
                keep_sink=True,
                keep_recent=True,
                matmul_mode="fp32",
            )
            torch.npu.synchronize()
            if index >= args.warmup:
                wall_us.append((time.perf_counter_ns() - begin) / 1000.0)
            if index == args.warmup + args.repeats - 1:
                kv_blocks_aligned = math.ceil(math.ceil(case.kv_seq / Q_BLOCK) / 32) * 32
                expected_mask = (case.batch, case.q_heads, math.ceil(case.q_seq / Q_BLOCK), kv_blocks_aligned)
                if tuple(mask.shape) != expected_mask or mask.dtype != torch.int8:
                    raise AssertionError(f"unexpected mask output: {tuple(mask.shape)}, {mask.dtype}")
                if tuple(count.shape) != expected_mask[:-1] or count.dtype != torch.int32:
                    raise AssertionError(f"unexpected count output: {tuple(count.shape)}, {count.dtype}")
            del mask, count
    p50_us = statistics.median(wall_us)
    print(f"RESULT baseline kv_seq={case.kv_seq} dtype={args.dtype} p50_us={p50_us:.3f} repeats={args.repeats}")


if __name__ == "__main__":
    main()
