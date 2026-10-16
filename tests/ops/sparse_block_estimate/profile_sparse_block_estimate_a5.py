#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""One large-shape forward for msprof op.

Default shape matches large-shape accuracy: Q [1, 32, 2304, 128], K [1, 4, 30757, 128].
--kv-seq replaces only the K sequence length (65536 = 64K, 425984 = 416K).
Without --data the inputs are the open 50/50 distribution. With --data they follow that snapshot.
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from empirical_inputs import load_snapshot_profile, make_query_key  # noqa: E402
from run_a5_correctness import (  # noqa: E402
    HEAD_DIM,
    PRESETS,
    STRIDE,
    Case,
    _load_custom_operator,
    _make_inputs,
    _torch_dtype,
)

Q_BLOCK = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--kv-seq", type=int, default=30757)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--msprof-mode", action="store_true")
    args = parser.parse_args()
    if args.device_id < 0 or args.seed < 0 or args.kv_seq <= 0:
        parser.error("--device-id and --seed must be non-negative, and --kv-seq must be positive")
    return args


def main() -> None:
    args = _parse_args()
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        raise SystemExit(
            f"ERROR: ASCEND_RT_VISIBLE_DEVICES is set ({visible!r}). Unset it and pass --device-id (npu-smi NPU ID)."
        )
    base = PRESETS["target"]
    case = Case(base.batch, base.q_heads, base.kv_heads, base.q_seq, args.kv_seq)
    dtype = _torch_dtype(args.dtype)
    if args.data is not None:
        profile, _source = load_snapshot_profile(args.data, case.q_heads, case.kv_heads, HEAD_DIM, args.seed)
        query, key = make_query_key(
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
    else:
        query, key = _make_inputs(case, dtype, args.seed)
    _load_custom_operator(args.device_id)
    query = query.to(f"npu:{args.device_id}")
    key = key.to(f"npu:{args.device_id}")
    torch.npu.synchronize()
    print(
        f"scenario=sparse_block_estimate_target device_id={args.device_id} "
        f"dtype={args.dtype} msprof_mode={args.msprof_mode} "
        f"Q={tuple(query.shape)} K={tuple(key.shape)}"
    )
    with torch.no_grad():
        mask, count = torch.ops.mindiesd.sparse_block_estimate(
            query=query,
            key=key,
            input_layout="BNSD",
            stride=STRIDE,
            sparse_size=Q_BLOCK,
            num_heads=case.q_heads,
            num_key_value_heads=case.kv_heads,
            scale_value=HEAD_DIM**-0.5 / STRIDE,
            threshold=0.85,
            causal=False,
            keep_sink=True,
            keep_recent=True,
            row_sparse=1.0,
        )
        torch.npu.synchronize()
    kv_blocks_aligned = math.ceil(math.ceil(case.kv_seq / Q_BLOCK) / 32) * 32
    expected_mask = (case.batch, case.q_heads, math.ceil(case.q_seq / Q_BLOCK), kv_blocks_aligned)
    if tuple(mask.shape) != expected_mask or mask.dtype != torch.int8:
        raise AssertionError(f"unexpected mask output: {tuple(mask.shape)}, {mask.dtype}")
    if tuple(count.shape) != expected_mask[:-1] or count.dtype != torch.int32:
        raise AssertionError(f"unexpected count output: {tuple(count.shape)}, {count.dtype}")


if __name__ == "__main__":
    main()
