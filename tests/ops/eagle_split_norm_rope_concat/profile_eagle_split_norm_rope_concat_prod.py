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

import argparse

import torch

from eagle_split_norm_rope_concat_accuracy_common import call_dut, init_device, make_inputs
from eagle_split_norm_rope_concat_cases import (
    add_profile_selection,
    resolve_profile_selection,
    PROD_HEADS,
    PROD_SHAPE,
    case_attrs,
    is_compact,
    output_shapes,
)


def main():
    parser = argparse.ArgumentParser(description="One fused call with CPU-prepared inputs for msprof op/simulator")
    parser.add_argument("--device-id", type=int, default=0)
    add_profile_selection(parser)
    parser.add_argument("--msprof-mode", action="store_true", help="Compatibility flag; always calls the target once")
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    args.case = resolve_profile_selection(parser, args)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0,2^32)")
    if args.device_id < 0:
        parser.error("--device-id must be nonnegative")
    print(f"[1] Initializing device npu:{args.device_id}", flush=True)
    device = init_device(args.device_id)
    attrs = case_attrs(args.case)
    print(f"[2] CPU input generation: case={args.case} distribution=normal seed={args.seed}", flush=True)
    host_inputs = make_inputs(
        PROD_SHAPE,
        PROD_HEADS,
        attrs["rotary_dim"],
        torch.bfloat16,
        "cpu",
        "normal",
        seed=args.seed,
        rope_rank=4,
        compact=is_compact(args.case),
    )
    print("[3] Copying inputs to device", flush=True)
    # Preserve non-tensor metadata and avoid random-generation kernels in the simulator.
    inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in host_inputs)
    torch.npu.synchronize()
    print("[4] Calling EagleSplitNormRopeConcat", flush=True)
    outputs = call_dut(inputs, **attrs)  # Exactly one target call; msprof owns warmup/replay.
    print("[5] Target call returned; synchronizing", flush=True)
    torch.npu.synchronize()
    if len(outputs) != 3:
        raise RuntimeError(f"Expected three outputs, got {len(outputs)}")
    for out, expected_shape in zip(outputs, output_shapes(args.case)):
        if tuple(out.shape) != expected_shape or out.dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected output {out.shape}, {out.dtype}; expected {expected_shape}, bfloat16")
        if not out.is_contiguous():
            raise RuntimeError("Output must be contiguous")
    # Numerical checks belong to the accuracy entrypoint, not the simulator workload.
    print("[6] Target completed; output metadata checked", flush=True)


if __name__ == "__main__":
    main()
