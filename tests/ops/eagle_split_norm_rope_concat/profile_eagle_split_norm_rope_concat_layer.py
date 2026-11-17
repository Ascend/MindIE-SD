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

"""Measure complete native BF16 RMSNorm/RoPE composition paths with NPU Events."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

from eagle_split_norm_rope_concat_cases import (
    add_profile_selection,
    resolve_profile_selection,
    PROD_HEADS,
    PROD_SHAPE,
    case_attrs,
    is_compact,
    output_shapes,
)
from eagle_split_norm_rope_concat_layer_common import (
    PATHS,
    PROTOCOL,
    input_config,
    run_path,
    scenario_name,
    tensor_state,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--path", choices=PATHS, required=True)
    add_profile_selection(parser)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--msprof-mode", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.case = resolve_profile_selection(parser, args)
    if args.device_id < 0 or args.iterations < 1 or not 0 <= args.seed < 2**32:
        parser.error("Require nonnegative device ID, positive iterations and seed in [0,2^32)")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Unset ASCEND_RT_VISIBLE_DEVICES; use a physical --device-id")
    if not args.output.is_absolute() or args.output.exists():
        parser.error("--output must be an absolute path to a new timing JSON")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device_id)
    torch.npu.set_compile_mode(jit_compile=False)
    import mindiesd  # noqa: F401

    from eagle_split_norm_rope_concat_accuracy_common import log_inputs, make_inputs

    attrs = case_attrs(args.case)
    config = input_config(args.case)
    inputs = make_inputs(
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
    # Match the fused profile's CPU RNG backend, seed and input order exactly.
    inputs = tuple(x.to(f"npu:{args.device_id}") if isinstance(x, torch.Tensor) else x for x in inputs)
    torch.npu.synchronize()
    log_inputs(inputs, **attrs)
    states = [tensor_state(x) if isinstance(x, torch.Tensor) else {"type": "int[]", "value": list(x)} for x in inputs]
    expected_shapes = [config["qkv_shape"], *config["weight_shapes"], *config["rope_shapes"], None]
    actual_shapes = [item.get("shape") for item in states]
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"Unexpected input shapes: expected {expected_shapes}, got {actual_shapes}")
    if inputs[5] != config["head_counts"]:
        raise RuntimeError(f"Unexpected head counts: {inputs[5]}")
    scenario = scenario_name(args.path, args.case)
    print(f"case={args.case} scenario={scenario} distribution=normal; hot-cache reused inputs", flush=True)
    internal_inputs = []
    with torch.inference_mode():
        starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
        for start, end in zip(starts, ends):
            start.record()
            end.record()
        for index in range(5):
            output = run_path(args.path, inputs, args.case, trace=internal_inputs if index == 0 else None)
            del output
        for index in range(args.iterations):
            starts[index].record()
            output = run_path(args.path, inputs, args.case)
            ends[index].record()
            if index + 1 < args.iterations:
                del output
        # elapsed_time waits internally; all calls are submitted before reading any sample.
        values = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise RuntimeError(f"Invalid Event duration: {values}")
        if len(output) != 3:
            raise RuntimeError("Unexpected output count")
        for tensor, expected_shape in zip(output, output_shapes(args.case)):
            if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"Unexpected output {tensor.shape}, {tensor.dtype}")
            if not tensor.is_contiguous() or not torch.isfinite(tensor).all().item():
                raise RuntimeError("Output must be contiguous and finite")
    print(f"Actual native BF16-chain intermediate inputs (first warmup): {json.dumps(internal_inputs)}", flush=True)
    report = {
        "protocol": PROTOCOL,
        "profiled": args.msprof_mode,
        "path": args.path,
        "case_id": args.case,
        "scenario": scenario,
        "distribution": "normal",
        "seed": args.seed,
        "device_id": args.device_id,
        "warmup": 5,
        "iterations": args.iterations,
        "samples": len(values),
        "event_samples_us": values,
        "event_us": min(values),
        "input_config": config,
        "actual_inputs": states,
        "internal_inputs": internal_inputs,
        "output_count": len(output),
        "runtime": {
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "device_name": torch.npu.get_device_name(args.device_id),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.output.with_name("event_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Path", "Sample", "Layer Duration(us)"])
        writer.writerows((args.path, index, value) for index, value in enumerate(values, 1))
    print(f"warmup=5 iterations={args.iterations} profiled={args.msprof_mode} min_event_us={min(values)}")


if __name__ == "__main__":
    main()
