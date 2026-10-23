#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Warm up five times, then report the minimum of individual layer Event timings."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

WARMUP = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--path", choices=("current", "legacy", "both"), default="both")
    parser.add_argument("--iterations", type=int, default=10, help="Measured calls per capture (default: 10).")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--msprof-mode", action="store_true", help="Label results collected under msprof.")
    parser.add_argument("--output", type=Path, required=True, help="Timing JSON file.")
    args = parser.parse_args()
    if args.device_id < 0 or args.iterations < 1:
        parser.error("device-id must be nonnegative and iterations must be at least 1")
    if args.msprof_mode and args.path == "both":
        parser.error("Use a separate msprof process for each path")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Unset ASCEND_RT_VISIBLE_DEVICES; use --device-id with the physical NPU ID")
    return args


def measure_calls(torch, function, iterations=10):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    # Initialize the lazily created events before warmup and outside measurement.
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        end.record()
    for _ in range(WARMUP):
        output = function()
        del output

    for index in range(iterations):
        starts[index].record()
        output = function()
        ends[index].record()
        if index + 1 < iterations:
            del output

    # Submit every measured call before elapsed_time waits for event completion.
    event_samples_us = []
    for start, end in zip(starts, ends):
        event_us = start.elapsed_time(end) * 1000
        if not math.isfinite(event_us) or event_us <= 0:
            raise RuntimeError(f"Invalid timing: event_us={event_us}")
        event_samples_us.append(event_us)
    return output, {"event_us": min(event_samples_us), "samples": iterations, "event_samples_us": event_samples_us}


def write_event_samples(destination, results):
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Path", "Sample", "Layer Duration(us)"])
        for name, timing in results.items():
            for index, value in enumerate(timing["event_samples_us"], 1):
                writer.writerow([name, index, value])


def main():
    args = parse_args()
    # Resolve this checkout even when launched from another working directory.
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    import torch
    import torch_npu  # noqa: F401 - registers torch.npu

    # MindIE-SD imports query the platform and can initialize the default device.
    torch.npu.set_device(args.device_id)
    from tests.layers.flash_attn.laser_attention.laser_attention_layer_common import LAYERS, make_inputs
    from mindiesd.utils.get_platform import is_a5_device

    if is_a5_device():
        raise SystemExit("LaserAttention is unsupported on A5")
    param, query, key, value = make_inputs(f"npu:{args.device_id}", args.seed)
    scale = param.head_dim**-0.5
    print(f"layer input layout=BSND scale={scale} mask=None seed={args.seed}", flush=True)
    for name, tensor in (("Q", query), ("K", key), ("V", value)):
        print(
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
            f"stride={tensor.stride()} contiguous={tensor.is_contiguous()}",
            flush=True,
        )
    print(
        "Inside layer: BNSD Q=[1,24,5376,128], K/V=[1,24,8960,128], "
        "valid Q/KV=5280/8800, pre_tokens=160. Current Q/K=BF16,V=FP16; legacy Q/K/V=FP16.",
        flush=True,
    )
    paths = ("current", "legacy")
    paths = paths if args.path == "both" else (args.path,)
    mode = "profiled diagnostic" if args.msprof_mode else "unprofiled comparison"
    runtime = {
        "pid": os.getpid(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device_name": torch.npu.get_device_name(args.device_id),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "ASCEND_LAUNCH_BLOCKING",
                "TASK_QUEUE_ENABLE",
                "CPU_AFFINITY_CONF",
                "PYTORCH_NPU_ALLOC_CONF",
            )
        },
    }
    print(f"Timing mode={mode}; path order={','.join(paths)}; runtime={json.dumps(runtime)}", flush=True)
    results = {}
    with torch.inference_mode():
        for name in paths:

            def function(name=name):
                return LAYERS[name](param, query, key, value, scale=scale)

            output, timing = measure_calls(torch, function, args.iterations)
            # Validation is outside the timed event interval.
            if output.shape != query.shape or output.dtype != torch.bfloat16:
                raise RuntimeError(f"{name}: unexpected output {output.shape}, {output.dtype}")
            if not torch.isfinite(output).all().item():
                raise RuntimeError(f"{name}: output contains NaN/Inf")
            results[name] = timing
            del output
            print(f"RESULT {name} [{mode}]: {json.dumps(timing)}", flush=True)
    report = {
        "profiled": args.msprof_mode,
        "path_order": list(paths),
        "runtime": runtime,
        "device_id": args.device_id,
        "warmup": WARMUP,
        "iterations": args.iterations,
        "seed": args.seed,
        "layout": "BSND",
        "dtype": "bfloat16",
        "q_shape": list(query.shape),
        "kv_shape": list(key.shape),
        "results": results,
    }
    if len(results) == 2:
        report["legacy_over_current"] = {"event_us": results["legacy"]["event_us"] / results["current"]["event_us"]}
        print(f"legacy/current (>1 means current faster): {report['legacy_over_current']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_event_samples(args.output.with_name("event_samples.csv"), results)
    print(f"Warmup={WARMUP}; measured calls={args.iterations}; event_us is their minimum.")
    if args.msprof_mode:
        print("Event timings measure the complete layer under msprof.")


if __name__ == "__main__":
    main()
