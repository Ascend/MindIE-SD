# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Profile one isolated kernel, or measure a full call with NPU Events."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from tests.ops.sigmoid_mul_dynamic_mx_quant.batch import prepare_vendor

from tests.ops.sigmoid_mul_dynamic_mx_quant.workload import SHAPE, fused, make_inputs, metadata, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("kernel", "native"), required=True)
    parser.add_argument("--precision-mode", type=int, choices=(1,), default=1)
    parser.add_argument("--scale-alg", type=int, choices=(0, 1), default=0)
    parser.add_argument("--msprof-mode", action="store_true")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device_id < 0 or args.iterations < 1 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("require a physical device ID, positive iterations and unset ASCEND_RT_VISIBLE_DEVICES")
    if args.path != "kernel" and args.output is None:
        parser.error("Event paths require --output")
    prepare_vendor()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device_id)
    import mindiesd  # noqa: F401

    print(
        f"Preparing BF16 CPU inputs {SHAPE}; precision_mode={args.precision_mode}, scale_alg={args.scale_alg}",
        flush=True,
    )
    inputs = make_inputs(SHAPE, "normal")
    device = torch.device(f"npu:{args.device_id}")
    print(f"Copying inputs to {device}", flush=True)
    x, gate = (v.to(device) for v in inputs)
    print(f"Calling {args.path} on {x.shape} and {gate.shape}, dtype={x.dtype}/{gate.dtype}", flush=True)
    if args.path == "kernel":
        torch.npu.synchronize()
        output = fused(x, gate, args.precision_mode, args.scale_alg)
        torch.npu.synchronize()
        metadata(output, SHAPE, device)
    else:
        pairs = [
            (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(args.iterations)
        ]
        for begin, end in pairs:
            begin.record()
            end.record()
        for _ in range(5):
            reference(x, gate, args.precision_mode, args.scale_alg)
        for index, (begin, end) in enumerate(pairs):
            begin.record()
            output = reference(x, gate, args.precision_mode, args.scale_alg)
            end.record()
            if index + 1 < args.iterations:
                del output
        samples = [begin.elapsed_time(end) * 1000 for begin, end in pairs]
        if not all(math.isfinite(v) and v > 0 for v in samples):
            raise ValueError(f"invalid Event durations {samples}")
        metadata(output, SHAPE, device)
        if not torch.isfinite(output[0].float()).all().item() or (output[1] == 255).any().item():
            raise ValueError("nonfinite FP8 value or E8M0 scale")
        args.output.write_text(
            json.dumps(
                {
                    "shape": SHAPE,
                    "device": str(device),
                    "precision_mode": args.precision_mode,
                    "scale_alg": args.scale_alg,
                    "path": args.path,
                    "profiled": args.msprof_mode,
                    "warmup": 5,
                    "iterations": args.iterations,
                    "samples": len(samples),
                    "event_samples_us": samples,
                    "event_us": min(samples),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Complete-call min={min(samples):.6f} us", flush=True)
        with args.output.with_name("event_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Path", "Sample", "Layer Duration(us)"))
            writer.writerows((args.path, index, value) for index, value in enumerate(samples, 1))
    print(f"Completed {args.path} with outputs {[(v.shape, v.dtype) for v in output]}", flush=True)


if __name__ == "__main__":
    main()
