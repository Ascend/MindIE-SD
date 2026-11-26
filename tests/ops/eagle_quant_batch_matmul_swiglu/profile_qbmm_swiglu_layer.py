"""Time native INT8 QBMM(BF16 output) followed by native SwiGLU with NPU Events."""

import argparse
import csv
import json
import math
from pathlib import Path

from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_common import (
    contract,
    initialize,
    make_inputs,
    metadata,
    production_scenario,
    to_device,
)
from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_state import save


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Defaults: M=5672 K=5376 N=28672, INT8 inputs, FP32 channel/token scales, "
            "BF16 intermediate and output, no bias. NZ packs W.T[N,K] before timing. "
            "Fixed 5 warmups; 10 individual Event samples by default; hot-cache inputs."
        ),
    )
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--weight-format", choices=["ND", "NZ"], required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--msprof-mode", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("m", 5672), ("k", 5376), ("n", 28672), ("seed", 20261030)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if min(args.m, args.k, args.n, args.iterations) <= 0 or args.device_id < 0 or args.n % 2:
        parser.error("positive dimensions/iterations, even N and nonnegative physical device required")
    s = production_scenario(args.weight_format, m=args.m, k=args.k, n=args.n, seed=args.seed)
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be a new absolute file path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch, torch_npu = initialize(args.device_id)
    cpu, parameters = make_inputs(s)
    inputs = to_device(cpu, args.device_id, args.weight_format)
    del cpu
    # Native QBMM infers the transpose from a strided view of the packed weight.
    weight = inputs["x2"].t() if s["transpose_x2"] else inputs["x2"]
    if s["transpose_x2"] and (weight.data_ptr() != inputs["x2"].data_ptr() or torch_npu.get_npu_format(weight) != 29):
        raise RuntimeError("native QBMM requires a transposed view sharing the NZ weight storage")
    details = metadata(inputs, s, parameters)
    details["dut"] = "torch_npu.npu_quant_matmul(BF16-out) -> torch_npu.npu_swiglu(dim=-1)"
    details["inputs"]["x2"].update(
        shape=list(weight.shape), stride=list(weight.stride()), contiguous=weight.is_contiguous()
    )
    print(json.dumps(details), flush=True)

    def function():
        intermediate = torch_npu.npu_quant_matmul(
            inputs["x1"],
            weight,
            inputs["scale"],
            pertoken_scale=inputs["pertoken_scale"],
            output_dtype=torch.bfloat16,
        )
        return torch_npu.npu_swiglu(intermediate, dim=-1)

    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, end in zip(starts, ends):
        start.record()
        end.record()
    for _ in range(5):
        output = function()
        del output
    for index in range(args.iterations):
        starts[index].record()
        output = function()
        ends[index].record()
        if index + 1 < args.iterations:
            del output
    values = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise RuntimeError(f"Invalid Event samples: {values}")
    contract(output, s)
    with (args.output.parent / "event_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Path", "Sample", "Layer Duration(us)"])
        writer.writerows(("torch_npu", i, value) for i, value in enumerate(values, 1))
    result = dict(
        path="torch_npu",
        weight_format=args.weight_format,
        device=args.device_id,
        scenario=s,
        metadata=details,
        warmup=5,
        iterations=args.iterations,
        samples=len(values),
        event_samples_us=values,
        event_us=min(values),
        profiled=args.msprof_mode,
        intermediate_dtype="BF16",
        output_dtype=str(output.dtype),
        input_preparation="CPU quantization and transfer; NZ packing before warmup/events",
        cache="hot",
    )
    save(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
