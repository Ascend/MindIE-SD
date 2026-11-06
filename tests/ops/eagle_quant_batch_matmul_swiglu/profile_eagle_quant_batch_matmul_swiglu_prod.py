"""One direct operator call for msprof op replay; no accuracy operations."""

import argparse
import json
from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_common import (
    dut,
    initialize,
    make_inputs,
    metadata,
    scenario,
    to_device,
)


def main():
    p = argparse.ArgumentParser(description="Single INT8 QBMM + FP32 SwiGLU; BF16 output; weight prepared as ND or NZ")
    p.add_argument("--device-id", type=int, required=True)
    p.add_argument("--weight-format", choices=["ND", "NZ"], default="ND")
    for name, default in (("m", 5672), ("k", 5376), ("n", 28672), ("seed", 20261030)):
        p.add_argument("--" + name, type=int, default=default)
    args = p.parse_args()
    if min(args.m, args.k, args.n) <= 0 or args.device_id < 0:
        p.error("positive dimensions and nonnegative physical device required")
    if args.n % 2:
        p.error("N must be even")
    if args.weight_format == "NZ" and args.n % 64:
        p.error("NZ weights without transpose require N/2 aligned to 32")
    torch, torch_npu = initialize(args.device_id)
    s = scenario(m=args.m, k=args.k, n=args.n, seed=args.seed)
    print("START CPU input generation", flush=True)
    cpu, parameters = make_inputs(s)
    print("START host-to-device transfer", flush=True)
    inputs = to_device(cpu, args.device_id)
    del cpu
    if args.weight_format == "NZ":
        inputs["x2"] = torch_npu.npu_format_cast(inputs["x2"], 29)
        if torch_npu.get_npu_format(inputs["x2"]) != 29:
            raise RuntimeError("weight preparation did not produce FRACTAL_NZ")
    details = metadata(inputs, s, parameters)
    details["inputs"]["x2"]["layout"] = args.weight_format
    print(json.dumps(details), flush=True)
    print("START input synchronization", flush=True)
    torch.npu.synchronize()
    print("START direct QBMM + SwiGLU invocation", flush=True)
    out = dut(inputs, s)
    print("START output synchronization", flush=True)
    torch.npu.synchronize()
    print(
        json.dumps(dict(output_shape=list(out.shape), output_dtype=str(out.dtype), output_device=str(out.device))),
        flush=True,
    )


if __name__ == "__main__":
    main()
