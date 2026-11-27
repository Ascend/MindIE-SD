# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Production-shape stability and single-kernel profiling for EagleHadamardQuant."""

import argparse
import csv
import fnmatch
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


CASES = {
    "k128-fp4": (81920, 4096, 128, "mxfp4"),
    "k32-fp4": (81920, 1536, 32, "mxfp4"),
    "k128-fp8": (81920, 4096, 128, "mxfp8"),
}
SHARED_CASE = "shared-k128-fp4"
SHAPES = {**CASES, SHARED_CASE: (4096, 4096, 128, "mxfp4")}
KERNEL = "*agle*adamard*uant*"


def make_inputs(torch, rows, cols, k, scenario):
    generator = torch.Generator().manual_seed(42)
    signs = torch.randint(2, (cols // k, 1, k), generator=torch.Generator().manual_seed(43)) * 2 - 1
    if scenario == "historical":
        x = torch.randn(rows, cols, dtype=torch.bfloat16, generator=generator)
        h = torch.ones(1, 1)
        while h.shape[0] < k:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)))
        rotation = (h / math.sqrt(k) * signs).bfloat16()
    else:
        x = torch.empty(rows, cols, dtype=torch.bfloat16)
        if scenario == "uniform":
            x.uniform_(-5, 5, generator=generator)
        else:
            mu = torch.empty(()).uniform_(-5, 5, generator=generator).item()
            sigma = torch.empty(()).uniform_(0.1, 2, generator=generator).item()
            print(f"seed=42 mu={mu:.8g} sigma={sigma:.8g}")
            x.normal_(mu, sigma, generator=generator)
        # Different signed permutations per block expose transposition and indexing errors.
        blocks = torch.arange(cols // k)[:, None]
        columns = torch.arange(k)
        rotation = torch.zeros(cols // k, k, k, dtype=torch.bfloat16)
        rotation[blocks, (5 * columns + blocks + 3) % k, columns] = signs[:, 0].bfloat16()
        # At both ends and around row tiles, exercise zero, near-zero and tie-to-even groups.
        anchors = torch.tensor([4, 0.25, -0.75, 1.25, -1.75, 3.5, -5, 0] * 4)
        for row in (0, 63, 64, 127, 128, rows - 1):
            if row < rows:
                x[row, :32] = anchors
                x[row, 32:64] = 0
                x[row, -32:] = anchors * (2**-18)
    return x, rotation


def reference_bytes(torch, rotated, fmt):
    """OCP E8M0 per 32 BF16 values. E2M1 packs the even column into the low nibble."""
    rows, cols = rotated.shape
    groups = rotated.bfloat16().float().reshape(rows, -1, 32)
    peak = groups.abs().amax(dim=-1)
    fp4 = fmt == "mxfp4"
    exponent = torch.frexp(peak).exponent - 1 - (2 if fp4 else 8)
    scales = torch.where(peak == 0, 0, exponent + 127).to(torch.uint8)
    values = torch.ldexp(groups, -exponent.unsqueeze(-1)).reshape(rows, cols)
    return encode_values(torch, values, fmt), scales.reshape(rows, cols // 64, 2)


def encode_values(torch, values, fmt):
    if fmt == "mxfp4":
        mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        magnitude = values.abs().contiguous()
        codes = torch.bucketize(magnitude, mid)
        # At a midpoint choose the even code, including zero.
        codes += (codes.remainder(2).bool() & (magnitude == mid[codes.clamp_max(6)])).long()
        codes = codes.to(torch.uint8) | (torch.signbit(values).to(torch.uint8) << 3)
        encoded = codes[:, ::2] | (codes[:, 1::2] << 4)
    else:
        encoded = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    return encoded


def ordered_codes(torch, y, fmt):
    if fmt == "mxfp4":
        y = torch.stack((y & 15, y >> 4), -1).flatten(1)
        sign = 8
    else:
        sign = 128
    magnitude = (y & (sign - 1)).short()
    return torch.where(y < sign, magnitude, -magnitude)


def compare_bytes(torch, actual, expected, label, row_offset=0):
    if torch.equal(actual, expected):
        return
    # Copy at most one small slab for diagnostics when a production tensor differs.
    for start in range(0, actual.shape[0], 128):
        a = actual[start : start + 128].cpu()
        b = expected[start : start + 128].cpu()
        if not torch.equal(a, b):
            pos = tuple((a != b).nonzero()[0].tolist())
            coordinate = (row_offset + start + pos[0], *pos[1:])
            raise AssertionError(f"{label}{coordinate}: got {a[pos].item()}, expected {b[pos].item()}")
    raise AssertionError(f"{label}: device comparison failed")


def check_accuracy(torch, x, rotation, outputs, fmt, scenario, label):
    # Chunking bounds host memory while covering every row, block and output byte.
    k = rotation.shape[-1]
    weights = rotation.double()
    for start in range(0, len(x), 256):
        chunk = x[start : start + 256]
        blocks = chunk.reshape(len(chunk), -1, k).transpose(0, 1).double()
        rotated = torch.bmm(blocks, weights).transpose(0, 1).reshape_as(chunk)
        expected = reference_bytes(torch, rotated, fmt)
        actual = [tensor[start : start + len(chunk)].view(torch.uint8).cpu() for tensor in outputs]
        if scenario == "historical":
            # Dense accumulation order can change BF16 rounding. Check the encoded
            # result first, then compare the affected slab in physical units.
            if any(not torch.equal(a, b) for a, b in zip(actual, expected)):
                # For a K-term FP32 dot product the absolute rounding error is bounded by
                # gamma_K * sum(abs(x_i * w_i)). A stable BF16 group has an exact MX encoding.
                unit = k * 2.0**-24
                bound = torch.bmm(blocks.abs(), weights.abs()).transpose(0, 1).reshape_as(chunk) * (unit / (1 - unit))
                low = (rotated - bound).bfloat16().float().reshape(len(chunk), -1, 32)
                high = (rotated + bound).bfloat16().float().reshape_as(low)
                min_abs = torch.where(low * high <= 0, 0, torch.minimum(low.abs(), high.abs()))
                max_abs = torch.maximum(low.abs(), high.abs())
                scales = actual[1].reshape(len(chunk), -1).int()

                def scale_code(values):
                    peak = values.amax(-1)
                    return torch.where(peak == 0, 0, torch.frexp(peak).exponent + 126 - (2 if fmt == "mxfp4" else 8))

                bad_scale = (scales < scale_code(min_abs)) | (scales > scale_code(max_abs))
                if bad_scale.any():
                    row, group = bad_scale.nonzero()[0].tolist()
                    raise AssertionError(f"{label} scale row={start + row} group={group} outside rounding bounds")

                def rank_bound(values, scale_codes):
                    scaled = torch.ldexp(values, 127 - scale_codes.unsqueeze(-1)).flatten(1)
                    return ordered_codes(torch, encode_values(torch, scaled, fmt), fmt)

                rank = ordered_codes(torch, actual[0], fmt)
                invalid = (rank < rank_bound(low, scales)) | (rank > rank_bound(high, scales))
                if invalid.any():
                    row, col = invalid.nonzero()[0].tolist()
                    raise AssertionError(f"{label} y row={start + row} col={col} outside rounding bounds")
                check_dense(torch, actual, expected, fmt, label, start)
        else:
            for name, observed, wanted in zip(("y", "scale"), actual, expected):
                compare_bytes(torch, observed, wanted, f"{label} {name}", start)


def check_dense(torch, actual, expected, fmt, label, start):
    def decode(pair):
        y, scale = pair
        if fmt == "mxfp4":
            codes = torch.stack((y & 15, y >> 4), -1).flatten(1)
            levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
            values = levels[(codes & 7).long()] * torch.where(codes < 8, 1, -1)
        else:
            values = y.view(torch.float8_e4m3fn).float()
        return torch.ldexp(values.reshape(len(y), -1, 32), scale.reshape(len(y), -1, 1).int() - 127).flatten(1)

    # Dense reductions may differ at BF16 rounding boundaries. An adjacent scale
    # exponent is permissible only if the reconstructed values still pass the numerical gate.
    delta = (actual[1].int() - expected[1].int()).abs()
    if delta.max().item() > 1:
        pos = tuple((delta > 1).nonzero()[0].tolist())
        raise AssertionError(f"{label} scale row={start + pos[0]} index={pos[1:]} delta={delta[pos].item()}")
    a, b = decode(actual), decode(expected)
    tol = 0.5 if fmt == "mxfp4" else 0.125
    errors = (a - b).abs()
    # Check each row so an isolated corrupted row cannot hide in a production tensor.
    denom = b.norm(dim=1)
    ratio = a.norm(dim=1) / denom.clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=1)
    matched = (errors <= tol * (1 + b.abs())).float().mean(dim=1)
    good = (ratio >= 0.9) & (ratio <= 1.1) & (cosine >= 0.99) & (matched >= 0.99)
    good = torch.where(denom == 0, (a == 0).all(dim=1), good)
    if not good.all():
        index = int((~good).nonzero()[0].item())
        raise AssertionError(
            f"{label} row={start + index} cosine={cosine[index]} norm_ratio={ratio[index]} "
            f"matched={matched[index]} max_abs={errors[index].max()}"
        )


def run_on_device(args, case, scenario):
    import torch
    import torch_npu

    device = torch.device("npu", args.device_id)
    torch_npu.npu.set_device(device)
    from mindiesd.utils.get_platform import get_npu_device
    from mindiesd.utils.utils import is_npu_available

    if not is_npu_available():
        raise RuntimeError("Eagle production tests require an Ascend NPU")
    label = f"{case}/{scenario}"
    print(f"{label}: {device} {get_npu_device().name}", flush=True)
    rows, cols, k, fmt = SHAPES[case]
    host_x, host_rotation = make_inputs(torch, rows, cols, k, scenario)
    nz = scenario == "normal"
    weight = host_rotation
    if case == SHARED_CASE:
        weight = host_rotation[0]
        host_rotation = weight.expand(cols // k, k, k)
    if nz:
        weight = weight.reshape(-1, k // 16, 16, k // 16, 16).permute(0, 3, 1, 2, 4).contiguous()
    x, rotation = host_x.to(device), weight.to(device)
    torch.npu.synchronize(device)

    def invoke():
        result = torch.ops.mindiesd.eagle_hadamard_quant(x, rotation, k, fmt, nz)
        torch.npu.synchronize(device)
        return result

    if args.mode == "worker":
        invoke()
        return

    specs = (
        ((rows, cols // 2 if fmt == "mxfp4" else cols), torch.uint8 if fmt == "mxfp4" else torch.float8_e4m3fn),
        ((rows, cols // 64, 2), torch.float8_e8m0fnu),
    )

    def checked():
        y, scale = invoke()
        for tensor, (shape, dtype) in zip((y, scale), specs):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
                raise AssertionError(
                    f"{label}: got {tensor.shape} {tensor.dtype} {tensor.device}, expected {shape} {dtype} {device}"
                )
        return y, scale

    def check_inputs():
        for name, current, original in (("x", x, host_x), ("sign_h", rotation, weight)):
            compare_bytes(torch, current.cpu().view(torch.uint8), original.view(torch.uint8), f"{label} input {name}")

    first = checked()
    check_accuracy(torch, host_x, host_rotation, first, fmt, scenario, label)
    check_inputs()
    if args.save_outputs or args.compare_outputs:
        folder = args.save_outputs or args.compare_outputs
        if args.save_outputs:
            folder.mkdir(parents=True, exist_ok=True)
        for name, tensor in zip(("y", "scale"), first):
            path = folder / f"{name}.bin"
            raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
            if args.save_outputs:
                path.write_bytes(raw)
            elif path.read_bytes() != raw:
                raise AssertionError(f"{label}: {name} differs from {path}")
    print(f"PASS {label} full-output accuracy", flush=True)
    if args.phase == "accuracy":
        return
    baseline = [item.view(torch.uint8).clone() for item in first]
    del first
    input_copies = (x.clone(), rotation.clone())
    completed = 0
    try:
        for run in range(1, args.stress_runs + 1):
            for name, tensor, expected in zip(("y", "scale"), checked(), baseline):
                compare_bytes(torch, tensor.view(torch.uint8), expected, f"{label} run {run} {name}")
            for name, current, original in zip(("x", "sign_h"), (x, rotation), input_copies):
                compare_bytes(
                    torch, current.view(torch.uint8), original.view(torch.uint8), f"{label} run {run} input {name}"
                )
            completed = run
    finally:
        status = "PASS" if completed == args.stress_runs else "PARTIAL"
        print(f"{status} {label}: {completed}/{args.stress_runs} stress iterations", flush=True)


def duration_us(folder, device):
    samples = []
    for path in folder.rglob("OpBasicInfo*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("Task Duration(us)", "").strip():
                    continue
                name = row.get("Kernel Name") or row.get("Op Name", "")
                if not fnmatch.fnmatchcase(name, KERNEL) or int(row.get("Device Id", row.get("Device ID"))) != device:
                    raise ValueError(f"{path}: unexpected task {row}")
                samples.append(float(row["Task Duration(us)"]))
    if len(samples) != 1 or not math.isfinite(samples[0]) or samples[0] <= 0:
        raise ValueError(f"{folder}: expected one positive finite Eagle kernel duration, got {samples}")
    return samples[0]


def profile(args, case, scenario):
    msprof = shutil.which("msprof")
    if not msprof:
        raise RuntimeError("msprof is required for performance collection")
    args.output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="eagle-", dir=args.output)).resolve()
    print(f"msprof output: {folder}", flush=True)
    samples = []
    for run in range(1, args.repeats + 1):
        raw = folder / f"{case}-{scenario}-{run:02d}"
        command = [
            msprof,
            "op",
            "--warm-up=20",
            "--launch-count=1",
            "--kill=false",
            "--replay-mode=kernel",
            f"--aic-metrics={args.metrics}",
            f"--kernel-name={KERNEL}",
            f"--output={raw}",
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--case",
            case,
            "--scenario",
            scenario,
            "--device-id",
            str(args.device_id),
        ]
        subprocess.run(command, check=True, env={**os.environ, "TASK_QUEUE_ENABLE": "0"})
        samples.append(duration_us(raw, args.device_id))
    print(
        f"{case}/{scenario} n={len(samples)} min={min(samples)} median={statistics.median(samples)} max={max(samples)} us"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("accuracy", "profile", "worker"):
        sub = modes.add_parser(mode)
        sub.add_argument("--case", choices=SHAPES, required=mode == "worker", help="default: run all three cases")
        sub.add_argument("--device-id", type=int, default=0, help="NPU ID (same physical and visible ID for msprof)")
        sub.add_argument("--validate-only", action="store_true", help="print the selected plan without using a device")
        sub.add_argument(
            "--scenario",
            choices=("historical", "uniform", "normal"),
            default="historical" if mode == "worker" else None,
        )
        if mode == "accuracy":
            outputs = sub.add_mutually_exclusive_group()
            outputs.add_argument("--save-outputs", type=Path)
            outputs.add_argument("--compare-outputs", type=Path)
            sub.add_argument(
                "--mode",
                dest="phase",
                choices=("all", "accuracy", "stress"),
                default="all",
                help="Stress always validates its independent baseline first",
            )
            sub.add_argument(
                "--stress-runs", type=int, default=1000, help="calls after independent baseline validation"
            )
        if mode == "profile":
            sub.add_argument("--metrics", default="BasicInfo", choices=("BasicInfo", "MemoryL0", "PipeUtilization"))
            sub.add_argument("--repeats", type=int, default=10)
            sub.add_argument("--output", type=Path, default=Path("msprof_eagle"))
    args = parser.parse_args()
    if args.device_id < 0 or getattr(args, "stress_runs", 1) < 1 or getattr(args, "repeats", 1) < 1:
        parser.error("require device-id >= 0, stress-runs >= 1 and repeats >= 1")
    cases = [args.case] if args.case else list(CASES)
    scenarios = [args.scenario] if args.scenario else ["historical", "uniform", "normal"]
    if args.case == SHARED_CASE:
        if args.scenario not in (None, "historical"):
            parser.error("shared-k128-fp4 requires the shared ND Hadamard input")
        scenarios = ["historical"]
    if (getattr(args, "save_outputs", None) or getattr(args, "compare_outputs", None)) and args.case != SHARED_CASE:
        parser.error("output comparison requires --case shared-k128-fp4")
    for case in cases:
        for scenario in scenarios:
            if args.validate_only:
                print(f"{args.mode} {case}/{scenario}: M,N,K,dst={SHAPES[case]} device={args.device_id}")
            elif args.mode == "profile":
                profile(args, case, scenario)
            else:
                run_on_device(args, case, scenario)


if __name__ == "__main__":
    main()
