#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Synthetic NPU benchmark for model-independent FIR Resample1D."""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

CASES = ((1, 64, 1024), (1, 128, 4096), (2, 256, 2048))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--implementation", choices=("target", "reference", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device_id < 0 or args.seed < 0:
        parser.error("--device-id and --seed must be non-negative")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    return args


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _measure(function: Callable[[], torch.Tensor], warmup: int, repeats: int) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            function()
        torch.npu.synchronize()
        samples_us = []
        for _ in range(repeats):
            torch.npu.synchronize()
            begin = time.perf_counter_ns()
            result = function()
            torch.npu.synchronize()
            samples_us.append((time.perf_counter_ns() - begin) / 1000.0)
        del result
    return {
        "p50_us": statistics.median(samples_us),
        "p90_us": _percentile(samples_us, 0.90),
        "min_us": min(samples_us),
    }


def _reference(x: torch.Tensor, fir_filter: torch.Tensor) -> torch.Tensor:
    input_length = x.shape[2]
    output_index = torch.arange(2 * input_length, device=x.device)
    output = torch.zeros(
        (x.shape[0], x.shape[1], 2 * input_length),
        dtype=torch.float32,
        device=x.device,
    )
    tap_offset = (output_index + 1).remainder(2)
    for phase_tap in range(6):
        tap = 2 * phase_tap + tap_offset
        source_position = output_index + tap - 5
        source_index = torch.div(source_position, 2, rounding_mode="floor").clamp(0, input_length - 1)
        output += x.index_select(2, source_index) * fir_filter.index_select(0, tap)
    return output


def _run_case(
    shape: tuple[int, int, int],
    implementation: str,
    warmup: int,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    from mindiesd import fir_resample1d, is_fir_resample1d_supported

    generator = torch.Generator().manual_seed(sum(shape) + seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
    fir_filter = torch.linspace(-0.25, 0.3, 12, dtype=torch.float32, device=device)
    if implementation in ("target", "both") and not is_fir_resample1d_supported(x, fir_filter):
        raise RuntimeError("FIR Resample1D optimized backend is unavailable")

    def target() -> torch.Tensor:
        return fir_resample1d(x, fir_filter)

    def reference() -> torch.Tensor:
        return _reference(x, fir_filter)

    result: dict[str, object] = {"shape": list(shape)}
    if implementation in ("target", "both"):
        target_output = target()
        reference_output = reference()
        torch.npu.synchronize(device)
        absolute_error = (target_output - reference_output).abs()
        relative_error = absolute_error / reference_output.abs().clamp_min(torch.finfo(torch.float32).tiny)
        max_abs_error = float(absolute_error.max().cpu())
        max_rel_error = float(relative_error.max().cpu())
        torch.testing.assert_close(target_output, reference_output, rtol=1e-5, atol=1e-5)
        result["accuracy"] = {
            "passed": True,
            "rtol": 1e-5,
            "atol": 1e-5,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
        }
    if implementation in ("target", "both"):
        result["target"] = _measure(target, warmup, repeats)
    if implementation in ("reference", "both"):
        result["reference"] = _measure(reference, warmup, repeats)
    if implementation == "both":
        result["speedup"] = result["reference"]["p50_us"] / result["target"]["p50_us"]
    return result


def main() -> None:
    args = _parse_args()
    import torch_npu

    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU is unavailable")
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    results = [_run_case(shape, args.implementation, args.warmup, args.repeats, args.seed, device) for shape in CASES]
    payload = {
        "device": torch_npu.npu.get_device_name(args.device_id),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
