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
import hashlib
import math
from datetime import datetime
from pathlib import Path
import tempfile

import torch

from tests.ops.eagle_split_norm_rope_concat.eagle_split_norm_rope_concat_accuracy_common import (
    PROD_HEADS,
    PROD_SHAPE,
    call_dut,
    compare,
    golden,
    init_device,
    make_inputs,
)
from tests.ops.eagle_split_norm_rope_concat.eagle_split_norm_rope_concat_cases import (
    ASYMMETRIC_CASE,
    BASE_CASE_IDS,
    CASE_IDS,
    accuracy_distributions,
    case_attrs,
    case_geometry,
    is_compact,
)


def input_fingerprints(inputs):
    fingerprints = []
    for value in inputs:
        data = (
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            if isinstance(value, torch.Tensor)
            else repr(value).encode("utf-8")
        )
        fingerprints.append(hashlib.sha256(data).hexdigest())
    return tuple(fingerprints)


def check_output_contract(outputs, inputs, case_id):
    if len(outputs) != 3:
        raise AssertionError(f"Expected q,k,v outputs, got {len(outputs)}")
    batch, seq, _, dim = inputs[0].shape
    shapes = tuple(
        (batch, heads, seq, dim) if case_attrs(case_id)["output_bnsd"] else (batch, seq, heads, dim)
        for heads in inputs[5]
    )
    for name, output, shape in zip(("q", "k", "v"), outputs, shapes):
        if (
            tuple(output.shape) != shape
            or output.dtype != inputs[0].dtype
            or output.device != inputs[0].device
            or not output.is_contiguous()
        ):
            raise AssertionError(
                f"{name}: invalid output contract: shape={output.shape} "
                f"dtype={output.dtype} device={output.device} stride={output.stride()}"
            )


def report_stress_failure(name, baseline, current, iteration, distribution, case_id, output_dir):
    changed = baseline.view(torch.int16) != current.view(torch.int16)
    a, g = current.float().flatten(), baseline.float().flatten()
    finite = torch.isfinite(a) & torch.isfinite(g)
    diff = (a - g).abs()
    finite_diff = torch.where(finite, diff, torch.zeros_like(diff))
    max_abs = float(finite_diff.max().item())
    max_rel = float((finite_diff / g.abs().clamp_min(1e-12)).max().item())
    count = int(changed.sum().item())
    print(
        f"FAIL stress case={case_id} distribution={distribution} iteration={iteration} output={name} "
        f"changed={count}/{baseline.numel()} max_abs={max_abs} max_rel={max_rel} "
        f"nonfinite={int((~finite).sum().item())}",
        flush=True,
    )
    # Include nonfinite locations first; signed-zero changes remain visible in the bit counts.
    scores = torch.where(finite, diff, torch.full_like(diff, float('inf')))
    scores.masked_fill_(~changed.flatten(), float('-inf'))
    for index in torch.topk(scores, min(8, count)).indices.tolist():
        print(f"  flat_index={index} baseline={float(g[index])} current={float(a[index])}")
    log_root = output_dir
    log_root.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(tempfile.mkdtemp(prefix=f"stress-{case_id}-{distribution}-{iteration}-", dir=log_root))
    torch.save(
        {
            "output": name,
            "iteration": iteration,
            "distribution": distribution,
            "case": case_id,
            "baseline": baseline,
            "current": current,
        },
        dump_dir / "mismatch.pt",
    )
    print(f"Stress mismatch saved to {dump_dir / 'mismatch.pt'}", flush=True)
    raise AssertionError(f"{name}: stress output is not bit-exact at iteration {iteration}")


def _run_stress(inputs, runs, distribution, before, case_id, output_dir):
    attrs = case_attrs(case_id)
    rope, layout = case_id.split("-")[:2]
    description = (
        f"shape={tuple(inputs[0].shape)} heads={tuple(inputs[5])} rope={rope} "
        f"output={layout} cos/sin_width={inputs[3].shape[-1]} R={attrs['rotary_dim']} dtype=bf16"
    )
    outputs = call_dut(inputs, **attrs)
    torch.npu.synchronize()
    check_output_contract(outputs, inputs, case_id)
    # Independent CPU copies cannot be overwritten by a later DUT invocation.
    baseline = tuple(x.detach().cpu().clone() for x in outputs)
    del outputs
    for name, value in zip(("q", "k", "v"), baseline):
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{name}: nonfinite stress baseline")
    if input_fingerprints(inputs) != before:
        raise AssertionError("Stress inputs changed during the baseline call")
    print(f"Stress {description} distribution={distribution} runs={runs} baseline_calls=1 policy=bit_exact", flush=True)
    for iteration in range(1, runs + 1):
        outputs = torch.ops.mindiesd.eagle_split_norm_rope_concat(*inputs, **attrs)
        torch.npu.synchronize()
        check_output_contract(outputs, inputs, case_id)
        for name, output, reference in zip(("q", "k", "v"), outputs, baseline):
            current = output.detach().cpu()
            if not torch.equal(current.view(torch.int16), reference.view(torch.int16)):
                if input_fingerprints(inputs) != before:
                    raise AssertionError(f"Stress inputs changed at iteration {iteration}")
                report_stress_failure(name, reference, current, iteration, distribution, case_id, output_dir)
            del current, output
        del outputs
        if iteration % 100 == 0 or iteration == runs:
            print(f"Stress {description} distribution={distribution} completed={iteration}/{runs}", flush=True)
    if input_fingerprints(inputs) != before:
        raise AssertionError("Stress inputs changed; output stability result is invalid")
    print(f"PASS stress {description} distribution={distribution} runs={runs} outputs=q,k,v bit_exact", flush=True)


def run_stress(inputs, runs, distribution, case_id, output_dir):
    before = input_fingerprints(inputs)
    try:
        _run_stress(inputs, runs, distribution, before, case_id, output_dir)
    except (AssertionError, RuntimeError) as error:
        if input_fingerprints(inputs) != before:
            raise AssertionError("Stress inputs changed; output stability result is invalid") from error
        raise


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mode", choices=("all", "accuracy", "stress"), default="all")
    parser.add_argument(
        "--distribution",
        choices=("all", "uniform", "normal"),
        default="all",
        help="all selects assigned distributions: original eight plus asymmetric normal and uniform (ten on Ascend950)",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("B", "S", "Htotal", "D"),
        help="Actual BSND input shape (default: 1 5672 168 128); D must be 128",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs=3,
        metavar=("Q", "K", "V"),
        help="Override head counts; Q+K+V=Htotal. Ascend950: each branch in [1,128]",
    )
    parser.add_argument("--case", choices=CASE_IDS, help="Select one named production case")
    parser.add_argument("--rope-type", choices=("all", "half", "interleave"), default="all")
    parser.add_argument(
        "--cache-width", choices=("all", "full", "half"), default="all", help="cos/sin width: full=R, half=R/2"
    )
    parser.add_argument("--rotary-dim", choices=("all", "96", "128"), default="all")
    parser.add_argument("--output-layout", choices=("all", "bsnd", "bnsd"), default="all")
    parser.add_argument(
        "--output",
        type=Path,
        help="Result directory (default: repository log/eagle_split_norm_rope_concat/accuracy_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--stress-runs",
        type=int,
        default=1000,
        help="Repeated DUT calls per distribution, excluding the baseline (default: 1000)",
    )
    args = parser.parse_args()
    custom_geometry = args.shape is not None or args.heads is not None
    if custom_geometry:
        args.shape = tuple(args.shape or PROD_SHAPE)
        default_heads = (1, 128, 128) if args.shape == (1, 5672, 257, 128) else PROD_HEADS
        args.heads = tuple(args.heads or default_heads)
    else:
        args.shape, args.heads = PROD_SHAPE, PROD_HEADS
    if args.device_id < 0:
        parser.error("--device-id must be nonnegative")
    if any(value <= 0 or value > 2147483647 for value in (*args.shape, *args.heads)):
        parser.error("--shape and --heads values must be positive int32 integers")
    if args.shape[3] != 128:
        parser.error("--shape D must be 128")
    if math.prod(args.shape) > 2147483647:
        parser.error("--shape numel must not exceed 2147483647")
    if sum(args.heads) != args.shape[2]:
        parser.error("--heads must satisfy Q+K+V=Htotal")
    candidates = CASE_IDS if not custom_geometry else BASE_CASE_IDS
    if custom_geometry and (args.shape, args.heads) == case_geometry(ASYMMETRIC_CASE):
        candidates = (ASYMMETRIC_CASE,)
    if args.case is not None:
        candidates = (args.case,)
    cases = tuple(
        case_id
        for case_id in candidates
        if args.rope_type in ("all", case_id.split("-", maxsplit=1)[0])
        and args.output_layout in ("all", case_id.split("-", maxsplit=2)[1])
        and (args.rotary_dim == "all" or case_attrs(case_id)["rotary_dim"] == int(args.rotary_dim))
        and (args.cache_width == "all" or is_compact(case_id) == (args.cache_width == "half"))
    )
    if not cases:
        parser.error(
            "No selected representative scenario; relax RoPE/layout/cache/R filters (preview with shell --dry-run)"
        )
    if args.stress_runs <= 0:
        parser.error("--stress-runs must be a positive integer")
    if not __debug__:
        parser.error("Run without Python optimization: accuracy checks require assertions")
    output_dir = (
        args.output.resolve()
        if args.output
        else (
            Path(__file__).resolve().parents[3]
            / "log"
            / "eagle_split_norm_rope_concat"
            / datetime.now().strftime("accuracy_%Y%m%d_%H%M%S")
        )
    )
    device = init_device(args.device_id)
    if torch.npu.get_soc_version() != 260:
        if args.case == ASYMMETRIC_CASE:
            parser.error("The asymmetric production case requires Ascend950")
        cases = tuple(key for key in cases if key != ASYMMETRIC_CASE)
        if not cases:
            parser.error("No selected case is supported on this device")
    stages = ("accuracy", "stress") if args.mode == "all" else (args.mode,)
    for stage in stages:
        for case_id in cases:
            shape, heads = (args.shape, args.heads) if custom_geometry else case_geometry(case_id)
            attrs = case_attrs(case_id)
            distributions = accuracy_distributions(case_id) if args.distribution == "all" else (args.distribution,)
            for distribution in distributions:
                rope, layout = case_id.split("-")[:2]
                rotary_dim = attrs["rotary_dim"]
                width = rotary_dim // 2 if is_compact(case_id) else rotary_dim
                description = (
                    f"shape={tuple(shape)} heads={tuple(heads)} "
                    f"rope={rope} output={layout} cos/sin_width={width} R={rotary_dim} dtype=bf16"
                )
                print(f"START {stage} {description} distribution={distribution}", flush=True)
                inputs = make_inputs(
                    tuple(shape),
                    tuple(heads),
                    rotary_dim,
                    torch.bfloat16,
                    device,
                    distribution,
                    rope_rank=4,
                    compact=is_compact(case_id),
                )
                if stage == "stress":
                    run_stress(inputs, args.stress_runs, distribution, case_id, output_dir)
                else:
                    actual = call_dut(inputs, **attrs)
                    check_output_contract(actual, inputs, case_id)
                    expected = golden(inputs, **attrs)  # All reference operations execute on this NPU.
                    compare(actual, expected, large=True)
                    assert torch.equal(actual[2].view(torch.int16), expected[2].view(torch.int16)), "V bits changed"
                    print(f"PASS accuracy {description} distribution={distribution}; mixed tolerance diagnostic only")
                    del actual, expected
                del inputs


if __name__ == "__main__":
    main()
