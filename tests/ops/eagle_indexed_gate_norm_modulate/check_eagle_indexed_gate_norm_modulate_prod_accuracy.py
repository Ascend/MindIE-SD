#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse

import torch

from eagle_indexed_gate_norm_modulate_accuracy_common import (
    DTYPES,
    NAMES,
    OUTPUT_NAMES,
    PROD_SCENARIOS,
    PROD_SHAPES,
    call_dut,
    compare,
    golden,
    init_device,
    make_inputs,
    scenario_name,
)


def check_unchanged(inputs, snapshots, label):
    for name, value, saved in zip(NAMES, inputs, snapshots):
        bits = value.view(torch.int16) if value.is_floating_point() else value
        saved_bits = saved.view(torch.int16) if saved.is_floating_point() else saved
        assert torch.equal(bits, saved_bits), f"{label}: input {name} mutated; stability result invalid"


def stress(inputs, output_dtype, runs, label):
    snapshots = tuple(x.clone() for x in inputs)
    snapshot_bytes = sum(x.numel() * x.element_size() for x in snapshots)
    print(f"{label}: NPU input snapshot bytes={snapshot_bytes}; two independent output baselines also stay on NPU")
    try:
        initial = call_dut(inputs, output_dtype)
        torch.npu.synchronize()
        check_unchanged(inputs, snapshots, f"{label} after baseline")
        assert len(initial) == 2
        for name, ref in zip(OUTPUT_NAMES, initial):
            assert ref.shape == inputs[0].shape and ref.dtype == output_dtype and ref.is_contiguous(), name
            assert bool(torch.isfinite(ref).all()), f"{label}: baseline {name} nonfinite"
        baseline = tuple(x.clone() for x in initial)
        del initial
        for run in range(1, runs + 1):
            actual = call_dut(inputs, output_dtype, log=False)
            torch.npu.synchronize()
            assert len(actual) == 2, f"{label} run={run}: output count={len(actual)}"
            for name, value, ref in zip(OUTPUT_NAMES, actual, baseline):
                assert value.shape == ref.shape and value.dtype == ref.dtype and value.is_contiguous(), (
                    label,
                    run,
                    name,
                    value.shape,
                    value.dtype,
                    value.stride(),
                )
                unequal = value.view(torch.int16) != ref.view(torch.int16)
                count = int(unequal.sum().item())
                if count:
                    diff = (value.float() - ref.float()).abs().flatten()
                    position = int(diff.argmax().item())
                    relative = diff / ref.float().flatten().abs().clamp_min(torch.finfo(torch.float32).tiny)
                    raise AssertionError(
                        f"{label} run={run} output={name} different_bits={count}/{value.numel()} "
                        f"ratio={count / value.numel():.9g} max_abs={diff.max().item()} "
                        f"max_rel={relative.max().item()} flat_index={position} "
                        f"coordinate={divmod(position, ref.shape[-1])} "
                        f"baseline={ref.flatten()[position].item()} current={value.flatten()[position].item()}"
                    )
            del actual
            if run % 100 == 0 or run == runs:
                print(f"PROGRESS stress {label} completed={run}/{runs}", flush=True)
        check_unchanged(inputs, snapshots, f"{label} after stress")
    except Exception:
        check_unchanged(inputs, snapshots, f"{label} on failure")
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Large ND accuracy (NPU FP32 composition) and deterministic stress; "
            "10 representative cases including distribution, filters select from the README matrix"
        )
    )
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--mode", choices=("all", "accuracy", "stress"), default="all")
    parser.add_argument("--stress-runs", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, choices=(2048, 5376, 6656, 7168))
    parser.add_argument("--rows", type=int, choices=PROD_SHAPES)
    parser.add_argument("--input-dtype", choices=DTYPES)
    parser.add_argument("--output-dtype", choices=DTYPES)
    args = parser.parse_args()
    if args.stress_runs <= 0 or args.device_id < 0:
        parser.error("--stress-runs must be positive and --device-id nonnegative")
    scenarios = [
        (n, k, d, i, o, distribution)
        for n, k, d, i, o, distribution in PROD_SCENARIOS
        if (args.hidden_size is None or args.hidden_size == d)
        and (args.rows is None or args.rows == n)
        and (args.input_dtype is None or args.input_dtype == i)
        and (args.output_dtype is None or args.output_dtype == o)
    ]
    if not scenarios:
        parser.error("No selected representative case matches these filters; see README for the case matrix")
    device = init_device(args.device_id)
    completed = []
    modes = ("accuracy", "stress") if args.mode == "all" else (args.mode,)
    for mode in modes:
        for rows, indexed_rows, hidden, input_name, output_name, distribution in scenarios:
            label = scenario_name(hidden, input_name, output_name, rows, indexed_rows).replace(
                "_normal", f"_{distribution}"
            )
            print(f"START {mode} {label}", flush=True)
            inputs = make_inputs(rows, hidden, DTYPES[input_name], device, distribution, indexed_rows=indexed_rows)
            if mode == "accuracy":
                print("golden=NPU FP32 elementwise/reduction composition; mixed tolerance is soft")
                expected = golden(inputs, DTYPES[output_name])
                actual = call_dut(inputs, DTYPES[output_name])
                compare(actual, expected, large=True)
                del expected, actual
            else:
                stress(inputs, DTYPES[output_name], args.stress_runs, label)
            completed.append(f"PASS {mode} {label}" + (f" runs={args.stress_runs}" if mode == "stress" else ""))
            del inputs
    print("\n".join(completed))


if __name__ == "__main__":
    main()
