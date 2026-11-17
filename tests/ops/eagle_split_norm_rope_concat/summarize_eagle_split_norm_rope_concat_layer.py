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

"""Summarize complete-call Event samples without importing torch or loading operators."""

import argparse
import json
import math
from pathlib import Path

from eagle_split_norm_rope_concat_cases import CASE_IDS
from eagle_split_norm_rope_concat_layer_common import PATHS, PROTOCOL, input_config, scenario_name


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Expected positive finite duration, got {value!r}")
    return value


def validate(report, path, case_id, reference, profiled):
    if report["protocol"] != PROTOCOL or not isinstance(report["profiled"], bool) or report["profiled"] is not profiled:
        raise ValueError("Timing protocol/profiling mode mismatch")
    if (
        report["path"] != path
        or report["case_id"] != case_id
        or report["scenario"] != scenario_name(path, case_id)
        or report["distribution"] != "normal"
    ):
        raise ValueError("Unexpected path/case/scenario/distribution")
    config = input_config(case_id)
    if report["input_config"] != config:
        raise ValueError(f"Input configuration mismatch: expected {config}, got {report['input_config']}")
    if report["warmup"] != 5:
        raise ValueError("Expected fixed warmup=5")
    iterations = report["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(report["device_id"], bool) or not isinstance(report["device_id"], int) or report["device_id"] < 0:
        raise ValueError("Invalid physical device ID")
    if isinstance(report["seed"], bool) or not isinstance(report["seed"], int) or not 0 <= report["seed"] < 2**32:
        raise ValueError("Invalid seed")
    # Input shape/stride vary by cache width and are validated against this case below.
    for key in ("device_id", "seed", "iterations", "warmup", "runtime"):
        if report[key] != reference[key]:
            raise ValueError(f"Inconsistent {key} across captures")
    width = config["rope_shapes"][0][-1]
    expected = [
        ([1, 5672, 168, 128], "torch.bfloat16", [5672 * 168 * 128, 168 * 128, 128, 1]),
        ([128], "torch.bfloat16", [1]),
        ([128], "torch.bfloat16", [1]),
        ([1, 5672, 1, width], "torch.bfloat16", [5672 * width, width, width, 1]),
        ([1, 5672, 1, width], "torch.bfloat16", [5672 * width, width, width, 1]),
        (None, None, None),
    ]
    if len(report["actual_inputs"]) != len(expected):
        raise ValueError("Unexpected input count")
    for index, (state, (shape, dtype, stride)) in enumerate(zip(report["actual_inputs"], expected)):
        if index == 5:
            expected_state = {"type": "int[]", "value": config["head_counts"]}
        else:
            expected_state = {
                "shape": shape,
                "dtype": dtype,
                "stride": stride,
                "device": f"npu:{report['device_id']}",
                "contiguous": True,
            }
        if state != expected_state:
            raise ValueError(f"Actual input {index} does not match production contract: {state}")
    if report["output_count"] != 3:
        raise ValueError("Output boundary mismatch")
    values = [positive(value) for value in report["event_samples_us"]]
    if (
        isinstance(report["samples"], bool)
        or not isinstance(report["samples"], int)
        or report["samples"] != iterations
        or len(values) != iterations
    ):
        raise ValueError("Event sample count mismatch")
    if positive(report["event_us"]) != min(values):
        raise ValueError("event_us must be the minimum of this capture's individual samples")


def capture_profile_dir(capture, profiled):
    profiles = [p for p in capture.glob("PROF_*") if p.is_dir()]
    if not profiled:
        if profiles:
            raise ValueError(f"Direct mode cannot contain PROF directories: {capture}")
        return None
    if len(profiles) != 1:
        raise ValueError(f"Expected exactly one PROF_* in {capture}")
    for pattern in ("op_summary*.csv", "msprof_*.json"):
        if not any(p.is_file() and p.stat().st_size for p in profiles[0].rglob(pattern)):
            raise ValueError(f"Missing nonempty {pattern} under {profiles[0]}")
    return str(profiles[0].resolve())


def summarize(output, runs, profiled, paths=PATHS, cases=CASE_IDS):
    rows = []
    reference = None
    for path in paths:
        for case_id in cases:
            for run in range(1, runs + 1):
                capture = output / case_id / "normal" / f"run_{run:02d}" / path
                filename = "profiled_timing.json" if profiled else "timing.json"
                other = "timing.json" if profiled else "profiled_timing.json"
                if (capture / other).exists():
                    raise ValueError(f"Mixed profiling modes in {capture}")
                timing_file = capture / filename
                report = json.loads(timing_file.read_text(encoding="utf-8"))
                if reference is None:
                    reference = report
                validate(report, path, case_id, reference, profiled)
                samples_csv = capture / "event_samples.csv"
                if not samples_csv.is_file() or samples_csv.stat().st_size == 0:
                    raise ValueError(f"Missing Event samples: {samples_csv}")
                profile_dir = capture_profile_dir(capture, profiled)
                rows.append(
                    {
                        "path": path,
                        "case_id": case_id,
                        "distribution": "normal",
                        "scenario": scenario_name(path, case_id) + "/normal",
                        "run": run,
                        "event_us": report["event_us"],
                        "event_samples_us": report["event_samples_us"],
                        "profile_dir": profile_dir,
                        "data_dir": profile_dir or str(capture.resolve()),
                        "timing_file": str(timing_file.resolve()),
                    }
                )
    best = [
        min((r for r in rows if r["path"] == path and r["case_id"] == case_id), key=lambda r: r["event_us"]).copy()
        for path in paths
        for case_id in cases
    ]
    return {
        "profiled": profiled,
        "protocol": PROTOCOL,
        "runs": runs,
        "warmup": 5,
        "iterations": reference["iterations"],
        "device_id": reference["device_id"],
        "seed": reference["seed"],
        "input_configs": {case_id: input_config(case_id) for case_id in cases},
        "paths": list(paths),
        "cases": list(cases),
        "rows": rows,
        "best_results": best,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--no-msprof", action="store_true")
    parser.add_argument("--paths", choices=PATHS, nargs="+", default=PATHS)
    parser.add_argument("--cases", choices=CASE_IDS, nargs="+", default=CASE_IDS)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    if len(set(args.paths)) != len(args.paths):
        parser.error("paths must not contain duplicates")
    if len(set(args.cases)) != len(args.cases):
        parser.error("cases must not contain duplicates")
    result = summarize(args.output.resolve(), args.runs, not args.no_msprof, args.paths, args.cases)
    mode = "under msprof" if result["profiled"] else "without msprof"
    width = max(len(row["scenario"]) for row in result["rows"])
    lines = [
        f"Complete-call NPU Event timings {mode}; physical NPU={result['device_id']}",
        f"Fixed warmup=5; {result['iterations']} samples per capture; {args.runs} captures per scenario; hot cache.",
        f"{'Scenario':{width}} Test  Layer Duration(us)  DataDir",
    ]
    for row in result["rows"]:
        lines.append(f"{row['scenario']:{width}} {row['run']:4} {row['event_us']:19.6f}  {row['data_dir']}")
    lines.append(f"Performance results: minimum of {args.runs} complete groups per scenario (us)")
    for row in result["best_results"]:
        lines.append(f"{row['scenario']:{width}} {row['event_us']:.6f}")
    lines.append(
        "Note: Some hardware models may show increased latency across runs. "
        "The minimum across all groups for each scenario is reported as the best performance observed under "
        "the current test conditions; this does not mean the variation has been eliminated."
    )
    rendered = "\n".join(lines) + "\n"
    (args.output / "summary.txt").write_text(rendered, encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
