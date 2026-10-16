#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Summarize minimum full-layer Event timings, grouped by path."""

import argparse
import json
import math
from pathlib import Path


SCENARIOS = {
    "current": "CastV(BF16->FP16)+LA(BF16-out)",
    "legacy": "CastQKV(BF16->FP16)+LA(FP32-out)+CastOut(FP32->FP16->BF16)",
}


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Expected finite positive timing, got {value!r}")
    return number


def validate_report(report, path, reference):
    if not isinstance(report["profiled"], bool) or report["profiled"] is not reference["profiled"]:
        raise ValueError("Inconsistent profiling mode")
    if set(report["results"]) != {path} or report["path_order"] != [path]:
        raise ValueError("Expected one path per run")
    if report["warmup"] != 5:
        raise ValueError("Expected fixed warmup=5")
    iterations = report["iterations"]
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("Expected positive integer iterations")
    for name in ("device_id", "warmup", "iterations", "seed", "layout", "dtype", "q_shape", "kv_shape"):
        if report[name] != reference[name]:
            raise ValueError(f"Inconsistent {name}: {report[name]!r} != {reference[name]!r}")
    timing = report["results"][path]
    values = [positive(value) for value in timing["event_samples_us"]]
    if timing["samples"] != iterations or len(values) != iterations:
        raise ValueError("Event sample count does not match iterations")
    if positive(timing["event_us"]) != min(values):
        raise ValueError("event_us must equal the minimum of Event samples")


def summarize(output, runs, profiled=True):
    rows = []
    reference = None
    for path in ("current", "legacy"):
        for index in range(1, runs + 1):
            capture = output / f"run_{index:02d}" / path
            timing_file = capture / ("profiled_timing.json" if profiled else "timing.json")
            report = json.loads(timing_file.read_text(encoding="utf-8"))
            if report["profiled"] is not profiled:
                raise ValueError(f"Unexpected profiling mode in {timing_file}")
            if reference is None:
                reference = report
            validate_report(report, path, reference)
            profile_dir = None
            if profiled:
                profiles = sorted(item for item in capture.glob("PROF_*") if item.is_dir())
                if len(profiles) != 1:
                    raise ValueError(f"Expected one PROF_* directory in {capture}, found {len(profiles)}")
                profile_dir = str(profiles[0].absolute())
            timing = report["results"][path]
            rows.append(
                {
                    "run": index,
                    "path": path,
                    "scenario": SCENARIOS[path],
                    "event_us": timing["event_us"],
                    "event_samples_us": timing["event_samples_us"],
                    "profile_dir": profile_dir,
                    "data_dir": profile_dir or str(capture.absolute()),
                    "timing_file": str(timing_file.absolute()),
                }
            )
    best_results = [
        min((row for row in rows if row["path"] == path), key=lambda row: row["event_us"]).copy() for path in SCENARIOS
    ]
    return {
        "runs": runs,
        "device_id": reference["device_id"],
        "warmup": reference["warmup"],
        "iterations": reference["iterations"],
        "profiled": profiled,
        "reduction": "min",
        "rows": rows,
        "best_results": best_results,
    }


def format_summary(report):
    mode = "under msprof" if report["profiled"] else "without msprof"
    width = max(map(len, SCENARIOS.values()))
    lines = [
        f"=== LaserAttention layer Event summary (device_id={report['device_id']}) ===",
        f"Fixed warmup=5; each result=min of {report['iterations']} full-layer Event timings {mode} (us).",
        "Both scenarios: BF16 QKV input/output; includes layout conversion, padding and output cropping.",
        f"{'Scenario':{width}} Test    Layer Duration(us)  DataDir",
    ]
    for row in report["rows"]:
        lines.append(f"{row['scenario']:{width}} {row['run']:4} {row['event_us']:21.6f}  {row['data_dir']}")
    lines.append(f"=== Performance results: minimum of {report['runs']} groups per scenario (us) ===")
    for row in report["best_results"]:
        lines.append(f"{row['scenario']:{width}} {row['event_us']:.6f}")
    lines.append(
        "Note: Some hardware models may show increased latency across runs. "
        "The minimum across all groups for each scenario is reported as the best performance "
        "observed under the current test conditions; this does not mean the variation has been eliminated."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--no-msprof", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    report = summarize(args.output, args.runs, profiled=not args.no_msprof)
    rendered = format_summary(report)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.txt").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
