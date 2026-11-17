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
import csv
import math
from pathlib import Path
import re


def read_duration(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(path).read_text(encoding="utf-8"))
    labels = re.findall(r"Task\s+Duration\s*\(us\)\s*:\s*([^\r\n]*)", text)
    if len(labels) != 1 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", labels[0].strip()):
        raise ValueError(f"{path}: expected exactly one numeric Task Duration(us); found {labels!r}")
    duration = float(labels[0])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: Task Duration(us) must be finite and positive")
    return duration


def table(rows):
    widths = [max(len(str(row[col])) for row in rows) for col in range(len(rows[0]))]
    # Pad to the next 8-column tab stop before adding a separator tab.
    widths = [((width // 8) + 1) * 8 for width in widths]
    return [
        "".join(str(value).ljust(width - 1) + "\t" for value, width in zip(row[:-1], widths[:-1])) + str(row[-1])
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser(description="Strict msprof Task Duration extraction and final fused summary")
    parser.add_argument("--read-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device-id", type=int)
    args = parser.parse_args()
    if args.read_log:
        print(f"{read_duration(args.read_log):.12g}")
        return
    if args.output_dir is None or args.test_count <= 0 or args.warmup < 0:
        parser.error("--output-dir, positive --test-count and nonnegative --warmup required")
    root = args.output_dir.resolve()
    with (root / "capture_rows.tsv").open(encoding="utf-8", newline="") as stream:
        captures = list(csv.reader(stream, delimiter="\t"))
    if not captures:
        raise ValueError("No captures to summarize")
    scenarios = {}
    detail = [("Scenario", "Test", "Metric", "Duration(us)", "DataDir")]
    for row in captures:
        if len(row) != 4:
            raise ValueError(f"Invalid capture row: {row!r}")
        scenario, test, value, directory = row
        duration = read_duration(Path(directory) / "msprof.log")
        if not math.isclose(float(value), duration, rel_tol=1e-10):
            raise ValueError(f"{directory}: duration record differs from log")
        scenarios.setdefault(scenario, []).append((int(test), duration))
        detail.append((scenario, test, "Task Duration", f"{duration:.9g}", str(Path(directory).resolve())))
    minima = [("Scenario", "Metric", "Min(us)", "Max(us)", "Spread(%)")]
    warnings = []
    for scenario, samples in scenarios.items():
        if [index for index, _ in samples] != list(range(1, args.test_count + 1)):
            raise ValueError(f"{scenario}: incomplete, duplicate or unordered capture rounds")
        values = [value for _, value in samples]
        low, high = min(values), max(values)
        spread = (high - low) / low * 100
        minima.append(
            (scenario, "Task Duration", f"{low:.9g}", f"{high:.9g}", f"{spread:.2f}" if len(samples) > 1 else "N/A")
        )
        if len(samples) < 2:
            warnings.append(f"{scenario}: need at least two samples to assess variation.")
        elif spread > 10:
            warnings.append(
                f"WARNING {scenario}: spread exceeds 10%; consider MSPROF_WARMUP=50 or 100. "
                "Frequency ramp-up is a possible cause, not a confirmed diagnosis; "
                "compare Current Freq when available."
            )
    lines = (
        table(detail)
        + [""]
        + table(minima)
        + [
            f"Physical NPU ID={args.device_id}; warm-up={args.warmup} (default 20); "
            f"independent captures={args.test_count} (default 20).",
            "After changing warm-ups, collect a fresh complete set (default 20 rounds) on the same device and inputs "
            "in a new output directory. Keep all samples and warnings if variation persists.",
            "Note: Some hardware models may show increased latency across runs. The minimum across all groups for "
            "each scenario is reported as the best performance observed under the current test conditions; "
            "this does not mean the variation has been eliminated.",
            *warnings,
        ]
    )
    report = "\n".join(lines) + "\n"
    (root / "task_duration_summary.txt").write_text(report, encoding="utf-8")
    (root / "final_summary.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
