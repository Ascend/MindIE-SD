#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Validate each msprof op duration and publish one complete performance report."""

import argparse
import math
import re
from pathlib import Path


def duration(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8", errors="replace"))
    matches = re.findall(r"Task\s+Duration\s*\(us\)\s*:\s*([^\r\n]+)", text)
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one Task Duration(us), got {len(matches)}")
    value = float(matches[0].strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: expected positive finite duration, got {value}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scene", action="append", help="Selected scene label; repeat for multiple layouts")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.check_log:
        duration(args.check_log)
        return
    if not args.output_dir or not args.scene or args.count <= 0 or args.warmup < 0:
        parser.error("require output-dir, scene, count>0 and warmup>=0")
    rows, summaries, notes = [("Scenario", "Test", "Metric", "Duration(us)", "DataDir")], [], []
    for scene in args.scene:
        for distribution in ("uniform", "normal"):
            values = []
            for index in range(1, args.count + 1):
                directory = (args.output_dir / scene / distribution / f"run_{index:02d}").resolve()
                value = duration(directory / "msprof.log")
                values.append(value)
                rows.append((f"{scene}/{distribution}", str(index), "Task Duration", f"{value:.6f}", str(directory)))
            low, high = min(values), max(values)
            summaries.append(f"MIN {scene}/{distribution} Task Duration(us)={low:.6f}")
            spread = (high - low) / low * 100
            notes.append(
                f"{scene}/{distribution}: warm-up={args.warmup}, captures={len(values)}, "
                f"max_us={high:.6f}, spread={spread:.2f}%"
            )
            if len(values) < 2:
                notes.append("WARNING: need at least two captures to assess spread.")
            elif spread > 10:
                notes.append(
                    "WARNING: spread exceeds 10%; consider MSPROF_WARMUP=50 or 100 and recapture "
                    "a complete set (default 20) on the same card/inputs in a new directory. "
                    "Frequency ramp-up is a possible cause; compare Current Freq when available."
                )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    table = ["\t".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows]
    report = (
        "\n".join(
            table
            + summaries
            + notes
            + [
                "Defaults: 20 warm-ups and 20 independent captures per layout/distribution.",
                "Note: Some hardware models may show increased latency across runs. The minimum across all groups "
                "for each scenario is reported as the best performance observed under the current test conditions; "
                "this does not mean the variation has been eliminated.",
            ]
        )
        + "\n"
    )
    for name in ("task_duration_summary.txt", "final_summary.txt"):
        (args.output_dir / name).write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
