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

"""Analyze kernel_details.csv: copy kernels + top kernels + prev/next attribution.

compilation-dev Phase 7 Copy 消减检测：统计 InplaceCopy/ViewCopy/TensorMove/StridedSlice
的 count 与耗时占比，输出耗时 Top kernel，并给出每个 Copy kernel 的前后算子归因
（定位膨胀来源：VAE 3D→2D / aot_autograd _to_copy / QKV 重组等）。

Usage:
    python analyze_copy_kernels.py --csv <kernel_details.csv> [--label <名称>]
"""

import argparse
import csv
import sys
from collections import Counter

COPY_KEYS = ("InplaceCopy", "ViewCopy", "TensorMove", "StridedSlice")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze copy kernels and top kernels in a profile"
    )
    parser.add_argument("--csv", required=True, help="kernel_details.csv path")
    parser.add_argument("--label", default="profile", help="label for output")
    args = parser.parse_args()

    with open(args.csv, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    total_ms = sum(float(r["Duration(us)"]) for r in rows) / 1000
    print(f"=== {args.label}: {len(rows)} kernels, total {total_ms:.2f} ms ===")

    copy_rows = [r for r in rows if any(k in r["Name"] for k in COPY_KEYS)]
    copy_total = sum(float(r["Duration(us)"]) for r in copy_rows) / 1000
    copy_pct = copy_total / total_ms * 100 if total_ms else 0.0
    print(f"Copy kernels: {len(copy_rows)} count, {copy_total:.3f} ms ({copy_pct:.1f}%)")
    for k in COPY_KEYS:
        sub = [r for r in copy_rows if k in r["Name"]]
        if sub:
            sub_ms = sum(float(r["Duration(us)"]) for r in sub) / 1000
            print(f"  {k}: {len(sub)} count, {sub_ms:.3f} ms")

    by_name = Counter()
    for r in rows:
        by_name[r["Name"]] += float(r["Duration(us)"])
    print("\nTop 15 kernels by duration:")
    for name, dur in by_name.most_common(15):
        print(f"  {dur / 1000:9.2f} ms  {name[:80]}")

    rows_sorted = sorted(rows, key=lambda r: float(r["Start Time(us)"]))
    print("\nCopy attribution (prev -> copy -> next):")
    shown = 0
    for i, r in enumerate(rows_sorted):
        if not any(k in r["Name"] for k in COPY_KEYS):
            continue
        prev = rows_sorted[i - 1]["Name"].split("_")[-1][:40] if i else "-"
        nxt = rows_sorted[i + 1]["Name"].split("_")[-1][:40] if i < len(rows_sorted) - 1 else "-"
        dur_ms = float(r["Duration(us)"]) / 1000
        name = r["Name"].split("_")[-1][:20]
        print(f"  [{i}] {name:<20s} {dur_ms:7.2f}ms <- {prev:<40s} -> {nxt}")
        shown += 1
        if shown > 40:
            print("  ... (truncated)")
            break


if __name__ == "__main__":
    sys.exit(main())
