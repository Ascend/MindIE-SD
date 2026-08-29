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

"""Compare kernel classes between two profiles (eager vs compile).

compilation-dev Phase 6/7 kernel diff 的轻量入口：按算子类聚合 count 与耗时，
快速确认 pattern 融合是否命中（融合 kernel 出现 / 原始 kernel 消失）。

Usage:
    python cmp_kernels.py --eager eager/kernel_details.csv --compile compile/kernel_details.csv
"""

import argparse
import collections
import csv
import sys

# 顺序即匹配优先级：更特化的关键词在前（如 AdaLayerNorm 必须先于 LayerNorm，
# 否则子串包含导致重复计入）。每个 kernel 只归入第一个命中的桶。
KEYWORDS = (
    "AdaLayerNorm",
    "LayerNorm",
    "RmsNorm",
    "residual_gate",
    "Rotary",
    "Gelu",
    "Dropout",
    "BroadcastTo",
    "Mul",
    "Add",
)


def load_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def summarize(rows, label):
    print(f"=== {label}: {len(rows)} kernels ===")
    bucket = collections.Counter()
    bucket_ms = collections.Counter()
    for r in rows:
        name = r["Name"]
        for key in KEYWORDS:
            if key in name:
                bucket[key] += 1
                try:
                    bucket_ms[key] += float(r["Duration(us)"])
                except (KeyError, ValueError):
                    pass
                break
    for key in KEYWORDS:
        if bucket[key]:
            total_ms = bucket_ms[key] / 1000
            print(f"  {key}: {bucket[key]} count, {total_ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Compare kernel classes between two profiles")
    parser.add_argument("--eager", required=True, help="eager kernel_details.csv path")
    parser.add_argument("--compile", required=True, help="compile kernel_details.csv path")
    args = parser.parse_args()
    summarize(load_rows(args.eager), "eager")
    summarize(load_rows(args.compile), "compile")


if __name__ == "__main__":
    sys.exit(main())
