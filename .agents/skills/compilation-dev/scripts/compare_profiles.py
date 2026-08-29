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

"""Compare two kernel CSVs (eager vs compile) with family rollups and per-kernel diff.

compilation-dev Phase 6 的 kernel diff 方法论文档化实现：按算子族聚合耗时、
逐 kernel 输出 delta，定位编译开销来源（同名 kernel 耗时差 / 新增 kernel）。

Usage:
    python compare_profiles.py --eager eager/kernel_details.csv --compile compile/kernel_details.csv
"""

import argparse
import collections
import csv
import sys

FAMILY = [
    ("FlashAttention", lambda n: "FlashAttention" in n),
    ("Addmm/MatMul", lambda n: "Addmm" in n or "MatMul" in n or "Gemm" in n or "BatchMatMul" in n),
    # 细分桶必须先于 "InplaceCopy(总)" 匹配，否则总桶吞掉细分统计（死代码）
    ("  ViewCopy", lambda n: "ViewCopy" in n),
    ("  TensorMove", lambda n: "TensorMove" in n),
    ("  TransposeCopy", lambda n: "InplaceCopy" in n and "Transpose" in n),
    ("  Cast", lambda n: "InplaceCopy" in n and "Cast" in n),
    ("InplaceCopy(总)", lambda n: "InplaceCopy" in n),
    ("Transpose(布局)", lambda n: "Transpose" in n and "InplaceCopy" not in n),
    ("Mul", lambda n: n.startswith("aclnnMul") or n.startswith("aclnnMuls")),
    ("Add", lambda n: n.startswith("aclnnAdd") or n.startswith("aclnnAdds")),
    ("LayerNorm", lambda n: "LayerNorm" in n and "AdaLayerNorm" not in n),
    ("RMSNorm", lambda n: "RmsNorm" in n or "rms_norm" in n),
    ("Pow/Mean/Rsqrt", lambda n: "Pow" in n or "Mean" in n or "Rsqrt" in n or "Square" in n),
    ("GELU", lambda n: "Gelu" in n),
    ("RoPE", lambda n: "Rotary" in n),
    ("Dropout", lambda n: "Dropout" in n),
    ("其他", lambda n: True),
]


def load(path):
    name_dur = collections.Counter()
    name_cnt = collections.Counter()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            n = row.get("Name", "")
            try:
                d = float(row.get("Duration(us)", 0))
            except ValueError:
                continue
            name_dur[n] += d
            name_cnt[n] += 1
    return name_dur, name_cnt


def main(eager_path, compile_path):
    ed, ec = load(eager_path)
    cd, cc = load(compile_path)
    et = sum(ed.values()) / 1000
    ct = sum(cd.values()) / 1000
    print(f"eager total  = {et:9.1f} ms ({sum(ec.values())} kernels)")
    print(f"compile total= {ct:9.1f} ms ({sum(cc.values())} kernels)")
    if et > 0:
        print(f"delta        = {ct - et:+9.1f} ms ({ct / et:.3f}x)")
    else:
        print(f"delta        = {ct - et:+9.1f} ms (eager total is 0)")

    print("\n--- family rollup (ms) ---")
    efam_d, efam_c = collections.Counter(), collections.Counter()
    cfam_d, cfam_c = collections.Counter(), collections.Counter()
    for n, d in ed.items():
        for label, fn in FAMILY:
            if fn(n):
                efam_d[label] += d / 1000
                efam_c[label] += ec[n]
                break
    for n, d in cd.items():
        for label, fn in FAMILY:
            if fn(n):
                cfam_d[label] += d / 1000
                cfam_c[label] += cc[n]
                break
    print(
        f"{'family':22s} {'eager ms':>10s} {'cmpl ms':>10s} "
        f"{'delta':>9s}  {'eager#':>6s} {'cmpl#':>6s}"
    )

    def sort_key(x):
        return -max(efam_d.get(x, 0), cfam_d.get(x, 0))

    labels = sorted(set(efam_d) | set(cfam_d), key=sort_key)
    for label in labels:
        delta = cfam_d.get(label, 0) - efam_d.get(label, 0)
        print(
            f"{label:22s} {efam_d.get(label, 0):10.1f} {cfam_d.get(label, 0):10.1f} "
            f"{delta:+9.1f}  {efam_c.get(label, 0):6d} {cfam_c.get(label, 0):6d}"
        )

    print("\n--- per-kernel diff (sorted by |delta|) ---")
    rows = []
    for n in set(ed) | set(cd):
        rows.append((n, ed.get(n, 0) / 1000, cd.get(n, 0) / 1000, ec.get(n, 0), cc.get(n, 0)))
    for n, e, c, ecnt, ccnt in sorted(rows, key=lambda r: -abs(r[2] - r[1])):
        if abs(c - e) < 0.05 and ecnt == ccnt:
            continue
        print(f"  {c - e:+9.2f} ms  {e:8.2f}->{c:8.2f}  #{ecnt:4d}->#{ccnt:4d}  {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eager vs compile kernel diff with family rollups")
    parser.add_argument("--eager", required=True, help="eager kernel_details.csv path")
    parser.add_argument("--compile", required=True, help="compile kernel_details.csv path")
    args = parser.parse_args()
    sys.exit(main(args.eager, args.compile))
