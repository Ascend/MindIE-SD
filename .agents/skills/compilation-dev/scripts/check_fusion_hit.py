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

"""融合命中判定：在 compile kernel_details.csv 中统计融合 kernel 出现次数。

注册成功/单元测试通过 ≠ pattern 命中真实图；最终判定看 kernel csv 中融合 kernel
是否出现且计数 == 期望站点数（graph-pattern-rewrite §5.3）。

Usage:
    python check_fusion_hit.py --csv compile/kernel_details.csv \
        --kernel GroupedMxMatmulSliceMSwigluMxQuant --expect 3
    # --name 只是显示标签（默认用 --kernel 的尾部可读名）
"""

import argparse
import csv
import sys


def count_kernel(path, needle):
    hits = 0
    total = 0
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            total += 1
            if needle in (row.get("Name") or ""):
                hits += 1
    return hits, total


def main():
    parser = argparse.ArgumentParser(description="Check fused-kernel hit in a kernel csv")
    parser.add_argument("--csv", required=True, help="kernel_details.csv path")
    parser.add_argument("--kernel", required=True,
                        help="substring of the fused kernel name (e.g. GroupedMxMatmulSliceMSwigluMxQuant)")
    parser.add_argument("--expect", type=int, default=None,
                        help="expected hit count (e.g. number of FFN sites); exit 1 if mismatch")
    parser.add_argument("--name", default=None, help="display label (defaults to --kernel tail)")
    parser.add_argument("--eager-csv", default=None,
                        help="optional eager csv to confirm fused kernel absent there")
    args = parser.parse_args()

    label = args.name or args.kernel.rsplit(":", 1)[-1].split(".")[-1]
    hits, total = count_kernel(args.csv, args.kernel)
    print(f"[compile] {label}: {hits} hit(s) / {total} kernels")
    ok = True
    if args.eager_csv:
        e_hits, e_total = count_kernel(args.eager_csv, args.kernel)
        print(f"[eager  ] {label}: {e_hits} hit(s) / {e_total} kernels (expect 0)")
        ok = ok and e_hits == 0
    if args.expect is not None:
        ok = ok and hits == args.expect
        print(f"expect {args.expect}: {'PASS' if hits == args.expect else 'FAIL'}")
    if args.expect is None and args.eager_csv is None:
        print("(no --expect/--eager-csv: informational)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
