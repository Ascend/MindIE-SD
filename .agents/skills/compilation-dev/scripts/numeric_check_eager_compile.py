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

"""图改写语义核验：同 seed 同进程 eager（无融合）vs compile（融合）输出对比。

手动改图（graph-pattern-rewrite §6）必须证明是等价替换：融合路径与 eager 基线
逐元素一致。对已验证位级一致的融合 kernel 断言 mean_rel==0.0；一般融合用阈值。

模板：替换 RUN_PIPE（返回 {name: tensor} dict）与底部模型构建段后运行。

Usage:
    python numeric_check_eager_compile.py [--tol 0.0]
"""

import argparse
import sys

import torch


def run_pipe(tag):
    """返回 {输出名: tensor}。子类在底部模型构建段中实现：
    eager 基线（无融合）与 compile（融合）各调一次，同 seed。"""
    raise NotImplementedError


def compare(va, vb, tol):
    ok = True
    for name in va:
        x = va[name].float().cpu()
        y = vb[name].float().cpu()
        denom = y.abs().mean().item() + 1e-6
        rel = ((x - y).abs() / denom).mean().item()
        maxabs = (x - y).abs().max().item()
        passed = rel <= tol and maxabs <= tol
        ok = ok and passed
        print(f"{name}: mean_rel={rel:.6f} max_abs={maxabs:.6f} "
              f"({'PASS' if passed else 'FAIL'})", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Numeric check eager vs compile")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="max allowed mean_rel/max_abs (default 0.0 = bit-identical)")
    args = parser.parse_args()
    va = run_pipe("eager_base")
    vb = run_pipe("compile_fused")
    ok = compare(va, vb, args.tol)
    print("NUMCHECK_DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
