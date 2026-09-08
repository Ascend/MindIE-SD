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

"""前缀逐级隔离：定位手写 pattern 链的首个不匹配节点（graph-pattern-rewrite §5.2）。

把目标链切成渐进变长的一组 pattern（p1 单节点 → p2 +1 节点 → ... → 全链），逐个
注册并打印真实图匹配数；第一个掉到 0 的就是断点。断点通常暴露：
- 该层节点 target/args 与真实图不一致；
- 该层引入的共享子节点未复用同一 PatternExpr 实例 / 缺 _users=MULTIPLE；
- 该层多了/少了 shape-noise 节点（view 尺寸未 Ignored）。

模板：替换 LEVEL_BUILDERS 与底部模型构建段后运行。

Usage:
    python isolate_pattern_prefix.py
"""

import os
import sys

import torch
import torch_npu  # noqa: F401
import torch._inductor.pattern_matcher as pm  # noqa: E402
from torch._inductor.pattern_matcher import (  # noqa: E402
    PatternMatcherPass,
    CallFunction,
    Arg,
    Ignored,
    MULTIPLE,
)

# --- 在此定义渐进链（返回 [(name, pattern_expr), ...]）---
# 例（MiniMax FFN 真实形态，F=14336）：
# def level_builders():
#     aten = torch.ops.aten
#     l1 = CallFunction(torch_npu.npu_quant_matmul, Arg(), Arg(), Arg())
#     l2 = CallFunction(aten.view.default, l1, Ignored())
#     l3 = CallFunction(aten.split.Tensor, l2, 14336, -1, _users=MULTIPLE)
#     g0 = CallFunction(operator.getitem, l3, 0, _users=MULTIPLE)
#     g1 = CallFunction(operator.getitem, l3, 1, _users=MULTIPLE)
#     l4 = CallFunction(aten.mul.Tensor, g0,
#                       CallFunction(aten.silu.default, g1))
#     return [("l1_qmm", l1), ("l2_view", l2), ("l3_split", l3), ("l4_act", l4)]
LEVEL_BUILDERS = None


def register_levels():
    passes = []
    for name, pat in LEVEL_BUILDERS():
        p = PatternMatcherPass(pass_name=f"inc_{name}")
        pm.GraphPatternEntry(pattern=pat, extra_check=lambda m: True,
                             handler=lambda m, *a, **k: None).register(p.patterns)
        passes.append((name, p))
    return passes


def _install_hook(level_passes, backend_cls=None):
    if backend_cls is None:
        from mindiesd.compilation import MindieSDBackend as backend_cls

    orig = backend_cls.apply_pattern_match_passes.__func__

    def patched(cls, graph, inputs):
        for name, p in level_passes:
            try:
                print(f"INC {name} MATCHED {p.apply(graph)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"INC {name} ERROR {type(e).__name__}: {e}", flush=True)
        return orig(cls, graph, inputs)

    backend_cls.apply_pattern_match_passes = classmethod(patched)


def main():
    if LEVEL_BUILDERS is None:
        print("LEVEL_BUILDERS 为空：先按文件头注释填入渐进链与模型构建段。")
        return
    _install_hook(register_levels())
    # --- 在此替换为模型构建 + compile + 前向 ---
    print("ISOLATE_TEMPLATE_DONE")


if __name__ == "__main__":
    main()
