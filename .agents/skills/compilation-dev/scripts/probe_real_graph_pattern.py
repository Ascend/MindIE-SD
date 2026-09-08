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

"""在真实 compile 图上注入 pattern 匹配 probe（graph-pattern-rewrite §5.1）。

用法：把本文件当作模板，在文件底部替换 BUILD/MODEL 段为你的模型构建代码，
并将 PROBE_PATTERNS 改为你要测试的 pattern 列表。脚本 monkey-patch
MindieSDBackend.apply_pattern_match_passes，在真实 pattern 跑之前先 apply 这些
probe pattern 并打印匹配数。

为什么不用本地 symbolic_trace/make_fx 验证：probe 图形态与真实 compile 图不同
（见 graph-pattern-rewrite-guide.md §5），必须在真实图上打。

Usage:
    python probe_real_graph_pattern.py   # 修改底部模型构建段后运行
"""

import os
import sys

import torch
import torch_npu  # noqa: F401
import torch._inductor.pattern_matcher as pm  # noqa: E402
from torch._inductor.pattern_matcher import PatternMatcherPass  # noqa: E402

# --- 在此替换为你的目标模型构建路径（示例为 MiniMax-H3 dummy）---
# sys.path.insert(0, "<dummy_run dir>")
# from model.minimax_h3_model import build_minimax_h3_pipeline
# MODEL_CFG = os.environ.get("MMX_CFG", "<config dir>")
# BUILD_KW = dict(num_layers=2, device="npu:0")
# PYTHONPATH 前置隔离 diffusers 版本时: sys.path.insert(0, "<dif040_site>")

# --- 在此列出要测的 pattern（GraphPatternEntry），并各自命名 ---
# 例：
# PROBE_PATTERNS = [("probe_qmm", qmm_pattern_expr)]
PROBE_PATTERNS = []


def register_probes(pass_dict):
    """把 PROBE_PATTERNS 注册到 pass_dict，返回 [(name, PatternMatcherPass)]。"""
    out = []
    for name, pat in PROBE_PATTERNS:
        p = PatternMatcherPass(pass_name=f"probe_{name}")
        pm.GraphPatternEntry(pattern=pat, extra_check=lambda m: True,
                             handler=lambda m, *a, **k: None).register(p.patterns)
        out.append((name, p))
    return out


def _install_hook(probe_passes, backend_cls=None):
    """Monkey-patch MindieSDBackend.apply_pattern_match_passes 注入 probe。"""
    if backend_cls is None:
        from mindiesd.compilation import MindieSDBackend as backend_cls

    orig = backend_cls.apply_pattern_match_passes.__func__

    def patched(cls, graph, inputs):
        for name, p in probe_passes:
            try:
                print(f"PROBE {name} MATCHED {p.apply(graph)}", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"PROBE {name} ERROR {type(e).__name__}: {e}", flush=True)
        return orig(cls, graph, inputs)

    backend_cls.apply_pattern_match_passes = classmethod(patched)


def main():
    if not PROBE_PATTERNS:
        print("PROBE_PATTERNS 为空：先按文件头注释填入待测 pattern 与模型构建段。")
        return
    probe_passes = register_probes(None)
    _install_hook(probe_passes)
    # --- 在此替换为模型构建 + compile + 前向（示例）---
    # pipe = build_minimax_h3_pipeline(MODEL_CFG, **BUILD_KW)
    # pipe.to("npu:0")
    # from minimax_h3_infer import _apply_compute_precision
    # _apply_compute_precision(pipe, "bf16")
    # from model.common import apply_w8a8_quant
    # apply_w8a8_quant(pipe, attrs=("transformer",))
    # compiled = torch.compile(pipe.transformer, backend=MindieSDBackend())
    # ...run once...
    print("PROBE_TEMPLATE_DONE")


if __name__ == "__main__":
    main()
