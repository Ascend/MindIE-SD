#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


"""Each registered replacement must rewrite its real FX primitive graph on CPU."""

import importlib

import pytest
import torch
from torch._inductor import pattern_matcher as pm
from torch._subclasses.fake_tensor import FakeTensorMode

from mindiesd.compilation._custom_decomposition import select_custom_decomp_table
from mindiesd.compilation.passes.pattern_match_pass import PatternMatchPass


GROUPS = [
    ("adalayernorm_pattern", "AdaLayerNormPatternGroup"),
    ("gelu_pattern", "GELUPatternGroup"),
    ("mul_add_pattern", "MulAddPatternGroup"),
    ("rms_norm_pattern", "RMSNormPatternGroup"),
    ("rope_pattern", "RopePatternGroup"),
    ("minimax_h3_adaln_pattern", "MiniMaxH3AdaLnPatternGroup"),
    ("minimax_h3_gate_pattern", "MiniMaxH3GatePatternGroup"),
    ("minimax_h3_rmsnorm_pattern", "MiniMaxH3RmsNormPatternGroup"),
    ("minimax_h3_rope_pattern", "MiniMaxH3RopePatternGroup"),
    ("minimax_h3_swiglu_pattern", "MiniMaxH3SwigluPatternGroup"),
    ("norm_out_adalayernorm_pattern", "NormOutAdaLayerNormPatternGroup"),
    ("qwen_rope_pattern", "QwenRopePatternGroup"),
    ("qwen_residual_gate_pattern", "QwenResidualGatePatternGroup"),
    ("wan_adalayernorm_pattern", "WanAdaLayerNormPatternGroup"),
    ("wan_residual_gate_pattern", "WanResidualGatePatternGroup"),
    ("wan_rmsnorm_pattern", "WanRmsNormPatternGroup"),
    ("wan_rope_pattern", "WanRopePatternGroup"),
]


@pytest.mark.parametrize("module_name,group_name", GROUPS)
def test_real_pattern_graph_is_rewritten(module_name, group_name):
    module = importlib.import_module(f"mindiesd.compilation.patterns.{module_name}")
    with FakeTensorMode():
        for pattern in getattr(module, group_name):
            inputs = [
                torch.zeros_like(value, device="cpu") if isinstance(value, torch.Tensor) else value
                for value in pattern.inputs()
            ]
            # Use the production decomposition path used by register_replacement.
            graph = pm.fwd_only(pattern.pattern, inputs, get_decomp_fn=select_custom_decomp_table)
            matcher = PatternMatchPass()
            matcher.register_pattern(
                pattern.name(),
                pattern.pattern,
                pattern.replacement,
                inputs,
                extra_check=getattr(pattern, "extra_check", None),
            )
            before = graph.code
            replacements = matcher.pattern_pass.apply(graph)
            assert replacements >= 1, pattern.name()
            graph.graph.lint()
            graph.recompile()
            assert graph.code != before, pattern.name()
            assert any(
                "mindiesd_test_cpu" in str(node.target) or "mindiesd." in str(node.target) for node in graph.graph.nodes
            ), pattern.name()
