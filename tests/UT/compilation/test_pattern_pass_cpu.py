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


import builtins
import importlib
import importlib.util
import logging
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from mindiesd.compilation.passes.pattern_match_pass import PatternMatchPass

module = importlib.import_module("mindiesd.compilation.passes.pattern_match_pass")


def add_twice(x):
    return x + x


def multiply_twice(x):
    return x * 2


def test_real_rewrite_runs_to_fixed_point_and_preserves_value():
    value = torch.randn(2, 3)
    graph = make_fx(add_twice)(value)
    matcher = PatternMatchPass()
    matcher.register_pattern("cpu-add-to-mul", add_twice, multiply_twice, [torch.empty(2, 3, device="meta")])
    saved_level = module.logger.level
    try:
        module.logger.setLevel(logging.DEBUG)
        matcher(graph)
    finally:
        module.logger.setLevel(saved_level)
    assert not any(node.target == torch.ops.aten.add.Tensor for node in graph.graph.nodes)
    assert any(node.target == torch.ops.aten.mul.Tensor for node in graph.graph.nodes)
    graph.recompile()
    torch.testing.assert_close(graph(value), value + value)
    with pytest.raises(ValueError, match="already registered"):
        matcher.register_pattern("cpu-add-to-mul", add_twice, multiply_twice, [value])


def test_registration_only_ignores_known_duplicate_errors():
    for message, raises in [("Duplicate pattern found", False), ("incompatible input", True)]:
        matcher = PatternMatchPass()
        with patch.object(module.pm, "register_replacement", side_effect=RuntimeError(message)):
            if raises:
                with pytest.raises(RuntimeError, match=message):
                    matcher.register_pattern(message, add_twice, multiply_twice, [torch.empty(2, 3, device="meta")])
            else:
                matcher.register_pattern(message, add_twice, multiply_twice, [torch.empty(2, 3, device="meta")])


def test_missing_modern_fwd_only_reports_unsupported_torch():
    saved = module.pm.fwd_only
    try:
        del module.pm.fwd_only
        with patch.object(module, "IS_TORCH_21", False), patch.object(module.pm, "register_replacement") as register:
            PatternMatchPass().register_pattern(
                "no-fwd-only", add_twice, multiply_twice, [torch.empty(2, 3, device="meta")]
            )
        register.assert_called_once()
    finally:
        module.pm.fwd_only = saved


def test_legacy_torch_21_trace_compatibility():
    spec = importlib.util.spec_from_file_location("mindiesd.compilation.passes._cpu_legacy_match", module.__file__)
    legacy = importlib.util.module_from_spec(spec)
    with patch.object(torch, "__version__", "2.1.0"):
        spec.loader.exec_module(legacy)
    assert legacy.IS_TORCH_21
    value = torch.randn(2, 3)
    table = legacy.select_custom_decomp_table()
    with patch.dict(table, table.copy(), clear=True):
        graph = legacy.mindie_inference_graph(multiply_twice, [value])
    torch.testing.assert_close(graph(value), value * 2)
