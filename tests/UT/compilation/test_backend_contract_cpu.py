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
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import fx

from mindiesd.compilation.compiliation_config import CompilationConfig
from mindiesd.compilation.passes.register_pattern_to_pass import PatternBase, register_pattern_to_pass

backend = importlib.import_module("mindiesd.compilation.mindie_sd_backend")


@pytest.mark.parametrize("name", ["name", "inputs", "pattern", "replacement"])
def test_abstract_pattern_contract(name):
    with pytest.raises(NotImplementedError):
        getattr(PatternBase, name)()


def test_registration_rejects_unsupported_input():
    for value in (None, object(), 1, "bad"):
        with pytest.raises(TypeError, match="not supported"):
            register_pattern_to_pass(value)
    with pytest.raises(RuntimeError, match="static class"):
        CompilationConfig()
    with pytest.raises(AttributeError):
        getattr(importlib.import_module("mindiesd.compilation"), "missing_export")


def test_graph_backend_default_and_acl_selection(monkeypatch):
    model = fx.symbolic_trace(torch.nn.ReLU())
    inputs = [torch.randn(2, 3)]
    instance = backend.MindieSDBackend()
    monkeypatch.setattr(backend, "npu_graph_available", False)
    compiled = Mock(return_value=model)
    monkeypatch.setattr(instance, "compile", compiled)
    assert instance(model, inputs) is model
    compiled.assert_called_once_with(model, inputs)
    for only, both, compile_calls in ((True, False, 0), (False, True, 1)):
        compiled.reset_mock()
        capture_backend = Mock(return_value="capture")
        monkeypatch.setattr(backend, "npu_graph_available", True)
        monkeypatch.setattr(CompilationConfig, "aclgraph_only", only)
        monkeypatch.setattr(CompilationConfig, "aclgraph_with_compile", both)
        with patch.object(backend, "create_aclgraph_backend", return_value=capture_backend):
            assert instance(model, inputs) == "capture"
        assert compiled.call_count == compile_calls
        capture_backend.assert_called_once_with(model, inputs)


def test_compile_freezing_disabled_runs_real_graph(monkeypatch):
    monkeypatch.setattr(CompilationConfig, "enable_freezing", False)
    module = fx.symbolic_trace(lambda x: (x.sin() + 2,))
    values = [torch.randn(2, 3)]
    compiled = backend.MindieSDBackend().compile(module, values)
    torch.testing.assert_close(compiled(*values)[0], module(*values)[0])


def legacy_backend_without_observer():
    original = builtins.__import__

    def filtered(name, *args, **kwargs):
        if name == "torch.fx.passes.graph_transform_observer":
            raise ImportError("legacy torch")
        return original(name, *args, **kwargs)

    path = Path(backend.__file__)
    spec = importlib.util.spec_from_file_location("mindiesd.compilation._cpu_legacy_backend", path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, "__import__", side_effect=filtered):
        spec.loader.exec_module(module)
    return module


def test_observer_compatibility_executes_passes():
    module = legacy_backend_without_observer()
    graph = fx.symbolic_trace(lambda x: x + 1)
    seen = []
    observer = module.GraphTransformObserver(graph, "cpu-pass")
    observer.apply_gm_pass(seen.append)
    observer.apply_graph_pass(seen.append)
    assert seen == [graph, graph.graph]


def test_decompose_compatibility_preserves_plain_graph():
    original = builtins.__import__

    def filtered(name, *args, **kwargs):
        if name == "torch._inductor.fx_passes.post_grad":
            raise ImportError("legacy torch")
        return original(name, *args, **kwargs)

    module = fx.symbolic_trace(lambda x: x.cos())
    with patch.object(builtins, "__import__", side_effect=filtered):
        assert backend.decompose_auto_functionalized(module.graph) is module.graph
    module.graph.lint()
