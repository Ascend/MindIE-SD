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


import contextlib
import importlib
from unittest.mock import MagicMock

import pytest

import pkgutil

import threading


def _cpu_graph_pool_handle():
    return "cpu-test-pool"


def _null_stream(_stream):
    return contextlib.nullcontext()


@pytest.fixture(autouse=True)
def cpu_fused_dispatch(monkeypatch):
    import torch
    import torch_npu
    import mindiesd
    from mindiesd.compilation import patterns
    from tests.UT.compilation.patterns import cpu_fused_ops as ops

    assert not torch.npu.is_available(), "Pattern tests must run without an available NPU"
    # Registrations capture replacement callables; isolate the real matcher so
    # test hardware doubles never leak into other domains after fixture teardown.
    registry = importlib.import_module("mindiesd.compilation.passes.register_pattern_to_pass")
    passes = importlib.import_module("mindiesd.compilation.passes")
    backend = importlib.import_module("mindiesd.compilation.mindie_sd_backend")
    from mindiesd.compilation.passes.pattern_match_pass import PatternMatchPass

    isolated_matcher = PatternMatchPass()
    monkeypatch.setattr(registry, "patterns", isolated_matcher)
    monkeypatch.setattr(backend, "patterns", isolated_matcher)
    monkeypatch.setattr(passes, "once_flag", threading.Event())
    for info in pkgutil.iter_modules(patterns.__path__):
        module = importlib.import_module(f"mindiesd.compilation.patterns.{info.name}")
        # Some production modules conditionally bind these names on device availability.
        monkeypatch.setattr(module, "mindiesd", mindiesd, raising=False)
        monkeypatch.setattr(module, "torch_npu", torch_npu, raising=False)
    monkeypatch.setattr(torch_npu, "npu_fast_gelu", ops.fast_gelu)
    monkeypatch.setattr(torch_npu, "npu_rms_norm", ops.rms)
    monkeypatch.setattr(torch_npu, "npu_rotary_mul", ops.rope)
    monkeypatch.setattr(torch_npu, "npu_group_norm_silu", ops.group_norm_silu._opoverload)
    monkeypatch.setattr(
        mindiesd,
        "layernorm_scale_shift",
        lambda layernorm, x, scale, shift, fused: ops.adaln(x, scale, shift, layernorm.eps),
    )
    monkeypatch.setattr(mindiesd.layers, "residual_gate_add", ops.gate)
    monkeypatch.setattr(mindiesd.layers, "qwen_residual_gate_add", ops.gate)
    mul_pattern = importlib.import_module("mindiesd.compilation.patterns.mul_add_pattern")
    monkeypatch.setattr(mul_pattern, "muls_add", ops.mul_add)
    # Do not patch torch.compile, PatternMatcherPass, or the backend: graph
    # tracing, matching, extra_check and graph rewrites all execute for real.
    yield
    torch._dynamo.reset()


@pytest.fixture(autouse=True)
def cpu_graph_dispatch(request, monkeypatch):
    import torch
    from tests.UT.compilation.cpu_graph_runtime import CapturedGraph, capture

    if request.node.path.name != "test_aclgraph_backend.py":
        yield
        return
    assert not torch.npu.is_available()
    acl = importlib.import_module("mindiesd.compilation.aclgraph_backend")
    backend = importlib.import_module("mindiesd.compilation.mindie_sd_backend")
    monkeypatch.setattr(acl, "_global_graph_pool", None)
    monkeypatch.setattr(acl, "npu_graph_available", True)
    monkeypatch.setattr(backend, "npu_graph_available", True)
    monkeypatch.setattr(torch.npu, "NPUGraph", CapturedGraph)
    monkeypatch.setattr(torch.npu, "graph", capture)
    monkeypatch.setattr(torch.npu, "graph_pool_handle", _cpu_graph_pool_handle)
    monkeypatch.setattr(torch.npu, "current_stream", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(torch.npu, "Stream", MagicMock(side_effect=MagicMock))
    monkeypatch.setattr(torch.npu, "stream", _null_stream)
    yield
    torch._dynamo.reset()
