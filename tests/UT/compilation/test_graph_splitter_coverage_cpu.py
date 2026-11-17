#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Validate breakpoint metadata through real CPU FX segment execution."""

import importlib
import operator

import pytest
import torch
from torch import fx

from mindiesd.compilation.graph_splitter import split_graph_by_breakpoints
from tests.UT.compilation.cpu_graph_runtime import CapturedGraph


def _capture_cpu(gm, inputs):
    with CapturedGraph() as captured:
        result = gm(*inputs)
    return captured, result


def _run_segmented(monkeypatch, gm, segments, output_spec, *inputs):
    backend = importlib.import_module("mindiesd.compilation.aclgraph_backend")
    monkeypatch.setattr(backend, "_capture_single_graph", _capture_cpu)

    # The first invocation clones the original inputs, so synchronous copy is
    # enough for this CPU boundary; graph capture/replay uses the shared double.
    def copy_inputs(entry, needs_copy):
        for destination, source in needs_copy:
            destination.copy_(source)

    monkeypatch.setattr(backend, "_copy_inputs_on_stream", copy_inputs)
    return backend.create_segmented_aclgraph_backend(gm, segments, output_spec)(*inputs)


@pytest.mark.parametrize("breakpoints", [None, [], ["missing"], ["x"]])
def test_no_effective_breakpoint_returns_original_graph(breakpoints):
    gm = fx.symbolic_trace(lambda x: x.sin() + 1)
    graphs, metadata, output_spec = split_graph_by_breakpoints(gm, breakpoints)
    assert len(graphs) == 1 and graphs[0] is gm
    assert metadata is None and output_spec is None


def test_nested_outputs_keep_early_values_repeated_leaves_and_constants(monkeypatch):
    graph = fx.Graph()
    x, y = (graph.placeholder(name) for name in ("x", "y"))
    add = graph.call_function(operator.add, (x, y))
    mul = graph.call_function(operator.mul, (add, add))
    neg = graph.call_function(operator.neg, (mul,))
    graph.output({"early": add, "late": (neg, [mul, None, "done", add]), "input": x})
    gm = fx.GraphModule({}, graph)
    graphs, segments, spec = split_graph_by_breakpoints(gm, [mul.name, neg.name])
    assert len(graphs) == len(segments) == 3
    assert [segment.num_outputs for segment in segments] == [2, 1, 1]
    assert segments[0].inputs == [("orig", "x"), ("orig", "y")]
    # add is used twice by mul, but it must create only one segment placeholder.
    assert segments[1].inputs == [("prev", 0, 1)]
    assert segments[2].inputs == [("prev", 1, 0)]
    assert spec.leaves == [(0, 1), (2, 0), (1, 0), (0, 1), (0, 0)]
    inputs = (torch.arange(6).reshape(2, 3).float(), torch.full((2, 3), 2.0))
    actual = _run_segmented(monkeypatch, gm, segments, spec, *inputs)
    expected = gm(*inputs)
    assert isinstance(actual, dict) and isinstance(actual["late"], tuple)
    assert isinstance(actual["late"][1], list)
    assert actual["late"][1][1:3] == [None, "done"]
    torch.testing.assert_close(actual["early"], expected["early"])
    torch.testing.assert_close(actual["late"][0], expected["late"][0])
    torch.testing.assert_close(actual["late"][1][0], expected["late"][1][0])
    torch.testing.assert_close(actual["late"][1][3], expected["late"][1][3])
    torch.testing.assert_close(actual["input"], expected["input"])


def test_split_preserves_module_attributes_and_tuple_valued_nodes(monkeypatch):
    module = torch.nn.Module()
    module.register_buffer("offset", torch.tensor([1.0, 2.0]))
    graph = fx.Graph()
    x = graph.placeholder("x")
    offset = graph.get_attr("offset")
    add = graph.call_function(operator.add, (x, offset))
    pair = graph.call_function(torch.ops.aten.split.Tensor, (add, 1, 0))
    first = graph.call_function(operator.getitem, (pair, 0))
    second = graph.call_function(operator.getitem, (pair, 1))
    result = graph.call_function(operator.add, (first, second))
    graph.output(result)
    gm = fx.GraphModule(module, graph)
    _, segments, spec = split_graph_by_breakpoints(gm, [pair.name, first.name])
    assert [segment.num_outputs for segment in segments] == [1, 1, 1]
    assert segments[1].inputs == [("prev", 0, 0)]
    assert segments[2].inputs == [("prev", 1, 0)]
    value = torch.arange(4).reshape(2, 2).float()
    torch.testing.assert_close(_run_segmented(monkeypatch, gm, segments, spec, value), gm(value))


def test_constant_output_and_unused_segment_remain_valid(monkeypatch):
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.call_function(operator.neg, (x,))
    unused = graph.call_function(torch.sin, (x,))
    graph.output((7, {"constant": None}))
    gm = fx.GraphModule({}, graph)
    _, segments, spec = split_graph_by_breakpoints(gm, [unused.name])
    assert [segment.num_outputs for segment in segments] == [1, 0]
    assert not spec.leaves
    assert _run_segmented(monkeypatch, gm, segments, spec, torch.ones(3)) == (7, {"constant": None})


def test_constant_only_graph_does_not_create_empty_segments():
    graph = fx.Graph()
    graph.output(42)
    gm = fx.GraphModule({}, graph)
    assert split_graph_by_breakpoints(gm, ["missing"]) == ([gm], None, None)
