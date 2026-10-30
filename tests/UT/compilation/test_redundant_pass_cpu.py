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


import operator

import pytest
import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx

from mindiesd.compilation.passes.gm_pass_base import GraphModulePass
from mindiesd.compilation.passes.redundant_node_elimination_pass import ReduandantNodeEliminationPass, get_node_shape


@pytest.mark.parametrize(
    "memory_format,removed", [(None, True), (torch.preserve_format, True), (torch.contiguous_format, False)]
)
def test_clone_elimination_preserves_nondefault_layout(memory_format, removed):
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.clone.default, (x,), {"memory_format": memory_format})
    graph.output(out)
    module = fx.GraphModule({}, graph)
    ReduandantNodeEliminationPass()(module)
    assert any(n.target == torch.ops.aten.clone.default for n in module.graph.nodes) is not removed
    value = torch.randn(2, 3)
    torch.testing.assert_close(module(value), value)


@pytest.mark.parametrize("index,removed", [(0, True), (None, False)])
def test_single_split_user_contract(index, removed):
    graph = fx.Graph()
    x = graph.placeholder("x")
    split = graph.call_function(torch.ops.aten.split_with_sizes.default, (x, [3], 0))
    out = split if index is None else graph.call_function(operator.getitem, (split, index))
    graph.output(out)
    module = fx.GraphModule({}, graph)
    ReduandantNodeEliminationPass()(module)
    assert any(n.target == torch.ops.aten.split_with_sizes.default for n in graph.nodes) is not removed


@pytest.mark.parametrize("shape,removed", [((2, 3), True), ((3, 2), False)])
def test_view_metadata_contract(shape, removed):
    value = torch.randn(2, 3)
    module = make_fx(lambda x: x.view(shape))(value)
    expected = module(value)
    ReduandantNodeEliminationPass()(module)
    assert any(n.target == torch.ops.aten.view.default for n in module.graph.nodes) is not removed
    torch.testing.assert_close(module(value), expected)


def test_missing_shape_and_pass_uuid():
    assert get_node_shape(object()) is None
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = "not a tensor"
    assert get_node_shape(x) is None
    p = ReduandantNodeEliminationPass()
    assert p.uuid() == ReduandantNodeEliminationPass().uuid()
    assert len(p.uuid()) == 64
