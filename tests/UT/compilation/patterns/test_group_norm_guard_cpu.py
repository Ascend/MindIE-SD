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


from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import fx

from mindiesd.compilation.patterns import group_norm_silu_pattern as group


def test_chain_and_argument_validation():
    graph = fx.Graph()
    x = graph.placeholder("x")
    empty = graph.call_function(torch.ops.aten.silu.default, ())
    assert group._chain_nodes(empty, False) == (None, None)
    literal = graph.call_function(torch.ops.aten.silu.default, (1,))
    assert group._chain_nodes(literal, False) == (None, None)
    wrong = graph.call_function(torch.ops.aten.neg.default, (x,))
    silu = graph.call_function(torch.ops.aten.silu.default, (wrong,))
    assert group._chain_nodes(silu, False) == (None, None)
    assert group._chain_nodes(silu, True) == (None, None)
    assert group._fused_args(wrong, False) is None
    assert group._fused_args(wrong, True) is None
    assert group._meta_value(1) == 1
    x.meta["example_value"] = torch.empty(2, 4, device="meta")
    assert group._meta_value(x) is x.meta["example_value"]
    match = SimpleNamespace(output_node=lambda: silu)
    assert not group._extra_check(match, False)


def test_fake_output_validation_and_hardware_rejection():
    meta = torch.empty(2, 4, 3, 3, device="meta")
    assert group._fake_outputs(None) is None
    assert group._fake_outputs((object(), None, None, 2, 1e-5)) is None
    assert group._fake_outputs((torch.empty(2, 4, 3, 3), None, None, 2, 1e-5)) is None
    assert group._fake_outputs((meta, "invalid", None, 2, 1e-5)) is None
    for result in (None, [], ["invalid"]):
        with patch.object(group.torch_npu, "npu_group_norm_silu", return_value=result):
            assert group._fake_outputs((meta, None, None, 2, 1e-5)) is None
    for error in (RuntimeError, TypeError, NotImplementedError, AttributeError):
        with patch.object(group.torch_npu, "npu_group_norm_silu", side_effect=error("unsupported")):
            assert group._fake_outputs((meta, None, None, 2, 1e-5)) is None


def test_group_norm_shared_output_is_not_fused():
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(1, 64, 4, 4, device="meta")
    norm = graph.call_function(torch.ops.aten.group_norm.default, (x, 32, None, None, 1e-5, True))
    silu = graph.call_function(torch.ops.aten.silu.default, (norm,))
    graph.output((norm, silu))
    assert not group._extra_check(SimpleNamespace(output_node=lambda: silu), False)
