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

"""Execute optional pass rewrites on real FX graphs with CPU dispatch doubles."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import fx

from mindiesd.compilation.compiliation_config import CompilationConfig, FusionPatterns
from mindiesd.compilation.passes.gelu_quant_pass import GeluQuantFusionPass
from mindiesd.compilation.passes.pattern_match_pass import PatternBackend, PatternMatchPass
from mindiesd.compilation.passes.rope_cast_pass import RopeCastFusionPass
from tests.UT.compilation.patterns import cpu_fused_ops


def npu_dynamic_mx_quant(x, *, dst_type=None, round_mode="rint", **kwargs):
    """CPU quantization boundary double; this does not model hardware MX scales."""
    return torch.round(x * 8).to(torch.int8), torch.full_like(x[..., :1], 0.125)


def npu_gelu_quant(x, *, approximate="none", quant_mode="dynamic", dst_type=None, round_mode="rint"):
    return npu_dynamic_mx_quant(
        torch.nn.functional.gelu(x, approximate=approximate), dst_type=dst_type, round_mode=round_mode
    )


def npu_fast_gelu(x):
    return cpu_fused_ops.fast_gelu(x)


def npu_rotary_mul(x, cos, sin, mode):
    return cpu_fused_ops.rope(x, cos, sin, mode)


def _gelu_graph(head, quant_kwargs, with_views=False):
    graph = fx.Graph()
    x = graph.placeholder("x")
    if head == "fast":
        value = graph.call_function(npu_fast_gelu, (x,))
    else:
        value = graph.call_function(torch.ops.aten.gelu.default, (x,), {"approximate": head})
    if with_views:
        value = graph.call_function(torch.ops.aten.view.default, (value, (2, 2, 4)))
        value = graph.call_function(torch.ops.aten.reshape.default, (value, (4, 4)))
    value = graph.call_function(npu_dynamic_mx_quant, (value,), quant_kwargs)
    graph.output(value)
    return fx.GraphModule({}, graph)


@pytest.mark.parametrize("head", ["none", "tanh", "fast"])
@pytest.mark.parametrize("with_views", [False, True])
def test_gelu_quant_rewrite_preserves_values_and_view_order(monkeypatch, head, with_views):
    monkeypatch.setattr(torch.ops.npu, "npu_gelu_quant", SimpleNamespace(default=npu_gelu_quant), raising=False)
    module = _gelu_graph(head, {"dst_type": 24, "round_mode": "rint"}, with_views)
    value = torch.linspace(-3, 3, 16).reshape(4, 4)
    expected = module(value)
    GeluQuantFusionPass(approximate="tanh")(module)
    targets = [node.target for node in module.graph.nodes]
    assert npu_dynamic_mx_quant not in targets
    assert torch.ops.aten.gelu.default not in targets
    assert npu_fast_gelu not in targets
    fused = next(node for node in module.graph.nodes if node.target is npu_gelu_quant)
    assert fused.kwargs == {
        "approximate": "tanh" if head == "fast" else head,
        "quant_mode": "dynamic",
        "dst_type": 24,
        "round_mode": "rint",
    }
    if with_views:
        assert fused.args[0].target is torch.ops.aten.reshape.default
        assert fused.args[0].args[0].target is torch.ops.aten.view.default
        assert fused.args[0].args[0].args[0].op == "placeholder"
    torch.testing.assert_close(module(value), expected)


@pytest.mark.parametrize("quant_kwargs", [{}, {"dst_type": 24, "axis": -1}])
def test_gelu_quant_keeps_unrepresentable_quantization(monkeypatch, quant_kwargs):
    monkeypatch.setattr(torch.ops.npu, "npu_gelu_quant", SimpleNamespace(default=npu_gelu_quant), raising=False)
    module = _gelu_graph("none", quant_kwargs)
    value = torch.randn(4, 4)
    expected = module(value)
    GeluQuantFusionPass()(module)
    assert any(node.target is npu_dynamic_mx_quant for node in module.graph.nodes)
    assert all(node.target is not npu_gelu_quant for node in module.graph.nodes)
    torch.testing.assert_close(module(value), expected)


def test_optional_passes_keep_graph_when_operator_or_chain_is_absent(monkeypatch):
    module = fx.symbolic_trace(lambda x: x + 1)
    before = str(module.graph)
    monkeypatch.setattr(torch.ops.npu, "npu_gelu_quant", SimpleNamespace(), raising=False)
    GeluQuantFusionPass()(module)
    monkeypatch.setattr(torch.ops.npu, "npu_gelu_quant", SimpleNamespace(default=npu_gelu_quant), raising=False)
    GeluQuantFusionPass()(module)
    RopeCastFusionPass()(module)
    assert str(module.graph) == before
    torch.testing.assert_close(module(torch.ones(2)), torch.full((2,), 2.0))
    with pytest.raises(ValueError, match="approximate"):
        GeluQuantFusionPass("unsupported")


def test_rope_cast_rewrite_reuses_small_cast_and_keeps_bf16_inputs():
    graph = fx.Graph()
    x, cos, sin = (graph.placeholder(name) for name in ("x", "cos", "sin"))
    for node, dtype in ((x, torch.bfloat16), (cos, torch.float32), (sin, torch.bfloat16)):
        node.meta["val"] = torch.empty(2, 4, dtype=dtype)
    results = []
    for mode in ("interleave", "half"):
        cast_in = graph.call_function(torch.ops.aten._to_copy.default, (x,), {"dtype": torch.float32})
        rope = graph.call_function(npu_rotary_mul, (cast_in, cos, sin, mode))
        results.append(graph.call_function(torch.ops.aten._to_copy.default, (rope,), {"dtype": torch.bfloat16}))
    graph.output(tuple(results))
    module = fx.GraphModule({}, graph)
    RopeCastFusionPass()(module)
    ropes = [node for node in module.graph.nodes if node.target is npu_rotary_mul]
    casts = [node for node in module.graph.nodes if node.target is torch.ops.aten._to_copy.default]
    assert len(ropes) == 2
    assert len(casts) == 1
    assert casts[0].args == (cos,)
    assert casts[0].kwargs == {"dtype": torch.bfloat16}
    assert all(node.args[0] is x and node.args[1] is casts[0] and node.args[2] is sin for node in ropes)
    values = (
        torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) / 8,
        torch.full((2, 4), 0.5),
        torch.full((2, 4), 0.25, dtype=torch.bfloat16),
    )
    expected = tuple(
        npu_rotary_mul(values[0], values[1].bfloat16(), values[2], mode) for mode in ("interleave", "half")
    )
    torch.testing.assert_close(module(*values), expected, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_pattern_activation_respects_switches_and_runs_once(monkeypatch, enabled):
    passes = importlib.import_module("mindiesd.compilation.passes")
    registry = importlib.import_module("mindiesd.compilation.passes.register_pattern_to_pass")
    patterns = importlib.import_module("mindiesd.compilation.patterns")
    config = FusionPatterns()
    for name in vars(config):
        if name.startswith("enable_"):
            setattr(config, name, enabled)
    monkeypatch.setattr(CompilationConfig, "fusion_patterns", config)
    register = Mock()
    group_norm = Mock()
    monkeypatch.setattr(registry, "register_pattern_to_pass", register)
    monkeypatch.setattr(patterns.GroupNormSiLUPatternGroup, "register", group_norm)
    passes.activate_pattern_once()
    passes.activate_pattern_once()
    group_norm.assert_called_once_with(registry.patterns.pattern_pass)
    assert passes.once_flag.is_set()
    groups = [call.args[0] for call in register.call_args_list]
    if enabled:
        assert len(groups) == 18
        assert groups.count(patterns.AddLayerNormPatternGroup) == 1
        assert groups.index(patterns.AddLayerNormPatternGroup) < groups.index(patterns.WanResidualGatePatternGroup)
        assert groups.index(patterns.QwenRopePatternGroup) < groups.index(patterns.WanResidualGatePatternGroup)
        assert groups.index(patterns.MiniMaxH3RopePatternGroup) < groups.index(patterns.WanResidualGatePatternGroup)
    else:
        assert groups == []


def test_lightweight_decomposition_rewrites_real_aten_graph(monkeypatch):
    passes = importlib.import_module("mindiesd.compilation.passes")
    decomposition = importlib.import_module("mindiesd.compilation._custom_decomposition")
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(torch.ops.aten.pow.Tensor_Scalar, (x, 2)))
    original = fx.GraphModule({}, graph)
    value = torch.randn(3, 4)

    def decompose_square(tensor, exponent):
        assert exponent == 2
        return tensor * tensor

    monkeypatch.setattr(
        decomposition, "select_custom_decomp_table", lambda: {torch.ops.aten.pow.Tensor_Scalar: decompose_square}
    )
    decomposed = passes.apply_lightweight_decomp(original, [value])
    assert decomposed is not original
    assert any(node.target is torch.ops.aten.mul.Tensor for node in decomposed.graph.nodes)
    assert all(node.target is not torch.ops.aten.pow.Tensor_Scalar for node in decomposed.graph.nodes)
    torch.testing.assert_close(decomposed(value), original(value))
    monkeypatch.setattr(decomposition, "select_custom_decomp_table", lambda: {})
    assert passes.apply_lightweight_decomp(original, [value]) is original
    monkeypatch.setattr(decomposition, "select_custom_decomp_table", Mock(side_effect=RuntimeError("trace failed")))
    assert passes.apply_lightweight_decomp(original, [value]) is original
    torch.testing.assert_close(original(value), value.square())


@pytest.mark.parametrize("extra", [False, True])
def test_npugraph_registration_forwards_contract(monkeypatch, extra):
    register = Mock()
    monkeypatch.setattr(torch.npu, "npugraph_ex", SimpleNamespace(register_replacement=register), raising=False)
    matcher = PatternMatchPass()

    def pattern(x):
        return x + 1

    def replacement(x):
        return x - (-1)

    def extra_check(match):
        return True

    example = [torch.empty(2, 3, device="meta")]
    check = extra_check if extra else None
    matcher.register_pattern("npugraph-contract", pattern, replacement, example, check, PatternBackend.NPUGRAPH_EX)
    register.assert_called_once_with(pattern, replacement, example, **({"extra_check": check} if extra else {}))
    assert matcher.pattern_replacements["npugraph-contract"] == (pattern, replacement)
    assert not matcher.pattern_pass.patterns
    with pytest.raises(ValueError, match="already registered"):
        matcher.register_pattern("npugraph-contract", pattern, replacement, example)


@pytest.mark.parametrize("failure", [None, ValueError("schema mismatch"), KeyError("legacy schema")])
def test_npugraph_unavailable_or_rejected_registration_is_nonfatal(monkeypatch, failure):
    module = importlib.import_module("mindiesd.compilation.passes.pattern_match_pass")
    register = None if failure is None else Mock(side_effect=failure)
    monkeypatch.setattr(torch.npu, "npugraph_ex", SimpleNamespace(register_replacement=register), raising=False)
    matcher = module.PatternMatchPass()

    def function(x):
        return x + 1

    matcher.register_pattern("optional-npugraph", function, function, [], backend="npugraph_ex")
    assert "optional-npugraph" in matcher.pattern_replacements
    if register is not None:
        register.assert_called_once()


def test_unknown_backend_falls_back_to_real_inductor_matching():
    from torch.fx.experimental.proxy_tensor import make_fx

    def pattern(x):
        return x + x

    def replacement(x):
        return x * 2

    matcher = PatternMatchPass()
    matcher.register_pattern(
        "fallback-backend", pattern, replacement, [torch.empty(2, 3, device="meta")], backend="typo"
    )
    value = torch.randn(2, 3)
    graph = make_fx(pattern)(value)
    matcher(graph)
    graph.recompile()
    assert any(node.target is torch.ops.aten.mul.Tensor for node in graph.graph.nodes)
    assert all(node.target is not torch.ops.aten.add.Tensor for node in graph.graph.nodes)
    torch.testing.assert_close(graph(value), pattern(value))


def test_legacy_matcher_constructor_keeps_real_rewrite_support(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    module = importlib.import_module("mindiesd.compilation.passes.pattern_match_pass")
    constructor = module.PatternMatcherPass
    calls = []

    def legacy_constructor(**kwargs):
        calls.append(kwargs)
        if kwargs:
            raise TypeError("pass_name not supported")
        return constructor()

    monkeypatch.setattr(module, "PatternMatcherPass", legacy_constructor)
    matcher = module.PatternMatchPass()

    def pattern(x):
        return x + x

    def replacement(x):
        return x * 2

    matcher.register_pattern("legacy-constructor-rewrite", pattern, replacement, [torch.empty(2, 3, device="meta")])
    value = torch.randn(2, 3)
    graph = make_fx(pattern)(value)
    assert matcher(graph) is graph
    graph.recompile()
    assert calls == [{"pass_name": "pattern_match_pass"}, {}]
    assert any(node.target is torch.ops.aten.mul.Tensor for node in graph.graph.nodes)
    assert all(node.target is not torch.ops.aten.add.Tensor for node in graph.graph.nodes)
    torch.testing.assert_close(graph(value), value + value)
