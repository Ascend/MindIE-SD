#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU contracts for graph backend routing, compatibility and capture boundaries."""

import builtins
import contextlib
import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch import fx

from mindiesd.compilation.compiliation_config import BackendMode, CompilationConfig, FusionPatterns
from tests.UT.compilation.cpu_graph_runtime import CapturedGraph, capture
from tests.UT.compilation.patterns import cpu_fused_ops

acl = importlib.import_module("mindiesd.compilation.aclgraph_backend")
backend = importlib.import_module("mindiesd.compilation.mindie_sd_backend")
npu_ex = importlib.import_module("mindiesd.compilation.npugraph_ex_backend")
splitter = importlib.import_module("mindiesd.compilation.graph_splitter")


@pytest.fixture(autouse=True)
def isolated_npugraph_hints(monkeypatch):
    monkeypatch.setattr(npu_ex, "_option_hint_done", set())


def test_compilation_public_exports_are_canonical():
    package = importlib.import_module("mindiesd.compilation")
    assert set(package.__all__) == {"MindieSDBackend", "CompilationConfig", "BackendMode"}
    assert package.BackendMode is BackendMode
    assert package.CompilationConfig is CompilationConfig
    assert package.MindieSDBackend is backend.MindieSDBackend


@pytest.fixture(name="cpu_capture")
def fixture_cpu_capture(monkeypatch):
    """Reuse the shared replay double; production input copying remains active."""
    graphs = []

    def make_graph():
        graph = CapturedGraph()
        graphs.append(graph)
        return graph

    monkeypatch.setattr(acl, "_global_graph_pool", None)
    monkeypatch.setattr(acl, "npu_graph_available", True)
    monkeypatch.setattr(torch.npu, "NPUGraph", make_graph)
    monkeypatch.setattr(torch.npu, "graph", capture)
    monkeypatch.setattr(torch.npu, "graph_pool_handle", Mock(return_value="cpu-pool"))
    monkeypatch.setattr(torch.npu, "Stream", MagicMock(side_effect=MagicMock))
    monkeypatch.setattr(torch.npu, "current_stream", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(torch.npu, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(CompilationConfig, "aclgraph_lazy_capture", False)
    monkeypatch.setattr(CompilationConfig, "safe_output_mode", True)
    monkeypatch.setattr(CompilationConfig, "aclgraph_max_entries", 0)
    return graphs


@pytest.mark.parametrize("options", [None, {}, {"custom": 7}])
def test_npugraph_options_safe_defaults_preserve_caller(options, monkeypatch):
    monkeypatch.setattr(npu_ex, "_option_hint_done", set())
    info = Mock()
    monkeypatch.setattr(npu_ex.logger, "info", info)
    original = None if options is None else dict(options)
    resolved = npu_ex._resolve_npugraph_ex_options(options)
    assert resolved["reuse_graph_pool_in_same_fx"] is False
    assert resolved["clone_output"] is True
    assert options == original
    assert npu_ex._resolve_npugraph_ex_options(options) == resolved
    assert info.call_count == 2


def test_npugraph_explicit_options_warn_once(monkeypatch):
    monkeypatch.setattr(npu_ex, "_option_hint_done", set())
    warning = Mock()
    monkeypatch.setattr(npu_ex.logger, "warning", warning)
    unsafe = {"reuse_graph_pool_in_same_fx": True, "clone_output": False}
    for _ in range(2):
        assert npu_ex._resolve_npugraph_ex_options(unsafe) == unsafe
    assert warning.call_count == 2
    safe = {"reuse_graph_pool_in_same_fx": False, "clone_output": True}
    assert npu_ex._resolve_npugraph_ex_options(safe) == safe
    assert warning.call_count == 2


@pytest.mark.parametrize("name", ["npu_dtype_cast", "_npu_dtype_cast"])
def test_resolve_native_cast_by_available_operator(name):
    native_cast = Mock()
    npu_ops = SimpleNamespace(**{name: SimpleNamespace(default=native_cast)})
    assert npu_ex._resolve_npu_cast_op(SimpleNamespace(ops=SimpleNamespace(npu=npu_ops))) is native_cast
    assert npu_ex._resolve_npu_cast_op(SimpleNamespace(ops=SimpleNamespace(npu=None))) is None
    assert npu_ex._resolve_npu_cast_op(SimpleNamespace(ops=SimpleNamespace(npu=SimpleNamespace()))) is None


def _cpu_dtype_cast(value, dtype):
    return value.to(dtype)


@pytest.mark.parametrize("keyword", [False, True])
def test_type_raw_rewrite_preserves_cast_result(keyword, monkeypatch):
    graph = fx.Graph()
    value = graph.placeholder("value")
    node = graph.call_method("type_raw", args=(value, torch.float64))
    graph.output(node)
    module = fx.GraphModule({}, graph)
    if keyword:
        # Code generation requires a positional receiver. Exercise the repair of
        # imported keyword-form nodes after constructing the valid container.
        node.args = ()
        node.kwargs = {"self": value, "dtype": torch.float64}
    monkeypatch.setattr(npu_ex, "_resolve_npu_cast_op", lambda torch_module: _cpu_dtype_cast)
    assert npu_ex._patch_type_raw_for_npugraph_ex(module) is module
    assert all(node.target != "type_raw" for node in graph.nodes)
    result = module(torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(result, torch.arange(4, dtype=torch.float64))


def test_type_raw_missing_arguments_and_operator_leave_graph_unchanged(monkeypatch):
    module = fx.symbolic_trace(lambda value: value + 1)
    original = module.code
    monkeypatch.setattr(npu_ex, "_resolve_npu_cast_op", lambda torch_module: None)
    assert npu_ex._patch_type_raw_for_npugraph_ex(module) is module
    assert module.code == original
    graph = fx.Graph()
    value = graph.placeholder("value")
    graph.output(graph.call_method("type_raw", args=(value,)))
    malformed = fx.GraphModule({}, graph)
    monkeypatch.setattr(npu_ex, "_resolve_npu_cast_op", lambda torch_module: _cpu_dtype_cast)
    warning = Mock()
    monkeypatch.setattr(npu_ex.logger, "warning", warning)
    assert npu_ex._patch_type_raw_for_npugraph_ex(malformed) is malformed
    assert any(node.target == "type_raw" for node in graph.nodes)
    warning.assert_called_once()


def test_type_raw_rewrite_error_is_reported_without_losing_graph(monkeypatch):
    module = fx.symbolic_trace(lambda value: value + 1)
    monkeypatch.setattr(npu_ex, "_resolve_npu_cast_op", Mock(side_effect=RuntimeError("cast unavailable")))
    warning = Mock()
    monkeypatch.setattr(npu_ex.logger, "warning", warning)
    assert npu_ex._patch_type_raw_for_npugraph_ex(module) is module
    assert warning.call_args.kwargs["exc_info"] is True
    assert "cast unavailable" in str(warning.call_args)


@pytest.mark.parametrize("style", ["options", "keywords", "fallback", "opaque"])
def test_compile_fx_signature_compatibility(style, monkeypatch):
    seen = []

    if style == "options":

        def compile_fx(graph, inputs, options):
            seen.append((graph, inputs, options))
            return graph
    elif style == "keywords":

        def compile_fx(graph, inputs, **kwargs):
            seen.append((graph, inputs, kwargs))
            return graph
    else:

        def compile_fx(graph, inputs, settings):
            seen.append((graph, inputs, settings))
            return graph

    if style == "opaque":
        monkeypatch.setattr(npu_ex.inspect, "signature", Mock(side_effect=ValueError("opaque signature")))
    fake_torch = SimpleNamespace(npu=SimpleNamespace(npugraph_ex=SimpleNamespace(compile_fx=compile_fx)))
    graph, inputs, options = object(), [object()], {"force_eager": True}
    assert npu_ex._call_npugraph_ex_compile_fx(fake_torch, graph, inputs, options) is graph
    assert seen == [(graph, inputs, options)]


def test_fx_dump_is_optional_and_reports_write_errors(tmp_path, monkeypatch):
    module = fx.symbolic_trace(lambda value: value + 1)
    monkeypatch.delenv("MINDIE_SD_DUMP_FX_DIR", raising=False)
    npu_ex._dump_fx_graph(module, "disabled")
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("MINDIE_SD_DUMP_FX_DIR", str(tmp_path))
    npu_ex._dump_fx_graph(module, "cpu_graph")
    paths = list(tmp_path.glob("cpu_graph_*.py"))
    assert len(paths) == 1
    assert paths[0].read_text(encoding="utf-8") == module.code
    warning = Mock()
    monkeypatch.setattr(npu_ex.logger, "warning", warning)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "open", Mock(side_effect=OSError("read-only destination")))
        npu_ex._dump_fx_graph(module, "failed")
    warning.assert_called_once()


def npu_rotary_mul(value, cos, sin, mode):
    return cpu_fused_ops.rope(value, cos, sin, mode)


def test_npugraph_fusion_options_rewrite_real_rope_chain(monkeypatch):
    monkeypatch.setattr(
        CompilationConfig,
        "fusion_patterns",
        FusionPatterns(enable_gelu_quant_fusion=True, enable_rope_cast_fusion=True),
    )
    graph = fx.Graph()
    value, cos, sin = (graph.placeholder(name) for name in ("value", "cos", "sin"))
    for node in (value, cos, sin):
        node.meta["val"] = torch.empty(2, 4, dtype=torch.bfloat16)
    cast = graph.call_function(torch.ops.aten._to_copy.default, (value,), {"dtype": torch.float32})
    rope = graph.call_function(npu_rotary_mul, (cast, cos, sin, "half"))
    output = graph.call_function(torch.ops.aten._to_copy.default, (rope,), {"dtype": torch.bfloat16})
    graph.output((output,))
    module = fx.GraphModule({}, graph)
    values = (
        torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) / 8,
        torch.full((2, 4), 0.5, dtype=torch.bfloat16),
        torch.full((2, 4), 0.25, dtype=torch.bfloat16),
    )
    expected = module(*values)[0]
    assert npu_ex._apply_fusion_patterns(module) is module
    assert all(node.target is not torch.ops.aten._to_copy.default for node in graph.nodes)
    (fused_rope,) = (node for node in graph.nodes if node.target is npu_rotary_mul)
    assert fused_rope.args == (value, cos, sin, "half")
    assert next(node for node in graph.nodes if node.op == "output").args == ((fused_rope,),)
    torch.testing.assert_close(module(*values)[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("force_eager,capture_available", [(False, False), (True, False), (True, True)])
def test_npugraph_backend_real_aot_and_optional_capture(force_eager, capture_available, cpu_capture, monkeypatch):
    compiled_graphs = []

    def compile_fx(graph, inputs, options):
        compiled_graphs.append((graph, options))
        return graph.forward

    monkeypatch.setattr(torch.npu, "npugraph_ex", SimpleNamespace(compile_fx=compile_fx), raising=False)
    monkeypatch.setattr(acl, "npu_graph_available", capture_available)
    configured = {"force_eager": force_eager}
    monkeypatch.setattr(CompilationConfig, "npugraph_ex_options", configured)
    module = fx.symbolic_trace(lambda value, scale: (value.sin() * scale,))
    values = torch.arange(4, dtype=torch.float32)
    compiled = npu_ex.create_npugraph_ex_backend()(module, [values, 2])
    for offset in (0, 1):
        actual = values + offset
        torch.testing.assert_close(compiled(actual, 2)[0], actual.sin() * 2)
    assert compiled_graphs
    assert compiled_graphs[0][1]["force_eager"] is force_eager
    assert configured == {"force_eager": force_eager}
    assert bool(cpu_capture) is (force_eager and capture_available)


def test_capture_static_inputs_scalar_keys_and_validation(cpu_capture, monkeypatch):
    node = SimpleNamespace(hint=5)
    symbolic = torch.SymInt(node)
    assert acl._symint_hint(object()) is None
    assert acl._symint_hint(SimpleNamespace(node=SimpleNamespace(hint="unknown"))) is None
    assert acl._concrete_scalar_input(symbolic) == 5
    monkeypatch.setattr(acl, "_symint_materialized_warned", False)
    assert acl._materialize_scalar_input(symbolic, 2) == 5
    assert acl._materialize_scalar_input(symbolic, 2) == 5
    assert acl._get_input_shape([torch.ones(2), 3, "tag", symbolic, None]) == (
        torch.Size([2]),
        3,
        "tag",
        5,
        (),
    )
    assert acl._safe_output(None, is_last=True) is None
    values = torch.arange(4, dtype=torch.float32)
    assert acl._safe_output(values, is_last=False) is values
    compiled = acl.create_aclgraph_backend()(lambda value, scale: value * scale, [values, 2])
    torch.testing.assert_close(compiled(values + 1, 2), (values + 1) * 2)
    entry = acl._ACLGraphEntry(cpu_capture[0], [values.detach().clone(), 2], values)
    with pytest.raises(RuntimeError, match="scalar input mismatch"):
        acl._validate_and_copy_inputs(entry, [values, 3])
    acl._validate_and_copy_inputs(entry, [None, 2])


def test_segmented_replay_preserves_nested_cross_segment_outputs(cpu_capture):
    module = fx.symbolic_trace(lambda value: {"early": value + 1, "late": [(value + 1).sin(), "constant"]})
    breakpoint_name = next(node.name for node in module.graph.nodes if node.target == "sin")
    _, segments, output_spec = splitter.split_graph_by_breakpoints(module, [breakpoint_name])
    compiled = acl.create_segmented_aclgraph_backend(module, segments, output_spec)
    first = torch.arange(4, dtype=torch.float32)
    initial = compiled(first)
    later = compiled(first + 2)
    torch.testing.assert_close(initial["early"], first + 1)
    torch.testing.assert_close(initial["late"][0], (first + 1).sin())
    torch.testing.assert_close(later["early"], first + 3)
    torch.testing.assert_close(later["late"][0], (first + 3).sin())
    assert later["late"][1] == "constant"


@pytest.mark.parametrize(
    "value,count,index,message",
    [
        (None, 1, 0, "produced no outputs"),
        (torch.ones(1), 1, 1, "single output"),
        (torch.ones(1), 2, 0, "declares 2 outputs"),
        ((torch.ones(1),), 2, 1, "produced 1 outputs"),
    ],
)
def test_segmented_capture_rejects_invalid_output_metadata(value, count, index, message, cpu_capture):
    module = fx.symbolic_trace(lambda value: value + 1)
    segments = [splitter.GraphSegment(lambda value: value, [("orig", "value")], count)]
    # The segment callable returns the selected malformed value, without mocking validation.
    segments[0].gm = lambda arg: value
    output_spec = splitter.OutputSpec(("leaf", 0), [(0, index)])
    compiled = acl.create_segmented_aclgraph_backend(module, segments, output_spec)
    with pytest.raises(RuntimeError, match=message):
        compiled(torch.zeros(1))


def test_capture_restores_patches_after_model_failure(cpu_capture):
    import gc

    original = gc.collect
    with pytest.raises(RuntimeError, match="model failure"):
        acl._capture_single_graph(Mock(side_effect=RuntimeError("model failure")), [])
    assert gc.collect is original


@pytest.mark.parametrize(
    "mode", [BackendMode.ACLGRAPH_ONLY, BackendMode.ACLGRAPH_WITH_COMPILE, BackendMode.NPUGRAPH_EX]
)
def test_backend_unavailable_falls_back_to_compiler(mode, monkeypatch):
    monkeypatch.setattr(CompilationConfig, "backend_mode", mode)
    monkeypatch.setattr(backend, "npu_graph_available", False)
    monkeypatch.setattr(backend, "npugraph_ex_available", False)
    instance = backend.MindieSDBackend()
    compiled = Mock()
    monkeypatch.setattr(instance, "compile", compiled)
    graph, inputs = object(), [object()]
    assert instance(graph, inputs) is compiled.return_value
    compiled.assert_called_once_with(graph, inputs)


@pytest.mark.parametrize("mode", [BackendMode.ACLGRAPH_ONLY, BackendMode.NPUGRAPH_EX])
def test_backend_routes_patterns_and_breakpoints(mode, cpu_capture, monkeypatch):
    monkeypatch.setattr(CompilationConfig, "backend_mode", mode)
    monkeypatch.setattr(CompilationConfig, "aclgraph_enable_patterns", True)
    monkeypatch.setattr(
        CompilationConfig,
        "fusion_patterns",
        FusionPatterns(
            enable_gelu_quant_fusion=True,
            enable_rope_cast_fusion=True,
        ),
    )
    monkeypatch.setattr(backend, "npu_graph_available", True)
    monkeypatch.setattr(backend, "npugraph_ex_available", True)
    monkeypatch.setattr(CompilationConfig, "aclgraph_breakpoints", ["sin"])
    module = fx.symbolic_trace(lambda value: (value + 1).sin())
    values = torch.arange(4, dtype=torch.float32)
    if mode == BackendMode.NPUGRAPH_EX:
        # Only the device-specific final compiler is doubled; routing and pattern passes run.
        factory = Mock(return_value=lambda graph, inputs: graph)
        monkeypatch.setattr(backend, "create_npugraph_ex_backend", factory)
    compiled = backend.MindieSDBackend()(module, [values])
    torch.testing.assert_close(compiled(values), (values + 1).sin())
    if mode == BackendMode.NPUGRAPH_EX:
        factory.assert_called_once_with()
    else:
        assert len(cpu_capture) >= 2


def test_compile_freezing_preserves_selected_inputs_and_boxed_contract(monkeypatch):
    monkeypatch.setattr(CompilationConfig, "enable_freezing", True)
    module = fx.symbolic_trace(lambda value: (value + 3,))
    values = torch.arange(4, dtype=torch.float32)
    monkeypatch.setattr(backend, "freeze", lambda original, graph, inputs: (module, [1]))

    def aot_autograd(*, fw_compiler, decompositions):
        return fw_compiler

    monkeypatch.setattr(backend, "aot_autograd", aot_autograd)
    compiled = backend.MindieSDBackend().compile(module, [torch.zeros(1), values])
    assert compiled._boxed_call is True
    boxed = [torch.zeros(1), values]
    torch.testing.assert_close(compiled(boxed)[0], values + 3)
    assert not boxed


def test_dynamo_probe_patch_only_changes_failed_optional_probes(monkeypatch):
    import torch.utils._triton as triton_utils
    import torch._dynamo.utils as dynamo_utils
    import torch._dynamo.variables.builder as dynamo_builder

    monkeypatch.setenv("MINDIE_SD_DYNAMO_COMPAT_PATCHES", "1")
    monkeypatch.setattr(backend, "_dynamo_triton_patch_done", False)
    failing = Mock(side_effect=RuntimeError("no CUDA capability"))
    healthy = Mock(return_value=True)

    def requires_device(device):
        raise AssertionError("required-argument probe must not be executed")

    # The compatibility helper updates multiple imported namespaces; register every
    # existing binding with monkeypatch so no process-wide patch escapes this test.
    for module in (triton_utils, dynamo_utils, dynamo_builder):
        for name in (
            "_device_supports_tma",
            "cuda_extra_check",
            "has_triton_experimental_host_tma",
            "has_triton",
            "has_triton_tma",
            "has_triton_tma_bulk_copy",
            "has_triton_stable_tma_api",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(triton_utils, "_device_supports_tma", failing, raising=False)
    monkeypatch.setattr(triton_utils, "cuda_extra_check", healthy, raising=False)
    monkeypatch.setattr(triton_utils, "has_triton_experimental_host_tma", requires_device, raising=False)
    backend._patch_dynamo_triton_device_probes()
    assert triton_utils._device_supports_tma() is False
    assert triton_utils.cuda_extra_check is healthy
    assert triton_utils.has_triton_experimental_host_tma is requires_device
    assert healthy.call_count == 1
    backend._patch_dynamo_triton_device_probes()
    assert healthy.call_count == 1


def test_restore_native_getenv_uses_saved_os_reference(monkeypatch):
    original = os.getenv

    def wrapped_getenv(name, default=None):
        return "wrapped"

    wrapped_getenv.__module__ = "torch_npu.utils.patch_getenv"
    saved = SimpleNamespace(original=original)
    import torch_npu

    utils = ModuleType("torch_npu.utils")
    utils.patch_getenv = saved
    monkeypatch.setitem(sys.modules, "torch_npu.utils", utils)
    monkeypatch.setattr(torch_npu, "utils", utils, raising=False)
    monkeypatch.setattr(os, "getenv", wrapped_getenv)
    monkeypatch.setattr(backend, "_torchnpu_getenv_restore_done", False)
    monkeypatch.setenv("MINDIE_SD_DYNAMO_COMPAT_PATCHES", "1")
    backend._restore_native_getenv()
    assert os.getenv is original
    backend._restore_native_getenv()
    assert os.getenv is original
