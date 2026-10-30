#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from mindiesd.compilation import MindieSDBackend, CompilationConfig  # pylint: disable=no-name-in-module


class _SingleIOModel(torch.nn.Module):
    def forward(self, x):
        return x * 2 + 1


class _MultiOutputModel(torch.nn.Module):
    def forward(self, x):
        return x * 2, x + 1


class _MixedInputModel(torch.nn.Module):
    def forward(self, x, scale):
        return x * scale


class _VariableLenModel(torch.nn.Module):
    """Model accepting variable-length 2D input [L, D] where L varies."""

    def __init__(self, dim):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


class TestAclGraphEventOrdering(unittest.TestCase):
    def test_copy_uses_events_without_host_synchronize(self):
        aclgraph_backend = importlib.import_module("mindiesd.compilation.aclgraph_backend")
        current_stream = MagicMock()
        copy_stream = MagicMock()
        input_ready_event = object()
        copy_complete_event = object()
        current_stream.record_event.return_value = input_ready_event
        copy_stream.record_event.return_value = copy_complete_event
        entry = aclgraph_backend._ACLGraphEntry(
            aclgraph=MagicMock(),
            static_inputs=[],
            output=MagicMock(),
            copy_stream=copy_stream,
        )
        static_input = torch.zeros(2, 2)
        dynamic_input = torch.ones(2, 2)

        with (
            patch.object(aclgraph_backend.torch.npu, "current_stream", return_value=current_stream),
            patch.object(aclgraph_backend.torch.npu, "stream", return_value=contextlib.nullcontext()),
        ):
            aclgraph_backend._copy_inputs_on_stream(entry, [(static_input, dynamic_input)])

        current_stream.synchronize.assert_not_called()
        copy_stream.wait_event.assert_called_once_with(input_ready_event)
        copy_stream.record_event.assert_called_once_with()
        current_stream.wait_event.assert_called_once_with(copy_complete_event)
        self.assertTrue(torch.equal(static_input, dynamic_input))


class TestAclGraphBackend(unittest.TestCase):
    def setUp(self):
        self._orig_aclgraph_only = CompilationConfig.aclgraph_only
        self._orig_aclgraph_with_compile = CompilationConfig.aclgraph_with_compile

    def tearDown(self):
        CompilationConfig.aclgraph_only = self._orig_aclgraph_only
        CompilationConfig.aclgraph_with_compile = self._orig_aclgraph_with_compile

    # ---  Output consistency tests  ---

    def test_aclgraph_only_output_consistent(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _SingleIOModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x = torch.randn(4, 4, dtype=torch.float32, device="cpu")

        output_compiled = compiled(x)
        output_original = model(x)

        self.assertTrue(
            torch.allclose(output_original, output_compiled),
            "ACLGraph-only output mismatch",
        )

    def test_aclgraph_with_compile_output_consistent(self):
        CompilationConfig.aclgraph_only = False
        CompilationConfig.aclgraph_with_compile = True

        model = _SingleIOModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x = torch.randn(4, 4, dtype=torch.float32, device="cpu")

        output_compiled = compiled(x)
        output_original = model(x)

        self.assertTrue(
            torch.allclose(output_original, output_compiled),
            "ACLGraph-with-compile output mismatch",
        )

    def test_both_modes_output_agree(self):
        model = _SingleIOModel()
        x = torch.randn(4, 4, dtype=torch.float32, device="cpu")

        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False
        compiled_only = torch.compile(model, backend=MindieSDBackend())
        out_only = compiled_only(x)

        CompilationConfig.aclgraph_only = False
        CompilationConfig.aclgraph_with_compile = True
        compiled_with = torch.compile(model, backend=MindieSDBackend())
        out_with = compiled_with(x)

        self.assertTrue(
            torch.allclose(out_only, out_with),
            "Two ACLGraph modes should produce same output",
        )

    # --- Graph caching tests ---

    def test_same_shape_reuses_cached_graph(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _SingleIOModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x = torch.randn(4, 4, dtype=torch.float32, device="cpu")

        out1 = compiled(x)
        self.assertTrue(torch.allclose(model(x), out1))

        out2 = compiled(x)
        self.assertTrue(torch.allclose(model(x), out2))

    def test_same_shape_with_new_storage_uses_async_copy(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _SingleIOModel()
        compiled = torch.compile(model, backend=MindieSDBackend())
        captured = torch.randn(4, 4, dtype=torch.float32, device="cpu")
        dynamic = torch.randn(4, 4, dtype=torch.float32, device="cpu")

        compiled(captured)
        output = compiled(dynamic)
        self.assertTrue(torch.allclose(model(dynamic), output), "Async copy replay output mismatch")

    def test_different_shape_triggers_recapture(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _SingleIOModel()
        # dynamic=False keeps a second shape a plain recompile; dynamic shapes make dynamo build a
        # C++ shape guard, which needs a toolchain that supports the host CPU's -march features.
        compiled = torch.compile(model, backend=MindieSDBackend(), dynamic=False)

        x1 = torch.randn(4, 4, dtype=torch.float32, device="cpu")
        out1 = compiled(x1)
        self.assertTrue(torch.allclose(model(x1), out1))

        x2 = torch.randn(8, 8, dtype=torch.float32, device="cpu")
        out2 = compiled(x2)
        self.assertTrue(torch.allclose(model(x2), out2))

    # --- Edge case tests ---

    def test_multiple_output_tensors(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _MultiOutputModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x = torch.randn(4, 4, dtype=torch.float32, device="cpu")

        out1, out2 = compiled(x)
        ref1, ref2 = model(x)

        self.assertTrue(torch.allclose(ref1, out1), "Output 1 mismatch")
        self.assertTrue(torch.allclose(ref2, out2), "Output 2 mismatch")

    def test_mixed_scalar_tensor_input(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _MixedInputModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x = torch.randn(4, 4, dtype=torch.float32, device="cpu")
        scale = 2.5

        output_compiled = compiled(x, scale)
        output_original = model(x, scale)

        self.assertTrue(
            torch.allclose(output_original, output_compiled),
            "Mixed scalar/tensor input output mismatch",
        )

    # --- Variable-length input tests (external padding) ---

    def test_padding_shorter_input_output_correct(self):
        max_len, dim = 64, 32
        model = _VariableLenModel(dim).to("cpu")
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False
        compiled = torch.compile(model, backend=MindieSDBackend())

        full_inp = torch.randn(max_len, dim, dtype=torch.float32, device="cpu")
        _ = compiled(full_inp)

        short_len = 20
        short_inp = torch.randn(short_len, dim, dtype=torch.float32, device="cpu")
        padding = (0, 0, 0, max_len - short_len)
        padded = F.pad(short_inp, padding)

        output_padded = compiled(padded)
        output_actual = output_padded[:short_len]
        golden = model(short_inp)

        self.assertTrue(
            torch.allclose(golden, output_actual),
            "Padded-path output should match golden after slicing",
        )

    def test_padding_equal_input_no_extra_copy(self):
        max_len, dim = 64, 32
        model = _VariableLenModel(dim).to("cpu")
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False
        compiled = torch.compile(model, backend=MindieSDBackend())

        full_inp = torch.randn(max_len, dim, dtype=torch.float32, device="cpu")
        out_captured = compiled(full_inp)
        self.assertTrue(torch.allclose(model(full_inp), out_captured))

        another = torch.randn(max_len, dim, dtype=torch.float32, device="cpu")
        out = compiled(another)
        self.assertTrue(torch.allclose(model(another), out))

    def test_padding_longer_input_triggers_recapture(self):
        """Longer input triggers re-capture rather than a crash."""
        capture_len, dim = 32, 16
        model = _VariableLenModel(dim).to("cpu")
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False
        # dynamic=False keeps the longer input a plain recompile instead of a dynamic-shape graph,
        # whose C++ shape guard needs a toolchain that supports the host CPU's -march features.
        compiled = torch.compile(model, backend=MindieSDBackend(), dynamic=False)

        inp = torch.randn(capture_len, dim, dtype=torch.float32, device="cpu")
        compiled(inp)  # trigger initial capture

        longer = torch.randn(capture_len * 2, dim, dtype=torch.float32, device="cpu")
        longer_out = compiled(longer)

        self.assertEqual(
            longer_out.shape[0], capture_len * 2, "Re-captured output should match longer input's first dim"
        )
        golden = model(longer)
        self.assertTrue(
            torch.allclose(golden, longer_out, atol=1e-5),
            "Re-captured output should match golden",
        )

    # --- Cleanup test ---

    def test_teardown_restores_config(self):
        self.assertEqual(
            (self._orig_aclgraph_only, self._orig_aclgraph_with_compile),
            (False, False),
            "Original config should default to (False, False)",
        )


if __name__ == "__main__":
    unittest.main()


def test_cpu_capture_eviction_lazy_outputs_and_dtype_validation(monkeypatch):
    import pytest

    acl = importlib.import_module("mindiesd.compilation.aclgraph_backend")
    monkeypatch.setattr(CompilationConfig, "aclgraph_lazy_capture", True)
    monkeypatch.setattr(CompilationConfig, "aclgraph_max_entries", 1)
    monkeypatch.setattr(CompilationConfig, "safe_output_mode", False)
    monkeypatch.setattr(acl.logger, "level", 10)
    calls = []

    def model(x, scale):
        calls.append(tuple(x.shape))
        return [x * scale, "metadata"]

    compiled = acl.create_aclgraph_backend()(model, [torch.zeros(2), 2])
    assert not calls
    x = torch.ones(2)
    result = compiled(x, 2)
    torch.testing.assert_close(result[0], x * 2)
    assert result[1] == "metadata"
    assert compiled(x, 2) is result
    compiled(torch.ones(3), 2)
    compiled(torch.zeros(2), 2)
    assert calls == [(2,), (3,), (2,)]
    with pytest.raises(RuntimeError, match="input mismatch"):
        compiled(torch.zeros(2, dtype=torch.float64), 2)


def test_cpu_capture_safe_mixed_outputs_and_builtin_patch(monkeypatch):
    acl = importlib.import_module("mindiesd.compilation.aclgraph_backend")
    monkeypatch.setattr(CompilationConfig, "safe_output_mode", True)
    compiled = acl.create_aclgraph_backend()(lambda x: (x + 1, "tag"), [torch.zeros(2)])
    first = compiled(torch.zeros(2))
    second = compiled(torch.ones(2))
    assert first[1] == second[1] == "tag"
    torch.testing.assert_close(first[0], torch.ones(2))
    torch.testing.assert_close(second[0], torch.full((2,), 2.0))
    scalar = acl.create_aclgraph_backend()(lambda x: "constant", [torch.zeros(1)])
    assert scalar(torch.zeros(1)) == "constant"
    import builtins

    original = builtins.abs
    with acl._patch_fn("abs", lambda value: 17):
        assert abs(-1) == 17
    assert builtins.abs is original
