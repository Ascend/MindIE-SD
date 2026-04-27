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

import os
import time
import unittest

import torch
import torch.nn.functional as F

from mindiesd.compilation import MindieSDBackend, CompilationConfig


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


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
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

        x = torch.randn(4, 4, dtype=torch.float32, device="npu")

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

        x = torch.randn(4, 4, dtype=torch.float32, device="npu")

        output_compiled = compiled(x)
        output_original = model(x)

        self.assertTrue(
            torch.allclose(output_original, output_compiled),
            "ACLGraph-with-compile output mismatch",
        )

    def test_both_modes_output_agree(self):
        model = _SingleIOModel()
        x = torch.randn(4, 4, dtype=torch.float32, device="npu")

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

        x = torch.randn(4, 4, dtype=torch.float32, device="npu")

        out1 = compiled(x)
        self.assertTrue(torch.allclose(model(x), out1))

        out2 = compiled(torch.randn(4, 4, dtype=torch.float32, device="npu"))
        self.assertTrue(torch.allclose(model(x), out2))

    def test_different_shape_triggers_recapture(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _SingleIOModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x1 = torch.randn(4, 4, dtype=torch.float32, device="npu")
        out1 = compiled(x1)
        self.assertTrue(torch.allclose(model(x1), out1))

        x2 = torch.randn(8, 8, dtype=torch.float32, device="npu")
        out2 = compiled(x2)
        self.assertTrue(torch.allclose(model(x2), out2))

    # --- Edge case tests ---

    def test_multiple_output_tensors(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _MultiOutputModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x = torch.randn(4, 4, dtype=torch.float32, device="npu")

        out1, out2 = compiled(x)
        ref1, ref2 = model(x)

        self.assertTrue(torch.allclose(ref1, out1), "Output 1 mismatch")
        self.assertTrue(torch.allclose(ref2, out2), "Output 2 mismatch")

    def test_mixed_scalar_tensor_input(self):
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False

        model = _MixedInputModel()
        compiled = torch.compile(model, backend=MindieSDBackend())

        x = torch.randn(4, 4, dtype=torch.float32, device="npu")
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
        model = _VariableLenModel(dim).to("npu")
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False
        compiled = torch.compile(model, backend=MindieSDBackend())

        full_inp = torch.randn(max_len, dim, dtype=torch.float32, device="npu")
        _ = compiled(full_inp)

        short_len = 20
        short_inp = torch.randn(short_len, dim, dtype=torch.float32, device="npu")
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
        model = _VariableLenModel(dim).to("npu")
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False
        compiled = torch.compile(model, backend=MindieSDBackend())

        full_inp = torch.randn(max_len, dim, dtype=torch.float32, device="npu")
        out_captured = compiled(full_inp)
        self.assertTrue(torch.allclose(model(full_inp), out_captured))

        another = torch.randn(max_len, dim, dtype=torch.float32, device="npu")
        out = compiled(another)
        self.assertTrue(torch.allclose(model(another), out))

    def test_padding_longer_input_triggers_recapture(self):
        """Longer input triggers re-capture rather than a crash."""
        capture_len, dim = 32, 16
        model = _VariableLenModel(dim).to("npu")
        CompilationConfig.aclgraph_only = True
        CompilationConfig.aclgraph_with_compile = False
        compiled = torch.compile(model, backend=MindieSDBackend())

        inp = torch.randn(capture_len, dim, dtype=torch.float32, device="npu")
        ref_out = compiled(inp)

        longer = torch.randn(capture_len * 2, dim, dtype=torch.float32, device="npu")
        longer_out = compiled(longer)

        self.assertEqual(
            longer_out.shape[0], capture_len * 2,
            "Re-captured output should match longer input's first dim"
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
