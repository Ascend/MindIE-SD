#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Affine residual LayerNorm must preserve the residual and respect graph users."""

import os
import unittest
from unittest.mock import patch

import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module


class ResidualLayerNormModel(torch.nn.Module):
    def __init__(self, hidden_size=3072, return_statistics=False, autocast=False, eps=1e-6):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, eps=eps)
        self.return_statistics = return_statistics
        self.autocast = autocast

    def forward(self, x, residual):
        summed = x + residual
        if self.return_statistics:
            normalized, mean, rstd = torch.native_layer_norm(
                summed, self.norm.normalized_shape, self.norm.weight, self.norm.bias, self.norm.eps
            )
            return normalized, summed, mean, rstd
        if self.autocast:
            with torch.autocast("npu", dtype=torch.bfloat16):
                return self.norm(summed).to(x.dtype), summed
        return self.norm(summed), summed


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Requires an Ascend NPU.")
class TestAddLayerNormPattern(unittest.TestCase):
    def _run(
        self,
        residual_shape,
        expect_fusion,
        return_statistics=False,
        autocast=False,
        dtype=torch.bfloat16,
        sequence=2048,
        hidden_size=3072,
        noncontiguous=False,
        eps=1e-6,
    ):
        torch.manual_seed(20261019)
        torch._dynamo.reset()
        model = ResidualLayerNormModel(hidden_size, return_statistics, autocast, eps)
        model = model.to("npu", dtype=dtype).eval()
        x = torch.randn(1, sequence, hidden_size * (2 if noncontiguous else 1), device="npu", dtype=dtype)
        if noncontiguous:
            x = x[..., ::2]
        residual = torch.randn(residual_shape, device="npu", dtype=dtype)
        targets = []
        original = MindieSDBackend.apply_pattern_match_passes.__func__

        def observe(cls, graph, inputs):
            original(cls, graph, inputs)
            targets.extend(str(node.target) for node in graph.graph.nodes if node.op == "call_function")

        with (
            torch.inference_mode(),
            patch.object(MindieSDBackend, "apply_pattern_match_passes", classmethod(observe)),
        ):
            expected = model(x, residual)
            compiled = torch.compile(model, backend=MindieSDBackend(), fullgraph=True)
            actual = compiled(x, residual)
            repeated = compiled(x, residual)
            torch.npu.synchronize()

        self.assertEqual(any("npu_add_layer_norm" in target for target in targets), expect_fusion, targets)
        self.assertTrue(
            torch.equal(actual[1], expected[1]),
            "The residual output must remain bitwise identical.",
        )
        self.assertEqual(actual[0].dtype, expected[0].dtype)
        self.assertTrue(torch.equal(actual[0], repeated[0]), "Repeated fused evaluation must be deterministic.")
        delta = actual[0].float() - expected[0].float()
        relative_l2 = delta.norm() / expected[0].float().norm().clamp_min(1e-12)
        self.assertLessEqual(relative_l2.item(), 1e-6 if dtype == torch.float32 else 0.005)
        for expected_statistic, actual_statistic in zip(expected[2:], actual[2:]):
            self.assertTrue(torch.equal(expected_statistic, actual_statistic))

    def test_affine_norm_preserves_shared_residual(self):
        self._run((1, 2048, 3072), expect_fusion=True)

    def test_broadcast_residual_is_not_fused(self):
        self._run((1, 1, 3072), expect_fusion=False)

    def test_autocast_norm_before_bfloat16_linear(self):
        self._run((1, 2048, 3072), expect_fusion=True, autocast=True)

    def test_float32_affine_norm(self):
        self._run((1, 2048, 3072), expect_fusion=True, dtype=torch.float32)

    def test_unvalidated_float16_is_not_fused(self):
        self._run((1, 2048, 3072), expect_fusion=False, dtype=torch.float16)

    def test_short_sequence_is_not_fused(self):
        self._run((1, 256, 3072), expect_fusion=False, sequence=256)

    def test_other_hidden_sizes_are_fused(self):
        for hidden_size in (768, 4096, 5120):
            with self.subTest(hidden_size=hidden_size):
                self._run((1, 2048, hidden_size), expect_fusion=True, hidden_size=hidden_size)

    def test_noncontiguous_input_is_not_fused(self):
        self._run((1, 2048, 3072), expect_fusion=False, noncontiguous=True)

    def test_different_epsilon_is_not_fused(self):
        self._run((1, 2048, 3072), expect_fusion=False, eps=1e-5)

    def test_live_statistics_are_not_discarded(self):
        self._run((1, 2048, 3072), expect_fusion=False, return_statistics=True)


if __name__ == "__main__":
    unittest.main()
