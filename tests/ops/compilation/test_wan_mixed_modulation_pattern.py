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

import os
import unittest
from unittest.mock import patch

import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module


class MixedModulationModel(torch.nn.Module):
    def forward(self, x, scale, shift):
        normalized = torch.native_layer_norm(x, [x.shape[-1]], None, None, 1e-6)[0]
        return normalized * (scale + 1) + shift


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Requires an Ascend NPU.")
class TestWanMixedModulationPattern(unittest.TestCase):
    def _run_case(self, sequence, tokenwise, expected_fusion):
        from mindiesd.layers.triton_utils import _TRITON_ON_ASCEND

        if not _TRITON_ON_ASCEND:
            self.skipTest("Mixed modulation fusion requires Triton-Ascend.")
        torch.manual_seed(42)
        torch._dynamo.reset()
        model = MixedModulationModel()
        x = torch.randn(1, sequence, 3072, dtype=torch.float32, device="npu")
        scale = torch.randn(1, sequence if tokenwise else 1, 3072, dtype=torch.bfloat16, device="npu")
        shift = torch.randn_like(scale)
        targets = []
        original = MindieSDBackend.apply_pattern_match_passes.__func__

        def observe(cls, graph, inputs):
            original(cls, graph, inputs)
            targets.extend(str(node.target) for node in graph.graph.nodes)

        with (
            torch.inference_mode(),
            patch.object(MindieSDBackend, "apply_pattern_match_passes", classmethod(observe)),
        ):
            expected = model(x, scale, shift)
            compiled = torch.compile(model, backend=MindieSDBackend(), fullgraph=True)
            actual = compiled(x, scale, shift)
            repeated = compiled(x, scale, shift)
            torch.npu.synchronize()
        self.assertEqual(any("mixed_precision_modulation" in target for target in targets), expected_fusion, targets)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual.bfloat16(), expected.bfloat16()))
        self.assertTrue(torch.equal(actual, repeated))

    def test_tokenwise_modulation_preserves_rounding(self):
        self._run_case(16384, True, True)

    def test_short_sequence_keeps_original_path(self):
        self._run_case(2048, True, False)

    def test_broadcast_modulation_keeps_original_path(self):
        self._run_case(16384, False, False)
