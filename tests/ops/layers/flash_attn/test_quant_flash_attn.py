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

"""Interface and NPU tests for FP8/MXFP8/MXFP4 attention and legacy compatibility."""

import importlib
import sys
import unittest
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
import torch_npu
from mindiesd import quant_attention
from mindiesd.quantization.mode import FP8FAMode
from mindiesd.utils.exception import ParametersInvalid
from mindiesd.utils.get_platform import is_a5_device

fp8 = importlib.import_module("mindiesd.layers.flash_attn.fused_infer_attention_score")
mxfp4 = importlib.import_module("mindiesd.layers.flash_attn.quant_attention_mxfp4")


class TestQuantAttentionMxfp8Npu(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "npu"), "Requires TorchNPU on an A5 device.")
    def test_real_npu_unequal_lengths_and_gqa(self):
        if not torch.npu.is_available() or not is_a5_device():
            self.skipTest("Requires an available A5 device.")
        # Constant V per batch/head has a known attention result for any Q/K.
        # Distinct values detect batch/head mixing without a quantization-error baseline.
        head_values = torch.tensor([[1.0, -1.0], [0.5, -0.5]], device="npu", dtype=torch.bfloat16)
        for layout, (q_len, kv_len) in product(("BNSD", "BSND"), ((128, 256), (256, 128))):
            with self.subTest(layout=layout, lengths=(q_len, kv_len)):
                q = torch.randn(2, 4, q_len, 64, device="npu", dtype=torch.bfloat16) * 0.1
                k = torch.randn(2, 2, kv_len, 64, device="npu", dtype=torch.bfloat16) * 0.1
                v = head_values.reshape(2, 2, 1, 1).expand(2, 2, kv_len, 64).contiguous()
                expected = head_values.repeat_interleave(2, dim=1).reshape(2, 4, 1, 1).expand_as(q)
                if layout == "BSND":
                    q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
                    expected = expected.transpose(1, 2)
                output = quant_attention(q, k, v, precision="mxfp8", layout=layout)
                actual = output.cpu()
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected.cpu(), rtol=0.02, atol=0.02)
