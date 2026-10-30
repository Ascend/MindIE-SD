# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

import math
import os
import unittest
from unittest.mock import patch

import torch

from tests.device import DEVICE_ID
from tests.ops.layers.flash_attn.laser_attention.laser_attention_layer_common import (
    LAYERS,
    LegacyLaserAttention,
    la,
    make_inputs,
)
from mindiesd.layers.flash_attn.common import AttentionParam


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE") == "CPU", "NPU integration test")
class TestLaserAttentionLayerNpu(unittest.TestCase):
    def test_business_shape_current_and_legacy_vs_fas(self):
        import torch_npu

        if not torch.npu.is_available():
            self.skipTest("NPU is unavailable")
        torch.npu.set_device(DEVICE_ID)
        if la.is_a5_device():
            self.skipTest("LA is unsupported on A5")
        param, query, key, value = make_inputs(f"npu:{torch.npu.current_device()}")
        with torch.inference_mode():
            golden = (
                torch_npu.npu_fusion_attention(
                    query,
                    key,
                    value,
                    head_num=24,
                    input_layout="BSND",
                    scale=128**-0.5,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                )[0]
                .float()
                .reshape(-1)
            )
            for name, forward in LAYERS.items():
                with self.subTest(path=name):
                    output = forward(param, query, key, value, scale=128**-0.5)
                    self.assertEqual(output.shape, query.shape)
                    self.assertEqual(output.dtype, torch.bfloat16)
                    actual = output.float().reshape(-1)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    cosine = torch.nn.functional.cosine_similarity(actual, golden, dim=0).item()
                    norm_ratio = (torch.linalg.vector_norm(actual) / torch.linalg.vector_norm(golden)).item()
                    print(f"{name}: cosine={cosine:.8f}, norm_ratio={norm_ratio:.8f}")
                    self.assertGreaterEqual(cosine, 0.99)
                    self.assertGreaterEqual(norm_ratio, 0.9)
                    self.assertLessEqual(norm_ratio, 1.1)
