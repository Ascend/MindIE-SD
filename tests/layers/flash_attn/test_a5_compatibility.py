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
"""Unit tests for the A5 backward-compatibility routing in mindiesd.layers.flash_attn.

These tests intentionally only cover the pure-Python routing logic so they can run
without a real NPU. The actual operator behaviour is exercised by other tests on
real hardware.
"""

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from mindiesd.layers.flash_attn.sparse_flash_attn import _resolve_sparse_type_for_a5
from mindiesd.utils.exception import ParametersInvalid


class TestResolveSparseTypeForA5(unittest.TestCase):
    @patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=False)
    def test_non_a5_passthrough(self, _mock):
        self.assertEqual(_resolve_sparse_type_for_a5("rf_v2", 0), ("rf_v2", 0))
        self.assertEqual(_resolve_sparse_type_for_a5("ada_bsa", 1), ("ada_bsa", 1))
        self.assertEqual(_resolve_sparse_type_for_a5("rf_v3", 4), ("rf_v3", 4))
        self.assertEqual(_resolve_sparse_type_for_a5(None, 0), (None, 0))

    @patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    @patch("mindiesd.layers.flash_attn.sparse_flash_attn.logger.info")
    def test_a5_rf_v2_routed_to_rf_v3_with_inner_precise_override(self, mock_info, _mock):
        sparse_type, inner_precise = _resolve_sparse_type_for_a5("rf_v2", 0)
        self.assertEqual(sparse_type, "rf_v3")
        self.assertEqual(inner_precise, 4)
        mock_info.assert_called_once()

    @patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    @patch("mindiesd.layers.flash_attn.sparse_flash_attn.logger.info")
    def test_a5_rf_v2_keeps_inner_precise_when_already_4(self, mock_info, _mock):
        sparse_type, inner_precise = _resolve_sparse_type_for_a5("rf_v2", 4)
        self.assertEqual(sparse_type, "rf_v3")
        self.assertEqual(inner_precise, 4)
        mock_info.assert_called_once()

    @patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    def test_a5_ada_bsa_raises_with_v2_message(self, _mock):
        with self.assertRaises(ParametersInvalid) as ctx:
            _resolve_sparse_type_for_a5("ada_bsa", 0)
        self.assertIn("ada_bsa", str(ctx.exception))
        self.assertIn("v2", str(ctx.exception))

    @patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    def test_a5_rf_v3_and_none_passthrough(self, _mock):
        self.assertEqual(_resolve_sparse_type_for_a5("rf_v3", 4), ("rf_v3", 4))
        self.assertEqual(_resolve_sparse_type_for_a5(None, 0), (None, 0))


class TestDirectSubInterfaceCallsRaiseOnA5(unittest.TestCase):
    """Bypassing the public API on A5 must surface a ParametersInvalid exception."""

    @patch("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2.is_a5_device", return_value=True)
    def test_rain_fusion_attention_raises_on_a5(self, _mock):
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention

        q = torch.zeros((1, 1, 1, 1))
        with self.assertRaises(ParametersInvalid) as ctx:
            rain_fusion_attention(q, q, q)
        self.assertIn("rf_v3", str(ctx.exception))

    @patch("mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa.is_a5_device", return_value=True)
    def test_ada_block_sparse_attention_raises_on_a5(self, _mock):
        from mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa import ada_block_sparse_attention

        q = torch.zeros((1, 1, 1, 1))
        with self.assertRaises(ParametersInvalid) as ctx:
            ada_block_sparse_attention(q, q, q, None, None)
        self.assertIn("ada_bsa", str(ctx.exception))
        self.assertIn("v2", str(ctx.exception))

    @patch("mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa.is_a5_device", return_value=True)
    def test_get_estimate_mask_raises_on_a5(self, _mock):
        from mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa import get_estimate_mask

        q = torch.zeros((1, 1, 1, 1))
        with self.assertRaises(ParametersInvalid) as ctx:
            get_estimate_mask(q, q, q, scale=1.0)
        self.assertIn("ada_bsa", str(ctx.exception))

    @patch("mindiesd.layers.flash_attn.ascend_laser_attention.is_a5_device", return_value=True)
    def test_ascend_laser_attention_forward_raises_on_a5(self, _mock):
        from mindiesd.layers.flash_attn.ascend_laser_attention import AscendLaserAttention
        from mindiesd.layers.flash_attn.common import AttentionParam

        attn_param = AttentionParam(1, 1, 64, 8192, 8192, torch.float16, True)
        q = torch.zeros((1, 1, 8192, 64), dtype=torch.float16)
        with self.assertRaises(ParametersInvalid) as ctx:
            AscendLaserAttention.forward_attn_bnsd(attn_param, q, q, q, scale=1.0)
        self.assertIn("ascend_laser_attention", str(ctx.exception))
        self.assertIn("attention_forward", str(ctx.exception))

    @patch("mindiesd.layers.flash_attn.prompt_flash_attn.is_a5_device", return_value=True)
    def test_prompt_flash_attn_forward_raises_on_a5(self, _mock):
        from mindiesd.layers.flash_attn.prompt_flash_attn import PromptFlashAttention
        from mindiesd.layers.flash_attn.common import AttentionParam

        attn_param = AttentionParam(1, 1, 64, 8, 8, torch.float16, True)
        q = torch.zeros((1, 1, 8, 64), dtype=torch.float16)

        for entry in ("forward_attn_bnsd", "forward_attn_bsnd", "forward_attn_bsh"):
            with self.subTest(entry=entry), self.assertRaises(ParametersInvalid) as ctx:
                getattr(PromptFlashAttention, entry)(attn_param, q, q, q, scale=1.0)
            self.assertIn("prompt_flash_attn", str(ctx.exception))
            self.assertIn("attention_forward", str(ctx.exception))

    @patch("mindiesd.layers.flash_attn.ascend_laser_preprocess.is_a5_device", return_value=True)
    def test_ascend_laser_preprocess_raises_on_a5(self, _mock):
        from mindiesd.layers.flash_attn.ascend_laser_preprocess import la_preprocess

        q = torch.zeros((1, 8, 1, 64), dtype=torch.float16)
        with self.assertRaises(ParametersInvalid) as ctx:
            la_preprocess(q, q, q)
        self.assertIn("ascend_laser_preprocess", str(ctx.exception))
        self.assertIn("attention_forward", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
