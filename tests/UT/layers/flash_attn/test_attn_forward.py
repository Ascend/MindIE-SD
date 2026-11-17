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

# pylint: disable=no-name-in-module
import unittest
from importlib import import_module
from unittest.mock import patch
import os

import torch
from device import DEVICE_ID
from mindiesd.layers.flash_attn.common import AttentionParam
from mindiesd.layers.flash_attn.attention_forward import attention_forward, get_manual_attention_op_type
from mindiesd.utils.exception import ParametersInvalid
from mindiesd.utils.get_platform import is_a5_device
from utils.utils.precision_compare import data_compare


_ATTENTION_FORWARD_MODULE = import_module("mindiesd.layers.flash_attn.attention_forward")


_SKIP_A5_LASER = "ascend_laser_attention is unsupported on A5; routed to fused_attn_score."


class TestAttentionForwardFallback(unittest.TestCase):
    @patch.object(_ATTENTION_FORWARD_MODULE, "is_a5_device", return_value=False)
    @patch.object(_ATTENTION_FORWARD_MODULE.logger, "debug")
    def test_manual_la_fallback_when_q_seqlen_lt_2048(self, mock_logger_debug, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 1024, 4096, torch.float16, False)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_debug.assert_called_once()

    @patch.object(_ATTENTION_FORWARD_MODULE, "is_a5_device", return_value=False)
    @patch.object(_ATTENTION_FORWARD_MODULE.logger, "debug")
    def test_manual_la_fallback_when_kv_seqlen_lt_2048(self, mock_logger_debug, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 4096, 1024, torch.float16, False)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_debug.assert_called_once()

    @patch.object(_ATTENTION_FORWARD_MODULE, "is_a5_device", return_value=False)
    @patch.object(_ATTENTION_FORWARD_MODULE.logger, "debug")
    def test_manual_la_fallback_when_head_first_true(self, mock_logger_debug, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 1024, 1024, torch.float16, True)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_debug.assert_called_once()

    @patch.object(_ATTENTION_FORWARD_MODULE, "is_a5_device", return_value=True)
    @patch.object(_ATTENTION_FORWARD_MODULE.logger, "warning")
    def test_manual_la_fallback_to_fas_on_a5(self, mock_logger_warning, _mock_is_a5):
        # On A5 LA is no longer the optimal operator; manual choice must be routed
        # to fused_attn_score regardless of the seqlen threshold used on A2/A3.
        attn_param = AttentionParam(2, 16, 64, 8192, 8192, torch.float16, True)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_warning.assert_called_once()

    @patch.object(_ATTENTION_FORWARD_MODULE, "is_a5_device", return_value=True)
    @patch.object(_ATTENTION_FORWARD_MODULE.logger, "warning")
    def test_manual_pfa_fallback_to_fas_on_a5(self, mock_logger_warning, _mock_is_a5):
        # On A5 PFA is no longer registered; manual choice must be routed to
        # fused_attn_score with a warning telling the user to migrate.
        attn_param = AttentionParam(2, 16, 64, 8192, 8192, torch.float16, True)

        op_type = get_manual_attention_op_type(attn_param, "prompt_flash_attn")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_warning.assert_called_once()

    @patch.object(_ATTENTION_FORWARD_MODULE, "is_a5_device", return_value=True)
    def test_manual_fas_unchanged_on_a5(self, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 8192, 8192, torch.float16, True)
        self.assertEqual(
            get_manual_attention_op_type(attn_param, "fused_attn_score"),
            "fused_attn_score",
        )

    @patch.object(_ATTENTION_FORWARD_MODULE, "is_a5_device", return_value=False)
    def test_manual_pfa_unchanged_on_non_a5(self, _mock_is_a5):
        # Non-A5 devices must keep PFA selection untouched.
        attn_param = AttentionParam(2, 16, 64, 8192, 8192, torch.float16, True)
        self.assertEqual(
            get_manual_attention_op_type(attn_param, "prompt_flash_attn"),
            "prompt_flash_attn",
        )


if __name__ == '__main__':
    import torch_npu

    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()
