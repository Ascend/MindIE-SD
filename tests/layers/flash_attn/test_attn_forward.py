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
from unittest.mock import patch
import os

import torch
from device import DEVICE_ID
from mindiesd.layers.flash_attn.common import AttentionParam
from mindiesd.layers.flash_attn.attention_forward import attention_forward, get_manual_attention_op_type
from mindiesd.utils.exception import ParametersInvalid
from mindiesd.utils.get_platform import is_a5_device
from utils.utils.precision_compare import data_compare


_SKIP_A5_LASER = "ascend_laser_attention is unsupported on A5; routed to fused_attn_score."


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
class TestAttentionFunc(unittest.TestCase):
    def test_attn_forward_no_fused_bsnd(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out = attention_forward(query, key, value, fused=False, head_first=False)
        self.assertIsNotNone(out)

    def test_attn_forward_runtime_bsnd(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(query, key, value, opt_mode="runtime", head_first=False)
        self.assertIsNotNone(out_fused)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=False)
        self.assertEqual(out_non_fused.shape, out_fused.shape)
        result, _, max_error = data_compare(out_fused.cpu(), out_non_fused.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_error}")

    def test_attn_forward_static_bsnd(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(query, key, value, head_first=False, opt_mode="static")
        self.assertIsNotNone(out_fused)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=False)
        self.assertEqual(out_non_fused.shape, out_fused.shape)
        result, _, max_error = data_compare(out_fused.cpu(), out_non_fused.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_error}")

    def test_attn_forward_manual_bsnd_pfa(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused_bnsd = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="prompt_flash_attn", layout="BNSD"
        )
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="prompt_flash_attn", layout="BSND"
        )
        out_fused_bsh = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="prompt_flash_attn", layout="BSH"
        )
        self.assertIsNotNone(out_fused_bnsd)
        self.assertIsNotNone(out_fused_bsnd)
        self.assertIsNotNone(out_fused_bsh)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=False)
        self.assertEqual(out_non_fused.shape, out_fused_bsnd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bnsd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bsh.shape)
        result_bsnd, _, max_error_bsnd = data_compare(out_fused_bsnd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsnd, "success", msg=f"Data compare failed. Max error is: {max_error_bsnd}")
        result_bnsd, _, max_error_bnsd = data_compare(out_fused_bnsd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bnsd, "success", msg=f"Data compare failed. Max error is: {max_error_bnsd}")
        result_bsh, _, max_error_bsh = data_compare(out_fused_bsh.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsh, "success", msg=f"Data compare failed. Max error is: {max_error_bsh}")

    def test_attn_forward_manual_bsnd_fas(self):
        attention_shape = [2, 32, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused_bnsd = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="fused_attn_score", layout="BNSD"
        )
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="fused_attn_score", layout="BSND"
        )
        out_fused_bsh = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="fused_attn_score", layout="BSH"
        )
        self.assertIsNotNone(out_fused_bnsd)
        self.assertIsNotNone(out_fused_bsnd)
        self.assertIsNotNone(out_fused_bsh)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=False)
        self.assertEqual(out_non_fused.shape, out_fused_bsnd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bnsd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bsh.shape)
        result_bsnd, _, max_error_bsnd = data_compare(out_fused_bsnd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsnd, "success", msg=f"Data compare failed. Max error is: {max_error_bsnd}")
        result_bnsd, _, max_error_bnsd = data_compare(out_fused_bnsd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bnsd, "success", msg=f"Data compare failed. Max error is: {max_error_bnsd}")
        result_bsh, _, max_error_bsh = data_compare(out_fused_bsh.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsh, "success", msg=f"Data compare failed. Max error is: {max_error_bsh}")

    @unittest.skipIf(is_a5_device(), _SKIP_A5_LASER)
    def test_attn_forward_manual_la_bsnd(self):
        attention_shape = [2, 5120, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD"
        )
        self.assertIsNotNone(out_fused)

    def test_attn_forward_no_fused_bnsd(self):
        attention_shape = [2, 16, 32, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out = attention_forward(query, key, value, fused=False, head_first=True)
        self.assertIsNotNone(out)

    def test_attn_forward_runtime_bnsd(self):
        attention_shape = [2, 16, 32, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(query, key, value, head_first=True, opt_mode="runtime")
        self.assertIsNotNone(out_fused)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=True)
        self.assertEqual(out_non_fused.shape, out_fused.shape)
        result, _, max_error = data_compare(out_fused.cpu(), out_non_fused.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_error}")

    def test_attn_forward_static_bnsd(self):
        attention_shape = [2, 16, 32, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(query, key, value, head_first=True, opt_mode="static")
        self.assertIsNotNone(out_fused)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=True)
        self.assertEqual(out_non_fused.shape, out_fused.shape)
        result, _, max_error = data_compare(out_fused.cpu(), out_non_fused.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_error}")

    def test_attn_forward_manual_bnsd_pfa(self):
        attention_shape = [2, 16, 32, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused_bnsd = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="prompt_flash_attn", layout="BNSD"
        )
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="prompt_flash_attn", layout="BSND"
        )
        out_fused_bsh = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="prompt_flash_attn", layout="BSH"
        )
        self.assertIsNotNone(out_fused_bnsd)
        self.assertIsNotNone(out_fused_bsnd)
        self.assertIsNotNone(out_fused_bsh)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=True)
        self.assertEqual(out_non_fused.shape, out_fused_bsnd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bnsd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bsh.shape)
        result_bsnd, _, max_error_bsnd = data_compare(out_fused_bsnd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsnd, "success", msg=f"Data compare failed. Max error is: {max_error_bsnd}")
        result_bnsd, _, max_error_bnsd = data_compare(out_fused_bnsd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bnsd, "success", msg=f"Data compare failed. Max error is: {max_error_bnsd}")
        result_bsh, _, max_error_bsh = data_compare(out_fused_bsh.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsh, "success", msg=f"Data compare failed. Max error is: {max_error_bsh}")

    def test_attn_forward_manual_bnsd_fas(self):
        attention_shape = [2, 16, 32, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused_bnsd = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="fused_attn_score", layout="BNSD"
        )
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="fused_attn_score", layout="BSND"
        )
        out_fused_bsh = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="fused_attn_score", layout="BSH"
        )
        self.assertIsNotNone(out_fused_bnsd)
        self.assertIsNotNone(out_fused_bsnd)
        self.assertIsNotNone(out_fused_bsh)

        out_non_fused = attention_forward(query, key, value, fused=False, head_first=True)
        self.assertEqual(out_non_fused.shape, out_fused_bsnd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bnsd.shape)
        self.assertEqual(out_non_fused.shape, out_fused_bsh.shape)
        result_bsnd, _, max_error_bsnd = data_compare(out_fused_bsnd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsnd, "success", msg=f"Data compare failed. Max error is: {max_error_bsnd}")
        result_bnsd, _, max_error_bnsd = data_compare(out_fused_bnsd.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bnsd, "success", msg=f"Data compare failed. Max error is: {max_error_bnsd}")
        result_bsh, _, max_error_bsh = data_compare(out_fused_bsh.cpu(), out_non_fused.cpu())
        self.assertEqual(result_bsh, "success", msg=f"Data compare failed. Max error is: {max_error_bsh}")

    @unittest.skipIf(is_a5_device(), _SKIP_A5_LASER)
    def test_attn_forward_manual_la_bnsd(self):
        attention_shape = [2, 16, 5120, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD"
        )
        self.assertIsNotNone(out_fused)

    def test_attn_forward_manual_env(self):
        attention_shape = [2, 16, 5120, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused_pfa = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="prompt_flash_attn", layout="BNSD"
        )
        os.environ["MINDIE_SD_FA_TYPE"] = "prompt_flash_attn"
        out_fused_env = attention_forward(query, key, value, head_first=True, opt_mode="manual", layout="BNSD")
        result, _, max_error = data_compare(out_fused_env.cpu(), out_fused_pfa.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_error}")

        os.environ["MINDIE_SD_FA_TYPE"] = "test"
        with self.assertRaises(ParametersInvalid):
            out_fused_env = attention_forward(query, key, value, head_first=True, opt_mode="manual", layout="BNSD")
        os.environ.pop("MINDIE_SD_FA_TYPE", None)


class TestAttentionForwardFallback(unittest.TestCase):
    @patch("mindiesd.layers.flash_attn.attention_forward.is_a5_device", return_value=False)
    @patch("mindiesd.layers.flash_attn.attention_forward.logger.debug")
    def test_manual_la_fallback_when_q_seqlen_lt_2048(self, mock_logger_debug, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 1024, 4096, torch.float16, False)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_debug.assert_called_once()

    @patch("mindiesd.layers.flash_attn.attention_forward.is_a5_device", return_value=False)
    @patch("mindiesd.layers.flash_attn.attention_forward.logger.debug")
    def test_manual_la_fallback_when_kv_seqlen_lt_2048(self, mock_logger_debug, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 4096, 1024, torch.float16, False)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_debug.assert_called_once()

    @patch("mindiesd.layers.flash_attn.attention_forward.is_a5_device", return_value=False)
    @patch("mindiesd.layers.flash_attn.attention_forward.logger.debug")
    def test_manual_la_fallback_when_head_first_true(self, mock_logger_debug, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 1024, 1024, torch.float16, True)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_debug.assert_called_once()

    @patch("mindiesd.layers.flash_attn.attention_forward.is_a5_device", return_value=True)
    @patch("mindiesd.layers.flash_attn.attention_forward.logger.warning")
    def test_manual_la_fallback_to_fas_on_a5(self, mock_logger_warning, _mock_is_a5):
        # On A5 LA is no longer the optimal operator; manual choice must be routed
        # to fused_attn_score regardless of the seqlen threshold used on A2/A3.
        attn_param = AttentionParam(2, 16, 64, 8192, 8192, torch.float16, True)

        op_type = get_manual_attention_op_type(attn_param, "ascend_laser_attention")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_warning.assert_called_once()

    @patch("mindiesd.layers.flash_attn.attention_forward.is_a5_device", return_value=True)
    @patch("mindiesd.layers.flash_attn.attention_forward.logger.warning")
    def test_manual_pfa_fallback_to_fas_on_a5(self, mock_logger_warning, _mock_is_a5):
        # On A5 PFA is no longer registered; manual choice must be routed to
        # fused_attn_score with a warning telling the user to migrate.
        attn_param = AttentionParam(2, 16, 64, 8192, 8192, torch.float16, True)

        op_type = get_manual_attention_op_type(attn_param, "prompt_flash_attn")

        self.assertEqual(op_type, "fused_attn_score")
        mock_logger_warning.assert_called_once()

    @patch("mindiesd.layers.flash_attn.attention_forward.is_a5_device", return_value=True)
    def test_manual_fas_unchanged_on_a5(self, _mock_is_a5):
        attn_param = AttentionParam(2, 16, 64, 8192, 8192, torch.float16, True)
        self.assertEqual(
            get_manual_attention_op_type(attn_param, "fused_attn_score"),
            "fused_attn_score",
        )

    @patch("mindiesd.layers.flash_attn.attention_forward.is_a5_device", return_value=False)
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
