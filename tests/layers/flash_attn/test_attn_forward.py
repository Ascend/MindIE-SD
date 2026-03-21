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

import unittest
from unittest.mock import patch

import sys
import time

import torch

sys.path.append('../')

from device import DEVICE_ID
from mindiesd.layers.flash_attn.common import AttentionParam
from mindiesd.layers.flash_attn.attention_forward import attention_forward
from tests.utils.utils.precision_compare import data_compare


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
            query, key, value, head_first=False, opt_mode="manual", op_type="prompt_flash_attn", layout="BNSD")
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="prompt_flash_attn", layout="BSND")
        out_fused_bsh = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="prompt_flash_attn", layout="BSH")
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
            query, key, value, head_first=False, opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="fused_attn_score", layout="BSND")
        out_fused_bsh = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="fused_attn_score", layout="BSH")
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

    def test_attn_forward_manual_la_bsnd(self):
        attention_shape = [2, 5120, 16, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(
            query, key, value, head_first=False, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
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
            query, key, value, head_first=True, opt_mode="manual", op_type="prompt_flash_attn", layout="BNSD")
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="prompt_flash_attn", layout="BSND")
        out_fused_bsh = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="prompt_flash_attn", layout="BSH")
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
            query, key, value, head_first=True, opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
        out_fused_bsnd = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="fused_attn_score", layout="BSND")
        out_fused_bsh = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="fused_attn_score", layout="BSH")
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

    def test_attn_forward_manual_la_bnsd(self):
        attention_shape = [2, 16, 5120, 64]
        device = "npu"
        query = torch.randn(attention_shape, dtype=torch.float16).to(device)
        key = torch.randn(attention_shape, dtype=torch.float16).to(device)
        value = torch.randn(attention_shape, dtype=torch.float16).to(device)
        out_fused = attention_forward(
            query, key, value, head_first=True, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
        self.assertIsNotNone(out_fused)

if __name__ == '__main__':
    import torch_npu

    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()