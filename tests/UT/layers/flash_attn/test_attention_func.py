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
import os
import unittest

import sys
import torch

from mindiesd.layers.flash_attn.common import AttentionParam
from mindiesd.layers.flash_attn.attention_func import get_attention_function_runtime
from importlib import import_module
from unittest.mock import patch
from mindiesd.utils.get_platform import NPUDevice


class TestAttentionFunction(unittest.TestCase):
    def test_attention_device_valid(self):
        """设备获取为910时,返回非空结果"""
        functions = import_module('mindiesd.layers.flash_attn.attention_func')
        for platform in (NPUDevice.A2, NPUDevice.A3, NPUDevice.A5, NPUDevice.Duo):
            length = list(NPUDevice).index(platform) + 11
            attn_param = AttentionParam(1, 2, 8, length, length, torch.float16, False)
            query = torch.ones(1, length, 2, 8, dtype=torch.float16)
            with (
                self.subTest(platform=platform),
                patch.object(functions, 'get_npu_device', return_value=platform),
                patch('mindiesd.layers.flash_attn.prompt_flash_attn.is_a5_device', return_value=False),
                patch('torch.npu.synchronize'),
                patch('torch_npu.npu_fusion_attention', side_effect=lambda q, *a, **kw: (q.clone(),)),
                patch('torch_npu.npu_prompt_flash_attention', side_effect=lambda q, *a, **kw: q.clone()),
            ):
                func = get_attention_function_runtime(attn_param, query, query, query)
                self.assertIsNotNone(func)
                self.assertEqual(func(query, query, query, None, 0.3).shape, query.shape)


if __name__ == '__main__':
    unittest.main()
