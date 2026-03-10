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


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestAttentionFunction(unittest.TestCase):
    def test_attention_device_valid(self):
        """设备获取为910时,返回非空结果"""
        sys.modules['torch_npu'].npu.get_device_name.return_value = 'Ascend910'

        attn_param = AttentionParam(2, 16, 64, 128, 128, torch.float16, False)
        query = torch.randn([2, 32, 16, 64], dtype=torch.float16)
        key = torch.randn([2, 32, 16, 64], dtype=torch.float16)
        value = torch.randn([2, 32, 16, 64], dtype=torch.float16)

        func = get_attention_function_runtime(attn_param, query, key, value)
        self.assertIsNotNone(func)


if __name__ == '__main__':
    unittest.main()

