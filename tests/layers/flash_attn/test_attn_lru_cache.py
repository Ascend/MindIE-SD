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
import os
import torch

from mindiesd.layers.flash_attn.common import lru_cache_by_attn_param, AttentionParam, attn_cache


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestCacheByAttnParam(unittest.TestCase):
    def test_update_cache(self):
        
        @lru_cache_by_attn_param()
        def test_cache(attn_param: AttentionParam):
            if attn_param.batch_size > 10:
                return "case 0"
            else:
                return "case 1"
        
        param = AttentionParam(20, 16, 64, 128, 128, torch.float32, False)
        out = test_cache(param)
        self.assertIn(param.to_hash(), attn_cache)
        self.assertEqual(attn_cache[param.to_hash()], "case 0")



if __name__ == '__main__':
    unittest.main()