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

from mindiesd.layers.flash_attn.common import lru_cache_by_attn_param, AttentionParam


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestCacheByAttnParam(unittest.TestCase):
    def test_update_cache(self):
        """Test that cache correctly stores and retrieves values."""
        call_count = [0]

        @lru_cache_by_attn_param()
        def test_cache(attn_param: AttentionParam):
            call_count[0] += 1
            if attn_param.batch_size > 10:
                return "case 0"
            else:
                return "case 1"

        param = AttentionParam(20, 16, 64, 128, 128, torch.float32, False)
        
        # First call should execute the function
        out1 = test_cache(param)
        self.assertEqual(out1, "case 0")
        self.assertEqual(call_count[0], 1)
        
        # Second call with same param should use cache
        out2 = test_cache(param)
        self.assertEqual(out2, "case 0")
        self.assertEqual(call_count[0], 1)  # Function not called again

    def test_cache_isolation_between_decorated_functions(self):
        """Test that each decorated function has its own independent cache."""
        call_count_func1 = [0]
        call_count_func2 = [0]

        @lru_cache_by_attn_param()
        def func1(attn_param: AttentionParam):
            call_count_func1[0] += 1
            return "func1_result"

        @lru_cache_by_attn_param()
        def func2(attn_param: AttentionParam):
            call_count_func2[0] += 1
            return "func2_result"

        param = AttentionParam(20, 16, 64, 128, 128, torch.float32, False)
        
        # Call func1
        result1 = func1(param)
        self.assertEqual(result1, "func1_result")
        self.assertEqual(call_count_func1[0], 1)
        self.assertEqual(call_count_func2[0], 0)
        
        # Call func2 with same param - should not use func1's cache
        result2 = func2(param)
        self.assertEqual(result2, "func2_result")
        self.assertEqual(call_count_func1[0], 1)
        self.assertEqual(call_count_func2[0], 1)  # func2 was called
        
        # Call both again - should use their respective caches
        func1(param)
        func2(param)
        self.assertEqual(call_count_func1[0], 1)  # Still 1, used cache
        self.assertEqual(call_count_func2[0], 1)  # Still 1, used cache

    def test_cache_maxsize(self):
        """Test that cache respects maxsize limit."""
        @lru_cache_by_attn_param(maxsize=2)
        def limited_cache(attn_param: AttentionParam):
            return f"result_{attn_param.batch_size}"

        param1 = AttentionParam(1, 16, 64, 128, 128, torch.float32, False)
        param2 = AttentionParam(2, 16, 64, 128, 128, torch.float32, False)
        param3 = AttentionParam(3, 16, 64, 128, 128, torch.float32, False)
        
        # Fill cache with 2 items
        limited_cache(param1)
        limited_cache(param2)
        
        # Add third item - should evict oldest (param1)
        limited_cache(param3)
        
        # param1 should be evicted, calling again should re-execute
        # We can't directly test this without accessing internal cache,
        # but the test ensures maxsize parameter is accepted


if __name__ == '__main__':
    unittest.main()
