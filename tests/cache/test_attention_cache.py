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
import sys

sys.path.append('../')

from mindiesd.cache_agent import CacheAgent, CacheConfig


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestAttentionCache(unittest.TestCase):
    def test_cache_func(self):
        result = [0, 1, 2, 3, 4,
                  5, 6, 7, 8, 9,
                  5, 6, 7, 8, 9,
                  5, 6, 7, 8, 9,
                  20, 21, 22, 23, 24,
                  20, 21, 22, 23, 24,
                  30, 31, 32, 33, 34]
        steps_count = 7
        blocks_count = 5

        config = CacheConfig(
            method="attention_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_end=5,
            step_interval=3)
        agent = CacheAgent(config)

        def test_cache_func(i):
            return i

        for _ in range(5):  # 多次运行测试
            cache_result = []
            for step in range(steps_count):
                for block in range(blocks_count):
                    res = agent.apply(test_cache_func, step * blocks_count + block)
                    cache_result.append(res)
            self.assertEqual(cache_result, result)
    
    def test_cache_func_two_result(self):
        result = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]
        steps_count = 7
        blocks_count = 5
        
        config = CacheConfig(
            method="attention_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_end=5,
            step_interval=3)
        agent = CacheAgent(config)

        def test_cache_func(i, j):
            return i, j

        for _ in range(5):  # 多次运行测试
            cache_result = []
            for step in range(steps_count):
                for block in range(blocks_count):
                    res = agent.apply(test_cache_func, step, block)
                    cache_result.append(res)
            self.assertEqual(cache_result, result)


if __name__ == '__main__':
    unittest.main()