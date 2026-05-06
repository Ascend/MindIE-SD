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
from mindiesd.utils.exception import ParametersInvalid, ModelExecError


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestDiTBlockCache(unittest.TestCase):
    def test_cache_func_one_output_with_one_input(self):
        """测试1个输入,1个输出场景"""
        result = [
            1, 2, 3, 4, 5, 6, 7,         # delta 1
            9, 11, 13, 15, 17, 19, 21,   # delta 2, update cache 19-11=8
            24, 27, 35, 35, 35, 35, 38,  # delta 3, reuse cache
            42, 46, 50, 54, 58, 62, 66,  # delta 4, update cache 19-11=16
            71, 76, 92, 92, 92, 92, 97]  # delta 5, reuse cache

        steps_count = 5
        blocks_count = 7

        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, delta):
            return i + delta

        for _ in range(5):  # 多次运行测试
            cache_result = []
            res = 0
            for step in range(steps_count):
                for _ in range(blocks_count):
                    res = agent.apply(test_cache_func, hidden_states=res, delta=(step + 1))
                    cache_result.append(res)
            self.assertEqual(cache_result, result)

    def test_cache_func_one_output_with_two_input(self):
        """测试2个输入,但只有1个输出场景,cache只缓存一个"""
        result = [
            1, 2, 3, 4, 5, 6, 7,         # delta 1
            9, 11, 13, 15, 17, 19, 21,   # delta 2, update cache 19-11=8
            24, 27, 35, 35, 35, 35, 38,  # delta 3, reuse cache
            42, 46, 50, 54, 58, 62, 66,  # delta 4, update cache 19-11=16
            71, 76, 92, 92, 92, 92, 97]  # delta 5, reuse cache

        steps_count = 5
        blocks_count = 7

        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, j, delta):
            return i + delta

        for _ in range(5):  # 多次运行测试
            cache_result = []
            hidden_states = 0
            encoder_hidden_states = 0
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states = agent.apply(
                        test_cache_func,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        delta=(step + 1))
                    cache_result.append((hidden_states))
            self.assertEqual(cache_result, result)

    def test_cache_func_invalid_two_output_with_one_input(self):
        """测试1个输入,2个输出场景"""
        result = [
            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
            (9, 0), (11, 0), (13, 0), (15, 0), (17, 0), (19, 0), (21, 0),
            (24, 0), (27, 0), (35, 0), (35, 0), (35, 0), (35, 0), (38, 0),
            (42, 0), (46, 0), (50, 0), (54, 0), (58, 0), (62, 0), (66, 0),
            (71, 0), (76, 0), (92, 0), (92, 0), (92, 0), (92, 0), (97, 0)]
        steps_count = 5
        blocks_count = 7
        cache_result = []
        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, delta):
            return i + delta, delta

        hidden_states = 0
        encoder_hidden_states = 0
        with self.assertRaises(ParametersInvalid) as context:
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states, encoder_hidden_states = agent.apply(
                        test_cache_func,
                        hidden_states=hidden_states,
                        delta=(step + 1))
                    cache_result.append((hidden_states, encoder_hidden_states))
        self.assertIn("DiTBlockCache] 'encoder_hidden_states' is required", str(context.exception))

    def test_cache_func_two_output_with_two_input(self):
        """测试两个输入,两个输出场景"""
        result = [
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
            (9, 9), (11, 11), (13, 13), (15, 15), (17, 17), (19, 19), (21, 21),
            (24, 24), (27, 27), (35, 35), (35, 35), (35, 35), (35, 35), (38, 38),
            (42, 42), (46, 46), (50, 50), (54, 54), (58, 58), (62, 62), (66, 66),
            (71, 71), (76, 76), (92, 92), (92, 92), (92, 92), (92, 92), (97, 97)]
        steps_count = 5
        blocks_count = 7

        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, j, delta):
            return i + delta, j + delta

        for _ in range(5):  # 多次运行测试
            cache_result = []
            hidden_states = 0
            encoder_hidden_states = 0
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states, encoder_hidden_states = agent.apply(
                        test_cache_func,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        delta=(step + 1))
                    cache_result.append((hidden_states, encoder_hidden_states))
            self.assertEqual(cache_result, result)

    def test_cache_func_invliad_function_output(self):
        """测试cache的函数有2个以上输出"""
        steps_count = 5
        blocks_count = 7
        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, j, delta):
            return i + delta, j + delta, i + j + delta

        hidden_states = 0
        encoder_hidden_states = 0
        with self.assertRaises(ModelExecError) as context:
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states, encoder_hidden_states, _ = agent.apply(
                        test_cache_func,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        delta=(step + 1))
        self.assertIn("[DiTBlockCache] The output count of cache function must be 1 or 2", str(context.exception))

    def test_cache_func_invliad_hidden_states_empty(self):
        """测试cache的输入没有hidden_states"""
        steps_count = 5
        blocks_count = 7
        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, delta):
            return i + delta

        hidden_states = None
        with self.assertRaises(ParametersInvalid) as context:
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states = agent.apply(
                        test_cache_func,
                        hidden_states,
                        delta=(step + 1))
        self.assertIn("[DiTBlockCache]: Cannot find 'hidden_states' in kwargs.", str(context.exception))

    def test_cache_func_invliad_hidden_states_none(self):
        """测试cache的输入hidden_states是None"""
        steps_count = 5
        blocks_count = 7
        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, j, delta):
            return i + delta, j + delta, i + j + delta

        hidden_states = None
        encoder_hidden_states = 0
        with self.assertRaises(ParametersInvalid) as context:
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states, encoder_hidden_states = agent.apply(
                        test_cache_func,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        delta=(step + 1))
        self.assertIn("[DiTBlockCache]: Input 'hidden_states' is None.", str(context.exception))

    def test_cache_func_invliad_output_none(self):
        """测试cache的function输出是None"""
        steps_count = 5
        blocks_count = 7
        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, delta):
            if i > 16:
                return None
            else:
                return i + delta

        hidden_states = 0
        with self.assertRaises(ModelExecError) as context:
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states = agent.apply(
                        test_cache_func,
                        hidden_states=hidden_states,
                        delta=(step + 1))
        self.assertIn("The output of cache function is None.", str(context.exception))

    def test_cache_func_invliad_two_output_none(self):
        """测试cache的function输出中的一个是None"""
        steps_count = 5
        blocks_count = 7
        config = CacheConfig(
            method="dit_block_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=1,
            step_interval=2,
            block_start=2,
            block_end=6)
        agent = CacheAgent(config)

        def test_cache_func(i, j, delta):
            return i + delta, None

        hidden_states = 0
        encoder_hidden_states = 0
        with self.assertRaises(ModelExecError) as context:
            for step in range(steps_count):
                for _ in range(blocks_count):
                    hidden_states, encoder_hidden_states = agent.apply(
                        test_cache_func,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        delta=(step + 1))
        self.assertIn("The output of cache function is None.", str(context.exception))


if __name__ == '__main__':
    unittest.main()