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
from mindiesd.utils.exception import ConfigError, ParametersInvalid


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestCacheConfig(unittest.TestCase):
    def test_valid_config(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_end=8,
            step_interval=3,
            block_start=2,
            block_end=1000)
        agent = CacheAgent(config)
        self.assertIsNotNone(agent)

    def test_invalid_method(self):
        config = CacheConfig(
            method="cache",
            blocks_count=10,
            steps_count=10,
            step_start=3,
            step_end=8,
            step_interval=3)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("not supported", str(context.exception))

    def test_invalid_steps_count(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=0,
            step_start=3,
            step_end=8,
            step_interval=3)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("The 'steps_count' in config must > 0", str(context.exception))

    def test_invalid_blocks_count(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=0,
            steps_count=10,
            step_start=3,
            step_end=8,
            step_interval=3)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("The 'blocks_count' in config must > 0", str(context.exception))

    def test_invalid_step_start(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=-1,
            step_end=8,
            step_interval=3)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("The 'step_start' in config must >= 0", str(context.exception))

    def test_invalid_step_interval(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_end=8,
            step_interval=0)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("The 'step_interval' in config must > 0", str(context.exception))

    def test_invalid_step_end(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_end=1,
            step_interval=3)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("The 'step_end' must >= 'step_start'", str(context.exception))

    def test_invalid_block_start(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_end=8,
            step_interval=3,
            block_start=-1,
            block_end=1000)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("The 'block_start' in config must >= 0", str(context.exception))

    def test_invalid_block_end(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_end=8,
            step_interval=3,
            block_start=10,
            block_end=8)
        with self.assertRaises(ConfigError) as context:
            CacheAgent(config)
        self.assertIn("The 'block_end' must >= 'block_start'", str(context.exception))

    def test_invalid_cache_function(self):
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_end=8,
            step_interval=3,
            block_start=2,
            block_end=1000)
        agent = CacheAgent(config)
        invalid_func = ""
        with self.assertRaises(ParametersInvalid) as context:
            agent.apply(invalid_func)
        self.assertIn("Input function must be callable.", str(context.exception))

    def test_cache_config_step_count_no_cache(self):
        """测试当step count <= step start的时候,直接执行传入函数"""
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=10,
            step_end=100,
            step_interval=2,
            block_start=2,
            block_end=8,
            )
        agent = CacheAgent(config)

        def func(x):
            return str(x)  # str类型的返回值没法缓存,因此如果走进了cache分支则会抛异常

        res = agent.apply(func, 20)
        self.assertEqual(res, func(20))

    def test_cache_config_step_start_no_cache(self):
        """测试当step start == step end 的时候,直接执行传入函数"""
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=5,
            step_end=5,
            step_interval=2,
            block_start=2,
            block_end=8,
            )
        agent = CacheAgent(config)

        def func(x):
            return str(x)  # str类型的返回值没法缓存,因此如果走进了cache分支则会抛异常

        res = agent.apply(func, 20)
        self.assertEqual(res, func(20))

    def test_cache_config_step_interval_no_cache(self):
        """测试当step interval = 1 的时候,直接执行传入函数"""
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_interval=1,
            step_end=5,
            block_start=2,
            block_end=8,
            )
        agent = CacheAgent(config)

        def func(x):
            return str(x)  # str类型的返回值没法缓存,因此如果走进了cache分支则会抛异常

        res = agent.apply(func, 20)
        self.assertEqual(res, func(20))

    def test_cache_config_block_count_no_cache(self):
        """测试当block count <= block start 的时候,直接执行传入函数"""
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_interval=2,
            step_end=5,
            block_start=10,
            block_end=100,
            )
        agent = CacheAgent(config)

        def func(x):
            return str(x)  # str类型的返回值没法缓存,因此如果走进了cache分支则会抛异常

        res = agent.apply(func, 20)
        self.assertEqual(res, func(20))

    def test_cache_config_block_start_no_cache(self):
        """测试当block start == block end 的时候,直接执行传入函数"""
        config = CacheConfig(
            method="attention_cache",
            blocks_count=10,
            steps_count=10,
            step_start=2,
            step_interval=2,
            step_end=5,
            block_start=5,
            block_end=5,
            )
        agent = CacheAgent(config)

        def func(x):
            return str(x)  # str类型的返回值没法缓存,因此如果走进了cache分支则会抛异常

        res = agent.apply(func, 20)
        self.assertEqual(res, func(20))


if __name__ == '__main__':
    unittest.main()