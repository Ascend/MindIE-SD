#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
import unittest
import logging
sys.path.append('../')

from mindiesd.utils.env import EnvVar, parser_env_to_dict, VALID_LOG_LEVELS

MINDIE_LOG_LEVEL = "MINDIE_LOG_LEVEL"


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestEnvs(unittest.TestCase):
    def test_parser_env_to_dict_case1(self):
        """测试正常配置" sd:info; debug"，解析sd的日志等级功能"""
        mindie_log_level = " sd:info; debug"
        log_level = parser_env_to_dict(mindie_log_level, VALID_LOG_LEVELS)
        self.assertEqual(log_level.get("sd"), "info")

    def test_parser_env_to_dict_case2(self):
        """测试正常配置"sd : info   "，解析sd的日志等级功能"""
        mindie_log_level = "sd : info   "
        log_level = parser_env_to_dict(mindie_log_level, VALID_LOG_LEVELS)
        self.assertEqual(log_level.get("sd"), "info")

    def test_parser_env_to_dict_case3(self):
        """测试重复配置"sd : info;sd:debug "，解析sd的日志等级功能"""
        mindie_log_level = "sd : info;sd:debug "
        log_level = parser_env_to_dict(mindie_log_level, VALID_LOG_LEVELS)
        self.assertEqual(log_level.get("sd"), "debug")

    def test_parser_env_to_dict_invalid_case1(self):
        """测试异常配置等级为不支持字符，字符串解析过滤功能"""
        mindie_log_level = "sd : dinfo   "
        log_level = parser_env_to_dict(mindie_log_level, VALID_LOG_LEVELS)
        self.assertIsNone(log_level.get("sd", None))

    def test_parser_env_to_dict_invalid_case2(self):
        """测试异常配置等级为不支持字符，字符串解析过滤功能"""
        mindie_log_level = "dinfo   "
        log_level = parser_env_to_dict(mindie_log_level, VALID_LOG_LEVELS)
        self.assertIsNone(log_level.get("*", None))

    def test_valid_env_log_level(self):
        """测试正常log_level配置"""

        test_envs = ["critical", "debug", "error", "info", "warn"]
        for log_level in test_envs:
            os.environ[MINDIE_LOG_LEVEL] = log_level
            env = EnvVar(os.getenv(MINDIE_LOG_LEVEL, ""))
            self.assertEqual(env.component_log_level, log_level)

            os.environ[MINDIE_LOG_LEVEL] = "sd:" + log_level
            env = EnvVar(os.getenv(MINDIE_LOG_LEVEL, ""))
            self.assertEqual(env.component_log_level, log_level)

            os.environ[MINDIE_LOG_LEVEL] = "other:" + log_level
            env = EnvVar(os.getenv(MINDIE_LOG_LEVEL, ""))
            self.assertEqual(env.component_log_level, "info")

    def test_valid_env_disable_log_level(self):
        """测试日志级别设置为null关掉所有日志"""
        disable_log_level = "null"
        os.environ[MINDIE_LOG_LEVEL] = disable_log_level
        env = EnvVar(os.getenv(MINDIE_LOG_LEVEL, ""))
        self.assertTrue(env.disable_log)

        os.environ[MINDIE_LOG_LEVEL] = "sd:  " + disable_log_level
        env = EnvVar(os.getenv(MINDIE_LOG_LEVEL, ""))
        self.assertTrue(env.disable_log)

        os.environ[MINDIE_LOG_LEVEL] = "other:  " + disable_log_level
        env = EnvVar(os.getenv(MINDIE_LOG_LEVEL, ""))
        self.assertFalse(env.disable_log)

    def test_invalid_env_log_level(self):
        long_string = "invalid env \n" * 50
        os.environ[MINDIE_LOG_LEVEL] = long_string
        with self.assertRaises(ValueError):
            env = EnvVar(os.getenv(MINDIE_LOG_LEVEL, ""))


if __name__ == '__main__':
    unittest.main()