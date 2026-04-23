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
from io import StringIO
import unittest
import logging
from unittest.mock import patch

sys.path.append('../')
from mindiesd.utils.logs.logging import logger, MAX_LOG_STRING_LEN


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestLog(unittest.TestCase):
    def setUp(self):
        """修改logger中的stream为StringIO,进行日志捕获"""
        self.string_io = StringIO()
        for handler in logger.handlers:
            if handler.__class__ is logging.StreamHandler:
                handler.stream = self.string_io

    def tearDown(self):
        """回退StringIO为正常日志功能"""
        for handler in logger.handlers:
            if handler.__class__ is logging.StreamHandler:
                handler.stream = sys.stdout

    def test_log_inject(self):
        inject_chars = [
            '\f', '\r', '\b', '\t', '\v', '\n',
            '\u000A', '\u000D', '\u000C', '\u000B',
            '\u0008', '\u007F', '\u0009'
        ]
        for inject in inject_chars:
            logger.info("test %s inject", inject)
            self.assertNotIn(inject, self.string_io.getvalue().rstrip('\n'))
            # 清空StringIO
            self.string_io.truncate(0)
            self.string_io.seek(0)

    def test_log_repetitive_space(self):
        error_log = "test" + "  " * 10 + "logs!"
        logger.info(error_log)
        self.assertNotIn("  ", self.string_io.getvalue().rstrip('\n'))
        # 清空StringIO
        self.string_io.truncate(0)
        self.string_io.seek(0)

    def test_log_long_str(self):
        error_log = "test_long_str " * 1024
        logger.info(error_log)
        self.assertLessEqual(len(self.string_io.getvalue().rstrip('\n')), MAX_LOG_STRING_LEN)
        # 清空StringIO
        self.string_io.truncate(0)
        self.string_io.seek(0)

    def test_log_func(self):
        try:
            logger.critical("Test critical!")
            logger.debug("Test debug!")
            logger.error("Test error!")
            logger.warning("Test warning!")
            logger.info("Test info!")
        except Exception as e:
            self.fail(f"An exception was raised: {e}")


if __name__ == '__main__':
    unittest.main()