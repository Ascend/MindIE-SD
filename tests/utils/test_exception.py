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

sys.path.append('../')
from mindiesd.utils import (ParametersInvalid, ConfigError,
    TorchError, ModelInitError, ModelExecError)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestException(unittest.TestCase):
    def test_parameter_exception(self):
        value_string = "Test parameter exception!"

        with self.assertRaises(Exception, msg="not raise") as context:
            raise ParametersInvalid(value_string)
        self.assertEqual(str(context.exception), "[MIE06E000001] Parameters invalid. " + value_string)

    def test_config_exception(self):
        value_string = "Test config exception!"

        with self.assertRaises(Exception, msg="not raise") as context:
            raise ConfigError(value_string)
        self.assertEqual(str(context.exception), "[MIE06E000002] Config parameter err. " + value_string)

    def test_torch_exception(self):
        value_string = "Test torch exception!"

        with self.assertRaises(Exception, msg="not raise") as context:
            raise TorchError(value_string)
        self.assertEqual(str(context.exception), "[MIE06E000003] Torch exec err. " + value_string)

    def test_model_init_exception(self):
        value_string = "Test model init exception!"

        with self.assertRaises(Exception, msg="not raise") as context:
            raise ModelInitError(value_string)
        self.assertEqual(str(context.exception), "[MIE06E000004] Model init err. " + value_string)

    def test_model_exec_exception(self):
        value_string = "Test model exec exception!"

        with self.assertRaises(Exception, msg="not raise") as context:
            raise ModelExecError(value_string)
        self.assertEqual(str(context.exception), "[MIE06E000005] Model exec err. " + value_string)


if __name__ == '__main__':
    unittest.main()