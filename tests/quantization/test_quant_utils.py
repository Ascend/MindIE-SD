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
import contextlib
from contextvars import ContextVar
import os
from unittest import mock
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from mindiesd.utils import ParametersInvalid, ConfigError
from mindiesd.quantization.utils import extract_constructor_args, replace_rank_suffix, get_quant_weight, TimestepManager


class MockSafeTensorHandler:
    def __init__(self, data):
        self.data = data

    def get_tensor(self, key):
        return self.data.get(key, None)

    def keys(self):
        return self.data.keys()


def create_mock_handler(mock_data):
    return MockSafeTensorHandler(mock_data)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestGetInitParams(unittest.TestCase):

    def test_extract_constructor_args(self):
        # Create a sample class with an __init__ method
        class SampleClass:
            def __init__(self, a, b, c=1):
                self.a = a
                self.b = b
                self.c = c

        # Create an instance of the sample class
        obj = SampleClass(1, 2, 3)

        # Test that extract_constructor_args returns the correct parameters
        expected_params = {'a': 1, 'b': 2, 'c': 3}
        self.assertEqual(extract_constructor_args(obj), expected_params)

        # Test that extract_constructor_args with a class argument returns the correct parameters
        expected_params_with_class = {'a': 1, 'b': 2, 'c': 3}
        self.assertEqual(extract_constructor_args(obj, SampleClass), expected_params_with_class)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestReplaceRankSuffix(unittest.TestCase):

    def test_no_rank_suffix(self):
        # Test file without rank suffix
        file_path = "/path/to/config.json"
        file_name = "config.json"
        new_path, new_file_name, rank = replace_rank_suffix(file_path)

        # Should not change the path and return -1 for rank
        self.assertEqual(new_path, file_path)
        self.assertEqual(new_file_name, file_name)
        self.assertEqual(rank, -1)

    def test_with_underscore_not_rank(self):
        # Test file with underscore but not a rank suffix
        file_path = "/path/to/config_name.json"
        file_name = "config_name.json"
        new_path, new_file_name, rank = replace_rank_suffix(file_path)

        # Should not change the path and return -1 for rank
        self.assertEqual(new_path, file_path)
        self.assertEqual(new_file_name, file_name)
        self.assertEqual(rank, -1)

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.get_rank')
    def test_with_rank_suffix_dist_initialized(self, mock_get_rank, mock_is_initialized):
        # Mock distributed environment
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 3

        # Test file with rank suffix
        file_path = "/path/to/config_0.json"
        expected_path = "/path/to/config_3.json"
        file_name = "config_3.json"

        new_path, new_file_name, rank = replace_rank_suffix(file_path)

        # Should change the rank suffix and return the current rank
        self.assertEqual(new_path, expected_path)
        self.assertEqual(new_file_name, file_name)
        self.assertEqual(rank, 3)

        # Verify mocks were called
        mock_is_initialized.assert_called_once()
        mock_get_rank.assert_called_once()

    @patch('torch.distributed.is_initialized')
    def test_with_rank_suffix_dist_not_initialized(self, mock_is_initialized):
        # Mock distributed environment not initialized
        mock_is_initialized.return_value = False

        # Test file with rank suffix
        file_path = "/path/to/config_0.json"

        # Should raise ConfigError
        with self.assertRaises(ConfigError):
            replace_rank_suffix(file_path)

        # Verify mock was called
        mock_is_initialized.assert_called_once()

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.get_rank')
    def test_complex_path_with_rank_suffix(self, mock_get_rank, mock_is_initialized):
        # Mock distributed environment
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 2

        # Test complex file path with rank suffix
        file_path = "/complex/path/with_underscores/config_file_1.json"
        expected_path = "/complex/path/with_underscores/config_file_2.json"
        file_name = "config_file_2.json"

        new_path, new_file_name, rank = replace_rank_suffix(file_path)

        # Should change the rank suffix and return the current rank
        self.assertEqual(new_path, expected_path)
        self.assertEqual(new_file_name, file_name)
        self.assertEqual(rank, 2)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestGetQuantWeight(unittest.TestCase):
    """测试get_quant_weight函数所有分支"""

    def test_normal_case(self):
        """测试正常获取量化权重"""
        # 准备模拟数据
        mock_tensor = MagicMock()
        weights = {'valid_key': mock_tensor}

        # 执行测试
        result = get_quant_weight(create_mock_handler(weights), 'valid_key')

        # 验证调用和返回
        self.assertEqual(result, mock_tensor)

    def test_key_not_exist(self):
        """测试键不存在的情况"""
        invalid_weights = {'other_key': 'value'}

        with self.assertRaises(ParametersInvalid) as cm:
            get_quant_weight(create_mock_handler(invalid_weights), 'missing_key')

        # 验证错误信息
        self.assertIn("Critical parameter missing: missing_key.", str(cm.exception))


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestTimestepManager(unittest.TestCase):

    def test_set_and_get(self):
        """测试正常设置和获取时间步索引"""
        TimestepManager.set_timestep_idx_max(3)
        TimestepManager.set_timestep_idx(3)
        self.assertEqual(TimestepManager.get_timestep_idx(), 3)

    def test_multiple_sets(self):
        """测试多次设置不同的值，确保最后的值正确"""
        TimestepManager.set_timestep_idx_max(20)
        TimestepManager.set_timestep_idx(10)
        TimestepManager.set_timestep_idx(20)
        self.assertEqual(TimestepManager.get_timestep_idx(), 20)
        self.assertEqual(TimestepManager.get_timestep_idx_max(), 20)

    def test_multiple_sets(self):
        """测试多次设置不同的值，确保最后的值正确"""
        TimestepManager.set_timestep_idx_max(5)
        with self.assertRaises(ParametersInvalid) as cm:
            TimestepManager.set_timestep_idx(10)

if __name__ == '__main__':
    unittest.main()
