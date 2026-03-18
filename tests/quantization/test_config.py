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
import sys
import unittest
from typing import Dict, List

from mindiesd.quantization.config import QuantConfig, LayerQuantConfig, TimestepPolicyConfig
from mindiesd.quantization.mode import W8A8_LIST
from mindiesd.quantization.mode import QuantAlgorithm, QuantMode

sys.path.append('../')


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestQuantConfig(unittest.TestCase):

    def test_parse_from_dict(self):
        # Test creating QuantConfig from a dictionary
        config_dict = {'quant_algo': QuantAlgorithm.W8A8}
        config = QuantConfig.parse_from_dict(config_dict)
        self.assertEqual(config.quant_algo, QuantAlgorithm.W8A8)

    def test_layer_quantization_mode(self):
        # Test when quant_algo is valid
        config = QuantConfig(quant_algo=QuantAlgorithm.W8A8)
        self.assertIsInstance(config.layer_quantization_mode, QuantMode)

        # Test when quant_algo is None
        config = QuantConfig(quant_algo=None)
        self.assertIsInstance(config.layer_quantization_mode, QuantMode)


    def test_serialize_to_dict(self):
        # Test converting QuantConfig to a dictionary
        config = QuantConfig(quant_algo=QuantAlgorithm.W8A8)
        config_dict = config.serialize_to_dict()
        self.assertEqual(config_dict['quant_algo'], QuantAlgorithm.W8A8)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestLayerQuantConfig(unittest.TestCase):

    def test_init(self):
        # Test initializing LayerQuantConfig with valid parameters
        quantized_layers = {'layer1': QuantConfig(quant_algo=QuantAlgorithm.W8A8)}
        config = LayerQuantConfig(quant_algo=QuantAlgorithm.W8A8, quantized_layers=quantized_layers)
        self.assertEqual(config.quant_algo, QuantAlgorithm.W8A8)
        self.assertEqual(config.quantized_layers, quantized_layers)

    def test_layer_quantization_mode(self):
        # Test when quantized_layers is not empty
        quantized_layers = {'layer1': QuantConfig(quant_algo=QuantAlgorithm.W8A8)}
        config = LayerQuantConfig(quantized_layers=quantized_layers)
        self.assertIsInstance(config.layer_quantization_mode, Dict)

        # Test when quantized_layers is empty
        config = LayerQuantConfig(quantized_layers={})
        self.assertIsInstance(config.layer_quantization_mode, Dict)

    def test_quant_algorithms_list(self):
        # Test when quantized_layers is not empty
        quantized_layers = {'layer1': QuantConfig(quant_algo=QuantAlgorithm.W8A8)}
        exclude_layers = ('layer2',)
        config = LayerQuantConfig(quantized_layers=quantized_layers, exclude_layers=exclude_layers)
        self.assertIsInstance(config.quant_algorithms_list, List)

        # Test when quantized_layers is empty
        config = LayerQuantConfig(quantized_layers={})
        self.assertIsInstance(config.quant_algorithms_list, List)


    def test_serialize_to_dict(self):
        # Test converting LayerQuantConfig to a dictionary
        quantized_layers = {'layer1': QuantConfig(quant_algo=QuantAlgorithm.W8A8)}
        config = LayerQuantConfig(quantized_layers=quantized_layers)
        config_dict = config.serialize_to_dict()
        self.assertIsInstance(config_dict, Dict)

    def test_parse_from_dict(self):
        # Test creating LayerQuantConfig from a dictionary
        config_dict = {'quantized_layers': {'layer1': {'quant_algo': QuantAlgorithm.W8A8}}}
        config = LayerQuantConfig.parse_from_dict(config_dict)
        self.assertIsInstance(config, LayerQuantConfig)
        self.assertIsInstance(config.quantized_layers['layer1'], QuantConfig)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestTimeStepPolicyConfig(unittest.TestCase):
    def setUp(self):
        """在每个测试方法前创建一个新的配置实例"""
        self.config = TimestepPolicyConfig()

    def test_register_and_get_strategy(self):
        """测试注册策略并能正确获取"""
        # 注册单个时间步的策略
        self.config.register(10, "static")  # 使用"static"而不是"fixed"
        self.assertEqual(self.config.get_strategy(10), "static")
        
        # 注册时间步范围
        self.config.register([20, 30, 40], "dynamic")  # 使用"dynamic"而不是"adaptive"
        self.assertEqual(self.config.get_strategy(20), "dynamic")
        self.assertEqual(self.config.get_strategy(30), "dynamic")
        self.assertEqual(self.config.get_strategy(40), "dynamic")
        
        # 测试默认策略
        self.assertEqual(self.config.get_strategy(5), "dynamic")

    def test_register_with_int_step(self):
        """测试使用整数作为step_range注册"""
        self.config.register(15, "static")  # 使用"static"
        self.assertEqual(self.config.get_strategy(15), "static")

    def test_register_with_range_step(self):
        """测试使用range对象作为step_range注册"""
        self.config.register(range(50, 53), "dynamic")  # 使用"dynamic"
        self.assertEqual(self.config.get_strategy(50), "dynamic")
        self.assertEqual(self.config.get_strategy(51), "dynamic")
        self.assertEqual(self.config.get_strategy(52), "dynamic")

    def test_invalid_strategy_type(self):
        """测试注册非字符串策略类型"""
        with self.assertRaises(TypeError):
            self.config.register(10, 123)  # 123不是字符串

    def test_invalid_strategy_value(self):
        """测试注册无效的策略值"""
        with self.assertRaises(ValueError):
            self.config.register(10, "invalid_strategy")  # 不在VALID_STRATEGIES中
        with self.assertRaises(ValueError):
            self.config.register(10, "fixed")  # "fixed"不在VALID_STRATEGIES中
        with self.assertRaises(ValueError):
            self.config.register(10, "adaptive")  # "adaptive"不在VALID_STRATEGIES中

    def test_invalid_step_range_type(self):
        """测试注册无效的step_range类型"""
        with self.assertRaises(TypeError):
            self.config.register("invalid", "static")  # 字符串不是有效的step_range类型

    def test_invalid_step_in_range(self):
        """测试step_range中包含非整数元素"""
        with self.assertRaises(TypeError):
            self.config.register([10, "20", 30], "static")  # "20"不是整数

    def test_get_strategy_for_unregistered_step(self):
        """测试获取未注册时间步的策略，应返回默认策略"""
        self.assertEqual(self.config.get_strategy(999), "dynamic")
if __name__ == '__main__':
    unittest.main()
