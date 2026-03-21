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

import importlib
import unittest
from unittest import mock
import json
import torch
import torch.nn as nn
from mindiesd.quantization.config import QuantConfig, LayerQuantConfig
from mindiesd.quantization.layer import W8A8QuantBaseLinear, WeightQuantLinear, W8A8MXFP8QuantLinear
from mindiesd.quantization.mode import QuantAlgorithm
from mindiesd.quantization.quantize import smooth_quantize_w8a8, smooth_quantize, quantize
from mindiesd.quantization.quantize import weight_quantize, w8a16_quantize
from mindiesd.quantization.quantize import get_cfg_and_weights
from mindiesd.utils import ParametersInvalid, ConfigError

quantize_module = importlib.import_module("mindiesd.quantization.quantize")


class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)


class MockSafeTensorHandler:
    def __init__(self, data):
        self.data = data
        
    def get_tensor(self, key):
        return self.data.get(key, None)

    def keys(self):
        return self.data.keys()


def create_mock_handler(mock_data):
    return MockSafeTensorHandler(mock_data)


class TestSmoothQuantize(unittest.TestCase):
    def setUp(self):
        in_features = 10
        out_features = 10
        self.weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(1, dtype=torch.float16),
            "0.input_offset": torch.ones(1, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        self.weights2 = {
            "0.linear.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.linear.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.linear.input_scale": torch.ones(1, dtype=torch.float16),
            "0.linear.input_offset": torch.ones(1, dtype=torch.float16),
            "0.linear.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.linear.bias": torch.ones(out_features, dtype=torch.float32),
            "0.div.mul_scale": torch.ones(out_features, dtype=torch.float32)
        }
        self.weights3 = {
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.weight_scale": torch.ones(out_features, out_features, dtype=torch.float32),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        self.weights4 = {
            "0.linear.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.linear.weight_scale": torch.ones(out_features, out_features, dtype=torch.float32),
            "0.linear.bias": torch.ones(out_features, dtype=torch.float32),
            "0.div.mul_scale": torch.ones(out_features, dtype=torch.float32)
        }

    def test_smooth_quantize_w8a8_with_linear(self):
        layer = nn.Linear(10, 10)
        cfg = QuantConfig()
        quant_layer, is_modified = smooth_quantize_w8a8("0", layer, cfg, create_mock_handler(self.weights))
        self.assertIsInstance(quant_layer, W8A8QuantBaseLinear)
        self.assertTrue(is_modified)

    def test_smooth_quantize_w8a8_with_anti_linear(self):
        layer = nn.Linear(10, 10)
        cfg = QuantConfig()
        quant_layer, is_modified = smooth_quantize_w8a8("0", layer, cfg, create_mock_handler(self.weights2))
        self.assertIsInstance(quant_layer, W8A8QuantBaseLinear)
        self.assertTrue(is_modified)

    def test_smooth_quantize_w8a8_with_fuse_linear(self):
        layer = nn.Linear(10, 10)
        layer.fuse_algo = QuantAlgorithm.W8A8
        cfg = QuantConfig()
        quant_layer, is_modified = smooth_quantize_w8a8("0", layer, cfg, create_mock_handler(self.weights))
        self.assertIsInstance(quant_layer, W8A8QuantBaseLinear)
        self.assertTrue(is_modified)

    def test_smooth_quantize_w8a8_with_unsupported_layer(self):
        layer = nn.ReLU()
        cfg = QuantConfig()
        quant_layer, is_modified = smooth_quantize_w8a8("0", layer, cfg, create_mock_handler(self.weights))
        self.assertEqual(quant_layer, layer)
        self.assertFalse(is_modified)

    def test_smooth_quantize_w8a8_mxfp8_with_linear(self):
        layer = nn.Linear(10, 10)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W8A8_MXFP8)
        quant_layer, is_modified = smooth_quantize_w8a8("0", layer, cfg, create_mock_handler(self.weights3))
        self.assertIsInstance(quant_layer, W8A8MXFP8QuantLinear)
        self.assertTrue(is_modified)

    def test_smooth_quantize_w8a8_mxfp8_with_anti_linear(self):
        layer = nn.Linear(10, 10)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W8A8_MXFP8)
        quant_layer, is_modified = smooth_quantize_w8a8("0", layer, cfg, create_mock_handler(self.weights4))
        self.assertIsInstance(quant_layer, W8A8MXFP8QuantLinear)
        self.assertTrue(is_modified)

    def test_smooth_quantize_with_supported_algo(self):
        layer = nn.Linear(10, 10)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W8A8)
        quant_layer, is_modified = smooth_quantize("0", layer, cfg, create_mock_handler(self.weights))
        self.assertIsInstance(quant_layer, W8A8QuantBaseLinear)
        self.assertTrue(is_modified)

    def test_smooth_quantize_with_unsupported_algo(self):
        layer = nn.Linear(10, 10)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.NO_QUANT)
        quant_layer, is_modified = smooth_quantize("0", layer, cfg, create_mock_handler(self.weights))
        self.assertEqual(quant_layer, layer)
        self.assertFalse(is_modified)


class TestQuantize(unittest.TestCase):
    def setUp(self):
        in_features = 10
        out_features = 10
        self.weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.weight_scale": torch.ones(1, dtype=torch.bfloat16),
            "0.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(1, dtype=torch.float16),
            "0.input_offset": torch.ones(1, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        self.weights2 = {
            "0.weight_scale": torch.ones(1, dtype=torch.bfloat16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_non_quant_config(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        cfg = LayerQuantConfig()
        mock_func.return_value = (cfg, create_mock_handler(self.weights))
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg)
        self.assertEqual(quantized_model, model)

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_empty_model(self, mock_func):
        model = nn.Sequential()
        cfg = LayerQuantConfig()
        mock_func.return_value = (cfg, create_mock_handler(self.weights))
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg)
        self.assertEqual(quantized_model, model)

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_excluded_layer(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        cfg = LayerQuantConfig(quantized_layers={"1": QuantConfig(quant_algo=QuantAlgorithm.W8A8,
            exclude_layers=tuple(["0"]))})
        mock_func.return_value = (cfg, create_mock_handler(self.weights))
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg)
        self.assertEqual(quantized_model, model)

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_w8a8_layer(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        cfg = LayerQuantConfig(quantized_layers={"0": QuantConfig(quant_algo=QuantAlgorithm.W8A8)})
        mock_func.return_value = cfg, create_mock_handler(self.weights)
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg)
        self.assertIsInstance(quantized_model[0], W8A8QuantBaseLinear)

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_w8a8_layer(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        cfg = LayerQuantConfig(quantized_layers={"0": QuantConfig(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)})
        mock_func.return_value = cfg, create_mock_handler(self.weights)
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg)
        self.assertIsInstance(quantized_model[0], W8A8QuantBaseLinear)

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_w8a8_layer(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        cfg = LayerQuantConfig(quantized_layers={"0": QuantConfig(quant_algo=QuantAlgorithm.W8A8_TIMESTEP)})
        mock_func.return_value = cfg, create_mock_handler(self.weights)
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg, t_idx=5)
        self.assertIsInstance(quantized_model[0], W8A8QuantBaseLinear)

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_custom_w8a8_layer(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        cfg = LayerQuantConfig(quantized_layers={"0": QuantConfig(quant_algo=QuantAlgorithm.W8A8)})
        mock_func.return_value = cfg, create_mock_handler(self.weights)
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg, map={CustomLinear: W8A8QuantBaseLinear})
        self.assertIsInstance(quantized_model[0], W8A8QuantBaseLinear)

    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_w8a8_fuse_layer(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        model[0].fuse_algo = QuantAlgorithm.W8A8
        cfg = LayerQuantConfig(quantized_layers={"0": QuantConfig(quant_algo=QuantAlgorithm.W8A8)})
        mock_func.return_value = cfg, create_mock_handler(self.weights)
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg)
        self.assertIsInstance(quantized_model[0], W8A8QuantBaseLinear)


    @mock.patch.object(quantize_module, "get_cfg_and_weights")  # 装饰器指定被mock的函数
    def test_quantize_with_w8a16_layer(self, mock_func):
        model = nn.Sequential(nn.Linear(10, 10))
        cfg = LayerQuantConfig(quantized_layers={"0": QuantConfig(quant_algo=QuantAlgorithm.W8A16)})
        mock_func.return_value = cfg, create_mock_handler(self.weights2)
        quantized_model = quantize.__wrapped__(model, "path", custom_cfg=cfg)
        self.assertIsInstance(quantized_model[0], WeightQuantLinear)

    @mock.patch("mindiesd.utils.file_utils.safe_open")
    @mock.patch("mindiesd.utils.file_utils.check_file_safety")
    def test_quantize_decorator_invalid_config(self, mock_check_safety, mock_safe_open):
        # Mock file with invalid config
        mock_file = mock.MagicMock()
        mock_file.read.return_value = json.dumps({"layer1": "W8A8"})
        mock_safe_open.return_value.__enter__.return_value = mock_file

        # Test invalid config case
        model = nn.Sequential(nn.Linear(10, 10))
        with self.assertRaises(ParametersInvalid):
            quantized_model = quantize(model, "path/to/quant_des.json")

    @mock.patch("mindiesd.utils.file_utils.safe_open")
    @mock.patch("mindiesd.utils.file_utils.check_file_safety")
    def test_quantize_decorator_file_error(self, mock_check_safety, mock_safe_open):
        # Mock file operation error
        mock_safe_open.side_effect = FileNotFoundError()

        # Test file error case
        model = nn.Sequential(nn.Linear(10, 10))
        with self.assertRaises(FileNotFoundError):
            quantized_model = quantize(model, "path/to/quant_des.json")


class TestWeightQuantize(unittest.TestCase):
    def setUp(self):
        in_features = 8
        out_features = 8
        self.weights = {
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32),
            "0.weight_scale": torch.ones(out_features, dtype=torch.float16),
            "0.weight_offset": torch.ones(out_features, dtype=torch.float16)
        }

    def test_weight_quantize_with_w8a16(self):
        layer = nn.Linear(8, 8)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W8A16)
        quant_layer, is_modified = weight_quantize("0", layer, cfg, create_mock_handler(self.weights))
        self.assertIsInstance(quant_layer, WeightQuantLinear)
        self.assertTrue(is_modified)

    def test_weight_quantize_with_w4a16(self):
        layer = nn.Linear(8, 8)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W4A16)
        quant_layer, is_modified = weight_quantize("0", layer, cfg, create_mock_handler(self.weights))
        self.assertIsInstance(quant_layer, WeightQuantLinear)
        self.assertTrue(is_modified)

    def test_weight_quantize_with_unsupported_algo(self):
        layer = nn.Linear(8, 8)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.NO_QUANT)
        quant_layer, is_modified = weight_quantize("0", layer, cfg, create_mock_handler(self.weights))
        self.assertEqual(quant_layer, layer)
        self.assertFalse(is_modified)

    def test_w8a16_quantize_with_linear(self):
        layer = nn.Linear(8, 8)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W8A16)
        quant_layer, is_modified = w8a16_quantize("0", layer, cfg, create_mock_handler(self.weights))
        self.assertIsInstance(quant_layer, WeightQuantLinear)
        self.assertTrue(is_modified)

    def test_w8a16_quantize_with_unsupported_layer(self):
        layer = nn.ReLU()
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W8A16)
        quant_layer, is_modified = w8a16_quantize("0", layer, cfg, create_mock_handler(self.weights))
        self.assertEqual(quant_layer, layer)
        self.assertFalse(is_modified)

    def test_w8a16_quantize_with_custom_map(self):
        layer = nn.Linear(8, 8)
        cfg = QuantConfig(quant_algo=QuantAlgorithm.W8A16)
        custom_map = {nn.Linear: WeightQuantLinear}
        quant_layer, is_modified = w8a16_quantize("0", layer, cfg, create_mock_handler(self.weights), map=custom_map)
        self.assertIsInstance(quant_layer, WeightQuantLinear)
        self.assertTrue(is_modified)


class TestGetCfgAndWeights(unittest.TestCase):
    def setUp(self):
        self.quant_des_path = "path/to/quant_des.json"
        self.quant_weight_path = "path/to/quant_model_weight_w8a8.safetensors"
        self.quant_des_dict = {
            "model_quant_type": "W8A8",
            "layer1": "W8A8",
            "layer2": "FLOAT"
        }
        self.quant_weights = {"weight": torch.ones(1)}

    @mock.patch("mindiesd.utils.file_utils.safe_open")
    @mock.patch("mindiesd.utils.file_utils.check_file_safety")
    @mock.patch("safetensors.safe_open")
    def test_get_cfg_and_weights_normal(self, mock_safe_open0, mock_check_safety, mock_safe_open1):
        # Mock file operations
        mock_file = mock.MagicMock()
        mock_file.read.return_value = json.dumps(self.quant_des_dict)
        mock_safe_open1.return_value.__enter__.return_value = mock_file
        mock_safe_open0.return_value.__enter__.return_value = create_mock_handler(self.quant_weights)

        # Test normal case
        cfg, weights = get_cfg_and_weights(self.quant_des_path)
        
        # Verify results
        self.assertEqual(cfg.quant_algo, QuantAlgorithm.W8A8)
        self.assertEqual(cfg.exclude_layers, tuple(["layer2"]))
        
        # Verify calls
        mock_safe_open1.assert_called_once()
        mock_check_safety.assert_called()
        mock_safe_open0.assert_called_once()


if __name__ == '__main__':
    unittest.main()
