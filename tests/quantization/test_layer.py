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

import unittest
from unittest import mock
from unittest.mock import patch
import torch
import torch_npu

from mindiesd.quantization.layer import W8A8QuantLinear, WeightQuantLinear, W8A8TimeStepQuantLinear, \
    W8A8MXFP8QuantLinear
from mindiesd.quantization.layer import QuantFA
from mindiesd.quantization.mode import QuantAlgorithm
from mindiesd.quantization.utils import get_quant_weight, TimestepManager


class MockSafeTensorHandler:
    def __init__(self, data):
        self.data = data
        
    def get_tensor(self, key):
        return self.data.get(key, None)

    def keys(self):
        return self.data.keys()


def create_mock_handler(mock_data):
    return MockSafeTensorHandler(mock_data)


def mock_npu_quant_matmul(*args, **kwargs):
    x1 = args[0] if len(args) >= 1 else None
    x2 = args[1] if len(args) >= 2 else None
    output_dtype = kwargs.get('output_dtype', torch.float16)

    batch_dims = x1.shape[:-1]
    out_features = x2.shape[-1] if x2 is not None else 0
    output_shape = batch_dims + (out_features,)

    output = torch.randn(*output_shape, dtype=output_dtype).to(x1.device)

    bias = kwargs.get('bias')
    if bias is not None:
        output += bias.to(output.dtype).to(output.device)
    return output


class TestQuantLinearFloat16(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        dtype_mocks = {
            'float8_e4m3fn': torch.float16,
            'float8_e8m0fnu': torch.float16
        }
        for dtype_name, dtype_val in dtype_mocks.items():
            if not hasattr(torch_npu, dtype_name):
                setattr(torch_npu, dtype_name, dtype_val)

        def mock_npu_dynamic_mx_quant(x, dst_type=None):
            scale = torch.ones(1, dtype=torch.float16).to(x.device)
            return x, scale

        def mock_npu_dtype_cast(tensor, dtype):
            return tensor
        torch_npu.npu_dtype_cast = mock_npu_dtype_cast

        if not hasattr(torch_npu, 'npu_dynamic_mx_quant'):
            torch_npu.npu_dynamic_mx_quant = mock_npu_dynamic_mx_quant

    def test_flatten_linear(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(1, dtype=torch.float16),
            "0.input_offset": torch.ones(1, dtype=torch.int8),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8QuantLinear(in_features, out_features, bias=True,
            weights=create_mock_handler(weights), prefix="0", dtype=torch.float16).npu()

        x = torch.randn(32, 8, 4, in_features).to(torch.float16).npu()
        output = linear(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (32, 8, 4, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_static(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(1, dtype=torch.float16),
            "0.input_offset": torch.ones(1, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8QuantLinear(in_features, out_features, bias=True,
            is_dynamic=False, weights=create_mock_handler(weights), prefix="0", dtype=torch.float16).npu()

        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.quant_matmul(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_timestep_static(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(100, out_features, dtype=torch.int32),
            "0.weight_scale": torch.ones(1, out_features, dtype=torch.float16),
            "0.deq_scale": torch.ones(100, out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(100, 1, dtype=torch.float16),
            "0.input_offset": torch.ones(100, 1, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        TimestepManager.set_timestep_idx_max(10)
        TimestepManager.set_timestep_idx(10)
        linear = W8A8TimeStepQuantLinear(in_features, out_features, bias=True,
            is_dynamic=False, weights=create_mock_handler(weights), prefix="0", dtype=torch.float16, t_idx=5).npu()
        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_timestep_dynamic(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(100, out_features, dtype=torch.int32),
            "0.weight_scale": torch.ones(1, out_features, dtype=torch.float16),
            "0.deq_scale": torch.ones(100, out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(100, 1, dtype=torch.float16),
            "0.input_offset": torch.ones(100, 1, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        TimestepManager.set_timestep_idx_max(10)
        TimestepManager.set_timestep_idx(1)
        linear = W8A8TimeStepQuantLinear(in_features, out_features, bias=True,
            is_dynamic=False, weights=create_mock_handler(weights), prefix="0", dtype=torch.float16, t_idx=5).npu()
        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_static_with_anti(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(1, dtype=torch.float16),
            "0.input_offset": torch.ones(1, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        mul_scale = torch.ones(in_features, dtype=torch.float32)
        linear = W8A8QuantLinear(in_features, out_features, bias=True, is_dynamic=False,
            weights=create_mock_handler(weights), prefix="0", dtype=torch.float16, mul_scale=mul_scale).npu()

        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_static_with_fuse(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(1, dtype=torch.float16),
            "0.input_offset": torch.ones(1, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8QuantLinear(in_features, out_features, bias=True, is_dynamic=False,
            weights=create_mock_handler(weights), prefix="0",
                dtype=torch.float16, fuse_algo=QuantAlgorithm.W8A8).npu()

        x = torch.randn(2, 32, in_features).to(torch.int8).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_quant_matmul_dynamic(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.weight_scale": torch.ones(out_features, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8QuantLinear(in_features, out_features, bias=True, is_dynamic=True,
            weights=create_mock_handler(weights), prefix="0", dtype=torch.float16).npu()

        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_dynamic_with_anti(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.weight_scale": torch.ones(out_features, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        mul_scale = torch.ones(in_features, dtype=torch.float32)
        linear = W8A8QuantLinear(in_features, out_features, bias=True, is_dynamic=True,
            weights=create_mock_handler(weights), prefix="0", dtype=torch.float16, mul_scale=mul_scale).npu()

        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_quant_matmul_w8a8mxfp8_dynamic_basic(self, _):
        in_features = 128
        out_features = 64
        weights = {
            "0.weight_scale": torch.ones(out_features, 2, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.float16),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8MXFP8QuantLinear(
            in_features, out_features, bias=True,
            weights=create_mock_handler(weights), prefix="0", dtype=torch.float16
        ).npu()

        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)

        self.stream.synchronize()

        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertEqual(output.dtype, torch.float16)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(linear.weight_scale.shape, (out_features, 1, 2))

    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_quant_matmul_w8a8mxfp8_dynamic_with_mul_scale(self, _):
        in_features = 128
        out_features = 64
        weights = {
            "0.weight_scale": torch.ones(out_features, 2, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.float16),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        mul_scale = torch.ones(in_features, dtype=torch.float32)

        linear = W8A8MXFP8QuantLinear(
            in_features, out_features, bias=True,
            weights=create_mock_handler(weights), prefix="0", dtype=torch.float16,
            mul_scale=mul_scale
        ).npu()

        x = torch.randn(4, 16, in_features).to(torch.float16).npu()
        output = linear.forward(x)

        self.stream.synchronize()

        self.assertEqual(output.shape, (4, 16, out_features))
        self.assertEqual(linear.mul_scale.shape, (in_features,))


class TestQuantLinearBFloat16(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()

    def test_flatten_linear(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.float),
            "0.input_scale": torch.ones(1, dtype=torch.bfloat16),
            "0.input_offset": torch.ones(1, dtype=torch.bfloat16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8QuantLinear(in_features, out_features, bias=True,
            weights=create_mock_handler(weights), prefix="0").npu()

        x = torch.randn(32, 8, 4, in_features).to(torch.bfloat16).npu()
        output = linear(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (32, 8, 4, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_static(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.float),
            "0.input_scale": torch.ones(1, dtype=torch.bfloat16),
            "0.input_offset": torch.ones(1, dtype=torch.bfloat16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8QuantLinear(in_features, out_features, bias=True,
            is_dynamic=False, weights=create_mock_handler(weights), prefix="0").npu()

        x = torch.randn(2, 32, in_features).to(torch.bfloat16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_quant_matmul_dynamic(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.weight_scale": torch.ones(out_features, dtype=torch.bfloat16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32)
        }
        linear = W8A8QuantLinear(in_features, out_features, bias=True,
            is_dynamic=True, weights=create_mock_handler(weights), prefix="0").npu()

        x = torch.randn(2, 32, in_features).to(torch.bfloat16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)


class TestFA3Float16(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        self.weights = {
            "test_layer.fa_q.scale": torch.ones(8, 1, dtype=torch.float16),
            "test_layer.fa_k.scale": torch.ones(8, 1, dtype=torch.float16),
            "test_layer.fa_v.scale": torch.ones(8, 1, dtype=torch.float16),
            "test_layer.fa_q.offset": torch.ones(8, 1, dtype=torch.float16),
            "test_layer.fa_k.offset": torch.ones(8, 1, dtype=torch.float16),
            "test_layer.fa_v.offset": torch.ones(8, 1, dtype=torch.float16)
        }
        # Mock torch_atb.Operation to avoid hardware dependencies
        self.operation_patcher = mock.patch('torch_atb.Operation')
        self.mock_operation = self.operation_patcher.start()
        # Configure the mock to return a callable with a forward method
        self.mock_operation_instance = mock.MagicMock()
        self.mock_operation_instance.forward.return_value = [torch.ones(32, 8, 16, dtype=torch.bfloat16).npu()]
        self.mock_operation.return_value = self.mock_operation_instance

    def tearDown(self):
        self.operation_patcher.stop()

    def test_init(self):
        # Test initializing QuantFA with valid parameters
        fa3 = QuantFA(ori_head_num=8, ori_inner_dim=64, prefix="test_layer",
            quant_weights=create_mock_handler(self.weights), dtype=torch.float16).npu()
        
        # Verify attributes are set correctly
        self.assertEqual(fa3.q_scale.shape, (8, 1))
        self.assertEqual(fa3.k_scale.shape, (8, 1))
        self.assertEqual(fa3.v_scale.shape, (8, 1))
        self.assertEqual(fa3.q_offset.shape, (8, 1))
        self.assertEqual(fa3.k_offset.shape, (8, 1))
        self.assertEqual(fa3.v_offset.shape, (8, 1))
        self.assertEqual(fa3.dtype, torch.float16)
        
        # Verify torch_atb.Operation was called the expected number of times
        self.assertEqual(self.mock_operation.call_count, 4)

    def test_forward(self):
        # Test forward pass of QuantFA
        fa3 = QuantFA(ori_head_num=8, ori_inner_dim=64, prefix="test_layer",
            quant_weights=create_mock_handler(self.weights)).npu()
        
        # Create test inputs
        query = torch.randn(32, 8, 8, dtype=torch.float16).npu()
        key = torch.randn(32, 8, 8, dtype=torch.float16).npu()
        value = torch.randn(32, 8, 8, dtype=torch.float16).npu()
        seq_len = [32]
        
        # Call forward method
        output = fa3.forward(query, key, value, seq_len)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (32, 8, 16))
        self.assertIsInstance(output, torch.Tensor)
        
        # Verify quant operations were called
        self.assertEqual(self.mock_operation_instance.forward.call_count, 4)


class TestFA3BFloat16(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        self.weights = {
            "test_layer.fa_q.scale": torch.ones(8, 1, dtype=torch.bfloat16),
            "test_layer.fa_k.scale": torch.ones(8, 1, dtype=torch.bfloat16),
            "test_layer.fa_v.scale": torch.ones(8, 1, dtype=torch.bfloat16),
            "test_layer.fa_q.offset": torch.ones(8, 1, dtype=torch.bfloat16),
            "test_layer.fa_k.offset": torch.ones(8, 1, dtype=torch.bfloat16),
            "test_layer.fa_v.offset": torch.ones(8, 1, dtype=torch.bfloat16)
        }
        # Mock torch_atb.Operation to avoid hardware dependencies
        self.operation_patcher = mock.patch('torch_atb.Operation')
        self.mock_operation = self.operation_patcher.start()
        # Configure the mock to return a callable with a forward method
        self.mock_operation_instance = mock.MagicMock()
        self.mock_operation_instance.forward.return_value = [torch.ones(32, 8, 16, dtype=torch.bfloat16).npu()]
        self.mock_operation.return_value = self.mock_operation_instance

    def tearDown(self):
        self.operation_patcher.stop()

    def test_init(self):
        # Test initializing QuantFA with valid parameters
        fa3 = QuantFA(ori_head_num=8, ori_inner_dim=64, prefix="test_layer",
            quant_weights=create_mock_handler(self.weights)).npu()
        
        # Verify attributes are set correctly
        self.assertEqual(fa3.q_scale.shape, (8, 1))
        self.assertEqual(fa3.k_scale.shape, (8, 1))
        self.assertEqual(fa3.v_scale.shape, (8, 1))
        self.assertEqual(fa3.q_offset.shape, (8, 1))
        self.assertEqual(fa3.k_offset.shape, (8, 1))
        self.assertEqual(fa3.v_offset.shape, (8, 1))
        self.assertEqual(fa3.dtype, torch.bfloat16)
        
        # Verify torch_atb.Operation was called the expected number of times
        self.assertEqual(self.mock_operation.call_count, 4)

    def test_forward(self):
        # Test forward pass of QuantFA
        fa3 = QuantFA(ori_head_num=8, ori_inner_dim=64, prefix="test_layer",
            quant_weights=create_mock_handler(self.weights)).npu()
        
        # Create test inputs
        query = torch.randn(32, 8, 8, dtype=torch.bfloat16).npu()
        key = torch.randn(32, 8, 8, dtype=torch.bfloat16).npu()
        value = torch.randn(32, 8, 8, dtype=torch.bfloat16).npu()
        seq_len = [32]
        
        # Call forward method
        output = fa3.forward(query, key, value, seq_len)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (32, 8, 16))
        self.assertIsInstance(output, torch.Tensor)
        
        # Verify quant operations were called
        self.assertEqual(self.mock_operation_instance.forward.call_count, 4)


class TestWeightQuantLinearBFloat16(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        self.in_features = 128
        self.out_features = 64
        self.weights = {
            "0.weight_scale": torch.ones(self.out_features, dtype=torch.bfloat16),
            "0.weight_offset": torch.ones(self.out_features, dtype=torch.bfloat16),
            "0.weight": torch.ones(self.out_features, self.in_features, dtype=torch.int8),
            "0.bias": torch.ones(self.out_features, dtype=torch.float32)
        }

    def test_init(self):
        # Test initialization of WeightQuantLinear
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0"
        ).npu()
        
        # Verify attributes are set correctly
        self.assertEqual(linear.weight_scale.shape, (self.out_features,))
        self.assertEqual(linear.weight.shape, (self.in_features, self.out_features))
        self.assertEqual(linear.bias.shape, (self.out_features,))
        self.assertEqual(linear.input_feature, self.in_features)
        self.assertEqual(linear.output_feature, self.out_features)
        self.assertEqual(linear.weight_scale.dtype, torch.bfloat16)

    def test_forward_2d(self):
        # Test forward pass with 2D input
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0"
        ).npu()
        
        x = torch.randn(32, self.in_features).to(torch.bfloat16).npu()
        output = linear(x)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (32, self.out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_3d(self):
        # Test forward pass with 3D input (testing _flatten_linear)
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0"
        ).npu()
        
        x = torch.randn(8, 32, self.in_features).to(torch.bfloat16).npu()
        output = linear(x)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (8, 32, self.out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_4d(self):
        # Test forward pass with 4D input (testing _flatten_linear with higher dimensions)
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0",
        ).npu()
        
        x = torch.randn(4, 8, 32, self.in_features).to(torch.bfloat16).npu()
        output = linear(x)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (4, 8, 32, self.out_features))
        self.assertIsInstance(output, torch.Tensor)


class TestWeightQuantLinearFloat(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        self.in_features = 128
        self.out_features = 64
        self.weights = {
            "0.weight_scale": torch.ones(self.out_features, dtype=torch.float16),
            "0.weight_offset": torch.ones(self.out_features, dtype=torch.float16),
            "0.weight": torch.ones(self.out_features, self.in_features, dtype=torch.int8),
            "0.bias": torch.ones(self.out_features, dtype=torch.float16)
        }

    def test_init(self):
        # Test initialization of WeightQuantLinear
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0",
            dtype=torch.float16
        ).npu()
        
        # Verify attributes are set correctly
        self.assertEqual(linear.weight_scale.shape, (self.out_features,))
        self.assertEqual(linear.weight.shape, (self.in_features, self.out_features))
        self.assertEqual(linear.bias.shape, (self.out_features,))
        self.assertEqual(linear.input_feature, self.in_features)
        self.assertEqual(linear.output_feature, self.out_features)
        self.assertEqual(linear.weight_scale.dtype, torch.float16)

    def test_forward_2d(self):
        # Test forward pass with 2D input
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0",
            dtype=torch.float16
        ).npu()
        
        x = torch.randn(32, self.in_features).to(torch.float16).npu()
        output = linear(x)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (32, self.out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_3d(self):
        # Test forward pass with 3D input (testing _flatten_linear)
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0",
            dtype=torch.float16
        ).npu()
        
        x = torch.randn(8, 32, self.in_features).to(torch.float16).npu()
        output = linear(x)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (8, 32, self.out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_4d(self):
        # Test forward pass with 4D input (testing _flatten_linear with higher dimensions)
        linear = WeightQuantLinear(
            self.in_features, 
            self.out_features, 
            bias=True, 
            weights=create_mock_handler(self.weights), 
            prefix="0",
            dtype=torch.float16
        ).npu()
        
        x = torch.randn(4, 8, 32, self.in_features).to(torch.float16).npu()
        output = linear(x)
        self.stream.synchronize()
        
        # Verify output shape and type
        self.assertEqual(output.shape, (4, 8, 32, self.out_features))
        self.assertIsInstance(output, torch.Tensor)


if __name__ == '__main__':
    unittest.main()