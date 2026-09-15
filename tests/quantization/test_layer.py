#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# pylint: disable=no-member,redefined-outer-name
import importlib
import os
import sys
import unittest
from itertools import product
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch_npu

from mindiesd import quant_attention
from mindiesd.quantization.layer import (
    W8A8QuantLinear,
    WeightQuantLinear,
    W8A8TimeStepQuantLinear,
    W8A8MXFP8QuantLinear,
    FP8RotateQuantFA,
    W8A8OnlineQuantLinear,
    W8A8MXFP8OnlineQuantLinear,
    W4A4MXFP4OnlineQuantLinear,
    W4A4MXFP4DualOnlineQuantLinear,
)
from mindiesd.quantization.config import QuantConfig, TimestepPolicyConfig
from mindiesd.quantization.mode import FP8FAMode, QuantAlgorithm
from mindiesd.utils import ParametersInvalid
from mindiesd.quantization.utils import TimestepManager
from mindiesd.utils.get_platform import is_a5_device


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


def mock_npu_dynamic_quant(x, *args, **kwargs):
    scale = torch.ones(x.shape[:-1].numel(), dtype=torch.float32, device=x.device)
    return torch.zeros_like(x, dtype=torch.int8), scale


def mock_npu_dynamic_mx_quant(x, *args, **kwargs):
    scale = torch.ones(x.shape[0], 2, dtype=torch.float32, device=x.device)
    return x, scale


def mock_npu_dynamic_dual_level_mx_quant(x, *args, **kwargs):
    fp4 = torch.zeros_like(x, dtype=torch.int8)
    l0_scale = torch.ones(x.shape[0], 1, dtype=torch.float32, device=x.device)
    l1_scale = torch.ones(x.shape[0], 2, dtype=torch.float32, device=x.device)
    return fp4, l0_scale, l1_scale


def mock_npu_dual_level_quant_matmul(*args, **kwargs):
    x1 = args[0]
    x2 = args[1]
    output_dtype = kwargs.get('output_dtype', torch.float16)
    bias = kwargs.get('bias')
    out_features = bias.shape[0] if bias is not None else x2.shape[-1]
    output_shape = x1.shape[:-1] + (out_features,)
    output = torch.randn(*output_shape, dtype=output_dtype).to(x1.device)
    if bias is not None:
        output += bias.to(output.dtype).to(output.device)
    return output


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
class TestQuantLinearFloat16(unittest.TestCase):
    def _patch_torch_npu_attr(self, name, value):
        patcher = patch.object(torch_npu, name, value, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        dtype_mocks = {'float8_e4m3fn': torch.float16, 'float8_e8m0fnu': torch.float16}
        dtype_mocks['float4_e2m1fn_x2'] = torch.int8
        for dtype_name, dtype_val in dtype_mocks.items():
            if not hasattr(torch_npu, dtype_name):
                self._patch_torch_npu_attr(dtype_name, dtype_val)

        def mock_dynamic_mx_quant(x, dst_type=None):
            scale = torch.ones(1, dtype=torch.float16).to(x.device)
            return x, scale

        def mock_npu_dtype_cast(tensor, dtype):
            return tensor

        def mock_npu_format_cast(tensor, *args, **kwargs):
            return tensor

        self._patch_torch_npu_attr('npu_dtype_cast', mock_npu_dtype_cast)
        self._patch_torch_npu_attr('npu_format_cast', mock_npu_format_cast)

        if not hasattr(torch_npu, 'npu_dynamic_mx_quant'):
            self._patch_torch_npu_attr('npu_dynamic_mx_quant', mock_dynamic_mx_quant)

        if not hasattr(torch_npu, 'npu_dynamic_dual_level_mx_quant'):
            self._patch_torch_npu_attr('npu_dynamic_dual_level_mx_quant', mock_npu_dynamic_dual_level_mx_quant)

        if not hasattr(torch_npu, 'npu_dual_level_quant_matmul'):
            self._patch_torch_npu_attr('npu_dual_level_quant_matmul', mock_npu_dual_level_quant_matmul)

    def test_flatten_linear(self):
        in_features = 128
        out_features = 64
        weights = {
            "0.quant_bias": torch.ones(out_features, dtype=torch.int32),
            "0.deq_scale": torch.ones(out_features, dtype=torch.int64),
            "0.input_scale": torch.ones(1, dtype=torch.float16),
            "0.input_offset": torch.ones(1, dtype=torch.int8),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.int8),
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8QuantLinear(
            in_features, out_features, bias=True, weights=create_mock_handler(weights), prefix="0", dtype=torch.float16
        ).npu()
        self.assertEqual(linear.input_offset.dtype, torch.int8)

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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8QuantLinear(
            in_features,
            out_features,
            bias=True,
            is_dynamic=False,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
        ).npu()

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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        TimestepManager.set_timestep_idx_max(10)
        TimestepManager.set_timestep_idx(10)
        linear = W8A8TimeStepQuantLinear(
            in_features,
            out_features,
            bias=True,
            is_dynamic=False,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
            t_idx=5,
        ).npu()
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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        TimestepManager.set_timestep_idx_max(10)
        TimestepManager.set_timestep_idx(1)
        linear = W8A8TimeStepQuantLinear(
            in_features,
            out_features,
            bias=True,
            is_dynamic=False,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
            t_idx=5,
        ).npu()
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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        mul_scale = torch.ones(in_features, dtype=torch.float32)
        linear = W8A8QuantLinear(
            in_features,
            out_features,
            bias=True,
            is_dynamic=False,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
            mul_scale=mul_scale,
        ).npu()

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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8QuantLinear(
            in_features,
            out_features,
            bias=True,
            is_dynamic=False,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
            fuse_algo=QuantAlgorithm.W8A8,
        ).npu()

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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8QuantLinear(
            in_features,
            out_features,
            bias=True,
            is_dynamic=True,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
        ).npu()

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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        mul_scale = torch.ones(in_features, dtype=torch.float32)
        linear = W8A8QuantLinear(
            in_features,
            out_features,
            bias=True,
            is_dynamic=True,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
            mul_scale=mul_scale,
        ).npu()

        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)

    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_quant_matmul_w8a8mxfp8_dynamic_basic(self, _, mock_dynamic_mx_quant):
        in_features = 128
        out_features = 64
        weights = {
            "0.weight_scale": torch.ones(out_features, 2, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.float16),
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8MXFP8QuantLinear(
            in_features, out_features, bias=True, weights=create_mock_handler(weights), prefix="0", dtype=torch.float16
        ).npu()

        x = torch.randn(2, 32, in_features).to(torch.float16).npu()
        output = linear.forward(x)

        self.stream.synchronize()

        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertEqual(output.dtype, torch.float16)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(linear.weight_scale.shape, (out_features, 1, 2))
        self.assertEqual(mock_dynamic_mx_quant.call_count, 1)

    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_quant_matmul_w8a8mxfp8_dynamic_with_mul_scale(self, _, mock_dynamic_mx_quant):
        in_features = 128
        out_features = 64
        weights = {
            "0.weight_scale": torch.ones(out_features, 2, dtype=torch.float16),
            "0.weight": torch.ones(out_features, in_features, dtype=torch.float16),
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        mul_scale = torch.ones(in_features, dtype=torch.float32)

        linear = W8A8MXFP8QuantLinear(
            in_features,
            out_features,
            bias=True,
            weights=create_mock_handler(weights),
            prefix="0",
            dtype=torch.float16,
            mul_scale=mul_scale,
        ).npu()

        x = torch.randn(4, 16, in_features).to(torch.float16).npu()
        output = linear.forward(x)

        self.stream.synchronize()

        self.assertEqual(output.shape, (4, 16, out_features))
        self.assertEqual(linear.mul_scale.shape, (in_features,))
        self.assertEqual(mock_dynamic_mx_quant.call_count, 1)

    @patch('torch_npu.npu_dynamic_quant', side_effect=mock_npu_dynamic_quant)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w8a8_online_quant_linear_forward(self, mock_quant_matmul, mock_dynamic_quant):
        in_features = 128
        out_features = 64
        linear = W8A8OnlineQuantLinear(nn.Linear(in_features, out_features), dtype=torch.float16).npu()

        x = torch.randn(2, 16, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()

        self.assertEqual(output.shape, (2, 16, out_features))
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(mock_dynamic_quant.call_count, 2)
        self.assertEqual(mock_quant_matmul.call_args.kwargs["output_dtype"], torch.float16)

    @patch('torch_npu.npu_dynamic_quant', side_effect=mock_npu_dynamic_quant)
    def test_w8a8_online_quant_linear_without_bias(self, mock_dynamic_quant):
        in_features = 128
        out_features = 64
        linear = W8A8OnlineQuantLinear(nn.Linear(in_features, out_features, bias=False), dtype=torch.float16).npu()

        self.assertIsNone(linear.bias)
        self.assertEqual(linear.weight.shape, (in_features, out_features))
        self.assertEqual(mock_dynamic_quant.call_count, 1)

    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w8a8_mxfp8_online_quant_linear_forward(self, mock_quant_matmul, mock_dynamic_mx_quant):
        in_features = 128
        out_features = 64
        linear = W8A8MXFP8OnlineQuantLinear(nn.Linear(in_features, out_features), dtype=torch.float16).npu()

        x = torch.randn(4, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()

        self.assertEqual(output.shape, (4, out_features))
        self.assertEqual(linear.weight_scale.shape, (out_features, 1, 2))
        self.assertEqual(mock_dynamic_mx_quant.call_count, 2)
        self.assertEqual(mock_quant_matmul.call_args.kwargs["group_sizes"], [1, 1, 32])

    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w4a4_mxfp4_online_quant_linear_uses_w4a4_by_default(self, mock_quant_matmul, mock_dynamic_mx_quant):
        in_features = 128
        out_features = 64
        linear = W4A4MXFP4OnlineQuantLinear(nn.Linear(in_features, out_features), dtype=torch.float16).npu()

        x = torch.randn(2, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()

        self.assertEqual(output.shape, (2, out_features))
        self.assertEqual(mock_dynamic_mx_quant.call_count, 2)
        self.assertEqual(mock_quant_matmul.call_args.kwargs["x1_dtype"], torch_npu.float4_e2m1fn_x2)
        self.assertEqual(mock_quant_matmul.call_args.kwargs["x2_dtype"], torch_npu.float4_e2m1fn_x2)

    @patch('torch_npu.npu_dynamic_quant', side_effect=mock_npu_dynamic_quant)
    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w4a4_mxfp4_online_quant_linear_fallback_timestep(
        self, mock_quant_matmul, mock_dynamic_mx_quant, mock_dynamic_quant
    ):
        in_features = 128
        out_features = 64
        timestep_config = TimestepPolicyConfig()
        timestep_config.register([5], "W4A8", target="w4a4_linear")
        linear = W4A4MXFP4OnlineQuantLinear(
            nn.Linear(in_features, out_features),
            dtype=torch.float16,
            quant_config=QuantConfig(timestep_config=timestep_config),
        ).npu()

        TimestepManager.set_timestep_idx_max(10)
        TimestepManager.set_timestep_idx(5)
        x = torch.randn(2, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()

        self.assertEqual(output.shape, (2, out_features))
        self.assertEqual(mock_dynamic_mx_quant.call_count, 2)
        mock_dynamic_quant.assert_not_called()
        self.assertNotIn("x1_dtype", mock_quant_matmul.call_args.kwargs)
        self.assertEqual(mock_quant_matmul.call_args.kwargs["x2_dtype"], torch_npu.float4_e2m1fn_x2)
        bias = mock_quant_matmul.call_args.kwargs["bias"]
        self.assertEqual(bias.shape, (1, out_features))
        self.assertEqual(bias.dtype, torch.bfloat16)

    @patch('torch_npu.npu_dynamic_dual_level_mx_quant', side_effect=mock_npu_dynamic_dual_level_mx_quant)
    @patch('torch_npu.npu_dual_level_quant_matmul', side_effect=mock_npu_dual_level_quant_matmul)
    def test_w4a4_mxfp4_dual_online_quant_linear_forward(self, mock_dual_matmul, mock_dual_quant):
        in_features = 128
        out_features = 64
        linear = W4A4MXFP4DualOnlineQuantLinear(nn.Linear(in_features, out_features), dtype=torch.float16).npu()

        x = torch.randn(2, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()

        self.assertEqual(output.shape, (2, out_features))
        self.assertEqual(mock_dual_quant.call_count, 2)
        self.assertEqual(mock_dual_matmul.call_args.kwargs["output_dtype"], torch.float16)

    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant)
    @patch('torch_npu.npu_dynamic_quant', side_effect=mock_npu_dynamic_quant)
    @patch('torch_npu.npu_dynamic_dual_level_mx_quant', side_effect=mock_npu_dynamic_dual_level_mx_quant)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w4a4_mxfp4_dual_online_quant_linear_fallback_timestep(
        self, mock_quant_matmul, mock_dual_quant, mock_dynamic_quant, mock_dynamic_mx_quant
    ):
        in_features = 128
        out_features = 64
        timestep_config = TimestepPolicyConfig()
        timestep_config.register([6], "W4A8", target="w4a4_linear")
        linear = W4A4MXFP4DualOnlineQuantLinear(
            nn.Linear(in_features, out_features),
            dtype=torch.float16,
            quant_config=QuantConfig(timestep_config=timestep_config),
        ).npu()

        TimestepManager.set_timestep_idx_max(10)
        TimestepManager.set_timestep_idx(6)
        x = torch.randn(2, in_features).to(torch.float16).npu()
        output = linear.forward(x)
        self.stream.synchronize()

        self.assertEqual(output.shape, (2, out_features))
        self.assertEqual(mock_dual_quant.call_count, 1)
        mock_dynamic_quant.assert_not_called()
        self.assertEqual(mock_dynamic_mx_quant.call_count, 1)
        self.assertEqual(mock_quant_matmul.call_args.kwargs["x2_dtype"], torch_npu.float4_e2m1fn_x2)
        bias = mock_quant_matmul.call_args.kwargs["bias"]
        self.assertEqual(bias.shape, (1, out_features))
        self.assertEqual(bias.dtype, torch.bfloat16)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8QuantLinear(
            in_features, out_features, bias=True, weights=create_mock_handler(weights), prefix="0"
        ).npu()
        self.assertEqual(linear.input_offset.dtype, torch.bfloat16)

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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8QuantLinear(
            in_features, out_features, bias=True, is_dynamic=False, weights=create_mock_handler(weights), prefix="0"
        ).npu()

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
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }
        linear = W8A8QuantLinear(
            in_features, out_features, bias=True, is_dynamic=True, weights=create_mock_handler(weights), prefix="0"
        ).npu()

        x = torch.randn(2, 32, in_features).to(torch.bfloat16).npu()
        output = linear.forward(x)
        self.stream.synchronize()
        self.assertEqual(output.shape, (2, 32, out_features))
        self.assertIsInstance(output, torch.Tensor)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
class TestWeightQuantLinearBFloat16(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        self.in_features = 128
        self.out_features = 64
        self.weights = {
            "0.weight_scale": torch.ones(self.out_features, dtype=torch.bfloat16),
            "0.weight_offset": torch.ones(self.out_features, dtype=torch.bfloat16),
            "0.weight": torch.ones(self.out_features, self.in_features, dtype=torch.int8),
            "0.bias": torch.ones(self.out_features, dtype=torch.float32),
        }

    def test_init(self):
        # Test initialization of WeightQuantLinear
        linear = WeightQuantLinear(
            self.in_features, self.out_features, bias=True, weights=create_mock_handler(self.weights), prefix="0"
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
            self.in_features, self.out_features, bias=True, weights=create_mock_handler(self.weights), prefix="0"
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
            self.in_features, self.out_features, bias=True, weights=create_mock_handler(self.weights), prefix="0"
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


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
class TestWeightQuantLinearFloat(unittest.TestCase):
    def setUp(self):
        self.stream = torch_npu.npu.current_stream()
        self.in_features = 128
        self.out_features = 64
        self.weights = {
            "0.weight_scale": torch.ones(self.out_features, dtype=torch.float16),
            "0.weight_offset": torch.ones(self.out_features, dtype=torch.float16),
            "0.weight": torch.ones(self.out_features, self.in_features, dtype=torch.int8),
            "0.bias": torch.ones(self.out_features, dtype=torch.float16),
        }

    def test_init(self):
        # Test initialization of WeightQuantLinear
        linear = WeightQuantLinear(
            self.in_features,
            self.out_features,
            bias=True,
            weights=create_mock_handler(self.weights),
            prefix="0",
            dtype=torch.float16,
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
            dtype=torch.float16,
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
            dtype=torch.float16,
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
            dtype=torch.float16,
        ).npu()

        x = torch.randn(4, 8, 32, self.in_features).to(torch.float16).npu()
        output = linear(x)
        self.stream.synchronize()

        # Verify output shape and type
        self.assertEqual(output.shape, (4, 8, 32, self.out_features))
        self.assertIsInstance(output, torch.Tensor)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
@unittest.skipIf(not is_a5_device(), "FP8 Quantization layer tests require A5 (950) NPU.")
class TestFP8RotateQuantFA(unittest.TestCase):
    """CPU-compatible tests for FP8RotateQuantFA. All NPU ops are mocked."""

    B = 1
    N = 8
    S = 16
    D = 64

    def setUp(self):
        if not hasattr(torch_npu, 'float8_e4m3fn'):
            torch_npu.float8_e4m3fn = torch.float16

    @staticmethod
    def _make_weights(d, scale=1.0):
        rot = scale * torch.eye(d)
        return create_mock_handler({"attn.q_rot": rot, "attn.k_rot": rot})

    @staticmethod
    def _mock_block_quant(tensor, dst_type=None, row_block_size=128, col_block_size=128):
        return tensor.to(torch.float16), torch.ones(1, dtype=torch.float32)

    @staticmethod
    def _mock_fa(*args, **kwargs):
        q = args[0]
        out_dtype = kwargs.get('out_dtype', torch.float32)
        return (torch.zeros(*q.shape, dtype=out_dtype),)

    def _make_model(self):
        return FP8RotateQuantFA(prefix="attn", weights=self._make_weights(self.D))

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_forward_bnsd_output_shape(self, mock_bq, mock_fa):
        mock_bq.side_effect = self._mock_block_quant
        mock_fa.side_effect = self._mock_fa

        model = self._make_model()
        q = torch.randn(self.B, self.N, self.S, self.D)
        k = torch.randn(self.B, self.N, self.S, self.D)
        v = torch.randn(self.B, self.N, self.S, self.D)

        out = model(q, k, v, layout="BNSD")

        self.assertEqual(out.shape, (self.B, self.N, self.S, self.D))

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_forward_bsnd_output_shape(self, mock_bq, mock_fa):
        mock_bq.side_effect = self._mock_block_quant
        mock_fa.side_effect = self._mock_fa

        model = self._make_model()
        q = torch.randn(self.B, self.S, self.N, self.D)
        k = torch.randn(self.B, self.S, self.N, self.D)
        v = torch.randn(self.B, self.S, self.N, self.D)

        out = model(q, k, v, layout="BSND")

        self.assertEqual(out.shape, (self.B, self.S, self.N, self.D))

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_forward_invalid_layout_raises_value_error(self, mock_bq, mock_fa):
        mock_bq.side_effect = self._mock_block_quant
        mock_fa.side_effect = self._mock_fa

        model = self._make_model()
        q = torch.randn(self.B, self.N, self.S, self.D)
        k = torch.randn(self.B, self.N, self.S, self.D)
        v = torch.randn(self.B, self.N, self.S, self.D)

        with self.assertRaises(ValueError):
            model(q, k, v, layout="NHWC")

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_forward_bnsd_output_trimmed_when_fa_returns_padded(self, mock_bq, mock_fa):
        """Verify the slice x[:, :, :s, :] when FA returns a sequence longer than s."""
        mock_bq.side_effect = self._mock_block_quant

        padded_s = self.S + 16

        def padded_fa(*args, **kwargs):
            q = args[0]
            out_dtype = kwargs.get('out_dtype', torch.float32)
            return (torch.zeros(q.shape[0], q.shape[1], padded_s, q.shape[3], dtype=out_dtype),)

        mock_fa.side_effect = padded_fa

        model = self._make_model()
        q = torch.randn(self.B, self.N, self.S, self.D)
        k = torch.randn(self.B, self.N, self.S, self.D)
        v = torch.randn(self.B, self.N, self.S, self.D)

        out = model(q, k, v, layout="BNSD")

        self.assertEqual(out.shape, (self.B, self.N, self.S, self.D))

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_forward_bsnd_rotation_applied_to_query(self, mock_bq, mock_fa):
        """Verify that q_rot is applied to query in BSND layout before block_quant."""
        captured = []

        def capture_bq(tensor, dst_type=None, row_block_size=128, col_block_size=128):
            captured.append(tensor.clone())
            return tensor.to(torch.float16), torch.ones(1)

        mock_bq.side_effect = capture_bq
        mock_fa.side_effect = self._mock_fa

        model = FP8RotateQuantFA(prefix="attn", weights=self._make_weights(self.D, scale=2.0))
        # BSND: (B, S, N, D)
        q = torch.ones(self.B, self.S, self.N, self.D)
        k = torch.ones(self.B, self.S, self.N, self.D)
        v = torch.ones(self.B, self.S, self.N, self.D)

        model(q, k, v, layout="BSND")

        # After rotation (2*eye) query values double; after transpose and squeeze: (N, S, D)
        self.assertEqual(len(captured), 3)
        expected_q = 2.0 * torch.ones(self.N, self.S, self.D)
        self.assertTrue(torch.allclose(captured[0], expected_q))

    def test_init_without_rotation_weights_does_not_raise(self):
        model = FP8RotateQuantFA(prefix="attn", weights=create_mock_handler({}))
        self.assertIsNone(model.q_rot)
        self.assertIsNone(model.k_rot)
        self.assertEqual(model.mode, FP8FAMode.HIGH_PRECISION)

    def test_default_mode_is_high_precision(self):
        model = self._make_model()
        self.assertEqual(model.mode, FP8FAMode.HIGH_PRECISION)

    def test_invalid_mode_raises_parameters_invalid(self):
        with self.assertRaises(ParametersInvalid):
            FP8RotateQuantFA(prefix="attn", weights=self._make_weights(self.D), mode="low_precision")

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_high_precision_keeps_original_fia_kwargs(self, mock_bq, mock_fa):
        captured_blocks = []

        def capture_bq(tensor, dst_type=None, row_block_size=128, col_block_size=128):
            captured_blocks.append(row_block_size)
            return tensor.to(torch.float16), torch.ones(1)

        mock_bq.side_effect = capture_bq
        mock_fa.side_effect = self._mock_fa

        model = self._make_model()
        q = torch.randn(self.B, self.N, self.S, self.D)
        k = torch.randn(self.B, self.N, self.S, self.D)
        v = torch.randn(self.B, self.N, self.S, self.D)
        model(q, k, v, layout="BNSD")

        self.assertEqual(captured_blocks, [128, 256, 256])
        fa_kwargs = mock_fa.call_args.kwargs
        self.assertEqual(fa_kwargs["query_quant_mode"], 7)
        self.assertEqual(fa_kwargs["key_quant_mode"], 7)
        self.assertEqual(fa_kwargs["value_quant_mode"], 7)
        self.assertEqual(fa_kwargs["num_query_heads"], self.N)
        self.assertEqual(fa_kwargs["num_key_value_heads"], self.N)
        self.assertNotIn("inner_precise", fa_kwargs)

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_c8v16_tiling512_aligns_quant_and_fia_kwargs(self, mock_bq, mock_fa):
        captured_blocks = []

        def capture_bq(tensor, dst_type=None, row_block_size=128, col_block_size=128):
            captured_blocks.append((row_block_size, col_block_size))
            return tensor.to(torch.float16), torch.ones(1)

        mock_bq.side_effect = capture_bq
        mock_fa.side_effect = self._mock_fa

        model = FP8RotateQuantFA(
            prefix="attn",
            weights=self._make_weights(self.D),
            mode=FP8FAMode.C8V16_TILING512,
        )
        q = torch.randn(self.B, self.N, self.S, self.D)
        k = torch.randn(self.B, self.N, self.S, self.D)
        v = torch.randn(self.B, self.N, self.S, self.D)
        model(q, k, v, layout="BNSD")

        self.assertEqual(captured_blocks, [(128, 128), (256, 128), (512, 64)])
        fa_kwargs = mock_fa.call_args.kwargs
        self.assertEqual(fa_kwargs["query_quant_mode"], 7)
        self.assertEqual(fa_kwargs["key_quant_mode"], 7)
        self.assertEqual(fa_kwargs["value_quant_mode"], 12)
        self.assertEqual(fa_kwargs["inner_precise"], 4)
        self.assertEqual(fa_kwargs["num_query_heads"], self.N)
        self.assertEqual(fa_kwargs["num_key_value_heads"], self.N)

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_gqa_passes_key_head_count_to_fia(self, mock_bq, mock_fa):
        mock_bq.side_effect = self._mock_block_quant
        mock_fa.side_effect = self._mock_fa

        kv_heads = 2
        model = self._make_model()
        q = torch.randn(self.B, self.N, self.S, self.D)
        k = torch.randn(self.B, kv_heads, self.S, self.D)
        v = torch.randn(self.B, kv_heads, self.S, self.D)
        model(q, k, v, layout="BNSD")

        fa_kwargs = mock_fa.call_args.kwargs
        self.assertEqual(fa_kwargs["num_query_heads"], self.N)
        self.assertEqual(fa_kwargs["num_key_value_heads"], kv_heads)

    @patch('torch.ops.mindiesd.fused_infer_attention_score_v2', create=True)
    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_forward_without_rotation_skips_matmul(self, mock_bq, mock_fa):
        captured = []

        def capture_bq(tensor, dst_type=None, row_block_size=128, col_block_size=128):
            captured.append(tensor.clone())
            return tensor.to(torch.float16), torch.ones(1)

        mock_bq.side_effect = capture_bq
        mock_fa.side_effect = self._mock_fa

        model = FP8RotateQuantFA(prefix="attn", weights=create_mock_handler({}))
        q = torch.ones(self.B, self.N, self.S, self.D)
        k = torch.ones(self.B, self.N, self.S, self.D)
        v = torch.ones(self.B, self.N, self.S, self.D)
        model(q, k, v, layout="BNSD")

        self.assertEqual(len(captured), 3)
        self.assertTrue(torch.allclose(captured[0], torch.ones(self.N, self.S, self.D)))


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
@unittest.skipIf(not is_a5_device(), "FP8 Quantization layer tests require A5 (950) NPU.")
class TestFaBlockQuantPreprocess(unittest.TestCase):
    """CPU-compatible tests for fa_block_quant_preprocess. All NPU ops are mocked."""

    B = 1
    N = 8
    S = 16
    D = 64

    def setUp(self):
        if not hasattr(torch_npu, 'float8_e4m3fn'):
            torch_npu.float8_e4m3fn = torch.float16

    @staticmethod
    def _mock_block_quant(tensor, dst_type=None, row_block_size=128, col_block_size=128):
        return tensor.to(torch.float16), torch.ones(1, dtype=torch.float32)

    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_bsnd_layout_calls_block_quant_with_transposed_shape(self, mock_bq):
        """Verify BSND input is transposed to BNSD before npu_dynamic_block_quant."""
        captured_shapes = []

        def capture(tensor, dst_type=None, row_block_size=128, col_block_size=128):
            captured_shapes.append(tensor.shape)
            return tensor.to(torch.float16), torch.ones(1)

        mock_bq.side_effect = capture
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

        x = torch.randn(self.B, self.S, self.N, self.D)
        fa_block_quant_preprocess(x, block_size=128, layout="BSND")

        # transpose(1,2) + squeeze(0) → (N, S, D)
        self.assertEqual(captured_shapes[0], torch.Size([self.N, self.S, self.D]))

    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_bnsd_layout_calls_block_quant_with_correct_shape(self, mock_bq):
        """Verify BNSD input is squeezed to (N, S, D) before npu_dynamic_block_quant."""
        captured_shapes = []

        def capture(tensor, dst_type=None, row_block_size=128, col_block_size=128):
            captured_shapes.append(tensor.shape)
            return tensor.to(torch.float16), torch.ones(1)

        mock_bq.side_effect = capture
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

        x = torch.randn(self.B, self.N, self.S, self.D)
        fa_block_quant_preprocess(x, block_size=128, layout="BNSD")

        # squeeze(0) → (N, S, D)
        self.assertEqual(captured_shapes[0], torch.Size([self.N, self.S, self.D]))

    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_invalid_dim_raises_parameters_invalid(self, mock_bq):
        mock_bq.side_effect = self._mock_block_quant
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess
        from mindiesd.utils import ParametersInvalid

        x = torch.randn(self.N, self.S, self.D)  # 3D, not 4D
        with self.assertRaises(ParametersInvalid):
            fa_block_quant_preprocess(x)

    @patch('torch_npu.npu_dynamic_block_quant', create=True)
    def test_block_size_forwarded_as_row_block_size(self, mock_bq):
        """Verify block_size is forwarded as row_block_size to npu_dynamic_block_quant."""
        captured_kwargs = {}

        def capture(tensor, dst_type=None, row_block_size=128, col_block_size=128):
            captured_kwargs['row_block_size'] = row_block_size
            return tensor.to(torch.float16), torch.ones(1)

        mock_bq.side_effect = capture
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

        x = torch.randn(self.B, self.N, self.S, self.D)
        fa_block_quant_preprocess(x, block_size=256, layout="BNSD")

        self.assertEqual(captured_kwargs['row_block_size'], 256)


class TestQuantAttentionMxfp8(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(1, 2, 17, 64)

    def test_unequal_lengths_and_heads_preserve_tnd_inputs_and_query_output(self):
        cases = product(("BNSD", "BSND"), (1, 2), ((17, 8, 2, 2), (8, 17, 4, 2), (128, 256, 4, 2)))
        for layout, batch, (q_len, kv_len, q_heads, kv_heads) in cases:
            q = torch.randn(batch, q_heads, q_len, 64)
            k = torch.randn(batch, kv_heads, kv_len, 64) + 10
            v = torch.randn_like(k) + 20
            packed_q = (2 * q).transpose(1, 2).reshape(batch * q_len, q_heads, 64)
            packed_k = k.transpose(1, 2).reshape(batch * kv_len, kv_heads, 64)
            packed_v = v.transpose(1, 2).reshape(batch * kv_len, kv_heads, 64)
            if layout == "BSND":
                q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
            with (
                self.subTest(layout=layout, batch=batch, lengths=(q_len, kv_len), heads=(q_heads, kv_heads)),
                patch.object(
                    torch_npu, "npu_dynamic_mx_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
                ) as quantize,
                patch.object(
                    torch_npu,
                    "npu_fused_infer_attention_score_v2",
                    create=True,
                    side_effect=lambda t, *args, **kw: (t + 1,),
                ) as execute,
            ):
                output = quant_attention(q, k, v, precision="mxfp8", layout=layout, q_rot=2 * torch.eye(64))
                torch.testing.assert_close(output, 2 * q + 1)
                for call, packed, axis in zip(quantize.call_args_list, (packed_q, packed_k, packed_v), (-1, -1, 0)):
                    torch.testing.assert_close(call.args[0], packed)
                    self.assertEqual(call.kwargs["axis"], axis)
                for actual, expected in zip(execute.call_args.args, (packed_q, packed_k, packed_v)):
                    torch.testing.assert_close(actual, expected)
                options = execute.call_args.kwargs
                self.assertEqual(options["input_layout"], "TND")
                self.assertEqual(options["num_query_heads"], q_heads)
                self.assertEqual(options["num_key_value_heads"], kv_heads)
                self.assertEqual(options["actual_seq_qlen"], [q_len] if batch == 1 else [q_len, 2 * q_len])
                self.assertEqual(options["actual_seq_kvlen"], [kv_len] if batch == 1 else [kv_len, 2 * kv_len])

    def test_options_are_rejected_before_rotation_or_quantization(self):
        with (
            patch.object(torch, "matmul") as rotate,
            patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quant,
        ):
            for options in ({"fp8_fa_mode": None}, {"softmax_scale": 0.5}, {"mxfp4_scale_alg": 2}, {"unknown": 1}):
                with self.subTest(options=options), self.assertRaisesRegex(TypeError, next(iter(options))):
                    quant_attention(self.q, self.q, self.q, precision="mxfp8", q_rot=torch.eye(64), **options)
            rotate.assert_not_called()
            quant.assert_not_called()

    def test_module_import_without_optional_mx_dtypes(self):
        impl = importlib.import_module("mindiesd.layers.flash_attn.fused_infer_attention_score")
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            spec = importlib.util.spec_from_file_location(impl.__name__ + "_probe", impl.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(callable(module._mxfp8_attention_forward))

    def test_mxfp8_tnd_sequence_metadata(self):
        q = self.q.expand(2, -1, -1, -1)
        with (
            patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quantize,
            patch.object(torch_npu, "npu_fused_infer_attention_score_v2", create=True) as execute,
        ):
            quantize.side_effect = lambda tensor, **kwargs: (tensor, torch.ones(1))
            execute.side_effect = lambda query, *args, **kwargs: (query,)
            out = quant_attention(q, q, q, precision="mxfp8")
            self.assertEqual(out.shape, q.shape)
            self.assertEqual(execute.call_args.kwargs["input_layout"], "TND")
            self.assertEqual(execute.call_args.kwargs["actual_seq_qlen"], [17, 34])
            self.assertEqual(execute.call_args.kwargs["value_quant_mode"], 8)

    def test_layout_rotation_scales_and_dtype_at_operator_boundary(self):
        for layout in ("BNSD", "BSND"):
            q = self.q.expand(2, -1, -1, -1)
            if layout == "BSND":
                q = q.transpose(1, 2)
            scales = [torch.tensor([index]) for index in range(3)]
            with (
                self.subTest(layout=layout),
                patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quantize,
                patch.object(torch_npu, "npu_fused_infer_attention_score_v2", create=True) as execute,
            ):
                quantize.side_effect = lambda t, **kw: (t, scales[quantize.call_count - 1])
                execute.side_effect = lambda t, *args, **kw: (t,)
                output = quant_attention(
                    q,
                    q,
                    q,
                    precision="mxfp8",
                    layout=layout,
                    k_rot=2 * torch.eye(64),
                    scale=0.5,
                    pre_tokens=7,
                    next_tokens=0,
                )
                torch.testing.assert_close(output, q)
                packed = (q.transpose(1, 2) if layout == "BNSD" else q).reshape(34, 2, 64)
                for call, factor, axis in zip(quantize.call_args_list, (1, 2, 1), (-1, -1, 0)):
                    torch.testing.assert_close(call.args[0], factor * packed)
                    self.assertEqual(call.kwargs, {"axis": axis, "dst_type": torch.float8_e4m3fn})
                options = execute.call_args.kwargs
                for tensor, mode, descale in zip(("query", "key", "value"), (6, 6, 8), scales):
                    self.assertEqual(options[tensor + "_quant_mode"], mode)
                    self.assertIs(options["dequant_scale_" + tensor], descale)
                    self.assertEqual(options["dequant_scale_" + tensor + "_dtype"], torch_npu.float8_e8m0fnu)
                    self.assertEqual(options[tensor + "_dtype"], torch.float8_e4m3fn)
                self.assertEqual(options["actual_seq_kvlen"], [17, 34])
                self.assertEqual(options["out_dtype"], q.dtype)
                self.assertEqual((options["softmax_scale"], options["pre_tokens"], options["next_tokens"]), (0.5, 7, 0))

    def test_invalid_rotation_fails_before_quantization(self):
        for options in (
            {"q_rot": torch.ones(64, 32)},
            {"k_rot": torch.eye(64, dtype=torch.float16)},
        ):
            with (
                self.subTest(options=options),
                patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quantize,
            ):
                with self.assertRaises(ValueError):
                    quant_attention(self.q, self.q, self.q, precision="mxfp8", **options)
                quantize.assert_not_called()

    def test_operator_exception_is_not_retried(self):
        with (
            patch.object(
                torch_npu, "npu_dynamic_mx_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
            ),
            patch.object(
                torch_npu, "npu_fused_infer_attention_score_v2", create=True, side_effect=RuntimeError("native failure")
            ) as execute,
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                quant_attention(self.q, self.q, self.q, precision="mxfp8")
            execute.assert_called_once()

    def test_legacy_class_delegates_and_preserves_historical_options(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        weights = SimpleNamespace(keys=lambda: ("a.q_rot", "a.k_rot"), get_tensor=lambda _: torch.eye(64))
        model = layer.MXFP8RotateQuantFA(prefix="a", weights=weights)
        with (
            patch.object(layer, "quant_attention", wraps=quant_attention) as entry,
            (
                patch.object(
                    torch_npu, "npu_dynamic_mx_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
                )
            ),
            patch.object(
                torch_npu, "npu_fused_infer_attention_score_v2", create=True, side_effect=lambda t, *a, **kw: (t,)
            ) as execute,
        ):
            output = model(self.q, self.q, self.q, softmax_scale=0.75, metadata=object(), precision="unused")
            torch.testing.assert_close(output, self.q)
            self.assertEqual(entry.call_args.kwargs["precision"], "mxfp8")
            self.assertIs(entry.call_args.kwargs["q_rot"], model.q_rot)
            self.assertEqual(execute.call_args.kwargs["softmax_scale"], 0.125)


class TestQuantAttentionMxfp8Npu(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "npu"), "Requires TorchNPU on an A5 device.")
    def test_real_npu_unequal_lengths_and_gqa(self):
        if not torch.npu.is_available() or not is_a5_device():
            self.skipTest("Requires an available A5 device.")
        # Constant V per batch/head has a known attention result for any Q/K.
        # Distinct values detect batch/head mixing without a quantization-error baseline.
        head_values = torch.tensor([[1.0, -1.0], [0.5, -0.5]], device="npu", dtype=torch.bfloat16)
        for layout, (q_len, kv_len) in product(("BNSD", "BSND"), ((128, 256), (256, 128))):
            with self.subTest(layout=layout, lengths=(q_len, kv_len)):
                q = torch.randn(2, 4, q_len, 64, device="npu", dtype=torch.bfloat16) * 0.1
                k = torch.randn(2, 2, kv_len, 64, device="npu", dtype=torch.bfloat16) * 0.1
                v = head_values.reshape(2, 2, 1, 1).expand(2, 2, kv_len, 64).contiguous()
                expected = head_values.repeat_interleave(2, dim=1).reshape(2, 4, 1, 1).expand_as(q)
                if layout == "BSND":
                    q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
                    expected = expected.transpose(1, 2)
                output = quant_attention(q, k, v, precision="mxfp8", layout=layout)
                actual = output.cpu()
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected.cpu(), rtol=0.02, atol=0.02)


if __name__ == '__main__':
    unittest.main()
