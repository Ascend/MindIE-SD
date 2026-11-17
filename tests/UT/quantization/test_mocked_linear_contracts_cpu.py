# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# Wrapper-contract tests migrated from the device suite: all tested native ops
# are mocked; CPU tensors execute the remaining real Python logic.


import unittest


from types import SimpleNamespace


from unittest.mock import patch


import torch


from torch import nn


import torch_npu


from mindiesd.quantization.layer import (
    WeightQuantLinear,
    W8A8MXFP8QuantLinear,
    W8A8OnlineQuantLinear,
    W8A8MXFP8OnlineQuantLinear,
    W4A4MXFP4OnlineQuantLinear,
    W4A4MXFP4DualOnlineQuantLinear,
)


from mindiesd.quantization.config import QuantConfig, TimestepPolicyConfig


from mindiesd.quantization.utils import TimestepManager


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


def mock_quant_four_over_six_a5(x, *args, **kwargs):
    """按真实算子语义返回：y uint8 打包 fp4（尾维减半），mxscale uint8（尾维 [dim/64, 2]）。"""
    y = torch.zeros(*x.shape[:-1], x.shape[-1] // 2, dtype=torch.uint8)
    scale_shape = list(x.shape)
    scale_shape[-1] = ((x.shape[-1] + 31) // 32 + 1) // 2
    scale_shape.append(2)
    mxscale = torch.zeros(*scale_shape, dtype=torch.uint8)
    return y, mxscale


def mock_npu_dynamic_mx_quant_packed(x, *args, axis=-1, **kwargs):
    """按 npu_dynamic_mx_quant(fp4) 语义返回：y 尾维减半，scale axis 维 [dim/64, 2] 并 append 2。"""
    y = torch.zeros(*x.shape[:-1], x.shape[-1] // 2, dtype=torch.uint8)
    scale_shape = list(x.shape)
    dim = axis if axis >= 0 else axis + x.dim()
    scale_shape[dim] = ((x.shape[dim] + 31) // 32 + 1) // 2
    scale_shape.append(2)
    scale = torch.zeros(*scale_shape, dtype=torch.uint8)
    return y, scale


def mock_quant_flash_attn_metadata(**kwargs):
    return object()


def mock_quant_flash_attn(q, k, v, *args, **kwargs):
    """按 quant_flash_attn 输出语义返回：out 形状随 layout_out，seq 维取 seqused_q 的 pad 长度。"""
    layout_out = kwargs.get("layout_out", "BNSD")
    seqused_q = kwargs.get("seqused_q")
    out_s = int(seqused_q.max()) if seqused_q is not None else q.shape[2]
    d = q.shape[-1] * 2  # q 为 uint8 打包（每字节 2 个 fp4）
    if layout_out == "BSND":
        output = torch.zeros(q.shape[0], out_s, q.shape[2], d)
    else:
        output = torch.zeros(q.shape[0], q.shape[1], out_s, d)
    return output, torch.zeros(0)


class CPUDeviceBoundary(unittest.TestCase):
    def setUp(self):
        super().setUp()
        for owner in (torch.Tensor, torch.nn.Module):
            patcher = patch.object(owner, "npu", lambda value, *args, **kwargs: value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)


class TestQuantLinearFloat16CPU(CPUDeviceBoundary):
    def _patch_torch_npu_attr(self, name, value):
        patcher = patch.object(torch_npu, name, value, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def setUp(self):
        super().setUp()
        self.stream = SimpleNamespace(synchronize=lambda: None)
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


class TestWeightQuantLinearBFloat16CPU(CPUDeviceBoundary):
    def setUp(self):
        super().setUp()
        self.stream = SimpleNamespace(synchronize=lambda: None)
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


class TestWeightQuantLinearFloatCPU(CPUDeviceBoundary):
    def setUp(self):
        super().setUp()
        self.stream = SimpleNamespace(synchronize=lambda: None)
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
