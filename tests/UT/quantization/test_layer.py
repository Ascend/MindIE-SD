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
# pylint: disable=no-member,redefined-outer-name,too-many-lines
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch_npu

from mindiesd.quantization.layer import (
    W8A8QuantLinear,
    WeightQuantLinear,
    W8A8TimeStepQuantLinear,
    W8A8MXFP8QuantLinear,
    FP8RotateQuantFA,
    W8A8OnlineQuantLinear,
    W8A8MXFP8OnlineQuantLinear,
    W4A4MXFP4FourOverSixQuantLinear,
    W4A4MXFP4OnlineQuantLinear,
    W4A4MXFP4DualOnlineQuantLinear,
    MXFP4QuantFA,
    MXFP4FourOverSixQuantFA,
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


class TestLegacyMXFP4Attention(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(1, 2, 17, 64)

    def test_bsa_scale_helper_import_remains_compatible(self):
        from mindiesd.quantization.layer import _reshape_mxfp4_v_scale_for_fa

        scale = torch.arange(32).reshape(1, 2, 4, 4)
        paired = _reshape_mxfp4_v_scale_for_fa(scale, "BNSD")
        torch.testing.assert_close(paired[..., 0], scale[:, :, 0::2, :])
        torch.testing.assert_close(paired[..., 1], scale[:, :, 1::2, :])
        self.assertIs(_reshape_mxfp4_v_scale_for_fa(paired, "BNSD"), paired)

    def test_legacy_mxfp4_timestep_fallback_filters_kernel_options(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        fp8 = importlib.import_module("mindiesd.layers.flash_attn.fused_infer_attention_score")
        model = layer.MXFP4QuantFA()
        for strategy in ("FP8", "FLOAT"):
            with (
                self.subTest(strategy=strategy),
                patch.object(model.timestep_config, "get_strategy", return_value=strategy),
                patch.object(fp8, "_fa_block_quant_preprocess", side_effect=lambda t, **kw: (t, torch.ones(1))),
                patch.object(
                    torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True, return_value=(self.q,)
                ) as execute,
                patch.object(torch_npu, "npu_fusion_attention", return_value=(self.q,), create=True) as floating,
            ):
                output = model(
                    self.q,
                    self.q,
                    self.q,
                    metadata=object(),
                    layout_out="BSND",
                    mask_mode=0,
                    return_softmax_lse=1,
                    softmax_scale=0.5,
                )
                torch.testing.assert_close(output, self.q)
                if strategy == "FP8":
                    self.assertEqual(execute.call_args.kwargs["softmax_scale"], 0.5)
                    self.assertNotIn("metadata", execute.call_args.kwargs)
                    floating.assert_not_called()
                else:
                    execute.assert_not_called()
                    self.assertEqual(floating.call_args.kwargs["scale"], 0.5)

    def test_legacy_quant_config_is_resolved_before_shared_quantization(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        config = SimpleNamespace(mxfp4_scale_alg=2, mxfp4_dst_type_max=7.25)
        with patch.object(
            torch_npu,
            "npu_dynamic_mx_quant",
            create=True,
            return_value=(self.q, torch.ones(1)),
        ) as quantize:
            layer._dynamic_mx_quant(
                self.q,
                torch_npu.float4_e2m1fn_x2,
                quant_config=config,
                dst_type_max=8.0,
            )
            self.assertEqual(
                quantize.call_args.kwargs,
                {
                    "dst_type": torch_npu.float4_e2m1fn_x2,
                    "scale_alg": 2,
                    "dst_type_max": 8.0,
                },
            )
            layer._dynamic_mx_quant(self.q, torch_npu.float8_e4m3fn, quant_config=config)
            self.assertEqual(quantize.call_args.kwargs, {"dst_type": torch_npu.float8_e4m3fn})

    def test_legacy_classes_use_function_and_timestep_policy(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        model = layer.MXFP4QuantFA(mxfp4_scale_alg=2, mxfp4_dst_type_max=7.25)
        with (
            patch.object(layer, "quant_attention", return_value=self.q) as execute,
            patch.object(torch_npu, "npu_fusion_attention", return_value=(self.q,), create=True) as floating,
        ):
            for strategy in ("FLOAT", "FP8", "MXFP4"):
                execute.reset_mock()
                floating.reset_mock()
                with patch.object(model.timestep_config, "get_strategy", return_value=strategy):
                    self.assertIs(model(self.q, self.q, self.q), self.q)
                    if strategy == "FLOAT":
                        execute.assert_not_called()
                        floating.assert_called_once()
                    else:
                        floating.assert_not_called()
                        self.assertEqual(execute.call_args.kwargs["precision"], strategy.lower())
                        if strategy == "MXFP4":
                            self.assertEqual(execute.call_args.kwargs["mxfp4_dst_type_max"], 7.25)


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


class TestW4A4MXFP4FourOverSixQuantLinear(unittest.TestCase):
    """四六自适应线性层测试：全部 NPU 算子 mock，CPU 可跑。"""

    IN_FEATURES = 64
    OUT_FEATURES = 64

    def _make_weights(self, out_features=None, in_features=None):
        out_features = out_features or self.OUT_FEATURES
        in_features = in_features or self.IN_FEATURES
        return {
            "0.weight": torch.ones(out_features, in_features, dtype=torch.uint8),
            "0.weight_scale": torch.ones(out_features, in_features // 32, dtype=torch.uint8),
            "0.bias": torch.ones(out_features, dtype=torch.float32),
        }

    def _make_layer(self, dtype=torch.float16, **kwargs):
        return W4A4MXFP4FourOverSixQuantLinear(
            self.IN_FEATURES,
            self.OUT_FEATURES,
            bias=True,
            weights=create_mock_handler(self._make_weights()),
            prefix="0",
            dtype=dtype,
            **kwargs,
        )

    @patch('mindiesd.quantization.layer.quant_four_over_six_a5', side_effect=mock_quant_four_over_six_a5)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w4a4_path_uses_four_over_six_op(self, mock_quant_matmul, mock_four_over_six):
        linear = self._make_layer(dtype=torch.float16)
        x = torch.randn(2, self.IN_FEATURES).to(torch.float16)
        output = linear.forward(x)

        self.assertEqual(output.shape, (2, self.OUT_FEATURES))
        self.assertEqual(mock_four_over_six.call_count, 1)
        # 算子仅接受 bf16：fp16 激活须先转换
        quant_input = mock_four_over_six.call_args.args[0]
        self.assertEqual(quant_input.dtype, torch.bfloat16)
        self.assertEqual(quant_input.shape, x.shape)
        self.assertEqual(mock_quant_matmul.call_args.kwargs["x1_dtype"], torch_npu.float4_e2m1fn_x2)

    @patch('mindiesd.quantization.layer.quant_four_over_six_a5', side_effect=mock_quant_four_over_six_a5)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w4a4_path_applies_mul_scale_before_quant(self, mock_quant_matmul, mock_four_over_six):
        mul_scale = torch.full((self.IN_FEATURES,), 2.0, dtype=torch.float16)
        linear = self._make_layer(dtype=torch.float16, mul_scale=mul_scale)
        x = torch.ones(1, self.IN_FEATURES, dtype=torch.float16)
        linear.forward(x)

        quant_input = mock_four_over_six.call_args.args[0]
        self.assertTrue(torch.allclose(quant_input.to(torch.float16), 2.0 * x))

    @patch('mindiesd.quantization.layer.quant_four_over_six_a5', side_effect=mock_quant_four_over_six_a5)
    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant_packed)
    @patch('torch_npu.npu_quant_matmul', side_effect=mock_npu_quant_matmul)
    def test_w4a8_fallback_keeps_dynamic_mx_quant(self, mock_quant_matmul, mock_dynamic_mx_quant, mock_four_over_six):
        timestep_config = TimestepPolicyConfig()
        timestep_config.register([5], "W4A8", target="w4a4_linear")
        linear = self._make_layer(dtype=torch.float16, quant_config=QuantConfig(timestep_config=timestep_config))

        TimestepManager.set_timestep_idx_max(10)
        TimestepManager.set_timestep_idx(5)
        x = torch.randn(2, self.IN_FEATURES).to(torch.float16)
        output = linear.forward(x)

        self.assertEqual(output.shape, (2, self.OUT_FEATURES))
        mock_four_over_six.assert_not_called()
        self.assertEqual(mock_dynamic_mx_quant.call_count, 1)
        self.assertNotIn("x1_dtype", mock_quant_matmul.call_args.kwargs)

    def test_rejects_unaligned_in_features(self):
        with self.assertRaises(ParametersInvalid):
            W4A4MXFP4FourOverSixQuantLinear(
                48, 48, bias=False, weights=create_mock_handler(self._make_weights(48, 48)), prefix="0"
            )


class TestMXFP4FourOverSixQuantFA(unittest.TestCase):
    """四六自适应 FA 测试：全部 NPU 算子/自定义算子 mock，CPU 可跑。"""

    B = 1
    N = 2
    S = 512
    D = 128

    def _make_fa(self, fa_cls=MXFP4FourOverSixQuantFA):
        return fa_cls(prefix="attn", weights=create_mock_handler({}))

    def _run_forward(self, fa, layout):
        if layout == "BSND":
            q = torch.randn(self.B, self.S, self.N, self.D)
            k = torch.randn(self.B, self.S, self.N, self.D)
            v = torch.randn(self.B, self.S, self.N, self.D)
        else:
            q = torch.randn(self.B, self.N, self.S, self.D)
            k = torch.randn(self.B, self.N, self.S, self.D)
            v = torch.randn(self.B, self.N, self.S, self.D)
        return q, k, v, fa(q, k, v, layout=layout)

    @patch.object(
        torch.ops.mindiesd, "quant_flash_attn_metadata", create=True, side_effect=mock_quant_flash_attn_metadata
    )
    @patch.object(torch.ops.mindiesd, "quant_flash_attn", create=True, side_effect=mock_quant_flash_attn)
    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant_packed)
    @patch('mindiesd.quantization.layer.quant_four_over_six_a5', side_effect=mock_quant_four_over_six_a5)
    def test_bnsd_qk_uses_four_over_six_and_v_uses_seq_axis(
        self, mock_four_over_six, mock_dynamic_mx_quant, _mock_attn, _mock_metadata
    ):
        fa = self._make_fa()
        q, k, v, output = self._run_forward(fa, "BNSD")

        self.assertEqual(output.shape, (self.B, self.N, self.S, self.D))
        # Q/K 走 46 自适应算子（尾轴），输入为 bf16 原形状
        self.assertEqual(mock_four_over_six.call_count, 2)
        for call in mock_four_over_six.call_args_list:
            self.assertEqual(call.args[0].dtype, torch.bfloat16)
            self.assertEqual(call.args[0].shape, q.shape)
        # V 沿 seq 轴（BNSD axis=2）保留 npu_dynamic_mx_quant
        self.assertEqual(mock_dynamic_mx_quant.call_count, 1)
        self.assertEqual(mock_dynamic_mx_quant.call_args.kwargs["axis"], 2)

    @patch.object(
        torch.ops.mindiesd, "quant_flash_attn_metadata", create=True, side_effect=mock_quant_flash_attn_metadata
    )
    @patch.object(torch.ops.mindiesd, "quant_flash_attn", create=True, side_effect=mock_quant_flash_attn)
    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant_packed)
    @patch('mindiesd.quantization.layer.quant_four_over_six_a5', side_effect=mock_quant_four_over_six_a5)
    def test_bsnd_layout_v_axis(self, mock_four_over_six, mock_dynamic_mx_quant, _mock_attn, _mock_metadata):
        fa = self._make_fa()
        q, k, v, output = self._run_forward(fa, "BSND")

        self.assertEqual(output.shape, (self.B, self.S, self.N, self.D))
        self.assertEqual(mock_four_over_six.call_count, 2)
        self.assertEqual(mock_dynamic_mx_quant.call_count, 1)
        self.assertEqual(mock_dynamic_mx_quant.call_args.kwargs["axis"], 1)

    @patch.object(
        torch.ops.mindiesd, "quant_flash_attn_metadata", create=True, side_effect=mock_quant_flash_attn_metadata
    )
    @patch.object(torch.ops.mindiesd, "quant_flash_attn", create=True, side_effect=mock_quant_flash_attn)
    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant_packed)
    @patch('mindiesd.quantization.layer.quant_four_over_six_a5', side_effect=mock_quant_four_over_six_a5)
    def test_unaligned_head_dim_raises(self, mock_four_over_six, mock_dynamic_mx_quant, _mock_attn, _mock_metadata):
        self.D = 100
        fa = self._make_fa()
        try:
            with self.assertRaises(ParametersInvalid):
                self._run_forward(fa, "BNSD")
        finally:
            self.D = 128

    @patch.object(
        torch.ops.mindiesd, "quant_flash_attn_metadata", create=True, side_effect=mock_quant_flash_attn_metadata
    )
    @patch.object(torch.ops.mindiesd, "quant_flash_attn", create=True, side_effect=mock_quant_flash_attn)
    @patch('torch_npu.npu_dynamic_mx_quant', side_effect=mock_npu_dynamic_mx_quant_packed)
    @patch('mindiesd.quantization.layer.quant_four_over_six_a5', side_effect=mock_quant_four_over_six_a5)
    def test_base_mxfp4_fa_still_uses_dynamic_mx_for_qkv(
        self, mock_four_over_six, mock_dynamic_mx_quant, _mock_attn, _mock_metadata
    ):
        """钩子重构回归：基类 MXFP4QuantFA 的 Q/K/V 均仍走 npu_dynamic_mx_quant。"""
        fa = self._make_fa(MXFP4QuantFA)
        q, k, v, output = self._run_forward(fa, "BNSD")

        self.assertEqual(output.shape, (self.B, self.N, self.S, self.D))
        mock_four_over_six.assert_not_called()
        self.assertEqual(mock_dynamic_mx_quant.call_count, 3)


if __name__ == '__main__':
    unittest.main()
