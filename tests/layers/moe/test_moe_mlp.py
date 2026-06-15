#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torch_npu

from mindiesd.layers.moe.moe_dataclass import MoEMlpComputeInput, MoEWeights
from mindiesd.layers.moe.moe_mlp import (
    unquant_apply_mlp,
    unified_apply_mlp,
    w8a8_dynamic_apply_mlp,
    w8a8_mxfp8_apply_mlp,
)
from mindiesd.layers.moe.moe_context import set_moe_context
from mindiesd.quantization.mode import QuantAlgorithm
from mindiesd.utils.get_platform import NPUDevice, get_npu_device

from .common import make_mxfp8_ones


def torch_mlp_reference(hidden_states, w13_weight, w2_weight, group_list, w13_bias=None, w2_bias=None):
    outputs = []
    start = 0
    for expert_id, end in enumerate(group_list.tolist()):
        if end <= start:
            continue
        gate_up = hidden_states[start:end] @ w13_weight[expert_id]
        if w13_bias is not None:
            gate_up = gate_up + w13_bias[expert_id]
        gate, up = gate_up.chunk(2, dim=-1)
        output = (F.silu(gate) * up) @ w2_weight[expert_id]
        if w2_bias is not None:
            output = output + w2_bias[expert_id]
        outputs.append(output)
        start = end
    return torch.cat(outputs, dim=0)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU",
    "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.",
)
class TestMoEMlpHelpers(unittest.TestCase):
    def setUp(self):
        set_moe_context()

    def tearDown(self):
        set_moe_context()

    def test_unquant_apply_mlp_selects_bias_dtype_by_input_dtype(self):
        cases = (
            dict(input_dtype=torch.bfloat16, bias_dtype=torch.float32),
            dict(input_dtype=torch.float16, bias_dtype=torch.float16),
        )
        for case in cases:
            with self.subTest(**case):
                input_dtype = case["input_dtype"]
                bias_dtype = case["bias_dtype"]
                mlp_input = MoEMlpComputeInput(
                    hidden_states=torch.randn(3, 4, dtype=input_dtype),
                    group_list=torch.tensor([2, 3]),
                    group_list_type=1,
                    weights=MoEWeights(
                        w13_weight=torch.randn(2, 4, 16, dtype=input_dtype),
                        w2_weight=torch.randn(2, 8, 4, dtype=input_dtype),
                        w13_bias=torch.randn(2, 16, dtype=input_dtype),
                        w2_bias=torch.randn(2, 4, dtype=input_dtype),
                    ),
                    mlp_output_dtype=input_dtype,
                )

                grouped_matmul_outputs = ([torch.randn(3, 16)], [torch.randn(3, 4)])
                with (
                    patch("torch_npu.npu_grouped_matmul", side_effect=grouped_matmul_outputs) as grouped_matmul,
                    patch(
                        "torch_npu.npu_swiglu",
                        return_value=torch.randn(3, 8),
                    ),
                ):
                    unquant_apply_mlp(mlp_input)

                first_bias = grouped_matmul.call_args_list[0].kwargs["bias"][0]
                second_bias = grouped_matmul.call_args_list[1].kwargs["bias"][0]
                self.assertEqual(first_bias.dtype, bias_dtype)
                self.assertEqual(second_bias.dtype, bias_dtype)

    def test_w8a8_dynamic_apply_mlp_uses_dispatch_quant_output(self):
        quant_hidden = torch.randint(-8, 8, (3, 4), dtype=torch.int8)
        per_token_scale = torch.randn(3, 1)
        swiglu_out = torch.randint(-8, 8, (3, 8), dtype=torch.int8)
        swiglu_out_scale = torch.randn(3, 1)
        expected = torch.randn(3, 4)
        mlp_input = MoEMlpComputeInput(
            hidden_states=quant_hidden,
            group_list=torch.tensor([2, 3]),
            group_list_type=1,
            weights=MoEWeights(
                w13_weight=torch.randint(-8, 8, (2, 4, 16), dtype=torch.int8),
                w2_weight=torch.randint(-8, 8, (2, 8, 4), dtype=torch.int8),
                w13_weight_scale=torch.randn(2, 16),
                w2_weight_scale=torch.randn(2, 4),
            ),
            mlp_output_dtype=torch.float32,
            dynamic_scale=per_token_scale,
        )

        with (
            patch("torch_npu.get_npu_format", return_value=29),
            patch("torch_npu.npu_dynamic_quant") as dynamic_quant,
            patch(
                "torch_npu.npu_grouped_matmul_swiglu_quant",
                return_value=(swiglu_out, swiglu_out_scale, None),
            ) as swiglu_quant,
            patch("torch_npu.npu_dequant_swiglu_quant") as dequant_swiglu,
            patch(
                "torch_npu.npu_grouped_matmul",
                return_value=[expected],
            ) as grouped_matmul,
        ):
            actual = w8a8_dynamic_apply_mlp(mlp_input)

        self.assertIs(actual, expected)
        dynamic_quant.assert_not_called()
        swiglu_quant.assert_called_once()
        self.assertIs(swiglu_quant.call_args.kwargs["x"], quant_hidden)
        self.assertIs(swiglu_quant.call_args.kwargs["x_scale"], per_token_scale)
        self.assertTrue(torch.equal(swiglu_quant.call_args.kwargs["group_list"], torch.tensor([2, 5])))
        self.assertIs(swiglu_quant.call_args.kwargs["weight_scale"], mlp_input.weights.w13_weight_scale)
        dequant_swiglu.assert_not_called()
        grouped_matmul.assert_called_once()
        self.assertEqual(grouped_matmul.call_args.kwargs["output_dtype"], torch.float32)

    def test_w8a8_mxfp8_apply_mlp_uses_dispatch_quant_output(self):
        quant_hidden = torch.empty(3, 4, dtype=torch.float8_e4m3fn)
        per_token_scale = torch.empty(3, 2, dtype=torch.uint8)
        swiglu_out = torch.empty(3, 8, dtype=torch.float8_e4m3fn)
        swiglu_out_scale = torch.empty(3, 1, 2, dtype=torch.uint8)
        expected = torch.randn(3, 4)
        mlp_input = MoEMlpComputeInput(
            hidden_states=quant_hidden,
            group_list=torch.tensor([2, 3]),
            group_list_type=1,
            weights=MoEWeights(
                w13_weight=torch.empty(2, 4, 16, dtype=torch.float8_e4m3fn),
                w2_weight=torch.empty(2, 8, 4, dtype=torch.float8_e4m3fn),
                w13_weight_scale=torch.empty(2, 16, dtype=torch.uint8),
                w2_weight_scale=torch.empty(2, 4, dtype=torch.uint8),
            ),
            mlp_output_dtype=torch.bfloat16,
            dynamic_scale=per_token_scale,
        )

        with (
            patch("torch_npu.npu_dynamic_mx_quant") as dynamic_mx_quant,
            patch(
                "torch_npu.npu_grouped_matmul_swiglu_quant_v2",
                return_value=(swiglu_out, swiglu_out_scale),
            ) as swiglu_quant,
            patch("torch_npu.npu_grouped_matmul", return_value=[expected]) as grouped_matmul,
        ):
            actual = w8a8_mxfp8_apply_mlp(mlp_input)

        self.assertIs(actual, expected)
        dynamic_mx_quant.assert_not_called()
        swiglu_quant.assert_called_once()
        self.assertIs(swiglu_quant.call_args.kwargs["x"], quant_hidden)
        self.assertEqual(swiglu_quant.call_args.kwargs["x_scale"].shape, torch.Size([3, 1, 2]))
        self.assertEqual(swiglu_quant.call_args.kwargs["quant_mode"], 2)
        self.assertEqual(swiglu_quant.call_args.kwargs["quant_dtype"], torch.float8_e4m3fn)
        self.assertIsNone(swiglu_quant.call_args.kwargs["x_dtype"])
        self.assertIsNone(swiglu_quant.call_args.kwargs["weight_dtype"])
        grouped_matmul.assert_called_once()
        self.assertEqual(grouped_matmul.call_args.kwargs["output_dtype"], torch.bfloat16)
        self.assertIsNone(grouped_matmul.call_args.kwargs["x_dtype"])
        self.assertIsNone(grouped_matmul.call_args.kwargs["weight_dtype"])
        self.assertIs(grouped_matmul.call_args.kwargs["per_token_scale"][0], swiglu_out_scale)

    def test_unified_apply_mlp_dispatches_w8a8_mxfp8(self):
        mlp_input = MoEMlpComputeInput(
            hidden_states=torch.empty(3, 4, dtype=torch.float8_e4m3fn),
            group_list=torch.tensor([2, 3]),
            group_list_type=1,
            weights=MoEWeights(
                w13_weight=torch.empty(2, 4, 16, dtype=torch.float8_e4m3fn),
                w2_weight=torch.empty(2, 8, 4, dtype=torch.float8_e4m3fn),
                w13_weight_scale=torch.empty(2, 16, dtype=torch.uint8),
                w2_weight_scale=torch.empty(2, 4, dtype=torch.uint8),
            ),
            mlp_output_dtype=torch.bfloat16,
            dynamic_scale=torch.empty(3, 2, dtype=torch.uint8),
        )
        expected = torch.randn(3, 4)

        set_moe_context(quant_algo=QuantAlgorithm.W8A8_MXFP8)
        with patch("mindiesd.layers.moe.moe_mlp.w8a8_mxfp8_apply_mlp", return_value=expected) as apply_mlp:
            actual = unified_apply_mlp(mlp_input)

        self.assertIs(actual, expected)
        apply_mlp.assert_called_once_with(mlp_input)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestMoEMlp(unittest.TestCase):
    def _make_w8a8_dynamic_mlp_input(self, device):
        return MoEMlpComputeInput(
            hidden_states=torch.randint(-8, 8, (3, 4), dtype=torch.int8, device=device),
            group_list=torch.tensor([2, 3], device=device),
            group_list_type=1,
            weights=MoEWeights(
                w13_weight=torch.randint(-8, 8, (2, 4, 16), dtype=torch.int8, device=device),
                w2_weight=torch.randint(-8, 8, (2, 8, 4), dtype=torch.int8, device=device),
                w13_weight_scale=torch.randn(2, 16, device=device),
                w2_weight_scale=torch.randn(2, 4, device=device),
            ),
            mlp_output_dtype=torch.bfloat16,
            dynamic_scale=torch.randn(3, 1, device=device),
        )

    @unittest.skipIf(
        get_npu_device() not in (NPUDevice.A2, NPUDevice.A3),
        "Skip INT8 MoE tests when device is not A2 or A3.",
    )
    def test_w8a8_dynamic_apply_mlp_casts_non_nz_weights_to_nz(self):
        device = torch.device("npu")
        mlp_input = self._make_w8a8_dynamic_mlp_input(device)
        w13_weight_nz = torch.empty_like(mlp_input.weights.w13_weight)
        w2_weight_nz = torch.empty_like(mlp_input.weights.w2_weight)

        with (
            patch("torch_npu.get_npu_format", return_value=0),
            patch("torch_npu.npu_format_cast", side_effect=(w13_weight_nz, w2_weight_nz)) as format_cast,
            patch(
                "torch_npu.npu_grouped_matmul_swiglu_quant",
                return_value=(torch.randn(3, 8, device=device), torch.randn(3, 1, device=device), None),
            ),
            patch("torch_npu.npu_grouped_matmul", return_value=[torch.randn(3, 4, device=device)]),
        ):
            w8a8_dynamic_apply_mlp(mlp_input)

        self.assertEqual(format_cast.call_count, 2)
        self.assertIs(format_cast.call_args_list[0].args[0], mlp_input.weights.w13_weight)
        self.assertEqual(format_cast.call_args_list[0].args[1], 29)
        self.assertIs(format_cast.call_args_list[1].args[0], mlp_input.weights.w2_weight)
        self.assertEqual(format_cast.call_args_list[1].args[1], 29)

    @unittest.skipIf(
        get_npu_device() not in (NPUDevice.A2, NPUDevice.A3),
        "Skip INT8 MoE tests when device is not A2 or A3.",
    )
    def test_w8a8_dynamic_apply_mlp_keeps_existing_nz_weights(self):
        device = torch.device("npu")
        mlp_input = self._make_w8a8_dynamic_mlp_input(device)

        with (
            patch("torch_npu.get_npu_format", return_value=29),
            patch("torch_npu.npu_format_cast") as format_cast,
            patch(
                "torch_npu.npu_grouped_matmul_swiglu_quant",
                return_value=(torch.randn(3, 8, device=device), torch.randn(3, 1, device=device), None),
            ),
            patch("torch_npu.npu_grouped_matmul", return_value=[torch.randn(3, 4, device=device)]),
        ):
            w8a8_dynamic_apply_mlp(mlp_input)

        format_cast.assert_not_called()

    @unittest.skipIf(
        get_npu_device() not in (NPUDevice.A2, NPUDevice.A3),
        "Skip INT8 MoE tests when device is not A2 or A3.",
    )
    def test_w8a8_dynamic_mlp_matches_prequantized_input(self):
        torch.manual_seed(2026)
        device = torch.device("npu")
        hidden_size = 32
        intermediate_size = 32
        cases = (
            dict(dtype=torch.bfloat16, weight_scale_dtype=torch.bfloat16),
            dict(dtype=torch.float16, weight_scale_dtype=torch.float32),
        )
        for case in cases:
            with self.subTest(**case):
                dtype = case["dtype"]
                weight_scale_dtype = case["weight_scale_dtype"]
                hidden_states = (torch.randn(4, hidden_size, device=device, dtype=dtype) / 10).contiguous()
                quant_hidden, per_token_scale = torch_npu.npu_dynamic_quant(hidden_states)
                w13_weight = torch.randint(
                    -8,
                    8,
                    (2, hidden_size, 2 * intermediate_size),
                    dtype=torch.int8,
                    device=device,
                )
                w2_weight = torch.randint(
                    -8,
                    8,
                    (2, intermediate_size, hidden_size),
                    dtype=torch.int8,
                    device=device,
                )
                weights = MoEWeights(
                    w13_weight=torch_npu.npu_format_cast(w13_weight, 29),
                    w2_weight=torch_npu.npu_format_cast(w2_weight, 29),
                    w13_weight_scale=torch.rand(2, 2 * intermediate_size, device=device, dtype=weight_scale_dtype),
                    w2_weight_scale=torch.rand(2, hidden_size, device=device, dtype=weight_scale_dtype),
                )
                group_list = torch.tensor([2, 2], dtype=torch.int64, device=device)

                expected = w8a8_dynamic_apply_mlp(
                    MoEMlpComputeInput(
                        hidden_states=hidden_states,
                        group_list=group_list,
                        group_list_type=1,
                        weights=weights,
                        mlp_output_dtype=hidden_states.dtype,
                    )
                )
                actual = w8a8_dynamic_apply_mlp(
                    MoEMlpComputeInput(
                        hidden_states=quant_hidden,
                        group_list=group_list,
                        group_list_type=1,
                        weights=weights,
                        mlp_output_dtype=hidden_states.dtype,
                        dynamic_scale=per_token_scale,
                    )
                )

                torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), atol=1e-3, rtol=1e-3)

    def test_unquant_apply_mlp_matches_torch_reference_with_bias(self):
        torch.manual_seed(2026)
        device = torch.device("npu")
        cases = (
            dict(dtype=torch.bfloat16),
            dict(dtype=torch.float16),
        )
        for case in cases:
            with self.subTest(**case):
                dtype = case["dtype"]
                hidden_states = torch.randn(3, 4) / 10
                w13_weight = torch.randn(2, 4, 16) / 10
                w2_weight = torch.randn(2, 8, 4) / 10
                w13_bias = torch.randn(2, 16) / 10
                w2_bias = torch.randn(2, 4) / 10
                group_list = torch.tensor([2, 3], dtype=torch.int64)
                expected = torch_mlp_reference(hidden_states, w13_weight, w2_weight, group_list, w13_bias, w2_bias)

                actual = unquant_apply_mlp(
                    MoEMlpComputeInput(
                        hidden_states=hidden_states.to(device=device, dtype=dtype),
                        group_list=group_list.to(device=device),
                        group_list_type=1,
                        weights=MoEWeights(
                            w13_weight=w13_weight.to(device=device, dtype=dtype),
                            w2_weight=w2_weight.to(device=device, dtype=dtype),
                            w13_bias=w13_bias.to(device=device, dtype=dtype),
                            w2_bias=w2_bias.to(device=device, dtype=dtype),
                        ),
                        mlp_output_dtype=dtype,
                    )
                )

                torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=5e-2, rtol=5e-2)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
@unittest.skipIf(get_npu_device() != NPUDevice.A5, "Skip MXFP8 MoE tests when device is not A5.")
class TestMoEMlpA5(unittest.TestCase):
    def test_w8a8_mxfp8_mlp_matches_prequantized_input(self):
        device = torch.device("npu")
        hidden_size = 128
        intermediate_size = 64
        hidden_states = (torch.randn(4, hidden_size, device=device, dtype=torch.bfloat16) / 10).contiguous()
        quant_hidden, per_token_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        w13_weight, w13_weight_scale = make_mxfp8_ones(2, hidden_size, 2 * intermediate_size, device=device)
        w2_weight, w2_weight_scale = make_mxfp8_ones(2, intermediate_size, hidden_size, device=device)
        weights = MoEWeights(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
        )
        group_list = torch.tensor([2, 2], dtype=torch.int64, device=device)

        expected = w8a8_mxfp8_apply_mlp(
            MoEMlpComputeInput(
                hidden_states=hidden_states,
                group_list=group_list,
                group_list_type=1,
                weights=weights,
                mlp_output_dtype=hidden_states.dtype,
            )
        )
        actual = w8a8_mxfp8_apply_mlp(
            MoEMlpComputeInput(
                hidden_states=quant_hidden,
                group_list=group_list,
                group_list_type=1,
                weights=weights,
                mlp_output_dtype=hidden_states.dtype,
                dynamic_scale=per_token_scale,
            )
        )

        torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
