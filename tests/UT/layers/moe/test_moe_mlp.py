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

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torch_npu

from mindiesd.layers.moe.moe_dataclass import (
    MoEMlpComputeInput,
    MoEWeights,
)
from mindiesd.layers.moe.moe_mlp import (
    unquant_apply_mlp,
    unified_apply_mlp,
    w8a8_dynamic_apply_mlp,
    w8a8_mxfp8_apply_mlp,
)
from mindiesd.layers.moe.moe_context import set_moe_context
from mindiesd.layers.moe.token_dispatcher import StaticDispatcher
from mindiesd.quantization.mode import QuantAlgorithm
from .common import a2_a3_test, a5_test, cpu_test, make_mxfp8_ones, make_static_metadata, npu_test


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


@cpu_test
class TestMoEMlpHelpers(unittest.TestCase):
    def test_w8a8_dynamic_without_static_metadata_returns_unfinalized_output(self):
        expected = torch.randn(3, 4)
        mlp_input = MoEMlpComputeInput(
            hidden_states=torch.randint(-8, 8, (3, 4), dtype=torch.int8),
            group_list=torch.tensor([2, 1]),
            group_list_type=1,
            weights=MoEWeights(
                w13_weight=torch.randint(-8, 8, (2, 4, 16), dtype=torch.int8),
                w2_weight=torch.randint(-8, 8, (2, 8, 4), dtype=torch.int8),
                w13_weight_scale=torch.randn(2, 16),
                w2_weight_scale=torch.randn(2, 4),
            ),
            mlp_output_dtype=torch.float32,
            dynamic_scale=torch.ones(3, 1),
        )

        with (
            patch("torch_npu.get_npu_format", return_value=29),
            patch(
                "torch_npu.npu_grouped_matmul_swiglu_quant",
                return_value=(torch.randint(-8, 8, (3, 8), dtype=torch.int8), torch.ones(3, 1), None),
            ),
            patch("torch_npu.npu_grouped_matmul", return_value=[expected]) as grouped_matmul,
            patch("torch_npu.npu_grouped_matmul_finalize_routing") as finalize_routing,
        ):
            actual = w8a8_dynamic_apply_mlp(mlp_input)

        self.assertIs(actual, expected)
        grouped_matmul.assert_called_once()
        finalize_routing.assert_not_called()

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
            patch.object(torch_npu, "float8_e8m0fnu", torch.uint8, create=True),
            patch("torch_npu.get_npu_format", return_value=29),
            patch("torch_npu.npu_dynamic_mx_quant", create=True) as dynamic_mx_quant,
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
        apply_mlp.assert_called_once_with(mlp_input, None)


if __name__ == "__main__":
    unittest.main()
