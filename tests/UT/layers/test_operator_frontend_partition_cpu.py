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

import unittest
from unittest.mock import patch
import pytest
import torch
from mindiesd.layers.activation import GELU
from mindiesd.layers.residual_gate_add import _normalize_triton_inputs
from mindiesd.layers.norm import fast_layernorm
from mindiesd.layers.moe.moe_dataclass import MoEMlpComputeInput, MoEWeights
from mindiesd.layers.moe.moe_mlp import w8a8_dynamic_apply_mlp
from mindiesd.utils import ParametersInvalid


class TestCpuMlpFormatRouting(unittest.TestCase):
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

    def test_w8a8_dynamic_apply_mlp_casts_non_nz_weights_to_nz(self):
        device = torch.device("cpu")
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

    def test_w8a8_dynamic_apply_mlp_keeps_existing_nz_weights(self):
        device = torch.device("cpu")
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


def test_gelu_invalid_approximation_is_python_validation():
    with pytest.raises(ParametersInvalid):
        GELU(approximate="test")(torch.randn(1, 2, 3))


def test_residual_full_sized_gate_is_python_validation():
    x = torch.randn(2, 16, 64)
    with pytest.raises(ValueError, match="gate must have shape"):
        _normalize_triton_inputs(x, torch.randn_like(x), torch.randn_like(x))


def test_layernorm_impl_mode_python_validation():
    layer = torch.nn.LayerNorm(8)
    x = torch.randn(2, 8)
    with pytest.raises(ParametersInvalid):
        fast_layernorm(layer, x, 5)
    with pytest.raises(ParametersInvalid):
        fast_layernorm(layer, x.bfloat16(), 2)
