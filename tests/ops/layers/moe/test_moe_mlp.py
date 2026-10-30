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
from tests.ops.layers.moe.common import a2_a3_test, a5_test, cpu_test, make_mxfp8_ones, make_static_metadata, npu_test


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


@npu_test
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

    @a2_a3_test
    def test_w8a8_dynamic_mlp_matches_prequantized_input(self):
        torch.manual_seed(2026)
        device = torch.device("npu")
        hidden_size = 256
        intermediate_size = 256
        cases = (
            dict(dtype=torch.bfloat16),
            dict(dtype=torch.float16),
        )
        for case in cases:
            with self.subTest(**case):
                dtype = case["dtype"]
                hidden_states = (torch.randn(4, hidden_size, device=device, dtype=dtype) / 10).contiguous()
                quant_hidden, per_token_scale = torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
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
                    w13_weight_scale=torch.rand(2, 2 * intermediate_size, device=device, dtype=torch.float32),
                    w2_weight_scale=torch.rand(
                        2,
                        hidden_size,
                        device=device,
                        dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float32,
                    ),
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

                torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)


@a5_test
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

    def test_w8a8_mxfp8_gmmfr_matches_unfused_combine(self):
        torch.manual_seed(2026)
        device = torch.device("npu")
        hidden_size = 128
        intermediate_size = 64
        hidden_states = (torch.randn(4, hidden_size, device=device, dtype=torch.bfloat16) / 10).contiguous()
        quant_hidden, per_token_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        w13_weight, w13_weight_scale = make_mxfp8_ones(2, hidden_size, 2 * intermediate_size, device=device)
        w2_weight, w2_weight_scale = make_mxfp8_ones(2, intermediate_size, hidden_size, device=device)
        mlp_input = MoEMlpComputeInput(
            hidden_states=quant_hidden,
            group_list=torch.tensor([2, 2], dtype=torch.int64, device=device),
            group_list_type=1,
            weights=MoEWeights(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                w13_weight_scale=w13_weight_scale,
                w2_weight_scale=w2_weight_scale,
            ),
            mlp_output_dtype=torch.bfloat16,
            dynamic_scale=per_token_scale,
        )
        metadata = make_static_metadata(quant_hidden, hidden_size)

        fused = w8a8_mxfp8_apply_mlp(mlp_input, metadata)
        unfused = StaticDispatcher.combine(w8a8_mxfp8_apply_mlp(mlp_input), metadata)

        torch.testing.assert_close(fused.cpu().float(), unfused.cpu().float(), atol=1e-3, rtol=1e-3)
