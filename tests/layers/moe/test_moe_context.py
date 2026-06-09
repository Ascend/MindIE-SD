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
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from mindiesd.layers.moe.moe_dataclass import (
    MoEMlpComputeInput,
    MoETokenDispatchOutput,
    MoEWeights,
    MoEPrepareInput,
    MoERoutingInput,
    MoETokenDispatchInput,
)
from mindiesd.layers.moe.moe_context import (
    MoECommType,
    build_mlp_compute_input,
    build_moe_weights,
    build_prepare_input,
    build_routing_input,
    build_token_dispatch_input,
    get_moe_comm_type,
    get_moe_group,
    get_moe_quant_algo,
    is_moe_quant,
    set_moe_comm_context,
    set_moe_context,
    validate_moe_inputs,
)
from mindiesd.quantization.config import QuantConfig
from mindiesd.quantization.mode import QuantAlgorithm
from mindiesd.utils import ParametersInvalid

from .common import make_moe_kwargs, make_w8a8_dynamic_quant_config


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU",
    "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.",
)
class TestMoEContext(unittest.TestCase):
    def setUp(self):
        set_moe_context()

    def test_validate_moe_inputs_accepts_valid_inputs(self):
        self.assertEqual(validate_moe_inputs(**make_moe_kwargs()), QuantAlgorithm.NO_QUANT)

    def test_validate_moe_inputs_resolves_empty_quant_config(self):
        self.assertEqual(validate_moe_inputs(**make_moe_kwargs(quant_config=QuantConfig())), QuantAlgorithm.NO_QUANT)

    def test_validate_moe_inputs_accepts_w8a8_dynamic_quant_scales(self):
        self.assertEqual(
            validate_moe_inputs(
                **make_moe_kwargs(
                    quant_config=make_w8a8_dynamic_quant_config(),
                    w13_weight=torch.randint(-8, 8, (2, 4, 16), dtype=torch.int8),
                    w2_weight=torch.randint(-8, 8, (2, 8, 4), dtype=torch.int8),
                    w13_weight_scale=torch.randn(2, 16),
                    w2_weight_scale=torch.randn(2, 4),
                )
            ),
            QuantAlgorithm.W8A8_DYNAMIC,
        )

    def test_validate_moe_inputs_rejects_invalid_parameters(self):
        invalid_cases = (
            dict(name="reduce_results", kwargs=dict(reduce_results="false")),
            dict(name="tokens_full", kwargs=dict(tokens_full="true")),
            dict(name="renormalize", kwargs=dict(renormalize="true")),
            dict(name="num_experts", kwargs=dict(num_experts="2")),
            dict(name="top_k", kwargs=dict(top_k=3)),
            dict(name="k_group", kwargs=dict(k_group=0)),
            dict(name="group_count", kwargs=dict(group_count=0)),
            dict(name="group_select_mode", kwargs=dict(group_select_mode=2)),
            dict(name="routing_method", kwargs=dict(routing_method="relu")),
            dict(name="routing_method_type", kwargs=dict(routing_method=0)),
            dict(name="routed_scaling_factor_int", kwargs=dict(routed_scaling_factor=1)),
            dict(name="routed_scaling_factor", kwargs=dict(routed_scaling_factor="1.0")),
            dict(name="quant_config_type", kwargs=dict(quant_config="int8")),
            dict(name="unsupported_quant_config", kwargs=dict(quant_config=QuantConfig(QuantAlgorithm.W4A16))),
            dict(name="none_quant_with_scale", kwargs=dict(w13_weight_scale=torch.randn(2, 16))),
            dict(name="dispatcher_type", kwargs=dict(dispatcher_type="auto")),
            dict(name="custom_routing_function", kwargs=dict(custom_routing_function=object())),
        )
        for case in invalid_cases:
            with self.subTest(case=case["name"]):
                kwargs = make_moe_kwargs()
                kwargs.update(case["kwargs"])
                with self.assertRaises(ParametersInvalid):
                    validate_moe_inputs(**kwargs)

    def test_validate_moe_inputs_rejects_missing_quant_scales(self):
        w8a8_dynamic_kwargs = make_moe_kwargs(
            quant_config=make_w8a8_dynamic_quant_config(),
            w13_weight=torch.randint(-8, 8, (2, 4, 16), dtype=torch.int8),
            w2_weight=torch.randint(-8, 8, (2, 8, 4), dtype=torch.int8),
            w13_weight_scale=torch.randn(2, 16),
            w2_weight_scale=torch.randn(2, 4),
        )
        cases = (
            dict(name="missing_w13_scale", kwargs=dict(w13_weight_scale=None)),
            dict(name="missing_w2_scale", kwargs=dict(w2_weight_scale=None)),
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                kwargs = dict(w8a8_dynamic_kwargs)
                kwargs.update(case["kwargs"])
                with self.assertRaises(ParametersInvalid):
                    validate_moe_inputs(**kwargs)

    def test_validate_moe_inputs_rejects_invalid_shapes(self):
        invalid_cases = (
            dict(router_logits=torch.randn(4, 2)),
            dict(w13_bias=torch.randn(2, 8)),
            dict(w2_bias=torch.randn(2, 8)),
            dict(w2_weight=torch.randn(2, 7, 5)),
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=tuple(overrides)):
                with self.assertRaises(ParametersInvalid):
                    validate_moe_inputs(**make_moe_kwargs(**overrides))

    def test_validate_moe_inputs_rejects_invalid_ep_partition(self):
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with self.assertRaises(ParametersInvalid):
            with patch("torch.distributed.get_world_size", return_value=3):
                validate_moe_inputs(**make_moe_kwargs(num_experts=4, ep_group=ep_group))

    def test_validate_moe_inputs_rejects_invalid_expert_grouping(self):
        cases = (
            dict(name="uneven_groups", kwargs=dict(num_experts=5, group_count=2)),
            dict(name="too_many_selected_groups", kwargs=dict(num_experts=4, k_group=3, group_count=2)),
            dict(
                name="topk_exceeds_selected_experts",
                kwargs=dict(num_experts=4, top_k=3, k_group=1, group_count=2),
            ),
            dict(
                name="top2_group_score_without_two_experts",
                kwargs=dict(num_experts=4, group_count=4, group_select_mode=1),
            ),
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(ParametersInvalid):
                    validate_moe_inputs(**make_moe_kwargs(**case["kwargs"]))

    def test_set_moe_comm_context_resolves_active_group(self):
        tp_group = MagicMock(spec=dist.ProcessGroup)
        ep_group = MagicMock(spec=dist.ProcessGroup)
        cases = (
            dict(
                name="prefers_ep",
                tp_group=tp_group,
                ep_group=ep_group,
                expected_type=MoECommType.EP,
                expected_group=ep_group,
            ),
            dict(
                name="uses_tp_without_ep",
                tp_group=tp_group,
                ep_group=None,
                expected_type=MoECommType.TP,
                expected_group=tp_group,
            ),
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                with patch("torch.distributed.get_world_size", return_value=2):
                    set_moe_comm_context(tp_group=case["tp_group"], ep_group=case["ep_group"])

                self.assertEqual(get_moe_comm_type(), case["expected_type"])
                self.assertIs(get_moe_group(), case["expected_group"])

    def test_is_moe_quant_resolves_quantization(self):
        set_moe_context()
        self.assertFalse(is_moe_quant())
        self.assertEqual(get_moe_quant_algo(), QuantAlgorithm.NO_QUANT)

        set_moe_context(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)
        self.assertTrue(is_moe_quant())
        self.assertEqual(get_moe_quant_algo(), QuantAlgorithm.W8A8_DYNAMIC)

    def test_build_input_wrappers(self):
        hidden_states = torch.randn(3, 4)
        router_logits = torch.randn(3, 2)
        topk_weights = torch.randn(3, 1)
        topk_ids = torch.zeros(3, 1, dtype=torch.int32)
        w13_weight = torch.randn(2, 4, 16)
        w2_weight = torch.randn(2, 8, 4)
        group_list = torch.tensor([2, 3])
        w13_weight_scale = torch.randn(2, 16)
        w2_weight_scale = torch.randn(2, 4)

        self.assertIsInstance(build_prepare_input(hidden_states, router_logits), MoEPrepareInput)
        routing_input = build_routing_input(
            hidden_states,
            router_logits,
            top_k=1,
            k_group=2,
            group_count=2,
            group_select_mode=1,
            routing_method="sigmoid",
            routed_scaling_factor=0.5,
        )
        self.assertIsInstance(routing_input, MoERoutingInput)
        self.assertEqual(routing_input.k_group, 2)
        self.assertEqual(routing_input.group_count, 2)
        self.assertEqual(routing_input.group_select_mode, 1)
        self.assertEqual(routing_input.norm_type, 1)
        self.assertEqual(routing_input.routed_scaling_factor, 0.5)
        weights = build_moe_weights(
            w13_weight,
            w2_weight,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
        )
        self.assertIsInstance(weights, MoEWeights)
        self.assertIs(weights.w13_weight_scale, w13_weight_scale)
        self.assertIs(weights.w2_weight_scale, w2_weight_scale)
        self.assertIsInstance(
            build_token_dispatch_input(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                num_experts=2,
                top_k=1,
                weights=weights,
            ),
            MoETokenDispatchInput,
        )
        dispatch_output = MoETokenDispatchOutput(
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=1,
            combine_metadata=object(),
        )
        mlp_input = build_mlp_compute_input(
            dispatch_output=dispatch_output,
            weights=weights,
            mlp_output_dtype=torch.float32,
        )
        self.assertIsInstance(mlp_input, MoEMlpComputeInput)
        self.assertIs(mlp_input.hidden_states, hidden_states)
        self.assertIs(mlp_input.group_list, group_list)
        self.assertIsInstance(mlp_input.weights, MoEWeights)
        self.assertIs(mlp_input.weights, weights)


if __name__ == "__main__":
    unittest.main()
