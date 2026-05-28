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
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from mindiesd.layers.moe.moe_dataclass import (
    MoEMlpComputeInput,
    MoEPrepareInput,
    MoERoutingInput,
    MoETokenDispatchInput,
)
from mindiesd.layers.moe.moe_context import (
    MoECommType,
    build_mlp_compute_input,
    build_prepare_input,
    build_routing_input,
    build_token_dispatch_input,
    get_moe_comm_type,
    get_moe_group,
    set_moe_comm_context,
    validate_moe_inputs,
)
from mindiesd.utils import ParametersInvalid

from .common import make_moe_kwargs


class TestMoEContext(unittest.TestCase):
    def setUp(self):
        set_moe_comm_context()

    def test_validate_moe_inputs_accepts_valid_inputs(self):
        validate_moe_inputs(**make_moe_kwargs())

    def test_validate_moe_inputs_rejects_invalid_parameters(self):
        invalid_cases = (
            dict(name="reduce_results", kwargs=dict(reduce_results="false")),
            dict(name="tokens_full", kwargs=dict(tokens_full="true")),
            dict(name="renormalize", kwargs=dict(renormalize="true")),
            dict(name="num_experts", kwargs=dict(num_experts="2")),
            dict(name="top_k", kwargs=dict(top_k=3)),
            dict(name="dispatcher_type", kwargs=dict(dispatcher_type="auto")),
            dict(name="custom_routing_function", kwargs=dict(custom_routing_function=object())),
        )
        for case in invalid_cases:
            with self.subTest(case=case["name"]):
                kwargs = make_moe_kwargs()
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

    def test_set_moe_comm_context_prefers_ep_group(self):
        tp_group = MagicMock(spec=dist.ProcessGroup)
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_comm_context(tp_group=tp_group, ep_group=ep_group)

        self.assertEqual(get_moe_comm_type(), MoECommType.EP)
        self.assertIs(get_moe_group(), ep_group)

    def test_set_moe_comm_context_uses_tp_when_ep_is_absent(self):
        tp_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_comm_context(tp_group=tp_group)

        self.assertEqual(get_moe_comm_type(), MoECommType.TP)
        self.assertIs(get_moe_group(), tp_group)

    def test_build_input_wrappers(self):
        hidden_states = torch.randn(3, 4)
        router_logits = torch.randn(3, 2)
        topk_weights = torch.randn(3, 1)
        topk_ids = torch.zeros(3, 1, dtype=torch.int32)
        w13_weight = torch.randn(2, 4, 16)
        w2_weight = torch.randn(2, 8, 4)
        group_list = torch.tensor([2, 3])

        self.assertIsInstance(build_prepare_input(hidden_states, router_logits), MoEPrepareInput)
        self.assertIsInstance(build_routing_input(hidden_states, router_logits, top_k=1), MoERoutingInput)
        self.assertIsInstance(
            build_token_dispatch_input(hidden_states, topk_weights, topk_ids, 2, 1, w13_weight),
            MoETokenDispatchInput,
        )
        self.assertIsInstance(
            build_mlp_compute_input(hidden_states, group_list, 1, w13_weight, w2_weight),
            MoEMlpComputeInput,
        )


if __name__ == "__main__":
    unittest.main()
