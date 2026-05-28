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
from unittest.mock import MagicMock

import torch

from mindiesd.layers.moe.experts_selector import select_experts
from mindiesd.layers.moe.moe_dataclass import MoERoutingInput


class TestExpertsSelector(unittest.TestCase):
    def test_default_router_selects_topk_and_renormalizes(self):
        router_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        routing_input = MoERoutingInput(
            hidden_states=torch.randn(2, 4),
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
        )
        topk_weights, topk_ids = select_experts(routing_input)

        self.assertEqual(topk_ids.dtype, torch.int32)
        self.assertEqual(topk_ids.shape, torch.Size([2, 2]))
        self.assertTrue(torch.allclose(topk_weights.sum(dim=-1), torch.ones(2)))

    def test_custom_router_output_is_forwarded(self):
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 3)
        custom_routing_function = MagicMock(
            return_value=(
                torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
                torch.tensor([[2, 1], [0, 2]], dtype=torch.int64),
            )
        )
        routing_input = MoERoutingInput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
            custom_routing_function=custom_routing_function,
        )
        topk_weights, topk_ids = select_experts(routing_input)

        custom_routing_function.assert_called_once_with(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=2,
            renormalize=True,
        )
        self.assertEqual(topk_ids.dtype, torch.int32)
        self.assertTrue(torch.equal(topk_weights, torch.tensor([[0.6, 0.4], [0.7, 0.3]])))
        self.assertTrue(torch.equal(topk_ids, torch.tensor([[2, 1], [0, 2]], dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
