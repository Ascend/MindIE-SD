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

from .common import cpu_test, npu_test


@cpu_test
class TestExpertsSelector(unittest.TestCase):
    def test_custom_router_output_is_forwarded(self):
        hidden_states = torch.randint(-8, 8, (2, 4), dtype=torch.int8)
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
            routed_scaling_factor=0.5,
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
        self.assertTrue(torch.equal(topk_weights, torch.tensor([[0.3, 0.2], [0.35, 0.15]])))
        self.assertTrue(torch.equal(topk_ids, torch.tensor([[2, 1], [0, 2]], dtype=torch.int32)))


def torch_grouped_topk_reference(
    router_logits,
    top_k,
    k_group=1,
    group_count=1,
    group_select_mode=0,
    norm_type=0,
    renormalize=False,
    routed_scaling_factor=1.0,
):
    dtype = router_logits.dtype
    router_logits = router_logits.float()
    scores = router_logits.softmax(dim=-1) if norm_type == 0 else router_logits.sigmoid()
    num_experts = router_logits.shape[-1]
    experts_per_group = num_experts // group_count
    grouped_scores = scores.view(scores.shape[0], group_count, experts_per_group)
    if group_select_mode == 0:
        group_scores = grouped_scores.max(dim=-1).values
    else:
        group_scores = grouped_scores.topk(2, dim=-1).values.sum(dim=-1)
    group_ids = group_scores.topk(k_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(dim=-1, index=group_ids, value=True)
    expert_mask = group_mask.repeat_interleave(experts_per_group, dim=-1)
    routed_scores = scores.masked_fill(~expert_mask, float("-inf"))
    topk_result = routed_scores.topk(top_k, dim=-1)
    topk_weights = topk_result.values
    if norm_type == 1:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    elif renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(dtype=dtype), topk_result.indices.to(torch.int32)


def torch_softmax_topk_reference(router_logits, top_k):
    dtype = router_logits.dtype
    router_logits = router_logits.float()
    topk_weights, topk_ids = router_logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(dtype=dtype), topk_ids.to(torch.int32)


if __name__ == "__main__":
    unittest.main()
