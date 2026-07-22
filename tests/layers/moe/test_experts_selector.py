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
from unittest.mock import MagicMock

import torch

from mindiesd.layers.moe.experts_selector import select_experts
from mindiesd.layers.moe.moe_dataclass import MoERoutingInput


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU",
    "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.",
)
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


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestExpertsSelectorNPU(unittest.TestCase):
    def test_gating_topk_matches_torch_reference(self):
        router_logits = torch.tensor(
            [[1.0, 4.0, 3.0, 2.0, 7.0, 5.0, 0.0, 6.0], [8.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0, 4.0]],
            device="npu",
        )
        cases = (
            dict(name="softmax", norm_type=0, renormalize=True, group_select_mode=0, dtype=torch.bfloat16),
            dict(
                name="grouped_softmax",
                norm_type=0,
                renormalize=True,
                k_group=1,
                group_count=2,
                group_select_mode=1,
                dtype=torch.bfloat16,
            ),
            dict(name="sigmoid", norm_type=1, group_select_mode=0, dtype=torch.bfloat16),
            dict(
                name="grouped_sigmoid",
                norm_type=1,
                k_group=1,
                group_count=2,
                group_select_mode=1,
                dtype=torch.bfloat16,
            ),
            dict(name="softmax", norm_type=0, renormalize=True, group_select_mode=0, dtype=torch.float16),
            dict(
                name="grouped_softmax",
                norm_type=0,
                renormalize=True,
                k_group=1,
                group_count=2,
                group_select_mode=1,
                dtype=torch.float16,
            ),
            dict(name="sigmoid", norm_type=1, group_select_mode=0, dtype=torch.float16),
            dict(
                name="grouped_sigmoid",
                norm_type=1,
                k_group=1,
                group_count=2,
                group_select_mode=1,
                dtype=torch.float16,
            ),
        )
        for case in cases:
            with self.subTest(**case):
                dtype = case["dtype"]
                case_kwargs = {key: value for key, value in case.items() if key not in ("name", "dtype")}
                router_logits_with_dtype = router_logits.to(dtype=dtype)
                routing_input = MoERoutingInput(
                    hidden_states=torch.randn(2, 4, device="npu", dtype=dtype),
                    router_logits=router_logits_with_dtype,
                    top_k=2,
                    routed_scaling_factor=0.5,
                    **case_kwargs,
                )
                topk_weights, topk_ids = select_experts(routing_input)
                expected_weights, expected_ids = torch_grouped_topk_reference(
                    router_logits_with_dtype.cpu(),
                    top_k=2,
                    routed_scaling_factor=0.5,
                    **case_kwargs,
                )

                torch.testing.assert_close(topk_weights.cpu(), expected_weights)
                self.assertTrue(torch.equal(topk_ids.cpu(), expected_ids))

    def test_gating_topk_softmax_matches_torch_reference(self):
        cases = (
            dict(B=2, num_experts=8, top_k=2, dtype=torch.bfloat16),
            dict(B=4, num_experts=16, top_k=4, dtype=torch.bfloat16),
            dict(B=1, num_experts=32, top_k=1, dtype=torch.bfloat16),
            dict(B=3, num_experts=64, top_k=8, dtype=torch.bfloat16),
            dict(B=2, num_experts=8, top_k=2, dtype=torch.float16),
            dict(B=4, num_experts=16, top_k=4, dtype=torch.float16),
            dict(B=1, num_experts=32, top_k=1, dtype=torch.float16),
            dict(B=3, num_experts=64, top_k=8, dtype=torch.float16),
        )
        for case in cases:
            with self.subTest(**case):
                B = case["B"]
                dtype = case["dtype"]
                top_k = case["top_k"]
                num_experts = case["num_experts"]
                router_logits = torch.stack(
                    [torch.randperm(num_experts, device="npu").to(torch.float32) for _ in range(B)]
                ).to(dtype=dtype)
                routing_input = MoERoutingInput(
                    hidden_states=torch.randn(B, 4, device="npu", dtype=dtype),
                    router_logits=router_logits,
                    top_k=top_k,
                    renormalize=True,
                )
                topk_weights, topk_ids = select_experts(routing_input)

                expected_weights, expected_ids = torch_softmax_topk_reference(router_logits.cpu(), top_k)

                torch.testing.assert_close(topk_weights.cpu(), expected_weights)
                self.assertTrue(
                    torch.equal(
                        topk_ids.cpu().sort(dim=-1).values,
                        expected_ids.cpu().to(torch.int32).sort(dim=-1).values,
                    )
                )


if __name__ == "__main__":
    unittest.main()
