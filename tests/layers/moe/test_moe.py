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
import torch.nn.functional as F

from mindiesd.layers.moe import moe
from mindiesd.layers.moe.moe import resolve_dispatcher_class
from mindiesd.layers.moe.moe_context import set_moe_comm_context
from mindiesd.layers.moe.token_dispatcher import DynamicDispatcher, StaticDispatcher
from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import NPUDevice

from .common import make_moe_kwargs


def mock_dispatch_result(num_tokens=3, hidden_size=4):
    return (
        torch.randn(num_tokens, hidden_size),
        torch.tensor([2, 1]),
        1,
        object(),
    )


def torch_moe_reference(
    hidden_states,
    w13_weight,
    w2_weight,
    topk_weights,
    topk_ids,
    w13_bias=None,
    w2_bias=None,
):
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[-1]
    expanded_hidden = hidden_states.view(num_tokens, 1, hidden_size).repeat(1, top_k, 1).reshape(-1, hidden_size)
    expert_out = torch.zeros(num_tokens * top_k, w2_weight.shape[-1], dtype=hidden_states.dtype)
    flat_topk_ids = topk_ids.reshape(-1)

    for expert_id in range(w13_weight.shape[0]):
        mask = flat_topk_ids == expert_id
        if not mask.any():
            continue
        gate_up = expanded_hidden[mask] @ w13_weight[expert_id]
        if w13_bias is not None:
            gate_up = gate_up + w13_bias[expert_id]
        gate, up = gate_up.chunk(2, dim=-1)
        expert_tokens = F.silu(gate) * up
        expert_out[mask] = expert_tokens @ w2_weight[expert_id]
        if w2_bias is not None:
            expert_out[mask] = expert_out[mask] + w2_bias[expert_id]

    weighted = expert_out.view(num_tokens, top_k, -1) * topk_weights.view(num_tokens, top_k, 1)
    return weighted.sum(dim=1)


def torch_select_experts(router_logits, top_k, renormalize):
    topk_result = router_logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_result.values
    topk_ids = topk_result.indices.to(torch.int32)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def run_static_moe(**kwargs):
    dispatch_result = kwargs.pop(
        "dispatch_result",
        mock_dispatch_result(kwargs["hidden_states"].shape[0], kwargs["hidden_states"].shape[-1]),
    )
    combine_output = kwargs.pop("combine_output", torch.randn_like(kwargs["hidden_states"]))
    dispatched_hidden_states = dispatch_result[0]
    mlp_output = kwargs.pop("mlp_output", torch.randn_like(dispatched_hidden_states))
    with patch.object(StaticDispatcher, "dispatch", return_value=dispatch_result) as mock_dispatch:
        with patch.object(StaticDispatcher, "combine", return_value=combine_output):
            with patch("mindiesd.layers.moe.moe_mlp.unquant_apply_mlp", return_value=mlp_output):
                output = moe(**kwargs)
    return output, mock_dispatch


def run_dynamic_moe(**kwargs):
    dispatch_result = kwargs.pop(
        "dispatch_result",
        mock_dispatch_result(kwargs["hidden_states"].shape[0], kwargs["hidden_states"].shape[-1]),
    )
    prepare_output = (kwargs["hidden_states"], kwargs["router_logits"], kwargs["hidden_states"].shape)
    with patch.object(DynamicDispatcher, "prepare", return_value=prepare_output) as mock_prepare:
        with patch.object(DynamicDispatcher, "dispatch", return_value=dispatch_result) as mock_dispatch:
            with patch.object(DynamicDispatcher, "combine", return_value=torch.randn_like(kwargs["hidden_states"])):
                with patch("mindiesd.layers.moe.moe_mlp.unquant_apply_mlp"):
                    with patch.object(
                        DynamicDispatcher,
                        "finalize",
                        return_value=torch.randn_like(kwargs["hidden_states"]),
                    ):
                        output = moe(**kwargs)
    return output, mock_prepare, mock_dispatch


class TestMoeFunction(unittest.TestCase):
    def setUp(self):
        DynamicDispatcher._split_cpu_buffers.clear()
        set_moe_comm_context()

    def test_static_moe_matches_torch_reference(self):
        torch.manual_seed(2026)
        cases = (
            dict(top_k=1, renormalize=False, with_bias=False, dtype=torch.float16),
            dict(top_k=2, renormalize=True, with_bias=True, dtype=torch.bfloat16),
        )
        for case in cases:
            with self.subTest(**case):
                device = torch.device("npu")
                dtype = case["dtype"]
                num_tokens = 5
                hidden_size = 4
                intermediate_size = 6
                num_experts = 3
                hidden_states = torch.randn(num_tokens, hidden_size) / 10
                router_logits = torch.randn(num_tokens, num_experts) / 10
                w13_weight = torch.randn(num_experts, hidden_size, 2 * intermediate_size) / 10
                w2_weight = torch.randn(num_experts, intermediate_size, hidden_size) / 10
                w13_bias = torch.randn(num_experts, 2 * intermediate_size) / 10 if case["with_bias"] else None
                w2_bias = torch.randn(num_experts, hidden_size) / 10 if case["with_bias"] else None
                topk_weights, topk_ids = torch_select_experts(
                    router_logits,
                    case["top_k"],
                    renormalize=case["renormalize"],
                )
                expected = torch_moe_reference(
                    hidden_states=hidden_states,
                    w13_weight=w13_weight,
                    w2_weight=w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    w13_bias=w13_bias,
                    w2_bias=w2_bias,
                )
                actual = moe(
                    hidden_states=hidden_states.to(device=device, dtype=dtype),
                    w13_weight=w13_weight.to(device=device, dtype=dtype),
                    w2_weight=w2_weight.to(device=device, dtype=dtype),
                    router_logits=router_logits.to(device=device, dtype=dtype),
                    num_experts=num_experts,
                    top_k=case["top_k"],
                    w13_bias=w13_bias.to(device=device, dtype=dtype) if w13_bias is not None else None,
                    w2_bias=w2_bias.to(device=device, dtype=dtype) if w2_bias is not None else None,
                    tokens_full=True,
                    reduce_results=False,
                    dispatcher_type="static",
                    renormalize=case["renormalize"],
                )

                torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=5e-2, rtol=5e-2)

    def test_default_dispatcher_selection_and_static_override(self):
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("mindiesd.layers.moe.moe.get_npu_device", return_value=NPUDevice.A3):
            with patch("torch.distributed.get_world_size", return_value=2):
                _, dynamic_prepare, dynamic_dispatch = run_dynamic_moe(
                    **make_moe_kwargs(num_experts=4, tokens_full=True),
                    ep_group=ep_group,
                )
                with patch.object(DynamicDispatcher, "dispatch") as unused_dynamic_dispatch:
                    with patch("torch.distributed.all_reduce"):
                        _, static_dispatch = run_static_moe(
                            **make_moe_kwargs(num_experts=4, tokens_full=True, dispatcher_type="static"),
                            ep_group=ep_group,
                        )

        dynamic_prepare.assert_called_once()
        dynamic_dispatch.assert_called_once()
        static_dispatch.assert_called_once()
        unused_dynamic_dispatch.assert_not_called()

    def test_resolve_dispatcher_class_by_device_and_override(self):
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_comm_context(ep_group=ep_group)
            with patch("mindiesd.layers.moe.moe.get_npu_device", return_value=NPUDevice.A3):
                self.assertIs(resolve_dispatcher_class(), DynamicDispatcher)
            with patch("mindiesd.layers.moe.moe.get_npu_device", return_value=NPUDevice.A2):
                self.assertIs(resolve_dispatcher_class(), StaticDispatcher)

        self.assertIs(resolve_dispatcher_class("static"), StaticDispatcher)
        self.assertIs(resolve_dispatcher_class("dynamic"), DynamicDispatcher)

    def test_resolve_dispatcher_class_rejects_dynamic_without_ep(self):
        with self.assertRaises(ParametersInvalid):
            resolve_dispatcher_class("dynamic")

    def test_dispatcher_type_overrides_default_routing_to_dynamic(self):
        kwargs = make_moe_kwargs(num_experts=4, tokens_full=True, dispatcher_type="dynamic")
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("mindiesd.layers.moe.moe.get_npu_device", return_value=NPUDevice.A2):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch.object(StaticDispatcher, "dispatch") as static_dispatch:
                    _, _, dynamic_dispatch = run_dynamic_moe(**kwargs, ep_group=ep_group)

        dynamic_dispatch.assert_called_once()
        static_dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
