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
import torch_npu

from mindiesd.layers.moe import moe
from mindiesd.layers.moe.moe import resolve_dispatcher_class
from mindiesd.layers.moe.moe_context import set_moe_comm_context
from mindiesd.layers.moe.moe_dataclass import (
    MoEPrepareOutput,
    MoEStaticCombineMetadata,
    MoETokenDispatchOutput,
)
from mindiesd.layers.moe.token_dispatcher import DynamicDispatcher, StaticDispatcher
from mindiesd.utils import ParametersInvalid
from .common import (
    a2_a3_test,
    a5_test,
    cpu_test,
    make_moe_kwargs,
    make_mxfp8_ones,
    make_w8a8_dynamic_quant_config,
    make_w8a8_mxfp8_quant_config,
    npu_test,
)


def mock_dispatch_result(num_tokens=3, hidden_size=4):
    return MoETokenDispatchOutput(
        hidden_states=torch.randn(num_tokens, hidden_size),
        dynamic_scale=None,
        group_list=torch.tensor([2, 1]),
        group_list_type=1,
        combine_metadata=object(),
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


def torch_select_experts(router_logits, top_k, renormalize, routed_scaling_factor=1.0):
    router_logits = router_logits.float()
    topk_result = router_logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_result.values
    topk_ids = topk_result.indices.to(torch.int32)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def make_mock_topk(hidden_states, top_k):
    num_tokens = hidden_states.shape[0]
    return (
        torch.ones(num_tokens, top_k),
        torch.zeros(num_tokens, top_k, dtype=torch.int32),
    )


def run_static_moe(**kwargs):
    dispatch_result = kwargs.pop(
        "dispatch_result",
        mock_dispatch_result(kwargs["hidden_states"].shape[0], kwargs["hidden_states"].shape[-1]),
    )
    combine_output = kwargs.pop("combine_output", torch.randn_like(kwargs["hidden_states"]))
    dispatched_hidden_states = dispatch_result.hidden_states
    mlp_output = kwargs.pop("mlp_output", torch.randn_like(dispatched_hidden_states))
    topk_weights, topk_ids = make_mock_topk(kwargs["hidden_states"], kwargs["top_k"])
    with (
        patch.object(StaticDispatcher, "dispatch", return_value=dispatch_result) as mock_dispatch,
        patch.object(StaticDispatcher, "combine", return_value=combine_output),
        patch("mindiesd.layers.moe.moe.unified_apply_mlp", return_value=mlp_output),
        patch("mindiesd.layers.moe.moe.select_experts", return_value=(topk_weights, topk_ids)),
    ):
        output = moe(**kwargs)
    return output, mock_dispatch


def run_dynamic_moe(**kwargs):
    dispatch_result = kwargs.pop(
        "dispatch_result",
        mock_dispatch_result(kwargs["hidden_states"].shape[0], kwargs["hidden_states"].shape[-1]),
    )
    prepare_output = MoEPrepareOutput(
        hidden_states=kwargs["hidden_states"],
        router_logits=kwargs["router_logits"],
        original_shape=kwargs["hidden_states"].shape,
        mlp_output_dtype=kwargs["hidden_states"].dtype,
    )
    topk_weights, topk_ids = make_mock_topk(kwargs["hidden_states"], kwargs["top_k"])
    with (
        patch.object(DynamicDispatcher, "prepare", return_value=prepare_output) as mock_prepare,
        patch.object(DynamicDispatcher, "dispatch", return_value=dispatch_result) as mock_dispatch,
        patch.object(DynamicDispatcher, "combine", return_value=torch.randn_like(kwargs["hidden_states"])),
        patch("mindiesd.layers.moe.moe.unified_apply_mlp", return_value=torch.empty(0)),
        patch.object(DynamicDispatcher, "finalize", return_value=torch.randn_like(kwargs["hidden_states"])),
        patch("mindiesd.layers.moe.moe.select_experts", return_value=(topk_weights, topk_ids)),
    ):
        output = moe(**kwargs)
    return output, mock_prepare, mock_dispatch


class TestMoeFunction(unittest.TestCase):
    def setUp(self):
        DynamicDispatcher._split_cpu_buffers.clear()
        DynamicDispatcher._split_copy_events.clear()
        set_moe_comm_context()

    @npu_test
    def test_static_moe_matches_torch_reference(self):
        torch.manual_seed(2026)
        cases = (
            dict(top_k=1, renormalize=False, routed_scaling_factor=1.0, with_bias=False, dtype=torch.bfloat16),
            dict(top_k=2, renormalize=True, routed_scaling_factor=0.5, with_bias=True, dtype=torch.bfloat16),
            dict(top_k=1, renormalize=False, routed_scaling_factor=1.0, with_bias=False, dtype=torch.float16),
            dict(top_k=2, renormalize=True, routed_scaling_factor=0.5, with_bias=True, dtype=torch.float16),
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
                    router_logits.to(dtype=dtype),
                    case["top_k"],
                    renormalize=case["renormalize"],
                    routed_scaling_factor=case["routed_scaling_factor"],
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
                    router_logits=router_logits.to(device=device, dtype=dtype),
                    num_experts=num_experts,
                    top_k=case["top_k"],
                    w13_weight=w13_weight.to(device=device, dtype=dtype),
                    w2_weight=w2_weight.to(device=device, dtype=dtype),
                    w13_bias=w13_bias.to(device=device, dtype=dtype) if w13_bias is not None else None,
                    w2_bias=w2_bias.to(device=device, dtype=dtype) if w2_bias is not None else None,
                    dispatcher_type="static",
                    inputs_sharded=False,
                    renormalize=case["renormalize"],
                    routed_scaling_factor=case["routed_scaling_factor"],
                    reduce_routed_out=False,
                )

                torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)

    @cpu_test
    def test_default_dispatcher_selection_and_static_override(self):
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            _, dynamic_prepare, dynamic_dispatch = run_dynamic_moe(
                **make_moe_kwargs(num_experts=4, inputs_sharded=False),
                ep_group=ep_group,
            )
            with patch.object(DynamicDispatcher, "dispatch") as unused_dynamic_dispatch:
                with patch("torch.distributed.all_reduce"):
                    _, static_dispatch = run_static_moe(
                        **make_moe_kwargs(num_experts=4, inputs_sharded=False, dispatcher_type="static"),
                        ep_group=ep_group,
                    )

        dynamic_prepare.assert_called_once()
        dynamic_dispatch.assert_called_once()
        static_dispatch.assert_called_once()
        unused_dynamic_dispatch.assert_not_called()

    @cpu_test
    def test_resolve_dispatcher_class_by_topk_and_override(self):
        ep_group = MagicMock(spec=dist.ProcessGroup)

        self.assertIs(resolve_dispatcher_class(1), StaticDispatcher)
        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_comm_context(ep_group=ep_group)
            self.assertIs(resolve_dispatcher_class(1), DynamicDispatcher)
            self.assertIs(resolve_dispatcher_class(2), StaticDispatcher)

        self.assertIs(resolve_dispatcher_class(1, "static"), StaticDispatcher)
        self.assertIs(resolve_dispatcher_class(1, "dynamic"), DynamicDispatcher)

    @cpu_test
    def test_resolve_dispatcher_class_rejects_dynamic_without_ep(self):
        with self.assertRaises(ParametersInvalid):
            resolve_dispatcher_class(1, "dynamic")

    @cpu_test
    def test_dispatcher_type_overrides_default_routing_to_dynamic(self):
        kwargs = make_moe_kwargs(num_experts=4, inputs_sharded=False, dispatcher_type="dynamic")
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            with patch.object(StaticDispatcher, "dispatch") as static_dispatch:
                _, _, dynamic_dispatch = run_dynamic_moe(**kwargs, ep_group=ep_group)

        dynamic_dispatch.assert_called_once()
        static_dispatch.assert_not_called()

    @cpu_test
    def test_return_dispatcher_type_returns_resolved_dispatcher(self):
        output, _ = run_static_moe(
            **make_moe_kwargs(dispatcher_type="static", return_dispatcher_type=True),
        )

        routed_out, dispatcher_type = output
        self.assertIsInstance(routed_out, torch.Tensor)
        self.assertEqual(dispatcher_type, "static")

    @cpu_test
    def test_static_w8a8_uses_dispatcher_combine(self):
        kwargs = make_moe_kwargs(
            hidden_size=256,
            intermediate_size=16,
            quant_config=make_w8a8_dynamic_quant_config(),
            w13_weight_scale=torch.randn(2, 32),
            w2_weight_scale=torch.randn(2, 256),
            dispatcher_type="static",
            reduce_routed_out=False,
        )
        topk_weights, topk_ids = make_mock_topk(kwargs["hidden_states"], kwargs["top_k"])
        metadata = MoEStaticCombineMetadata(
            topk_weights=topk_weights,
            expanded_row_idx=torch.arange(kwargs["hidden_states"].shape[0], dtype=torch.int32),
            restore_shape=kwargs["hidden_states"].shape,
        )
        dispatch_output = MoETokenDispatchOutput(
            hidden_states=torch.randint(-8, 8, kwargs["hidden_states"].shape, dtype=torch.int8),
            dynamic_scale=torch.ones(kwargs["hidden_states"].shape[0], 1),
            group_list=torch.tensor([2, 1]),
            group_list_type=1,
            combine_metadata=metadata,
        )
        expert_output = torch.randn_like(kwargs["hidden_states"])
        expected = torch.randn_like(kwargs["hidden_states"])

        with (
            patch.object(StaticDispatcher, "dispatch", return_value=dispatch_output) as dispatch,
            patch.object(StaticDispatcher, "combine", return_value=expected) as combine,
            patch("mindiesd.layers.moe.moe.select_experts", return_value=(topk_weights, topk_ids)),
            patch(
                "mindiesd.layers.moe.moe.unified_apply_mlp",
                return_value=expert_output,
            ) as apply_mlp,
        ):
            actual = moe(**kwargs)

        torch.testing.assert_close(actual, expected)
        apply_mlp.assert_called_once()
        self.assertIsNone(apply_mlp.call_args.args[1])
        self.assertFalse(dispatch.call_args.args[0].use_gmm_finalize_routing)
        combine.assert_called_once_with(hidden_states=expert_output, combine_metadata=metadata)


@a2_a3_test
class TestMoeW8A8Dynamic(unittest.TestCase):
    def setUp(self):
        set_moe_comm_context()

    def test_w8a8_dynamic_moe_produces_correct_output_shape_and_dtype(self):
        torch.manual_seed(2026)
        device = torch.device("npu")
        num_tokens = 4
        hidden_size = 256
        intermediate_size = 256
        num_experts = 2
        dtype = torch.bfloat16

        hidden_states = (torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) / 10).contiguous()
        router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype) / 10

        # Int8 weights in NZ format
        w13_weight = torch_npu.npu_format_cast(
            torch.randint(-8, 8, (num_experts, hidden_size, 2 * intermediate_size), dtype=torch.int8, device=device),
            29,
        )
        w2_weight = torch_npu.npu_format_cast(
            torch.randint(-8, 8, (num_experts, intermediate_size, hidden_size), dtype=torch.int8, device=device),
            29,
        )
        w13_weight_scale = torch.rand(
            num_experts,
            2 * intermediate_size,
            device=device,
            dtype=torch.float32,
        )
        w2_weight_scale = torch.rand(num_experts, hidden_size, device=device, dtype=dtype)

        output = moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
            num_experts=num_experts,
            top_k=2,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_config=make_w8a8_dynamic_quant_config(),
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            dispatcher_type="static",
            inputs_sharded=False,
            reduce_routed_out=False,
        )

        self.assertEqual(tuple(output.shape), (num_tokens, hidden_size))
        self.assertEqual(output.dtype, dtype)


@a5_test
class TestMoeW8A8Mxfp8(unittest.TestCase):
    def setUp(self):
        set_moe_comm_context()

    def test_w8a8_mxfp8_moe_produces_correct_output_shape_and_dtype(self):
        device = torch.device("npu")
        num_tokens = 4
        hidden_size = 128
        intermediate_size = 64
        num_experts = 2
        dtype = torch.bfloat16

        hidden_states = (torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) / 10).contiguous()
        router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype) / 10
        w13_weight, w13_weight_scale = make_mxfp8_ones(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            device=device,
        )
        w2_weight, w2_weight_scale = make_mxfp8_ones(
            num_experts,
            intermediate_size,
            hidden_size,
            device=device,
        )

        output = moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
            num_experts=num_experts,
            top_k=1,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_config=make_w8a8_mxfp8_quant_config(),
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            dispatcher_type="static",
            inputs_sharded=False,
            reduce_routed_out=False,
        )

        self.assertEqual(tuple(output.shape), (num_tokens, hidden_size))
        self.assertEqual(output.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
