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
import torch_npu

from mindiesd.layers.moe.moe_dataclass import (
    MoEDynamicCombineMetadata,
    MoEPrepareInput,
    MoEPrepareOutput,
    MoEStaticCombineMetadata,
)
from mindiesd.layers.moe.moe_context import set_moe_comm_context, set_moe_context
from mindiesd.layers.moe.moe_mlp import _normalize_mxfp_scale_layout
from mindiesd.layers.moe.token_dispatcher import DynamicDispatcher, StaticDispatcher
from mindiesd.quantization.mode import QuantAlgorithm
from .common import a2_a3_test, a5_test, cpu_test, make_token_dispatch_input, npu_test


class TestTokenDispatcher(unittest.TestCase):
    def setUp(self):
        DynamicDispatcher._split_cpu_buffers.clear()
        DynamicDispatcher._split_copy_events.clear()
        set_moe_context()

    @cpu_test
    def test_dynamic_ep_prepare_pads_and_slices_full_inputs(self):
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        router_logits = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.get_rank", return_value=1):
                set_moe_comm_context(ep_group=ep_group)
                prepare_input = MoEPrepareInput(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    inputs_sharded=False,
                )
                prepare_output = DynamicDispatcher.prepare(prepare_input)
        set_moe_comm_context()

        self.assertIsInstance(prepare_output, MoEPrepareOutput)
        self.assertEqual(prepare_output.original_shape, (hidden_states.shape, hidden_states.shape[0]))
        self.assertIsNone(prepare_output.dynamic_scale)
        self.assertTrue(torch.equal(prepare_output.hidden_states[0], hidden_states[2]))
        self.assertTrue(torch.equal(prepare_output.router_logits[0], router_logits[2]))
        self.assertTrue(torch.equal(prepare_output.hidden_states[1], torch.zeros(4)))
        self.assertTrue(torch.equal(prepare_output.router_logits[1], torch.zeros(4, dtype=router_logits.dtype)))

    @npu_test
    def test_dynamic_ep_dispatch_uses_all_to_all_path(self):
        device = torch.device("npu")
        hidden_states = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device)
        topk_weights = torch.ones(2, 2, device=device)
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_comm_context(ep_group=ep_group)
            split_copy_event = MagicMock()
            with patch.object(
                DynamicDispatcher,
                "_preprocess",
                return_value=(
                    torch.tensor([2, 2], dtype=torch.int32),
                    torch.tensor([2, 2], dtype=torch.int32),
                    torch.tensor([[2], [2]], device=device),
                    split_copy_event,
                    4,
                ),
            ):
                with patch(
                    "mindiesd.layers.moe.token_dispatcher.all_to_all_single",
                    side_effect=lambda input_tensor, output_splits, input_splits, group: input_tensor,
                ) as all_to_all:
                    token_dispatch_input = make_token_dispatch_input(
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        num_experts=2,
                        top_k=2,
                        local_num_experts=1,
                    )
                    dispatch_output = DynamicDispatcher.dispatch(token_dispatch_input)
            set_moe_comm_context()

        self.assertEqual(dispatch_output.group_list_type, 1)
        self.assertEqual(all_to_all.call_count, 1)
        split_copy_event.synchronize.assert_called_once()
        self.assertIsNone(dispatch_output.dynamic_scale)
        self.assertEqual(dispatch_output.hidden_states.shape, torch.Size([4, 4]))

    @cpu_test
    def test_dynamic_w8a8_dynamic_dispatch_quantizes_before_all_to_all(self):
        hidden_states = torch.randn(2, 4)
        quant_hidden = torch.empty(4, 4, dtype=torch.int8)
        dynamic_scale = torch.randn(4)
        topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        topk_weights = torch.ones(2, 2)
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_context(ep_group=ep_group, quant_algo=QuantAlgorithm.W8A8_DYNAMIC)
            split_copy_event = MagicMock()
            with patch.object(
                DynamicDispatcher,
                "_preprocess",
                return_value=(
                    torch.tensor([2, 2], dtype=torch.int32),
                    torch.tensor([2, 2], dtype=torch.int32),
                    torch.tensor([[2], [2]]),
                    split_copy_event,
                    4,
                ),
            ):
                with patch(
                    "torch_npu.npu_moe_token_permute",
                    return_value=(hidden_states.repeat(2, 1), torch.arange(4)),
                ):
                    with patch(
                        "torch_npu.npu_dynamic_quant", return_value=(quant_hidden, dynamic_scale)
                    ) as dynamic_quant:
                        with patch(
                            "mindiesd.layers.moe.token_dispatcher.all_to_all_single",
                            side_effect=lambda input_tensor, output_splits, input_splits, group: input_tensor,
                        ) as all_to_all:
                            token_dispatch_input = make_token_dispatch_input(
                                hidden_states=hidden_states,
                                topk_weights=topk_weights,
                                topk_ids=topk_ids,
                                num_experts=2,
                                top_k=2,
                                local_num_experts=1,
                            )
                            dispatch_output = DynamicDispatcher.dispatch(token_dispatch_input)
            set_moe_context()

        self.assertIs(dispatch_output.hidden_states, quant_hidden)
        self.assertIs(dispatch_output.dynamic_scale, dynamic_scale)
        dynamic_quant.assert_called_once()
        self.assertEqual(dynamic_quant.call_args.kwargs["dst_type"], torch.int8)
        self.assertEqual(all_to_all.call_count, 2)
        split_copy_event.synchronize.assert_called_once()

    @cpu_test
    def test_dynamic_w8a8_mxfp8_dispatch_defers_quantization_to_mlp(self):
        hidden_states = torch.randn(2, 4)
        permuted_hidden = hidden_states.repeat(2, 1)
        topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        topk_weights = torch.ones(2, 2)
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_context(ep_group=ep_group, quant_algo=QuantAlgorithm.W8A8_MXFP8)
            split_copy_event = MagicMock()
            with patch.object(
                DynamicDispatcher,
                "_preprocess",
                return_value=(
                    torch.tensor([2, 2], dtype=torch.int32),
                    torch.tensor([2, 2], dtype=torch.int32),
                    torch.tensor([[2], [2]]),
                    split_copy_event,
                    4,
                ),
            ):
                with patch(
                    "torch_npu.npu_moe_token_permute",
                    return_value=(permuted_hidden, torch.arange(4)),
                ):
                    with patch("torch_npu.npu_dynamic_mx_quant", create=True) as dynamic_mx_quant:
                        with patch(
                            "mindiesd.layers.moe.token_dispatcher.all_to_all_single",
                            side_effect=lambda input_tensor, output_splits, input_splits, group: input_tensor,
                        ) as all_to_all:
                            dispatch_output = DynamicDispatcher.dispatch(
                                make_token_dispatch_input(
                                    hidden_states=hidden_states,
                                    topk_weights=topk_weights,
                                    topk_ids=topk_ids,
                                    num_experts=2,
                                    top_k=2,
                                    local_num_experts=1,
                                )
                            )
            set_moe_context()

        dynamic_mx_quant.assert_not_called()
        self.assertIs(dispatch_output.hidden_states, permuted_hidden)
        self.assertIsNone(dispatch_output.dynamic_scale)
        self.assertEqual(all_to_all.call_count, 1)
        split_copy_event.synchronize.assert_called_once()

    @cpu_test
    def test_static_combine_casts_topk_weights_to_output_dtype(self):
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        metadata = MoEStaticCombineMetadata(
            topk_weights=torch.ones(2, 1, dtype=torch.float32),
            restore_shape=hidden_states.shape,
            expanded_row_idx=torch.arange(2, dtype=torch.int32),
        )

        with patch("torch_npu.npu_moe_token_unpermute", return_value=hidden_states) as token_unpermute:
            output = StaticDispatcher.combine(hidden_states, metadata)

        self.assertEqual(output.dtype, hidden_states.dtype)
        self.assertEqual(token_unpermute.call_args.kwargs["probs"].dtype, hidden_states.dtype)

    @cpu_test
    def test_dynamic_combine_casts_topk_weights_to_output_dtype(self):
        hidden_states = torch.randn(4, 4, dtype=torch.bfloat16)
        metadata = MoEDynamicCombineMetadata(
            topk_weights=torch.ones(2, 2, dtype=torch.float32),
            hidden_shape=torch.Size([2, 4]),
            input_splits=[2, 2],
            output_splits=[2, 2],
            local_unpermute_indices=torch.arange(4, dtype=torch.int32),
            global_unpermute_indices=None,
        )

        with (
            patch(
                "mindiesd.layers.moe.token_dispatcher.all_to_all_single",
                side_effect=lambda input_tensor, output_splits, input_splits, group: input_tensor,
            ),
            patch(
                "torch_npu.npu_moe_token_unpermute",
                return_value=torch.randn(2, 4, dtype=torch.bfloat16),
            ) as unpermute,
        ):
            output = DynamicDispatcher.combine(hidden_states, metadata)

        self.assertEqual(output.dtype, hidden_states.dtype)
        self.assertEqual(unpermute.call_args.kwargs["probs"].dtype, hidden_states.dtype)

    @npu_test
    def test_dynamic_ep_combine_uses_all_to_all_path(self):
        device = torch.device("npu")
        ep_group = MagicMock(spec=dist.ProcessGroup)
        metadata = MoEDynamicCombineMetadata(
            topk_weights=torch.ones(2, 2, device=device),
            hidden_shape=torch.Size([2, 4]),
            input_splits=[2, 2],
            output_splits=[2, 2],
            local_unpermute_indices=torch.arange(4, dtype=torch.int32, device=device),
            global_unpermute_indices=torch.arange(4, dtype=torch.int32, device=device),
        )

        with patch(
            "mindiesd.layers.moe.token_dispatcher.all_to_all_single",
            side_effect=lambda input_tensor, output_splits, input_splits, group: input_tensor,
        ) as all_to_all:
            with patch("torch.distributed.get_world_size", return_value=2):
                set_moe_comm_context(ep_group=ep_group)
                output = DynamicDispatcher.combine(
                    hidden_states=torch.randn(4, 4, device=device, dtype=torch.bfloat16),
                    combine_metadata=metadata,
                )
            set_moe_comm_context()

        self.assertEqual(output.shape, torch.Size([2, 4]))
        self.assertEqual(all_to_all.call_count, 1)

    @npu_test
    def test_static_ep_masks_nonlocal_topk_weights(self):
        device = torch.device("npu")
        topk_ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32, device=device)
        topk_weights = torch.ones(2, 2, device=device)

        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.get_rank", return_value=1):
                set_moe_comm_context(ep_group=MagicMock(spec=dist.ProcessGroup))
                token_dispatch_input = make_token_dispatch_input(
                    hidden_states=torch.randn(2, 4, device=device, dtype=torch.bfloat16),
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    num_experts=4,
                    top_k=2,
                    local_num_experts=2,
                )
                dispatch_output = StaticDispatcher.dispatch(token_dispatch_input)
                set_moe_comm_context()

        self.assertTrue(
            torch.equal(
                dispatch_output.combine_metadata.topk_weights.cpu(),
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            )
        )

    @cpu_test
    def test_static_w8a8_dynamic_dispatch_uses_expected_row_index_type(self):
        hidden_states = torch.randn(2, 4)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
        topk_weights = torch.ones(2, 1)
        sorted_hidden_states = torch.randint(-8, 8, (2, 4), dtype=torch.int8)
        expanded_row_idx = torch.arange(2, dtype=torch.int32)
        expert_tokens = torch.tensor([1, 1], dtype=torch.int32)
        dynamic_scale = torch.randn(2)
        for use_gmm_finalize_routing, expected_row_idx_type in ((False, 0), (True, 1)):
            with self.subTest(use_gmm_finalize_routing=use_gmm_finalize_routing):
                token_dispatch_input = make_token_dispatch_input(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    use_gmm_finalize_routing=use_gmm_finalize_routing,
                )
                set_moe_context(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)
                with patch(
                    "torch_npu.npu_moe_init_routing_v2",
                    return_value=(sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale),
                ) as init_routing:
                    dispatch_output = StaticDispatcher.dispatch(token_dispatch_input)

                self.assertIs(dispatch_output.hidden_states, sorted_hidden_states)
                self.assertIs(dispatch_output.dynamic_scale, dynamic_scale)
                self.assertEqual(dispatch_output.group_list_type, 0)
                self.assertEqual(init_routing.call_args.kwargs["quant_mode"], 1)
                self.assertEqual(init_routing.call_args.kwargs["expert_tokens_num_type"], 0)
                self.assertEqual(init_routing.call_args.kwargs["row_idx_type"], expected_row_idx_type)
        set_moe_context()

    @cpu_test
    def test_static_w8a8_mxfp8_dispatch_uses_mx_quant_mode(self):
        hidden_states = torch.randn(2, 4)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
        topk_weights = torch.ones(2, 1)
        sorted_hidden_states = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
        expanded_row_idx = torch.arange(2, dtype=torch.int32)
        expert_tokens = torch.tensor([1, 1], dtype=torch.int32)
        dynamic_scale = torch.empty(2, dtype=torch.uint8)
        token_dispatch_input = make_token_dispatch_input(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        set_moe_context(quant_algo=QuantAlgorithm.W8A8_MXFP8)
        with patch(
            "torch_npu.npu_moe_init_routing_v2",
            return_value=(sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale),
        ) as init_routing:
            dispatch_output = StaticDispatcher.dispatch(token_dispatch_input)
        set_moe_context()

        self.assertIs(dispatch_output.hidden_states, sorted_hidden_states)
        self.assertIs(dispatch_output.dynamic_scale, dynamic_scale)
        self.assertEqual(init_routing.call_args.kwargs["quant_mode"], 3)

    @cpu_test
    def test_static_w8a8_dynamic_dispatch_reuses_prepare_quant_scale(self):
        hidden_states = torch.randint(-8, 8, (2, 4), dtype=torch.int8)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
        topk_weights = torch.ones(2, 1)
        dynamic_scale = torch.randn(2)
        sorted_hidden_states = torch.randint(-8, 8, (2, 4), dtype=torch.int8)
        expanded_row_idx = torch.arange(2, dtype=torch.int32)
        expert_tokens = torch.tensor([1, 1], dtype=torch.int32)
        token_dispatch_input = make_token_dispatch_input(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            dynamic_scale=dynamic_scale,
        )

        set_moe_context(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)
        with patch(
            "torch_npu.npu_moe_init_routing_v2",
            return_value=(sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale),
        ) as init_routing:
            dispatch_output = StaticDispatcher.dispatch(token_dispatch_input)
        set_moe_context()

        self.assertIs(dispatch_output.dynamic_scale, dynamic_scale)
        self.assertIs(init_routing.call_args.kwargs["scale"], dynamic_scale)
        self.assertEqual(init_routing.call_args.kwargs["quant_mode"], -1)

    @a2_a3_test
    def test_static_w8a8_dynamic_dispatch_matches_prequantized_input(self):
        device = torch.device("npu")
        cases = (
            dict(name="bf16", dtype=torch.bfloat16),
            dict(name="fp16", dtype=torch.float16),
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                dtype = case["dtype"]
                hidden_states = (torch.randn(4, 4, device=device, dtype=dtype) / 10).contiguous()
                topk_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device=device)
                topk_weights = torch.ones(4, 1, device=device)

                set_moe_context(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)
                internal_quant_output = StaticDispatcher.dispatch(
                    make_token_dispatch_input(
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                    )
                )
                quant_hidden, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
                prepare_quant_output = StaticDispatcher.dispatch(
                    make_token_dispatch_input(
                        hidden_states=quant_hidden,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        dynamic_scale=dynamic_scale,
                    )
                )
                set_moe_context()

                torch.testing.assert_close(prepare_quant_output.hidden_states, internal_quant_output.hidden_states)
                torch.testing.assert_close(prepare_quant_output.dynamic_scale, internal_quant_output.dynamic_scale)
                torch.testing.assert_close(prepare_quant_output.group_list, internal_quant_output.group_list)

    @a5_test
    def test_static_w8a8_mxfp8_dispatch_matches_prequantized_input(self):
        device = torch.device("npu")
        hidden_states = (torch.randn(4, 32, device=device, dtype=torch.bfloat16) / 10).contiguous()
        topk_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32, device=device)
        topk_weights = torch.ones(4, 1, device=device)

        set_moe_context(quant_algo=QuantAlgorithm.W8A8_MXFP8)
        internal_quant_output = StaticDispatcher.dispatch(
            make_token_dispatch_input(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
        )
        quant_hidden, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        dynamic_scale = _normalize_mxfp_scale_layout(dynamic_scale)
        prepare_quant_output = StaticDispatcher.dispatch(
            make_token_dispatch_input(
                hidden_states=quant_hidden,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                dynamic_scale=dynamic_scale,
            )
        )
        set_moe_context()

        torch.testing.assert_close(
            prepare_quant_output.hidden_states.cpu().float(),
            internal_quant_output.hidden_states.cpu().float(),
        )
        prepare_scale = prepare_quant_output.dynamic_scale.reshape(prepare_quant_output.dynamic_scale.shape[0], -1)
        internal_scale = internal_quant_output.dynamic_scale.view(torch.uint8).reshape(
            internal_quant_output.dynamic_scale.shape[0],
            -1,
        )
        self.assertTrue(torch.equal(prepare_scale.cpu(), internal_scale.cpu()))
        torch.testing.assert_close(prepare_quant_output.group_list, internal_quant_output.group_list)

    @cpu_test
    def test_static_prepare_gathers_partial_inputs(self):
        hidden_states = torch.randn(4, 4)
        router_logits = torch.randn(4, 4)
        ep_group = MagicMock(spec=dist.ProcessGroup)
        prepare_input = MoEPrepareInput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            inputs_sharded=True,
        )

        with patch("torch.distributed.all_gather_into_tensor") as all_gather:
            with patch("torch.distributed.get_world_size", return_value=2):
                set_moe_comm_context(ep_group=ep_group)
                prepare_output = StaticDispatcher.prepare(prepare_input)
            set_moe_comm_context()

        self.assertIsInstance(prepare_output, MoEPrepareOutput)
        self.assertEqual(prepare_output.original_shape, hidden_states.shape)
        self.assertIsNone(prepare_output.dynamic_scale)
        self.assertEqual(prepare_output.hidden_states.shape, torch.Size([8, 4]))
        self.assertEqual(prepare_output.router_logits.shape, torch.Size([8, 4]))
        self.assertEqual(all_gather.call_count, 2)

    @cpu_test
    def test_static_w8a8_dynamic_prepare_quantizes_before_all_gather(self):
        hidden_states = torch.randn(4, 4)
        router_logits = torch.randn(4, 4)
        quant_hidden = torch.randint(-8, 8, hidden_states.shape, dtype=torch.int8)
        dynamic_scale = torch.randn(hidden_states.shape[0])
        ep_group = MagicMock(spec=dist.ProcessGroup)
        prepare_input = MoEPrepareInput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            inputs_sharded=True,
        )

        def fake_all_gather(tensor, group):
            return tensor.repeat(2, *([1] * (tensor.dim() - 1)))

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_context(ep_group=ep_group, quant_algo=QuantAlgorithm.W8A8_DYNAMIC)
            with patch("torch_npu.npu_dynamic_quant", return_value=(quant_hidden, dynamic_scale)) as dynamic_quant:
                with patch("mindiesd.layers.moe.token_dispatcher.all_gather", side_effect=fake_all_gather):
                    prepare_output = StaticDispatcher.prepare(prepare_input)
            set_moe_context()

        dynamic_quant.assert_called_once()
        self.assertTrue(torch.equal(dynamic_quant.call_args.args[0], hidden_states))
        self.assertEqual(dynamic_quant.call_args.kwargs["dst_type"], torch.int8)
        self.assertIsInstance(prepare_output, MoEPrepareOutput)
        self.assertEqual(prepare_output.original_shape, hidden_states.shape)
        self.assertEqual(prepare_output.hidden_states.dtype, torch.int8)
        self.assertTrue(torch.equal(prepare_output.hidden_states, quant_hidden.repeat(2, 1)))
        self.assertTrue(torch.equal(prepare_output.router_logits, router_logits.repeat(2, 1)))
        self.assertTrue(torch.equal(prepare_output.dynamic_scale, dynamic_scale.repeat(2)))

    @cpu_test
    def test_static_finalize_uses_reduce_scatter_for_partial_inputs(self):
        hidden_states = torch.randn(8, 4)
        original_shape = torch.Size([4, 4])
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.reduce_scatter_tensor") as reduce_scatter:
            reduce_scatter.side_effect = lambda out, inp, group: out.copy_(inp[: out.shape[0]])
            with patch("torch.distributed.get_world_size", return_value=2):
                set_moe_comm_context(ep_group=ep_group)
                output = StaticDispatcher.finalize(
                    routed_out=hidden_states,
                    original_shape=original_shape,
                    inputs_sharded=True,
                    reduce_routed_out=False,
                )
            set_moe_comm_context()

        self.assertEqual(output.shape, original_shape)
        reduce_scatter.assert_called_once()
        self.assertIs(reduce_scatter.call_args.kwargs["group"], ep_group)

    @cpu_test
    def test_static_finalize_reduce_routed_out_controls_all_reduce(self):
        hidden_states = torch.randn(4, 4)
        tp_group = MagicMock(spec=dist.ProcessGroup)

        for reduce_routed_out, expected_calls in ((True, 1), (False, 0)):
            with self.subTest(reduce_routed_out=reduce_routed_out):
                with patch("torch.distributed.all_reduce") as all_reduce:
                    with patch("torch.distributed.get_world_size", return_value=2):
                        set_moe_comm_context(tp_group=tp_group)
                        output = StaticDispatcher.finalize(
                            routed_out=hidden_states.clone(),
                            original_shape=hidden_states.shape,
                            inputs_sharded=False,
                            reduce_routed_out=reduce_routed_out,
                        )
                    set_moe_comm_context()

                self.assertEqual(output.shape, hidden_states.shape)
                self.assertEqual(all_reduce.call_count, expected_calls)


if __name__ == "__main__":
    unittest.main()
