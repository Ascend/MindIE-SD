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
    MoEDynamicCombineMetadata,
    MoEPrepareInput,
    MoETokenDispatchInput,
)
from mindiesd.layers.moe.moe_context import set_moe_comm_context
from mindiesd.layers.moe.token_dispatcher import DynamicDispatcher, StaticDispatcher


class TestTokenDispatcher(unittest.TestCase):
    def setUp(self):
        DynamicDispatcher._split_cpu_buffers.clear()
        set_moe_comm_context()

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
                    tokens_full=True,
                )
                prepare_output = DynamicDispatcher.prepare(prepare_input)
        set_moe_comm_context()

        prepared_hidden_states, prepared_router_logits, original_shape = prepare_output
        self.assertEqual(original_shape, (hidden_states.shape, hidden_states.shape[0]))
        self.assertTrue(torch.equal(prepared_hidden_states[0], hidden_states[2]))
        self.assertTrue(torch.equal(prepared_router_logits[0], router_logits[2]))
        self.assertTrue(torch.equal(prepared_hidden_states[1], torch.zeros(4)))
        self.assertTrue(torch.equal(prepared_router_logits[1], torch.zeros(4)))

    def test_dynamic_ep_dispatch_uses_all_to_all_path(self):
        device = torch.device("npu")
        hidden_states = torch.randn(2, 4, device=device)
        topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device)
        topk_weights = torch.ones(2, 2, device=device)
        ep_group = MagicMock(spec=dist.ProcessGroup)

        with patch("torch.distributed.get_world_size", return_value=2):
            set_moe_comm_context(ep_group=ep_group)
            with patch.object(
                DynamicDispatcher,
                "_preprocess",
                return_value=(
                    torch.tensor([2], device=device),
                    torch.tensor([2, 2], dtype=torch.int32),
                    torch.tensor([2, 2], dtype=torch.int32),
                    None,
                    4,
                ),
            ):
                with patch(
                    "mindiesd.layers.moe.token_dispatcher.all_to_all_single",
                    side_effect=lambda input_tensor, output_splits, input_splits, group: input_tensor,
                ) as all_to_all:
                    token_dispatch_input = MoETokenDispatchInput(
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        num_experts=2,
                        top_k=2,
                        local_num_experts=1,
                    )
                    global_tokens, _, group_list_type, _ = DynamicDispatcher.dispatch(token_dispatch_input)
            set_moe_comm_context()

        self.assertEqual(group_list_type, 1)
        self.assertEqual(all_to_all.call_count, 1)
        self.assertEqual(global_tokens.shape, torch.Size([4, 4]))

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
                    hidden_states=torch.randn(4, 4, device=device),
                    combine_metadata=metadata,
                )
            set_moe_comm_context()

        self.assertEqual(output.shape, torch.Size([2, 4]))
        self.assertEqual(all_to_all.call_count, 1)

    def test_static_ep_masks_nonlocal_topk_weights(self):
        device = torch.device("npu")
        topk_ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32, device=device)
        topk_weights = torch.ones(2, 2, device=device)

        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.get_rank", return_value=1):
                set_moe_comm_context(ep_group=MagicMock(spec=dist.ProcessGroup))
                token_dispatch_input = MoETokenDispatchInput(
                    hidden_states=torch.randn(2, 4, device=device),
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    num_experts=4,
                    top_k=2,
                    local_num_experts=2,
                )
                _, _, _, combine_metadata = StaticDispatcher.dispatch(token_dispatch_input)
                set_moe_comm_context()

        self.assertTrue(
            torch.equal(
                combine_metadata.topk_weights.cpu(),
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            )
        )

    def test_static_prepare_gathers_partial_inputs(self):
        hidden_states = torch.randn(4, 4)
        router_logits = torch.randn(4, 4)
        ep_group = MagicMock(spec=dist.ProcessGroup)
        prepare_input = MoEPrepareInput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            tokens_full=False,
        )

        with patch("torch.distributed.all_gather_into_tensor") as all_gather:
            with patch("torch.distributed.get_world_size", return_value=2):
                set_moe_comm_context(ep_group=ep_group)
                prepared_hidden_states, prepared_router_logits, original_shape = StaticDispatcher.prepare(prepare_input)
            set_moe_comm_context()

        self.assertEqual(original_shape, hidden_states.shape)
        self.assertEqual(prepared_hidden_states.shape, torch.Size([8, 4]))
        self.assertEqual(prepared_router_logits.shape, torch.Size([8, 4]))
        self.assertEqual(all_gather.call_count, 2)

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
                    tokens_full=False,
                    reduce_results=False,
                )
            set_moe_comm_context()

        self.assertEqual(output.shape, original_shape)
        reduce_scatter.assert_called_once()
        self.assertIs(reduce_scatter.call_args.kwargs["group"], ep_group)

    def test_static_finalize_reduce_results_controls_all_reduce(self):
        hidden_states = torch.randn(4, 4)
        tp_group = MagicMock(spec=dist.ProcessGroup)

        for reduce_results, expected_calls in ((True, 1), (False, 0)):
            with self.subTest(reduce_results=reduce_results):
                with patch("torch.distributed.all_reduce") as all_reduce:
                    with patch("torch.distributed.get_world_size", return_value=2):
                        set_moe_comm_context(tp_group=tp_group)
                        output = StaticDispatcher.finalize(
                            routed_out=hidden_states.clone(),
                            original_shape=hidden_states.shape,
                            tokens_full=True,
                            reduce_results=reduce_results,
                        )
                    set_moe_comm_context()

                self.assertEqual(output.shape, hidden_states.shape)
                self.assertEqual(all_reduce.call_count, expected_calls)


if __name__ == "__main__":
    unittest.main()
