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
from tests.ops.layers.moe.common import a2_a3_test, a5_test, cpu_test, make_token_dispatch_input, npu_test


class TestTokenDispatcher(unittest.TestCase):
    def setUp(self):
        DynamicDispatcher._split_cpu_buffers.clear()
        DynamicDispatcher._split_copy_events.clear()
        set_moe_context()

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
        # This block distinguishes OCP scaling from the fused ceil scale rule.
        hidden_states[1].fill_(0.015625)
        hidden_states[1, 0] = 0.234375
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
        # Fused MoE routing uses ceil-based scaling, rather than the OCP default.
        quant_hidden, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
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
