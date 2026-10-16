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

import torch
import torch_npu

from mindiesd.layers.moe.moe_dataclass import MoEStaticCombineMetadata, MoETokenDispatchInput
from mindiesd.quantization.config import QuantConfig
from mindiesd.quantization.mode import QuantAlgorithm
from mindiesd.utils.get_platform import NPUDevice, get_npu_device


TEST_MODE = os.environ.get("MINDIE_TEST_MODE", "ALL")
NPU_DEVICE = None if TEST_MODE == "CPU" else get_npu_device()

cpu_test = unittest.skipIf(TEST_MODE == "NPU", "CPU-compatible test")
npu_test = unittest.skipIf(TEST_MODE == "CPU", "NPU-dependent test")
a2_a3_test = unittest.skipIf(
    TEST_MODE == "CPU" or NPU_DEVICE not in (NPUDevice.A2, NPUDevice.A3),
    "A2/A3 INT8 test",
)
a5_test = unittest.skipIf(TEST_MODE == "CPU" or NPU_DEVICE != NPUDevice.A5, "A5 MXFP8 test")


def make_w8a8_dynamic_quant_config():
    return QuantConfig(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)


def make_w8a8_mxfp8_quant_config():
    return QuantConfig(quant_algo=QuantAlgorithm.W8A8_MXFP8)


def make_mxfp8_ones(*shape, device):
    num_experts, k_size, n_size = shape
    quant_weight, weight_scale = torch_npu.npu_dynamic_mx_quant(
        torch.ones(num_experts, n_size, k_size, device=device, dtype=torch.bfloat16),
        dst_type=torch.float8_e4m3fn,
    )
    weight_scale = weight_scale.reshape(num_experts, n_size, -1, 2)
    return quant_weight.transpose(1, 2), weight_scale.transpose(1, 2)


def make_static_metadata(hidden_states, output_size=None, top_k=1):
    num_tokens = hidden_states.shape[0]
    output_size = output_size or hidden_states.shape[-1]
    return MoEStaticCombineMetadata(
        topk_weights=torch.ones(num_tokens, top_k, device=hidden_states.device),
        expanded_row_idx=torch.arange(num_tokens * top_k, dtype=torch.int32, device=hidden_states.device),
        restore_shape=torch.Size([num_tokens, output_size]),
    )


def make_token_dispatch_input(
    hidden_states,
    topk_weights,
    topk_ids,
    num_experts=2,
    top_k=1,
    local_num_experts=2,
    dynamic_scale=None,
    use_gmm_finalize_routing=False,
):
    return MoETokenDispatchInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_num_experts=local_num_experts,
        dynamic_scale=dynamic_scale,
        use_gmm_finalize_routing=use_gmm_finalize_routing,
    )


def make_moe_kwargs(
    num_tokens=3,
    num_experts=2,
    hidden_size=4,
    intermediate_size=8,
    dtype=torch.float32,
    **overrides,
):
    kwargs = dict(
        hidden_states=torch.randn(num_tokens, hidden_size, dtype=dtype),
        router_logits=torch.randn(num_tokens, num_experts, dtype=dtype),
        num_experts=num_experts,
        top_k=1,
        w13_weight=torch.randn(num_experts, hidden_size, 2 * intermediate_size, dtype=dtype),
        w2_weight=torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype),
    )
    kwargs.update(overrides)
    return kwargs
