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

import torch

from mindiesd.quantization.config import QuantConfig
from mindiesd.quantization.mode import QuantAlgorithm


def make_w8a8_dynamic_quant_config():
    return QuantConfig(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)


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
