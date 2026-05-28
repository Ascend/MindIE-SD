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

from .moe_dataclass import MoERoutingInput


def select_experts(
    routing_input: MoERoutingInput,
):
    hidden_states = routing_input.hidden_states
    router_logits = routing_input.router_logits
    top_k = routing_input.top_k
    renormalize = routing_input.renormalize
    custom_routing_function = routing_input.custom_routing_function

    if custom_routing_function is not None:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
    else:
        topk_result = router_logits.softmax(dim=-1).topk(top_k, dim=-1)
        topk_weights = topk_result.values
        topk_ids = topk_result.indices
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_weights = topk_weights.reshape(-1, top_k)
    topk_ids = topk_ids.reshape(-1, top_k).to(torch.int32)
    return topk_weights, topk_ids
