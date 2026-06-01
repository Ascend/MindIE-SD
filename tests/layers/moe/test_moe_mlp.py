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
import torch.nn.functional as F

from mindiesd.layers.moe.moe_mlp import unquant_apply_mlp


def torch_mlp_reference(hidden_states, w13_weight, w2_weight, group_list, w13_bias=None, w2_bias=None):
    outputs = []
    start = 0
    for expert_id, end in enumerate(group_list.tolist()):
        if end <= start:
            continue
        gate_up = hidden_states[start:end] @ w13_weight[expert_id]
        if w13_bias is not None:
            gate_up = gate_up + w13_bias[expert_id]
        gate, up = gate_up.chunk(2, dim=-1)
        output = (F.silu(gate) * up) @ w2_weight[expert_id]
        if w2_bias is not None:
            output = output + w2_bias[expert_id]
        outputs.append(output)
        start = end
    return torch.cat(outputs, dim=0)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestMoEMlp(unittest.TestCase):
    def test_unquant_apply_mlp_matches_torch_reference_with_bias(self):
        torch.manual_seed(2026)
        device = torch.device("npu")
        dtype = torch.bfloat16
        hidden_states = torch.randn(3, 4) / 10
        w13_weight = torch.randn(2, 4, 16) / 10
        w2_weight = torch.randn(2, 8, 4) / 10
        w13_bias = torch.randn(2, 16) / 10
        w2_bias = torch.randn(2, 4) / 10
        group_list = torch.tensor([2, 3], dtype=torch.int64)
        expected = torch_mlp_reference(hidden_states, w13_weight, w2_weight, group_list, w13_bias, w2_bias)

        actual = unquant_apply_mlp(
            hidden_states=hidden_states.to(device=device, dtype=dtype),
            w13_weight=w13_weight.to(device=device, dtype=dtype),
            w2_weight=w2_weight.to(device=device, dtype=dtype),
            group_list=group_list.to(device=device),
            w13_bias=w13_bias.to(device=device, dtype=dtype),
            w2_bias=w2_bias.to(device=device, dtype=dtype),
        )

        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
