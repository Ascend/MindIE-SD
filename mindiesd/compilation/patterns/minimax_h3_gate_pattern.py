#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""MiniMax-H3 residual-gate fusion pattern (register_replacement).

MiniMax-H3 block applies a per-row gate to attention/FF outputs:
    hidden = hidden + gate_msa[adaln_indices] * attn_output   (msa)
    hidden = hidden + gate_mlp[adaln_indices] * ff_output     (mlp)
where gate tables are [3, D] (per-modality rows, L2-resident).

Real graph (verified by dump):
    index_select = gate_table.index_select(0, indices)   # [S, D]
    mul          = index_select * value                   # gate * attn/ff out
    add          = residual + mul

Replacement: mindiesd gather_residual_gate(residual, value, gate_table,
indices) -> ONE triton kernel (scalar idx gather + i32 + 3 rows/program,
bench 61us vs 108us eager chain). Must be registered BEFORE
wan_residual_gate so the generic `x + y*gate` pattern does not match this
index_select form (F2/R4 lesson).
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()

from mindiesd.layers.scale_shift import (
    gather_residual_gate as mindie_gather_residual_gate,  # noqa: E402
)


def create(dtype):
    class MiniMaxH3GatePattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            residual = torch.empty(1, 4, 5376, dtype=dtype, device="meta")
            value = torch.empty(1, 4, 5376, dtype=dtype, device="meta")
            gate_table = torch.empty(3, 5376, dtype=dtype, device="meta")
            indices = torch.empty(4, dtype=torch.int64, device="meta")
            return [residual, value, gate_table, indices]

        @staticmethod
        def pattern(residual, value, gate_table, indices):
            def func(residual, value, gate_table, indices):
                gate = torch.ops.aten.index_select.default(gate_table, 0, indices)
                mul = torch.ops.aten.mul.Tensor(gate, value)
                return torch.ops.aten.add.Tensor(residual, mul)

            return func(residual, value, gate_table, indices)

        @staticmethod
        def replacement(residual, value, gate_table, indices):
            def func(residual, value, gate_table, indices):
                return mindie_gather_residual_gate(residual, value, gate_table, indices)

            return func(residual, value, gate_table, indices)

    return MiniMaxH3GatePattern


MiniMaxH3GatePatternGroup = [
    create(dtype=torch.bfloat16),
    create(dtype=torch.float32),
]
