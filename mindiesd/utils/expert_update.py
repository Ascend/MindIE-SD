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

"""Typed payload for one expert re-placement.

`mindiesd.eplb` produces it and `mindiesd.layers.moe` consumes it. It lives in
`mindiesd.utils` so both sides can import it without pulling in the NPU-only
modules of the other side.

The field order matches the historical tuple the dispatcher used to return, and
the dataclass stays unpackable, so a third-party dispatcher written against the
old contract keeps working while new code reads named fields.
"""

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ExpertUpdate:
    """New expert weights plus the layout they belong to.

    Args:
        weight1 (Any):
            Re-placed gate/up projection weights, row `k` belongs to the `k`-th
            expert of `local_expert_list` sorted by ascending global expert id.
        weight2 (Any):
            Re-placed down projection weights, same row order as `weight1`.
        local_expert_num (int):
            Number of experts this rank holds after the re-placement.
        device_indices_map (Any):
            `[num_experts]` map from a global expert id to the rank holding it.
        local_expert_indices_map (Any):
            `[num_experts]` per-rank row map. Kept for the legacy contract; the MoE
            forward does not consume it, because a rank only knows its own rows and
            stores -1 for every other rank's experts.
        local_expert_list (list | None):
            Global expert ids held by this rank, in the dispatcher's row order.
    """

    weight1: Any = None
    weight2: Any = None
    local_expert_num: int | None = None
    device_indices_map: Any = None
    local_expert_indices_map: Any = None
    local_expert_list: list = field(default_factory=list)

    def __iter__(self) -> Iterator:
        """Unpack like the historical 6-tuple."""
        yield self.weight1
        yield self.weight2
        yield self.local_expert_num
        yield self.device_indices_map
        yield self.local_expert_indices_map
        yield self.local_expert_list

    def __len__(self) -> int:
        return 6
