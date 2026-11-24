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

"""Single definition of the expert-ordering rule used by the MoE forward.

A re-placement can move any expert to any rank, so the forward cannot rely on a
rank owning a contiguous slice. One rule keeps the three places that touch the
layout consistent:

    everything is ordered by ascending global expert id, and the row of an expert
    inside its owner is its position among the experts that rank owns.

That covers the weight rows the hook publishes, the send order of the all-to-all,
and the column order of the receiver. Before this module the rule was spelled out
three times, so a change in one place could silently disagree with the others.

The helpers are pure torch and take the device map published by the EPLB
dispatcher, which every rank knows.
"""

import torch


def owned_columns(device_map: torch.Tensor, rank: int) -> torch.Tensor:
    """Ascending global expert ids held by `rank`.

    Args:
        device_map (torch.Tensor): `[num_experts]` map from expert id to rank.
        rank (int): The rank to select.

    Returns:
        torch.Tensor: 1-D tensor of expert ids, ascending.
    """
    flat = device_map.to(torch.long).reshape(-1)
    return torch.nonzero(flat == rank, as_tuple=False).flatten()


def within_rank_positions(device_map: torch.Tensor) -> torch.Tensor:
    """Position of every expert inside its owner's ascending-id list.

    Args:
        device_map (torch.Tensor): `[num_experts]` map from expert id to rank.

    Returns:
        torch.Tensor: `[num_experts]` int64 positions, expert id indexed.
    """
    flat = device_map.to(torch.long).reshape(-1)
    num_experts = flat.numel()
    if num_experts == 0:
        return flat

    sorted_experts = torch.argsort(flat, stable=True)
    num_ranks = int(flat.max().item()) + 1
    per_rank = torch.bincount(flat, minlength=num_ranks)
    first_index = torch.cumsum(per_rank, 0) - per_rank
    within = torch.empty_like(sorted_experts)
    within[sorted_experts] = torch.arange(num_experts, device=flat.device) - first_index[flat[sorted_experts]]
    return within


def sort_order(experts) -> list:
    """Row order that sorts a per-expert payload by ascending global expert id.

    Args:
        experts (Sequence[int]): Expert ids in the payload's current row order.

    Returns:
        list: Row indices, so `payload[order]` is sorted by expert id.
    """
    return sorted(range(len(experts)), key=lambda position: experts[position])
