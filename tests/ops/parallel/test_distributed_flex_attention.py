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
"""Run with torchrun --nproc-per-node=4 -m pytest <this-file> --test-mode=NPU.

The tests compare true HCCL Ulysses/Ring/hybrid execution with a global FP32
reference with padded rows and replicated joint context.
They intentionally require four real NPUs and never replace kernel/transport.
"""

import os

import pytest
import torch
import torch.distributed as dist

from mindiesd import FlexAttentionMask, make_block_causal_mask
from mindiesd.parallel import distributed_flex_attention
from mindiesd.utils.get_platform import is_a5_device


@pytest.fixture(scope="module", name="distributed_device")
def _distributed_device_fixture():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires torchrun with exactly four NPU ranks")
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    if not is_a5_device():
        pytest.skip("CATLASS DSL FlexAttention requires Ascend 950")
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("hccl")
    yield device
    if owned:
        dist.destroy_process_group()


def _groups(ulysses_size, ring_size):
    rank = dist.get_rank()
    ulysses_group = ring_group = None
    owned_groups = []
    for r in range(ring_size):
        members = list(range(r * ulysses_size, (r + 1) * ulysses_size))
        group = dist.new_group(members)
        if rank in members:
            ulysses_group = group
            owned_groups.append(group)
    for u in range(ulysses_size):
        members = [r * ulysses_size + u for r in range(ring_size)]
        group = dist.new_group(members)
        if rank in members:
            ring_group = group
            owned_groups.append(group)
    return ulysses_group, ring_group, owned_groups


@pytest.mark.parametrize("ulysses_size,ring_size", [(4, 1), (2, 2), (1, 4)])
@torch.inference_mode()
def test_parallel_flex_matches_global_reference(distributed_device, ulysses_size, ring_size):
    device = distributed_device
    ulysses_group, ring_group, owned_groups = _groups(ulysses_size, ring_size)
    try:
        generator = torch.Generator().manual_seed(127)
        local_length, prefix_length = 129, 3
        length, heads, kv_heads, width = 4 * local_length, 8, 4, 128
        query = torch.randn(1, length, heads, width, generator=generator).bfloat16().to(device)
        key = torch.randn(1, length, kv_heads, width, generator=generator).bfloat16().to(device)
        value = torch.randn(1, length, kv_heads, width, generator=generator).bfloat16().to(device)
        joint_key = torch.randn(1, prefix_length, kv_heads, width, generator=generator).bfloat16().to(device)
        joint_value = torch.randn(1, prefix_length, kv_heads, width, generator=generator).bfloat16().to(device)

        valid = torch.ones(length + prefix_length, dtype=torch.bool, device=device)
        valid[prefix_length + 17] = False
        spans = [(17, 301)]
        rank = dist.get_rank()
        shard = slice(rank * local_length, (rank + 1) * local_length)
        mask = make_block_causal_mask(spans, length + prefix_length, device=device, valid_tokens=valid)
        arguments = dict(
            mask=mask,
            ulysses_group=ulysses_group,
            ring_group=ring_group,
            joint_key=joint_key,
            joint_value=joint_value,
            enable_gqa=True,
            return_lse=True,
        )
        # First execution includes compilation; repeat exercises the same request mask cache.
        actual, lse = distributed_flex_attention(query[:, shard], key[:, shard], value[:, shard], **arguments)
        repeated, repeated_lse = distributed_flex_attention(
            query[:, shard], key[:, shard], value[:, shard], **arguments
        )
        torch.npu.synchronize()
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
        torch.testing.assert_close(repeated_lse, lse, rtol=0, atol=0)
        q = query[:, shard].transpose(1, 2).float()
        k = torch.cat((joint_key, key), 1).transpose(1, 2).float().repeat_interleave(2, dim=1)
        v = torch.cat((joint_value, value), 1).transpose(1, 2).float().repeat_interleave(2, dim=1)
        m = torch.arange(prefix_length + rank * local_length, prefix_length + (rank + 1) * local_length, device=device)[
            None, None, :, None
        ]
        n = torch.arange(prefix_length + length, device=device)[None, None, None, :]
        same_block = (m >= 17) & (m < 301) & (n >= 17) & (n < 301)
        keep = ((m >= n) | same_block) & valid[m] & valid[n]
        scores = (q @ k.transpose(-2, -1) * width**-0.5).masked_fill(~keep, -torch.inf)
        expected = (torch.nan_to_num(scores.softmax(-1)) @ v).transpose(1, 2)
        torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=2e-2)
        torch.testing.assert_close(lse, scores.logsumexp(-1), rtol=3e-3, atol=3e-2)
        if rank == 0:
            assert torch.equal(actual[:, 17], torch.zeros_like(actual[:, 17]))
            assert torch.isneginf(lse[:, :, 17]).all()
    finally:
        for group in reversed(owned_groups):
            dist.destroy_process_group(group)


@pytest.mark.parametrize("invalid", ["shape", "map"])
@torch.inference_mode()
def test_peer_validation_failure_is_reported_on_every_rank(distributed_device, invalid):
    device = distributed_device
    ulysses_group, ring_group, owned_groups = _groups(2, 2)
    try:
        failing = dist.get_rank() == 0
        length = 130 if failing and invalid == "shape" else 129
        x = torch.zeros(1, length, 4, 128, dtype=torch.bfloat16, device=device)
        key = value = x
        arguments = {}
        if failing and invalid == "map":
            arguments["query_positions"] = torch.arange(length - 1, device=device)
        with pytest.raises(ValueError):
            distributed_flex_attention(
                x,
                key,
                value,
                mask=FlexAttentionMask(),
                ulysses_group=ulysses_group,
                ring_group=ring_group,
                **arguments,
            )
    finally:
        for group in reversed(owned_groups):
            dist.destroy_process_group(group)
