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
"""Coordinate and online-softmax contracts without a device compiler."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from mindiesd import flex_attention

flex = importlib.import_module("mindiesd.layers.flash_attn.flex_attention")
parallel = importlib.import_module("mindiesd.parallel.distributed_flex_attention")


@pytest.fixture(name="dense_runtime")
def _dense_runtime_fixture(monkeypatch):
    builds = []

    def build(mask, q_map, kv_map, heads, batch_size):
        builds.append((q_map, kv_map))
        return mask, q_map, kv_map, heads

    def run(q, k, v, prepared, scale, return_lse):
        mask, q_map, kv_map, heads = prepared
        q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))
        ratio = q.shape[1] // k.shape[1]
        k, v = (t.repeat_interleave(ratio, dim=1) for t in (k, v))
        b = torch.arange(q.shape[0])[:, None, None, None]
        h = heads[None, :, None, None]
        m = q_map[None, None, :, None]
        n = kv_map[None, None, None, :]
        scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5 if scale is None else scale)
        if mask.score_mod is not None:
            scores = mask.score_mod(scores, b, h, m, n)
        keep = (m >= 0) & (n >= 0)
        if mask.mask_mod is not None:
            keep = keep & mask.mask_mod(b, h, m, n)
        scores = scores.masked_fill(~keep, -torch.inf)
        lse = scores.logsumexp(-1)
        output = (torch.nan_to_num(scores.softmax(-1)) @ v).transpose(1, 2)
        return (output, lse) if return_lse else output

    monkeypatch.setattr(parallel, "_load_dsl", lambda device: SimpleNamespace(prepare=build, attention=run))
    monkeypatch.setattr(flex, "_load_dsl", lambda device: SimpleNamespace(prepare=build, attention=run))
    monkeypatch.setattr(flex, "check_flex_attention_device", lambda device: None)
    monkeypatch.setattr(parallel.FlexAttentionMask, "_validate", lambda self, batch_size: None)

    def validate(*args, local_error=None):
        if local_error is not None:
            raise local_error

    monkeypatch.setattr(parallel, "_validate_peer_contracts", validate)
    return builds


def test_stable_merge_preserves_empty_rows_and_later_visible_blocks():
    empty_lse = torch.full((1, 2, 3), -torch.inf)
    empty_out = torch.zeros(1, 3, 2, 8)
    out, lse = parallel._merge_attention(empty_out, empty_lse, empty_out, empty_lse)
    assert torch.equal(out, empty_out)
    assert torch.equal(lse, empty_lse)
    visible_out = torch.randn_like(empty_out)
    visible_lse = torch.randn_like(empty_lse)
    out, lse = parallel._merge_attention(out, lse, visible_out, visible_lse)
    torch.testing.assert_close(out, visible_out)
    torch.testing.assert_close(lse, visible_lse)


@torch.no_grad()
def test_ring_rotates_kv_and_merges_joint_context_once(monkeypatch, dense_runtime):
    torch.manual_seed(17)
    q = torch.randn(1, 3, 2, 128, dtype=torch.bfloat16)
    k, v = (torch.randn(1, 6, 2, 128, dtype=torch.bfloat16) for _ in range(2))
    joint_k, joint_v = (torch.randn(1, 1, 2, 128, dtype=torch.bfloat16) for _ in range(2))
    query_positions = torch.tensor([0, 3, 4])
    key_positions = torch.tensor([3, 4])
    group = object()
    monkeypatch.setattr(parallel, "_group_size", lambda g: 3 if g is group else 1)
    monkeypatch.setattr(parallel, "_group_rank", lambda g: 1 if g is group else 0)
    owners = iter((0, 2))

    def rotate(block_k, block_v, positions, process_group):
        owner = next(owners)
        return (
            k[:, owner * 2 : owner * 2 + 2],
            v[:, owner * 2 : owner * 2 + 2],
            torch.arange(owner * 2 + 1, owner * 2 + 3),
        )

    monkeypatch.setattr(parallel, "_rotate_kv", rotate)
    mask = parallel.FlexAttentionMask(lambda b, h, m, n: m >= n, batch_shared=True, head_shared=True)
    actual, actual_lse = parallel.distributed_flex_attention(
        q,
        k[:, 2:4],
        v[:, 2:4],
        mask=mask,
        ring_group=group,
        query_positions=query_positions,
        key_positions=key_positions,
        joint_key=joint_k,
        joint_value=joint_v,
        joint_key_positions=torch.tensor([0]),
        return_lse=True,
    )
    expected, expected_lse = parallel.distributed_flex_attention(
        q,
        torch.cat((joint_k, k), 1),
        torch.cat((joint_v, v), 1),
        mask=parallel.FlexAttentionMask(lambda b, h, m, n: m >= n),
        query_positions=query_positions,
        key_positions=torch.arange(7),
        return_lse=True,
    )
    # The dense test runtime returns FP32; Ring restores the query dtype.
    torch.testing.assert_close(actual, expected.to(q.dtype), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-6, rtol=2e-5)


@torch.no_grad()
def test_rejects_missing_joint_pair_before_collectives(dense_runtime):
    x = torch.zeros(1, 3, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="joint_key and joint_value"):
        parallel.distributed_flex_attention(x, x, x, mask=parallel.FlexAttentionMask(), joint_key=x)


@torch.no_grad()
def test_bnsd_rejects_malformed_joint_before_transpose(dense_runtime):
    x = torch.zeros(1, 2, 3, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="four-dimensional"):
        parallel.distributed_flex_attention(
            x, x, x, mask=parallel.FlexAttentionMask(), layout="BNSD", joint_key=object(), joint_value=object()
        )


def test_peer_cache_reuse_requires_collective_agreement(monkeypatch):
    group = object()
    monkeypatch.setattr(parallel, "_group_size", lambda g: 2 if g is group else 1)
    monkeypatch.setattr(parallel, "_group_rank", lambda g: 0)
    gathered_calls = []

    def gather(received, descriptor, group):
        gathered_calls.append(1)
        received[0].copy_(descriptor)
        received[1].copy_(descriptor)
        # Ring allows different peer Q lengths. A peer changing this length
        # must still enter validation when our local plan is already cached.
        received[1][1] = 4

    def reduce(tensor, op, group):
        # Emulate another rank missing its cache; validation has no errors.
        if op == parallel.dist.ReduceOp.MIN:
            tensor[1] = 0
        else:
            tensor.zero_()

    monkeypatch.setattr(parallel.dist, "all_gather", gather)
    monkeypatch.setattr(parallel.dist, "all_reduce", reduce)
    x = torch.zeros(1, 3, 2, 128, dtype=torch.bfloat16)
    mask = parallel.FlexAttentionMask()
    for _ in range(2):
        parallel._validate_peer_contracts(mask, x, x, x, 0, None, group, True)
    assert len(gathered_calls) == 2


def test_unequal_ring_queries_require_explicit_global_maps(monkeypatch):
    group = object()
    monkeypatch.setattr(parallel, "_group_size", lambda g: 2 if g is group else 1)
    monkeypatch.setattr(parallel, "_group_rank", lambda g: 0)

    def gather(received, descriptor, group):
        for item in received:
            item.copy_(descriptor)
        received[1][1] = descriptor[1] + 1

    monkeypatch.setattr(parallel.dist, "all_gather", gather)
    monkeypatch.setattr(parallel.dist, "all_reduce", lambda *args, **kwargs: None)
    x = torch.zeros(1, 3, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="explicit query_positions"):
        parallel._validate_peer_contracts(parallel.FlexAttentionMask(), x, x, x, 0, None, group)


def test_ulysses_exchanges_sequence_and_heads_and_restores_local_shard(monkeypatch):
    rank = 1
    group = object()
    size = 3
    shards = [torch.arange(96).reshape(2, 2, 6, 4) + peer * 1000 for peer in range(size)]
    head_shards = [torch.cat([tensor.chunk(size, dim=2)[peer] for tensor in shards], dim=1) for peer in range(size)]
    expected_sends = iter(
        (torch.stack(shards[rank].chunk(size, dim=2)), torch.stack(head_shards[rank].chunk(size, dim=1)))
    )
    received = iter(
        (
            torch.stack([tensor.chunk(size, dim=2)[rank] for tensor in shards]),
            torch.stack([tensor.chunk(size, dim=1)[rank] for tensor in head_shards]),
        )
    )
    monkeypatch.setattr(parallel.dist, "get_world_size", lambda process_group: size if process_group is group else 1)

    def exchange(output, packed, group):
        assert packed.is_contiguous()
        torch.testing.assert_close(packed, next(expected_sends))
        output.copy_(next(received))

    monkeypatch.setattr(parallel.dist, "all_to_all_single", exchange)
    exchanged = parallel._ulysses_forward(shards[rank], group)
    torch.testing.assert_close(exchanged, head_shards[rank])
    restored = parallel._ulysses_reverse(exchanged, group)
    torch.testing.assert_close(restored, shards[rank])


def test_ulysses_reverse_rejects_uneven_query_shards_before_exchange(monkeypatch):
    monkeypatch.setattr(parallel.dist, "get_world_size", lambda group: 2)
    with pytest.raises(ValueError, match="query length must be divisible"):
        parallel._ulysses_reverse(torch.empty(1, 3, 2, 4), object())


def test_position_gather_preserves_source_order_and_int64_coordinates(monkeypatch):
    group = object()
    positions = torch.tensor([2**33, -5, 2**33 + 1, -5])[::2]
    peers = (positions, torch.tensor([7, -1]))
    monkeypatch.setattr(parallel.dist, "get_world_size", lambda process_group: 2)

    def gather(received, local, group):
        assert local.is_contiguous()
        torch.testing.assert_close(local, positions)
        for destination, source in zip(received, peers):
            destination.copy_(source)

    monkeypatch.setattr(parallel.dist, "all_gather", gather)
    torch.testing.assert_close(parallel._gather_positions(positions, group), torch.tensor([2**33, 2**33 + 1, 7, -1]))


def test_ring_rotation_uses_global_peer_ranks_and_waits_for_all_buffers(monkeypatch):
    rank = 2
    group = object()
    global_ranks = [2, 5, 7]
    monkeypatch.setattr(parallel.dist, "get_world_size", lambda process_group: 3)
    monkeypatch.setattr(parallel.dist, "get_rank", lambda process_group: rank)
    monkeypatch.setattr(parallel.dist, "get_global_rank", lambda process_group, local_rank: global_ranks[local_rank])
    tensors = (torch.arange(24).reshape(1, 3, 2, 4).transpose(1, 2), torch.ones(1, 2, 3, 4), torch.arange(6)[::2])
    expected = tuple(tensor.contiguous() + 10 for tensor in tensors)
    completed = []

    def operation(op, tensor, peer, process_group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=process_group)

    def exchange(operations):
        assert len(operations) == 6
        requests = []
        for index, (send, receive) in enumerate(zip(operations[::2], operations[1::2])):
            assert send.op is parallel.dist.isend and receive.op is parallel.dist.irecv
            assert send.peer == global_ranks[(rank + 1) % 3]
            assert receive.peer == global_ranks[(rank - 1) % 3]
            assert send.group is group and receive.group is group
            assert send.tensor.is_contiguous()
            torch.testing.assert_close(send.tensor, tensors[index])
            requests.append(SimpleNamespace(wait=lambda i=index: completed.append(("send", i))))

            def complete(i=index, buffer=receive.tensor):
                buffer.copy_(expected[i])
                completed.append(("receive", i))

            requests.append(SimpleNamespace(wait=complete))
        return requests

    monkeypatch.setattr(parallel.dist, "P2POp", operation)
    monkeypatch.setattr(parallel.dist, "batch_isend_irecv", exchange)
    actual = parallel._rotate_kv(*tensors, group)
    assert len(completed) == 6
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)


def test_restore_output_converts_bnsd_and_fp32_lse():
    output = torch.arange(48).reshape(1, 3, 2, 8).to(torch.bfloat16)
    lse = torch.arange(6).reshape(1, 2, 3).to(torch.bfloat16)
    restored, restored_lse = parallel._restore_output(output, lse, None, "BNSD")
    torch.testing.assert_close(restored, output.transpose(1, 2))
    torch.testing.assert_close(restored_lse, lse.float())
    assert restored.is_contiguous() and restored_lse.is_contiguous()


@torch.no_grad()
def test_local_bnsd_joint_prefix_and_public_wrapper_match_dense_attention(dense_runtime):
    torch.manual_seed(29)
    q, k, v = (torch.randn(1, 2, 3, 128, dtype=torch.bfloat16) for _ in range(3))
    joint_k, joint_v = (torch.randn(1, 2, 1, 128, dtype=torch.bfloat16) for _ in range(2))
    actual, actual_lse = parallel.distributed_flex_attention(
        q,
        k,
        v,
        mask=parallel.FlexAttentionMask(),
        layout="BNSD",
        joint_key=joint_k,
        joint_value=joint_v,
        return_lse=True,
    )
    expected, expected_lse = flex_attention(
        q, torch.cat((joint_k, k), 2), torch.cat((joint_v, v), 2), layout="BNSD", return_lse=True
    )
    torch.testing.assert_close(actual, expected.to(q.dtype), atol=0.008, rtol=0.008)
    torch.testing.assert_close(actual_lse, expected_lse)
