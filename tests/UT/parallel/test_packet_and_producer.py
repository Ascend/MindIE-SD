# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU contracts for the mixed packet wire format and producer preparation."""

# pylint: disable=redefined-outer-name,no-member,no-name-in-module

import pytest

torch = pytest.importorskip("torch")

from cpu_support import DeviceTensor


def test_packet_round_trip_owns_payload_scale_and_pool_regions(producer):
    packet = producer.modules["packet"]
    with pytest.raises(ValueError, match="align"):
        packet.derive_packet_layout(world=2, s_local=3, hs=1, nblk=1, npool=2)
    world, chunks, s_local, hidden, nblk, npool = 2, 2, 2, 2, 2, 2
    layout = packet.derive_packet_layout(world=world, s_local=s_local, hs=hidden, nblk=nblk, npool=npool)
    assert layout.pool_off % packet.PACKET_ALIGN_BYTES == layout.payload_bytes % packet.PACKET_ALIGN_BYTES
    payload_q = torch.arange(chunks * world * s_local * hidden, dtype=torch.int8).reshape(
        chunks, world, s_local, hidden
    )
    payload_k = payload_q + 32
    scale_q = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]])
    scale_k = torch.tensor([[[8.0, 5.0], [6.0, 7.0]]])
    pool_q = torch.tensor([[[4.0, 8.0]], [[12.0, 16.0]]])
    pool_k = pool_q + 8
    mappings = [{"cells_unique": torch.tensor([0])} for _ in range(world)]
    unpacker = packet.Unpacker(layout, rank_mappings=mappings, pool_cell_token_counts=torch.tensor([4.0]))

    packed_q = packet.pack_qk_packet(payload_q, scale_q, pool_q, 1, layout=layout, chunks=chunks)
    packed_k = packet.pack_qk_packet(payload_k, scale_k, pool_k, 1, layout=layout, chunks=chunks)
    q8, k8, q_scale, k_scale, pooled_q, pooled_k = unpacker.unpack_pair_batched(packed_q, packed_k)
    assert torch.equal(q8.reshape(-1), payload_q[1].reshape(-1))
    assert torch.equal(k8.reshape(-1), payload_k[1].reshape(-1))
    assert torch.equal(q_scale, torch.tensor([[[3.0, 4.0]]]))
    assert torch.equal(k_scale, torch.tensor([[[8.0, 7.0]]]))
    assert torch.equal(pooled_q.float(), torch.tensor([[[[4.0, 6.0]]]]))
    assert torch.equal(pooled_k.float(), torch.tensor([[[[8.0, 10.0]]]]))
    with pytest.raises(RuntimeError, match="dummy pool row"):
        packet.Unpacker(
            packet.derive_packet_layout(world=world, s_local=s_local, hs=hidden, nblk=nblk, npool=1),
            rank_mappings=mappings,
            pool_cell_token_counts=torch.tensor([4.0]),
        )


def test_mixed_producer_builds_one_reduce_operand_and_all_chunk_payloads(producer):
    chain = producer.build()
    shape = (1, 68, 4, 2)
    query = (torch.arange(torch.tensor(shape).prod()).reshape(shape) / 64).to(torch.bfloat16)
    key = query * 2
    value = query * 0.5
    reduce_operand = chain.produce(query, key, value)
    boundary_values = chain.reorder_mapping.num_cross_rank_quant_blocks * 4 * 2
    assert reduce_operand.numel() == boundary_values + 8
    assert torch.equal(reduce_operand[boundary_values:].view(1, 4, 2), chain.compute_v_amax(value))
    with pytest.raises(ValueError, match="all_reduce operand"):
        chain.pack(reduce_operand[:-1])

    packs, v_scales = chain.pack(reduce_operand)
    assert len(packs) == chain.reorder_mapping.chunking
    assert tuple(v_scales.shape) == (1, 4, 2)
    value_send = chain.pack_v(value, v_scales)
    assert tuple(value_send.shape) == (2, 2, 68, 1, 1, 2)
    for chunk, (query_send, key_send) in enumerate(packs):
        prepared = chain.prepare_chunk(
            chunk,
            query_send,
            key_send,
            value_send[chunk],
            v_scales,
            scale=0.25,
            sparsity=0.75,
        )
        q8, k8, q_scale, k_scale, value_global, value_scale, mask = prepared
        assert tuple(q8.shape) == tuple(k8.shape) == (1, 136, 1, 2)
        assert tuple(q_scale.shape) == tuple(k_scale.shape) == (1, 1, chain.reorder_mapping.nblk)
        assert tuple(value_global.shape) == (1, 136, 1, 2)
        assert torch.equal(value_scale, chain.take_v_scale(v_scales, chunk))
        assert mask.dtype == torch.int8
    retained = chain.retained_blocks["retained"]
    chain.make_mask(prepared[-1].float(), prepared[-1].float(), scale=1.0, sparsity=0.5)
    assert chain.retained_blocks["retained"] is retained


def test_v_amax_excludes_padding_and_zero_rank_is_explicit(producer):
    chain = producer.build(rank=1)
    value = torch.ones(1, 68, 4, 2, dtype=torch.bfloat16)
    value[:, 64:] = 1000
    assert torch.equal(chain.compute_v_amax(value), torch.ones(1, 4, 2))
    chain.reorder_mapping.local_rank_mapping["n_tok"] = 0
    assert torch.equal(chain.compute_v_amax(value), torch.zeros(1, 4, 2))
    if "pipeline" in producer.modules:
        assert producer.build(precision="bf16").precision == "bf16"
    else:
        with pytest.raises(ValueError, match="precision='mix'"):
            producer.build(precision="bf16")


def test_execution_record_stream_helper_covers_nested_tensors_when_available(producer):
    if "pipeline" not in producer.modules:
        pytest.skip("execution dependency is not installed yet")
    value = DeviceTensor.wrap(torch.ones(1))
    producer.modules["pipeline"]._record_stream((value, [value]), object())
