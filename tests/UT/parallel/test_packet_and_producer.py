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
    pool_q, pool_k = chain.unpacker.unpack_pair_batched(packs[0][0], packs[0][1])[-2:]
    chain.make_mask(pool_q, pool_k, scale=1.0, sparsity=0.5)
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


@pytest.fixture
def pure(producer):
    from types import SimpleNamespace

    return SimpleNamespace(packet=producer.modules["packet"], quant=producer.modules["distributed_quant"])


@pytest.mark.parametrize("pre_rearranged", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_optimized_producer_quantizes_once_after_global_max(producer, monkeypatch, pre_rearranged, rank):
    """Avoid provisional writes while preserving final scales, payloads and pools."""
    reference = producer.build(rank=rank, pre_rearranged=pre_rearranged, optimize=False)
    optimized = producer.build(rank=rank, pre_rearranged=pre_rearranged, optimize=True)
    torch.manual_seed(42)
    tensors = [(torch.randn(1, 68, 4, 2) * 3).to(torch.bfloat16) for _ in range(3)]
    vector = reference.produce(*tensors) * 2
    reference.pack(vector)
    quant = producer.modules["distributed_quant"]
    original = quant._quantize_into
    calls = []

    def counted(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(quant, "_quantize_into", counted)
    actual_vector = optimized.produce(*tensors) * 2
    assert not calls, "provisional Q/K quantization is overwritten after MAX"
    assert torch.equal(actual_vector, vector)
    optimized.pack(actual_vector)
    assert calls == [optimized.st_q, optimized.st_k]
    for actual, expected in [(optimized.st_q, reference.st_q), (optimized.st_k, reference.st_k)]:
        for name in ["payload", "scales", "pool_sum"]:
            assert torch.equal(getattr(actual, name), getattr(expected, name))


def test_mixed_mask_reuses_device_retention_and_two_pool_parts(producer, monkeypatch):
    from cpu_support import native_rf_functions

    stock = native_rf_functions()["get_multi_span_blockwise_mask"]
    chain = producer.build(optimize=True)
    chain.reorder_mapping.token_block_mapping["dense_blocks"] = {2}
    chain.reorder_mapping.token_block_mapping["first_frame_blocks"] = {5}
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return stock(*args, **kwargs)

    monkeypatch.setattr(producer.layers, "get_multi_span_blockwise_mask", observed)
    torch.manual_seed(42)
    q, k = [torch.randn(1, 16, 1, 2).to(torch.bfloat16) for _ in range(2)]
    expected = stock(torch.cat((q, k, k)), 0.7, 0.25, {2}, {5}, "BSND", return_binary=True)
    first = chain.make_mask(q, k, scale=0.25, sparsity=0.7)
    second = chain.make_mask(q, k, scale=0.25, sparsity=0.7)
    assert torch.equal(first, expected) and torch.equal(second, expected)
    assert calls[0]["pool_parts"] == 2
    assert calls[0]["retained_idx"].numel() == 0
    assert calls[0]["retained_mask"] is calls[1]["retained_mask"]


@pytest.mark.parametrize("rank", [0, 1])
def test_mixed_pack_reuses_buffers_and_updates_valid_fields(producer, rank):
    actual_chains = [producer.build(rank=source, optimize=True) for source in range(2)]
    reference_chains = [producer.build(rank=source, optimize=False) for source in range(2)]
    torch.manual_seed(42)
    first = [(torch.randn(1, 68, 4, 2) * 3).to(torch.bfloat16) for _ in range(3)]
    initial = [chain.pack(chain.produce(*first))[0] for chain in actual_chains]
    pointers = [(q.data_ptr(), k.data_ptr()) for q, k in initial[rank]]
    second = [tensor * 1.5 for tensor in first]
    actual_vectors = [chain.produce(*second) for chain in actual_chains]
    reference_vectors = [chain.produce(*second) for chain in reference_chains]
    shared_actual = torch.maximum(*actual_vectors)
    shared_reference = torch.maximum(*reference_vectors)
    actual = [chain.pack(shared_actual)[0] for chain in actual_chains]
    expected = [chain.pack(shared_reference)[0] for chain in reference_chains]
    assert [(q.data_ptr(), k.data_ptr()) for q, k in actual[rank]] == pointers
    for chunk in range(2):
        received_actual = [
            torch.stack([actual[source][chunk][field][rank] for source in range(2)]) for field in range(2)
        ]
        received_expected = [
            torch.stack([expected[source][chunk][field][rank] for source in range(2)]) for field in range(2)
        ]
        got = actual_chains[rank].unpacker.unpack_pair_batched(*received_actual)
        want = reference_chains[rank].unpacker.unpack_pair_batched(*received_expected)
        assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize('rank', [0, 1])
def test_direct_producer_preserves_all_received_fields_across_calls(producer, monkeypatch, rank):
    actual = [producer.build(rank=r, optimize=True) for r in range(2)]
    expected = [producer.build(rank=r, optimize=False) for r in range(2)]
    monkeypatch.setattr(type(actual[0]), '_can_write_direct_packet', lambda self: self.optimize_attention_pipeline)
    torch.manual_seed(72)
    inputs = [(torch.randn(1, 68, 4, 2) * 3).to(torch.bfloat16) for _ in range(3)]
    pointers = None
    for repeat in range(2):
        values = [v * (repeat + 1) for v in inputs]
        va = [c.produce(*values) for c in actual]
        ve = [c.produce(*values) for c in expected]
        pa = [c.pack(torch.maximum(*va)) for c in actual]
        pe = [c.pack(torch.maximum(*ve)) for c in expected]
        sends = []
        for i, chain in enumerate(actual):
            vbytes = chain.pack_v(values[2], pa[i][1])
            sends.append([(q, k, vbytes[j]) for j, (q, k) in enumerate(pa[i][0])])
            assert all(t.is_contiguous() for triple in sends[-1] for t in triple)
            wire = chain._direct_qkv_send
            assert wire.ndim == 1
            q_packet, k_packet = chain._qk_packet_buffers
            assert k_packet.data_ptr() - q_packet.data_ptr() == q_packet.numel()
            assert vbytes.data_ptr() - k_packet.data_ptr() == k_packet.numel()
            assert (
                chain.st_q.payload.untyped_storage().data_ptr() == chain._direct_qkv_send.untyped_storage().data_ptr()
            )
        current = [c._direct_qkv_send.data_ptr() for c in actual]
        if pointers is not None:
            assert current == pointers
        pointers = current
        for chunk in range(2):
            got_packets = [torch.stack([sends[src][chunk][field][rank] for src in range(2)]) for field in range(2)]
            ref_packets = [torch.stack([pe[src][0][chunk][field][rank] for src in range(2)]) for field in range(2)]
            got = actual[rank].unpacker.unpack_pair_batched(*got_packets)
            want = expected[rank].unpacker.unpack_pair_batched(*ref_packets)
            assert all(torch.equal(a, b) for a, b in zip(got, want))
            expected_v = [expected[src].pack_v(values[2], pe[src][1])[chunk][rank] for src in range(2)]
            assert torch.equal(torch.stack([sends[src][chunk][2][rank] for src in range(2)]), torch.stack(expected_v))


def test_optimized_mix_rejects_old_mask_helper_before_collectives(producer, monkeypatch):
    monkeypatch.setattr(producer.layers, "get_multi_span_blockwise_mask", lambda pooled, sparsity, scale: pooled)
    with pytest.raises(RuntimeError, match="matching RF mask helper"):
        producer.build(optimize=True)
    assert not producer.events
