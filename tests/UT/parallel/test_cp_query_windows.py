# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
from cpu_support import load_parallel_module, new_package


@pytest.fixture(name="hybrid")
def hybrid_module(monkeypatch):
    return load_parallel_module(new_package("_hybrid_window_test", monkeypatch), "hybrid_sparse_attention", monkeypatch)


@pytest.mark.parametrize(
    "phase,length,used,expected",
    [
        (0, 26144, 52282, (0, 26144, 0, 26240)),
        (1, 26144, 52282, (26144, 52282, 26112, 52282)),
        (0, 144, 287, (0, 144, 0, 256)),
        (1, 144, 287, (144, 287, 128, 287)),
        (0, 256, 500, (0, 256, 0, 256)),
        (1, 256, 500, (256, 500, 256, 500)),
        (1, 144, 58, (58, 58, 58, 58)),
    ],
)
def test_global_query_window_preserves_block_boundaries(hybrid, phase, length, used, expected):
    result = hybrid.derive_query_window(phase, length, used, 2)
    assert tuple(result[k] for k in ("owned_start", "owned_end", "start", "end")) == expected
    assert result["end"] <= used
    if result["owned_end"] > result["owned_start"]:
        assert result["start"] % 128 == 0
    assert result["output_start"] == result["owned_start"] - result["start"]


@pytest.mark.parametrize("length,used", [(144, 287), (256, 500), (8, 15)])
def test_boundary_exchange_reconstructs_exact_global_query(hybrid, length, used):
    full = torch.arange(2 * length * 128).reshape(1, 2 * length, 1, 128).to(torch.bfloat16)
    shards = [full[:, :length], full[:, length:]]
    edge = min(length, 128)
    boundary = torch.cat((shards[0][:, -edge:], shards[1][:, :edge]), dim=1)
    for phase in range(2):
        window = hybrid.derive_query_window(phase, length, used)
        result = hybrid.expand_query_window(shards[phase], boundary, phase, window)
        assert torch.equal(result, full[:, window["start"] : window["end"]])
        begin = window["output_start"]
        owned = window["owned_end"] - window["owned_start"]
        assert torch.equal(result[:, begin : begin + owned], full[:, window["owned_start"] : window["owned_end"]])


def test_empty_cp_phase_keeps_bf16_collectives_and_zero_consumers(producer, monkeypatch):
    hybrid = importlib.import_module(producer.package.__name__ + ".hybrid_sparse_attention")
    mapping = producer.build(precision="bf16", pre_rearranged=True).reorder_mapping
    state = producer.modules["state"].ParallelState(group=producer.group, overlap_comm=False)
    chain = producer.modules["pipeline"].Chain(mapping, state=state, precision="mix", pre_rearranged=True)
    gathers = []

    def gather(out, value, group):
        gathers.append(value.dtype)
        out.copy_(value.repeat(2))

    monkeypatch.setattr(hybrid.dist, "all_gather_into_tensor", gather)
    counts = dict(allreduce=0, a2a_calls=0, fused_stages=0)
    q = torch.ones(1, 68, 4, 2, dtype=torch.bfloat16)
    output, counts = hybrid.run_ulysses_kv_gather(
        chain, q, q, q, kv_group=producer.group, kv_rank=1, scale=0.5, sparsity=0.8, inner_precise=4, counts=counts
    )
    assert gathers == [torch.bfloat16] * 4
    assert counts["kv_gather_calls"] == 4 and counts["a2a_calls"] == 8
    assert counts["quantized_bsa_after_comm_calls"] == 0 and counts["quantized_collective_calls"] == 0
    assert torch.count_nonzero(output) == 0
    assert all(event[1].dtype == torch.bfloat16 for event in producer.events)
