# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU contracts for collective order, execution modes and committed diagnostics."""

# pylint: disable=redefined-outer-name

import pytest

torch = pytest.importorskip("torch")


def make_inputs():
    shape = (1, 68, 4, 2)
    query = (torch.arange(torch.tensor(shape).prod()).reshape(shape) / 64).to(torch.bfloat16)
    return query, query * 2, query * 0.5


def call(producer, state, *, precision="mix", sparse_type="rf_v3", return_stats=True):
    query, key, value = make_inputs()
    return producer.entry(
        query,
        key,
        value,
        [{"start": 4, "latent_shape": [1, 8, 16]}],
        chunking=2,
        group=producer.group,
        used_len=132,
        state=state,
        precision=precision,
        sparse_type=sparse_type,
        return_stats=return_stats,
    )


def test_serial_mix_commits_exact_collectives_stats_and_chain_reuse(producer, monkeypatch):
    pipeline = producer.modules["pipeline"]
    monkeypatch.setattr(pipeline, "_create_comm_stream", lambda device: pytest.fail("serial created a stream"))
    state = producer.modules["state"].ParallelState(group=producer.group)
    output, stats = call(producer, state)
    assert tuple(output.shape) == (1, 68, 4, 2)
    assert stats["tally"] == {
        "allreduce": 1,
        "fused_stages": 2,
        "consumer": "quantized",
        "a2a_calls": 8,
    }
    assert [event[0] for event in producer.events] == ["reduce"] + ["a2a"] * 8
    assert state.last_call_stats["tally"] == stats["tally"]
    first_chain = state._last_chain
    call(producer, state)
    assert state._last_chain is first_chain and len(state.chains) == 1
    assert state.stats()["allreduce_total"] == 2 and state.stats()["a2a_total"] == 16


@pytest.mark.parametrize("sparse_type", ["rf_v2", "rf_v3"])
def test_serial_bf16_uses_no_scale_reduce_and_routes_consumer(producer, monkeypatch, sparse_type):
    monkeypatch.setattr(producer.layers, "is_a5_device", lambda: sparse_type == "rf_v3")
    state = producer.modules["state"].ParallelState(group=producer.group)
    output, stats = call(producer, state, precision="bf16", sparse_type=sparse_type)
    assert tuple(output.shape) == (1, 68, 4, 2)
    assert stats["tally"]["allreduce"] == 0 and stats["tally"]["a2a_calls"] == 8
    assert stats["tally"]["consumer"] == sparse_type
    expected = "rain_fusion" if sparse_type == "rf_v2" else "cann"
    assert state.last_call_stats["actual"]["bsa_backend"] == expected


def test_overlap_stream_protocol_matches_serial_counts_and_finishes_state(producer, fake_npu, monkeypatch):
    pipeline = producer.modules["pipeline"]
    fake_npu.install(pipeline, monkeypatch, trace_collectives=True)
    state = producer.modules["state"].ParallelState(group=producer.group, overlap_comm=True)
    output, stats = call(producer, state)
    assert tuple(output.shape) == (1, 68, 4, 2)
    assert stats["tally"]["allreduce"] == 1 and stats["tally"]["a2a_calls"] == 8
    assert state._execution_active is False and state._last_execution_event is not None
    assert fake_npu.streams and fake_npu.events
    assert sum(event[0] == "a2a" for event in fake_npu.timeline) == 8
    assert any(event[0] == "record" for event in fake_npu.timeline)
    assert any(event[0] == "wait" for event in fake_npu.timeline)


def test_collective_count_rejection_never_publishes_success(producer, monkeypatch):
    entry_module = producer.modules["distributed_sparse_attention"]
    pipeline = producer.modules["pipeline"]
    state = producer.modules["state"].ParallelState(group=producer.group)

    def wrong_run(self, q, k, v, **kwargs):
        self.state._last_chain = self
        return q, {"allreduce": 0, "a2a_calls": 0, "fused_stages": 0}

    monkeypatch.setattr(pipeline.Chain, "run", wrong_run)
    with pytest.raises(RuntimeError, match="exactly ONE all_reduce") as error:
        call(producer, state)
    assert error.value.parallel_call_counts["allreduce"] == 0
    assert state.last_call_stats is None and state._last_chain is None
    with pytest.raises(RuntimeError, match="bf16 must not issue"):
        entry_module._require_collective_counts({"allreduce": 1, "a2a_calls": 8}, "bf16", 2)
    with pytest.raises(RuntimeError, match="exactly 8 all_to_all"):
        entry_module._require_collective_counts({"allreduce": 1, "a2a_calls": 7}, "mix", 2)


def test_validation_precedes_chain_allocation_and_clears_old_snapshot(producer, monkeypatch):
    state = producer.modules["state"].ParallelState(group=producer.group)
    call(producer, state)
    monkeypatch.setattr(producer.modules["pipeline"], "Chain", lambda *args, **kwargs: pytest.fail("built chain"))
    query, key, value = make_inputs()
    with pytest.raises(ValueError, match="block_size_kv"):
        producer.entry(
            query,
            key,
            value,
            [{"start": 4, "latent_shape": [1, 8, 16]}],
            chunking=2,
            group=producer.group,
            used_len=132,
            block_size_kv=64,
            state=state,
        )
    assert state.last_call_stats is None
