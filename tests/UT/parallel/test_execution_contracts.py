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

import logging

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


def test_serial_mix_commits_exact_collectives_stats_and_chain_reuse(producer, monkeypatch, caplog):
    caplog.set_level("DEBUG", logger="mindie-sd")
    monkeypatch.setattr(logging.getLogger("mindie-sd"), "handlers", [caplog.handler])
    monkeypatch.setattr(logging.getLogger("mindie-sd"), "propagate", False)
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
    records = [r for r in caplog.records if r.name == "mindie-sd"]
    assert len(records) == 1 and records[0].levelno == logging.DEBUG and records[0].args


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


@pytest.mark.parametrize("optimize", [False, True])
def test_overlap_stream_protocol_matches_serial_counts_and_finishes_state(producer, fake_npu, monkeypatch, optimize):
    pipeline = producer.modules["pipeline"]
    fake_npu.install(pipeline, monkeypatch, trace_collectives=True)
    state = producer.modules["state"].ParallelState(
        group=producer.group, overlap_comm=True, optimize_attention_pipeline=optimize
    )
    output, stats = call(producer, state)
    assert tuple(output.shape) == (1, 68, 4, 2)
    expected = 8
    assert stats["tally"]["allreduce"] == 1 and stats["tally"]["a2a_calls"] == expected
    assert state._execution_active is False and state._last_execution_event is not None
    assert fake_npu.streams and fake_npu.events
    assert sum(event[0] == "a2a" for event in fake_npu.timeline) == expected
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


def test_npu_serial_tracks_tail_without_comm_stream(producer, fake_npu, monkeypatch):
    from cpu_support import DeviceTensor

    pipeline = producer.modules["pipeline"]
    fake_npu.install(pipeline, monkeypatch, trace_collectives=True)
    state = producer.modules["state"].ParallelState(group=producer.group, overlap_comm=False)
    chain = producer.build()
    chain.state = state
    monkeypatch.setattr(type(chain), "_run_mix", lambda self, q, k, v, **kw: (q, kw["counts"]))
    query = DeviceTensor.wrap(torch.ones(1, 68, 4, 2, dtype=torch.bfloat16))
    chain.run(query, query, query, scale=1.0, sparsity=0.8, inner_precise=4)
    assert state._comm_stream is None and state._last_execution_event is not None
    assert not state._execution_active
    mapping = chain.reorder_mapping
    assert pipeline._can_use_common_reverse_receive_buffer(query, mapping, "mix", "rf_v3", True)
    from types import SimpleNamespace

    assert not pipeline._can_use_common_reverse_receive_buffer(query, SimpleNamespace(bs=2, w=1), "bf16", "rf_v3", True)


def test_serial_npu_protocol_failure_invalidates_state(producer, monkeypatch):
    import inspect
    from cpu_support import DeviceTensor

    entry = producer.modules["distributed_sparse_attention"]
    state = producer.modules["state"].ParallelState(group=producer.group, overlap_comm=False)
    chain = producer.build()
    chain.state = state
    monkeypatch.setattr(entry, "get_attention_reorder_mapping", lambda *a, **kw: chain.reorder_mapping)
    monkeypatch.setattr(
        producer.modules["pipeline"].Chain, "run", lambda self, q, *a, **kw: (q, {"allreduce": 0, "a2a_calls": 0})
    )
    monkeypatch.setattr(producer.modules["pipeline"], "Chain", lambda *a, **kw: chain)
    q = DeviceTensor.wrap(make_inputs()[0])
    options = {"group": producer.group} if "group" in inspect.signature(producer.entry).parameters else {}
    with pytest.raises(RuntimeError, match="all_reduce|all_to_all|BF16 communication"):
        producer.entry(
            q, q, q, [{"start": 4, "latent_shape": [1, 8, 16]}], used_len=132, chunking=2, state=state, **options
        )
    assert state._execution_failed and state.last_call_stats is None
    with pytest.raises(RuntimeError, match="failed execution"):
        state._check_execution()


def test_independent_mix_baseline_cache_is_bounded(producer):
    from types import SimpleNamespace

    state = producer.modules["state"].ParallelState(group=producer.group, optimize_attention_pipeline=False)
    waits = []
    state._last_execution_event = SimpleNamespace(synchronize=lambda: waits.append(len(state.chains)))
    for index in range(6):
        mapping = SimpleNamespace(config=SimpleNamespace(key=(index,)))
        for cache in (state.token_block_mapping_cache, state.reorder_mapping_cache):
            cache[index] = object()
        chain = state.chain(mapping, build=lambda *a, **kw: object())
        assert state.chain(mapping, build=lambda *a, **kw: pytest.fail("rebuilt hot geometry")) is chain
        assert len(state.chains) == 1 and len(waits) == index + 1
        assert len(state.token_block_mapping_cache) <= 1 and len(state.reorder_mapping_cache) <= 1
    assert waits == [0, 1, 1, 1, 1, 1]
