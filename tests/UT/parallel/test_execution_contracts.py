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

import importlib
import logging
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dependency_missing", [False, True])
def test_native_mxfp8_runtime_errors_precede_collectives(producer, monkeypatch, dependency_missing):
    name = producer.package.__name__ + "._native_mxfp8"
    if dependency_missing:

        def require_runtime():
            raise ModuleNotFoundError("missing runtime dependency", name="runtime_dependency")

        monkeypatch.setitem(sys.modules, name, SimpleNamespace(require_native_mxfp8_runtime=require_runtime))
        error, message = ModuleNotFoundError, "missing runtime dependency"
    else:
        monkeypatch.setitem(sys.modules, name, None)
        error, message = ValueError, "supplied by the native consumer PR"
    monkeypatch.setattr(producer.layers, "is_a5_device", lambda: True)
    q = torch.ones(1, 68, 4, 128, dtype=torch.bfloat16)
    state = producer.modules["state"].ParallelState(group=producer.group)
    with pytest.raises(error, match=message):
        producer.entry(q, q, q, [], state=state, precision="mxfp8")
    assert not producer.events and state.last_call_stats is None


def make_inputs():
    shape = (1, 68, 4, 2)
    query = (torch.arange(torch.tensor(shape).prod()).reshape(shape) / 64).to(torch.bfloat16)
    return query, query * 2, query * 0.5


def call(producer, state, *, precision="mix", sparse_type="rf_v3", return_stats=True):
    query, key, value = make_inputs()
    if precision == "fp16":
        query, key, value = (t.to(torch.float16) for t in (query, key, value))
    return producer.entry(
        query,
        key,
        value,
        [{"start": 4, "latent_shape": [1, 8, 16]}],
        chunking=2,
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
    state = producer.modules["state"].ParallelState(quantized_comm=True, group=producer.group)
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


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("sparse_type", ["rf_v2", "rf_v3"])
def test_serial_bf16_uses_no_scale_reduce_and_routes_consumer(producer, monkeypatch, sparse_type, precision):
    monkeypatch.setattr(producer.layers, "is_a5_device", lambda: sparse_type == "rf_v3")
    state = producer.modules["state"].ParallelState(quantized_comm=True, group=producer.group)
    output, stats = call(producer, state, precision=precision, sparse_type=sparse_type)
    assert tuple(output.shape) == (1, 68, 4, 2)
    assert stats["tally"]["allreduce"] == 0 and stats["tally"]["a2a_calls"] == 8
    assert stats["tally"]["consumer"] == sparse_type
    expected = "rain_fusion" if sparse_type == "rf_v2" else "eagle"
    assert state.last_call_stats["actual"]["bsa_backend"] == expected
    if sparse_type == "rf_v3":
        assert producer.layers.calls[-1][1]["bsa_backend"] == expected


@pytest.mark.parametrize("optimize", [False, True])
def test_overlap_stream_protocol_matches_serial_counts_and_finishes_state(producer, fake_npu, monkeypatch, optimize):
    pipeline = producer.modules["pipeline"]
    fake_npu.install(pipeline, monkeypatch, trace_collectives=True)
    state = producer.modules["state"].ParallelState(
        quantized_comm=True,
        group=producer.group,
        overlap_comm=True,
        optimize_attention_pipeline=optimize,
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


@pytest.mark.parametrize(
    "case", ["state", "tensor", "shape", "dtype", "batch", "a5", "inner", "block", "mx_shape", "mx_device"]
)
def test_entry_rejects_invalid_contract_before_exchange(producer, monkeypatch, case):
    q = torch.ones(1, 68, 4, 128, dtype=torch.bfloat16)
    state = producer.modules["state"].ParallelState(group=producer.group)
    args = [q, q, q, [{"start": 4, "latent_shape": [1, 8, 16]}]]
    options = dict(state=state, used_len=132, chunking=2)
    if case == "state":
        options["state"] = object()
    elif case == "tensor":
        args[0] = None
    elif case == "shape":
        args[1] = q[:, :1]
    elif case == "dtype":
        args[:3] = [q.float()] * 3
    elif case == "batch":
        options["bs"] = 2
    elif case == "a5":
        monkeypatch.setattr(producer.layers, "is_a5_device", lambda: False)
    elif case == "inner":
        options.update(precision="bf16", inner_precise=4)
    elif case == "block":
        options["block_size_kv"] = 64
    elif case == "mx_shape":
        args[:3] = [q[..., :64]] * 3
        options["precision"] = "mxfp8"
    else:
        options["precision"] = "mxfp8"
        monkeypatch.setattr(producer.layers, "is_a5_device", lambda: False)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        producer.entry(*args, **options)
    assert not producer.events and state.last_call_stats is None


def test_cp_entry_commits_lazy_diagnostics_and_rejects_incomplete_protocol(producer, monkeypatch):
    hybrid = importlib.import_module(producer.package.__name__ + ".hybrid_sparse_attention")
    monkeypatch.setattr(hybrid, "resolve_cp_rank", lambda *args: 0)
    state = producer.modules["state"].ParallelState(group=producer.group, cp_group=producer.group)
    q = torch.ones(1, 68, 4, 128, dtype=torch.bfloat16)
    counts = dict(allreduce=0, a2a_calls=8, fused_stages=0, kv_gather_calls=4, query_boundary_gather_calls=2)
    monkeypatch.setattr(producer.modules["pipeline"].Chain, "run", lambda self, q, *a, **kw: (q, counts))
    options = dict(state=state, used_len=264, chunking=2, pre_rearranged=True)
    spans = [{"start": 8, "latent_shape": [1, 8, 32]}]
    with pytest.raises(ValueError, match="pre-rearranged"):
        producer.entry(q, q, q, spans, **{**options, "pre_rearranged": False})
    with pytest.raises(ValueError, match="head_dim"):
        producer.entry(q[..., :64], q[..., :64], q[..., :64], spans, **options)
    assert producer.entry(q, q, q, spans, **options) is q
    stats = state.last_call_stats
    assert stats["plan"]["sequence_world"] == 4 and stats["hybrid"]["kv_gather_calls"] == 4
    assert stats["hybrid"]["forward_precision"] == "bf16"
    for field, message in [("kv_gather_calls", "KV gather"), ("query_boundary_gather_calls", "query-block boundary")]:
        fresh = producer.modules["state"].ParallelState(group=producer.group, cp_group=producer.group)
        original = counts[field]
        counts[field] = 0
        with pytest.raises(RuntimeError, match=message):
            producer.entry(q, q, q, spans, **{**options, "state": fresh})
        assert fresh._execution_failed and fresh.last_call_stats is None
        counts[field] = original


def test_collective_count_rejection_never_publishes_success(producer, monkeypatch):
    entry_module = producer.modules["distributed_sparse_attention"]
    pipeline = producer.modules["pipeline"]
    state = producer.modules["state"].ParallelState(quantized_comm=True, group=producer.group)

    def wrong_run(self, q, k, v, **kwargs):
        self.state._last_chain = self
        return q, {"allreduce": 0, "a2a_calls": 0, "fused_stages": 0}

    monkeypatch.setattr(pipeline.Chain, "run", wrong_run)
    with pytest.raises(RuntimeError, match="mix requires 1 all_reduce") as error:
        call(producer, state)
    assert error.value.parallel_call_counts["allreduce"] == 0
    assert state.last_call_stats is None and state._last_chain is None
    with pytest.raises(RuntimeError, match="bf16 requires 0 all_reduce"):
        entry_module._require_collective_counts({"allreduce": 1, "a2a_calls": 8}, "bf16", 2)
    with pytest.raises(RuntimeError, match="exactly 8 all_to_all"):
        entry_module._require_collective_counts({"allreduce": 1, "a2a_calls": 7}, "mix", 2)


def test_validation_precedes_chain_allocation_and_clears_old_snapshot(producer, monkeypatch):
    state = producer.modules["state"].ParallelState(quantized_comm=True, group=producer.group)
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
    monkeypatch.setattr(type(chain), "_run_bf16", lambda self, q, k, v, **kw: (q, kw["counts"]))
    query = DeviceTensor.wrap(torch.ones(1, 68, 4, 2, dtype=torch.bfloat16))
    chain.run(query, query, query, scale=1.0, sparsity=0.8, inner_precise=4)
    assert state._comm_stream is None and state._last_execution_event is not None
    assert not state._execution_active
    mapping = chain.reorder_mapping
    assert pipeline._can_use_common_reverse_receive_buffer(query, mapping, "mix", "rf_v3", True)
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


@pytest.mark.parametrize(
    "precision,overlap,phase,cp",
    [("bf16", False, 0, 2), ("mix", True, 1, 2), ("bf16", True, 1, 4), ("mix", False, 3, 4), ("bf16", False, 0, 0)],
)
def test_cp_protocol(producer, fake_npu, monkeypatch, precision, overlap, phase, cp):
    hybrid = importlib.import_module(producer.package.__name__ + ".hybrid_sparse_attention")
    pipeline = producer.modules["pipeline"]
    fake_npu.install(pipeline, monkeypatch, trace_collectives=True)
    chain = producer.build(precision="bf16", pre_rearranged=True)
    chain.precision = precision
    State = producer.modules["state"].ParallelState
    chain.state = State(group=producer.group, overlap_comm=overlap, quantized_comm=precision == "mix")
    q = torch.ones(1, 48 if cp else 68, 4, 2, dtype=torch.bfloat16)
    group = SimpleNamespace(world=cp, rank=phase) if cp else None
    monkeypatch.setattr(
        hybrid.dist, "all_gather_into_tensor", lambda out, val, group: out.copy_(val.repeat(group.world))
    )
    if overlap:
        chain.state._begin_execution(
            torch.npu.current_stream(q.device), build=lambda: pipeline._create_comm_stream(q.device)
        )
    counts = dict(allreduce=0, a2a_calls=0, fused_stages=0)
    options = dict(kv_group=group, kv_rank=phase, scale=0.5, sparsity=0.8, inner_precise=4, counts=counts)
    result, counts = hybrid.run_ulysses_kv_gather(chain, q, q, q, **options)
    assert result.shape == q.shape and torch.isfinite(result).all()
    assert counts["a2a_calls"] == 8
    assert counts["kv_gather_calls"] == 4 * bool(cp) and counts["query_boundary_gather_calls"] == 2 * bool(cp)
    assert counts["allreduce"] == 2 * (precision == "mix")
    if phase >= 2:
        assert torch.count_nonzero(result) == 0


def test_group_membership_contract(producer, monkeypatch):
    hybrid = importlib.import_module(producer.package.__name__ + ".hybrid_sparse_attention")
    ug = SimpleNamespace(world=8, rank=0, members=list(range(8)))
    kv = SimpleNamespace(world=2, rank=0, members=[0, 8])
    monkeypatch.setattr(hybrid.dist, "get_world_size", lambda g=None: 16 if g is None else g.world)
    monkeypatch.setattr(hybrid.dist, "get_rank", lambda g=None: 0 if g is None else g.rank)
    monkeypatch.setattr(hybrid.dist, "get_process_group_ranks", lambda g: g.members)
    assert hybrid.resolve_cp_rank(ug, kv, 8, 0) == 0
    monkeypatch.setattr(hybrid.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(hybrid.dist, "new_group", lambda members, **kwargs: members)
    assert hybrid.create_sparse_parallel_groups() == (list(range(8)), [0, 8])
    assert hybrid.create_ulysses_kv_gather_groups() == (list(range(8)), [0, 8])
    with pytest.raises(ValueError):
        hybrid.derive_sparse_parallel_groups(12)
    with pytest.raises(ValueError):
        hybrid.derive_sparse_parallel_groups(16, 3)
    kv.members = [1, 9]
    with pytest.raises(ValueError):
        hybrid.resolve_cp_rank(ug, kv, 8, 0)


@pytest.mark.parametrize("phase,rank", [(0, 0), (1, 1)])
def test_global_mix_packets_refresh_values_and_reuse_geometry(producer, phase, rank):
    wire = importlib.import_module(producer.package.__name__ + ".quantized_attention_wire")
    mapping = SimpleNamespace(world=2, rank=rank, hc=4, hs=128, chunking=2, used_len=264)

    def chain(enabled):
        return SimpleNamespace(
            reorder_mapping=mapping,
            pre_rearranged=True,
            precision="mix",
            group=producer.group,
            optimize_attention_pipeline=enabled,
            retained_blocks={},
        )

    actual, reference = chain(True), chain(False)
    torch.manual_seed(72)
    values = [torch.randn(1, 68, 4, 128).bfloat16() for _ in range(3)]
    pointer = None
    for repeat in range(2):
        values_now = [v * (repeat + 1) for v in values]
        counts = {"allreduce": 0}
        got = wire.prepare_quantized_wire(actual, *values_now, kv_group=None, phase=phase, counts=counts)
        wanted = wire.prepare_quantized_wire(
            reference, *values_now, kv_group=None, phase=phase, counts={"allreduce": 0}
        )
        assert counts["allreduce"] == 2
        assert all(torch.equal(x, y) for aa, bb in zip(got, wanted) for x, y in zip(aa, bb))
        positions = next(iter(actual.retained_blocks["mix_wire_geometry"].values()))["positions"]
        if pointer is not None:
            assert positions.data_ptr() == pointer
        pointer = positions.data_ptr()


def test_default_bf16_wire_consumes_after_four_transfers_per_chunk(producer, monkeypatch):
    calls = []

    def consume(precision, tensors, options):
        assert precision == "mix" and all(t.dtype == torch.bfloat16 for t in tensors)
        assert any(event[0] == "a2a" for event in producer.events)
        calls.append(tensors[0].shape)
        return tensors[0]

    monkeypatch.setattr(producer.layers, "run_bf16_bsa", consume, raising=False)
    state = producer.modules["state"].ParallelState(group=producer.group, overlap_comm=False)
    assert state.quantized_comm is False
    output, stats = call(producer, state)
    assert output.shape == (1, 68, 4, 2) and len(calls) == 2
    assert stats["tally"]["allreduce"] == 0 and stats["tally"]["a2a_calls"] == 8
    assert all(event[1].dtype == torch.bfloat16 for event in producer.events)
