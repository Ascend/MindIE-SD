# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Behavioral contracts for configuration, ownership and caller-owned state."""

# pylint: disable=redefined-outer-name  # pytest fixtures are requested by parameter name.

import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from cpu_support import load_parallel_module, new_package


@pytest.fixture
def foundation(monkeypatch):
    name = "_parallel_foundation_contracts"
    package = new_package(name, monkeypatch)

    def load(relative):
        return load_parallel_module(package, relative, monkeypatch)

    return SimpleNamespace(config=load("config"), topology=load("topology"), state=load("state"), load=load, name=name)


def configuration(foundation, **changes):
    arguments = dict(world=2, hc=4, s_local=68, used_len=132, spans=[{"start": 4, "latent_shape": [1, 8, 16]}], hs=4)
    arguments.update(changes)
    return foundation.config.derive_parallel_config(**arguments)


def test_configuration_is_immutable_and_uses_the_attention_window(foundation):
    config = configuration(foundation)
    assert (config.s_global, config.used_len, config.head_groups, config.chunking, config.w) == (136, 132, 2, 2, 1)
    assert config.key == configuration(foundation).key
    assert config.key == configuration(foundation, spans=[SimpleNamespace(start=4, latent_grid=[1, 8, 16])]).key
    with pytest.raises(FrozenInstanceError):
        config.used_len = 136


@pytest.mark.parametrize(
    "changes,exception,match",
    [
        ({"world": 0}, RuntimeError, "world"),
        ({"hc": 3}, RuntimeError, "head split"),
        ({"chunking": 0}, RuntimeError, "head split"),
        ({"rank": 2}, ValueError, "rank"),
        ({"used_len": 137}, ValueError, "used_len"),
        ({"spans": [{"start": 5, "latent_shape": [1, 8, 16]}]}, ValueError, "outside"),
        ({"spans": None}, ValueError, "spans"),
        ({"used_len": None}, ValueError, "required"),
        ({"chunking": 1}, RuntimeError, "ONE head"),
        ({"spans": [{}]}, ValueError, "latent_shape"),
        ({"spans": [SimpleNamespace()]}, ValueError, "latent_grid"),
        ({"spans": [{"start": 4, "latent_shape": [0, 8, 16]}]}, ValueError, "positive ints"),
        ({"spans": [{"start": 0, "latent_shape": [1, 8, 16]}]}, ValueError, "equal used_len"),
        ({"spans": [{"start": 0, "latent_shape": [1, 8, 16]}] * 2}, ValueError, "previous span"),
    ],
)
def test_invalid_geometry_is_rejected_before_execution(foundation, changes, exception, match):
    with pytest.raises(exception, match=match):
        configuration(foundation, **changes)


@pytest.mark.parametrize("batch,width", [(1, 1), (1, 3), (2, 3)])
def test_reverse_layout_preserves_each_ranks_rows(foundation, batch, width):
    source = torch.arange(batch * 12 * width * 4).reshape(batch, 12, width, 4)
    received = foundation.topology.merge_ranks(foundation.topology.split_into_ranks(source, 3))
    for rank in range(3):
        assert torch.equal(received[:, :, rank], source[:, rank * 4 : (rank + 1) * 4])
    assert foundation.topology.derive_rank_window(1, 68, 132) == (68, 132, 64)
    forward = foundation.topology.head_to_row(2, 3)
    reverse = foundation.topology.row_to_head(2, 3)
    assert [reverse[row] for row in forward] == list(range(6))


@pytest.mark.parametrize("changed", [{"world": 3}, {"rank": 1}, {"device": "npu:1"}])
def test_state_rejects_context_reuse(foundation, changed):
    state = foundation.state.ParallelState()
    context = dict(group=object(), world=2, rank=0, device="cpu")
    state.bind_or_check(**context)
    with pytest.raises(RuntimeError, match="bound"):
        state.bind_or_check(**dict(context, **changed))
    with pytest.raises(RuntimeError, match="different process group"):
        state.bind_or_check(**dict(context, group=object()))


def test_state_cross_checks_live_group_and_rejects_destroyed_group(foundation, monkeypatch):
    dist = foundation.state.dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 1)
    state = foundation.state.ParallelState()
    group = object()
    with pytest.raises(RuntimeError, match="reports"):
        state.bind_or_check(group=group, world=2, rank=0, device="cpu")

    def destroyed(group):
        raise RuntimeError("group destroyed")

    monkeypatch.setattr(dist, "get_world_size", destroyed)
    with pytest.raises(RuntimeError, match="not usable"):
        state.bind_or_check(group=group, world=2, rank=1, device="cpu")


def test_state_reuses_a_chain_only_for_the_same_configuration(foundation):
    state = foundation.state.ParallelState()
    calls = []

    def build(mapping, **kwargs):
        chain = SimpleNamespace(mapping=mapping, state=kwargs["state"])
        calls.append(chain)
        return chain

    first = SimpleNamespace(config=SimpleNamespace(key=("first",)))
    second = SimpleNamespace(config=SimpleNamespace(key=("second",)))
    chain = state.chain(first, build=build)
    assert state.chain(first, build=build) is chain
    assert state.chain(second, build=build) is not chain
    assert len(calls) == state.stats()["engines"] == 2


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("backend", ["cann", "eagle"])
def test_state_configuration_is_explicit_and_independent(foundation, monkeypatch, optimize, backend):
    monkeypatch.setenv("MINDIESD_MINIMAX_H3_FEATURES", "unknown_feature")
    monkeypatch.setenv("H3_BSA_OP", "invalid")
    state = foundation.state.ParallelState(optimize_attention_pipeline=optimize, bsa_backend=backend)
    assert state.optimize_attention_pipeline is optimize
    assert state.bsa_backend == backend
    assert state.overlap_comm is False
    with pytest.raises(AttributeError):
        state.optimize_attention_pipeline = not optimize
    with pytest.raises(AttributeError):
        state.bsa_backend = "eagle" if backend == "cann" else "cann"


def test_state_preserves_default_optimization_and_backend(foundation):
    state = foundation.state.ParallelState()
    assert state.optimize_attention_pipeline is True
    assert state.bsa_backend == "cann"


@pytest.mark.parametrize("optimize", [0, 1, None, "true"])
def test_state_rejects_non_boolean_pipeline_configuration(foundation, optimize):
    with pytest.raises(ValueError, match="optimize_attention_pipeline"):
        foundation.state.ParallelState(optimize_attention_pipeline=optimize)


@pytest.mark.parametrize("backend", [None, "", "invalid"])
def test_state_rejects_unknown_bsa_backend(foundation, backend):
    with pytest.raises(ValueError, match="bsa_backend"):
        foundation.state.ParallelState(bsa_backend=backend)


def qkv_plan(foundation, **changes):
    arguments = dict(
        head_count=56,
        head_dim=128,
        batch_size=1,
        world=8,
        chunking=7,
        pre_rearranged=True,
        overlap_comm=True,
    )
    arguments.update(changes)
    module = sys.modules[foundation.state.QKVLayoutPlan.__module__]
    return module.plan_qkv_layout(**arguments)


@pytest.mark.parametrize("world,chunking", [(8, 7), (4, 14)])
def test_qkv_layout_plan_accepts_only_supported_geometries(foundation, world, chunking):
    plan = qkv_plan(foundation, world=world, chunking=chunking)
    assert plan.supported and plan.output_bnsd
    assert plan.layout == "head_major" and plan.fallback_reason is None
    with pytest.raises(FrozenInstanceError):
        plan.layout = "bsnd"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"requested": False}, "disabled"),
        ({"overlap_comm": False}, "overlap_disabled"),
        ({"pre_rearranged": False}, "input_not_pre_rearranged"),
        ({"head_dim": 64}, "geometry_unsupported"),
    ],
)
def test_qkv_layout_plan_reports_explicit_fallbacks(foundation, changes, reason):
    plan = qkv_plan(foundation, **changes)
    assert not plan.supported and not plan.output_bnsd
    assert plan.layout == "bsnd" and plan.fallback_reason == reason


def test_state_owns_and_validates_the_shared_qkv_plan(foundation):
    plan = qkv_plan(foundation)
    state = foundation.state.ParallelState(qkv_layout_plan=plan)
    assert state.qkv_layout_plan is plan


@pytest.mark.parametrize(
    "mode",
    [
        ("bf16", "rf_v3", 1),
        ("fp32", "rf_v3", False),
        ("bf16", "invalid", False),
        ("bf16", "rf_v2", True),
        ("mix", "rf_v2", False),
    ],
)
def test_invalid_attention_mode_is_rejected(foundation, mode):
    with pytest.raises(ValueError):
        foundation.config._validate_attention_mode(*mode)


def test_world_rank_resolution_requires_a_live_group(foundation, monkeypatch):
    dist = foundation.state.dist
    with pytest.raises(RuntimeError, match="no process group"):
        foundation.topology.resolve_world_rank(None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    group = object()
    with pytest.raises(RuntimeError, match="not initialized"):
        foundation.topology.resolve_world_rank(group)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda value: 2)
    monkeypatch.setattr(dist, "get_rank", lambda value: 1)
    assert foundation.topology.resolve_world_rank(group) == (2, 1)
    assert foundation.topology.resolve_world_rank(group, 4, 3) == (4, 3)


@pytest.mark.parametrize("chunks,world", [(7, 8), (14, 4)])
def test_round_robin_merge_restores_global_head_order(foundation, chunks, world):
    raw = torch.arange(chunks * world * 3 * 128).reshape(chunks, world, 1, 3, 1, 128)
    merged = foundation.topology.merge_reverse_receive_chunks_round_robin(raw)
    heads = merged.reshape(1, 3, chunks * world, 128)
    common = foundation.topology.merge_reverse_receive_chunks(raw)
    for chunk in range(chunks):
        for rank in range(world):
            assert torch.equal(heads[0, :, chunk * world + rank], raw[chunk, rank, 0, :, 0])
            assert torch.equal(common[0, :, rank, chunk], raw[chunk, rank, 0, :, 0])


def test_execution_lifecycle_poisoning_and_stats_snapshot(foundation):
    state = foundation.state.ParallelState()
    state._last_call_stats = {"nested": {"values": [1]}}
    snapshot = state.last_call_stats
    snapshot["nested"]["values"].append(2)
    assert state.last_call_stats == {"nested": {"values": [1]}}

    waited, stream, event = [], object(), object()
    compute = SimpleNamespace(wait_event=waited.append)
    assert state._begin_execution(compute, build=lambda: stream) is stream
    with pytest.raises(RuntimeError, match="active execution"):
        state._begin_execution(compute, build=lambda: stream)
    state._finish_execution(event)
    assert not state._execution_active and state._last_execution_event is event
    assert state._begin_execution(compute, build=lambda: stream) is stream
    assert waited == [event]
    state._fail_execution()
    assert state.last_call_stats is None
    with pytest.raises(RuntimeError, match="failed execution"):
        state._begin_execution(compute, build=lambda: stream)


def test_stream_creation_failure_is_retryable(foundation):
    state = foundation.state.ParallelState()

    def fail():
        raise RuntimeError("stream creation")

    with pytest.raises(RuntimeError, match="stream creation"):
        state._begin_execution(SimpleNamespace(), build=fail)
    assert not state._execution_active and not state._execution_failed
    assert state._begin_execution(SimpleNamespace(), build=object) is not None
