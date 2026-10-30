# pylint: disable=redefined-outer-name

"""Dispatcher bookkeeping using CPU math and explicit native/communication boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mindiesd.layers.moe import token_dispatcher as module
from mindiesd.layers.moe.moe_dataclass import MoEPrepareInput
from mindiesd.utils import ParametersInvalid


@pytest.fixture
def cpu_boundaries(monkeypatch):
    cls = module.DynamicDispatcher
    monkeypatch.setattr(cls, "_split_cpu_buffers", {})
    monkeypatch.setattr(cls, "_split_copy_events", {})
    monkeypatch.setattr(module, "get_moe_group", lambda: None)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(module, "all_gather", lambda value, group: value.clone())
    monkeypatch.setattr(module, "all_to_all_single", lambda value, out, inp, group: value.clone())
    monkeypatch.setattr(torch.npu, "Event", lambda: SimpleNamespace(record=Mock(), synchronize=Mock()))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: "cpu-test-stream")
    empty = torch.empty

    def host_empty(*args, **kwargs):
        # Pinned allocation is a device-runtime boundary unavailable in CPU builds.
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)


def permute(tokens, indices, num_out_tokens=None):
    indices = indices.reshape(tokens.shape[0], -1)
    assert torch.equal(indices, indices.round())
    order = indices.reshape(-1).argsort(stable=True)
    rows = order // indices.shape[1]
    return tokens[rows], order


def unpermute(permuted_tokens, sorted_indices, probs=None, restore_shape=None):
    restored = torch.empty_like(permuted_tokens)
    restored[sorted_indices.long()] = permuted_tokens
    if probs is not None:
        restored = restored.reshape(probs.shape[0], probs.shape[1], -1)
        restored = (restored * probs.unsqueeze(-1)).sum(1)
    return restored


@pytest.mark.parametrize("quant_shape", [None, "vector", "matrix"])
@pytest.mark.parametrize("experts", [1, 2])
def test_dynamic_dispatch_combine_roundtrip(cpu_boundaries, monkeypatch, quant_shape, experts):
    monkeypatch.setattr(module.torch_npu, "npu_moe_token_permute", permute)
    monkeypatch.setattr(module.torch_npu, "npu_moe_token_unpermute", unpermute)
    monkeypatch.setattr(module, "is_moe_int_quant", lambda: quant_shape is not None)

    def quantize(value):
        scale = torch.full((value.shape[0],), 2.0)
        if quant_shape == "matrix":
            scale = scale[:, None]
        return value / 2, scale

    monkeypatch.setattr(module, "dynamic_quant", quantize)
    hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    # Integral FP32 indices let CPU histc execute its real arithmetic. Native
    # integer-index support is covered in the separate real operator suite.
    ids = (torch.arange(5) % experts).float().reshape(5, 1)
    request = SimpleNamespace(
        hidden_states=hidden,
        topk_ids=ids,
        topk_weights=torch.ones(5, 1),
        num_experts=experts,
        local_num_experts=experts,
    )
    dispatched = module.DynamicDispatcher.dispatch(request)
    assert dispatched.group_list.tolist() == ([5.0] if experts == 1 else [3.0, 2.0])
    assert dispatched.group_list_type == 1
    restored = module.DynamicDispatcher.combine(dispatched.hidden_states, dispatched.combine_metadata)
    torch.testing.assert_close(restored, hidden if quant_shape is None else hidden / 2)
    if quant_shape is not None:
        assert torch.equal(dispatched.dynamic_scale, torch.full_like(dispatched.dynamic_scale, 2))
        assert dispatched.dynamic_scale.ndim == (1 if quant_shape == "vector" else 2)
    else:
        assert dispatched.dynamic_scale is None
    event = module.DynamicDispatcher._get_split_copy_event(ids.device)
    event.record.assert_called_once_with("cpu-test-stream")
    event.synchronize.assert_called_once_with()


def test_dynamic_preprocess_counts_splits_and_reuses_buffers(cpu_boundaries, monkeypatch):
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 1)
    peer_counts = torch.tensor([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(module, "all_gather", lambda counts, group: torch.cat((peer_counts, counts)))
    ids = torch.tensor([[0.0, 2.0], [2.0, 3.0]])
    inp, out, counts, event, total = module.DynamicDispatcher._preprocess(ids, 4, 2, None)
    assert inp.tolist() == [1, 3]
    assert out.tolist() == [7, 3]
    assert counts.tolist() == [[3.0, 4.0], [2.0, 1.0]]
    assert total == ids.numel()
    assert module.DynamicDispatcher._get_split_cpu_buffer(2, "input") is inp
    assert module.DynamicDispatcher._get_split_copy_event(ids.device) is event
    changed = module.DynamicDispatcher._copy_split_sizes_to_cpu(torch.tensor([4, 5]), 2, "input")
    assert changed is inp and changed.tolist() == [4, 5]
    with pytest.raises(ParametersInvalid, match="evenly partitioned"):
        module.DynamicDispatcher._preprocess(ids, 5, 2, None)


@pytest.mark.parametrize("sharded", [False, True])
def test_prepare_and_finalize_cpu_shapes(cpu_boundaries, monkeypatch, sharded):
    hidden = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    router = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    request = MoEPrepareInput(hidden_states=hidden, router_logits=router, inputs_sharded=sharded)
    result = module.DynamicDispatcher.prepare(request)
    torch.testing.assert_close(result.hidden_states, hidden.reshape(6, 3))
    restored = module.DynamicDispatcher.finalize(result.hidden_states, result.original_shape, inputs_sharded=sharded)
    torch.testing.assert_close(restored, hidden)
    static = module.StaticDispatcher.prepare(request)
    torch.testing.assert_close(static.hidden_states, hidden.reshape(6, 3))
    torch.testing.assert_close(module.StaticDispatcher.finalize(static.hidden_states, static.original_shape), hidden)
