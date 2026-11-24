# pylint: disable=redefined-outer-name

"""Expert-map routing contracts for the dynamic MoE dispatcher.

These tests pin the arithmetic a re-placement relies on: the all-to-all chunks must
be ordered by (destination rank, local row), and the received order must match the
local expert column order. They run on CPU with the native/communication boundaries
stubbed out, exactly like `test_token_dispatcher_cpu_paths.py`.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mindiesd.layers.moe import token_dispatcher as module
from mindiesd.layers.moe.moe_dataclass import MoETokenDispatchInput
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
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)


def permute(tokens, indices, num_out_tokens=None):
    indices = indices.reshape(tokens.shape[0], -1)
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


def two_rank(monkeypatch, ep_size=2, ep_rank=0):
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: ep_size)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: ep_rank)


def gather_two_ranks(monkeypatch, rank0_counts, rank1_counts):
    def fake_all_gather(value, group):
        return torch.cat([rank0_counts, rank1_counts])

    monkeypatch.setattr(module, "all_gather", fake_all_gather)


def test_preprocess_map_branch_orders_owned_columns_by_global_expert_id(cpu_boundaries, monkeypatch):
    """The published map decides ownership; columns then follow ascending global id."""
    two_rank(monkeypatch)
    num_experts = 4
    # device_map[e] is the rank holding expert e: expert 0 -> rank 1, expert 1 ->
    # rank 0, expert 2 -> rank 1, expert 3 -> rank 0. Rank 0 owns {1, 3}, and its
    # columns are ordered by ascending id, i.e. [expert 1, expert 3].
    device_map = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    rank0_counts = torch.tensor([1.0, 2.0, 0.0, 3.0])
    rank1_counts = torch.tensor([0.0, 5.0, 4.0, 0.0])
    gather_two_ranks(monkeypatch, rank0_counts, rank1_counts)

    # Integral FP32 indices keep CPU histc on its real arithmetic path.
    topk_ids = torch.tensor([[0, 3], [0, 1]], dtype=torch.float32)
    input_splits, output_splits, counts, _, _ = module.DynamicDispatcher._preprocess(
        topk_ids=topk_ids,
        num_experts=num_experts,
        local_num_experts=2,
        ep_group=object(),
        device_indices_map=device_map,
    )

    # This rank sends expert 1 and expert 3 to rank 0, experts 0 and 2 to rank 1.
    assert input_splits.tolist() == [2, 2]
    # Columns follow ascending global expert id: rank 0 owns experts 1 and 3.
    assert counts.tolist() == [[2.0, 3.0], [5.0, 0.0]]
    assert output_splits.tolist() == [5, 5]


def test_dispatch_orders_tokens_by_destination_rank_then_local_row(cpu_boundaries, monkeypatch):
    """The permute index must encode (destination rank, local row)."""
    two_rank(monkeypatch)
    captured = {}

    def capturing_permute(tokens, indices, num_out_tokens=None):
        captured["indices"] = indices.clone()
        return permute(tokens, indices, num_out_tokens)

    monkeypatch.setattr(module.torch_npu, "npu_moe_token_permute", capturing_permute)
    monkeypatch.setattr(module.torch_npu, "npu_moe_token_unpermute", unpermute)
    monkeypatch.setattr(module, "is_moe_int_quant", lambda: False)

    num_experts = 4
    device_map = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    counts = torch.tensor([1.0, 1.0, 1.0, 1.0])
    gather_two_ranks(monkeypatch, counts, counts)

    # Exercise the preprocess stage directly: it owns the outgoing token order, and
    # the downstream all-to-all is replaced by a stub in this CPU harness.
    module.DynamicDispatcher._dispatch_preprocess(
        hidden_states=torch.zeros(2, 3),
        topk_ids=torch.tensor([[0, 3], [1, 2]], dtype=torch.float32),
        num_experts=num_experts,
        local_num_experts=2,
        ep_group=object(),
        device_indices_map=device_map,
    )

    # Ordering key = destination rank * L + position in that rank's ascending id list.
    # rank0 owns {1, 3} -> 1:0, 3:1 ; rank1 owns {0, 2} -> 0:0, 2:1
    # expert 0 -> rank1 pos0 -> 2 ; expert 3 -> rank0 pos1 -> 1 ; expert 1 -> rank0 pos0 -> 0
    assert captured["indices"].tolist() == [[2, 1], [0, 3]]


def test_preprocess_accepts_the_device_map_alone(cpu_boundaries, monkeypatch):
    """Only the device map matters: the row order is ascending expert id."""
    two_rank(monkeypatch)
    gather_two_ranks(monkeypatch, torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 6.0, 7.0, 8.0]))

    input_splits, output_splits, counts, _, _ = module.DynamicDispatcher._preprocess(
        topk_ids=torch.tensor([[0, 3]], dtype=torch.float32),
        num_experts=4,
        local_num_experts=2,
        ep_group=object(),
        device_indices_map=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
    )
    # rank 0 owns experts {1, 3}; the columns are taken in ascending id order.
    assert input_splits.tolist() == [1, 1]
    assert counts.tolist() == [[2.0, 4.0], [6.0, 8.0]]
    assert output_splits.tolist() == [6, 14]


def test_legacy_path_is_untouched_when_maps_are_absent(cpu_boundaries, monkeypatch):
    """Even partition without maps keeps the historical splits."""
    two_rank(monkeypatch)
    monkeypatch.setattr(module, "all_gather", lambda value, group: torch.cat([value, value]))

    # Integral FP32 indices keep CPU histc on its real arithmetic path.
    topk_ids = torch.tensor([[0, 3], [0, 1]], dtype=torch.float32)
    input_splits, output_splits, counts, _, _ = module.DynamicDispatcher._preprocess(
        topk_ids=topk_ids,
        num_experts=4,
        local_num_experts=2,
        ep_group=object(),
    )
    assert input_splits.tolist() == [3, 1]
    assert counts.tolist() == [[2.0, 1.0], [2.0, 1.0]]
    assert output_splits.tolist() == [3, 3]
