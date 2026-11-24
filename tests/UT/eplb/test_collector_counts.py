# pylint: disable=redefined-outer-name

"""Contract of the count-based collection path.

`collect_expert_counts` exists so the MoE hot path can skip sorting the routing
tensor: it accepts a cumulative histogram (what `torch.histc` + `torch.cumsum`
produce) instead of a sorted index list. The accumulation and the dtype conversion
are what the forward depends on, so they are pinned here.

The collector was written for the device: it pins a host buffer and the index-based
path calls a CANN operator. Both are stubbed so the contract can run on a CPU-only
machine.
"""

from unittest.mock import MagicMock

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from mindiesd.eplb.collector import ExpertLoadCollector  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_collector(monkeypatch):
    """Replace the device-only pieces with CPU equivalents."""
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *args, **kwargs: self)

    def fake_expert_tokens(indices, num_experts):
        counts = torch.bincount(indices.long(), minlength=num_experts)
        return torch.cumsum(counts, 0)

    monkeypatch.setattr(torch_npu, "npu_moe_compute_expert_tokens", fake_expert_tokens)


def make_collector(num_experts=4):
    return ExpertLoadCollector(num_experts, lb_interval=1)


def test_accumulates_cumulative_counts_as_long():
    collector = make_collector()

    # torch.histc returns float32, the buffer is int64.
    collector.collect_expert_counts(torch.tensor([1.0, 2.0, 2.0, 4.0]))

    assert collector.expert_group_list.dtype == torch.long
    assert collector.expert_group_list.tolist() == [1, 2, 2, 4]


def test_accumulates_across_calls():
    collector = make_collector()

    collector.collect_expert_counts(torch.tensor([1.0, 2.0, 2.0, 4.0]))
    collector.collect_expert_counts(torch.tensor([0.0, 1.0, 1.0, 2.0]))

    assert collector.expert_group_list.tolist() == [1, 3, 3, 6]


def test_emits_the_profile_task_when_attached():
    collector = make_collector()
    collector.task_transfer = MagicMock()

    collector.collect_expert_counts(torch.tensor([1.0, 1.0, 1.0, 1.0]))

    collector.task_transfer.profile_emit_task.assert_called_once()


def test_matches_the_index_based_path_for_the_same_routing():
    """The fast path and the sorted-index path must report the same counts."""
    indices = torch.tensor([3, 0, 1, 1, 2, 1], dtype=torch.int32)
    counts = torch.bincount(indices.long(), minlength=4).to(torch.float32)
    cumulative = torch.cumsum(counts, 0)

    by_counts = make_collector()
    by_counts.collect_expert_counts(cumulative)

    by_indices = make_collector()
    by_indices.collect_expert_load(torch.sort(indices).values)

    assert by_counts.expert_group_list.tolist() == by_indices.expert_group_list.tolist()
