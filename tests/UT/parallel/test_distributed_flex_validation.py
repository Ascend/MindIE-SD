#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Check coordinated validation using bounded CPU collective participation."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
from types import SimpleNamespace

import pytest
import torch

parallel = importlib.import_module("mindiesd.parallel.distributed_flex_attention")
flex = importlib.import_module("mindiesd.layers.flash_attn.flex_attention")


class _Collectives:
    """Synchronize real validation code across CPU threads, without a backend."""

    def __init__(self):
        self.local = threading.local()
        self.condition = threading.Condition()
        self.counts = {}
        self.rounds = {}

    def reduce(self, tensor, op, group):
        rank = self.local.rank
        with self.condition:
            counter = rank, group
            step = self.counts.get(counter, 0)
            self.counts[counter] = step + 1
            state = self.rounds.setdefault((group, step), {"values": {}, "ready": False})
            state["values"][rank] = tensor
            if len(state["values"]) == len(group):
                values = torch.stack(list(state["values"].values()))
                result = values.amin(0) if op == parallel.dist.ReduceOp.MIN else values.amax(0)
                for value in state["values"].values():
                    value.copy_(result)
                state["ready"] = True
                self.condition.notify_all()
            if not self.condition.wait_for(lambda: state["ready"], timeout=2):
                raise TimeoutError("a peer left before the matching validation collective")


@pytest.mark.parametrize("invalid", ["map", "import"])
def test_local_failure_reaches_every_rank_before_payload_exchange(monkeypatch, invalid):
    communication = _Collectives()
    ulysses_size, ring_size = 2, 2
    world_size = ulysses_size * ring_size
    rows = [tuple(range(r * ulysses_size, (r + 1) * ulysses_size)) for r in range(ring_size)]
    columns = [tuple(r * ulysses_size + u for r in range(ring_size)) for u in range(ulysses_size)]
    monkeypatch.setattr(flex, "check_flex_attention_device", lambda device: None)
    monkeypatch.setattr(parallel, "_group_size", lambda group: len(group) if group is not None else 1)
    monkeypatch.setattr(parallel, "_group_rank", lambda group: group.index(communication.local.rank) if group else 0)
    monkeypatch.setattr(parallel.dist, "all_reduce", communication.reduce)

    def load(device):
        if invalid == "import" and communication.local.rank == 0:
            raise ImportError("missing optional DSL dependency")
        return SimpleNamespace()

    def unexpected(*args, **kwargs):
        raise AssertionError("payload/shape exchange started before all validation errors were propagated")

    monkeypatch.setattr(parallel, "_load_dsl", load)
    monkeypatch.setattr(parallel.dist, "all_gather", unexpected)
    monkeypatch.setattr(parallel.dist, "all_to_all_single", unexpected)

    def worker(rank):
        communication.local.rank = rank
        query = torch.zeros(1, 3, 4, 128, dtype=torch.bfloat16)
        key = value = query
        mask = parallel.FlexAttentionMask()
        arguments = {
            "mask": mask,
            "ulysses_group": rows[rank // ulysses_size],
            "ring_group": columns[rank % ulysses_size],
        }
        if rank == 0 and invalid == "map":
            arguments["query_positions"] = torch.arange(2)
        try:
            with torch.no_grad():
                parallel.distributed_flex_attention(query, key, value, **arguments)
        except Exception as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=world_size) as executor:
        errors = list(executor.map(worker, range(world_size)))
    assert all(isinstance(error, (ValueError, TypeError, ImportError)) for error in errors), errors
    assert all("another rank" in str(error) for error in errors[1:]), errors


def test_peer_local_error_is_not_hidden_by_a_cached_plan(monkeypatch):
    group = object()
    monkeypatch.setattr(parallel, "_group_size", lambda selected: 2 if selected is group else 1)
    monkeypatch.setattr(parallel, "_group_rank", lambda selected: 0)
    query = torch.zeros(1, 3, 4, 128, dtype=torch.bfloat16)
    mask = parallel.FlexAttentionMask()

    def reduce(tensor, op, group):
        tensor[0] = 0

    monkeypatch.setattr(parallel.dist, "all_reduce", reduce)
    plan = (
        group,
        None,
        tuple(query.shape),
        tuple(query.shape),
        tuple(query.shape),
        query.dtype,
        str(query.device),
        0,
        False,
    )
    mask._validated_plans.add(plan)
    with pytest.raises(ValueError, match="another rank"):
        parallel._validate_peer_contracts(mask, query, query, query, 0, group, None)
