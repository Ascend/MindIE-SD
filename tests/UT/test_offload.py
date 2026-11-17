# pylint: disable=redefined-outer-name

# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch

from mindiesd.offload import enable_offload


class Block(torch.nn.Module):
    def __init__(self, sliced=False):
        super().__init__()
        values = torch.arange(12, dtype=torch.float32)
        self.weight = torch.nn.Parameter(values[::2] if sliced else values.clone())
        self.register_buffer("bias", torch.ones(3))

    def forward(self, value):
        return value + self.weight.sum() + self.bias.sum()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block(), Block(True), Block()])

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


@pytest.fixture
def transfer_boundary(monkeypatch):
    streams, events = [], []

    def stream():
        value = MagicMock()
        streams.append(value)
        return value

    def event():
        value = MagicMock()
        events.append(value)
        return value

    monkeypatch.setattr(torch.npu, "Stream", stream)
    monkeypatch.setattr(torch.npu, "Event", event)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    current = MagicMock()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current)
    module_to = torch.nn.Module.to

    def to(module, *args, **kwargs):
        if args == ("npu",):
            return module
        return module_to(module, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", to)
    empty = torch.empty

    def allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    return streams, events, current


@pytest.mark.parametrize("reserved", [0, 1, 2])
def test_offload_restores_real_cpu_storage_and_preserves_output(transfer_boundary, reserved):
    model = Model()
    value = torch.tensor(5.0)
    with torch.no_grad():
        expected = model(value)
        enable_offload(model, model.blocks, reserved)
        for index, block in enumerate(model.blocks):
            if index >= reserved:
                assert block.weight.untyped_storage().size() == 0
                assert block.weight.p_cpu.device.type == "cpu"
        for _ in range(2):
            torch.testing.assert_close(model(value), expected)
        assert model.blocks[2].weight.is_slice_tensor
        before = [len(block._forward_pre_hooks) for block in model.blocks]
        enable_offload(model, model.blocks, reserved)
        assert before == [len(block._forward_pre_hooks) for block in model.blocks]
    streams, events, current = transfer_boundary
    assert len(streams) == 2
    assert all(not event.record.called for event in events[:reserved])
    assert all(event.record.called for event in events[reserved:])
    assert current.wait_stream.called
    assert current.wait_event.called


@pytest.mark.parametrize(
    "model,blocks,reserved,error",
    [
        (object(), torch.nn.ModuleList([Block()]), 0, TypeError),
        (Model(), [], 0, TypeError),
        (Model(), torch.nn.ModuleList(), 0, ValueError),
        (Model(), torch.nn.ModuleList([Block()]), "1", TypeError),
        (Model(), torch.nn.ModuleList([Block()]), -1, ValueError),
        (Model(), torch.nn.ModuleList([Block()]), 1, ValueError),
    ],
)
def test_invalid_configuration_fails_before_device_access(model, blocks, reserved, error):
    with pytest.raises(error):
        enable_offload(model, blocks, reserved)
