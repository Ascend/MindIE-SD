# pylint: disable=redefined-outer-name

# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import zmq

import mindiesd.share_memory as sm
from mindiesd.utils.exception import ParametersInvalid
from mindiesd.utils.safe_pickle import safe_dumps, safe_loads


@pytest.fixture(autouse=True)
def transport(monkeypatch):
    context = MagicMock()
    monkeypatch.setattr(sm, 'ZMQ_CONTEXT', context)
    monkeypatch.setattr(sm, 'manager', None)
    return context


@pytest.mark.parametrize('rank', [0, 1])
def test_manager_broadcast_roundtrip(transport, rank):
    manager = sm.init_share_memory(2, rank, '127.0.0.1', 8110)
    assert sm.get_share_memory_manager() is manager
    assert manager.is_master == (rank == 0)
    socket = transport.socket.return_value
    socket.recv.return_value = safe_dumps((23, 42))
    result = manager.broadcast_handle((23, 42) if rank == 0 else None)
    assert result == (23, 42)
    if rank == 0:
        assert safe_loads(socket.send.call_args.args[0]) == (23, 42)
    else:
        socket.send.assert_called_once_with(b'READY')
    old_close_count = socket.close.call_count
    sm.init_share_memory(2, rank)
    assert socket.close.call_count > old_close_count
    sm.ShareMemoryManager.cleanup()
    assert sm.manager is None
    assert sm.ZMQ_CONTEXT is None
    transport.term.assert_called_once()


@pytest.mark.parametrize('rank', [0, 1])
def test_transport_timeout_is_explicit(transport, rank):
    manager = sm.ShareMemoryManager(2, rank)
    transport.socket.return_value.recv.side_effect = [b'ACK', zmq.Again()] if rank else zmq.Again()
    with pytest.raises(TimeoutError):
        manager.broadcast_handle(None)


def test_cleanup_failure_still_releases_socket_references(transport):
    manager = sm.ShareMemoryManager(2, 0)
    transport.socket.return_value.close.side_effect = OSError('closed transport')
    manager.close()
    assert manager.rep_socket is None and manager.pub_socket is None
    sm.ShareMemoryManager.cleanup()
    with pytest.raises(ParametersInvalid, match='not been initialized'):
        sm.get_share_memory_manager()


@pytest.mark.parametrize('device', [None, 'cpu', 'meta'])
def test_real_cpu_or_meta_fallback(device):
    model = torch.nn.Linear(3, 2)
    result = sm.share_memory(model, device=device, dtype=torch.float64)
    assert result is model
    assert model.weight.dtype == torch.float64
    assert model.weight.device.type == ('meta' if device == 'meta' else 'cpu')


class DeviceTorch:
    """Local adapter for backend-only device and shared-handle interfaces."""

    def __init__(self, token, rebuild):
        self.token = token
        self.UntypedStorage = SimpleNamespace(_new_shared_npu=rebuild)

    def device(self, name):
        return self.token if str(name).startswith('npu') else torch.device(name)

    def empty_like(self, tensor, device=None):
        assert device is self.token
        return torch.empty_like(tensor, device='cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def device_boundary(monkeypatch):
    token = object()
    stored = torch.arange(6, dtype=torch.float32)
    rebuild = MagicMock(return_value=stored.untyped_storage())
    monkeypatch.setattr(sm, 'torch', DeviceTorch(token, rebuild))
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if kwargs.get('device') is token:
            kwargs['device'] = torch.device('cpu')
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', to)
    monkeypatch.setattr(torch.storage.TypedStorage, '_share_npu_', lambda storage: (17,), raising=False)
    monkeypatch.setattr(torch.npu, 'empty_cache', MagicMock())
    return token, stored, rebuild


def test_device_validation_without_device_allocation(device_boundary):
    token, _, _ = device_boundary
    model = torch.nn.Linear(3, 2)
    with pytest.raises(ParametersInvalid, match='floating point or complex'):
        sm._check_device_and_dtype(model, token, torch.int32)
    already_on_device = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=token)]))
    fallback, result, _, _ = sm._check_device_and_dtype(already_on_device, token, None)
    assert fallback and result is already_on_device
    with pytest.raises(ParametersInvalid):
        sm.share_memory(object(), 'cpu')


@pytest.mark.parametrize('rank', [0, 1])
def test_handle_exchange_preserves_real_cpu_tensor_values(monkeypatch, device_boundary, rank):
    token, stored, rebuild = device_boundary
    manager = SimpleNamespace(instance_id=rank, broadcast_handle=MagicMock(return_value=(17,)))
    monkeypatch.setattr(sm, 'manager', manager)
    model = torch.nn.Linear(3, 2, bias=False)
    before = model.weight.detach().clone()
    assert sm.share_memory(model, device=token) is model
    manager.broadcast_handle.assert_called_once_with((17,) if rank == 0 else None)
    if rank == 0:
        torch.testing.assert_close(model.weight, before)
        rebuild.assert_not_called()
    else:
        rebuild.assert_called_once_with(17)
        torch.testing.assert_close(model.weight, stored.reshape(2, 3))
    torch.npu.empty_cache.assert_called_once()
