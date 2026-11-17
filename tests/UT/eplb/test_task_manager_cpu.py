"""Worker control flow with real queues and handlers, without device/IPC startup."""

import queue
import threading
from multiprocessing.connection import AuthenticationError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mindiesd.eplb import task_manager as module
from mindiesd.eplb.task_payload import TaskPayload, TaskType
from mindiesd.utils.exception import ParametersInvalid


def test_worker_auth_precedence(monkeypatch):
    monkeypatch.setenv("EPLB_AUTH_KEY", "environment-secret")
    assert module._resolve_worker_auth_key("explicit-secret") == "explicit-secret"
    assert module._resolve_worker_auth_key(None) == "environment-secret"
    monkeypatch.delenv("EPLB_AUTH_KEY")
    monkeypatch.setattr(module.secrets, "token_hex", lambda count: "generated-secret")
    assert module._resolve_worker_auth_key(None) == "generated-secret"


@pytest.mark.parametrize("error", [None, AuthenticationError("denied"), OSError("unavailable")])
def test_worker_connection_returns_rank_queues_or_propagates(monkeypatch, error):
    incoming, outgoing = queue.Queue(), queue.Queue()
    manager = SimpleNamespace(
        connect=Mock(side_effect=error),
        get_instruction_queues=Mock(return_value=incoming),
        get_upload_queues=Mock(return_value=outgoing),
    )
    factory = Mock(return_value=manager)
    monkeypatch.setattr(module, "get_manager_client", factory)
    if error:
        with pytest.raises(type(error), match=str(error)):
            module.connect_to_schedule_manager(3, "localhost", 51001, "test-key")
        manager.get_instruction_queues.assert_not_called()
    else:
        assert module.connect_to_schedule_manager(3, "localhost", 51001, "test-key") == (incoming, outgoing)
        manager.get_instruction_queues.assert_called_once_with(rank=3)
        manager.get_upload_queues.assert_called_once_with(rank=3)
    factory.assert_called_once_with(("localhost", 51001), "test-key")


@pytest.mark.parametrize("sentinel", [None, "exit"])
def test_pool_profiles_real_collector_and_ignores_foreign_instruction(monkeypatch, sentinel):
    incoming, outgoing = queue.Queue(), queue.Queue()
    collector = SimpleNamespace(lb_interval=3, get_expert_load=lambda: torch.tensor([3, 8]))
    dispatcher = SimpleNamespace(update_lock=threading.Lock(), local_expert_list=[1, 0])
    child = SimpleNamespace(dispatcher=dispatcher, expert_load_collector=collector)
    model = SimpleNamespace(named_modules=lambda: [("root", object()), ("moe", child)])
    monkeypatch.setattr(module.torch_npu.npu, "Stream", lambda device: object())
    incoming.put("foreign-object")
    incoming.put(TaskPayload(TaskType.PROFILE, moe_layer_idx=0))
    incoming.put(sentinel)
    module.expert_info_transfer_pool(model, incoming, outgoing, "cpu")
    report = outgoing.get_nowait()
    assert report["moe_layer_idx"] == 0
    assert report["load"].tolist() == [3, 8]
    assert report["local_expert_list"] == [1, 0]
    assert outgoing.empty() and incoming.empty()
    assert collector.task_transfer is not None


def test_pool_rejects_unknown_task_with_original_handler(monkeypatch):
    incoming = queue.Queue()
    instruction = TaskPayload(TaskType.PROFILE)
    instruction.task_type = "unsupported"
    incoming.put(instruction)
    model = SimpleNamespace(named_modules=lambda: [])
    monkeypatch.setattr(module.torch_npu.npu, "Stream", lambda device: object())
    with pytest.raises(ParametersInvalid, match="Unknown task type"):
        module.expert_info_transfer_pool(model, incoming, queue.Queue(), "cpu")


@pytest.mark.parametrize("missing", [False, True])
def test_construct_worker_uses_resolved_auth_and_thread_boundary(monkeypatch, missing):
    incoming, outgoing = queue.Queue(), queue.Queue()
    connect = Mock(return_value=(None, outgoing) if missing else (incoming, outgoing))
    monkeypatch.setattr(module, "connect_to_schedule_manager", connect)
    thread = SimpleNamespace(start=Mock())
    factory = Mock(return_value=thread)
    monkeypatch.setattr(module.threading, "Thread", factory)
    model = object()
    result = module.construct_expert_info_transfer_pool(
        module=model, rank_in_group=1, device="cpu", ip="localhost", port=51001, auth_key="test-key"
    )
    connect.assert_called_once_with(1, "localhost", 51001, "test-key")
    if missing:
        assert result == (None, None)
        factory.assert_not_called()
    else:
        assert result == (thread, incoming)
        factory.assert_called_once_with(
            target=module.expert_info_transfer_pool, args=(model, incoming, outgoing, "cpu"), daemon=True
        )
        thread.start.assert_called_once_with()
