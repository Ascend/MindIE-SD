"""Scheduler startup and monitor boundaries, retaining real queue processing."""

import argparse
import queue
from unittest.mock import Mock

import pytest

from mindiesd.eplb import eplb_scheduler as module


def arguments():
    return argparse.Namespace(
        world_size=2,
        host="localhost",
        port=51001,
        expert_num=4,
        block_num=2,
        max_move=1,
        redundant=0,
        mode="A2A",
        auth_key="test-key",
    )


def test_context_initializes_per_rank_queues_and_per_layer_buffers(monkeypatch):
    monkeypatch.setattr(module, "upload_queues", {})
    monkeypatch.setattr(module, "instruction_queues", {})
    server = Mock()
    monkeypatch.setattr(module, "start_manager_server", server)
    args = arguments()
    context = module._init_scheduler_context(args)
    server.assert_called_once_with(("localhost", 51001), "test-key")
    assert context.experts_set == {0, 1, 2, 3}
    assert context.experts_per_rank == 2 and context.update_count == 0
    assert context.load_report_buffer == {0: {}, 1: {}}
    assert context.local_expert_buffer == {0: {}, 1: {}}
    assert set(module.upload_queues) == set(module.instruction_queues) == {0, 1}
    assert module.upload_queues[0] is not module.upload_queues[1]
    assert module._process_rank_report(context, 0)


@pytest.mark.parametrize("exit_signal", [KeyboardInterrupt, SystemExit])
def test_monitor_sleeps_for_empty_queues_and_stops_on_signal(monkeypatch, exit_signal):
    monkeypatch.setattr(module, "upload_queues", {})
    monkeypatch.setattr(module, "instruction_queues", {})
    monkeypatch.setattr(module, "start_manager_server", Mock())
    original_queue = queue.Queue

    class ControlledQueue(original_queue):
        def __init__(self):
            super().__init__()
            self.polls = 0

        def get_nowait(self):
            self.polls += 1
            if self.polls > 1:
                raise exit_signal()
            return super().get_nowait()

    monkeypatch.setattr(module.queue, "Queue", ControlledQueue)
    sleep = Mock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    module.run_scheduler(arguments())
    sleep.assert_called_once_with(0.1)
    assert module.upload_queues[0].polls == 2
    assert module.upload_queues[1].polls == 1
