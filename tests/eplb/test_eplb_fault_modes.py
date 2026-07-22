#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import queue
import socket
import sys
import types
import unittest
from importlib import util
from multiprocessing.connection import AuthenticationError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch_npu  # noqa: F401
except Exception:
    sys.modules["torch_npu"] = mock.MagicMock()

ROOT = Path(__file__).resolve().parents[2]


def _ensure_package(package_name, package_path):
    package = sys.modules.get(package_name)
    if package is not None:
        return package
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package
    return package


def _load_module(module_name, relative_path):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = util.spec_from_file_location(module_name, ROOT / relative_path)
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("mindiesd", ROOT / "mindiesd")
_ensure_package("mindiesd.utils", ROOT / "mindiesd" / "utils")
_ensure_package("mindiesd.utils.logs", ROOT / "mindiesd" / "utils" / "logs")
_ensure_package("mindiesd.eplb", ROOT / "mindiesd" / "eplb")

_load_module("mindiesd.utils.env", "mindiesd/utils/env.py")
_load_module("mindiesd.utils.logs.logging", "mindiesd/utils/logs/logging.py")
exception_module = _load_module("mindiesd.utils.exception", "mindiesd/utils/exception.py")
_load_module("mindiesd.eplb.task_payload", "mindiesd/eplb/task_payload.py")
_load_module("mindiesd.eplb.task_transfer", "mindiesd/eplb/task_transfer.py")
greedy_module = _load_module("mindiesd.eplb.greedy_algorithm", "mindiesd/eplb/greedy_algorithm.py")
eplb_scheduler = _load_module("mindiesd.eplb.eplb_scheduler", "mindiesd/eplb/eplb_scheduler.py")
task_handler_module = _load_module("mindiesd.eplb.task_handler", "mindiesd/eplb/task_handler.py")
task_manager_module = _load_module("mindiesd.eplb.task_manager", "mindiesd/eplb/task_manager.py")

SchedulerContext = eplb_scheduler.SchedulerContext
ProfileTaskTransfer = sys.modules["mindiesd.eplb.task_transfer"].ProfileTaskTransfer
handle_unknown_task = task_handler_module.handle_unknown_task
connect_to_schedule_manager = task_manager_module.connect_to_schedule_manager
A2ARedundantExpertService = greedy_module.A2ARedundantExpertService
ExpertExchangeService = greedy_module.ExpertExchangeService
LoadData = greedy_module.LoadData
eplb_greedy = greedy_module.eplb_greedy
ModelExecError = exception_module.ModelExecError
ParametersInvalid = exception_module.ParametersInvalid


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _scheduler_args(**overrides):
    args = {
        "world_size": 2,
        "host": "127.0.0.1",
        "port": _get_free_port(),
        "expert_num": 4,
        "block_num": 1,
        "max_move": 1,
        "redundant": 0,
        "mode": "A2A",
        "auth_key": "secret_key",
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class TestEplbFaultModeReproduction(unittest.TestCase):
    def setUp(self):
        eplb_scheduler.upload_queues.clear()
        eplb_scheduler.instruction_queues.clear()

    def tearDown(self):
        eplb_scheduler.upload_queues.clear()
        eplb_scheduler.instruction_queues.clear()

    def test_fm001_fm003_scheduler_not_running_causes_worker_connection_failure(self):
        port = _get_free_port()

        with self.assertRaises((ConnectionRefusedError, OSError)):
            connect_to_schedule_manager(0, "127.0.0.1", port, "secret_key")

    def test_fm002_scheduler_port_unavailable(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]

            with self.assertRaises(OSError):
                eplb_scheduler._init_scheduler_context(_scheduler_args(port=port))

    def test_fm004_auth_key_mismatch_rejects_worker(self):
        args = _scheduler_args()
        eplb_scheduler._init_scheduler_context(args)

        with self.assertRaises((AuthenticationError, OSError)):
            connect_to_schedule_manager(0, args.host, args.port, "wrong_key")

    def test_fm005_profile_task_enqueue_failed_when_instruction_queue_full(self):
        instruction_queue = queue.Queue(maxsize=1)
        instruction_queue.put_nowait("occupied")
        transfer = ProfileTaskTransfer(instruction_queue, moe_layer_idx=0, lb_interval=1)

        with self.assertLogs("mindie-sd", level="WARNING") as captured:
            transfer.profile_emit_task()

        self.assertIn("EPLB profile task enqueue failed", "\n".join(captured.output))

    def test_fm006_scheduler_report_missing_field_raises_model_exec_error(self):
        eplb_scheduler.upload_queues[0] = queue.Queue()
        eplb_scheduler.instruction_queues[0] = queue.Queue()
        eplb_scheduler.upload_queues[0].put_nowait({"moe_layer_idx": 0})
        context = SchedulerContext(
            scheduler_args=_scheduler_args(),
            world_size=1,
            redundant=0,
            experts_set=set(range(4)),
            experts_per_rank=4,
            load_report_buffer={0: {}},
            local_expert_buffer={0: {}},
        )

        with self.assertRaisesRegex(ModelExecError, "EPLB scheduler failed"):
            eplb_scheduler._process_rank_report(context, 0)

    def test_fm007_expert_initial_placement_failed_when_capacity_is_insufficient(self):
        response = {
            0: np.array([1, 2, 3], dtype=np.int64),
            1: np.array([1, 2, 3], dtype=np.int64),
        }

        with self.assertRaises(MemoryError):
            eplb_greedy(
                response=response,
                algorithm_type="A2A",
                device_to_expert={0: [0], 1: [1]},
                world_size=2,
                expert_num=3,
                max_move=1,
                redundant=0,
            )

    def test_fm008_shared_expert_placement_failed_when_device_memory_is_insufficient(self):
        service = A2ARedundantExpertService(
            num_devices=2,
            num_experts=2,
            expert_mems={0: 2.0, 1: 1.0},
            device_mems={0: 1.0, 1: 1.0},
            cost_local=1,
            cost_remote=10,
            max_move_number=1,
            load_balance_threshold=100,
        )
        placement = np.zeros((2, 2), dtype=int)
        used_mems = {0: 0.0, 1: 0.0}

        with self.assertRaises(MemoryError):
            service.process_share_expert(placement, shared_expert_id=0, used_mems=used_mems)

    def test_fm009_expert_exchange_state_inconsistent_when_rank_layout_is_missing(self):
        service = ExpertExchangeService(
            num_devices=2,
            num_experts=4,
            expert_mems={idx: 1.0 for idx in range(4)},
            device_mems={0: 2.0, 1: 2.0},
            cost_local=1,
            cost_remote=10,
            max_move_number=1,
            load_balance_threshold=0,
        )
        load_data = LoadData(
            placement=np.zeros((2, 4), dtype=int),
            shared_expert_id=None,
            total_traffic=np.array([[10, 9, 1, 1], [1, 1, 10, 9]], dtype=np.int64),
            used_mems={0: 0.0, 1: 0.0},
            origin_device_to_expert={0: [0, 1]},
            expert_trans_tensor=None,
        )

        with self.assertRaises(KeyError):
            service.initial_placement(load_data)

    def test_fm010_layout_not_updated_when_greedy_reports_no_update(self):
        context = SchedulerContext(
            scheduler_args=_scheduler_args(),
            world_size=1,
            redundant=0,
            experts_set=set(range(4)),
            experts_per_rank=4,
            load_report_buffer={0: {0: np.array([0, 0, 0, 0], dtype=np.int64)}},
            local_expert_buffer={0: {0: [0, 1, 2, 3]}},
        )
        transfer = mock.Mock()

        with mock.patch(
            "mindiesd.eplb.eplb_scheduler.eplb_greedy",
            return_value=(False, [], [], [], None),
        ):
            eplb_scheduler._emit_layer_update(context, 0, transfer)

        transfer.update_emit_task.assert_not_called()
        self.assertEqual(context.update_count, 0)

    def test_fm011_unknown_task_instruction_raises_parameters_invalid(self):
        instruction = SimpleNamespace(task_type="UNKNOWN")

        with self.assertRaisesRegex(ParametersInvalid, "Unknown task type"):
            handle_unknown_task(instruction, None, None, None, None)


if __name__ == "__main__":
    unittest.main()
