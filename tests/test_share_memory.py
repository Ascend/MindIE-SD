#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
import sys
import os
import zmq
import torch
import torch.nn as nn
import torch_npu
from unittest.mock import Mock, patch, MagicMock, call, ANY
import logging

sys.path.append('../')
try:
    from device import DEVICE_ID
except ImportError:
    DEVICE_ID = 0

import mindiesd.share_memory as msm
mock_zmq_ctx = MagicMock(spec=zmq.Context)
msm.ZMQ_CONTEXT = mock_zmq_ctx

from mindiesd.share_memory import (
    ShareMemoryManager, init_share_memory, get_share_memory_manager, share_memory,
    _check_device_and_dtype, ZMQ_CONTEXT, manager as global_manager
)

logging.disable(logging.CRITICAL)
os.environ["ZMQ_DISABLE_IPV6"] = "1"
os.environ["ASCEND_SIMULATOR"] = "1"


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestShareMemoryManager(unittest.TestCase):

    def setUp(self):
        global global_manager
        self.original_manager = global_manager
        global_manager = None

        self.mock_socket = MagicMock()
        self.mock_socket.setsockopt.return_value = None
        self.mock_socket.send_pyobj.return_value = None
        self.mock_socket.recv_pyobj.return_value = 123456
        mock_zmq_ctx.socket.return_value = self.mock_socket

        self.mock_socket.reset_mock()

        self.device_id = 0
        self.world_size = 3
        self.default_master_addr = "127.0.0.1"
        self.default_base_port = 5555

    def tearDown(self):
        global global_manager
        global_manager = self.original_manager

    def test_manager_init_rank0(self):
        manager = ShareMemoryManager(
            instance_world_size=self.world_size,
            instance_id=0,
        )
        self.assertEqual(manager.instance_world_size, self.world_size)
        self.assertEqual(manager.instance_id, 0)
        self.assertEqual(manager.master_addr, self.default_master_addr)
        self.assertEqual(manager.base_port, self.default_base_port)
        self.assertTrue(manager.is_master)

    def test_manager_init_rank1(self):
        manager = ShareMemoryManager(
            instance_world_size=self.world_size,
            instance_id=1,
            master_addr="192.168.1.100",
            base_port=6666
        )
        self.assertEqual(manager.instance_id, 1)
        self.assertFalse(manager.is_master)
        self.assertEqual(manager.master_addr, "192.168.1.100")
        self.assertEqual(manager.base_port, 6666)

    def test_broadcast_handle_master(self):
        manager = ShareMemoryManager(instance_world_size=2, instance_id=0)
        pub_port = self.default_base_port + self.device_id + 100

        ret_handle = manager.broadcast_handle(99999)

        self.assertEqual(ret_handle, 99999)


    def test_broadcast_handle_slave(self):
        manager = ShareMemoryManager(instance_world_size=2, instance_id=1)
        pub_port = self.default_base_port + self.device_id + 100

        ret_handle = manager.broadcast_handle(None)

        self.assertEqual(ret_handle, 123456)
        self.mock_socket.setsockopt.assert_has_calls([
            call(zmq.SUBSCRIBE, b""),
            call(zmq.RCVTIMEO, 5000)
        ])


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestGetShareMemoryManager(unittest.TestCase):

    def setUp(self):
        global global_manager
        self.original_manager = global_manager
        global_manager = None
        msm.manager = None

    def tearDown(self):
        global global_manager
        global_manager = self.original_manager

    def test_singleton_pattern(self):
        manager1 = init_share_memory(
            instance_world_size=4,
            instance_id=0,
            master_addr="192.168.1.100",
            base_port=6666
        )
        self.assertIsInstance(manager1, ShareMemoryManager)

        manager2 = get_share_memory_manager()
        self.assertIs(manager1, manager2)

    def test_init_without_required_params(self):
        with self.assertRaises(msm.ParametersInvalid) as ctx:
            get_share_memory_manager()
        self.assertIn("ShareMemoryManager has not been initialized", str(ctx.exception))

    def test_dynamic_config_addr_port(self):
        manager = init_share_memory(
            instance_world_size=2,
            instance_id=0,
            master_addr="10.0.0.5",
            base_port=7777
        )
        self.assertEqual(manager.master_addr, "10.0.0.5")
        self.assertEqual(manager.base_port, 7777)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestMemoryShareTo(unittest.TestCase):

    def setUp(self):
        global global_manager
        self.original_manager = global_manager
        self.device = f'npu:{DEVICE_ID}'
        self.dtype = torch.float16

        mock_socket = MagicMock()
        mock_socket.setsockopt.return_value = None
        mock_socket.send_pyobj.return_value = None
        mock_socket.recv_pyobj.return_value = 123456
        mock_zmq_ctx.socket.return_value = mock_socket
        global_manager = ShareMemoryManager(
            instance_world_size=2,
            instance_id=0,
        )

    def tearDown(self):
        global global_manager
        global_manager = self.original_manager

    def test_share_memory_basic(self):
        module = nn.Linear(10, 10).cpu()
        result_module = share_memory(module, device=self.device, dtype=self.dtype)

        self.assertIs(result_module, module)
        self.assertEqual(next(module.parameters()).dtype, self.dtype)

    def test_share_memory_cpu_fallback(self):
        module = nn.Linear(5, 5).cpu()
        result_module = share_memory(module, device="cpu")
        self.assertIsInstance(result_module, nn.Linear)
        self.assertEqual(next(result_module.parameters()).device, torch.device("cpu"))


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestCheckDeviceDtype(unittest.TestCase):

    def test_check_device_dtype_npu_match(self):
        module = nn.Linear(10,10).to(f'npu:{DEVICE_ID}')
        target_device = torch.device(f'npu:{DEVICE_ID}')

        should_fallback, result, _, _ = _check_device_and_dtype(module, target_device, torch.float16)
        self.assertTrue(should_fallback)
        self.assertIs(result, module)

    def test_check_invalid_dtype(self):
        module = nn.Linear(5,5).cpu()
        target_device = torch.device(f'npu:{DEVICE_ID}')

        with self.assertRaises(msm.ParametersInvalid):
            _check_device_and_dtype(module, target_device, torch.int32)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestAllInOnePipeline(unittest.TestCase):
    def test_full_pipeline(self):
        manager = init_share_memory(
            instance_world_size=2,
            instance_id=0,
        )
        self.assertIsInstance(manager, ShareMemoryManager)

        model = nn.Linear(10, 10)
        model = share_memory(model, device="npu:0", dtype=torch.bfloat16)

        self.assertIsInstance(model, nn.Linear)
        self.assertTrue(next(model.parameters()).is_npu)


if __name__ == '__main__':
    unittest.main(verbosity=2, failfast=True)