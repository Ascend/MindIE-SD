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

import os
import queue
import unittest

from mindiesd.eplb.task_transfer import UpdateTaskTransfer
from mindiesd.eplb.task_payload import TaskType


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU",
    "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.",
)
class TestUpdateTaskTransfer(unittest.TestCase):
    def test_update_emit_task_does_not_block_on_full_queue(self):
        """Test that update_emit_task does not block when queue is full."""
        # Create queues with maxsize=1 to easily fill them
        instruction_queues = {rank: queue.Queue(maxsize=1) for rank in range(2)}
        
        transfer = UpdateTaskTransfer(instruction_queues, moe_layer_idx=0)
        
        # Fill the queues
        for rank in range(2):
            instruction_queues[rank].put("dummy")
        
        # This should not block even though queues are full
        # It should log a warning and continue
        device_indices_list = [[0], [1]]
        local_expert_indices_list = [[0], [0]]
        local_expert_list = [[0], [1]]
        expert_trans_tensor = "dummy_tensor"
        
        # This call should complete quickly (not block)
        transfer.update_emit_task(
            device_indices_list,
            local_expert_indices_list,
            local_expert_list,
            expert_trans_tensor,
            world_size=2,
        )
        
        # Verify queues are still full (items were not added)
        for rank in range(2):
            self.assertTrue(instruction_queues[rank].full())

    def test_update_emit_task_adds_to_empty_queue(self):
        """Test that update_emit_task successfully adds items to empty queues."""
        instruction_queues = {rank: queue.Queue(maxsize=10) for rank in range(2)}
        
        transfer = UpdateTaskTransfer(instruction_queues, moe_layer_idx=0)
        
        device_indices_list = [[0], [1]]
        local_expert_indices_list = [[0], [0]]
        local_expert_list = [[0], [1]]
        expert_trans_tensor = "dummy_tensor"
        
        transfer.update_emit_task(
            device_indices_list,
            local_expert_indices_list,
            local_expert_list,
            expert_trans_tensor,
            world_size=2,
        )
        
        # Verify items were added to queues
        for rank in range(2):
            self.assertFalse(instruction_queues[rank].empty())
            task = instruction_queues[rank].get_nowait()
            self.assertEqual(task.task_type, TaskType.UPDATE_LAYOUT)
            self.assertEqual(task.worker_rank, rank)
            self.assertEqual(task.moe_layer_idx, 0)


if __name__ == "__main__":
    unittest.main()
