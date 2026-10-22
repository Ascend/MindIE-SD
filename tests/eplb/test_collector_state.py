#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest import mock

import torch

from mindiesd.eplb.collector import ExpertLoadCollector


class TestCollectorState(unittest.TestCase):
    @staticmethod
    def count_tokens(indices, expert_num):
        return torch.bincount(indices.long(), minlength=expert_num)

    def test_collection_accumulates_counts_and_read_resets(self):
        with mock.patch.object(torch.Tensor, "pin_memory", new=lambda tensor: tensor):
            collector = ExpertLoadCollector(4)
        with mock.patch("mindiesd.eplb.collector.torch_npu.npu_moe_compute_expert_tokens", self.count_tokens):
            collector.collect_expert_load(torch.tensor([0, 0, 2], dtype=torch.int32))
            collector.collect_expert_load(torch.tensor([1, 3, 3], dtype=torch.int32))
        self.assertTrue(torch.equal(collector.get_expert_load(), torch.tensor([2, 1, 1, 2])))
        self.assertTrue(torch.equal(collector.get_expert_load(), torch.zeros(4, dtype=torch.long)))

    def test_reset_clears_pending_counts(self):
        with mock.patch.object(torch.Tensor, "pin_memory", new=lambda tensor: tensor):
            collector = ExpertLoadCollector(4)
        with mock.patch("mindiesd.eplb.collector.torch_npu.npu_moe_compute_expert_tokens", self.count_tokens):
            collector.collect_expert_load(torch.tensor([1, 1], dtype=torch.int32))
        collector.reset()
        self.assertTrue(torch.equal(collector.get_expert_load(), torch.zeros(4, dtype=torch.long)))

    def test_collector_notifies_transfer_on_each_collection(self):
        with mock.patch.object(torch.Tensor, "pin_memory", new=lambda tensor: tensor):
            collector = ExpertLoadCollector(4, lb_interval=3)
        collector.task_transfer = mock.Mock()
        with mock.patch("mindiesd.eplb.collector.torch_npu.npu_moe_compute_expert_tokens", self.count_tokens):
            collector.collect_expert_load(torch.tensor([], dtype=torch.int32))
            collector.collect_expert_load(torch.tensor([0], dtype=torch.int32))
        self.assertEqual(collector.task_transfer.profile_emit_task.call_count, 2)
        self.assertEqual(collector.lb_interval, 3)


if __name__ == "__main__":
    unittest.main()
