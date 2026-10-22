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

from mindiesd.eplb.dispatcher import DispatcherBase, DynamicDispatcher


class TestDispatcherState(unittest.TestCase):
    def setUp(self):
        self.weight1 = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        self.weight2 = self.weight1.transpose(1, 2).contiguous()

    def make_dispatcher(self):
        with mock.patch.object(torch.Tensor, "pin_memory", new=lambda tensor: tensor):
            return DynamicDispatcher(4, self.weight1, self.weight2, ep_rank=1, ep_size=2)

    def test_initial_mapping_routes_each_expert_to_its_owner(self):
        dispatcher = DispatcherBase(4, self.weight1, self.weight2, ep_rank=1, ep_size=2)
        indices = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.equal(dispatcher.get_device_indices(indices), torch.tensor([0, 0, 1, 1])))
        self.assertTrue(torch.equal(dispatcher.get_local_expert_indices(indices)[2:], torch.tensor([0, 1])))

    def test_custom_mapping_supports_reordered_experts(self):
        dispatcher = DispatcherBase(4, self.weight1, self.weight2, ep_rank=0, ep_size=2)
        dispatcher.set_device_and_local_expert_map(torch.tensor([1, 0, 1, 0]), torch.tensor([0, 0, 1, 1]))
        indices = torch.tensor([3, 2, 1, 0])
        self.assertTrue(torch.equal(dispatcher.get_device_indices(indices), torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.equal(dispatcher.get_local_expert_indices(indices), torch.tensor([1, 1, 0, 0])))

    def test_pending_update_waits_for_every_rank(self):
        dispatcher = self.make_dispatcher()
        dispatcher.update_valid_tensor.fill_(1)
        with mock.patch("mindiesd.eplb.dispatcher.dist.all_reduce", side_effect=lambda tensor, op: tensor.fill_(0)):
            self.assertFalse(dispatcher.check_consistency())
        self.assertEqual(dispatcher.update_valid_tensor.item(), 1)
        with mock.patch("mindiesd.eplb.dispatcher.dist.all_reduce", side_effect=lambda tensor, op: tensor.fill_(1)):
            self.assertTrue(dispatcher.check_consistency())

    def test_weight_and_map_update_is_consumed_once(self):
        dispatcher = self.make_dispatcher()
        new_weight1, new_weight2 = self.weight1[[3, 1]], self.weight2[[3, 1]]
        transform = torch.eye(4)[[0, 2, 3, 1]]
        dispatcher.copy_module_weight_and_map(
            weight1=new_weight1,
            weight2=new_weight2,
            device_indices_map=torch.tensor([0, 1, 0, 1]),
            local_expert_indices_map=torch.tensor([0, 1, 1, 0]),
            local_expert_list=[3, 1],
            expert_trans_tensor=transform,
            layer_idx=0,
        )
        self.assertEqual(dispatcher.update_valid_tensor.item(), 1)
        with mock.patch("mindiesd.eplb.dispatcher.dist.all_reduce"):
            self.assertTrue(dispatcher.check_consistency())
        weight1, weight2, count, device_map, local_map, expert_list = dispatcher.update_module_weight_and_map()
        self.assertTrue(torch.equal(weight1, new_weight1))
        self.assertTrue(torch.equal(weight2, new_weight2))
        self.assertEqual((count, expert_list), (2, [3, 1]))
        self.assertTrue(torch.equal(device_map, torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.equal(local_map, torch.tensor([0, 1, 1, 0])))
        self.assertIs(dispatcher.get_expert_trans_tensor(), transform)
        self.assertEqual(dispatcher.update_valid_tensor.item(), 0)
        self.assertFalse(dispatcher.update_flag)
        with mock.patch("mindiesd.eplb.dispatcher.dist.all_reduce"):
            self.assertFalse(dispatcher.check_consistency())


if __name__ == "__main__":
    unittest.main()
