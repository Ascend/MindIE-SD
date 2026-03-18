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
import unittest
import sys
import numpy as np

sys.path.append('../')

from mindiesd.eplb.eplb_scheduler import eplb_greedy

RESPONSE = {
    0: np.array(
        [450, 1, 3892, 4017, 226, 67, 1321, 3, 0, 214, 1376, 722, 318, 428, 211, 13, 3, 86, 10, 8, 44, 28, 39, 159,
            0, 2787, 4083, 0, 219, 8, 4018, 0, 2, 1, 53, 6, 3, 5, 19, 0, 0, 12, 208, 3987, 2171, 0, 27, 0, 30, 0, 1,
            11, 0, 4096, 0, 0, 3, 1, 6, 1, 281, 2, 46, 3982, 0, 0, 4093, 2, 1, 0, 27, 3, 1, 4, 11, 3, 0, 0, 3, 1, 0,
            774, 0, 0, 89, 211, 1567, 0, 43, 3982, 135, 4096, 15, 0, 325, 320, 1, 0, 4, 0, 94, 333, 0, 3, 9, 26, 444,
            0, 0, 0, 13, 107, 3472, 106, 1, 4079, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 329, 1, 0, 55, 0, 0, 0, 32, 4, 0, 1,
            11, 2, 0, 0, 0, 0, 4096, 0, 1, 0, 0, 0, 0, 0, 2, 4096, 8, 0, 0, 2, 0, 0, 0, 13, 69, 5, 0, 0, 0, 203, 112,
            20, 7, 0, 1, 0, 0, 0, 0, 9, 0, 21, 0, 0, 0, 21, 11, 0, 40, 1, 39, 11, 0, 2, 72, 1, 0, 5, 0, 0, 4001, 0, 2,
            1, 0, 0, 1, 0, 86, 0, 4096, 3, 12, 2, 4096, 0, 0, 4096, 0, 0, 0, 3664, 1, 0, 1, 5, 0, 14, 0, 0, 92, 0, 6,
            0, 0, 6, 3, 3982, 714, 0, 4096, 2, 0, 0, 1993, 0, 6, 0, 0, 10, 0, 1, 0, 1, 3, 96, 659, 4096, 55, 0, 0, 0,
            1, 0, 8, 129, 0, 2, 21, 0, 89, 19, 46, 0, 0, 7, 3, 19, 0, 0, 108, 0, 0, 0, 40, 4096, 1, 40, 36, 0, 2, 0, 0,
            71, 2840, 6, 1732, 0, 6, 0, 5, 2, 0, 3, 0, 0, 1, 0, 0, 4, 4096, 4094, 66, 1, 3, 0, 1, 37, 0, 30, 4096, 0,
            0]),
    1: np.array(
        [450, 1, 3892, 4017, 226, 67, 1321, 3, 0, 214, 1376, 722, 318, 428, 211, 13, 3, 86, 10, 8, 44, 28, 39, 159,
            0, 2787, 4083, 0, 219, 8, 4018, 0, 2, 1, 53, 6, 3, 5, 19, 0, 0, 12, 208, 3987, 2171, 0, 27, 0, 30, 0, 1,
            11, 0, 4096, 0, 0, 3, 1, 6, 1, 281, 2, 46, 3982, 0, 0, 4093, 2, 1, 0, 27, 3, 1, 4, 11, 3, 0, 0, 3, 1, 0,
            774, 0, 0, 89, 211, 1567, 0, 43, 3982, 135, 4096, 15, 0, 325, 320, 1, 0, 4, 0, 94, 333, 0, 3, 9, 26, 444,
            0, 0, 0, 13, 107, 3472, 106, 1, 4079, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 329, 1, 0, 55, 0, 0, 0, 32, 4, 0, 1,
            11, 2, 0, 0, 0, 0, 4096, 0, 1, 0, 0, 0, 0, 0, 2, 4096, 8, 0, 0, 2, 0, 0, 0, 13, 69, 5, 0, 0, 0, 203, 112,
            20, 7, 0, 1, 0, 0, 0, 0, 9, 0, 21, 0, 0, 0, 21, 11, 0, 40, 1, 39, 11, 0, 2, 72, 1, 0, 5, 0, 0, 4001, 0, 2,
            1, 0, 0, 1, 0, 86, 0, 4096, 3, 12, 2, 4096, 0, 0, 4096, 0, 0, 0, 3664, 1, 0, 1, 5, 0, 14, 0, 0, 92, 0, 6,
            0, 0, 6, 3, 3982, 714, 0, 4096, 2, 0, 0, 1993, 0, 6, 0, 0, 10, 0, 1, 0, 1, 3, 96, 659, 4096, 55, 0, 0, 0,
            1, 0, 8, 129, 0, 2, 21, 0, 89, 19, 46, 0, 0, 7, 3, 19, 0, 0, 108, 0, 0, 0, 40, 4096, 1, 40, 36, 0, 2, 0, 0,
            71, 2840, 6, 1732, 0, 6, 0, 5, 2, 0, 3, 0, 0, 1, 0, 0, 4, 4096, 4094, 66, 1, 3, 0, 1, 37, 0, 30, 4096, 0,
            0]),
    2: np.array(
        [450, 1, 3892, 4017, 226, 67, 1321, 3, 0, 214, 1376, 722, 318, 428, 211, 13, 3, 86, 10, 8, 44, 28, 39, 159,
            0, 2787, 4083, 0, 219, 8, 4018, 0, 2, 1, 53, 6, 3, 5, 19, 0, 0, 12, 208, 3987, 2171, 0, 27, 0, 30, 0, 1,
            11, 0, 4096, 0, 0, 3, 1, 6, 1, 281, 2, 46, 3982, 0, 0, 4093, 2, 1, 0, 27, 3, 1, 4, 11, 3, 0, 0, 3, 1, 0,
            774, 0, 0, 89, 211, 1567, 0, 43, 3982, 135, 4096, 15, 0, 325, 320, 1, 0, 4, 0, 94, 333, 0, 3, 9, 26, 444,
            0, 0, 0, 13, 107, 3472, 106, 1, 4079, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 329, 1, 0, 55, 0, 0, 0, 32, 4, 0, 1,
            11, 2, 0, 0, 0, 0, 4096, 0, 1, 0, 0, 0, 0, 0, 2, 4096, 8, 0, 0, 2, 0, 0, 0, 13, 69, 5, 0, 0, 0, 203, 112,
            20, 7, 0, 1, 0, 0, 0, 0, 9, 0, 21, 0, 0, 0, 21, 11, 0, 40, 1, 39, 11, 0, 2, 72, 1, 0, 5, 0, 0, 4001, 0, 2,
            1, 0, 0, 1, 0, 86, 0, 4096, 3, 12, 2, 4096, 0, 0, 4096, 0, 0, 0, 3664, 1, 0, 1, 5, 0, 14, 0, 0, 92, 0, 6,
            0, 0, 6, 3, 3982, 714, 0, 4096, 2, 0, 0, 1993, 0, 6, 0, 0, 10, 0, 1, 0, 1, 3, 96, 659, 4096, 55, 0, 0, 0,
            1, 0, 8, 129, 0, 2, 21, 0, 89, 19, 46, 0, 0, 7, 3, 19, 0, 0, 108, 0, 0, 0, 40, 4096, 1, 40, 36, 0, 2, 0, 0,
            71, 2840, 6, 1732, 0, 6, 0, 5, 2, 0, 3, 0, 0, 1, 0, 0, 4, 4096, 4094, 66, 1, 3, 0, 1, 37, 0, 30, 4096, 0,
            0]),

    3: np.array(
        [450, 1, 3892, 4017, 226, 67, 1321, 3, 0, 214, 1376, 722, 318, 428, 211, 13, 3, 86, 10, 8, 44, 28, 39, 159,
            0, 2787, 4083, 0, 219, 8, 4018, 0, 2, 1, 53, 6, 3, 5, 19, 0, 0, 12, 208, 3987, 2171, 0, 27, 0, 30, 0, 1,
            11, 0, 4096, 0, 0, 3, 1, 6, 1, 281, 2, 46, 3982, 0, 0, 4093, 2, 1, 0, 27, 3, 1, 4, 11, 3, 0, 0, 3, 1, 0,
            774, 0, 0, 89, 211, 1567, 0, 43, 3982, 135, 4096, 15, 0, 325, 320, 1, 0, 4, 0, 94, 333, 0, 3, 9, 26, 444,
            0, 0, 0, 13, 107, 3472, 106, 1, 4079, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 329, 1, 0, 55, 0, 0, 0, 32, 4, 0, 1,
            11, 2, 0, 0, 0, 0, 4096, 0, 1, 0, 0, 0, 0, 0, 2, 4096, 8, 0, 0, 2, 0, 0, 0, 13, 69, 5, 0, 0, 0, 203, 112,
            20, 7, 0, 1, 0, 0, 0, 0, 9, 0, 21, 0, 0, 0, 21, 11, 0, 40, 1, 39, 11, 0, 2, 72, 1, 0, 5, 0, 0, 4001, 0, 2,
            1, 0, 0, 1, 0, 86, 0, 4096, 3, 12, 2, 4096, 0, 0, 4096, 0, 0, 0, 3664, 1, 0, 1, 5, 0, 14, 0, 0, 92, 0, 6,
            0, 0, 6, 3, 3982, 714, 0, 4096, 2, 0, 0, 1993, 0, 6, 0, 0, 10, 0, 1, 0, 1, 3, 96, 659, 4096, 55, 0, 0, 0,
            1, 0, 8, 129, 0, 2, 21, 0, 89, 19, 46, 0, 0, 7, 3, 19, 0, 0, 108, 0, 0, 0, 40, 4096, 1, 40, 36, 0, 2, 0, 0,
            71, 2840, 6, 1732, 0, 6, 0, 5, 2, 0, 3, 0, 0, 1, 0, 0, 4, 4096, 4094, 66, 1, 3, 0, 1, 37, 0, 30, 4096, 0,
            0])
}

EXPERT_DICT = {
    0: list(range(0, 80)),  # 第0组: 0-79
    1: list(range(80, 160)),  # 第1组: 80-159
    2: list(range(160, 240)),  # 第2组: 160-239
    3: list(range(240, 320))  # 第3组: 240-319
}

EXPERT_DICT_REDUNDANT = {
    0: list(range(0, 84)),  # 第0组: 0-83
    1: list(range(80, 164)),  # 第1组: 80-163
    2: list(range(160, 244)),  # 第2组: 160-243
    3: list(range(240, 324))  # 第3组: 240-323
}


@unittest.skipIf(True, "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestEplbScheduler(unittest.TestCase):
    def test_A2A_algo(self):
        result = eplb_greedy(RESPONSE, "A2A", EXPERT_DICT, world_size=4, expert_num=320)
        self.assertIsNotNone(result)
        update, device_indices_list, local_expert_indices_list, local_expert_list, expert_trans_tensor = result
        self.assertEqual(update, True)
        self.assertEqual(len(device_indices_list), 4)
        self.assertEqual(len(local_expert_indices_list), 4)
        self.assertEqual(len(local_expert_indices_list[0]), 320)
        for rank, experts in enumerate(local_expert_list):
            unique_experts = set(experts)
            self.assertEqual(len(experts), len(unique_experts))
        expected_shape = (320, 320)
        self.assertEqual(expert_trans_tensor.shape, expected_shape)
        
    def test_AG_algo(self):
        result = eplb_greedy(RESPONSE, "AG", EXPERT_DICT, world_size=4, expert_num=320)
        self.assertIsNotNone(result)
        update, device_indices_list, local_expert_indices_list, local_expert_list, expert_trans_tensor = result
        self.assertEqual(update, True)
        self.assertEqual(len(device_indices_list), 4)
        self.assertEqual(len(local_expert_indices_list), 4)
        self.assertEqual(len(local_expert_indices_list[0]), 320)
        for rank, experts in enumerate(local_expert_list):
            unique_experts = set(experts)
            self.assertEqual(len(experts), len(unique_experts))
        expected_shape = (320, 320)
        self.assertEqual(expert_trans_tensor.shape, expected_shape)
    
    def test_EX_algo(self):
        result = eplb_greedy(RESPONSE, "EX", EXPERT_DICT, world_size=4, expert_num=320)
        self.assertIsNotNone(result)
        update, device_indices_list, local_expert_indices_list, local_expert_list, expert_trans_tensor = result
        self.assertEqual(update, True)
        self.assertEqual(len(device_indices_list), 4)
        self.assertEqual(len(local_expert_indices_list), 4)
        self.assertEqual(len(local_expert_indices_list[0]), 320)
        for rank, experts in enumerate(local_expert_list):
            unique_experts = set(experts)
            self.assertEqual(len(experts), len(unique_experts))
        expected_shape = (320, 320)
        self.assertEqual(expert_trans_tensor.shape, expected_shape)

    def test_A2A_redundant_algo(self):
        result = eplb_greedy(RESPONSE, "A2A", EXPERT_DICT_REDUNDANT, 4, 320, 84, 4)
        self.assertIsNotNone(result)
        _, _, _, local_expert_list, _ = result
        for rank, experts in enumerate(local_expert_list):
            unique_experts = set(experts)
            self.assertEqual(len(experts), len(unique_experts))
            self.assertEqual(len(unique_experts), 84)

    def test_AG_redundant_algo(self):
        result = eplb_greedy(RESPONSE, "AG", EXPERT_DICT_REDUNDANT, 4, 320, 84, 4)
        self.assertIsNotNone(result)
        _, _, _, local_expert_list, _ = result
        for rank, experts in enumerate(local_expert_list):
            unique_experts = set(experts)
            self.assertEqual(len(experts), len(unique_experts))
            self.assertEqual(len(unique_experts), 84)

if __name__ == '__main__':
    unittest.main()