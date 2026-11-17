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

from mindiesd.eplb.task_payload import TaskPayload, TaskType
from mindiesd.utils.exception import ParametersInvalid


class TestTaskPayload(unittest.TestCase):
    def test_profile_payloads_have_independent_data(self):
        first = TaskPayload(TaskType.PROFILE)
        second = TaskPayload(TaskType.PROFILE)
        first.data["load"] = [1, 2]
        self.assertEqual(second.data, {})
        self.assertIsNone(first.moe_layer_idx)

    def test_update_layout_accepts_zero_layer_index(self):
        payload = TaskPayload(TaskType.UPDATE_LAYOUT, worker_rank=0, moe_layer_idx=0, data={"layout": [0, 1]})
        self.assertEqual(payload.moe_layer_idx, 0)
        self.assertEqual(payload.data, {"layout": [0, 1]})

    def test_invalid_type_is_rejected(self):
        for value in ["PROFILE", 1, None]:
            with self.subTest(value=value), self.assertRaisesRegex(ParametersInvalid, "TaskType Enum"):
                TaskPayload(value)

    def test_non_dictionary_data_is_rejected(self):
        for value in [[], None, "layout"]:
            with self.subTest(value=value), self.assertRaisesRegex(ParametersInvalid, "dictionary"):
                TaskPayload(TaskType.PROFILE, data=value)

    def test_update_requires_data(self):
        with self.assertRaisesRegex(ParametersInvalid, "empty"):
            TaskPayload(TaskType.UPDATE_LAYOUT, moe_layer_idx=0)

    def test_update_requires_layer_index(self):
        with self.assertRaisesRegex(ParametersInvalid, "moe_layer_idx"):
            TaskPayload(TaskType.UPDATE_LAYOUT, data={"layout": [0, 1]})


if __name__ == "__main__":
    unittest.main()
