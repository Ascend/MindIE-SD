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
import os
import torch

from mindiesd.layers.flash_attn.common import AttentionParam
from mindiesd.layers.flash_attn.attention_operate import (
    AttentionOperateBase, register_op_duo, register_op_800, device_duo_op, device_800_op)


@register_op_800("test_op")
@register_op_duo("test_op")
class TestOperator(AttentionOperateBase):
    supported_layout = ["BNSD", "BSH"]
    supported_dtype = [torch.float32]

    @classmethod
    def is_supported_shape(cls, attn_param: AttentionParam) -> bool:
        return True

    @classmethod
    def forward_attn_bnsd(cls, attn_param, query, key, value, mask=None, scale=None) -> None:
        return "bnsd"

    @classmethod
    def forward_attn_bsnd(cls, attn_param, query, key, value, mask=None, scale=None) -> None:
        return "bsnd"

    @classmethod
    def forward_attn_bsh(cls, attn_param, query, key, value, mask=None, scale=None) -> None:
        return "bsh"


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestAttentionOperator(unittest.TestCase):
    def test_operator_registry_duo(self):
        op = device_duo_op.get_all()
        self.assertIn("test_op", op)
        test_op = device_duo_op.get("test_op")

    def test_operator_registry_800(self):
        op = device_800_op.get_all()
        self.assertIn("test_op", op)
        test_op = device_800_op.get("test_op")


if __name__ == '__main__':
    unittest.main()