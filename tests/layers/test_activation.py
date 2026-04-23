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
import torch
import torch.nn as nn
import torch_npu

from device import DEVICE_ID
from mindiesd.layers.activation import get_activation_layer, GELU
from mindiesd.utils import ParametersInvalid


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestActivation(unittest.TestCase):
    def test_get_activation_layer_valid(self):
        func = get_activation_layer('gelu')
        self.assertIsInstance(func, nn.Module)

    def test_get_activation_layer_invalid(self):
        with self.assertRaises(ParametersInvalid):
            func = get_activation_layer('test')

    def test_gelu(self):
        tensor = torch.randn(size=(1, 2, 3)).to(f"npu:{DEVICE_ID}")
        gelu = GELU(approximate="test")
        with self.assertRaises(ParametersInvalid):
            output = gelu(tensor)


if __name__ == '__main__':
    torch.manual_seed(1234)
    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()