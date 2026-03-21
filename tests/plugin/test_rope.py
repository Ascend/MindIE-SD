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
import torch
import torch.nn as nn
import torch_npu

# 加载自定义库
torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


class TestRopeMindieSd(unittest.TestCase):
    def setUp(self):
        # 设置NPU设备
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        # 定义输入张量的形状和数据类型
        self.x_shape = (2, 48, 128, 64)
        self.cos_sin_shape = (1, 1, 128, 64)
        self.dtype = torch.float32

        # 创建随机张量
        self.x = torch.randn(self.x_shape, device=self.device, dtype=self.dtype)
        self.cos = torch.randn(self.cos_sin_shape, device=self.device, dtype=self.dtype)
        self.sin = torch.randn(self.cos_sin_shape, device=self.device, dtype=self.dtype)

    def test_rope_mindie_sd_output_shape(self):
        output = torch.ops.mindiesd.rope(self.x, self.cos, self.sin, mode=1)
        expected_shape = self.x_shape
        self.assertEqual(output.shape, expected_shape,
                         "Output shape does not match expected shape.")

    def test_rope_mindie_sd_consistency(self):
        output1 = torch.ops.mindiesd.rope(self.x, self.cos, self.sin, mode=1)
        output2 = torch.ops.mindiesd.rope(self.x, self.cos, self.sin, mode=1)
        self.assertTrue(torch.allclose(output1, output2),
                        "Multiple runs of the operator produce inconsistent results.")

    def test_rope_mindie_sd_mode_0_and_1(self):
        output_mode_0 = torch.ops.mindiesd.rope(self.x, self.cos, self.sin, mode=0)
        output_mode_1 = torch.ops.mindiesd.rope(self.x, self.cos, self.sin, mode=1)
        self.assertFalse(torch.allclose(output_mode_0, output_mode_1),
                         "Outputs for different modes should be different.")


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)