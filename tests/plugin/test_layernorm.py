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

# 加载自定义库
if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        # 设置NPU设备
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        # 定义输入张量的形状和数据类型
        self.x_shape = (2, 48, 128)
        self.dtype = torch.bfloat16

        self.layernorm_origin = nn.LayerNorm(normalized_shape=128).npu()

        # 创建随机张量
        self.x = torch.randn(self.x_shape, device=self.device, dtype=self.dtype)

    def test_layernorm_output_shape(self):
        output = torch.ops.mindiesd.layernorm(
            self.x, list(self.layernorm_origin.normalized_shape), self.layernorm_origin.weight, self.layernorm_origin.bias,
            self.layernorm_origin.eps, impl_mode=0)[0]
        expected_shape = self.x_shape
        self.assertEqual(output.shape, expected_shape,
                         "Output shape does not match expected shape.")

    def test_layernorm(self):
        output_0 = torch.ops.mindiesd.layernorm(
            self.x, list(self.layernorm_origin.normalized_shape), self.layernorm_origin.weight, self.layernorm_origin.bias,
            self.layernorm_origin.eps, impl_mode=0)[0].reshape(1, -1)
        output_1 = torch.ops.mindiesd.layernorm(
            self.x, list(self.layernorm_origin.normalized_shape), self.layernorm_origin.weight, self.layernorm_origin.bias,
            self.layernorm_origin.eps, impl_mode=1)[0].reshape(1, -1)
        origin = self.layernorm_origin(self.x).reshape(1, -1)

        self.assertGreater(torch.cosine_similarity(output_0, origin)[0], 2**-7)
        self.assertGreater(torch.cosine_similarity(output_1, origin)[0], 2**-7)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)