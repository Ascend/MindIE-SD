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
class TestAdaLayerNorm(unittest.TestCase):
    def setUp(self):
        # 设置NPU设备
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        # 定义输入张量的形状和数据类型
        self.batch_size = 1
        self.seqence_length = 1024
        self.hidden_size = 128
        self.dtype = torch.float32
        self.eps = 1e-5

        self.layernorm = nn.LayerNorm(normalized_shape=128, eps=self.eps, elementwise_affine=True).npu()

        # 创建随机张量
        self.x = torch.randn([self.batch_size, self.seqence_length, self.hidden_size], device=self.device, dtype=self.dtype)
        self.scale = torch.randn([self.batch_size, self.hidden_size], device=self.device, dtype=self.dtype)
        self.shift = torch.randn([self.batch_size, self.hidden_size], device=self.device, dtype=self.dtype)

    def test_adalayernorm_output_shape(self):
        output = torch.ops.mindiesd.adaln_v2(
            self.x, self.scale, self.shift, self.layernorm.weight, self.layernorm.bias, self.layernorm.eps
        )[0]
        expected = [self.batch_size, self.seqence_length, self.hidden_size]
        self.assertEqual(list(output.shape), expected, "Output shape does not match expected shape.")

    def test_adalayernorm(self):
        output = torch.ops.mindiesd.adaln_v2(
            self.x, self.scale, self.shift, self.layernorm.weight, self.layernorm.bias, self.layernorm.eps
        )[0].reshape(1, -1)
        origin = self.layernorm(self.x).reshape(1, -1)

        self.assertGreater(torch.cosine_similarity(output, origin)[0], 2**-7)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)