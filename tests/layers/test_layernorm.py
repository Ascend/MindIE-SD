#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
import sys
import torch
import torch_npu
import torch.nn as nn

sys.path.append('../')
from device import DEVICE_ID
from mindiesd import fast_layernorm
from mindiesd.utils import ParametersInvalid


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        device = "npu"
        self.x = torch.randn([2, 1024, 128], dtype=torch.float32).npu()
        self.layernorm_have_param = nn.LayerNorm(normalized_shape=128).npu()
        self.layernorm_non_param = nn.LayerNorm(normalized_shape=128, elementwise_affine=False, bias=False).npu()

    def test_layernorm_have_param(self):
        out_npu = fast_layernorm(self.layernorm_have_param, self.x, 0).reshape(1, -1)
        origin = self.layernorm_have_param(self.x).reshape(1, -1)
        self.assertGreater(torch.cosine_similarity(out_npu, origin)[0], 2**-7)
    
    def test_layernorm_non_param(self):
        out_npu = fast_layernorm(self.layernorm_non_param, self.x, 0).reshape(1, -1)
        origin = self.layernorm_non_param(self.x).reshape(1, -1)
        self.assertGreater(torch.cosine_similarity(out_npu, origin)[0], 2**-7)
    
    def test_impl_mode(self):
        with self.assertRaises(ParametersInvalid):
            out = fast_layernorm(self.layernorm_have_param, self.x, 5)
        with self.assertRaises(ParametersInvalid):
            out = fast_layernorm(self.layernorm_have_param, self.x.to(torch.bfloat16), 2)

        out_npu = fast_layernorm(self.layernorm_have_param, self.x, 1).reshape(1, -1)
        origin = self.layernorm_have_param(self.x).reshape(1, -1)
        self.assertGreater(torch.cosine_similarity(out_npu, origin)[0], 2**-7)


if __name__ == '__main__':
    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()