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
import os
import unittest
import torch
import torch_npu
from torch import nn

from device import DEVICE_ID
from mindiesd import fast_layernorm
from mindiesd.utils import ParametersInvalid


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
class TestLayerNorm(unittest.TestCase):
    def setUp(self):
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
            fast_layernorm(self.layernorm_have_param, self.x, 5)
        with self.assertRaises(ParametersInvalid):
            fast_layernorm(self.layernorm_have_param, self.x.to(torch.bfloat16), 2)

        out_npu = fast_layernorm(self.layernorm_have_param, self.x, 1).reshape(1, -1)
        origin = self.layernorm_have_param(self.x).reshape(1, -1)
        self.assertGreater(torch.cosine_similarity(out_npu, origin)[0], 2**-7)

    def test_normalized_shape_too_large(self):
        """Test that normalized_shape with more dims than input raises ParametersInvalid."""
        # input is 3-D [2, 1024, 128], normalized_shape is 4-D
        bad_layernorm = nn.LayerNorm(normalized_shape=(2, 1024, 128, 64)).npu()
        with self.assertRaises(ParametersInvalid) as ctx:
            fast_layernorm(bad_layernorm, self.x, 0)
        self.assertIn("normalized_shape must fit within input dimensions", str(ctx.exception))

    def test_normalized_shape_equal_dims(self):
        """Test edge case where normalized_shape ndim equals input ndim."""
        layernorm = nn.LayerNorm(normalized_shape=(2, 1024, 128)).npu()
        x = torch.randn([2, 1024, 128], dtype=torch.float32).npu()
        out = fast_layernorm(layernorm, x, 0)
        self.assertEqual(out.shape, x.shape)

    def test_normalized_shape_less_than_dims(self):
        """Test normal case where normalized_shape ndim is less than input ndim."""
        layernorm = nn.LayerNorm(normalized_shape=(128,)).npu()
        x = torch.randn([2, 1024, 128], dtype=torch.float32).npu()
        out = fast_layernorm(layernorm, x, 0)
        self.assertEqual(out.shape, x.shape)


if __name__ == '__main__':
    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()
