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
import sys
import torch
import torch_npu
import torch.nn as nn

sys.path.append('../')
from device import DEVICE_ID
from mindiesd import layernorm_scale_shift
from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import NPUDevice, get_npu_device
from unittest.mock import Mock

class TestAdaLayerNorm(unittest.TestCase):
    def setUp(self):
        self.norm_eps = 1e-5

    def test_layernorm_type(self):
        device = "npu"
        layernorm = nn.GroupNorm(4, 64).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)


    def test_x_type(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = [2, 1024, 128]
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)


    def test_scale_type(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = [2, 128]
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)


    def test_shift_type(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = [2, 128]
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)


    def test_fused_type(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = "True"

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)


    def test_x_dim(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_scale_dim(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 1, 1024, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_scale_second_dim(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_shift_dim(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 1, 1024, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_shift_second_dim(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_x_scale_dim_equal(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 64], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_scale_shift_dim_equal(self):
        device = "npu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 64], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)


    @torch.no_grad()
    def test_layernorm_scale_shift_2d_non_affine(self):
        device = "npu"
        batch_size = 2
        sentence_length = 1024
        hidden_size = 128
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)

        x = torch.randn([batch_size, sentence_length, hidden_size], dtype=torch.float32).to(device)
        scale = torch.randn([batch_size, hidden_size], dtype=torch.float32).to(device)
        shift = torch.randn([batch_size, hidden_size], dtype=torch.float32).to(device)

        out_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=True)
        out_non_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=False)

        self.assertEqual(out_non_fused.shape, out_fused.shape)

        out_fused = out_fused.reshape(1, -1).to(torch.float32)
        out_non_fused = out_non_fused.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_non_fused, out_fused)[0], 2**-7)


    @torch.no_grad()
    def test_layernorm_scale_shift_2d_use_affine(self):
        device = "npu"
        batch_size = 2
        sentence_length = 1024
        hidden_size = 128
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=True).to(device)

        x = torch.randn([batch_size, sentence_length, hidden_size], dtype=torch.float32).to(device)
        scale = torch.randn([batch_size, hidden_size], dtype=torch.float32).to(device)
        shift = torch.randn([batch_size, hidden_size], dtype=torch.float32).to(device)

        out_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=True)
        out_non_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=False)

        self.assertEqual(out_non_fused.shape, out_fused.shape)

        out_fused = out_fused.reshape(1, -1).to(torch.float32)
        out_non_fused = out_non_fused.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_non_fused, out_fused)[0], 2**-7)


    @torch.no_grad()
    def test_layernorm_scale_shift_3d_non_affine(self):
        device = "npu"
        batch_size = 2
        sentence_length = 1024
        hidden_size = 128
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)

        x = torch.randn([batch_size, sentence_length, hidden_size], dtype=torch.float32).to(device)
        scale = torch.randn([batch_size, 1, hidden_size], dtype=torch.float32).to(device)
        shift = torch.randn([batch_size, 1, hidden_size], dtype=torch.float32).to(device)

        out_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=True)
        out_non_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=False)

        self.assertEqual(out_non_fused.shape, out_fused.shape)

        out_fused = out_fused.reshape(1, -1).to(torch.float32)
        out_non_fused = out_non_fused.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_non_fused, out_fused)[0], 2**-7)


    @torch.no_grad()
    def test_layernorm_scale_shift_3d_use_affine(self):
        device = "npu"
        batch_size = 2
        sentence_length = 1024
        hidden_size = 128
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=True).to(device)

        x = torch.randn([batch_size, sentence_length, hidden_size], dtype=torch.float32).to(device)
        scale = torch.randn([batch_size, 1, hidden_size], dtype=torch.float32).to(device)
        shift = torch.randn([batch_size, 1, hidden_size], dtype=torch.float32).to(device)

        out_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=True)
        out_non_fused = layernorm_scale_shift(layernorm, x, scale, shift, fused=False)

        self.assertEqual(out_non_fused.shape, out_fused.shape)

        out_fused = out_fused.reshape(1, -1).to(torch.float32)
        out_non_fused = out_non_fused.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_non_fused, out_fused)[0], 2**-7)


    @torch.no_grad()
    def test_layernorm_scale_shift_3d_use_affine_and_a5(self):
        device = "npu"
        batch_size = 1
        sentence_length = 1024
        hidden_size = 128
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=True).to(device)

        x = torch.randn([batch_size, sentence_length, hidden_size], dtype=torch.float32).to(device)
        scale = torch.randn([batch_size, hidden_size], dtype=torch.float32).to(device)
        shift = torch.randn([batch_size, hidden_size], dtype=torch.float32).to(device)
        
        origin_ops_v2 = torch.ops.mindie.adaln_v2_mindie_sd
        origin_ops_v1 = torch.ops.mindie.adaln_mindie_sd
        ops_mock_v2 = Mock()
        ops_mock_v1 = Mock()
        
        def mock_ops_v2(*args, **kwargs):
            ops_mock_v2()
            return origin_ops_v2(*args, **kwargs)
        
        def mock_ops_v1(*args, **kwargs):
            ops_mock_v1()
            return origin_ops_v1(*args, **kwargs)
        
        torch.ops.mindie.adaln_v2_mindie_sd = mock_ops_v2
        torch.ops.mindie.adaln_mindie_sd = mock_ops_v1
        try:
            out = layernorm_scale_shift(layernorm, x, scale, shift, fused=True)
            if get_npu_device() == NPUDevice.A5:
                ops_mock_v2.assert_called_once()
            else:
                ops_mock_v1.assert_called_once()
        finally:
            torch.ops.mindie.adaln_v2_mindie_sd = origin_ops_v2
            torch.ops.mindie.adaln_mindie_sd = origin_ops_v1
           

if __name__ == "__main__":
    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()