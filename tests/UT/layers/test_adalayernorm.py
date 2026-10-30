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
# pylint: disable=no-name-in-module
import os
import unittest
from unittest.mock import Mock
import torch
import torch_npu
from torch import nn

from device import DEVICE_ID
from utils.utils.precision_compare import data_compare
from mindiesd import layernorm_scale_shift
from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import NPUDevice, get_npu_device


class TestAdaLayerNorm(unittest.TestCase):
    def setUp(self):
        self.norm_eps = 1e-5

    def test_layernorm_type(self):
        device = "cpu"
        layernorm = nn.GroupNorm(4, 64).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_x_type(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = [2, 1024, 128]
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_scale_type(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = [2, 128]
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_shift_type(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = [2, 128]
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_fused_type(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = "True"

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_x_dim(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_scale_dim(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 1, 1024, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_scale_invalid_second_dim(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 512, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_shift_dim(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 1, 1024, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_shift_invalid_second_dim(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 512, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_x_scale_dim_equal(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 64], dtype=torch.float32).to(device)
        shift = torch.randn([2, 128], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)

    def test_scale_shift_dim_equal(self):
        device = "cpu"
        layernorm = nn.LayerNorm(128, self.norm_eps, elementwise_affine=False).to(device)
        x = torch.randn([2, 1024, 128], dtype=torch.float32).to(device)
        scale = torch.randn([2, 128], dtype=torch.float32).to(device)
        shift = torch.randn([2, 64], dtype=torch.float32).to(device)
        fused = True

        with self.assertRaises(ParametersInvalid):
            layernorm_scale_shift(layernorm, x, scale, shift, fused)


if __name__ == "__main__":
    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()
