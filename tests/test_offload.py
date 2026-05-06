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
import torch_npu
from mindiesd.offload import enable_offload


class MockDITBlock(torch.nn.Module):
    def __init__(self, has_slice_tensor: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.float32))
        self.weight.is_slice_tensor = False
        self.bias = torch.nn.Parameter(torch.randn(16, dtype=torch.float32))
        self.bias.is_slice_tensor = False
        self.img_feat = torch.nn.Parameter(torch.randn(32, 32, dtype=torch.float32))
        self.img_feat.is_slice_tensor = False
        self.register_buffer('running_mean', torch.randn(16, dtype=torch.float32))
        self.running_mean.is_slice_tensor = False
        self.slice_param = None

        if has_slice_tensor:
            full_tensor = torch.randn(64, dtype=torch.float32)
            self.slice_param = torch.nn.Parameter(full_tensor[::2])
            self.slice_param.is_slice_tensor = True

    def forward(self, x):
        sum_params = self.weight.sum() + self.bias.sum() + self.img_feat.sum()
        sum_bufs = self.running_mean.sum()
        if self.slice_param is not None:
            sum_params += self.slice_param.sum()
        return x + sum_params + sum_bufs


class MockDITModel(torch.nn.Module):
    def __init__(self, num_blocks: int = 4):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(MockDITBlock(has_slice_tensor=(i >= 2)))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestDITOffload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU 环境不可用，跳过所有测试")
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        cls.cpu_device = torch.device("cpu")

    def setUp(self):
        self.num_blocks = 4
        self.model = MockDITModel(num_blocks=self.num_blocks)
        self.model.to(self.cpu_device)

        self.original_params = {}
        for blk_idx, blk in enumerate(self.model.blocks):
            self.original_params[blk_idx] = {
                'weight': blk.weight.data.clone(),
                'bias': blk.bias.data.clone(),
                'img_feat': blk.img_feat.data.clone(),
                'running_mean': blk.running_mean.clone()
            }
            if blk.slice_param is not None:
                self.original_params[blk_idx]['slice_param'] = blk.slice_param.data.clone()

    def tearDown(self):
        torch.npu.empty_cache()

    def test_enable_dit_offload_initialization(self):
        enable_offload(self.model, self.model.blocks)

        self.assertTrue(hasattr(self.model, 'h2d_stream'))
        self.assertTrue(hasattr(self.model, 'd2h_stream'))
        self.assertEqual(self.model.min_reserved_blocks_count, 2)
        self.assertEqual(len(self.model.event), self.num_blocks)

        for blk_idx in range(2):
            blk = self.model.blocks[blk_idx]
            for _, param in blk.named_parameters():
                self.assertEqual(param.data.device, self.device)
                self.assertFalse(hasattr(param, 'p_cpu'))
                self.assertNotEqual(param.data.untyped_storage().size(), 0)
            for _, buf in blk.named_buffers():
                self.assertEqual(buf.device, self.device)
                self.assertFalse(hasattr(buf, 'p_cpu'))
                self.assertNotEqual(buf.data.untyped_storage().size(), 0)

        for blk_idx in range(2, self.num_blocks):
            blk = self.model.blocks[blk_idx]
            for name, param in blk.named_parameters():
                self.assertTrue(hasattr(param, 'p_cpu'))
                self.assertEqual(param.p_cpu.device, self.cpu_device)
                self.assertTrue(param.p_cpu.is_pinned())
                self.assertEqual(param.data.shape, self.original_params[blk_idx][name].shape)
                self.assertEqual(param.data.untyped_storage().size(), 0)
                self.assertTrue(torch.allclose(param.p_cpu, self.original_params[blk_idx][name], atol=1e-6))

            for name, buf in blk.named_buffers():
                self.assertTrue(hasattr(buf, 'p_cpu'))
                self.assertEqual(buf.p_cpu.device, self.cpu_device)
                self.assertTrue(buf.p_cpu.is_pinned())
                self.assertEqual(buf.data.shape, self.original_params[blk_idx][name].shape)
                self.assertEqual(buf.data.untyped_storage().size(), 0)
                self.assertTrue(torch.allclose(buf.p_cpu, self.original_params[blk_idx][name], atol=1e-6))
    def test_full_forward_flow(self):
        enable_offload(self.model, self.model.blocks)
        for idx, blk in enumerate(self.model.blocks):
            blk.index = idx

        x = torch.randn(32, 16).to(self.device)
        output = self.model(x)

        self.assertIsNotNone(output)
        self.assertEqual(output.device, self.device)
        self.assertEqual(output.shape, x.shape)

        for blk_idx in range(2, self.num_blocks):
            blk = self.model.blocks[blk_idx]
            for name, param in blk.named_parameters():
                self.assertEqual(param.data.untyped_storage().size(), 0)
                self.assertEqual(param.data.shape, self.original_params[blk_idx][name].shape)

        for blk_idx in range(2):
            blk = self.model.blocks[blk_idx]
            for _, param in blk.named_parameters():
                self.assertEqual(param.data.device, self.device)
                self.assertNotEqual(param.data.untyped_storage().size(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)