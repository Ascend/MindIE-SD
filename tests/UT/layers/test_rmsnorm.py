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
from unittest.mock import patch
import torch
from torch import nn
from mindiesd import RMSNorm
from mindiesd.utils import ParametersInvalid


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.batch_size = 2
        self.sequence_length = 5
        self.hidden_dim = 4
        self.eps = 1e-6
        self.rmsnorm = RMSNorm(hidden_size=self.hidden_dim, eps=self.eps)
        self.hidden_states = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)

    def test_fused_vs_non_fused(self):
        output_non_fused = self.rmsnorm(self.hidden_states, if_fused=False)
        with patch('torch_npu.npu_rms_norm', return_value=(output_non_fused, None), create=True) as backend:
            output_fused = self.rmsnorm(self.hidden_states, if_fused=True)
        backend.assert_called_once_with(self.hidden_states, self.rmsnorm.weight, epsilon=self.eps)

        self.assertTrue(
            torch.allclose(output_non_fused, output_fused, atol=1e-6), "Fused and Non-Fused outputs do not match!"
        )

    def test_invalid_rank_rejected_before_backend(self):
        for shape in ((4,), (1,) * 9):
            with self.subTest(shape=shape), self.assertRaises(ParametersInvalid):
                self.rmsnorm(torch.ones(shape))


if __name__ == "__main__":
    unittest.main()
