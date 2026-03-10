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
from mindiesd import RMSNorm


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.batch_size = 2
        self.sequence_length = 5
        self.hidden_dim = 4
        self.eps = 1e-6
        self.rmsnorm = RMSNorm(hidden_size=self.hidden_dim, eps=self.eps).to("npu")
        self.hidden_states = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to("npu")

    def test_fused_vs_non_fused(self):
        output_non_fused = self.rmsnorm(self.hidden_states, if_fused=False)
        output_fused = self.rmsnorm(self.hidden_states, if_fused=True)

        self.assertTrue(
            torch.allclose(output_non_fused, output_fused, atol=1e-6),
            "Fused and Non-Fused outputs do not match!"
        )


if __name__ == "__main__":
    unittest.main()