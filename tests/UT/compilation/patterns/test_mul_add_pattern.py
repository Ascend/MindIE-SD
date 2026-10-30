#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
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

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module

SCALE = 1.5


class MulAddModel(torch.nn.Module):
    def forward(self, x, y):
        return x * SCALE + y


class TestMulAddCompilationCase(unittest.TestCase):
    def test_mul_add_pattern_bfloat16(self):
        model = MulAddModel()
        x = torch.randn(4, 4096, dtype=torch.bfloat16, device="cpu")
        y = torch.randn(4, 4096, dtype=torch.bfloat16, device="cpu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        output_compiled_f32 = output_compiled.reshape(1, -1).to(torch.float32)
        output_original_f32 = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled_f32, output_original_f32, rtol=1e-2, atol=1e-2)

    def test_mul_add_pattern_float16(self):
        model = MulAddModel()
        x = torch.randn(4, 4096, dtype=torch.float16, device="cpu")
        y = torch.randn(4, 4096, dtype=torch.float16, device="cpu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        output_compiled_f32 = output_compiled.reshape(1, -1).to(torch.float32)
        output_original_f32 = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled_f32, output_original_f32, rtol=1e-2, atol=1e-2)

    def test_mul_add_pattern_float32(self):
        model = MulAddModel()
        x = torch.randn(4, 4096, dtype=torch.float32, device="cpu")
        y = torch.randn(4, 4096, dtype=torch.float32, device="cpu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        torch.testing.assert_close(output_compiled.reshape(1, -1), output_original.reshape(1, -1), rtol=1e-2, atol=1e-2)

    def test_mul_add_large_tensor(self):
        model = MulAddModel()
        x = torch.randn(32, 8192, dtype=torch.bfloat16, device="cpu")
        y = torch.randn(32, 8192, dtype=torch.bfloat16, device="cpu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        output_compiled_f32 = output_compiled.reshape(1, -1).to(torch.float32)
        output_original_f32 = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled_f32, output_original_f32, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
