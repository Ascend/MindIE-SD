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

from mindiesd.compilation import MindieSDBackend

SCALE = 1.5


class MulAddModel(torch.nn.Module):
    def forward(self, x, y):
        return x * SCALE + y


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestMulAddCompilationCase(unittest.TestCase):
    def test_mul_add_pattern_bfloat16(self):
        model = MulAddModel()
        x = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
        y = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)
        torch.npu.synchronize()

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        output_compiled_f32 = output_compiled.reshape(1, -1).to(torch.float32)
        output_original_f32 = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(
            torch.cosine_similarity(output_compiled_f32, output_original_f32)[0],
            2 ** -7,
            msg="pattern replacement output mismatch",
        )

    def test_mul_add_pattern_float16(self):
        model = MulAddModel()
        x = torch.randn(4, 4096, dtype=torch.float16, device="npu")
        y = torch.randn(4, 4096, dtype=torch.float16, device="npu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)
        torch.npu.synchronize()

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        output_compiled_f32 = output_compiled.reshape(1, -1).to(torch.float32)
        output_original_f32 = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(
            torch.cosine_similarity(output_compiled_f32, output_original_f32)[0],
            2 ** -7,
            msg="pattern replacement output mismatch",
        )

    def test_mul_add_pattern_float32(self):
        model = MulAddModel()
        x = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        y = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)
        torch.npu.synchronize()

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        self.assertGreater(
            torch.cosine_similarity(
                output_compiled.reshape(1, -1), output_original.reshape(1, -1)
            )[0],
            2 ** -7,
            msg="pattern replacement output mismatch",
        )

    def test_mul_add_large_tensor(self):
        model = MulAddModel()
        x = torch.randn(32, 8192, dtype=torch.bfloat16, device="npu")
        y = torch.randn(32, 8192, dtype=torch.bfloat16, device="npu")
        args = (x, y)

        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(*args)
        torch.npu.synchronize()

        output_compiled = compiled_model(*args)
        output_original = model(*args)
        output_compiled_f32 = output_compiled.reshape(1, -1).to(torch.float32)
        output_original_f32 = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(
            torch.cosine_similarity(output_compiled_f32, output_original_f32)[0],
            2 ** -7,
            msg="pattern replacement output mismatch",
        )


if __name__ == "__main__":
    unittest.main()
