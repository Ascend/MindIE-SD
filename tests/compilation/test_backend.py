import unittest
import torch
import time

import sys
from packaging.version import Version
from pathlib import Path
from mindiesd.compilation import MindieSDBackend
from mindiesd.compilation.passes.register_pattern_to_pass import register_pattern_to_pass, PatternBase


class SamplePass(PatternBase):
    @staticmethod
    def name():
        return __class__.__name__

    @staticmethod
    def inputs():
        x = torch.empty(2, 2, device="meta")
        y = torch.empty(2, 2, device="meta")
        z = torch.empty(2, 2, device="meta")
        return [x, y, z]

    @staticmethod
    def pattern(x, y, z):
        def source_pattern(x, y, z):
            x_mul = torch.mul(x, z)
            y_mul = torch.mul(y, z)
            return torch.add(x_mul, y_mul)
        return source_pattern(x, y, z)

    @staticmethod
    def replacement(x, y, z):
        def target_pattern(x, y, z):
            # 实际场景可替换为硬件优化内核（如 cuDNN 融合算子）
            def fused_mul_add(x, y, z):
                s = x + y
                return torch.mul(s, z)
            return fused_mul_add(x, y, z)
        return target_pattern(x, y, z)

register_pattern_to_pass(SamplePass)

class TestModel(torch.nn.Module):
    def forward(self, x, y, z):
        x_mul = torch.mul(x, z)
        y_mul = torch.mul(y, z)
        return torch.add(x_mul, y_mul)  # 匹配源模式


class TestCompilationCase(unittest.TestCase):
    def test_fused_kernel_perf_and_accuracy(self):
        model = TestModel()
        x = torch.randint(0, 10, (32, 64))
        y = torch.randint(0, 10, (32, 64))
        z = torch.randint(0, 10, (32, 64))

        # 关键：用自定义后端编译模型，自动触发 replace_pattern
        compiled_model = torch.compile(model, backend=MindieSDBackend())

        # 运行编译后的模型（首次运行会触发编译，后续复用缓存）
        t1 = time.perf_counter()
        output_compiled = compiled_model(x, y, z)
        t2 = time.perf_counter()
        output_original = model(x, y, z)
        t3 = time.perf_counter()

        # 验证输出一致性
        assert torch.allclose(output_original, output_compiled), "模式替换后输出不一致！"
        self.assertLess(t3-t2, t2-t1, msg="函数耗时超过预期阈值")

if __name__ == '__main__':
    unittest.main()
