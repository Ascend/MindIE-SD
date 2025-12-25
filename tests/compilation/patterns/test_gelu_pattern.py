import unittest
import time
import torch

from mindiesd.compilation import MindieSDBackend


class GeluPatternModel(torch.nn.Module):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.gelu = torch.nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x)


class TestGeluCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x):
        compiled_model = torch.compile(model, backend=MindieSDBackend())

        t1 = time.perf_counter()
        output_compiled = compiled_model(x)
        t2 = time.perf_counter()
        output_original = model(x)
        t3 = time.perf_counter()

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(output_compiled, output_original)[0], 2**-7, msg="模式替换后输出不一致！")
        self.assertLess(t3-t2, t2-t1, msg="函数耗时超过预期阈值")

    def test_gelu_pattern_tanh_approx_bfloat16(self):
        model = GeluPatternModel(approximate="tanh")
        x = torch.randn(1, 4608, 12288, dtype=torch.bfloat16, device="npu")

        self._run_test_and_measure_time(model, x)

if __name__ == "__main__":
    unittest.main()
