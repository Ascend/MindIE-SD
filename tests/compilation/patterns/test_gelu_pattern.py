import os
import unittest
import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


class GeluPatternModel(torch.nn.Module):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.gelu = torch.nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestGeluCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x)
        torch.npu.synchronize()

        compiled_time = benchmark(compiled_model, (x,))
        original_time = benchmark(model, (x,))

        output_compiled = compiled_model(x)
        output_original = model(x)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(output_compiled, output_original)[0], 2**-7, msg="模式替换后输出不一致！")
        self.assertLess(compiled_time, original_time, msg="compiled={:.6f}s >= original={:.6f}s".format(compiled_time, original_time))

    def test_gelu_pattern_tanh_approx_bfloat16(self):
        model = GeluPatternModel(approximate="tanh")
        x = torch.randn(4, 4608, 12288, dtype=torch.bfloat16, device="npu")

        self._run_test_and_measure_time(model, x)

if __name__ == "__main__":
    unittest.main()
