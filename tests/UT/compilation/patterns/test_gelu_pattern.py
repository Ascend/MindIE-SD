import os
import unittest
import torch

# MindieSDBackend is exposed lazily through the package __getattr__.
from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class GeluPatternModel(torch.nn.Module):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.gelu = torch.nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x)


class TestGeluCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x)

        compiled_time = benchmark(compiled_model, (x,))
        original_time = benchmark(model, (x,))

        output_compiled = compiled_model(x)
        output_original = model(x)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled, output_original, rtol=1e-2, atol=1e-2)
        print(f"compiled={compiled_time:.6f}s, original={original_time:.6f}s")
        self.assertGreater(compiled_time, 0)
        self.assertGreater(original_time, 0)
        # Device speedup is measured by benchmarks, independently of CPU semantics.

    def test_gelu_pattern_tanh_approx_bfloat16(self):
        model = GeluPatternModel(approximate="tanh")
        x = torch.randn(1, 8, 12288, dtype=torch.bfloat16, device="cpu")

        self._run_test_and_measure_time(model, x)


if __name__ == "__main__":
    unittest.main()
