import os
import unittest

import torch
import torch.nn.functional as F

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class RMSNormPatternModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 128, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, normalized_shape=(self.hidden_size,), weight=weight, eps=self.eps)


@unittest.skipIf(torch.__version__.startswith("2.1"), "")
class TestRMSNormCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, weight):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, weight)

        compiled_args = (x, weight)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        output_compiled = compiled_model(x, weight)
        output_original = model(x, weight)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled, output_original, rtol=1e-2, atol=1e-2)

        self.assertGreater(compiled_time, 0)
        self.assertGreater(original_time, 0)

    def test_rmsnorm_pattern_bfloat16(self):
        model = RMSNormPatternModel(hidden_size=128, eps=1e-6)
        x = torch.randn(1, 8, 24, 128, dtype=torch.bfloat16, device="cpu")
        weight = torch.randn(128, dtype=torch.bfloat16, device="cpu")
        self._run_test_and_measure_time(model, x, weight)


if __name__ == "__main__":
    unittest.main()
