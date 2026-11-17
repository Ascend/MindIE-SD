import os
import unittest
import torch

# MindieSDBackend is exposed lazily through the package __getattr__.
from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class AdaLayerNormZeroPatternDiffusersModel(torch.nn.Module):
    # Reference: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/normalization.py#L131
    def __init__(self, embedding_dim: int, epsilon: float = 1e-06) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=epsilon)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        out = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return out


class TestAdaLayerNormPatternCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, scale, shift):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, scale, shift)

        compiled_args = (x, scale, shift)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        output_compiled = compiled_model(x, scale, shift)
        output_original = model(x, scale, shift)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled, output_original, rtol=1e-2, atol=1e-2)
        print(f"compiled={compiled_time:.6f}s, original={original_time:.6f}s")
        self.assertGreater(compiled_time, 0)
        self.assertGreater(original_time, 0)
        # CPU unit tests verify graph semantics; device performance belongs to benchmarks.

    def test_adalayernorm_zero_pattern_diffusers_bfloat16(self):
        B, S, N, D = 2, 8, 24, 128  # FLux.1-dev

        embedding_dim = N * D
        eps = 1e-06
        model = AdaLayerNormZeroPatternDiffusersModel(embedding_dim, epsilon=eps)

        x = torch.randn(B, S, embedding_dim, dtype=torch.bfloat16, device="cpu")
        scale = torch.randn(B, embedding_dim, dtype=torch.bfloat16, device="cpu")
        shift = torch.randn(B, embedding_dim, dtype=torch.bfloat16, device="cpu")

        self._run_test_and_measure_time(model, x, scale, shift)


if __name__ == '__main__':
    unittest.main()
