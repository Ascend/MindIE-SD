import os
import unittest
import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
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


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestAdaLayerNormPatternCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, scale, shift):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, scale, shift)
        torch.npu.synchronize()

        compiled_args = (x, scale, shift)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        output_compiled = compiled_model(x, scale, shift)
        output_original = model(x, scale, shift)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(output_compiled, output_original)[0], 2**-7, msg="模式替换后输出不一致！")
        self.assertLess(compiled_time, original_time, msg="compiled={:.6f}s >= original={:.6f}s".format(compiled_time, original_time))

    def test_adalayernorm_zero_pattern_diffusers_bfloat16(self):
        B, S, N, D = 4, 4096, 24, 128   # FLux.1-dev

        embedding_dim = N * D
        eps = 1e-06
        model = AdaLayerNormZeroPatternDiffusersModel(embedding_dim, epsilon=eps)

        x = torch.randn(B, S, embedding_dim, dtype=torch.bfloat16, device="npu")
        scale = torch.randn(B, embedding_dim, dtype=torch.bfloat16, device="npu")
        shift = torch.randn(B, embedding_dim, dtype=torch.bfloat16, device="npu")

        self._run_test_and_measure_time(model, x, scale, shift)

if __name__ == '__main__':
    unittest.main()