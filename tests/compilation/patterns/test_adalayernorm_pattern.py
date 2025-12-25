import unittest
import torch
import time

from mindiesd.compilation import MindieSDBackend

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

        t1 = time.perf_counter()
        output_compiled = compiled_model(x, scale, shift)
        t2 = time.perf_counter()
        output_original = model(x, scale, shift)
        t3 = time.perf_counter()

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(output_compiled, output_original)[0], 2**-7, msg="模式替换后输出不一致！")
        self.assertLess(t3-t2, t2-t1, msg="函数耗时超过预期阈值")

    def test_adalayernorm_zero_pattern_diffusers_bfloat16(self):
        B, S, N, D = 1, 4096, 24, 128   # FLux.1-dev

        embedding_dim = N * D
        eps = 1e-06
        model = AdaLayerNormZeroPatternDiffusersModel(embedding_dim, epsilon=eps)

        x = torch.randn(B, S, embedding_dim, dtype=torch.bfloat16, device="npu")
        scale = torch.randn(B, embedding_dim, dtype=torch.bfloat16, device="npu")
        shift = torch.randn(B, embedding_dim, dtype=torch.bfloat16, device="npu")

        self._run_test_and_measure_time(model, x, scale, shift)

if __name__ == '__main__':
    unittest.main()