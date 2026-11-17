import os
import unittest
import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class RopePatternModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        x_out = (x * cos + x_rotated * sin).to(x.dtype)
        return x_out


class RopePatternModelDiffusersFlux(torch.nn.Module):
    # Example Codes Based on diffusers.models.embeddings.apply_rotary_emb
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        x_out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return x_out


class TestRopeCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, cos, sin):
        # 关键：用自定义后端编译模型，自动触发 replace_pattern
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, cos, sin)

        compiled_args = (x, cos, sin)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        # 验证输出一致性
        output_compiled = compiled_model(x, cos, sin)
        output_original = model(x, cos, sin)

        # 验证输出一致性
        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled, output_original, rtol=1e-2, atol=1e-2)
        self.assertGreater(compiled_time, 0)
        self.assertGreater(original_time, 0)

    def test_rope_pattern_base(self):
        model = RopePatternModel()
        x = torch.randn(1, 8, 24, 128, dtype=torch.bfloat16, device="cpu")
        cos = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cpu")
        sin = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cpu")

        self._run_test_and_measure_time(model, x, cos, sin)

    def test_rope_pattern_diffusers_flux(self):
        model = RopePatternModelDiffusersFlux()
        x = torch.randn(1, 8, 24, 128, dtype=torch.bfloat16, device="cpu")
        cos = torch.randn(1, 8, 1, 128, dtype=torch.float32, device="cpu")
        sin = torch.randn(1, 8, 1, 128, dtype=torch.float32, device="cpu")

        self._run_test_and_measure_time(model, x, cos, sin)


if __name__ == '__main__':
    unittest.main()
