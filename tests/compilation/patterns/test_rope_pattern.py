import os
import unittest
import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark

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


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestRopeCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, cos, sin):
        # 关键：用自定义后端编译模型，自动触发 replace_pattern
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, cos, sin)
        torch.npu.synchronize()

        compiled_args = (x, cos, sin)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        # 验证输出一致性
        output_compiled = compiled_model(x, cos, sin)
        output_original = model(x, cos, sin)

        # 验证输出一致性
        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(output_compiled, output_original)[0], 2**-7, msg="模式替换后输出不一致！")
        self.assertLess(compiled_time, original_time, msg="compiled={:.6f}s >= original={:.6f}s".format(compiled_time, original_time))

    def test_rope_pattern_base(self):
        model = RopePatternModel()
        x = torch.randn(1, 4608, 24, 128, dtype=torch.bfloat16, device="npu")
        cos = torch.randn(1, 4608, 1, 128, dtype=torch.bfloat16, device="npu")
        sin = torch.randn(1, 4608, 1, 128, dtype=torch.bfloat16, device="npu")

        self._run_test_and_measure_time(model, x, cos, sin)

    def test_rope_pattern_diffusers_flux(self):
        model = RopePatternModelDiffusersFlux()
        x = torch.randn(1, 4608, 24, 128, dtype=torch.bfloat16, device="npu")
        cos = torch.randn(1, 4608, 1, 128, dtype=torch.float32, device="npu")
        sin = torch.randn(1, 4608, 1, 128, dtype=torch.float32, device="npu")

        self._run_test_and_measure_time(model, x, cos, sin)

if __name__ == '__main__':
    unittest.main()
