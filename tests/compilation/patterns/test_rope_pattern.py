import unittest
import torch
import time

from mindiesd.compilation import MindieSDBackend

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

        # 运行编译后的模型（首次运行会触发编译，后续复用缓存）
        t1 = time.perf_counter()
        output_compiled = compiled_model(x, cos, sin)
        t2 = time.perf_counter()
        output_original = model(x, cos, sin)
        t3 = time.perf_counter()

        # 验证输出一致性
        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(output_compiled, output_original)[0], 2**-7, msg="模式替换后输出不一致！")
        self.assertLess(t3-t2, t2-t1, msg="函数耗时超过预期阈值")

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
