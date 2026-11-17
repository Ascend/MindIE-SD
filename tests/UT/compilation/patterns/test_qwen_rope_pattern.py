import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class QwenRopePatternModel(torch.nn.Module):
    """复刻 diffusers transformer_qwenimage.apply_rotary_emb_qwen(use_real=False)
    在 compute-precision bf16 改写后的实数域等价形式(与 pattern 代码一致)。
    """

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        xr, xi = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        cos = freqs.real.unsqueeze(1).to(x.dtype)  # [S, 1, D//2]
        sin = freqs.imag.unsqueeze(1).to(x.dtype)
        out_real = xr * cos - xi * sin
        out_imag = xr * sin + xi * cos
        x_out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
        return x_out.type_as(x)


class TestQwenRopeCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, freqs):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, freqs)

        compiled_args = (x, freqs)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        output_compiled = compiled_model(x, freqs)
        output_original = model(x, freqs)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled, output_original, rtol=1e-2, atol=1e-2)
        self.assertGreater(compiled_time, 0)
        self.assertGreater(original_time, 0)

    def test_qwen_rope_pattern_bf16(self):
        model = QwenRopePatternModel()
        x = torch.randn(1, 8, 24, 128, dtype=torch.bfloat16, device="cpu")
        freqs = torch.complex(
            torch.randn(8, 64, dtype=torch.float32, device="cpu"),
            torch.randn(8, 64, dtype=torch.float32, device="cpu"),
        )
        self._run_test_and_measure_time(model, x, freqs)

    def test_qwen_rope_pattern_fp32(self):
        model = QwenRopePatternModel()
        x = torch.randn(1, 8, 24, 128, dtype=torch.float32, device="cpu")
        freqs = torch.complex(
            torch.randn(8, 64, dtype=torch.float32, device="cpu"),
            torch.randn(8, 64, dtype=torch.float32, device="cpu"),
        )
        self._run_test_and_measure_time(model, x, freqs)


if __name__ == "__main__":
    unittest.main()
