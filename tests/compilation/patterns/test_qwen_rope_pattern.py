import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.compilation.test_bench_utils import benchmark


class QwenRopePatternModel(torch.nn.Module):
    """复刻 diffusers transformer_qwenimage.apply_rotary_emb_qwen(use_real=False)
    在 compute-precision bf16 改写后的实数域等价形式(与 pattern 代码一致)。"""

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        xr, xi = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        cos = freqs.real.unsqueeze(1).to(x.dtype)  # [S, 1, D//2]
        sin = freqs.imag.unsqueeze(1).to(x.dtype)
        out_real = xr * cos - xi * sin
        out_imag = xr * sin + xi * cos
        x_out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
        return x_out.type_as(x)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestQwenRopeCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, freqs):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, freqs)
        torch.npu.synchronize()

        compiled_args = (x, freqs)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        output_compiled = compiled_model(x, freqs)
        output_original = model(x, freqs)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        self.assertGreater(
            torch.cosine_similarity(output_compiled, output_original)[0],
            2**-7,
            msg="模式替换后输出不一致！",
        )
        self.assertGreater(compiled_time, 0)
        self.assertGreater(original_time, 0)

    def test_qwen_rope_pattern_bf16(self):
        model = QwenRopePatternModel()
        x = torch.randn(1, 4096, 24, 128, dtype=torch.bfloat16, device="npu")
        freqs = torch.complex(
            torch.randn(4096, 64, dtype=torch.float32, device="npu"),
            torch.randn(4096, 64, dtype=torch.float32, device="npu"),
        )
        self._run_test_and_measure_time(model, x, freqs)

    def test_qwen_rope_pattern_fp32(self):
        model = QwenRopePatternModel()
        x = torch.randn(1, 512, 24, 128, dtype=torch.float32, device="npu")
        freqs = torch.complex(
            torch.randn(512, 64, dtype=torch.float32, device="npu"),
            torch.randn(512, 64, dtype=torch.float32, device="npu"),
        )
        self._run_test_and_measure_time(model, x, freqs)


if __name__ == "__main__":
    unittest.main()
