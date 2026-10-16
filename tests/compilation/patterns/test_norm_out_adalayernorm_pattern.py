import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.compilation.test_bench_utils import benchmark


class NormOutAdaLayerNormModel(torch.nn.Module):
    """复刻 diffusers FluxLayerNorm0 / qwen norm_out 的 (1+scale)[:, None] 形态
    (unsqueeze 在 add 之后, 与 norm1/norm2 的 modulation 不同)。"""

    def __init__(self, dim=512, eps=1e-6):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        ln_out = self.norm(x)
        return ln_out * (1 + scale).unsqueeze(1) + shift.unsqueeze(1)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestNormOutAdaLayerNormCompilationCase(unittest.TestCase):
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
        self.assertGreater(
            torch.cosine_similarity(output_compiled, output_original)[0],
            2**-7,
            msg="模式替换后输出不一致！",
        )
        self.assertGreater(compiled_time, 0)
        self.assertGreater(original_time, 0)

    def test_norm_out_adaln_bf16(self):
        model = NormOutAdaLayerNormModel(dim=3072)
        x = torch.randn(1, 4096, 3072, dtype=torch.bfloat16, device="npu")
        scale = torch.randn(1, 3072, dtype=torch.bfloat16, device="npu")
        shift = torch.randn(1, 3072, dtype=torch.bfloat16, device="npu")
        self._run_test_and_measure_time(model, x, scale, shift)

    def test_norm_out_adaln_fp32(self):
        model = NormOutAdaLayerNormModel(dim=3072)
        x = torch.randn(1, 512, 3072, dtype=torch.float32, device="npu")
        scale = torch.randn(1, 3072, dtype=torch.float32, device="npu")
        shift = torch.randn(1, 3072, dtype=torch.float32, device="npu")
        self._run_test_and_measure_time(model, x, scale, shift)


if __name__ == "__main__":
    unittest.main()
