import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class MiniMaxRmsNormModel(torch.nn.Module):
    # torch 2.11: torch.rms_norm 在 freeze 前已被 Dynamo 分解为 pow/mean/rsqrt 链,
    # minimax_h3_rmsnorm_pattern 手写该链并在 before_freezing 命中。
    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = epsilon

    def forward(self, x, weight):
        return torch.rms_norm(x, (self.dim,), weight, self.eps)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestMiniMaxRmsNormPatternCompilationCase(unittest.TestCase):
    def test_minimax_rmsnorm_pattern_bfloat16_3d(self):
        b, s, d = 2, 16, 5376
        model = MiniMaxRmsNormModel(d)
        x = torch.randn(b, s, d, dtype=torch.bfloat16, device="npu")
        weight = torch.randn(d, dtype=torch.bfloat16, device="npu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(x, weight)
        torch.npu.synchronize()

        out_c = compiled(x, weight).reshape(1, -1).to(torch.float32)
        out_o = model(x, weight).reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="MiniMax RMSNorm pattern (3D) replacement output mismatch!")
        compiled_time = benchmark(compiled, (x, weight))
        original_time = benchmark(model, (x, weight))
        print(f"3D compiled={compiled_time:.4f}s original={original_time:.4f}s")

    def test_minimax_rmsnorm_pattern_bfloat16_4d(self):
        b, s, h, d = 2, 16, 56, 128
        model = MiniMaxRmsNormModel(d)
        x = torch.randn(b, s, h, d, dtype=torch.bfloat16, device="npu")
        weight = torch.randn(d, dtype=torch.bfloat16, device="npu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(x, weight)
        torch.npu.synchronize()

        out_c = compiled(x, weight).reshape(1, -1).to(torch.float32)
        out_o = model(x, weight).reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="MiniMax RMSNorm pattern (4D) replacement output mismatch!")
        compiled_time = benchmark(compiled, (x, weight))
        original_time = benchmark(model, (x, weight))
        print(f"4D compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
