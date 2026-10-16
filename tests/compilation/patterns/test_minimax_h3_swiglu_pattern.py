import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class MiniMaxH3SwigluModel(torch.nn.Module):
    # 与 minimax_h3_swiglu_pattern 匹配: split.Tensor(proj, F, -1) -> silu(gate) -> mul
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, proj):
        hidden, gate = torch.split(proj, self.hidden_dim, dim=-1)
        return hidden * torch.nn.functional.silu(gate)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestMiniMaxH3SwigluPatternCompilationCase(unittest.TestCase):
    def test_minimax_h3_swiglu_pattern_bfloat16(self):
        # 真实尺寸: proj [S, 2*14336], split_size=14336 (pattern 常量精确匹配)
        b, s, half = 1, 4, 14336
        model = MiniMaxH3SwigluModel(half)
        proj = torch.randn(b, s, 2 * half, dtype=torch.bfloat16, device="npu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(proj)
        torch.npu.synchronize()

        out_c = compiled(proj).reshape(1, -1).to(torch.float32)
        out_o = model(proj).reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="MiniMax SwiGLU pattern replacement output mismatch!")
        compiled_time = benchmark(compiled, (proj,))
        original_time = benchmark(model, (proj,))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
