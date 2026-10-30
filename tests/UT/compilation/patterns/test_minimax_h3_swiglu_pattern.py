import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class MiniMaxH3SwigluModel(torch.nn.Module):
    # 与 minimax_h3_swiglu_pattern 匹配: split.Tensor(proj, F, -1) -> silu(gate) -> mul
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, proj):
        hidden, gate = torch.split(proj, self.hidden_dim, dim=-1)
        return hidden * torch.nn.functional.silu(gate)


class TestMiniMaxH3SwigluPatternCompilationCase(unittest.TestCase):
    def test_minimax_h3_swiglu_pattern_bfloat16(self):
        # 真实尺寸: proj [S, 2*14336], split_size=14336 (pattern 常量精确匹配)
        b, s, half = 1, 4, 14336
        model = MiniMaxH3SwigluModel(half)
        proj = torch.randn(b, s, 2 * half, dtype=torch.bfloat16, device="cpu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(proj)

        out_c = compiled(proj).reshape(1, -1).to(torch.float32)
        out_o = model(proj).reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(out_c, out_o, rtol=1e-2, atol=1e-2)
        compiled_time = benchmark(compiled, (proj,))
        original_time = benchmark(model, (proj,))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
