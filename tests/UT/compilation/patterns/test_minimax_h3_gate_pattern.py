import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class MiniMaxH3GateModel(torch.nn.Module):
    # 与 minimax_h3_gate_pattern 匹配: residual + gate_table[idx] * value
    def forward(self, residual, value, gate_table, indices):
        gate = gate_table.index_select(0, indices)
        return residual + gate * value


class TestMiniMaxH3GatePatternCompilationCase(unittest.TestCase):
    def test_minimax_h3_gate_pattern_bfloat16(self):
        b, s, d = 1, 64, 5376
        model = MiniMaxH3GateModel()
        residual = torch.randn(b, s, d, dtype=torch.bfloat16, device="cpu")
        value = torch.randn(b, s, d, dtype=torch.bfloat16, device="cpu") * 0.1
        gate_table = torch.randn(3, d, dtype=torch.bfloat16, device="cpu") * 0.1
        indices = torch.randint(0, 3, (s,), dtype=torch.int64, device="cpu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(residual, value, gate_table, indices)

        out_c = compiled(residual, value, gate_table, indices).reshape(1, -1).to(torch.float32)
        out_o = model(residual, value, gate_table, indices).reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(out_c, out_o, rtol=1e-2, atol=1e-2)
        compiled_time = benchmark(compiled, (residual, value, gate_table, indices))
        original_time = benchmark(model, (residual, value, gate_table, indices))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
