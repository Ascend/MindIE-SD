import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class MiniMaxH3AdaLnModel(torch.nn.Module):
    # 与 minimax_h3_adaln_pattern 匹配: index_select 表行 -> x*(1+scale)+shift
    def forward(self, x, scale_table, shift_table, indices):
        scale = scale_table.index_select(0, indices)
        shift = shift_table.index_select(0, indices)
        return x * (1.0 + scale) + shift


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestMiniMaxH3AdaLnPatternCompilationCase(unittest.TestCase):
    def test_minimax_h3_adaln_pattern_bfloat16(self):
        b, s, d = 1, 64, 5376
        model = MiniMaxH3AdaLnModel()
        x = torch.randn(b, s, d, dtype=torch.bfloat16, device="npu")
        scale_table = torch.randn(3, d, dtype=torch.bfloat16, device="npu") * 0.1
        shift_table = torch.randn(3, d, dtype=torch.bfloat16, device="npu") * 0.1
        indices = torch.randint(0, 3, (s,), dtype=torch.int64, device="npu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(x, scale_table, shift_table, indices)
        torch.npu.synchronize()

        out_c = compiled(x, scale_table, shift_table, indices).reshape(1, -1).to(torch.float32)
        out_o = model(x, scale_table, shift_table, indices).reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="MiniMax AdaLN pattern replacement output mismatch!")
        compiled_time = benchmark(compiled, (x, scale_table, shift_table, indices))
        original_time = benchmark(model, (x, scale_table, shift_table, indices))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
