import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class MiniMaxH3RopeModel(torch.nn.Module):
    # 与 minimax_h3_rope_pattern 匹配: 对每 head 前 96/128 通道做 rotate_half
    def __init__(self, rotary_dim: int = 96):
        super().__init__()
        self.rotary_dim = rotary_dim

    def forward(self, x, cos, sin):
        rotary_dim = self.rotary_dim
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        cos4 = cos.to(x.dtype)[None, :, None, :]
        sin4 = sin.to(x.dtype)[None, :, None, :]
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        out = x_rot * cos4 + rotated * sin4
        return torch.cat((out, x_pass), dim=-1).contiguous()


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestMiniMaxH3RopePatternCompilationCase(unittest.TestCase):
    def test_minimax_h3_rope_pattern_bfloat16(self):
        b, s, h, d = 1, 4, 56, 128
        model = MiniMaxH3RopeModel()
        x = torch.randn(b, s, h, d, dtype=torch.bfloat16, device="npu")
        cos = torch.randn(s, 96, dtype=torch.float32, device="npu")
        sin = torch.randn(s, 96, dtype=torch.float32, device="npu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(x, cos, sin)
        torch.npu.synchronize()

        out_c = compiled(x, cos, sin).reshape(1, -1).to(torch.float32)
        out_o = model(x, cos, sin).reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="MiniMax RoPE pattern replacement output mismatch!")
        compiled_time = benchmark(compiled, (x, cos, sin))
        original_time = benchmark(model, (x, cos, sin))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
