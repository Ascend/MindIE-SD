import os
import sys
import unittest
import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark

_INT64_MAX = sys.maxsize


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class WanRopeModel(torch.nn.Module):
    # matches wan_rope_pattern exactly (interleaved rotation, 4D slice_scatter write-back)
    def forward(self, x, cos, sin):
        x_shape = list(x.shape)
        x_view = torch.ops.aten.view.default(x, x_shape[:-1] + [x_shape[-1] // 2, 2])
        x1, x2 = torch.ops.aten.unbind.int(x_view, -1)
        cos_e = torch.ops.aten.slice.Tensor(cos, 3, 0, _INT64_MAX, 2)
        sin_o = torch.ops.aten.slice.Tensor(sin, 3, 1, _INT64_MAX, 2)
        out = torch.ops.aten.empty_like.default(x)
        sub = x1 * cos_e - x2 * sin_o
        s0 = torch.ops.aten.slice.Tensor(out, 3, 0, _INT64_MAX, 2)
        c0 = torch.ops.aten.copy.default(s0, sub)
        ss0 = torch.ops.aten.slice_scatter.default(out, c0, 3, 0, _INT64_MAX, 2)
        add = x1 * sin_o + x2 * cos_e
        s1 = torch.ops.aten.slice.Tensor(ss0, 3, 1, _INT64_MAX, 2)
        c1 = torch.ops.aten.copy.default(s1, add)
        return torch.ops.aten.slice_scatter.default(ss0, c1, 3, 1, _INT64_MAX, 2)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestWanRopePatternCompilationCase(unittest.TestCase):
    def test_wan_rope_pattern_bfloat16(self):
        B, S, H, D = 2, 16, 4, 128
        model = WanRopeModel()
        x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="npu")
        cos = torch.randn(B, S, 1, D, dtype=torch.float32, device="npu")
        sin = torch.randn(B, S, 1, D, dtype=torch.float32, device="npu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(x, cos, sin)
        torch.npu.synchronize()

        out_c = compiled(x, cos, sin).reshape(1, -1).to(torch.float32)
        out_o = model(x, cos, sin).reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="Wan RoPE pattern replacement output mismatch!")
        compiled_time = benchmark(compiled, (x, cos, sin))
        original_time = benchmark(model, (x, cos, sin))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
