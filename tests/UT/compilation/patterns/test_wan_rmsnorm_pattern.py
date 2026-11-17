import os
import unittest
import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class WanRmsNormModel(torch.nn.Module):
    # matches wan_rmsnorm_pattern exactly (fp32, no casts)
    def __init__(self, dim: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        # 真实图常量: eps 经 float32 舍入 (9.999999974752427e-07)
        self.eps = torch.tensor(epsilon, dtype=torch.float32).item()

    def forward(self, x, weight):
        variance = torch.ops.aten.pow.Tensor_Scalar(x, 2)
        mean = torch.ops.aten.mean.dim(variance, [x.dim() - 1], True)
        add = torch.ops.aten.add.Scalar(mean, self.eps)
        rsqrt = torch.ops.aten.rsqrt.default(add)
        result = x * rsqrt
        return result * weight


class TestWanRmsNormPatternCompilationCase(unittest.TestCase):
    def test_wan_rmsnorm_pattern_bfloat16(self):
        B, S, D = 2, 16, 5120
        eps = 1e-6
        model = WanRmsNormModel(D, epsilon=eps)
        x = torch.randn(B, S, D, dtype=torch.float32, device="cpu")
        weight = torch.randn(D, dtype=torch.bfloat16, device="cpu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(x, weight)

        out_c = compiled(x, weight).reshape(1, -1).to(torch.float32)
        out_o = model(x, weight).reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(out_c, out_o, rtol=1e-2, atol=1e-2)
        compiled_time = benchmark(compiled, (x, weight))
        original_time = benchmark(model, (x, weight))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
