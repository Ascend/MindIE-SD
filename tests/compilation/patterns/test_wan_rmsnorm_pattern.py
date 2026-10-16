import os
import unittest
import torch

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
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


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestWanRmsNormPatternCompilationCase(unittest.TestCase):
    def test_wan_rmsnorm_pattern_bfloat16(self):
        B, S, D = 2, 16, 5120
        eps = 1e-6
        model = WanRmsNormModel(D, epsilon=eps)
        x = torch.randn(B, S, D, dtype=torch.float32, device="npu")
        weight = torch.randn(D, dtype=torch.bfloat16, device="npu")

        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(x, weight)
        torch.npu.synchronize()

        out_c = compiled(x, weight).reshape(1, -1).to(torch.float32)
        out_o = model(x, weight).reshape(1, -1).to(torch.float32)
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="Wan RMSNorm pattern replacement output mismatch!")
        compiled_time = benchmark(compiled, (x, weight))
        original_time = benchmark(model, (x, weight))
        print(f"compiled={compiled_time:.4f}s original={original_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
