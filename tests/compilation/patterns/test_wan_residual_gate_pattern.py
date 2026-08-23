import os
import unittest
import torch
import torch.nn.functional as F

from mindiesd.compilation import MindieSDBackend
from tests.compilation.test_bench_utils import benchmark


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class WanResidualGateModel(torch.nn.Module):
    """Positive case: 3D residual `x + y*gate` (register_replacement pattern must fire)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.ln = torch.nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, y, gate):
        res = x + y * gate
        return self.ln(res)


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class WanResidualGateNoLnModel(torch.nn.Module):
    """Negative case 1: residual add WITHOUT a native_layer_norm consumer (still correct)."""

    def forward(self, x, y, gate):
        return x + y * gate


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class WanResidualGateRopeLikeModel(torch.nn.Module):
    """Negative case 2: add(mul, mul) RoPE-like 4D chain -> must not be broken by the fusion."""

    def forward(self, a, b, c, d):
        m1 = a * c
        m2 = b * d
        return m1 + m2


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestWanResidualGatePassCompilationCase(unittest.TestCase):
    def _compare(self, model, *inputs):
        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(*inputs)
        torch.npu.synchronize()
        out_c = compiled(*inputs).reshape(1, -1).to(torch.float32)
        out_o = model(*inputs).reshape(1, -1).to(torch.float32)
        torch.npu.synchronize()
        self.assertGreater(torch.cosine_similarity(out_c, out_o)[0], 2**-7,
                           msg="residual+gate fused output mismatch!")
        return benchmark(compiled, inputs), benchmark(model, inputs)

    def test_residual_gate_anchored_bf16(self):
        B, S, D = 2, 16, 5120
        x = torch.randn(B, S, D, dtype=torch.bfloat16, device="npu")
        y = torch.randn(B, S, D, dtype=torch.bfloat16, device="npu")
        gate = torch.randn(B, 1, D, dtype=torch.bfloat16, device="npu")
        ct, ot = self._compare(WanResidualGateModel(D), x, y, gate)
        print(f"anchored: compiled={ct:.4f}s original={ot:.4f}s")

    def test_residual_gate_no_ln_no_crash(self):
        B, S, D = 2, 16, 5120
        x = torch.randn(B, S, D, dtype=torch.bfloat16, device="npu")
        y = torch.randn(B, S, D, dtype=torch.bfloat16, device="npu")
        gate = torch.randn(B, 1, D, dtype=torch.bfloat16, device="npu")
        self._compare(WanResidualGateNoLnModel(), x, y, gate)

    def test_rope_like_add_mul_mul_no_crash(self):
        B, S, H, Dh = 2, 16, 8, 64
        a = torch.randn(B, S, H, Dh, dtype=torch.bfloat16, device="npu")
        b = torch.randn(B, S, H, Dh, dtype=torch.bfloat16, device="npu")
        c = torch.randn(B, S, 1, Dh, dtype=torch.bfloat16, device="npu")
        d = torch.randn(B, S, 1, Dh, dtype=torch.bfloat16, device="npu")
        self._compare(WanResidualGateRopeLikeModel(), a, b, c, d)


if __name__ == '__main__':
    unittest.main()
