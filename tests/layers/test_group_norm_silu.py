import os
import unittest

import torch
import torch.nn.functional as F

from mindiesd import group_norm_silu


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestGroupNormSiLU(unittest.TestCase):
    def test_matches_group_norm_then_silu(self):
        dtype_configs = (
            (torch.float16, 1e-3, 1e-3),
            (torch.bfloat16, 1e-2, 1e-2),
        )
        for dtype, rtol, atol in dtype_configs:
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 64, 8, 8, dtype=dtype, device="npu")
                weight = torch.randn(64, dtype=x.dtype, device=x.device)
                bias = torch.randn(64, dtype=x.dtype, device=x.device)

                actual = group_norm_silu(x, weight, bias, group=32, eps=1e-5)
                expected = F.silu(F.group_norm(x, 32, weight, bias, eps=1e-5))
                torch.npu.synchronize()

                self.assertEqual(actual.shape, x.shape)
                self.assertEqual(actual.dtype, x.dtype)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


class TestGroupNormSiLUReference(unittest.TestCase):
    def test_fused_false_uses_pytorch_implementation(self):
        x = torch.randn(2, 8, 4, 4, dtype=torch.float32)
        weight = torch.randn(8, dtype=x.dtype)
        bias = torch.randn(8, dtype=x.dtype)

        actual = group_norm_silu(x, weight, bias, group=4, eps=1e-5, fused=False)
        expected = F.silu(F.group_norm(x, 4, weight, bias, eps=1e-5))

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, x.dtype)


if __name__ == "__main__":
    unittest.main()
