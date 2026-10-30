import os
import unittest

import torch
import torch.nn.functional as F

from mindiesd import group_norm_silu


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
