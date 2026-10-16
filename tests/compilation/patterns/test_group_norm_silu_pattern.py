import os
import unittest

import torch
import torch.nn.functional as F
import torch_npu
from torch.fx.experimental.proxy_tensor import make_fx

from mindiesd.compilation import MindieSDBackend


class GroupNormSiLUModel(torch.nn.Module):
    def forward(self, x, weight, bias):
        return F.silu(F.group_norm(x, 32, weight, bias, eps=1e-5))


class TestGroupNormSiLUPatternRegistration(unittest.TestCase):
    def test_native_group_norm_pattern_ignores_shape_scalars(self):
        model = GroupNormSiLUModel()
        x = torch.randn(1, 64, 16, 16, dtype=torch.float16)
        weight = torch.randn(64, dtype=x.dtype)
        bias = torch.randn(64, dtype=x.dtype)
        graph = make_fx(model, tracing_mode="fake")(x, weight, bias)

        MindieSDBackend.apply_pattern_match_passes(graph, (x, weight, bias))

        self.assertTrue(any(node.target is torch_npu.npu_group_norm_silu for node in graph.graph.nodes))
        self.assertFalse(any(node.target is torch.ops.aten.native_group_norm.default for node in graph.graph.nodes))


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestGroupNormSiLUCompilation(unittest.TestCase):
    def test_compile_matches_eager_for_different_spatial_shape(self):
        model = GroupNormSiLUModel()
        dtype_configs = (
            (torch.float16, 1e-3, 1e-3),
            (torch.bfloat16, 1e-2, 1e-2),
        )
        for dtype, rtol, atol in dtype_configs:
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 64, 16, 16, dtype=dtype, device="npu")
                weight = torch.randn(64, dtype=x.dtype, device=x.device)
                bias = torch.randn(64, dtype=x.dtype, device=x.device)

                compiled = torch.compile(model, backend=MindieSDBackend())
                actual = compiled(x, weight, bias)
                expected = model(x, weight, bias)
                torch.npu.synchronize()

                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
