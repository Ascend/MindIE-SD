import os
import unittest
import torch

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from tests.UT.compilation.test_bench_utils import benchmark


class WanAdaLayerNormModel(torch.nn.Module):
    # Reference: diffusers WanTransformerBlock.forward
    #   norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa)
    #                         + shift_msa).type_as(hidden_states)
    # The .float() cast is upstream; here x is already fp32 (same as the pattern).
    def __init__(self, dim: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = epsilon

    def forward(self, x, scale, shift):
        ln_out = torch.ops.aten.native_layer_norm(x, [self.dim], None, None, self.eps)[0]
        return ln_out * (scale + 1) + shift


class TestWanAdaLayerNormPatternCompilationCase(unittest.TestCase):
    def _run_test_and_measure_time(self, model, x, scale, shift):
        compiled_model = torch.compile(model, backend=MindieSDBackend())
        compiled_model(x, scale, shift)

        compiled_args = (x, scale, shift)
        compiled_time = benchmark(compiled_model, compiled_args)
        original_time = benchmark(model, compiled_args)

        output_compiled = compiled_model(x, scale, shift)
        output_original = model(x, scale, shift)

        output_compiled = output_compiled.reshape(1, -1).to(torch.float32)
        output_original = output_original.reshape(1, -1).to(torch.float32)
        torch.testing.assert_close(output_compiled, output_original, rtol=1e-2, atol=1e-2)
        return compiled_time, original_time

    def test_wan_adalayernorm_pattern_bfloat16(self):
        # Wan2.2-T2V-A14B: dim = 40 heads * 128 head_dim = 5120, eps=1e-6
        B, S, D = 2, 16, 5120
        eps = 1e-6
        model = WanAdaLayerNormModel(D, epsilon=eps)

        x = torch.randn(B, S, D, dtype=torch.float32, device="cpu")
        scale = torch.randn(B, 1, D, dtype=torch.float32, device="cpu")
        shift = torch.randn(B, 1, D, dtype=torch.float32, device="cpu")

        self._run_test_and_measure_time(model, x, scale, shift)


if __name__ == '__main__':
    unittest.main()
