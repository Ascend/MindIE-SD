import os
import unittest

import torch

from mindiesd.compilation import MindieSDBackend
from mindiesd.compilation.patterns.qwen_residual_gate_pattern import _is_qwen_residual_gate_match


class _Node:
    def __init__(self, value):
        self.meta = {"val": value}


class _Match:
    def __init__(self, residual, branch, gate):
        self.kwargs = {
            "residual": _Node(residual),
            "branch": _Node(branch),
            "gate": _Node(gate),
        }


class TestQwenResidualGateMatchGuard(unittest.TestCase):
    @staticmethod
    def _match(residual_shape, branch_shape, gate_shape, dtype=torch.bfloat16):
        return _Match(
            torch.empty(residual_shape, dtype=dtype, device="meta"),
            torch.empty(branch_shape, dtype=dtype, device="meta"),
            torch.empty(gate_shape, dtype=dtype, device="meta"),
        )

    def test_accepts_qwen_broadcast_contract(self):
        self.assertTrue(_is_qwen_residual_gate_match(self._match((2, 16, 64), (2, 16, 64), (2, 1, 64))))

    def test_rejects_non_qwen_shapes(self):
        self.assertFalse(_is_qwen_residual_gate_match(self._match((2, 16, 64), (2, 16, 64), (2, 16, 64))))
        self.assertFalse(_is_qwen_residual_gate_match(self._match((2, 16, 64), (2, 8, 64), (2, 1, 64))))
        self.assertFalse(_is_qwen_residual_gate_match(self._match((2, 16, 8, 64), (2, 16, 8, 64), (2, 1, 64))))


class QwenResidualGateModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, residual, branch, gate):
        return self.norm(residual + gate * branch)


class QwenResidualGateCommutativeModel(torch.nn.Module):
    def forward(self, residual, branch, gate):
        return residual + branch * gate


class QwenResidualGateInvalidGateModel(torch.nn.Module):
    def forward(self, residual, branch, gate):
        return residual + gate * branch


class RopeLikeModel(torch.nn.Module):
    def forward(self, a, b, cos, sin):
        return a * cos + b * sin


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestQwenResidualGatePattern(unittest.TestCase):
    @staticmethod
    def _assert_close(model, *inputs):
        compiled = torch.compile(model, backend=MindieSDBackend())
        actual = compiled(*inputs)
        expected = model(*inputs)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_qwen_gate_broadcast_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch_size, sequence_length in ((1, 16), (2, 31)):
                with self.subTest(dtype=dtype, batch_size=batch_size, sequence_length=sequence_length):
                    residual = torch.randn(batch_size, sequence_length, 64, dtype=dtype, device="npu")
                    branch = torch.randn_like(residual)
                    gate = torch.randn(batch_size, 1, 64, dtype=dtype, device="npu")
                    self._assert_close(QwenResidualGateModel(64), residual, branch, gate)

    def test_commutative_mul_order_is_numerically_safe(self):
        residual = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")
        branch = torch.randn_like(residual)
        gate = torch.randn(2, 1, 64, dtype=torch.bfloat16, device="npu")
        self._assert_close(QwenResidualGateCommutativeModel(), residual, branch, gate)

    def test_full_sized_gate_is_not_replaced(self):
        residual = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")
        branch = torch.randn_like(residual)
        full_gate = torch.randn_like(residual)
        self._assert_close(QwenResidualGateInvalidGateModel(), residual, branch, full_gate)

    def test_rope_like_add_mul_mul_is_not_replaced(self):
        a = torch.randn(2, 16, 8, 64, dtype=torch.bfloat16, device="npu")
        b = torch.randn_like(a)
        cos = torch.randn(2, 16, 1, 64, dtype=torch.bfloat16, device="npu")
        sin = torch.randn_like(cos)
        self._assert_close(RopeLikeModel(), a, b, cos, sin)
