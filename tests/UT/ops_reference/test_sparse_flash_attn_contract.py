# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""CPU-only contracts of the sparse flash attention frontends (no device kernel)."""

import unittest
from unittest import mock

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover - torch_npu is present in CI images
    torch_npu = None


def _has_mxfp4_dtypes():
    """Whether torch_npu exposes the FP4 (E2M1) and E8M0 scale dtypes.

    The MXFP4 path threads these dtype codes into the kernel call and evaluates
    them as dst_type arguments inside _mxfp4_quant_qkv, so even fully mocked
    tests need the attributes to exist. Skip instead of failing on torch_npu
    builds without them.
    """
    return torch_npu is not None and hasattr(torch_npu, "float4_e2m1fn_x2") and hasattr(torch_npu, "float8_e8m0fnu")


class TestResolveSparseTypeForA5(unittest.TestCase):
    """Pure-Python unit tests for _resolve_sparse_type_for_a5 (no NPU required).

    Guards the A5 inner_precise policy added for yjy_ac review 3: explicit
    rf_v3 requires inner_precise=4 on A5, while rf_v2 keeps its remap.
    """

    @mock.patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    def test_rf_v3_rejects_wrong_inner_precise_on_a5(self, _mock_is_a5):
        """Explicit rf_v3 with inner_precise != 4 must raise ParametersInvalid on A5."""
        from mindiesd.layers.flash_attn.sparse_flash_attn import _resolve_sparse_type_for_a5
        from mindiesd.utils.exception import ParametersInvalid

        with self.assertRaises(ParametersInvalid):
            _resolve_sparse_type_for_a5("rf_v3", 0)
        # inner_precise=4 passes through untouched
        self.assertEqual(_resolve_sparse_type_for_a5("rf_v3", 4), ("rf_v3", 4))

    @mock.patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    def test_rf_v2_still_remaps_to_4_on_a5(self, _mock_is_a5):
        """rf_v2 remap must still force inner_precise=4 (vllm-omni depends on it)."""
        from mindiesd.layers.flash_attn.sparse_flash_attn import _resolve_sparse_type_for_a5

        self.assertEqual(_resolve_sparse_type_for_a5("rf_v2", 0), ("rf_v3", 4))

    @mock.patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=False)
    def test_non_a5_passthrough(self, _mock_is_a5):
        """Non-A5 devices pass sparse_type/inner_precise through unchanged."""
        from mindiesd.layers.flash_attn.sparse_flash_attn import _resolve_sparse_type_for_a5

        self.assertEqual(_resolve_sparse_type_for_a5("rf_v3", 0), ("rf_v3", 0))
        self.assertEqual(_resolve_sparse_type_for_a5("rf_v2", 0), ("rf_v2", 0))


@unittest.skipUnless(_has_mxfp4_dtypes(), "MXFP4 tests require torch_npu.float4_e2m1fn_x2 / float8_e8m0fnu.")
class TestMxfp4ScaleReshapeGuards(unittest.TestCase):
    """CPU-runnable tests for the MXFP4 scale reshapes in _mxfp4_quant_qkv.

    Regression for the EZ1001 6D-scale bug: newer torch_npu returns the
    byte-grouped 5D scale directly from npu_dynamic_mx_quant, so the q/k scale
    reshape must be idempotent (dim==4 guard) and the V scale must go through
    the shared helper. npu_dynamic_mx_quant is mocked, so no NPU op runs.
    """

    B, S, N, D = 1, 128, 2, 64

    def _fake_dynamic_mx_quant(self, scale_dim5):
        """Return (data, scale) mimicking npu_dynamic_mx_quant for axis -1 / 2."""

        def _quant(tensor, dst_type, **kwargs):
            axis = kwargs.get("axis", -1)
            b, n, s, d = tensor.shape
            data = torch.zeros(b, n, s, d // 2, dtype=torch.uint8)
            if axis == -1:
                scale = torch.ones(b, n, s, d // 32, dtype=torch.uint8)
                if scale_dim5:
                    scale = scale.reshape(b, n, s, d // 64, 2)
            else:
                scale = torch.ones(b, n, s // 32, d, dtype=torch.uint8)
                if scale_dim5:
                    scale = scale.reshape(b, n, s // 64, d, 2)
            return data, scale

        return _quant

    def _run_quant_qkv(self, scale_dim5, **quant_kwargs):
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import _mxfp4_quant_qkv

        shape_bsnd = (self.B, self.S, self.N, self.D)
        q = torch.randn(shape_bsnd, dtype=torch.float32)
        k = torch.randn(shape_bsnd, dtype=torch.float32)
        v = torch.randn(shape_bsnd, dtype=torch.float32)
        q_rot = torch.eye(self.D, dtype=torch.float32)
        k_rot = torch.eye(self.D, dtype=torch.float32)

        with mock.patch(
            "mindiesd.quantization.layer._dynamic_mx_quant", side_effect=self._fake_dynamic_mx_quant(scale_dim5)
        ) as quant_mock:
            result = _mxfp4_quant_qkv(q, k, v, q_rot, k_rot, layout="BSND", **quant_kwargs)
        return result, quant_mock

    def test_4d_scales_are_regrouped_to_5d(self):
        """Older torch_npu returns 4D scales: they must be reshaped into the V3 5D layout."""
        (q_fp4, k_fp4, v_fp4, q_scale, k_scale, v_scale), _ = self._run_quant_qkv(scale_dim5=False)

        # q/k scale: [B,N,S,D/32] -> [B,N,S,D/64,2]; v scale: [B,N,S/32,D] -> [B,N,S/64,D,2].
        self.assertEqual(tuple(q_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(k_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(v_scale.shape), (self.B, self.N, self.S // 64, self.D, 2))

    def test_5d_scales_pass_through_unchanged(self):
        """Newer torch_npu returns byte-grouped 5D scales: reshapes must be idempotent."""
        (q_fp4, k_fp4, v_fp4, q_scale, k_scale, v_scale), _ = self._run_quant_qkv(scale_dim5=True)

        self.assertEqual(tuple(q_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(k_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(v_scale.shape), (self.B, self.N, self.S // 64, self.D, 2))

    def test_quant_kwargs_threading(self):
        """dst_type_max>0 and scale_alg must be forwarded to the quantizer; absent when unset."""
        _, quant_mock = self._run_quant_qkv(scale_dim5=False, scale_alg=2, dst_type_max=7.25)
        forwarded = quant_mock.call_args.kwargs
        self.assertEqual(forwarded["scale_alg"], 2)
        self.assertEqual(forwarded["dst_type_max"], 7.25)
        # Q/K quantize rowwise (axis -1), V columnwise along the sequence dim (axis 2).
        axes = [call.kwargs["axis"] for call in quant_mock.call_args_list]
        self.assertEqual(axes, [-1, -1, 2])

        _, quant_mock = self._run_quant_qkv(scale_dim5=False)
        forwarded = quant_mock.call_args.kwargs
        self.assertNotIn("scale_alg", forwarded)
        self.assertNotIn("dst_type_max", forwarded)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
