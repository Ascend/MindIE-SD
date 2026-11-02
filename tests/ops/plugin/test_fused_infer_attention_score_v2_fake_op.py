#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
# This file must run in its own process, separate from
# tests/ops/fused_infer_attention_score/test_fused_infer_attention_score_v2.py.
# Importing _custom_ops registers fake ops into PyTorch's abstract impl table
# (process-level global state). aclnn shape inference reads this table and calls
# the Python fake in non-Meta paths, returning invalid values (0xFFFFFFFF).
# Keeping tests in separate files ensures each gets a clean process.

import os
import sys
import unittest

import torch

# Add project root to sys.path for mindiesd import.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Importing _custom_ops registers the fake ops into torch.ops.mindiesd: register_ops
# auto-loads the plugin .so when NPU is available, so the op schema exists for the
# Meta dispatch below.
import mindiesd.layers._custom_ops  # noqa: F401, E402  triggers fake op registration


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestFusedInferAttentionScoreV2FakeOp(unittest.TestCase):
    """Verify the eagle FIA v2 fake op output shape/dtype on the Meta device.

    ``fused_infer_attention_score_v2_fake`` (mindiesd/layers/_custom_ops.py) was added
    so that Dynamo/AOT fake-tensor propagation can infer this custom op's output
    metadata — without it the FP8 attention path breaks the graph at the op call
    site. These tests need the NPU environment only for the op schema (plugin .so,
    auto-loaded by register_ops.py); they exercise the Meta (abstract) implementation
    only — no real NPU kernel is dispatched, so no A5 gate is required (unlike
    test_block_sparse_attention.py, which runs the real A5-only kernel).
    """

    META = torch.device("meta")
    MAX_TOKEN = 2147483647

    def _meta_scales(self, shape):
        return torch.empty(shape, dtype=torch.float32, device=self.META)

    def _call_fake(
        self,
        q,
        k,
        v,
        layout="BNSD",
        out_dtype=None,
        scales=None,
        return_softmax_lse=False,
        num_heads=None,
        num_kv_heads=None,
        query_rope=None,
        quant_scale_out=None,
    ):
        """Mirror the kwargs of the real call site (fused_infer_attention_score
        ._fp8_attention_forward) so the fake is exercised through the same schema.

        ``num_heads`` / ``num_kv_heads`` override the layout-derived head counts, so
        cases that are not derivable from shape alone (NTD, explicit kv-head counts)
        can be exercised.

        The op schema defaults ``input_layout`` to ``'BSH'``, so an omitted layout must
        still derive the head counts from the BSH positions (``q.shape[2]``), otherwise
        the fake sees a bogus ``num_query_heads`` and its output metadata is wrong.
        """
        if scales is None:
            q_scale = k_scale = v_scale = None
        else:
            q_scale, k_scale, v_scale = scales
        head_layout = layout or "BSH"
        if num_heads is None:
            num_heads = q.shape[1] if head_layout != "BSH" else q.shape[2]
        if num_kv_heads is None:
            num_kv_heads = k.shape[1] if head_layout != "BSH" else k.shape[2]
        kwargs = {
            "num_query_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "softmax_scale": q.shape[-1] ** -0.5,
            "pre_tokens": self.MAX_TOKEN,
            "next_tokens": self.MAX_TOKEN,
            "query_quant_mode": 7,
            "key_quant_mode": 7,
            "value_quant_mode": 7,
            "dequant_scale_query": q_scale,
            "dequant_scale_key": k_scale,
            "dequant_scale_value": v_scale,
        }
        if layout is not None:
            kwargs["input_layout"] = layout
        if out_dtype is not None:
            kwargs["out_dtype"] = out_dtype
        if return_softmax_lse:
            kwargs["return_softmax_lse"] = True
        if query_rope is not None:
            kwargs["query_rope"] = query_rope
        if quant_scale_out is not None:
            kwargs["quant_scale_out"] = quant_scale_out
        return torch.ops.mindiesd.fused_infer_attention_score_v2(q, k, v, **kwargs)

    def test_fake_lse_empty_by_default(self):
        """Default ``return_softmax_lse=False`` must return an empty ``[0]`` lse.

        Mirrors ``InferLseOut`` in csrc/plugin/fused_infer_attention_score.cpp.
        """
        b, n, s, d = 1, 4, 128, 128
        q = torch.empty(b, n, s, d, dtype=torch.float16, device=self.META)
        k = torch.empty(b, n, s, d, dtype=torch.float16, device=self.META)
        v = torch.empty(b, n, s, d, dtype=torch.float16, device=self.META)
        _, lse = self._call_fake(q, k, v, layout="BNSD")
        self.assertEqual(tuple(lse.shape), (0,))
        self.assertEqual(lse.dtype, torch.float32)

    def test_fake_bnsd_shape(self):
        """BNSD: out matches query (fp8 path), softmax_lse=[B,N,S,1] fp32."""
        b, n, s, d = 1, 24, 512, 128
        q = torch.empty(b, n, s, d, dtype=torch.bfloat16, device=self.META)
        k = torch.empty(b, n, s, d, dtype=torch.bfloat16, device=self.META)
        v = torch.empty(b, n, s, d, dtype=torch.bfloat16, device=self.META)
        # Mirror the block-FP8 path: dequant scales present, out_dtype follows query.
        scales = (
            self._meta_scales([b, n, s, 1]),
            self._meta_scales([b, n, s, 1]),
            self._meta_scales([b, n, s, 1]),
        )
        attn_out, lse = self._call_fake(q, k, v, layout="BNSD", scales=scales, return_softmax_lse=True)
        self.assertEqual(tuple(attn_out.shape), (b, n, s, d))
        self.assertEqual(attn_out.dtype, torch.bfloat16)
        self.assertEqual(tuple(lse.shape), (b, n, s, 1))
        self.assertEqual(lse.dtype, torch.float32)

    def test_fake_tnd_shape(self):
        """TND: softmax_lse=[T,N,1]."""
        t, n, d = 1024, 8, 128
        q = torch.empty(t, n, d, dtype=torch.float16, device=self.META)
        k = torch.empty(t, n, d, dtype=torch.float16, device=self.META)
        v = torch.empty(t, n, d, dtype=torch.float16, device=self.META)
        attn_out, lse = self._call_fake(q, k, v, layout="TND", return_softmax_lse=True)
        self.assertEqual(tuple(attn_out.shape), (t, n, d))
        self.assertEqual(attn_out.dtype, torch.float16)
        self.assertEqual(tuple(lse.shape), (t, n, 1))

    def test_fake_bsh_explicit_shape(self):
        """BSH (explicit): softmax_lse=[B,H,S,1]."""
        b, s, h = 2, 256, 16
        q = torch.empty(b, s, h, dtype=torch.bfloat16, device=self.META)
        k = torch.empty(b, s, h, dtype=torch.bfloat16, device=self.META)
        v = torch.empty(b, s, h, dtype=torch.bfloat16, device=self.META)
        attn_out, lse = self._call_fake(q, k, v, layout="BSH", return_softmax_lse=True)
        self.assertEqual(tuple(attn_out.shape), (b, s, h))
        self.assertEqual(tuple(lse.shape), (b, h, s, 1))

    def test_fake_default_layout_is_bsh(self):
        """Omitting input_layout falls back to the BSH branch: lse=[B,H,S,1]."""
        b, s, h = 1, 128, 4
        q = torch.empty(b, s, h, dtype=torch.float16, device=self.META)
        k = torch.empty(b, s, h, dtype=torch.float16, device=self.META)
        v = torch.empty(b, s, h, dtype=torch.float16, device=self.META)
        attn_out, lse = self._call_fake(q, k, v, layout=None, return_softmax_lse=True)
        self.assertEqual(tuple(attn_out.shape), (b, s, h))
        self.assertEqual(tuple(lse.shape), (b, h, s, 1))

    def test_fake_out_dtype_override(self):
        """out_dtype kwarg wins over query dtype (query fp16 -> out bf16)."""
        b, n, s, d = 1, 4, 128, 128
        q = torch.empty(b, n, s, d, dtype=torch.float16, device=self.META)
        k = torch.empty(b, n, s, d, dtype=torch.float16, device=self.META)
        v = torch.empty(b, n, s, d, dtype=torch.float16, device=self.META)
        attn_out, lse = self._call_fake(q, k, v, layout="BNSD", out_dtype=torch.bfloat16)
        self.assertEqual(attn_out.dtype, torch.bfloat16)
        self.assertEqual(lse.dtype, torch.float32)

    def test_fake_output_dv_differs_from_dq(self):
        """Last output dim comes from value, not query (Dq != Dv).

        Mirrors ``GetValueD`` + ``InferAttentionOut``: the attention output's last
        dim is valueD, so a fake that copies ``query.shape`` produces a shape
        mismatch downstream in AOT shape propagation.
        """
        b, n, s, dq, dv = 2, 4, 7, 64, 128
        q = torch.empty(b, n, s, dq, dtype=torch.bfloat16, device=self.META)
        k = torch.empty(b, n, s, dq, dtype=torch.bfloat16, device=self.META)
        v = torch.empty(b, n, s, dv, dtype=torch.bfloat16, device=self.META)
        attn_out, lse = self._call_fake(q, k, v, layout="BNSD", num_heads=n, num_kv_heads=n, return_softmax_lse=True)
        self.assertEqual(tuple(attn_out.shape), (b, n, s, dv))
        self.assertEqual(attn_out.dtype, torch.bfloat16)
        self.assertEqual(tuple(lse.shape), (b, n, s, 1))

    def test_fake_int8_query_defaults_to_fp16(self):
        """int8 query without query_rope falls back to fp16 output (InferOutputDtype)."""
        b, n, s, d = 1, 4, 32, 128
        q = torch.empty(b, n, s, d, dtype=torch.int8, device=self.META)
        k = torch.empty(b, n, s, d, dtype=torch.int8, device=self.META)
        v = torch.empty(b, n, s, d, dtype=torch.int8, device=self.META)
        attn_out, _ = self._call_fake(q, k, v, layout="BNSD")
        self.assertEqual(attn_out.dtype, torch.float16)
        self.assertEqual(tuple(attn_out.shape), (b, n, s, d))

    def test_fake_int8_query_uses_query_rope_dtype(self):
        """int8 query with query_rope follows query_rope's dtype."""
        b, n, s, d = 1, 4, 32, 128
        q = torch.empty(b, n, s, d, dtype=torch.int8, device=self.META)
        k = torch.empty(b, n, s, d, dtype=torch.int8, device=self.META)
        v = torch.empty(b, n, s, d, dtype=torch.int8, device=self.META)
        rope = torch.empty(b, n, s, 64, dtype=torch.bfloat16, device=self.META)
        attn_out, _ = self._call_fake(q, k, v, layout="BNSD", query_rope=rope)
        self.assertEqual(attn_out.dtype, torch.bfloat16)

    def test_fake_quant_scale_out_forces_int8(self):
        """quant_scale_out present forces int8 output even for float query dtype."""
        b, n, s, d = 1, 4, 32, 128
        q = torch.empty(b, n, s, d, dtype=torch.bfloat16, device=self.META)
        k = torch.empty(b, n, s, d, dtype=torch.bfloat16, device=self.META)
        v = torch.empty(b, n, s, d, dtype=torch.bfloat16, device=self.META)
        scale_out = self._meta_scales([1])
        attn_out, _ = self._call_fake(q, k, v, layout="BNSD", quant_scale_out=scale_out)
        self.assertEqual(attn_out.dtype, torch.int8)

    def test_fake_ntd_lse_is_t_by_n(self):
        """NTD: query is [N, T, D], so lse is [T, N, 1] (GetQueryTnd + InferLseOut)."""
        n, t, d = 4, 7, 64
        q = torch.empty(n, t, d, dtype=torch.float16, device=self.META)
        k = torch.empty(n, t, d, dtype=torch.float16, device=self.META)
        v = torch.empty(n, t, d, dtype=torch.float16, device=self.META)
        attn_out, lse = self._call_fake(q, k, v, layout="NTD", num_heads=n, num_kv_heads=n, return_softmax_lse=True)
        self.assertEqual(tuple(attn_out.shape), (n, t, d))
        self.assertEqual(tuple(lse.shape), (t, n, 1))

    def test_fake_rejects_unknown_layout(self):
        """An input_layout outside FIA_LAYOUT_MAP must raise, not guess a shape."""
        q = torch.empty(1, 4, 8, 32, dtype=torch.float16, device=self.META)
        k = torch.empty(1, 4, 8, 32, dtype=torch.float16, device=self.META)
        v = torch.empty(1, 4, 8, 32, dtype=torch.float16, device=self.META)
        # NBSD is an output layout only; it is not a valid input_layout.
        with self.assertRaises(RuntimeError) as ctx:
            self._call_fake(q, k, v, layout="NBSD", num_heads=4, num_kv_heads=4)
        self.assertIn("unsupported", str(ctx.exception))

    def test_fake_rejects_zero_num_heads(self):
        """num_query_heads <= 0 must raise (matches the C++ TORCH_CHECK)."""
        q = torch.empty(1, 4, 8, 32, dtype=torch.float16, device=self.META)
        k = torch.empty(1, 4, 8, 32, dtype=torch.float16, device=self.META)
        v = torch.empty(1, 4, 8, 32, dtype=torch.float16, device=self.META)
        with self.assertRaises(RuntimeError) as ctx:
            self._call_fake(q, k, v, layout="BNSD", num_heads=0, num_kv_heads=4)
        self.assertIn("num_query_heads", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
