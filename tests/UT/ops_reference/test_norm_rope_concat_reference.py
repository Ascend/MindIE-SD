#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""CPU reference tests for mindiesd.norm_rope_concat."""

import unittest

import torch

from tests.UT.ops_reference.reference_math import torch_norm_rope_concat_ref


class TestNormRopeConcatReference(unittest.TestCase):
    """Tests for the PyTorch reference implementation — always runs."""

    def test_ref_no_norm_no_rope_no_encoder(self):
        """Reference: pure transpose only."""
        B, S, H, D = 1, 4, 2, 8
        query = torch.randn(B, S, H, D, dtype=torch.float32)

        q_out, k_out, v_out, *_ = torch_norm_rope_concat_ref(query, query, query)

        self.assertEqual(q_out.shape, (B, H, S, D))
        torch.testing.assert_close(q_out, query.permute(0, 2, 1, 3), atol=1e-5, rtol=0)

    def test_ref_layer_norm_identity_weight(self):
        """Reference: LayerNorm with weight=1, bias=0 should match standard LN."""
        B, S, H, D = 2, 3, 4, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)

        q_out, k_out, v_out, q_mean, q_rstd, *_ = torch_norm_rope_concat_ref(
            query,
            query,
            query,
            norm_query_weight=nqw,
            norm_query_bias=nqb,
            norm_key_weight=nqw,
            norm_key_bias=nqb,
            norm_type=2,
            is_training=True,
        )

        # Compare with torch.nn.functional.layer_norm
        ln_ref = torch.nn.functional.layer_norm(query.float(), (D,), weight=nqw, bias=nqb, eps=1e-5)
        ln_ours = q_out.permute(0, 2, 1, 3)  # back to B,S,H,D
        torch.testing.assert_close(ln_ours.float(), ln_ref, atol=1e-5, rtol=1e-5)

    def test_ref_rms_norm_unit_weight(self):
        """Reference: RMSNorm with weight=1."""
        B, S, H, D = 1, 2, 4, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)

        q_out, *_ = torch_norm_rope_concat_ref(
            query,
            query,
            query,
            norm_query_weight=nqw,
            norm_key_weight=nqw,
            norm_type=4,
            is_training=False,
        )

        # RMS norm reference
        x = query.float()
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
        expected = (x / rms).permute(0, 2, 1, 3)
        torch.testing.assert_close(q_out.float(), expected, atol=1e-5, rtol=1e-5)

    def test_ref_concat_order_before(self):
        """Reference: concat_order=0 puts self before encoder."""
        B, H, D = 1, 2, 4
        Sq, Seq = 3, 2
        q_self = torch.ones(B, Sq, H, D, dtype=torch.float32)
        q_enc = torch.ones(B, Seq, H, D, dtype=torch.float32) * 2

        q_out, *_ = torch_norm_rope_concat_ref(
            q_self, q_self, q_self, encoder_query=q_enc, encoder_key=q_enc, encoder_value=q_enc, concat_order=0
        )

        # self (1.0) before encoder (2.0): first Sq values should be 1, last Seq values 2
        q_bhsd = q_out.permute(0, 2, 1, 3)  # back to (B, S'+S_encoder, H, D)
        self.assertTrue((q_bhsd[:, :Sq] == 1.0).all())
        self.assertTrue((q_bhsd[:, Sq:] == 2.0).all())

    def test_ref_concat_order_after(self):
        """Reference: concat_order=1 puts encoder before self."""
        B, H, D = 1, 2, 4
        Sq, Seq = 3, 2
        q_self = torch.ones(B, Sq, H, D, dtype=torch.float32)
        q_enc = torch.ones(B, Seq, H, D, dtype=torch.float32) * 2

        q_out, *_ = torch_norm_rope_concat_ref(
            q_self, q_self, q_self, encoder_query=q_enc, encoder_key=q_enc, encoder_value=q_enc, concat_order=1
        )

        # encoder (2.0) before self (1.0)
        q_bhsd = q_out.permute(0, 2, 1, 3)
        self.assertTrue((q_bhsd[:, :Seq] == 2.0).all())
        self.assertTrue((q_bhsd[:, Seq:] == 1.0).all())

    def test_ref_rope_interleave(self):
        """Reference: RoPE INTERLEAVE — verify (x1, x2) → (-x2, x1) pattern."""
        D = 4
        query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)
        rope_sin = torch.ones(1, D)
        rope_cos = torch.ones(1, D)

        q_out, *_ = torch_norm_rope_concat_ref(query, query, query, rope_sin=rope_sin, rope_cos=rope_cos, rope_type=1)

        # With cos=1, sin=1: (x1,x2) → (x1*1 + (-x2)*1, x2*1 + x1*1) → (x1-x2, x2+x1)
        # (1,2)→(1-2, 2+1)=(-1,3); (3,4)→(3-4,4+3)=(-1,7)
        q_bhsd = q_out
        self.assertAlmostEqual(q_bhsd[0, 0, 0, 0].item(), -1.0, places=5)
        self.assertAlmostEqual(q_bhsd[0, 0, 0, 1].item(), 3.0, places=5)
        self.assertAlmostEqual(q_bhsd[0, 0, 0, 2].item(), -1.0, places=5)
        self.assertAlmostEqual(q_bhsd[0, 0, 0, 3].item(), 7.0, places=5)

    def test_ref_mean_rstd_output_shapes(self):
        """Reference: training mode produces correct mean/rstd shapes."""
        B, S, H, D = 2, 5, 3, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)

        result = torch_norm_rope_concat_ref(
            query,
            query,
            query,
            norm_query_weight=nqw,
            norm_query_bias=nqb,
            norm_key_weight=nqw,
            norm_key_bias=nqb,
            norm_type=2,
            is_training=True,
        )

        # result tuple: (q_out, k_out, v_out, nq_mean, nq_rstd, nk_mean, nk_rstd, ...)
        self.assertEqual(len(result), 11)
        # LayerNorm mean/rstd
        self.assertEqual(result[3].shape, (B, S, H, 1))  # norm_query_mean
        self.assertEqual(result[4].shape, (B, S, H, 1))  # norm_query_rstd

    def test_ref_all_norm_type_combinations(self):
        """Reference: run all norm_type values to ensure no crash."""
        B, S, H, D = 1, 3, 2, 8
        q = torch.randn(B, S, H, D, dtype=torch.float32)
        w = torch.ones(D, dtype=torch.float32)
        b = torch.zeros(D, dtype=torch.float32)

        for norm_type in [0, 1, 2, 3, 4]:
            result = torch_norm_rope_concat_ref(
                q,
                q,
                q,
                norm_query_weight=w,
                norm_query_bias=b,
                norm_key_weight=w,
                norm_key_bias=b,
                norm_type=norm_type,
                is_training=(norm_type in (1, 2)),
            )
            self.assertEqual(result[0].shape, (B, H, S, D), f"Failed for norm_type={norm_type}")
            self.assertEqual(result[1].shape, (B, H, S, D), f"Failed for norm_type={norm_type}")
            self.assertEqual(result[2].shape, (B, H, S, D), f"Failed for norm_type={norm_type}")


if __name__ == '__main__':
    unittest.main()
