#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
Unit tests for mindiesd.norm_rope_concat operator.

Tests the fused Norm + RoPE + Concat operator across various configurations:
  - norm_type: 0(NONE), 1(LAYER_NORM), 2(LAYER_NORM_AFFINE), 3(RMS_NORM), 4(RMS_NORM_AFFINE)
  - rope_type: 0(NONE), 1(INTERLEAVE), 2(HALF)
  - concat_order: 0(BEFORE_ENCODER), 1(AFTER_ENCODER)
  - is_training: True/False
"""

import os
import unittest

import torch


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================


def torch_layer_norm(x, weight=None, bias=None, eps=1e-5):
    """Reference LayerNorm: normalizes over the last dimension."""
    x_fp32 = x.float()
    mean = torch.mean(x_fp32, dim=-1, keepdim=True)
    var = torch.var(x_fp32, dim=-1, correction=0, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    out = (x_fp32 - mean) * rstd
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out, mean.squeeze(-1), rstd.squeeze(-1)


def torch_rms_norm(x, weight=None, eps=1e-5):
    """Reference RMSNorm: normalizes over the last dimension (no bias)."""
    x_fp32 = x.float()
    rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + eps)
    out = x_fp32 / rms
    if weight is not None:
        out = out * weight.float()
    return out


def torch_rotary_emb(x, rope_sin, rope_cos, rope_type):
    """Reference RoPE: INTERLEAVE(1) or HALF(2)."""
    if rope_type == 1:  # INTERLEAVE
        x_reshaped = x.float().view(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(3)
    elif rope_type == 2:  # HALF
        half_dim = x.shape[-1] // 2
        x1, x2 = x.float()[..., :half_dim], x.float()[..., half_dim:]
        rotated = torch.cat([-x2, x1], dim=-1)
    else:
        return x
    return x.float() * rope_cos.float() + rotated * rope_sin.float()


def torch_norm_rope_concat_ref(
    query, key, value,
    encoder_query=None, encoder_key=None, encoder_value=None,
    norm_query_weight=None, norm_query_bias=None,
    norm_key_weight=None, norm_key_bias=None,
    norm_added_query_weight=None, norm_added_query_bias=None,
    norm_added_key_weight=None, norm_added_key_bias=None,
    rope_sin=None, rope_cos=None,
    norm_type=0, norm_added_type=0, rope_type=0,
    concat_order=0, eps=1e-5, is_training=False,
):
    """
    Pure PyTorch reference for norm_rope_concat.
    Input layout: (B, S, H, D)  →  Output layout: (B, H, S', D) where S' = S + S_encoder

    norm_type / norm_added_type:
        0=NONE, 1=LAYER_NORM, 2=LAYER_NORM_AFFINE, 3=RMS_NORM, 4=RMS_NORM_AFFINE
    rope_type: 0=NONE, 1=INTERLEAVE, 2=HALF
    concat_order: 0=BEFORE_ENCODER (self first), 1=AFTER_ENCODER (encoder first)
    """
    origin_dtype = query.dtype
    B, Sq, H, D = query.shape
    Sk = key.shape[1]
    Sv = value.shape[1]

    Seq = encoder_query.shape[1] if encoder_query is not None else 0
    Sek = encoder_key.shape[1] if encoder_key is not None else 0
    Sev = encoder_value.shape[1] if encoder_value is not None else 0

    # ---------- Norm on query ----------
    norm_q_fn = {
        0: lambda x, w, b: (x, None, None),  # NONE
        1: lambda x, w, b: torch_layer_norm(x, eps=eps),                   # LAYER_NORM
        2: lambda x, w, b: torch_layer_norm(x, w, b, eps=eps),             # LAYER_NORM_AFFINE
        3: lambda x, w, b: (torch_rms_norm(x, eps=eps), None, None),       # RMS_NORM
        4: lambda x, w, b: (torch_rms_norm(x, w, eps=eps), None, None),    # RMS_NORM_AFFINE
    }
    q_out, q_mean, q_rstd = norm_q_fn.get(norm_type, norm_q_fn[0])(query, norm_query_weight, norm_query_bias)

    # ---------- Norm on key ----------
    k_out, k_mean, k_rstd = norm_q_fn.get(norm_type, norm_q_fn[0])(key, norm_key_weight, norm_key_bias)

    # ---------- Norm on encoder_query ----------
    eq_out, eq_mean, eq_rstd = norm_q_fn.get(norm_added_type, norm_q_fn[0])(
        encoder_query, norm_added_query_weight, norm_added_query_bias)

    # ---------- Norm on encoder_key ----------
    ek_out, ek_mean, ek_rstd = norm_q_fn.get(norm_added_type, norm_q_fn[0])(
        encoder_key, norm_added_key_weight, norm_added_key_bias)

    # ---------- Concat (B, S, H, D) → (B, S', H, D) ----------
    def concat_tensors(a, b, order):
        if b is None:
            return a
        if order == 0:  # BEFORE_ENCODER: self first
            return torch.cat([a, b], dim=1)
        else:           # AFTER_ENCODER: encoder first
            return torch.cat([b, a], dim=1)

    q_cat = concat_tensors(q_out, eq_out, concat_order)
    k_cat = concat_tensors(k_out, ek_out, concat_order)
    v_cat = concat_tensors(value, encoder_value, concat_order)

    # ---------- Transpose to (B, H, S', D) ----------
    # Use .clone() to ensure independent tensors — avoids aliasing when
    # query/key/value share the same underlying storage (e.g. in tests).
    q_out_final = q_cat.permute(0, 2, 1, 3).contiguous().clone()
    k_out_final = k_cat.permute(0, 2, 1, 3).contiguous().clone()
    v_out_final = v_cat.permute(0, 2, 1, 3).contiguous().clone()

    # ---------- RoPE ----------
    if rope_type != 0 and rope_sin is not None and rope_cos is not None:
        rope_len = rope_sin.shape[0]
        q_out_final[:, :, :rope_len, :] = torch_rotary_emb(
            q_out_final[:, :, :rope_len, :], rope_sin, rope_cos, rope_type)
        k_out_final[:, :, :rope_len, :] = torch_rotary_emb(
            k_out_final[:, :, :rope_len, :], rope_sin, rope_cos, rope_type)

    # ---------- Prepare mean/rstd outputs ----------
    def make_mean_rstd(mean, rstd, B, S, H, is_layer_norm, is_training):
        if is_layer_norm and is_training and mean is not None:
            return mean.view(B, S, H, 1).float(), rstd.view(B, S, H, 1).float()
        else:
            dummy = torch.zeros(1, dtype=torch.float32, device=query.device)
            return dummy, dummy

    is_ln_q = norm_type in (1, 2)
    is_ln_k = norm_type in (1, 2)
    is_ln_eq = norm_added_type in (1, 2)
    is_ln_ek = norm_added_type in (1, 2)

    nq_mean, nq_rstd = make_mean_rstd(q_mean, q_rstd, B, Sq, H, is_ln_q, is_training)
    nk_mean, nk_rstd = make_mean_rstd(k_mean, k_rstd, B, Sk, H, is_ln_k, is_training)
    naq_mean, naq_rstd = make_mean_rstd(eq_mean, eq_rstd, B, Seq, H, is_ln_eq, is_training)
    nak_mean, nak_rstd = make_mean_rstd(ek_mean, ek_rstd, B, Sek, H, is_ln_ek, is_training)

    return (q_out_final.to(origin_dtype), k_out_final.to(origin_dtype), v_out_final.to(origin_dtype),
            nq_mean, nq_rstd, nk_mean, nk_rstd, naq_mean, naq_rstd, nak_mean, nak_rstd)


# ============================================================================
# Test Helpers
# ============================================================================


def _is_npu_available():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def _has_mindiesd_op():
    try:
        import mindiesd  # noqa: F401
        return hasattr(torch.ops, 'mindiesd') and hasattr(torch.ops.mindiesd, 'norm_rope_concat')
    except Exception:
        return False


NPU_AVAILABLE = _is_npu_available() and _has_mindiesd_op()


def _to_npu(*tensors):
    """Move tensors to NPU if available, otherwise keep on CPU."""
    if NPU_AVAILABLE:
        device = torch.device("npu:0")
        return tuple(t.to(device) for t in tensors)
    return tensors


# ============================================================================
# Test Cases
# ============================================================================


@unittest.skipUnless(NPU_AVAILABLE, "NPU or mindiesd operator not available")
class TestNormRopeConcatNPU(unittest.TestCase):
    """NPU tests: run operator on real hardware and compare against reference."""

    def _run_and_check(self, query, key, value,
                       encoder_query=None, encoder_key=None, encoder_value=None,
                       norm_query_weight=None, norm_query_bias=None,
                       norm_key_weight=None, norm_key_bias=None,
                       norm_added_query_weight=None, norm_added_query_bias=None,
                       norm_added_key_weight=None, norm_added_key_bias=None,
                       rope_sin=None, rope_cos=None,
                       norm_type=0, norm_added_type=0, rope_type=0,
                       concat_order=0, eps=1e-5, is_training=False,
                       atol=1e-2, rtol=1e-2):
        """Run operator on NPU and compare with reference."""
        # Reference on CPU
        def _to_cpu(t):
            return t.cpu() if t is not None else None
        ref_result = torch_norm_rope_concat_ref(
            _to_cpu(query), _to_cpu(key), _to_cpu(value),
            _to_cpu(encoder_query), _to_cpu(encoder_key), _to_cpu(encoder_value),
            _to_cpu(norm_query_weight), _to_cpu(norm_query_bias),
            _to_cpu(norm_key_weight), _to_cpu(norm_key_bias),
            _to_cpu(norm_added_query_weight), _to_cpu(norm_added_query_bias),
            _to_cpu(norm_added_key_weight), _to_cpu(norm_added_key_bias),
            _to_cpu(rope_sin), _to_cpu(rope_cos),
            norm_type, norm_added_type, rope_type,
            concat_order, eps, is_training,
        )

        # Operator on NPU
        npu_query, npu_key, npu_value = _to_npu(query, key, value)
        npu_eq = _to_npu(encoder_query)[0] if encoder_query is not None else None
        npu_ek = _to_npu(encoder_key)[0] if encoder_key is not None else None
        npu_ev = _to_npu(encoder_value)[0] if encoder_value is not None else None
        npu_nqw = _to_npu(norm_query_weight)[0] if norm_query_weight is not None else None
        npu_nqb = _to_npu(norm_query_bias)[0] if norm_query_bias is not None else None
        npu_nkw = _to_npu(norm_key_weight)[0] if norm_key_weight is not None else None
        npu_nkb = _to_npu(norm_key_bias)[0] if norm_key_bias is not None else None
        npu_naqw = _to_npu(norm_added_query_weight)[0] if norm_added_query_weight is not None else None
        npu_naqb = _to_npu(norm_added_query_bias)[0] if norm_added_query_bias is not None else None
        npu_nakw = _to_npu(norm_added_key_weight)[0] if norm_added_key_weight is not None else None
        npu_nakb = _to_npu(norm_added_key_bias)[0] if norm_added_key_bias is not None else None
        npu_sin = _to_npu(rope_sin)[0] if rope_sin is not None else None
        npu_cos = _to_npu(rope_cos)[0] if rope_cos is not None else None

        result = torch.ops.mindiesd.norm_rope_concat(
            npu_query, npu_key, npu_value,
            npu_eq, npu_ek, npu_ev,
            npu_nqw, npu_nqb, npu_nkw, npu_nkb,
            npu_naqw, npu_naqb, npu_nakw, npu_nakb,
            npu_sin, npu_cos,
            norm_type=norm_type, norm_added_type=norm_added_type,
            rope_type=rope_type, concat_order=concat_order,
            eps=eps, is_training=is_training,
        )

        q_out, k_out, v_out = result[0], result[1], result[2]
        nq_mean, nq_rstd = result[3], result[4]
        nk_mean, nk_rstd = result[5], result[6]
        naq_mean, naq_rstd = result[7], result[8]
        nak_mean, nak_rstd = result[9], result[10]

        # Check shapes
        ref_q, ref_k, ref_v = ref_result[0], ref_result[1], ref_result[2]
        self.assertEqual(q_out.shape, ref_q.shape, f"query output shape mismatch: {q_out.shape} vs {ref_q.shape}")
        self.assertEqual(k_out.shape, ref_k.shape, f"key output shape mismatch: {k_out.shape} vs {ref_k.shape}")
        self.assertEqual(v_out.shape, ref_v.shape, f"value output shape mismatch: {v_out.shape} vs {ref_v.shape}")

        # Check numerical (loose tolerance for fp16/mixed precision)
        self.assertTrue(torch.allclose(q_out.cpu(), ref_q, atol=atol, rtol=rtol),
                        f"query output mismatch (max diff: {(q_out.cpu() - ref_q).abs().max()})")
        self.assertTrue(torch.allclose(k_out.cpu(), ref_k, atol=atol, rtol=rtol),
                        f"key output mismatch (max diff: {(k_out.cpu() - ref_k).abs().max()})")

    # ---------- Basic tests ----------

    def test_no_norm_no_rope_no_encoder(self):
        """norm_type=0, rope_type=0, no encoder — pure transpose only."""
        B, S, H, D = 1, 4, 2, 8
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        self._run_and_check(query, key, value)

    def test_layer_norm_affine_no_rope_no_encoder(self):
        """norm_type=2 (LAYER_NORM_AFFINE), rope_type=0."""
        B, S, H, D = 1, 5, 4, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)
        self._run_and_check(query, key, value,
                          norm_query_weight=nqw, norm_query_bias=nqb,
                          norm_key_weight=nkw, norm_key_bias=nkb,
                          norm_type=2, is_training=False)

    def test_layer_norm_affine_training(self):
        """norm_type=2, is_training=True — should produce mean/rstd."""
        B, S, H, D = 1, 3, 2, 8
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)
        self._run_and_check(query, key, value,
                          norm_query_weight=nqw, norm_query_bias=nqb,
                          norm_key_weight=nkw, norm_key_bias=nkb,
                          norm_type=2, is_training=True)

    def test_rms_norm_affine(self):
        """norm_type=4 (RMS_NORM_AFFINE)."""
        B, S, H, D = 1, 5, 4, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        self._run_and_check(query, key, value,
                          norm_query_weight=nqw, norm_key_weight=nkw,
                          norm_type=4, is_training=False)

    def test_rms_norm_no_affine(self):
        """norm_type=3 (RMS_NORM without affine)."""
        B, S, H, D = 1, 3, 4, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        self._run_and_check(query, key, value, norm_type=3, is_training=False)

    # ---------- RoPE tests ----------

    def test_rope_interleave(self):
        """rope_type=1 (INTERLEAVE), with norm_type=2."""
        B, S, H, D = 1, 4, 2, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)
        rope_sin = torch.randn(S, D, dtype=torch.float32)
        rope_cos = torch.randn(S, D, dtype=torch.float32)
        self._run_and_check(query, key, value,
                          norm_query_weight=nqw, norm_query_bias=nqb,
                          norm_key_weight=nkw, norm_key_bias=nkb,
                          rope_sin=rope_sin, rope_cos=rope_cos,
                          norm_type=2, rope_type=1, is_training=False)

    def test_rope_half(self):
        """rope_type=2 (HALF)."""
        B, S, H, D = 1, 4, 2, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)
        rope_sin = torch.randn(S, D, dtype=torch.float32)
        rope_cos = torch.randn(S, D, dtype=torch.float32)
        self._run_and_check(query, key, value,
                          norm_query_weight=nqw, norm_query_bias=nqb,
                          norm_key_weight=nkw, norm_key_bias=nkb,
                          rope_sin=rope_sin, rope_cos=rope_cos,
                          norm_type=2, rope_type=2, is_training=False)

    # ---------- Encoder (cross-attention) tests ----------

    def test_with_encoder_before(self):
        """concat_order=0 (BEFORE_ENCODER): self query before encoder query."""
        B, H, D = 1, 4, 16
        Sq, Sk, Sv = 4, 4, 4
        Seq, Sek, Sev = 2, 2, 2

        query = torch.randn(B, Sq, H, D, dtype=torch.float32)
        key = torch.randn(B, Sk, H, D, dtype=torch.float32)
        value = torch.randn(B, Sv, H, D, dtype=torch.float32)
        encoder_query = torch.randn(B, Seq, H, D, dtype=torch.float32)
        encoder_key = torch.randn(B, Sek, H, D, dtype=torch.float32)
        encoder_value = torch.randn(B, Sev, H, D, dtype=torch.float32)

        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)
        naqw = torch.ones(D, dtype=torch.float32)
        naqb = torch.zeros(D, dtype=torch.float32)
        nakw = torch.ones(D, dtype=torch.float32)
        nakb = torch.zeros(D, dtype=torch.float32)

        total_seq = Sq + Seq
        rope_sin = torch.randn(total_seq, D, dtype=torch.float32)
        rope_cos = torch.randn(total_seq, D, dtype=torch.float32)

        self._run_and_check(query, key, value,
                          encoder_query, encoder_key, encoder_value,
                          nqw, nqb, nkw, nkb, naqw, naqb, nakw, nakb,
                          rope_sin, rope_cos,
                          norm_type=2, norm_added_type=2, rope_type=1,
                          concat_order=0, is_training=True)

    def test_with_encoder_after(self):
        """concat_order=1 (AFTER_ENCODER): encoder query before self query."""
        B, H, D = 1, 4, 16
        Sq, Sk, Sv = 4, 4, 4
        Seq, Sek, Sev = 2, 2, 2

        query = torch.randn(B, Sq, H, D, dtype=torch.float32)
        key = torch.randn(B, Sk, H, D, dtype=torch.float32)
        value = torch.randn(B, Sv, H, D, dtype=torch.float32)
        encoder_query = torch.randn(B, Seq, H, D, dtype=torch.float32)
        encoder_key = torch.randn(B, Sek, H, D, dtype=torch.float32)
        encoder_value = torch.randn(B, Sev, H, D, dtype=torch.float32)

        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)
        naqw = torch.ones(D, dtype=torch.float32)
        naqb = torch.zeros(D, dtype=torch.float32)
        nakw = torch.ones(D, dtype=torch.float32)
        nakb = torch.zeros(D, dtype=torch.float32)

        total_seq = Sq + Seq
        rope_sin = torch.randn(total_seq, D, dtype=torch.float32)
        rope_cos = torch.randn(total_seq, D, dtype=torch.float32)

        self._run_and_check(query, key, value,
                          encoder_query, encoder_key, encoder_value,
                          nqw, nqb, nkw, nkb, naqw, naqb, nakw, nakb,
                          rope_sin, rope_cos,
                          norm_type=2, norm_added_type=2, rope_type=1,
                          concat_order=1, is_training=True)

    # ---------- Shape / output size tests ----------

    def test_output_shapes_no_encoder(self):
        """Verify output shapes without encoder."""
        B, S, H, D = 2, 6, 2, 32
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)

        npu_query, npu_key, npu_value = _to_npu(query, key, value)
        npu_nqw, npu_nqb = _to_npu(nqw, nqb)
        npu_nkw, npu_nkb = _to_npu(nkw, nkb)

        result = torch.ops.mindiesd.norm_rope_concat(
            npu_query, npu_key, npu_value,
            norm_query_weight=npu_nqw, norm_query_bias=npu_nqb,
            norm_key_weight=npu_nkw, norm_key_bias=npu_nkb,
            norm_type=2, is_training=True,
        )

        self.assertEqual(result[0].shape, (B, H, S, D))      # query_output
        self.assertEqual(result[1].shape, (B, H, S, D))      # key_output
        self.assertEqual(result[2].shape, (B, H, S, D))      # value_output
        self.assertEqual(result[3].shape, (B, S, H, 1))       # norm_query_mean
        self.assertEqual(result[4].shape, (B, S, H, 1))       # norm_query_rstd
        self.assertEqual(result[5].shape, (B, S, H, 1))       # norm_key_mean
        self.assertEqual(result[6].shape, (B, S, H, 1))       # norm_key_rstd

    def test_output_shapes_with_encoder(self):
        """Verify output shapes with encoder (cross-attention)."""
        B, H, D = 2, 3, 16
        Sq, Sk, Sv = 5, 5, 5
        Seq, Sek, Sev = 3, 3, 3

        query = torch.randn(B, Sq, H, D, dtype=torch.float32)
        key = torch.randn(B, Sk, H, D, dtype=torch.float32)
        value = torch.randn(B, Sv, H, D, dtype=torch.float32)
        eq_ = torch.randn(B, Seq, H, D, dtype=torch.float32)
        ek_ = torch.randn(B, Sek, H, D, dtype=torch.float32)
        ev_ = torch.randn(B, Sev, H, D, dtype=torch.float32)

        npu_q, npu_k, npu_v = _to_npu(query, key, value)
        npu_eq, npu_ek, npu_ev = _to_npu(eq_, ek_, ev_)

        result = torch.ops.mindiesd.norm_rope_concat(
            npu_q, npu_k, npu_v,
            encoder_query=npu_eq, encoder_key=npu_ek, encoder_value=npu_ev,
        )

        self.assertEqual(result[0].shape, (B, H, Sq + Seq, D))
        self.assertEqual(result[1].shape, (B, H, Sk + Sek, D))
        self.assertEqual(result[2].shape, (B, H, Sv + Sev, D))

    # ---------- Mixed norm types ----------

    def test_query_layernorm_encoder_none(self):
        """norm_type=2 (LAYER_NORM_AFFINE), norm_added_type=0 (NONE)."""
        B, H, D = 1, 2, 8
        Sq, Sk, Sv = 4, 4, 4
        Seq = 2

        query = torch.randn(B, Sq, H, D, dtype=torch.float32)
        key = torch.randn(B, Sk, H, D, dtype=torch.float32)
        value = torch.randn(B, Sv, H, D, dtype=torch.float32)
        eq_ = torch.randn(B, Seq, H, D, dtype=torch.float32)
        ek_ = torch.randn(B, Seq, H, D, dtype=torch.float32)
        ev_ = torch.randn(B, Seq, H, D, dtype=torch.float32)

        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)
        nkw = torch.ones(D, dtype=torch.float32)
        nkb = torch.zeros(D, dtype=torch.float32)

        self._run_and_check(query, key, value,
                          eq_, ek_, ev_,
                          norm_query_weight=nqw, norm_query_bias=nqb,
                          norm_key_weight=nkw, norm_key_bias=nkb,
                          norm_type=2, norm_added_type=0,
                          is_training=False)

    def test_fp16_input(self):
        """Test with FP16 input data."""
        B, S, H, D = 1, 4, 2, 16
        query = torch.randn(B, S, H, D, dtype=torch.float16)
        key = torch.randn(B, S, H, D, dtype=torch.float16)
        value = torch.randn(B, S, H, D, dtype=torch.float16)
        nqw = torch.ones(D, dtype=torch.float16)
        nqb = torch.zeros(D, dtype=torch.float16)
        nkw = torch.ones(D, dtype=torch.float16)
        nkb = torch.zeros(D, dtype=torch.float16)
        rope_sin = torch.randn(S, D, dtype=torch.float16)
        rope_cos = torch.randn(S, D, dtype=torch.float16)
        self._run_and_check(query, key, value,
                          norm_query_weight=nqw, norm_query_bias=nqb,
                          norm_key_weight=nkw, norm_key_bias=nkb,
                          rope_sin=rope_sin, rope_cos=rope_cos,
                          norm_type=2, rope_type=1, is_training=False,
                          atol=5e-2, rtol=5e-2)


@unittest.skipUnless(NPU_AVAILABLE, "NPU or mindiesd operator not available")
class TestNormRopeConcatIntegration(unittest.TestCase):
    """Integration-style tests that exercise realistic model scenarios."""

    def test_cross_attention_scenario(self):
        """Simulate a cross-attention block: self Q, encoder K/V with norm+rope."""
        B, H, D = 1, 8, 64
        Sq, Skv = 10, 20

        query = torch.randn(B, Sq, H, D, dtype=torch.float32)
        encoder_key = torch.randn(B, Skv, H, D, dtype=torch.float32)
        encoder_value = torch.randn(B, Skv, H, D, dtype=torch.float32)

        # For cross-attention: key=encoder_key, value=encoder_value, no self key/value
        # Use placeholder key/value matching query shape
        key = torch.randn(B, Sq, H, D, dtype=torch.float32)
        value = torch.randn(B, Sq, H, D, dtype=torch.float32)

        nqw = torch.randn(D, dtype=torch.float32)
        nqb = torch.randn(D, dtype=torch.float32)
        nakw = torch.randn(D, dtype=torch.float32)
        nakb = torch.randn(D, dtype=torch.float32)
        rope_sin = torch.randn(Sq + Skv, D, dtype=torch.float32)
        rope_cos = torch.randn(Sq + Skv, D, dtype=torch.float32)

        npu_q, npu_k, npu_v = _to_npu(query, key, value)
        npu_ek, npu_ev = _to_npu(encoder_key, encoder_value)
        npu_nqw, npu_nqb = _to_npu(nqw, nqb)
        npu_nakw, npu_nakb = _to_npu(nakw, nakb)
        npu_sin, npu_cos = _to_npu(rope_sin, rope_cos)

        result = torch.ops.mindiesd.norm_rope_concat(
            npu_q, npu_k, npu_v,
            encoder_key=npu_ek, encoder_value=npu_ev,
            norm_query_weight=npu_nqw, norm_query_bias=npu_nqb,
            norm_added_key_weight=npu_nakw, norm_added_key_bias=npu_nakb,
            rope_sin=npu_sin, rope_cos=npu_cos,
            norm_type=2, norm_added_type=2, rope_type=1,
            is_training=False,
        )

        # Output shapes: query has self+encoder seq, key=encoder seq only (key=Sq=10 so 10+20=30)
        self.assertEqual(result[0].shape, (B, H, Sq + Skv, D))
        self.assertEqual(result[1].shape, (B, H, Sq + Skv, D))
        self.assertEqual(result[2].shape, (B, H, Sq + Skv, D))

    def test_different_seq_lengths(self):
        """Test with query, key, value having different sequence lengths."""
        B, H, D = 2, 4, 16
        Sq, Sk, Sv = 8, 12, 12

        query = torch.randn(B, Sq, H, D, dtype=torch.float32)
        key = torch.randn(B, Sk, H, D, dtype=torch.float32)
        value = torch.randn(B, Sv, H, D, dtype=torch.float32)

        npu_q, npu_k, npu_v = _to_npu(query, key, value)
        result = torch.ops.mindiesd.norm_rope_concat(npu_q, npu_k, npu_v)

        self.assertEqual(result[0].shape, (B, H, Sq, D))
        self.assertEqual(result[1].shape, (B, H, Sk, D))
        self.assertEqual(result[2].shape, (B, H, Sv, D))

    def test_bsh_format_consistency(self):
        """Validate that input (B,S,H,D) → output (B,H,S',D) transform is consistent."""
        B, S, H, D = 1, 6, 3, 32
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        key = torch.randn(B, S, H, D, dtype=torch.float32)
        value = torch.randn(B, S, H, D, dtype=torch.float32)

        npu_q, npu_k, npu_v = _to_npu(query, key, value)
        result = torch.ops.mindiesd.norm_rope_concat(npu_q, npu_k, npu_v)

        q_out, k_out, v_out = result[0].cpu(), result[1].cpu(), result[2].cpu()

        # Verify that data permute is correct: input (B,S,H,D), output (B,H,S,D)
        # Element at query[0, s, h, d] should go to q_out[0, h, s, d]
        for s in range(S):
            for h in range(H):
                for d in range(D):
                    self.assertAlmostEqual(
                        query[0, s, h, d].item(),
                        q_out[0, h, s, d].item(),
                        places=5,
                        msg=f"Mismatch at (0,{h},{s},{d})")


# ============================================================================
# Reference-Only Tests (always runnable, no NPU needed)
# ============================================================================


class TestNormRopeConcatReference(unittest.TestCase):
    """Tests for the PyTorch reference implementation — always runs."""

    def test_ref_no_norm_no_rope_no_encoder(self):
        """Reference: pure transpose only."""
        B, S, H, D = 1, 4, 2, 8
        query = torch.randn(B, S, H, D, dtype=torch.float32)

        q_out, k_out, v_out, *_ = torch_norm_rope_concat_ref(
            query, query, query)

        self.assertEqual(q_out.shape, (B, H, S, D))
        torch.testing.assert_close(q_out, query.permute(0, 2, 1, 3), atol=1e-5, rtol=0)

    def test_ref_layer_norm_identity_weight(self):
        """Reference: LayerNorm with weight=1, bias=0 should match standard LN."""
        B, S, H, D = 2, 3, 4, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)
        nqb = torch.zeros(D, dtype=torch.float32)

        q_out, k_out, v_out, q_mean, q_rstd, *_ = torch_norm_rope_concat_ref(
            query, query, query,
            norm_query_weight=nqw, norm_query_bias=nqb,
            norm_key_weight=nqw, norm_key_bias=nqb,
            norm_type=2, is_training=True,
        )

        # Compare with torch.nn.functional.layer_norm
        ln_ref = torch.nn.functional.layer_norm(
            query.float(), (D,), weight=nqw, bias=nqb, eps=1e-5)
        ln_ours = q_out.permute(0, 2, 1, 3)  # back to B,S,H,D
        torch.testing.assert_close(ln_ours.float(), ln_ref, atol=1e-5, rtol=1e-5)

    def test_ref_rms_norm_unit_weight(self):
        """Reference: RMSNorm with weight=1."""
        B, S, H, D = 1, 2, 4, 16
        query = torch.randn(B, S, H, D, dtype=torch.float32)
        nqw = torch.ones(D, dtype=torch.float32)

        q_out, *_ = torch_norm_rope_concat_ref(
            query, query, query,
            norm_query_weight=nqw, norm_key_weight=nqw,
            norm_type=4, is_training=False,
        )

        # RMS norm reference
        x = query.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
        expected = (x / rms).permute(0, 2, 1, 3)
        torch.testing.assert_close(q_out.float(), expected, atol=1e-5, rtol=1e-5)

    def test_ref_concat_order_before(self):
        """Reference: concat_order=0 puts self before encoder."""
        B, H, D = 1, 2, 4
        Sq, Seq = 3, 2
        q_self = torch.ones(B, Sq, H, D, dtype=torch.float32)
        q_enc = torch.ones(B, Seq, H, D, dtype=torch.float32) * 2

        q_out, *_ = torch_norm_rope_concat_ref(
            q_self, q_self, q_self,
            encoder_query=q_enc, encoder_key=q_enc, encoder_value=q_enc,
            concat_order=0)

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
            q_self, q_self, q_self,
            encoder_query=q_enc, encoder_key=q_enc, encoder_value=q_enc,
            concat_order=1)

        # encoder (2.0) before self (1.0)
        q_bhsd = q_out.permute(0, 2, 1, 3)
        self.assertTrue((q_bhsd[:, :Seq] == 2.0).all())
        self.assertTrue((q_bhsd[:, Seq:] == 1.0).all())

    def test_ref_rope_interleave(self):
        """Reference: RoPE INTERLEAVE — verify (x1, x2) → (-x2, x1) pattern."""
        B, H, D = 1, 1, 4
        query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)
        rope_sin = torch.ones(1, D)
        rope_cos = torch.ones(1, D)

        q_out, *_ = torch_norm_rope_concat_ref(
            query, query, query,
            rope_sin=rope_sin, rope_cos=rope_cos,
            rope_type=1)

        # With cos=1, sin=1: (x1,x2) → (x1*1 + (-x2)*1, x2*1 + x1*1) → (x1-x2, x2+x1)
        # (1,2)→(1-2, 2+1)=(-1,3); (3,4)→(3-4,4+3)=(-1,7)
        expected = torch.tensor([[[[-1.0, 3.0], [-1.0, 7.0]]]], dtype=torch.float32)
        # output is (B=1, H=1, S=1, D=4) = [[[[1,2,3,4]]]] after permute back,
        # RoPE applies on (B,H,S,D), so all S=1 dims are transformed
        q_bhsd = q_out
        # cos=1,sin=1: x1-cos + (-x2)*sin, x2*cos + x1*sin → x1-x2, x2+x1
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
            query, query, query,
            norm_query_weight=nqw, norm_query_bias=nqb,
            norm_key_weight=nqw, norm_key_bias=nqb,
            norm_type=2, is_training=True,
        )

        # result tuple: (q_out, k_out, v_out, nq_mean, nq_rstd, nk_mean, nk_rstd, ...)
        self.assertEqual(len(result), 11)
        # LayerNorm mean/rstd
        self.assertEqual(result[3].shape, (B, S, H, 1))   # norm_query_mean
        self.assertEqual(result[4].shape, (B, S, H, 1))   # norm_query_rstd

    def test_ref_all_norm_type_combinations(self):
        """Reference: run all norm_type values to ensure no crash."""
        B, S, H, D = 1, 3, 2, 8
        q = torch.randn(B, S, H, D, dtype=torch.float32)
        w = torch.ones(D, dtype=torch.float32)
        b = torch.zeros(D, dtype=torch.float32)

        for norm_type in [0, 1, 2, 3, 4]:
            result = torch_norm_rope_concat_ref(
                q, q, q,
                norm_query_weight=w, norm_query_bias=b,
                norm_key_weight=w, norm_key_bias=b,
                norm_type=norm_type, is_training=(norm_type in (1, 2)),
            )
            self.assertEqual(result[0].shape, (B, H, S, D), f"Failed for norm_type={norm_type}")
            self.assertEqual(result[1].shape, (B, H, S, D), f"Failed for norm_type={norm_type}")
            self.assertEqual(result[2].shape, (B, H, S, D), f"Failed for norm_type={norm_type}")


# ============================================================================
# Main
# ============================================================================


if __name__ == '__main__':
    unittest.main()

