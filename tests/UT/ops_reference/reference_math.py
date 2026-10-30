# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Pure PyTorch references shared by CPU unit tests and device operator tests."""

import torch


def torch_mul_add_ref(a, b, c):
    """PyTorch reference for mul_add: out = a + b * c."""
    return a.float() + b.float() * c.float()


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
    rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + eps)
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
    query,
    key,
    value,
    encoder_query=None,
    encoder_key=None,
    encoder_value=None,
    norm_query_weight=None,
    norm_query_bias=None,
    norm_key_weight=None,
    norm_key_bias=None,
    norm_added_query_weight=None,
    norm_added_query_bias=None,
    norm_added_key_weight=None,
    norm_added_key_bias=None,
    rope_sin=None,
    rope_cos=None,
    norm_type=0,
    norm_added_type=0,
    rope_type=0,
    concat_order=0,
    eps=1e-5,
    is_training=False,
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

    Seq = encoder_query.shape[1] if encoder_query is not None else 0
    Sek = encoder_key.shape[1] if encoder_key is not None else 0

    # ---------- Norm on query ----------
    norm_q_fn = {
        0: lambda x, w, b: (x, None, None),  # NONE
        1: lambda x, w, b: torch_layer_norm(x, eps=eps),  # LAYER_NORM
        2: lambda x, w, b: torch_layer_norm(x, w, b, eps=eps),  # LAYER_NORM_AFFINE
        3: lambda x, w, b: (torch_rms_norm(x, eps=eps), None, None),  # RMS_NORM
        4: lambda x, w, b: (torch_rms_norm(x, w, eps=eps), None, None),  # RMS_NORM_AFFINE
    }
    q_out, q_mean, q_rstd = norm_q_fn.get(norm_type, norm_q_fn[0])(query, norm_query_weight, norm_query_bias)

    # ---------- Norm on key ----------
    k_out, k_mean, k_rstd = norm_q_fn.get(norm_type, norm_q_fn[0])(key, norm_key_weight, norm_key_bias)

    # ---------- Norm on encoder_query ----------
    eq_out, eq_mean, eq_rstd = norm_q_fn.get(norm_added_type, norm_q_fn[0])(
        encoder_query, norm_added_query_weight, norm_added_query_bias
    )

    # ---------- Norm on encoder_key ----------
    ek_out, ek_mean, ek_rstd = norm_q_fn.get(norm_added_type, norm_q_fn[0])(
        encoder_key, norm_added_key_weight, norm_added_key_bias
    )

    # ---------- Concat (B, S, H, D) → (B, S', H, D) ----------
    def concat_tensors(a, b, order):
        if b is None:
            return a
        if order == 0:  # BEFORE_ENCODER: self first
            return torch.cat([a, b], dim=1)
        else:  # AFTER_ENCODER: encoder first
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
            q_out_final[:, :, :rope_len, :], rope_sin, rope_cos, rope_type
        )
        k_out_final[:, :, :rope_len, :] = torch_rotary_emb(
            k_out_final[:, :, :rope_len, :], rope_sin, rope_cos, rope_type
        )

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

    return (
        q_out_final.to(origin_dtype),
        k_out_final.to(origin_dtype),
        v_out_final.to(origin_dtype),
        nq_mean,
        nq_rstd,
        nk_mean,
        nk_rstd,
        naq_mean,
        naq_rstd,
        nak_mean,
        nak_rstd,
    )
