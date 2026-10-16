#!/usr/bin/env python
# pylint: disable=duplicate-code
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import threading

import torch
import torch.nn.functional as F
import torch_npu

from .sparse_flash_attn_rf_v2 import (
    avgpool,
    do_tensor_rearrange_pooling,
    do_multi_span_tensor_rearrange_pooling,
    rearrange_with_remaining,
    get_blockwise_mask,
    get_multi_span_blockwise_mask,
    do_tensor_inv_rearrange,
)


_ROT_MATRIX_LOCK = threading.Lock()
_ROT_MATRIXS: dict = {}


def _hadamard_order(n: int) -> torch.Tensor:
    """Sylvester Hadamard matrix of order ``n`` (power of 2), unscaled."""
    if n == 1:
        return torch.ones(1, 1)
    h = _hadamard_order(n // 2)
    top = torch.cat([h, h], dim=1)
    bottom = torch.cat([h, -h], dim=1)
    return torch.cat([top, bottom], dim=0)


def _get_rot_matrices(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
) -> tuple:
    """Hadamard rotation matrices for the FP8 path, cached per (device, dtype, head_dim).

    Same construction as vllm-omni kv_quant_npu (QuaRotMode.HADAMARD): a scaled
    Sylvester Hadamard matrix sliced to ``head_dim``. Q and K share one rotation.
    """
    key = (str(device), dtype, head_dim)
    with _ROT_MATRIX_LOCK:
        rot = _ROT_MATRIXS.get(key)
        if rot is None:
            order = 1 << (head_dim - 1).bit_length()
            rot = _hadamard_order(order) / math.sqrt(order)
            rot = rot[:head_dim, :head_dim].to(device=device, dtype=dtype).contiguous()
            _ROT_MATRIXS[key] = rot
        return rot, rot


@torch.no_grad()
def _perblock_quant(input_tensor, block_size=64, dst_type=torch.int8, smooth=False):
    """Quantize Q/K per block along the sequence dimension (1D per-block scales).

    Input must be BNSD [B, N, S, D]. Padding is applied to meet block
    alignment and sliced off afterwards.

    Returns:
        int8 [B, N, S, D] quantized tensor and fp32 scales [B, N, ceil(S / block_size)].
    """
    assert len(input_tensor.shape) == 4, (
        f"eagle qbsa per-block quant only supports 4D qkv, got {len(input_tensor.shape)} dims."
    )
    b, n, s, d = input_tensor.shape

    if smooth:
        input_tensor = input_tensor - input_tensor.mean(dim=2, keepdim=True)

    if s % block_size != 0:
        padding_length = (block_size - (s % block_size)) % block_size
        input_tensor = F.pad(input_tensor, (0, 0, 0, padding_length))

    input_tensor = input_tensor.reshape(b, n, math.ceil(s / block_size), -1)
    input_quant, input_scale = torch_npu.npu_dynamic_quant(input_tensor, dst_type=dst_type)

    input_quant = input_quant.reshape(b, n, -1, d)[:, :, :s, :]
    return input_quant.contiguous(), input_scale.contiguous()


@torch.no_grad()
def _eagle_qbsa_quant_qkv(q, k, v, block_size_q=64, layout="BSND"):
    """EagleQBSA quantization: Q/K per-block INT8, V per-channel FP8.

    q/k/v are BF16 [B, S, N, D] (BSND) or [B, N, S, D] (BNSD).
    Returns BNSD int8 Q/K, BNSD FP8 V, and fp32 scales.
    """
    if layout == "BSND":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    q_q, q_scales = _perblock_quant(q, block_size=block_size_q, dst_type=torch.int8)
    k_q, k_scales = _perblock_quant(k, block_size=block_size_q, dst_type=torch.int8)
    v_q, v_scales = torch_npu.npu_dynamic_quant(v.transpose(-1, -2), dst_type=torch_npu.float8_e4m3fn)
    v_q = v_q.transpose(-1, -2).contiguous()

    return q_q, k_q, v_q, q_scales, k_scales, v_scales


def _fp8_quant_qkv(q, k, v, q_rot, k_rot, block_size_q=128, block_size_kv=256, layout="BSND"):
    """Rotate Q/K then block-quantize Q/K/V to FP8.

    Mirrors FP8RotateQuantFA.forward: rotation on Q/K, block quant on all three.
    q_rot / k_rot should be generated once per attention instance and reused.

    Note: fa_block_quant_preprocess always outputs BNSD regardless of input layout.
    The returned tensors are in BNSD layout.
    """
    from ..quant.block_quant import fa_block_quant_preprocess

    # Rotation on Q and K (value is NOT rotated)
    q = torch.matmul(q, q_rot)
    k = torch.matmul(k, k_rot)

    q_fp8, q_scale = fa_block_quant_preprocess(
        q, block_size=block_size_q, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    k_fp8, k_scale = fa_block_quant_preprocess(
        k, block_size=block_size_kv, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    v_fp8, v_scale = fa_block_quant_preprocess(
        v, block_size=block_size_kv, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )

    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


@torch.no_grad()
def _mxfp4_quant_qkv(q, k, v, q_rot, k_rot, scale_alg=None, dst_type_max=0.0, layout="BSND"):
    """Rotate Q/K then MXFP4-quantize Q/K/V (E2M1 data + E8M0 scales).

    Mirrors MXFP4QuantFA._forward_mxfp4: Q/K are quantized rowwise along the
    head dim (per-32-element groups), V columnwise along the sequence dim. The
    Hadamard rotation of Q/K is retained from the FP8 path — with a 4-bit
    mantissa budget, outliers dominate the 32-element group scale, so rotation
    must break them up before quantization (V is NOT rotated).

    S is padded to a 64 base so the V-scale 32-row blocks pair up for the V3
    kernel layout. The caller must pass the ORIGINAL sequence lengths to the
    kernel (so the padded tail is skipped) and crop the output afterwards.

    Returns BNSD packed FP4 tensors (UINT8 storage, last dim D/2) and E8M0
    scales in the V3 layout: q/k scale [B, N, S, D/64, 2],
    v scale [B, N, ceil(S/64), D, 2].
    """
    from ...quantization.layer import _dynamic_mx_quant, _reshape_mxfp4_v_scale_for_fa

    # Rotation on Q and K (value is NOT rotated)
    q = torch.matmul(q, q_rot)
    k = torch.matmul(k, k_rot)

    if layout == "BSND":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    # Pad S to a 64 base: the V scale reshape pairs two consecutive 32-row blocks.
    pad_len = (64 - q.shape[2] % 64) % 64
    if pad_len:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))

    quant_kwargs = {}
    if scale_alg is not None:
        quant_kwargs["scale_alg"] = scale_alg
    if dst_type_max and dst_type_max > 0:
        quant_kwargs["dst_type_max"] = float(dst_type_max)

    q_fp4, q_scale = _dynamic_mx_quant(q, dst_type=torch_npu.float4_e2m1fn_x2, axis=-1, **quant_kwargs)
    k_fp4, k_scale = _dynamic_mx_quant(k, dst_type=torch_npu.float4_e2m1fn_x2, axis=-1, **quant_kwargs)
    v_fp4, v_scale = _dynamic_mx_quant(v, dst_type=torch_npu.float4_e2m1fn_x2, axis=2, **quant_kwargs)

    # q/k scale [B,N,S,D/32] -> [B,N,S,D/64,2]; v scale handled by the shared helper.
    # Newer torch_npu already returns the byte-grouped 5D scale for float4_e2m1fn_x2,
    # so guard the reshape for idempotence (same as _reshape_mxfp4_v_scale_for_fa).
    if q_scale.dim() == 4:
        q_scale = q_scale.reshape(*q_scale.shape[:-1], q_scale.shape[-1] // 2, 2).contiguous()
    if k_scale.dim() == 4:
        k_scale = k_scale.reshape(*k_scale.shape[:-1], k_scale.shape[-1] // 2, 2).contiguous()
    v_scale = _reshape_mxfp4_v_scale_for_fa(v_scale, "BNSD")
    return q_fp4, k_fp4, v_fp4, q_scale, k_scale, v_scale


def _bsa_inv_rearrange(out, tq, hq, wq, input_layout="BSND"):
    """Inverse of do_tensor_rearrange_pooling (text_len=0).

    Supports BSND [B, S, N, D] and BNSD [B, N, S, D] without extra transposes.
    Aligned path (hq%8==0 and wq%8==0): un-block-rearrange all tq frames.
    Remainder path: first frame is unchanged; remaining (tq-1) frames are un-rearranged.
    """
    bnsd = input_layout == "BNSD"
    b = out.shape[0]
    n = out.shape[1] if bnsd else out.shape[2]
    d = out.shape[3]
    hn, wn = hq // 8, wq // 8

    if hq % 8 == 0 and wq % 8 == 0:
        # aligned: (f hn wn hb wb) -> (f hn hb wn wb)
        if bnsd:
            out = (
                out.reshape(b, n, tq, hn, wn, 8, 8, d)
                .permute(0, 1, 2, 3, 5, 4, 6, 7)
                .contiguous()
                .reshape(b, n, tq * hq * wq, d)
            )
        else:
            out = (
                out.reshape(b, tq, hn, wn, 8, 8, n, d)
                .permute(0, 1, 2, 4, 3, 5, 6, 7)
                .contiguous()
                .reshape(b, tq * hq * wq, n, d)
            )
        return out

    # remainder path: split first frame (unchanged) from rest
    first_frame_len = hq * wq
    hq_block = (hq // 8) * 8
    wq_block = (wq // 8) * 8
    hq_rem = hq % 8
    wq_rem = wq % 8
    block_size = hn * wn * 64  # block-rearranged tokens/frame
    h_rem_size = hq_rem * wq  # h-remainder tokens/frame

    if bnsd:
        out_first = out[:, :, :first_frame_len, :]
        out_rest = out[:, :, first_frame_len:, :]

        out_rest = out_rest.reshape(b, n, tq - 1, hq * wq, d)
        t_block = out_rest[:, :, :, :block_size, :]
        t_h_r = out_rest[:, :, :, block_size : block_size + h_rem_size, :] if hq_rem > 0 else None
        t_w_r = out_rest[:, :, :, block_size + h_rem_size :, :] if wq_rem > 0 else None

        t_block = (
            t_block.reshape(b, n, tq - 1, hn, wn, 8, 8, d)
            .permute(0, 1, 2, 3, 5, 4, 6, 7)
            .contiguous()
            .reshape(b, n, tq - 1, hq_block, wq_block, d)
        )
        if wq_rem > 0:
            t_block = torch.cat([t_block, t_w_r.reshape(b, n, tq - 1, hq_block, wq_rem, d)], dim=4)
        if hq_rem > 0:
            t_block = torch.cat([t_block, t_h_r.reshape(b, n, tq - 1, hq_rem, wq, d)], dim=3)

        out_rest = t_block.reshape(b, n, (tq - 1) * hq * wq, d)
        return torch.cat([out_first, out_rest], dim=2)
    else:
        out_first = out[:, :first_frame_len, :, :]
        out_rest = out[:, first_frame_len:, :, :]

        out_rest = out_rest.reshape(b, tq - 1, hq * wq, n, d)
        t_block = out_rest[:, :, :block_size, :, :]
        t_h_r = out_rest[:, :, block_size : block_size + h_rem_size, :, :] if hq_rem > 0 else None
        t_w_r = out_rest[:, :, block_size + h_rem_size :, :, :] if wq_rem > 0 else None

        t_block = (
            t_block.reshape(b, tq - 1, hn, wn, 8, 8, n, d)
            .permute(0, 1, 2, 4, 3, 5, 6, 7)
            .contiguous()
            .reshape(b, tq - 1, hq_block, wq_block, n, d)
        )
        if wq_rem > 0:
            t_block = torch.cat([t_block, t_w_r.reshape(b, tq - 1, hq_block, wq_rem, n, d)], dim=3)
        if hq_rem > 0:
            t_block = torch.cat([t_block, t_h_r.reshape(b, tq - 1, hq_rem, wq, n, d)], dim=2)

        out_rest = t_block.reshape(b, (tq - 1) * hq * wq, n, d)
        return torch.cat([out_first, out_rest], dim=1)


def do_tensor_rearrange_only(q, k, v, txt_len, latent_shape_q, latent_shape_k, input_layout):
    """Spatial rearrange only (no avgpool), used when mask is cached."""
    tensor = torch.cat((q, k, v), dim=0)
    if txt_len != 0:
        if input_layout == "BSND":
            tensor_t = tensor[:, :txt_len, :, :]
            tensor_i = tensor[:, txt_len:, :, :]
        else:  # BNSD
            tensor_t = tensor[:, :, :txt_len, :]
            tensor_i = tensor[:, :, txt_len:, :]
        tensor_i = rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
        if input_layout == "BSND":
            tensor = torch.cat((tensor_i, tensor_t), dim=1)
        else:
            tensor = torch.cat((tensor_i, tensor_t), dim=2)
    else:
        tensor = rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout)
    q_, k_, v_ = torch.chunk(tensor, 3, dim=0)
    return q_, k_, v_


def _adapt_mask_for_block_sizes(mask, block_size_q, block_size_kv, pool_size):
    """Adapt block_sparse_mask from pool_size granularity to target block sizes.

    Used for cached masks that were generated at a uniform pool_size.  Merges
    adjacent blocks via any-merge so that mask dimensions match the target.
    block_size_q / block_size_kv must be multiples of pool_size.
    """
    if block_size_q == pool_size and block_size_kv == pool_size:
        return mask

    b, n, qb, kb = mask.shape

    if block_size_q != pool_size:
        ratio = block_size_q // pool_size
        if block_size_q % pool_size != 0:
            raise ValueError(f"block_size_q ({block_size_q}) must be a multiple of pool_size ({pool_size})")
        qb_padded = (qb + ratio - 1) // ratio * ratio
        if qb_padded != qb:
            pad = mask.new_zeros(b, n, qb_padded - qb, kb, dtype=mask.dtype)
            mask = torch.cat([mask, pad], dim=2)
        mask = mask.reshape(b, n, qb_padded // ratio, ratio, mask.shape[3])
        mask = mask.any(dim=3).to(mask.dtype)

    if block_size_kv != pool_size:
        ratio = block_size_kv // pool_size
        if block_size_kv % pool_size != 0:
            raise ValueError(f"block_size_kv ({block_size_kv}) must be a multiple of pool_size ({pool_size})")
        kb = mask.shape[3]
        kb_padded = (kb + ratio - 1) // ratio * ratio
        if kb_padded != kb:
            pad = mask.new_zeros(b, n, mask.shape[2], kb_padded - kb, dtype=mask.dtype)
            mask = torch.cat([mask, pad], dim=3)
        mask = mask.reshape(b, n, mask.shape[2], kb_padded // ratio, ratio)
        mask = mask.any(dim=4).to(mask.dtype)

    return mask


def _generate_mask_direct(
    q_pool,
    k_pool,
    txt_len,
    sparsity,
    scale,
    block_size_q,
    block_size_kv,
    latent_shape_q,
    input_layout,
    protect_first_frame=True,
):
    """Generate mask directly at block_size_q × block_size_kv granularity.

    Unlike ``get_blockwise_mask`` which assumes a uniform pool_size for both Q
    and KV, this function works with separately-pooled Q and K tensors to
    produce a rectangular mask [B, N, q_blocks, kv_blocks] at the exact target
    block sizes — no post-hoc merging needed.
    """
    if input_layout == "BSND":
        scores = torch.einsum("blnd,bsnd->bnls", q_pool, k_pool) * scale
    else:
        scores = torch.einsum("bnld,bnsd->bnls", q_pool, k_pool) * scale

    probs = torch.nn.functional.softmax(scores, dim=-1)

    cols = probs.shape[-1]
    keep_len = math.ceil(cols * (1 - sparsity))
    topk_values, _ = torch.topk(probs, k=keep_len, dim=-1)
    thresholds = topk_values[..., -1:]
    mask = probs >= thresholds

    tq, hq, wq = latent_shape_q
    first_frame_len = hq * wq

    # Text & first-frame protection: separate block counts for Q and KV.
    text_block_num_q = (txt_len + block_size_q - 1) // block_size_q
    text_block_num_kv = (txt_len + block_size_kv - 1) // block_size_kv

    if text_block_num_q > 0:
        mask[:, :, -text_block_num_q:, :] = True
        mask[:, :, :, -text_block_num_kv:] = True

    if protect_first_frame:
        firstframe_block_num_q = (first_frame_len + block_size_q - 1) // block_size_q
        firstframe_block_num_kv = (first_frame_len + block_size_kv - 1) // block_size_kv
        if firstframe_block_num_q > 0:
            mask[:, :, :firstframe_block_num_q, :] = True
            mask[:, :, :, :firstframe_block_num_kv] = True

    return mask.to(torch.int8)


def rain_fusion_attention_v3(
    query,
    key,
    value,
    block_sparse_mask,
    scale=None,
    head_num=None,
    num_key_value_heads=None,
    input_layout="BNSD",
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    block_size_q=128,
    block_size_kv=None,
    inner_precise=4,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    quant_mode=-1,
    dst_type_max=0.0,
    q_dtype=None,
    k_dtype=None,
    v_dtype=None,
    q_scale_dtype=None,
    k_scale_dtype=None,
    v_scale_dtype=None,
):
    """Sparse attention forward using aclnnBlockSparseAttentionV3/V2.

    Supported precision paths:
      - BF16/FP16: pass dequant scales as None (default).
      - FP8: pass pre-quantized FP8 QKV (must be BNSD) with FLOAT32 dequant scales.
      - MXFP4: pass packed FP4 QKV (UINT8 storage, must be BNSD) with E8M0
        dequant scales, quant_mode=2 (OCP) or 3 (CX), the CANN dtype codes
        (q_dtype etc. = torch_npu.float4_e2m1fn_x2, *_scale_dtype =
        torch_npu.float8_e8m0fnu) and optional dst_type_max for CX.

    Args:
        query / key / value: BNSD [B,N,S,D] or BSND [B,S,N,D].
                             BF16 when scales=None, quantized when scales provided.
                             Quantized tensors must already be in BNSD layout (caller handles conversion).
        block_sparse_mask:   int8 [B, N, q_blocks, kv_blocks]
        scale:               attention scale, default head_dim ** -0.5
        head_num:            number of query heads
        num_key_value_heads: number of KV heads (GQA), default equals head_num
        input_layout:        'BNSD' or 'BSND' — only affects BF16 tensors;
                             quantized tensors (with scales) must be BNSD
        actual_seq_lengths:  per-batch query sequence lengths
        actual_seq_lengths_kv: per-batch KV sequence lengths
        block_size_q:        block size for Q dimension (blockShapeX), default 128
        block_size_kv:       block size for KV dimension (blockShapeY). BF16: defaults
                             to block_size_q. FP8/MXFP4: defaults to 256 (FP8 requires
                             a 256 multiple, MXFP4 a 64 multiple, per CANN constraint).
        inner_precise:       precision mode; 950 chip requires 4
        q/k/v_dequant_scale: dequant scales — FLOAT32 for FP8, E8M0 (UINT8
                             storage + dtype code) for MXFP4; BNSD layout
        quant_mode:          V3 quantization mode: -1 auto (default), 0 none,
                             1 FP8, 2 MXFP4 OCP, 3 MXFP4 CX
        dst_type_max:        MXFP4 CX quantization range; 0.0 means dtype max
        q_dtype/k_dtype/v_dtype: CANN dtype code override for packed MXFP4 tensors
        q/k/v_scale_dtype:   CANN dtype code override for MXFP4 E8M0 scales

    Returns:
        out (Tensor): same layout and dtype as input
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if num_key_value_heads is None:
        num_key_value_heads = head_num

    fp8_mode = q_dequant_scale is not None

    # FP8: blockShapeY must be a multiple of 256; MXFP4: 64 (CANN tiling constraint).
    # BF16: blockShapeY equals block_size_q (no extra constraint).
    if block_size_kv is None:
        block_size_kv = 256 if fp8_mode else block_size_q

    # For BF16 path: convert BSND→BNSD if needed.
    # For quantized (FP8/MXFP4) path: tensors are already BNSD (produced by the caller).
    permuted = False
    if not fp8_mode and input_layout == "BSND":
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()
        permuted = True

    layout = "BNSD"

    kwargs = dict(
        query=query,
        key=key,
        value=value,
        block_sparse_mask=block_sparse_mask,
        block_shape=[block_size_q, block_size_kv],
        q_input_layout=layout,
        kv_input_layout=layout,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale,
        inner_precise=inner_precise,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        softmax_lse_flag=0,
    )
    if fp8_mode:
        kwargs.update(
            q_dequant_scale=q_dequant_scale,
            k_dequant_scale=k_dequant_scale,
            v_dequant_scale=v_dequant_scale,
        )
    if quant_mode != -1:
        kwargs["quant_mode"] = quant_mode
    if dst_type_max != 0.0:
        kwargs["dst_type_max"] = dst_type_max
    for name, dtype in (
        ("q_dtype", q_dtype),
        ("k_dtype", k_dtype),
        ("v_dtype", v_dtype),
        ("q_scale_dtype", q_scale_dtype),
        ("k_scale_dtype", k_scale_dtype),
        ("v_scale_dtype", v_scale_dtype),
    ):
        if dtype is not None:
            kwargs[name] = dtype

    attention_out, _ = torch.ops.mindiesd.block_sparse_attention(**kwargs)

    if permuted:
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()

    return attention_out


def bsa_sparse_attention_v3(
    q,
    k,
    v,
    latent_shape_q,
    latent_shape_k=None,
    txt_len=0,
    sparsity=0.5,
    input_layout="BSND",
    head_num=None,
    num_key_value_heads=None,
    scale=None,
    inner_precise=4,
    cached_mask=None,
    protect_first_frame=True,
    q_rot=None,
    k_rot=None,
    block_size=128,
    block_size_kv=None,
    video_spans=None,
    precision="bf16",
    mxfp4_dst_type_max=0.0,
    mxfp4_scale_alg=None,
):
    """End-to-end rf_v3 sparse attention: rearrange -> mask -> [quant] -> BSA -> inv-rearrange.

    Kernel precision is selected by ``precision``:
      - ``'mix'`` → EagleQBSA path: Q/K per-block INT8, V per-channel
        FP8, dispatched to
        ``torch.ops.mindiesd.eagle_quant_block_sparse_attention``.
      - ``'fp8'`` → BSA FP8 path: Hadamard rotation of Q/K plus full FP8 block
        quantization of Q/K/V before the BSA kernel. Rotation matrices are
        generated internally and cached per (device, dtype, head_dim) unless
        q_rot/k_rot are provided by the caller.
      - ``'mxfp4'`` → BSA MXFP4 path (aclnnBlockSparseAttentionV3): Hadamard
        rotation of Q/K (retained from the FP8 path — outliers would otherwise
        dominate the 32-element group scales at 4-bit precision), then FP4
        E2M1 data + E8M0 scales: Q/K rowwise along the head dim, V columnwise
        along the sequence dim. mxfp4_dst_type_max > 0 selects the CX scaling
        strategy (quantMode=3, ceil truncation with a custom range in [6, 12]);
        otherwise OCP (quantMode=2, floor truncation).
      - ``'bf16'`` (default) → BF16 path (no quantization).

    Mask generation always operates on BF16 tensors (before quantization).

    Args:
        q / k / v:           BF16 tensors [B, S, N, D] (BSND) or [B, N, S, D] (BNSD)
        latent_shape_q:      (t, h, w) for query; t*h*w == S. Omit in multi-span mode.
        latent_shape_k:      (t, h, w) for key/value, default equals latent_shape_q
        txt_len:             text token length (currently only 0 is supported)
        sparsity:            sparsity ratio [0, 1); 0 means no sparsity
        input_layout:        'BSND' or 'BNSD'
        head_num:            number of query heads; inferred from q if None
        num_key_value_heads: number of KV heads (GQA), default equals head_num
        scale:               attention scale; default head_dim ** -0.5
        inner_precise:       precision mode; resolved by hardware constraint
        cached_mask:         cached int8 block_sparse_mask from a previous step
        protect_first_frame: protect first frame generation
        q_rot / k_rot:       rotation matrices for FP8/MXFP4 paths.
                              Generate once per attention instance (e.g. via QR on randn).
                              Used by the FP8 and MXFP4 quantization paths.
        block_size:          Block size for Q dimension: used for rearrangement pooling,
                             CANN operator blockShapeX, and FP8 Q quantization. Default 128.
        block_size_kv:       KV block size for CANN operator blockShapeY and FP8 KV quant.
                             FP8: must be a multiple of 256, defaults to 256.
                             MXFP4: must be a multiple of 64, defaults to 256.
                             BF16: defaults to block_size.
        video_spans:
                             Multi-video layout over an unpadded input sequence.
                             The spans use
                             ``{"start": int, "latent_shape": [T, H, W]}``.
                             The public ``sparse_attention`` API validates
                             its compatibility with legacy layout arguments.
                             The same layout reorders Q, K, and V.
        precision:           Execution precision for the sparse kernel:
                             'bf16' (default, no quantization), 'mix' (EagleQBSA),
                             'fp8' (BSA FP8 with Hadamard rotation), or 'mxfp4'
                             (BSA MXFP4 via aclnnBlockSparseAttentionV3).
        mxfp4_dst_type_max:  MXFP4 quantization range (dstTypeMax). 0 (default) selects
                             OCP scaling (quantMode=2); a positive value (typically in
                             [6, 12]) selects CX scaling (quantMode=3).
        mxfp4_scale_alg:     Optional scale_alg forwarded to npu_dynamic_mx_quant.

    Returns:
        out (Tensor):      BF16 attention output, same layout as input
        new_mask (Tensor): int8 block_sparse_mask for caching
    """
    multi_span = video_spans is not None
    if not multi_span and latent_shape_k is None:
        latent_shape_k = latent_shape_q
    if head_num is None:
        head_num = q.shape[1] if input_layout == "BNSD" else q.shape[2]
    if num_key_value_heads is None:
        num_key_value_heads = head_num
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5

    # Resolve the execution mode from ``precision``. The FP8 and MXFP4 paths
    # rotate Q/K with a Hadamard matrix before quantization. If the caller did
    # not provide rotation matrices, generate them once per (device, dtype,
    # head_dim) and reuse across steps.
    if precision not in ("mix", "fp8", "mxfp4", "bf16"):
        raise ValueError(f"precision must be one of 'bf16', 'fp8', 'mxfp4', 'mix'; got {precision!r}.")
    if precision in ("fp8", "mxfp4") and (q_rot is None or k_rot is None):
        q_rot, k_rot = _get_rot_matrices(q.device, q.dtype, q.shape[-1])

    # S dimension index: dim 2 for BNSD, dim 1 for BSND
    s_dim = 2 if input_layout == "BNSD" else 1

    fp8_mode = precision == "fp8"
    mxfp4_mode = precision == "mxfp4"

    # Resolve effective KV block size for the CANN operator.
    # FP8: blockShapeY must be a multiple of 256; MXFP4: a multiple of 64
    # (CANN tiling constraint; 256 also satisfies MXFP4 and is the default).
    # BF16: blockShapeY = block_size (no extra constraint; block_size_kv is ignored).
    if fp8_mode or mxfp4_mode:
        effective_block_size_kv = block_size_kv if block_size_kv is not None else 256
    else:
        effective_block_size_kv = block_size
    if multi_span and effective_block_size_kv != block_size:
        raise ValueError(
            "rf_v3 multi-video spans currently require identical Q and KV block sizes; "
            "the FP8 KV=256 path is not yet span-aware."
        )

    new_mask = None
    inverse = None
    if multi_span:
        q_, k_, v_, tensor_pool, inverse, dense_blocks, first_frame_blocks = (
            do_multi_span_tensor_rearrange_pooling(
                q, k, v, video_spans, block_size, input_layout
            )
        )
        if cached_mask is None:
            new_mask = get_multi_span_blockwise_mask(
                tensor_pool,
                sparsity,
                scale,
                dense_blocks,
                first_frame_blocks,
                input_layout,
                return_binary=True,
            )
    elif cached_mask is None:
        # --- Mask generation ---
        if effective_block_size_kv == block_size:
            # Same Q/KV granularity — rearrange + pool once, reuse tensor_pool.
            q_, k_, v_, tensor_pool = do_tensor_rearrange_pooling(
                q,
                k,
                v,
                text_len=txt_len,
                pool_size=block_size,
                latent_shape_q=latent_shape_q,
                latent_shape_k=latent_shape_k,
                input_layout=input_layout,
            )
            new_mask = get_blockwise_mask(
                tensor_pool,
                txt_len,
                sparsity,
                scale,
                block_size,
                latent_shape_q,
                latent_shape_k,
                input_layout,
                return_binary=True,
                protect_first_frame=protect_first_frame,
            )
        else:
            # Separate Q/KV granularity — rearrange only (no tensor_pool),
            # then pool Q and K separately at their respective block sizes.
            q_, k_, v_ = do_tensor_rearrange_only(
                q,
                k,
                v,
                txt_len=txt_len,
                latent_shape_q=latent_shape_q,
                latent_shape_k=latent_shape_k,
                input_layout=input_layout,
            )
            q_pool = avgpool(q_, pool_size=block_size, input_layout=input_layout)
            k_pool = avgpool(k_, pool_size=effective_block_size_kv, input_layout=input_layout)
            new_mask = _generate_mask_direct(
                q_pool,
                k_pool,
                txt_len,
                sparsity,
                scale,
                block_size,
                effective_block_size_kv,
                latent_shape_q,
                input_layout,
                protect_first_frame=protect_first_frame,
            )
    else:
        # rearrange only, reuse cached mask
        q_, k_, v_ = do_tensor_rearrange_only(
            q,
            k,
            v,
            txt_len=txt_len,
            latent_shape_q=latent_shape_q,
            latent_shape_k=latent_shape_k,
            input_layout=input_layout,
        )

    seqlen = q_.shape[s_dim]
    actual_seq_lens = [seqlen] * q_.shape[0]

    # Skip mask adaptation if cached mask already matches target block sizes.
    if cached_mask is not None:
        expected_q_blocks = math.ceil(seqlen / block_size)
        expected_kv_blocks = math.ceil(seqlen / effective_block_size_kv)
        if cached_mask.shape[2] == expected_q_blocks and cached_mask.shape[3] == expected_kv_blocks:
            new_mask = cached_mask
        else:
            new_mask = _adapt_mask_for_block_sizes(
                cached_mask, block_size, effective_block_size_kv, pool_size=block_size
            )

    if precision == "mix":
        # EagleQBSA: Q/K per-block INT8 + V per-channel FP8 (BNSD inside).
        q_q, k_q, v_q, q_scales, k_scales, v_scales = _eagle_qbsa_quant_qkv(
            q_, k_, v_, block_size_q=64, layout=input_layout
        )
        out, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
            query=q_q,
            key=k_q,
            value=v_q.view(torch.int8),
            block_sparse_mask=new_mask.view(torch.int8),
            block_shape=[block_size, block_size],
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=num_key_value_heads,
            scale_value=scale,
            inner_precise=inner_precise,
            softmax_lse_flag=0,
            actual_seq_lengths=actual_seq_lens,
            actual_seq_lengths_kv=actual_seq_lens,
            query_scale=q_scales,
            key_scale=k_scales,
            value_scale=v_scales,
            query_dtype=torch.int8,
            key_dtype=torch.int8,
            value_dtype=torch_npu.float8_e4m3fn,
            output_dtype=torch.bfloat16,
        )
        if input_layout == "BSND":
            out = out.permute(0, 2, 1, 3).contiguous()
    else:
        # FP8/MXFP4: rotate Q/K, quantize Q/K/V (output BNSD).
        q_scale = k_scale = v_scale = None
        quant_mode = -1
        dst_type_max = 0.0
        q_dtype = k_dtype = v_dtype = None
        q_scale_dtype = k_scale_dtype = v_scale_dtype = None
        if mxfp4_mode:
            # V3 quant mode: 2 = MXFP4 OCP (floor), 3 = MXFP4 CX (custom range, ceil).
            quant_mode = 2 if mxfp4_dst_type_max <= 0 else 3
            dst_type_max = float(mxfp4_dst_type_max) if mxfp4_dst_type_max > 0 else 0.0
            q_, k_, v_, q_scale, k_scale, v_scale = _mxfp4_quant_qkv(
                q_,
                k_,
                v_,
                q_rot,
                k_rot,
                scale_alg=mxfp4_scale_alg,
                dst_type_max=mxfp4_dst_type_max,
                layout=input_layout,
            )
            # Packed FP4 data / E8M0 scales are UINT8 storage: carry CANN dtype codes.
            q_dtype = k_dtype = v_dtype = torch_npu.float4_e2m1fn_x2
            q_scale_dtype = k_scale_dtype = v_scale_dtype = torch_npu.float8_e8m0fnu
        elif fp8_mode:
            q_, k_, v_, q_scale, k_scale, v_scale = _fp8_quant_qkv(
                q_,
                k_,
                v_,
                q_rot,
                k_rot,
                block_size_q=block_size,
                block_size_kv=effective_block_size_kv,
                layout=input_layout,
            )

        # BSA kernel (V3: BF16/FP8/MXFP4; V2: BF16/FP8)
        bsa_layout = "BNSD" if (fp8_mode or mxfp4_mode) else input_layout
        out = rain_fusion_attention_v3(
            q_,
            k_,
            v_,
            block_sparse_mask=new_mask,
            scale=scale,
            head_num=head_num,
            num_key_value_heads=num_key_value_heads,
            input_layout=bsa_layout,
            actual_seq_lengths=actual_seq_lens,
            actual_seq_lengths_kv=actual_seq_lens,
            block_size_q=block_size,
            block_size_kv=effective_block_size_kv,
            inner_precise=inner_precise,
            q_dequant_scale=q_scale,
            k_dequant_scale=k_scale,
            v_dequant_scale=v_scale,
            quant_mode=quant_mode,
            dst_type_max=dst_type_max,
            q_dtype=q_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            q_scale_dtype=q_scale_dtype,
            k_scale_dtype=k_scale_dtype,
            v_scale_dtype=v_scale_dtype,
        )

        # FP8/MXFP4 output is BNSD; convert back for inv-rearrange.
        if (fp8_mode or mxfp4_mode) and input_layout == "BSND":
            out = out.permute(0, 2, 1, 3).contiguous()

        # MXFP4 pads S to a 64 base before quantization (kernel skips the padded
        # tail via the original actual_seq_lengths); crop the output back.
        if mxfp4_mode and out.shape[s_dim] != seqlen:
            out = out.narrow(s_dim, 0, seqlen).contiguous()

    # inverse rearrange to restore (t, h, w) order
    if multi_span:
        out = out.index_select(1 if input_layout == "BSND" else 2, inverse)
    elif txt_len > 0:
        out = do_tensor_inv_rearrange(out, txt_len, latent_shape_q, latent_shape_k, input_layout)
    else:
        tq, hq, wq = latent_shape_q
        out = _bsa_inv_rearrange(out, tq, hq, wq, input_layout)
    return out, new_mask
