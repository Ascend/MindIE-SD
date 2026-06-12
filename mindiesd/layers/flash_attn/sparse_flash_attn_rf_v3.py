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
import torch
import torch_npu
from .sparse_flash_attn_rf_v2 import (
    avgpool,
    do_tensor_rearrange_pooling,
    rearrange_with_remaining,
    get_blockwise_mask,
    do_tensor_inv_rearrange,
)
from ..quant.block_quant import fa_block_quant_preprocess


def _fp8_quant_qkv(q, k, v, q_rot, k_rot, block_size_q=128, block_size_kv=256, layout="BSND"):
    """Rotate Q/K then block-quantize Q/K/V to FP8.

    Mirrors FP8RotateQuantFA.forward: rotation on Q/K, block quant on all three.
    q_rot / k_rot should be generated once per attention instance and reused.

    Note: fa_block_quant_preprocess always outputs BNSD regardless of input layout.
    The returned tensors are in BNSD layout.
    """
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
):
    """Sparse attention forward using aclnnBlockSparseAttentionV2.

    Supports both BF16 and FP8 paths via the V2 kernel:
      - BF16/FP16: pass dequant scales as None (default).
      - FP8: pass pre-quantized FP8 QKV (must be BNSD) with FLOAT32 dequant scales.

    Args:
        query / key / value: BNSD [B,N,S,D] or BSND [B,S,N,D].
                             BF16 when scales=None, FP8 when scales provided.
                             FP8 tensors must already be in BNSD layout (caller handles conversion).
        block_sparse_mask:   int8 [B, N, q_blocks, kv_blocks]
        scale:               attention scale, default head_dim ** -0.5
        head_num:            number of query heads
        num_key_value_heads: number of KV heads (GQA), default equals head_num
        input_layout:        'BNSD' or 'BSND' — only affects BF16 tensors;
                             FP8 tensors (with scales) must be BNSD
        actual_seq_lengths:  per-batch query sequence lengths
        actual_seq_lengths_kv: per-batch KV sequence lengths
        block_size_q:        block size for Q dimension (blockShapeX), default 128
        block_size_kv:       block size for KV dimension (blockShapeY). BF16: defaults
                             to block_size_q. FP8: must be a multiple of 256 (per CANN
                             constraint), defaults to 256.
        inner_precise:       precision mode; 950 chip requires 4
        q/k/v_dequant_scale: optional FLOAT32 dequant scales for FP8 path (BNSD layout)

    Returns:
        out (Tensor): same layout and dtype as input
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if num_key_value_heads is None:
        num_key_value_heads = head_num

    fp8_mode = q_dequant_scale is not None

    # FP8: blockShapeY must be a multiple of 256 (CANN tiling constraint).
    # BF16: blockShapeY equals block_size_q (no extra constraint).
    if block_size_kv is None:
        block_size_kv = 256 if fp8_mode else block_size_q

    # For BF16 path: convert BSND→BNSD if needed.
    # For FP8 path: tensors are already BNSD (produced by fa_block_quant_preprocess).
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
):
    """End-to-end rf_v3 sparse attention: rearrange -> mask -> [quant] -> BSA -> inv-rearrange.

    FP8 vs BF16 is controlled by the caller:
      - Pass q_rot/k_rot → FP8 path (rotation + block quantization before BSA kernel)
      - No q_rot/k_rot  → BF16 path (no quantization)

    Mask generation always operates on BF16 tensors (before quantization).

    Args:
        q / k / v:           BF16 tensors [B, S, N, D] (BSND) or [B, N, S, D] (BNSD)
        latent_shape_q:      (t, h, w) for query; t*h*w == S
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
        q_rot / k_rot:       rotation matrices for FP8 path.
                              Generate once per attention instance (e.g. via QR on randn).
                              When provided, enables FP8 quantization path.
        block_size:          Block size for Q dimension: used for rearrangement pooling,
                             CANN operator blockShapeX, and FP8 Q quantization. Default 128.
        block_size_kv:       KV block size for CANN operator blockShapeY and FP8 KV quant.
                             FP8: must be a multiple of 256, defaults to 256.
                             BF16: defaults to block_size.

    Returns:
        out (Tensor):      BF16 attention output, same layout as input
        new_mask (Tensor): int8 block_sparse_mask for caching
    """
    if latent_shape_k is None:
        latent_shape_k = latent_shape_q
    if head_num is None:
        head_num = q.shape[1] if input_layout == "BNSD" else q.shape[2]
    if num_key_value_heads is None:
        num_key_value_heads = head_num
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5

    tq, hq, wq = latent_shape_q
    # S dimension index: dim 2 for BNSD, dim 1 for BSND
    s_dim = 2 if input_layout == "BNSD" else 1

    fp8_mode = q_rot is not None and k_rot is not None

    # Resolve effective KV block size for the CANN operator.
    # FP8: blockShapeY must be a multiple of 256 (CANN tiling constraint).
    # BF16: blockShapeY = block_size (no extra constraint; block_size_kv is ignored).
    if fp8_mode:
        effective_block_size_kv = block_size_kv if block_size_kv is not None else 256
    else:
        effective_block_size_kv = block_size

    new_mask = None
    if cached_mask is None:
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

    # FP8: rotate Q/K, block-quantize Q/K/V (output BNSD).
    q_scale = k_scale = v_scale = None
    if fp8_mode:
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

    # BSA kernel (V2: BF16 + FP8)
    bsa_layout = "BNSD" if fp8_mode else input_layout
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
    )

    # FP8 output is BNSD; convert back for inv-rearrange.
    if fp8_mode and input_layout == "BSND":
        out = out.permute(0, 2, 1, 3).contiguous()

    # inverse rearrange to restore (t, h, w) order
    if txt_len > 0:
        out = do_tensor_inv_rearrange(out, txt_len, latent_shape_q, latent_shape_k, input_layout)
    else:
        out = _bsa_inv_rearrange(out, tq, hq, wq, input_layout)
    return out, new_mask
