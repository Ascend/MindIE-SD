#!/usr/bin/env python
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

import torch
from .sparse_flash_attn_rf_v2 import (
    do_tensor_rearrange_pooling,
    rearrange_with_remaining,
    get_blockwise_mask,
    do_tensor_inv_rearrange,
)


def _bsa_inv_rearrange(out, tq, hq, wq, input_layout="BSND"):
    """Inverse of do_tensor_rearrange_pooling (text_len=0).

    Supports BSND [B, S, N, D] and BNSD [B, N, S, D] without extra transposes.
    Aligned path (hq%8==0 and wq%8==0): un-block-rearrange all tq frames.
    Remainder path: first frame is unchanged; remaining (tq-1) frames are un-rearranged.
    """
    bnsd = (input_layout == "BNSD")
    b = out.shape[0]
    n = out.shape[1] if bnsd else out.shape[2]
    d = out.shape[3]
    hn, wn = hq // 8, wq // 8

    if hq % 8 == 0 and wq % 8 == 0:
        # aligned: (f hn wn hb wb) -> (f hn hb wn wb)
        if bnsd:
            out = (out
                   .reshape(b, n, tq, hn, wn, 8, 8, d)
                   .permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                   .reshape(b, n, tq * hq * wq, d))
        else:
            out = (out
                   .reshape(b, tq, hn, wn, 8, 8, n, d)
                   .permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()
                   .reshape(b, tq * hq * wq, n, d))
        return out

    # remainder path: split first frame (unchanged) from rest
    first_frame_len = hq * wq
    hq_block = (hq // 8) * 8
    wq_block = (wq // 8) * 8
    hq_rem = hq % 8
    wq_rem = wq % 8
    block_size = hn * wn * 64   # block-rearranged tokens/frame
    h_rem_size = hq_rem * wq    # h-remainder tokens/frame

    if bnsd:
        out_first = out[:, :, :first_frame_len, :]
        out_rest = out[:, :, first_frame_len:, :]

        out_rest = out_rest.reshape(b, n, tq - 1, hq * wq, d)
        t_block = out_rest[:, :, :, :block_size, :]
        t_h_r = out_rest[:, :, :, block_size:block_size + h_rem_size, :] if hq_rem > 0 else None
        t_w_r = out_rest[:, :, :, block_size + h_rem_size:, :] if wq_rem > 0 else None

        t_block = (t_block
                   .reshape(b, n, tq - 1, hn, wn, 8, 8, d)
                   .permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                   .reshape(b, n, tq - 1, hq_block, wq_block, d))
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
        t_h_r = out_rest[:, :, block_size:block_size + h_rem_size, :, :] if hq_rem > 0 else None
        t_w_r = out_rest[:, :, block_size + h_rem_size:, :, :] if wq_rem > 0 else None

        t_block = (t_block
                   .reshape(b, tq - 1, hn, wn, 8, 8, n, d)
                   .permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()
                   .reshape(b, tq - 1, hq_block, wq_block, n, d))
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
    sparse_size=128,
    inner_precise=4,
):
    """Sparse attention forward using block_sparse_attention op.

    Args:
        query / key / value: BNSD [B,N,S,D] or BSND [B,S,N,D]
        block_sparse_mask:   int8 [B, N, q_blocks, kv_blocks]
        scale:               attention scale, default head_dim ** -0.5
        head_num:            number of query heads
        num_key_value_heads: number of KV heads (GQA), default equals head_num
        input_layout:        'BNSD' or 'BSND'
        actual_seq_lengths:  per-batch query sequence lengths
        actual_seq_lengths_kv: per-batch KV sequence lengths
        sparse_size:         block size, default 128 (blockShapeY must be multiple of 128)
        inner_precise:       precision mode; 950 chip requires 4

    Returns:
        out (Tensor): same layout and dtype as query
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if num_key_value_heads is None:
        num_key_value_heads = head_num

    # block_sparse_attention does not support BSND; convert to BNSD internally.
    permuted = False
    if input_layout == "BSND":
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()
        layout = "BNSD"
        permuted = True
    else:
        layout = input_layout

    kwargs = dict(
        query=query,
        key=key,
        value=value,
        block_sparse_mask=block_sparse_mask,
        block_shape=[sparse_size, sparse_size],
        q_input_layout=layout,
        kv_input_layout=layout,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale,
        inner_precise=inner_precise,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        softmax_lse_flag=0,
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
    pool_size=128,
    sparsity=0.5,
    input_layout="BSND",
    head_num=None,
    num_key_value_heads=None,
    scale=None,
    inner_precise=4,
    cached_mask=None,
    protect_first_frame=True,
):
    """End-to-end rf_v3 sparse attention: rearrange -> mask -> BSA -> inv-rearrange.

    Skips avgpool+mask generation when cached_mask is provided (~863µs saved per step).
    Operator handles unaligned S natively via actual_seq_lengths.

    Args:
        q / k / v:           BSND [B, S, N, D] or BNSD [B, N, S, D]
        latent_shape_q:      (t, h, w) for query; t*h*w == S
        latent_shape_k:      (t, h, w) for key/value, default equals latent_shape_q
        txt_len:             text token length (currently only 0 is supported)
        pool_size:           block size, must be a multiple of 128, default 128
        sparsity:            sparsity ratio [0, 1); 0 means no sparsity
        input_layout:        'BSND' or 'BNSD'; BNSD avoids extra transposes in the BSA call
        head_num:            number of query heads; inferred from q if None
        num_key_value_heads: number of KV heads (GQA), default equals head_num
        scale:               attention scale; default head_dim ** -0.5
        inner_precise:       precision mode; 950 chip requires 4
        cached_mask:         cached int8 block_sparse_mask from a previous step;
                             skips pool+mask generation when provided
        protect_first_frame: protect first frame generation

    Returns:
        out (Tensor):      same layout as input, token order matches input
        new_mask (Tensor): int8 block_sparse_mask, can be passed as cached_mask next step
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

    if cached_mask is None:
        # rearrange + avgpool -> generate new mask
        q_, k_, v_, tensor_pool = do_tensor_rearrange_pooling(
            q, k, v,
            text_len=txt_len,
            pool_size=pool_size,
            latent_shape_q=latent_shape_q,
            latent_shape_k=latent_shape_k,
            input_layout=input_layout,
        )
        new_mask = get_blockwise_mask(
            tensor_pool, txt_len, sparsity, scale, pool_size,
            latent_shape_q, latent_shape_k, input_layout,
            return_binary=True,
            protect_first_frame=protect_first_frame,
        )
    else:
        # rearrange only, reuse cached mask
        q_, k_, v_ = do_tensor_rearrange_only(
            q, k, v,
            txt_len=txt_len,
            latent_shape_q=latent_shape_q,
            latent_shape_k=latent_shape_k,
            input_layout=input_layout,
        )
        new_mask = cached_mask

    # block sparse attention
    actual_seq_lens = [q_.shape[s_dim]] * q_.shape[0]
    out = rain_fusion_attention_v3(
        q_, k_, v_,
        block_sparse_mask=new_mask,
        scale=scale,
        head_num=head_num,
        num_key_value_heads=num_key_value_heads,
        input_layout=input_layout,
        actual_seq_lengths=actual_seq_lens,
        actual_seq_lengths_kv=actual_seq_lens,
        sparse_size=pool_size,
        inner_precise=inner_precise,
    )

    # inverse rearrange to restore (t, h, w) order
    if txt_len > 0:
        out = do_tensor_inv_rearrange(out, txt_len, latent_shape_q, latent_shape_k, input_layout)
    else:
        out = _bsa_inv_rearrange(out, tq, hq, wq, input_layout)
    return out, new_mask
