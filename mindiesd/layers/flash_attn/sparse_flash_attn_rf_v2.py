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

import math
from collections.abc import Mapping, Sequence
import torch
from einops import rearrange
from .. import _custom_ops as ops
from ...utils.exception import ParametersInvalid
from ...utils.get_platform import is_a5_device


_A5_RF_V2_UNSUPPORTED_MSG = (
    "sparse_flash_attn_rf_v2 (rain_fusion_attention) is not supported on A5 devices. "
    "Please use the public API 'mindiesd.layers.flash_attn.sparse_attention' (which routes 'rf_v2' "
    "to 'rf_v3' automatically on A5), or call 'sparse_flash_attn_rf_v3.bsa_sparse_attention_v3' "
    "directly for the higher-performance equivalent."
)


def avgpool(input_tensor, pool_size=128, input_layout='BNSD'):  # BSND in,  BSND out
    if input_layout == "BSND":
        batch, seqlen, headnum, dim = input_tensor.shape

        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size

        if num_full_blocks > 0:
            full_blocks = input_tensor[:, : num_full_blocks * pool_size, :, :]
            full_blocks_reshaped = full_blocks.view(batch, num_full_blocks, pool_size, headnum, dim)
            full_pooled = full_blocks_reshaped.mean(dim=2)
        else:
            full_pooled = torch.empty(0, device=input_tensor.device)
        if tail_size > 0:
            tail_block = input_tensor[:, num_full_blocks * pool_size :, :, :]
            tail_reshaped = tail_block.view(batch, 1, tail_size, headnum, dim)
            tail_pooled = tail_reshaped.mean(dim=2)
        else:
            tail_pooled = torch.empty(0, device=input_tensor.device)

        if num_full_blocks > 0 and tail_size > 0:
            output_tensor = torch.cat([full_pooled, tail_pooled], dim=1)
        elif num_full_blocks > 0:
            output_tensor = full_pooled
        else:
            output_tensor = tail_pooled
    else:
        batch, headnum, seqlen, dim = input_tensor.shape
        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size
        if num_full_blocks > 0:
            full_blocks = input_tensor[:, :, : num_full_blocks * pool_size, :]
            full_blocks_reshaped = full_blocks.view(batch, headnum, num_full_blocks, pool_size, dim)
            full_pooled = full_blocks_reshaped.mean(dim=3)
        else:
            full_pooled = torch.empty(0, device=input_tensor.device)
        if tail_size > 0:
            tail_block = input_tensor[:, :, num_full_blocks * pool_size :, :]
            tail_reshaped = tail_block.view(batch, headnum, 1, tail_size, dim)
            tail_pooled = tail_reshaped.mean(dim=3)
        else:
            tail_pooled = torch.empty(0, device=input_tensor.device)

        if num_full_blocks > 0 and tail_size > 0:
            output_tensor = torch.cat([full_pooled, tail_pooled], dim=2)
        elif num_full_blocks > 0:
            output_tensor = full_pooled
        else:
            output_tensor = tail_pooled
    return output_tensor


def get_mask_index(mask):
    b, n, s, _ = mask.shape
    device = mask.device

    mask_reshaped = mask.reshape(-1, s, s)
    batch_size = mask_reshaped.shape[0]

    row_indices = torch.arange(s, device=device).expand(batch_size, s, -1)
    sorted_vals = torch.where(mask_reshaped, row_indices, 1e9).to(torch.float32)
    sorted_vals, _ = torch.sort(sorted_vals, dim=-1)
    valid_count = mask_reshaped.sum(dim=-1, keepdim=True)
    keep_mask = row_indices < valid_count
    result = torch.where(keep_mask, sorted_vals, -1)

    pos_matrix = result.reshape(b, n, s, s).to(torch.int64)
    return pos_matrix


def get_blockwise_mask(
    qkv_pool,
    txt_len,
    sparsity,
    scale,
    pool_size,
    latent_shape_q,
    latent_shape_k,
    input_layout,
    return_binary=False,
    protect_first_frame=True,
):
    tq, hq, wq = latent_shape_q
    first_frame_len = hq * wq

    query_pool, key_pool, value_pool = torch.chunk(qkv_pool, 3, dim=0)
    if input_layout == "BSND":
        attn_scores_head = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
    else:
        attn_scores_head = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale
    score_matrix = torch.nn.functional.softmax(attn_scores_head, dim=-1)

    cols = score_matrix.shape[-1]

    keep_len = math.ceil(cols * (1 - sparsity))
    topk_values, _ = torch.topk(score_matrix, k=keep_len, dim=-1)
    thresholds = topk_values[..., -1:]
    mask = score_matrix >= thresholds
    text_block_num = (txt_len + pool_size - 1) // pool_size

    if text_block_num > 0:
        mask[:, :, -text_block_num:, :] = True
        mask[:, :, :, -text_block_num:] = True

    if protect_first_frame:
        firstframe_block_num = (first_frame_len + pool_size - 1) // pool_size
        if firstframe_block_num > 0:
            mask[:, :, :firstframe_block_num, :] = True
            mask[:, :, :, :firstframe_block_num] = True

    if return_binary:
        return mask.to(torch.int8)

    select_idx = get_mask_index(mask)
    select_idx = select_idx[0].transpose(0, 1)
    select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
    return select_idx, select_num_idx


def rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout):
    '''
    b (f hn hb wn wb) n d -> b (f hn wn hb wb) n d
    or
    b n (f hn hb wn wb) d -> b n (f hn wn hb wb) d
    '''
    tq, hq, wq = latent_shape_q
    first_frame_len, frame_num = hq * wq, tq
    if input_layout == "BSND":
        b, s, n, d = tensor.shape

        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_h_r = None
            tensor_w_r = None
            tensor_first = tensor[:, :first_frame_len, :, :]
            tensor = tensor[:, first_frame_len:, :, :]
            tensor_hwt = rearrange(tensor, 'b (f h w) n d -> b f h w n d', f=frame_num - 1, h=hq, w=wq)
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=2)
                tensor_h_r = tensor_h_r.reshape(b, frame_num - 1, -1, n, d)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=3)
                tensor_w_r = tensor_w_r.reshape(b, frame_num - 1, -1, n, d)
            tensor_hwt = rearrange(
                tensor_hwt,
                'b f (hn hb) (wn wb) n d -> b f (hn wn hb wb) n d',
                f=frame_num - 1,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=2)
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=2)
            tensor_hwt = tensor_hwt.reshape(b, -1, n, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=1)
        else:
            tensor_hwt = rearrange(
                tensor,
                'b (f hn hb wn wb) n d -> b (f hn wn hb wb) n d',
                f=frame_num,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
    else:
        b, n, s, d = tensor.shape
        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_h_r = None
            tensor_w_r = None
            tensor_first = tensor[:, :, :first_frame_len, :]
            tensor = tensor[:, :, first_frame_len:, :]
            tensor_hwt = rearrange(tensor, 'b n (f h w) d -> b n f h w d', f=frame_num - 1, h=hq, w=wq)
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=3)
                tensor_h_r = tensor_h_r.reshape(b, n, frame_num - 1, -1, d)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=4)
                tensor_w_r = tensor_w_r.reshape(b, n, frame_num - 1, -1, d)
            tensor_hwt = rearrange(
                tensor_hwt,
                'b n f (hn hb) (wn wb) d -> b n f (hn wn hb wb) d',
                f=frame_num - 1,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=3)
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=3)
            tensor_hwt = tensor_hwt.reshape(b, n, -1, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=2)
        else:
            tensor_hwt = rearrange(
                tensor,
                'b n (f hn hb wn wb) d -> b n (f hn wn hb wb) d',
                f=frame_num,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
    return tensor_hwt


def inv_rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout):
    '''
    b (f hn wn hb wb) n d -> b (f hn hb wn wb) n d
    or
    b n (f hn wn hb wb) d -> b n (f hn hb wn wb) d
    '''
    tq, hq, wq = latent_shape_q
    first_frame_len, frame_num = hq * wq, tq
    r_h = hq % 8
    r_w = wq % 8
    h_main = hq - r_h
    w_main = wq - r_w

    if input_layout == "BSND":
        b, s, n, d = tensor.shape

        if (r_h != 0) or (r_w != 0):
            tensor_first = tensor[:, :first_frame_len, :, :]
            tensor = tensor[:, first_frame_len:, :, :]
            tensor = tensor.reshape(b, frame_num - 1, hq * wq, n, d)

            split_sizes = [h_main * w_main]
            if r_h != 0:
                split_sizes.append(r_h * wq)
            if r_w != 0:
                split_sizes.append(h_main * r_w)

            parts = torch.split(tensor, split_sizes, dim=2)
            tensor_hwt = parts[0]
            idx = 1
            if r_h != 0:
                tensor_h_r = parts[idx]
                idx += 1
            if r_w != 0:
                tensor_w_r = parts[idx]

            tensor_hwt = rearrange(
                tensor_hwt,
                'b f (hn wn hb wb) n d -> b f (hn hb) (wn wb) n d',
                f=frame_num - 1,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )

            if r_w != 0:
                tensor_w_r = tensor_w_r.reshape(b, frame_num - 1, h_main, r_w, n, d)
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=3)

            if r_h != 0:
                tensor_h_r = tensor_h_r.reshape(b, frame_num - 1, r_h, wq, n, d)
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=2)

            tensor_hwt = tensor_hwt.reshape(b, -1, n, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=1)
        else:
            tensor_hwt = rearrange(
                tensor,
                'b (f hn wn hb wb) n h -> b (f hn hb wn wb) n h',
                f=frame_num,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
    else:
        b, n, s, d = tensor.shape
        if (r_h != 0) or (r_w != 0):
            tensor_first = tensor[:, :, :first_frame_len, :]
            tensor = tensor[:, :, first_frame_len:, :]
            tensor = tensor.reshape(b, n, frame_num - 1, hq * wq, d)

            split_sizes = [h_main * w_main]
            if r_h != 0:
                split_sizes.append(r_h * wq)
            if r_w != 0:
                split_sizes.append(h_main * r_w)

            parts = torch.split(tensor, split_sizes, dim=3)
            tensor_hwt = parts[0]
            idx = 1
            if r_h != 0:
                tensor_h_r = parts[idx]
                idx += 1
            if r_w != 0:
                tensor_w_r = parts[idx]

            tensor_hwt = rearrange(
                tensor_hwt,
                'b n f (hn wn hb wb) d -> b n f (hn hb) (wn wb) d',
                f=frame_num - 1,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )

            if r_w != 0:
                tensor_w_r = tensor_w_r.reshape(b, n, frame_num - 1, h_main, r_w, d)
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=4)

            if r_h != 0:
                tensor_h_r = tensor_h_r.reshape(b, n, frame_num - 1, r_h, wq, d)
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=3)

            tensor_hwt = tensor_hwt.reshape(b, n, -1, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=2)
        else:
            tensor_hwt = rearrange(
                tensor,
                'b n (f hn wn hb wb) h -> b n (f hn hb wn wb) h',
                f=frame_num,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
    return tensor_hwt


def do_tensor_rearrange_pooling(query, key, value, text_len, pool_size, latent_shape_q, latent_shape_k, input_layout):
    '''
    张量的分块重排 + 池化操作
    '''
    tensor = torch.cat((query, key, value), dim=0)
    if text_len != 0:
        if input_layout == "BSND":
            tensor_t = tensor[:, :text_len, :, :]
            tensor_i = tensor[:, text_len:, :, :]
        else:
            tensor_t = tensor[:, :, :text_len, :]
            tensor_i = tensor[:, :, text_len:, :]
        tensor_i_2 = rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
        if input_layout == "BSND":
            tensor = torch.concat((tensor_i_2, tensor_t), dim=1)
        else:
            tensor = torch.concat((tensor_i_2, tensor_t), dim=2)
        tensor_pool = avgpool(tensor, pool_size, input_layout)
    else:
        tensor = rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout)
        tensor_pool = avgpool(tensor, pool_size, input_layout)
    query_, key_, value_ = torch.chunk(tensor, 3, dim=0)
    return query_, key_, value_, tensor_pool


def do_tensor_inv_rearrange(tensor, text_len, latent_shape_q, latent_shape_k, input_layout):
    if text_len != 0:
        if input_layout == "BSND":
            tensor_t = tensor[:, -text_len:, :, :]
            tensor_i = tensor[:, :-text_len, :, :]

            tensor_i = inv_rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
            tensor = torch.concat((tensor_t, tensor_i), dim=1)
        elif input_layout == "BNSD":
            tensor_t = tensor[:, :, -text_len:, :]
            tensor_i = tensor[:, :, :-text_len, :]
            tensor_i = inv_rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
            tensor = torch.concat((tensor_t, tensor_i), dim=2)
    else:
        tensor = inv_rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout)

    return tensor


def _sequence_length(tensor, input_layout):
    return tensor.shape[1] if input_layout == "BSND" else tensor.shape[2]


def _index_sequence(tensor, indices, input_layout):
    dim = 1 if input_layout == "BSND" else 2
    return tensor.index_select(dim, indices)


def _normalize_video_spans(video_spans, sequence_len):
    """Validate video slices against the unpadded Q/K/V sequence."""
    if not isinstance(video_spans, Sequence) or isinstance(video_spans, (str, bytes)):
        raise ParametersInvalid("video_spans must be a sequence of span descriptors.")
    spans = []
    previous_end = 0
    for index, span in enumerate(video_spans):
        if not isinstance(span, Mapping):
            raise ParametersInvalid(f"video_spans[{index}] must be a mapping.")
        try:
            start = int(span["start"])
            shape = tuple(int(dim) for dim in span["latent_shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ParametersInvalid(
                f"video_spans[{index}] requires integer start and latent_shape=[T,H,W]."
            ) from exc
        if len(shape) != 3 or any(dim <= 0 for dim in shape):
            raise ParametersInvalid(
                f"video_spans[{index}].latent_shape must contain three positive integers."
            )
        length = math.prod(shape)
        if start < previous_end or start + length > sequence_len:
            raise ParametersInvalid(
                f"video_spans[{index}] [{start}, {start + length}) overlaps another span "
                f"or exceeds the input sequence length={sequence_len}."
            )
        spans.append((start, shape, length))
        previous_end = start + length
    if not spans:
        raise ParametersInvalid("video_spans must contain at least one video span.")
    return spans


def _span_rearranged_indices(start, shape, input_layout, device):
    """Return this clip's rf_v2 spatial reordering as source row indices."""
    length = math.prod(shape)
    if input_layout == "BSND":
        indices = torch.arange(start, start + length, device=device).reshape(1, length, 1, 1)
    else:
        indices = torch.arange(start, start + length, device=device).reshape(1, 1, length, 1)
    return rearrange_with_remaining(indices, shape, shape, input_layout).reshape(-1).to(torch.long)


def _multi_span_permutation(spans, sequence_len, pool_size, input_layout, device):
    """Pack individually tiled clips into one block plan without crossing clips.

    Dense rows are used to finish a partial clip block before the next clip.
    Those mixed blocks are marked dense, which preserves full access for every
    text/image/audio token. If there are not enough dense rows, a fixed-size
    block kernel cannot represent the layout safely and the caller must fall
    back to dense attention.
    """
    all_indices = torch.arange(sequence_len, device=device, dtype=torch.long)
    is_video = torch.zeros(sequence_len, device=device, dtype=torch.bool)
    for start, _, length in spans:
        is_video[start : start + length] = True
    dense_indices = all_indices[~is_video]
    dense_cursor = 0
    position = 0
    parts = []
    dense_blocks: set[int] = set()
    first_frame_blocks: set[int] = set()

    for clip_index, (start, shape, length) in enumerate(spans):
        # Every clip begins on a fresh sparse block. The final clip may share
        # its tail block with dense context, but never with another clip.
        if position % pool_size:
            raise AssertionError("multi-span packing lost block alignment")
        clip_block = position // pool_size
        first_frame_blocks.update(
            range(clip_block, clip_block + math.ceil((shape[1] * shape[2]) / pool_size))
        )
        parts.append(_span_rearranged_indices(start, shape, input_layout, device))
        position += length
        if clip_index != len(spans) - 1 and position % pool_size:
            needed = pool_size - position % pool_size
            if dense_cursor + needed > dense_indices.numel():
                raise ParametersInvalid(
                    "rf_v2 multi-video spans need dense rows to isolate clip block boundaries; "
                    "fall back to dense attention for this layout."
                )
            dense_blocks.add(position // pool_size)
            parts.append(dense_indices[dense_cursor : dense_cursor + needed])
            dense_cursor += needed
            position += needed

    if dense_cursor < dense_indices.numel():
        dense_end = position + dense_indices.numel() - dense_cursor
        dense_blocks.update(range(position // pool_size, math.ceil(dense_end / pool_size)))
        parts.append(dense_indices[dense_cursor:])
        position = dense_end
    if position != sequence_len:
        raise AssertionError("multi-span permutation does not cover the input sequence")
    permutation = torch.cat(parts)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(sequence_len, device=device, dtype=torch.long)
    return permutation, inverse, dense_blocks, first_frame_blocks


def get_multi_span_blockwise_mask(
    qkv_pool, sparsity, scale, dense_blocks, first_frame_blocks, input_layout, return_binary=False
):
    """Global multi-video ranking with dense context and first frame per clip retained."""
    query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)
    if input_layout == "BSND":
        scores = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
    else:
        scores = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale
    score_matrix = torch.nn.functional.softmax(scores, dim=-1)
    keep_len = math.ceil(score_matrix.shape[-1] * (1 - sparsity))
    if keep_len < 1:
        keep_len = 1
    threshold = torch.topk(score_matrix, k=keep_len, dim=-1).values[..., -1:]
    mask = score_matrix >= threshold
    retained = sorted(dense_blocks | first_frame_blocks)
    if retained:
        retained_idx = torch.tensor(retained, device=mask.device, dtype=torch.long)
        mask[:, :, retained_idx, :] = True
        mask[:, :, :, retained_idx] = True
    if return_binary:
        return mask.to(torch.int8)
    select_idx = get_mask_index(mask)[0].transpose(0, 1)
    select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
    return select_idx, select_num_idx


def do_multi_span_tensor_rearrange_pooling(query, key, value, video_spans, pool_size, input_layout):
    """Prepare one global rf_v2 call for an unpadded packed video sequence."""
    sequence_len = _sequence_length(query, input_layout)
    if (
        _sequence_length(key, input_layout) != sequence_len
        or _sequence_length(value, input_layout) != sequence_len
    ):
        raise ParametersInvalid(
            "Q, K, and V must have identical sequence lengths in rf_v2 multi-video mode."
        )
    spans = _normalize_video_spans(video_spans, sequence_len)
    permutation, inverse, dense_blocks, first_frame_blocks = _multi_span_permutation(
        spans, sequence_len, pool_size, input_layout, query.device
    )
    q_rf = _index_sequence(query, permutation, input_layout)
    k_rf = _index_sequence(key, permutation, input_layout)
    v_rf = _index_sequence(value, permutation, input_layout)
    qkv_pool = avgpool(torch.cat((q_rf, k_rf, v_rf), dim=0), pool_size, input_layout)
    return q_rf, k_rf, v_rf, qkv_pool, inverse, dense_blocks, first_frame_blocks


def do_tensor_pooling(tensor, text_len):
    tensor_t = tensor[:, :text_len, :, :]
    tensor_i = tensor[:, text_len:, :, :]

    tensor_i_pool = avgpool(tensor_i, pool_size=128)
    tensor_t_pool = avgpool(tensor_t, pool_size=128)

    tensor_pool = torch.concat((tensor_t_pool, tensor_i_pool), dim=1)
    return tensor_pool


def rain_fusion_attention(
    query,
    key,
    value,
    scale=None,
    head_num=None,
    input_layout="TND",
    select_idx=None,
    select_num_idx=None,
    blockshape=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    inner_precise=0,
):
    if is_a5_device():
        raise ParametersInvalid(_A5_RF_V2_UNSUPPORTED_MSG)

    out, _ = ops.rain_fusion_attention(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        blockshape,
        attn_mask=None,
        actual_seq_qlen=actual_seq_lengths,
        actual_seq_kvlen=actual_seq_lengths_kv,
        block_table=None,
        q_input_layout=input_layout,
        kv_input_layout=input_layout,
        head_num=head_num,
        mask_type=0,
        scale=scale,
        inner_precise=inner_precise,
        block_size=0,
    )

    return out
