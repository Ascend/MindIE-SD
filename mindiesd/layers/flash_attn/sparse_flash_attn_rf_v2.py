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
    # The unaligned path keeps the first frame in its original order.
    if frame_num == 1 and (hq % 8 != 0 or wq % 8 != 0):
        return tensor
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
    if frame_num == 1 and (hq % 8 != 0 or wq % 8 != 0):
        return tensor
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
            raise ParametersInvalid(f"video_spans[{index}] requires integer start and latent_shape=[T,H,W].") from exc
        if len(shape) != 3 or any(dim <= 0 for dim in shape):
            raise ParametersInvalid(f"video_spans[{index}].latent_shape must contain three positive integers.")
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


def _normalize_image_spans(image_spans, sequence_len):
    """Validate independently described reference-image grids."""
    if not isinstance(image_spans, Sequence) or isinstance(image_spans, (str, bytes)):
        raise ParametersInvalid("image_spans must be a sequence of span descriptors.")
    spans = []
    previous_end = 0
    for span in image_spans:
        try:
            start = int(span["start"])
            shape = tuple(int(dim) for dim in span["latent_shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ParametersInvalid("image_spans requires start and latent_shape=[H,W].") from exc
        if len(shape) != 2 or any(dim <= 0 for dim in shape):
            raise ParametersInvalid("image_spans latent_shape must contain two positive integers.")
        length = math.prod(shape)
        if start < previous_end or start + length > sequence_len:
            raise ParametersInvalid("image_spans overlaps another image or exceeds the sequence.")
        spans.append((start, shape, length))
        previous_end = start + length
    return spans


def _multi_span_permutation(
    spans, sequence_len, pool_size, input_layout, device, image_spans=(), *, return_reference_plan=False
):
    """Pack individually tiled clips into one block plan without crossing clips.

    Dense rows are used to finish a partial clip block before the next clip.
    Those mixed blocks are marked dense, which preserves full access for every
    text/image/audio token. If there are not enough dense rows, a fixed-size
    block kernel cannot represent the layout safely and the caller must fall
    back to dense attention.
    """
    spans = sorted([(*span, False) for span in spans] + [(*span, True) for span in image_spans])
    previous_end = 0
    for start, _, length, _ in spans:
        if start < previous_end:
            raise ParametersInvalid("image_spans and video_spans must not overlap.")
        previous_end = start + length
    all_indices = torch.arange(sequence_len, device=device, dtype=torch.long)
    is_video = torch.zeros(sequence_len, device=device, dtype=torch.bool)
    for start, _, length, _ in spans:
        is_video[start : start + length] = True
    dense_indices = all_indices[~is_video]
    dense_cursor = 0
    position = 0
    parts = []
    dense_blocks: set[int] = set()
    first_frame_blocks: set[int] = set()

    image_groups = []
    target_blocks = []
    for clip_index, (start, shape, length, is_image) in enumerate(spans):
        # Every clip begins on a fresh sparse block. The final clip may share
        # its tail block with dense context, but never with another clip.
        if position % pool_size:
            raise AssertionError("multi-span packing lost block alignment")
        clip_block = position // pool_size
        if is_image:
            image_groups.append(tuple(range(clip_block, clip_block + length // pool_size)))
            dense_blocks.update(range(clip_block, clip_block + math.ceil(length / pool_size)))
            spatial_shape = (1, *shape)
        else:
            if image_spans:
                target_blocks.extend(range(clip_block, clip_block + length // pool_size))
            first_frame_blocks.update(range(clip_block, clip_block + math.ceil((shape[1] * shape[2]) / pool_size)))
            spatial_shape = shape
        parts.append(_span_rearranged_indices(start, spatial_shape, input_layout, device))
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
    result = permutation, inverse, dense_blocks, first_frame_blocks
    return (*result, (tuple(image_groups), tuple(target_blocks))) if return_reference_plan else result


def pool_qk_separately(query, key, pool_size=128, input_layout="BSND"):
    """Pool independent Q/K inputs; concatenate pooled values, not full payloads.

    The single-batch/single-head BSND path retains the native Mean reduction
    and tail divisor while avoiding the large concat and strided full-block
    reduction input. Other layouts/geometries retain the stock implementation.
    No input is modified; output batch order remains Q followed by K.
    """
    if query.shape != key.shape or query.dtype != key.dtype or query.device != key.device:
        raise ValueError("query/key pooling requires equal shapes, dtypes and devices")
    if input_layout != "BSND" or query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != 1:
        return avgpool(torch.cat((query, key), dim=0), pool_size, input_layout)
    if pool_size <= 0 or query.shape[1] <= 0:
        raise ValueError("pool_size and sequence length must be positive")
    _, length, heads, dim = query.shape
    full_cells, tail_size = divmod(length, pool_size)
    pooled = []
    if full_cells:
        full = [
            tensor[:, : full_cells * pool_size].view(1, full_cells, pool_size, heads, dim).mean(dim=2)
            for tensor in (query, key)
        ]
        pooled.append(torch.cat(full, dim=0))
    if tail_size:
        tail = torch.cat((query[:, -tail_size:], key[:, -tail_size:]), dim=0)
        pooled.append(tail.view(2, 1, tail_size, heads, dim).mean(dim=2))
    return torch.cat(pooled, dim=1) if len(pooled) == 2 else pooled[0]


def get_multi_span_blockwise_mask(
    qkv_pool,
    sparsity,
    scale,
    dense_blocks,
    first_frame_blocks,
    input_layout,
    return_binary=False,
    reference_plan=((), ()),
    image_sparsity=0.0,
    *,
    retained_idx=None,
    retained_mask=None,
    pool_parts=3,
    async_index_transfer=True,
):
    """Global multi-video ranking with dense context and first frame per clip retained."""
    if pool_parts not in (2, 3):
        raise ValueError("pool_parts must be 2 (Q/K) or 3 (Q/K/V)")
    if not isinstance(async_index_transfer, bool):
        raise ValueError("async_index_transfer must be a bool")
    if qkv_pool.shape[0] == 0 or qkv_pool.shape[0] % pool_parts:
        raise ValueError("pooled batch dimension must be divisible by pool_parts")
    query_pool, key_pool = torch.chunk(qkv_pool, pool_parts, dim=0)[:2]
    if input_layout == "BSND":
        scores = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
    else:
        scores = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale
    score_matrix = torch.nn.functional.softmax(scores, dim=-1)
    keep_len = math.ceil(score_matrix.shape[-1] * (1 - sparsity))
    if keep_len < 1:  # pylint: disable=consider-using-max-builtin  # Preserve upstream mask selection.
        keep_len = 1
    threshold = torch.topk(score_matrix, k=keep_len, dim=-1).values[..., -1:]
    mask = score_matrix >= threshold
    if retained_mask is not None:
        if not return_binary:
            raise ValueError("retained_mask requires return_binary=True")
        mask.logical_or_(retained_mask.to(torch.bool))
    if retained_idx is None:
        if return_binary:
            # Pageable host memory still makes an NPU async copy wait for the
            # stream. Pin these small indices; the parallel path skips the copy.
            retained_idx = torch.tensor(
                sorted(dense_blocks | first_frame_blocks),
                dtype=torch.long,
                pin_memory=mask.device.type == "npu" and async_index_transfer,
            ).to(device=mask.device, non_blocking=async_index_transfer)
        else:
            retained_idx = torch.tensor(sorted(dense_blocks | first_frame_blocks), device=mask.device, dtype=torch.long)
    if retained_idx.numel():
        mask[:, :, retained_idx, :] = True
        mask[:, :, :, retained_idx] = True
    image_groups, target_blocks = reference_plan
    rows = sorted(set(target_blocks) - dense_blocks - first_frame_blocks)
    if image_groups and image_sparsity > 0 and rows:
        row_idx = torch.tensor(rows, device=mask.device, dtype=torch.long)
        for group in image_groups:
            if not group:
                continue
            columns = torch.tensor(group, device=mask.device, dtype=torch.long)
            values = scores.index_select(-2, row_idx).index_select(-1, columns)
            keep = max(1, math.ceil(len(group) * (1 - image_sparsity)))
            selected = values.topk(keep, dim=-1).indices
            selected_mask = torch.zeros_like(values, dtype=torch.bool).scatter_(-1, selected, True)
            mask[..., row_idx[:, None], columns[None, :]] = selected_mask
    if return_binary:
        return mask.to(torch.int8)
    select_idx = get_mask_index(mask)[0].transpose(0, 1)
    select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
    return select_idx, select_num_idx


def do_multi_span_tensor_rearrange_pooling(
    query,
    key,
    value,
    video_spans,
    pool_size,
    input_layout,
    image_spans=None,
    *,
    pool_parts=3,
    return_reference_plan=False,
):
    """Prepare RF tensors; opt in to the eighth reference-image metadata result."""
    sequence_len = _sequence_length(query, input_layout)
    if _sequence_length(key, input_layout) != sequence_len or _sequence_length(value, input_layout) != sequence_len:
        raise ParametersInvalid("Q, K, and V must have identical sequence lengths in rf_v2 multi-video mode.")
    spans = _normalize_video_spans(video_spans, sequence_len)
    images = _normalize_image_spans(image_spans, sequence_len) if image_spans is not None else []
    permutation, inverse, dense_blocks, first_frame_blocks, reference_plan = _multi_span_permutation(
        spans, sequence_len, pool_size, input_layout, query.device, images, return_reference_plan=True
    )
    if pool_parts == 3:
        q_rf = _index_sequence(query, permutation, input_layout)
        k_rf = _index_sequence(key, permutation, input_layout)
        v_rf = _index_sequence(value, permutation, input_layout)
        qkv_pool = avgpool(torch.cat((q_rf, k_rf, v_rf), dim=0), pool_size, input_layout)
    elif pool_parts in (0, 2):
        reordered = _index_sequence(torch.cat((query, key, value), dim=0), permutation, input_layout)
        q_rf, k_rf, v_rf = torch.chunk(reordered, 3, dim=0)
        qkv_pool = avgpool(reordered[: 2 * query.shape[0]], pool_size, input_layout) if pool_parts else None
    else:
        raise ValueError("pool_parts must be 0 (no pool), 2 (Q/K) or 3 (Q/K/V)")
    result = q_rf, k_rf, v_rf, qkv_pool, inverse, dense_blocks, first_frame_blocks
    return (*result, reference_plan) if return_reference_plan else result


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
