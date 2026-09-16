#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


"""MXFP4 preparation, QuantFlashAttn execution and output restoration."""

import torch
import torch_npu
import functools
import torch.nn.functional as F  # noqa: N812

MXFP4_FA_SEQ_PAD_BASE = 512
MXFP4_Q_QUANT_MODE = 3
MXFP4_K_QUANT_MODE = 3
MXFP4_V_QUANT_MODE = 3


def _mxfp4_attention_forward(
    query,
    key,
    value,
    *,
    layout,
    layout_kv,
    layout_out,
    scale,
    win_left,
    win_right,
    q_rot=None,
    k_rot=None,
    mxfp4_scale_alg=None,
    mxfp4_dst_type_max=None,
    metadata=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    mask_mode=0,
    block_table=None,
    sinks=None,
    attn_mask=None,
    return_softmax_lse=0,
):
    """Pack MXFP4 and pass the padded sequence lengths, as in the legacy class."""
    # Layouts, Q/K/V and rotations have already been validated by quant_attention.
    if layout == "BNSD":
        batch, heads, q_len, head_dim = query.shape
    else:
        batch, q_len, heads, head_dim = query.shape
    if q_rot is not None:
        query = torch.matmul(query, q_rot)
    if k_rot is not None:
        key = torch.matmul(key, k_rot)
    quant_options = {}
    if mxfp4_scale_alg is not None:
        quant_options["scale_alg"] = mxfp4_scale_alg
    if mxfp4_dst_type_max is not None:
        quant_options["dst_type_max"] = mxfp4_dst_type_max
    query = _pad_fa_seq_before_quant(query, MXFP4_FA_SEQ_PAD_BASE, layout)
    key = _pad_fa_seq_before_quant(key, MXFP4_FA_SEQ_PAD_BASE, layout_kv)
    value = _pad_fa_seq_before_quant(value, MXFP4_FA_SEQ_PAD_BASE, layout_kv)
    if layout == "BNSD":
        _, _, padded_q_len, _ = query.shape
    else:
        _, padded_q_len, _, _ = query.shape
    if layout_kv == "BNSD":
        _, kv_heads, padded_kv_len, _ = key.shape
    else:
        _, padded_kv_len, kv_heads, _ = key.shape
    seqused_q, seqused_kv = _get_qfa_seqused(batch, padded_q_len, padded_kv_len, query.device)
    q, q_scale = torch_npu.npu_dynamic_mx_quant(query, dst_type=torch_npu.float4_e2m1fn_x2, axis=-1, **quant_options)
    # QuantFlashAttn reads Q descales as BNSD even when Q itself is BSND
    # (GetQueryScaleGmFormat). K/V descales follow their own layout contracts.
    if layout == "BSND":
        q_scale = q_scale.transpose(1, 2).contiguous()
    k, k_scale = torch_npu.npu_dynamic_mx_quant(key, dst_type=torch_npu.float4_e2m1fn_x2, axis=-1, **quant_options)
    v, v_scale = torch_npu.npu_dynamic_mx_quant(
        value,
        dst_type=torch_npu.float4_e2m1fn_x2,
        axis=2 if layout_kv == "BNSD" else 1,
        **quant_options,
    )
    v_scale = _reshape_mxfp4_v_scale_for_fa(v_scale, layout_kv)

    # Metadata and execution must agree on layouts, lengths, quant modes and windows.
    kernel_options = {
        "q_quant_mode": MXFP4_Q_QUANT_MODE,
        "k_quant_mode": MXFP4_K_QUANT_MODE,
        "v_quant_mode": MXFP4_V_QUANT_MODE,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "q_dtype": torch_npu.float4_e2m1fn_x2,
        "k_dtype": torch_npu.float4_e2m1fn_x2,
        "v_dtype": torch_npu.float4_e2m1fn_x2,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "mask_mode": mask_mode,
        "win_left": win_left,
        "win_right": win_right,
        "layout_q": layout,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    }
    if metadata is None:
        metadata = torch.ops.mindiesd.quant_flash_attn_metadata(
            num_heads_q=heads,
            num_heads_kv=kv_heads,
            head_dim=head_dim,
            batch_size=batch,
            **kernel_options,
        )
    output, _ = torch.ops.mindiesd.quant_flash_attn(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        metadata=metadata,
        block_table=block_table,
        sinks=sinks,
        attn_mask=attn_mask,
        q_descale_dtype=torch_npu.float8_e8m0fnu,
        k_descale_dtype=torch_npu.float8_e8m0fnu,
        v_descale_dtype=torch_npu.float8_e8m0fnu,
        softmax_scale=scale,
        return_softmax_lse=return_softmax_lse,
        **kernel_options,
    )
    return _crop_fa_output(output, q_len, layout_out)


def _pad_fa_seq_before_quant(tensor, base, layout):
    sequence = tensor.shape[2 if layout == "BNSD" else 1]
    pad_len = (-sequence) % base
    padding = (0, 0, 0, pad_len) if layout == "BNSD" else (0, 0, 0, 0, 0, pad_len)
    return F.pad(tensor, padding) if pad_len else tensor


def _reshape_mxfp4_v_scale_for_fa(v_scale, layout):
    if layout == "BNSD":
        scale_blocks = v_scale.shape[2]
        if v_scale.dim() == 5:
            return v_scale
        if scale_blocks % 2 != 0:
            raise ValueError(f"V scale S blocks must be even for layout BNSD, got {scale_blocks}.")
        return (
            v_scale.reshape(v_scale.shape[0], v_scale.shape[1], scale_blocks // 2, 2, v_scale.shape[3])
            .transpose(-1, -2)
            .contiguous()
        )
    if layout == "BSND":
        scale_blocks = v_scale.shape[1]
        if v_scale.dim() == 5:
            return v_scale
        if scale_blocks % 2 != 0:
            raise ValueError(f"V scale S blocks must be even for layout BSND, got {scale_blocks}.")
        return (
            v_scale.reshape(v_scale.shape[0], scale_blocks // 2, 2, v_scale.shape[2], v_scale.shape[3])
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
    raise ValueError(f"Unsupported layout: {layout}, expected 'BNSD' or 'BSND'.")


@functools.lru_cache(maxsize=512)
def _get_qfa_seqused(batch_size, query_length, kv_length, device):
    device = torch.device(device)
    seqused_q = torch.full((batch_size,), query_length, dtype=torch.int32, device=device)
    seqused_kv = torch.full((batch_size,), kv_length, dtype=torch.int32, device=device)
    return seqused_q, seqused_kv


def _crop_fa_output(output, seq_len, layout):
    if layout == "BNSD" and output.shape[2] != seq_len:
        output = output[:, :, :seq_len, :]
    elif layout == "BSND" and output.shape[1] != seq_len:
        output = output[:, :seq_len, :, :]
    return output
