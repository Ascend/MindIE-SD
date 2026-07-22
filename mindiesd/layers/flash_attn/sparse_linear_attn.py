#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import torch
from torch import nn
import torch.nn.functional as F

from ...utils.exception import ParametersInvalid
from .._custom_ops import block_sparse_attention
from .sparse_linear_attn_triton import _attention, get_block_map

_SUPPORTED_SPARSE_BACKENDS = ('triton', 'ascendc')
# aclnnBlockSparseAttention tiling constraints (see block_sparse_attention_tiling.cpp).
_ASCENDC_SUPPORTED_HEAD_DIMS = (64, 128)
_ASCENDC_BLKK_ALIGN = 128
# SparseLinearAttention triton path: get_block_map compress_kernel UB limit on NPU.
# _attention.apply also asserts {16,32,64,128,256}, but D=256 cannot compile in get_block_map.
_TRITON_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)
# Triton SLA product matrix (original triton-ascend PR).
_TRITON_SUPPORTED_BLOCK_SIZES = (64, 128)


def _ascendc_shape_supported(head_dim, blkq, blkk):
    return blkq > 0 and blkk > 0 and head_dim in _ASCENDC_SUPPORTED_HEAD_DIMS and blkk % _ASCENDC_BLKK_ALIGN == 0


def _triton_shape_supported(head_dim, blkq, blkk):
    return (
        blkq > 0
        and blkk > 0
        and head_dim in _TRITON_SUPPORTED_HEAD_DIMS
        and blkq in _TRITON_SUPPORTED_BLOCK_SIZES
        and blkk in _TRITON_SUPPORTED_BLOCK_SIZES
    )


def _resolve_sparse_attn_backend(head_dim, blkq, blkk, *, where=''):
    if _ascendc_shape_supported(head_dim, blkq, blkk):
        return 'ascendc'
    if _triton_shape_supported(head_dim, blkq, blkk):
        return 'triton'
    suffix = f' ({where})' if where else ''
    raise ParametersInvalid(
        f"No sparse attention backend supports head_dim={head_dim}, BLKQ={blkq}, BLKK={blkk}{suffix}. "
        f"ascendc requires head_dim in {_ASCENDC_SUPPORTED_HEAD_DIMS} and BLKK multiple of "
        f"{_ASCENDC_BLKK_ALIGN}; triton requires head_dim in {_TRITON_SUPPORTED_HEAD_DIMS} and "
        f"BLKQ/BLKK in {_TRITON_SUPPORTED_BLOCK_SIZES}."
    )


def _validate_inner_precise(inner_precise, is_950, use_bf16):
    if is_950:
        if inner_precise != 4:
            raise ParametersInvalid(f"inner_precise must be 4 on 950 series devices, got {inner_precise}.")
    elif use_bf16:
        if inner_precise != 0:
            raise ParametersInvalid(f"inner_precise must be 0 with bf16 on non-950 devices, got {inner_precise}.")
    else:
        if inner_precise not in (0, 1):
            raise ParametersInvalid(f"inner_precise must be 0 or 1 with fp16 on non-950 devices, got {inner_precise}.")


class SparseLinearAttention(nn.Module):
    def __init__(
        self,
        head_dim,
        topk,
        feature_map='softmax',
        BLKQ=64,
        BLKK=64,
        use_bf16=True,
        tie_feature_map_qk=True,
        inner_precise=None,
    ):
        R'''
        Args:
            head_dim: dimension of each head. ascendc: 64 or 128; triton fallback: 16/32/64/128.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query. ascendc: positive; triton fallback: 64 or 128.
            BLKK: block size for key. ascendc: positive multiple of 128; triton fallback: 64 or 128.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
            inner_precise: precision mode for ascendc backend only (triton ignores this).
                Supported values:
                  0 — high precision; supports both bf16 and fp16.
                  1 — high performance; fp16 only, NOT supported with bf16, may reduce precision.
                  4 — required for 950 series chips; supports both bf16 and fp16.
                When None (default), auto-selects based on device: 950 -> 4, otherwise -> 0.
            Sparse flash backend is auto-selected: ascendc when shape constraints are met,
            otherwise triton when its constraints are met.
        '''
        super().__init__()
        if BLKQ <= 0 or BLKK <= 0:
            raise ParametersInvalid(f"BLKQ and BLKK must be positive, got BLKQ={BLKQ}, BLKK={BLKK}.")
        _resolve_sparse_attn_backend(head_dim, BLKQ, BLKK, where='head_dim')
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.inner_precise = inner_precise
        try:
            dev_name = torch.npu.get_device_properties(torch.npu.current_device()).name
            self._is_950 = '950' in dev_name
        except Exception:
            self._is_950 = False
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':

            def elu_feature_map(x):
                return F.elu(x) + 1

            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':

            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)

            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
        '''
        dtype = q.dtype

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        head_dim = q.shape[-1]
        if head_dim != self.proj_l.in_features:
            raise ParametersInvalid(
                f"query head dimension {head_dim} does not match module head_dim {self.proj_l.in_features}."
            )

        if q.device.type != 'npu':
            raise ParametersInvalid(f"SparseLinearAttention requires query/key/value on NPU; got device {q.device}.")

        sparse_attn_backend = _resolve_sparse_attn_backend(head_dim, self.BLKQ, self.BLKK, where='query head dimension')

        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        scale = q.shape[-1] ** -0.5

        if sparse_attn_backend == 'triton':
            atten_mask = None
            causal = False
            o_s, _ = _attention.apply(
                q, k, v, sparse_map, lut, real_topk, atten_mask, causal, scale, self.BLKQ, self.BLKK
            )
        else:
            inner_precise = self.inner_precise if self.inner_precise is not None else (4 if self._is_950 else 0)
            _validate_inner_precise(inner_precise, self._is_950, self.dtype == torch.bfloat16)
            o_s, _ = block_sparse_attention(
                query=q,
                key=k,
                value=v,
                block_sparse_mask=sparse_map,
                block_shape=[self.BLKQ, self.BLKK],
                q_input_layout='BNSD',
                kv_input_layout='BNSD',
                num_key_value_heads=q.shape[1],
                scale_value=scale,
                inner_precise=inner_precise,
                softmax_lse_flag=0,
            )
        q = self.feature_map_q(q).contiguous().to(self.dtype)
        k = self.feature_map_k(k).contiguous().to(self.dtype)

        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))

        o_l = calc_linear(q, k, v)

        with torch.amp.autocast('npu', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        o = (o_s + o_l).to(dtype)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o
