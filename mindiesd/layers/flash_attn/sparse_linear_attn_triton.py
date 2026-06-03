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
# pylint: disable=too-many-lines

import torch

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
from triton.runtime import driver


def get_npu_aicore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]


# ---------------------------------------utils---------------------------------#
@triton.jit
def compress_kernel(x, mean_x, L: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr):
    idx_bh = tl.program_id(0)
    nproc = tl.program_id(1)

    comp_l_len = (L + BLOCK_L - 1) // BLOCK_L
    comp_l_start = nproc
    comp_l_step = tl.num_programs(1)
    for comp_l_idx in tl.range(
        comp_l_start,
        comp_l_len,
        comp_l_step,
    ):
        l_idx = comp_l_idx * BLOCK_L  # index of x dimension 3
        start_x = x + idx_bh * L * D + l_idx * D
        start_mean_x = mean_x + idx_bh * comp_l_len * D + comp_l_idx * D

        # load x range: [][][BLOCK_L][D]
        range_x = tl.arange(0, BLOCK_L)[:, None] * D + tl.arange(0, D)[None, :]
        mask_x = l_idx + tl.arange(0, BLOCK_L)[:, None] < L
        # save mean_x range: [][][1][D]
        range_mean_x = tl.arange(0, D)

        # load, compute, save
        i = tl.load(start_x + range_x, mask=mask_x)  # shape: (BLOCK_L, D)
        len_i = min(BLOCK_L, L - l_idx)
        mean_i = tl.sum(i, axis=0) / len_i
        tl.store(start_mean_x + range_mean_x, mean_i)


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (B * H, min(L_BLOCKS, 32768 // (B * H)))
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


@triton.jit
def softmax_with_mask_with_update(
    vtaskId,
    qk,
    sm_scale,
    attn_mask_ptr,
    m_i_buffer,
    l_i_buffer,
    alpha_buffer,
    p_nz_buffer,
    qk_scale,
    cast_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    sub_vec_id = al.sub_vec_id()
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    alpha = bl.to_tensor(alpha_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij)
    tmp_max = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    tmp_max = bl.to_tensor(tmp_max)

    atten_mask = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.int8)

    if STAGE == 1:
        curr_mask_ptr = tl.advance(attn_mask_ptr, ((sub_vec_id * BLOCK_M // 2).to(tl.int32), 0))
        atten_mask = tl.load(curr_mask_ptr)

    BLOCK_N_UNROLL: tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            if STAGE == 1:
                mask = al.extract_slice(atten_mask, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1]).to(
                    tl.int1
                )  # 64*128 -> 1*64
                qk_loop = qk_loop + tl.where(mask, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            if STAGE == 1:
                mask_unroll = al.extract_slice(atten_mask, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1]).to(
                    tl.int1
                )
                qk_loop_unroll = qk_loop_unroll + tl.where(mask_unroll, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)

            tmp_max = al.insert_slice(tmp_max, row_max_agg, [loop], [1], [1])
        m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)

            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1])

            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(
                p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL // 16, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1]
            )

            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

    al.copy(bl.to_buffer(m_ij, al.ascend_address_space.UB), m_i_buffer)

    bl.to_buffer(l_i, bind_buffer=l_i_buffer)
    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)
    bl.to_buffer(alpha, bind_buffer=alpha_buffer)


@triton.jit
def softmax_no_mask_with_update(
    vtaskId,
    qk,
    sm_scale,
    m_i_buffer,
    l_i_buffer,
    alpha_buffer,
    p_nz_buffer,
    qk_scale,
    cast_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    alpha = bl.to_tensor(alpha_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij)
    tmp_max = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    tmp_max = bl.to_tensor(tmp_max)

    BLOCK_N_UNROLL: tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)

            tmp_max = al.insert_slice(tmp_max, row_max_agg, [loop], [1], [1])

        m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)

            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1])

            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(
                p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL // 16, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1]
            )

            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

    al.copy(bl.to_buffer(m_ij, al.ascend_address_space.UB), m_i_buffer)

    bl.to_buffer(l_i, bind_buffer=l_i_buffer)
    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)
    bl.to_buffer(alpha, bind_buffer=alpha_buffer)


@triton.jit
def softmax_with_mask_no_update(
    qk,
    sm_scale,
    attn_mask_ptr,
    m_i_buffer,
    l_i_buffer,
    p_nz_buffer,
    qk_scale,
    cast_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    sub_vec_id = al.sub_vec_id()
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij_buffer)
    m_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    m_ij = bl.to_tensor(m_ij_buffer)

    atten_mask = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.int8)
    if STAGE == 1:
        curr_mask_ptr = tl.advance(attn_mask_ptr, ((sub_vec_id * BLOCK_M // 2).to(tl.int32), 0))
        atten_mask = tl.load(curr_mask_ptr)  # 64*128

    BLOCK_N_UNROLL: tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            if STAGE == 1:
                mask = al.extract_slice(atten_mask, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1]).to(
                    tl.int1
                )  # 64*128 -> 1*64
                qk_loop = qk_loop + tl.where(mask, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            if STAGE == 1:
                mask_unroll = al.extract_slice(atten_mask, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1]).to(
                    tl.int1
                )
                qk_loop_unroll = qk_loop_unroll + tl.where(mask_unroll, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)

            m_ij = al.insert_slice(m_ij, row_max_agg, [loop], [1], [1])

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)

            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1])

            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(
                p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL // 16, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1]
            )

            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    al.copy(m_ij_buffer, m_i_buffer)
    al.copy(l_ij_buffer, l_i_buffer)

    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)


@triton.jit
def softmax_no_mask_no_update(
    qk,
    sm_scale,
    m_i_buffer,
    l_i_buffer,
    p_nz_buffer,
    qk_scale,
    cast_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij_buffer)
    m_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    m_ij = bl.to_tensor(m_ij_buffer)

    BLOCK_N_UNROLL: tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)

            m_ij = al.insert_slice(m_ij, row_max_agg, [loop], [1], [1])

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)

            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1])

            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(
                p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL // 16, loop, 0], [BLOCK_N_UNROLL // 16, 1, 16], [1, 1, 1]
            )

            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    al.copy(l_ij_buffer, l_i_buffer)
    al.copy(m_ij_buffer, m_i_buffer)

    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)


@triton.jit
def softmax_vf_select(
    vtaskId,
    need_mask,
    need_update,
    qk,
    sm_scale,
    attn_mask_ptr,
    m_i_buffer,
    l_i_buffer,
    alpha_buffer,
    p_nz_buffer,
    qk_scale,
    cast_dtype,
    BLOCK_M,
    BLOCK_N,
    STAGE,
):
    if need_mask & need_update:
        softmax_with_mask_with_update(
            vtaskId,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_buffer,
            l_i_buffer,
            alpha_buffer,
            p_nz_buffer,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif need_mask & ~need_update:
        softmax_with_mask_no_update(
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_buffer,
            l_i_buffer,
            p_nz_buffer,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif ~need_mask & need_update:
        softmax_no_mask_with_update(
            vtaskId,
            qk,
            sm_scale,
            m_i_buffer,
            l_i_buffer,
            alpha_buffer,
            p_nz_buffer,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    else:
        softmax_no_mask_no_update(
            qk, sm_scale, m_i_buffer, l_i_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE
        )


@triton.jit
def process_v1(
    qk_ub_ping,
    qk_ub_pong,
    p_l1_ping,
    p_l1_pong,
    attn_mask_ptr,
    m_i_tb0,
    m_i_tb1,
    m_i_tb2,
    l_i_tb0,
    l_i_tb1,
    l_i_tb2,
    alpha_tb0,
    alpha_tb1,
    alpha_tb2,
    sm_scale,
    vtaskId,
    v_s1_task_mod3,
    cast_dtype,
    need_mask,
    need_update,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    sub_vec_id = al.sub_vec_id()

    p_nz = bl.alloc(cast_dtype, [BLOCK_N // 16, BLOCK_M // 32 * 16, 16], al.ascend_address_space.UB)
    al.multibuffer(p_nz, 2)

    qk_scale = bl.alloc(tl.float32, [BLOCK_M // 2, BLOCK_N], al.ascend_address_space.UB)
    qk_scale = bl.to_tensor(qk_scale)

    if (vtaskId & 1) == 1:
        qk = bl.to_tensor(qk_ub_ping)
    else:
        qk = bl.to_tensor(qk_ub_pong)

    if v_s1_task_mod3 == 0 and (vtaskId - 1) % 3 == 0:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb0,
            l_i_tb0,
            alpha_tb0,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif v_s1_task_mod3 == 0 and (vtaskId - 1) % 3 == 1:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb0,
            l_i_tb0,
            alpha_tb1,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif v_s1_task_mod3 == 0 and (vtaskId - 1) % 3 == 2:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb0,
            l_i_tb0,
            alpha_tb2,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif v_s1_task_mod3 == 1 and (vtaskId - 1) % 3 == 0:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb1,
            l_i_tb1,
            alpha_tb0,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif v_s1_task_mod3 == 1 and (vtaskId - 1) % 3 == 1:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb1,
            l_i_tb1,
            alpha_tb1,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif v_s1_task_mod3 == 1 and (vtaskId - 1) % 3 == 2:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb1,
            l_i_tb1,
            alpha_tb2,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif v_s1_task_mod3 == 2 and (vtaskId - 1) % 3 == 0:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb2,
            l_i_tb2,
            alpha_tb0,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    elif v_s1_task_mod3 == 2 and (vtaskId - 1) % 3 == 1:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb2,
            l_i_tb2,
            alpha_tb1,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )
    else:
        softmax_vf_select(
            vtaskId,
            need_mask,
            need_update,
            qk,
            sm_scale,
            attn_mask_ptr,
            m_i_tb2,
            l_i_tb2,
            alpha_tb2,
            p_nz,
            qk_scale,
            cast_dtype,
            BLOCK_M,
            BLOCK_N,
            STAGE,
        )

    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    p_nz = bl.to_tensor(p_nz)
    if (vtaskId & 1) == 1:
        p_l1_ping_sub = bl.subview(
            p_l1_ping,
            [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0],
            [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16],
            [1, 1, 1, 1],
        )
        al.copy_from_ub_to_l1(
            bl.to_buffer(p_nz.reshape(BLOCK_N // 16, BLOCK_M // 32, 16, 16), al.ascend_address_space.UB), p_l1_ping_sub
        )
    else:
        p_l1_pong_sub = bl.subview(
            p_l1_pong,
            [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0],
            [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16],
            [1, 1, 1, 1],
        )
        al.copy_from_ub_to_l1(
            bl.to_buffer(p_nz.reshape(BLOCK_N // 16, BLOCK_M // 32, 16, 16), al.ascend_address_space.UB), p_l1_pong_sub
        )

    al.sync_block_set("vector", "cube", 4, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)


@triton.jit
def vec_prefree_s_ub():
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def vec_prefree_pv_ub():
    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def vec_postwait_p_l1():
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)


@triton.jit
def cube_prefree_p_l1():
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)


@triton.jit
def cube_postwait_s_ub():
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def cube_postwait_pv_ub():
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _qk_matmul(q, K_block_ptr, qk_ub_ping, qk_ub_pong, qk_l0c, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, sid):
    k = tl.load(K_block_ptr)
    trans_k = tl.trans(k)
    qk = tl.dot(q, trans_k)
    bl.to_buffer(qk, bind_buffer=qk_l0c)
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    if (sid & 1) == 0:
        qk_ub = bl.to_tensor(qk_ub_ping)
    else:
        qk_ub = bl.to_tensor(qk_ub_pong)

    al.fixpipe(
        qk, bl.to_buffer(qk_ub, al.ascend_address_space.UB), al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT
    )

    al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)


@triton.jit
def _pv_matmul(
    p_l1_ping,
    p_l1_pong,
    pv_ub_ping,
    pv_ub_pong,
    pv_l0c,
    V_block_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pvid,
):
    v = tl.load(V_block_ptr)

    # wait for vec to complete p's transfer
    al.sync_block_wait("vector", "cube", 4, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)

    if (pvid & 1) == 0:
        p_l1 = bl.to_tensor(p_l1_ping, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub_ping)
    else:
        p_l1 = bl.to_tensor(p_l1_pong, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub_pong)

    pv = tl.dot(p_l1, v)
    bl.to_buffer(pv, bind_buffer=pv_l0c)

    # m free P buffer to vec
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

    # fixpipe allocate PV buffer from vec
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.fixpipe(
        pv, bl.to_buffer(pv_ub, al.ascend_address_space.UB), al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT
    )

    # fixpipe indicate to vec that PV transfer completes
    al.sync_block_set("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)


@triton.jit
def _flash_update(
    pv_ub_ping,
    pv_ub_pong,
    alpha_tb0,
    alpha_tb1,
    alpha_tb2,
    acc_buffer,
    v_s1_idx_mod3,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    pvid,
    update_acc,
):
    al.sync_block_wait("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    if (pvid & 1) == 1:
        pv = bl.to_tensor(pv_ub_ping)
    else:
        pv = bl.to_tensor(pv_ub_pong)
    if (pvid - 3) % 3 == 0:
        alpha = bl.to_tensor(alpha_tb0)
    elif (pvid - 3) % 3 == 1:
        alpha = bl.to_tensor(alpha_tb1)
    else:
        alpha = bl.to_tensor(alpha_tb2)

    if update_acc:
        acc_tensor = bl.to_tensor(acc_buffer)
        acc_tensor = acc_tensor * alpha[:, None] + pv
        bl.to_buffer(acc_tensor, bind_buffer=acc_buffer)
    else:
        if (pvid & 1) == 1:
            al.copy(pv_ub_ping, acc_buffer)
        else:
            al.copy(pv_ub_pong, acc_buffer)

    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def update_s2_loop(taskId_mod3, cur_s2_idx, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = cur_s2_idx, s2_idx_2, s2_idx_3, s2_idx_4  # pylint: disable=self-assigning-variable
    elif taskId_mod3 == 1:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = s2_idx_1, cur_s2_idx, s2_idx_3, s2_idx_4  # pylint: disable=self-assigning-variable
    elif taskId_mod3 == 2:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = s2_idx_1, s2_idx_2, cur_s2_idx, s2_idx_4  # pylint: disable=self-assigning-variable
    else:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = s2_idx_1, s2_idx_2, s2_idx_3, cur_s2_idx  # pylint: disable=self-assigning-variable
    return s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4


@triton.jit
def is_need_update(taskId_mod3, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        is_need = not s2_idx_1 == 0
    elif taskId_mod3 == 1:
        is_need = not s2_idx_2 == 0
    elif taskId_mod3 == 2:
        is_need = not s2_idx_3 == 0
    else:
        is_need = not s2_idx_4 == 0
    return is_need


@triton.jit
def is_last_skv(taskId_mod3, s2_idx_1, s2_size_1, s2_idx_2, s2_size_2, s2_idx_3, s2_size_3, s2_idx_4, s2_size_4):
    if taskId_mod3 == 0:
        is_reach = s2_idx_1 == s2_size_1 - 1
    elif taskId_mod3 == 1:
        is_reach = s2_idx_2 == s2_size_2 - 1
    elif taskId_mod3 == 2:
        is_reach = s2_idx_3 == s2_size_3 - 1
    else:
        is_reach = s2_idx_4 == s2_size_4 - 1
    return is_reach


@triton.jit
def is_first_skv_loop(taskId_mod3, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        is_first = s2_idx_1 == 0
    elif taskId_mod3 == 1:
        is_first = s2_idx_2 == 0
    elif taskId_mod3 == 2:
        is_first = s2_idx_3 == 0
    else:
        is_first = s2_idx_4 == 0
    return is_first


@triton.jit
def get_s_offset(taskId_mod3, head_num, seqlen, stride, b_idx, n_idx, s_idx):
    s_offset = (b_idx.to(tl.int64) * head_num + n_idx.to(tl.int64)) * seqlen + s_idx.to(tl.int64) * stride
    return s_offset.to(tl.int64)


@triton.jit
def get_cur_task(
    taskId_mod3,
    b_idx_1,
    n_idx_1,
    s1_idx_1,
    s2_idx_1,
    s2_size_1,
    b_idx_2,
    n_idx_2,
    s1_idx_2,
    s2_idx_2,
    s2_size_2,
    b_idx_3,
    n_idx_3,
    s1_idx_3,
    s2_idx_3,
    s2_size_3,
    b_idx_4,
    n_idx_4,
    s1_idx_4,
    s2_idx_4,
    s2_size_4,
):
    if taskId_mod3 == 0:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1
    elif taskId_mod3 == 1:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2
    elif taskId_mod3 == 2:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3
    else:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4
    return cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size


@triton.jit
def update_pos(s1_cur_idx, s1_step, batch_size, head_num, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N):
    bn_idx = s1_cur_idx // NUM_BLOCKS_M  # total batch_offset * head_num_offset
    s1_idx = s1_cur_idx % NUM_BLOCKS_M  # remain s_offset
    b_idx = bn_idx // head_num  # batch_offset
    n_idx = bn_idx % head_num  # head_num_offset
    if STAGE == 1:
        hi = s1_idx + 1
    else:
        hi = (N_CTX + BLOCK_N - 1) // BLOCK_N

    return b_idx, n_idx, s1_idx, hi


@triton.jit
def update_task(taskId, task_cnt, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4):
    s1_task_mod3 = task_cnt % 3
    if taskId & 3 == 0:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = (  # pylint: disable=self-assigning-variable
            s1_task_mod3,
            v_s1_task_mod3_2,
            v_s1_task_mod3_3,
            v_s1_task_mod3_4,
        )
    elif taskId & 3 == 1:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = (  # pylint: disable=self-assigning-variable
            v_s1_task_mod3_1,
            s1_task_mod3,
            v_s1_task_mod3_3,
            v_s1_task_mod3_4,
        )
    elif taskId & 3 == 2:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = (  # pylint: disable=self-assigning-variable
            v_s1_task_mod3_1,
            v_s1_task_mod3_2,
            s1_task_mod3,
            v_s1_task_mod3_4,
        )
    else:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = (  # pylint: disable=self-assigning-variable
            v_s1_task_mod3_1,
            v_s1_task_mod3_2,
            v_s1_task_mod3_3,
            s1_task_mod3,
        )

    return v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4


@triton.jit
def get_s_task(taskId, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4):
    if taskId & 3 == 0:
        cur_s1_task_mod3 = v_s1_task_mod3_1
    elif taskId & 3 == 1:
        cur_s1_task_mod3 = v_s1_task_mod3_2
    elif taskId & 3 == 2:
        cur_s1_task_mod3 = v_s1_task_mod3_3
    else:
        cur_s1_task_mod3 = v_s1_task_mod3_4

    return cur_s1_task_mod3


@triton.jit
def create_task(
    taskId,
    b_idx,
    n_idx,
    s1_idx,
    s2_idx,
    s2_size,
    b_idx_1,
    n_idx_1,
    s1_idx_1,
    s2_idx_1,
    s2_size_1,
    b_idx_2,
    n_idx_2,
    s1_idx_2,
    s2_idx_2,
    s2_size_2,
    b_idx_3,
    n_idx_3,
    s1_idx_3,
    s2_idx_3,
    s2_size_3,
    b_idx_4,
    n_idx_4,
    s1_idx_4,
    s2_idx_4,
    s2_size_4,
):
    if taskId & 3 == 0:
        b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    elif taskId & 3 == 1:
        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    elif taskId & 3 == 2:
        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    else:
        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4 = b_idx, n_idx, s1_idx, s2_idx, s2_size

    return (
        b_idx_1,
        n_idx_1,
        s1_idx_1,
        s2_idx_1,
        s2_size_1,
        b_idx_2,
        n_idx_2,
        s1_idx_2,
        s2_idx_2,
        s2_size_2,
        b_idx_3,
        n_idx_3,
        s1_idx_3,
        s2_idx_3,
        s2_size_3,
        b_idx_4,
        n_idx_4,
        s1_idx_4,
        s2_idx_4,
        s2_size_4,
    )


@triton.jit
def create_and_get_basic_pos(s1_cur_idx, s1_step, batch, head_num, s1_start, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N):
    b_idx, n_idx, s1_idx, s2_size = update_pos(
        s1_cur_idx, s1_step, batch, head_num, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N
    )

    return (b_idx, n_idx, s1_idx, s2_size)


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    ATTEN_MASK,
    M,
    Out,
    sparse_start_idx,
    sm_scale: tl.constexpr,  #
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,  #
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,  #
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,  #
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,  #
    stride_am: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_BLOCKS_PER_CORE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
    CORE_NUM: tl.constexpr,
    TOPK: tl.constexpr,
    LUT,
):
    pid = tl.program_id(0)
    # warm up stage nums
    preload = 3
    cast_dtype = Q.dtype.element_ty

    qk_ub_ping = bl.alloc(tl.float32, [BLOCK_M // 2, BLOCK_N], al.ascend_address_space.UB)
    qk_ub_pong = bl.alloc(tl.float32, [BLOCK_M // 2, BLOCK_N], al.ascend_address_space.UB)

    p_l1_ping = bl.alloc(cast_dtype, [BLOCK_N // 16, BLOCK_M // 16, 16, 16], al.ascend_address_space.L1)
    p_l1_pong = bl.alloc(cast_dtype, [BLOCK_N // 16, BLOCK_M // 16, 16, 16], al.ascend_address_space.L1)

    pv_ub_ping = bl.alloc(tl.float32, [BLOCK_M // 2, HEAD_DIM], al.ascend_address_space.UB, is_mem_unique=True)
    pv_ub_pong = bl.alloc(tl.float32, [BLOCK_M // 2, HEAD_DIM], al.ascend_address_space.UB, is_mem_unique=True)

    multi_core_limit = NUM_BLOCKS
    last_loop = 0
    last_second_loop = 0
    last_third_loop = 0
    if STAGE != 1:
        last_third_loop = (NUM_BLOCKS - pid + CORE_NUM - 1) // CORE_NUM * CORE_NUM + pid
        last_second_loop = last_third_loop + CORE_NUM
        last_loop = last_second_loop + CORE_NUM
        multi_core_limit += 3 * CORE_NUM
        start_block, end_block, step = pid, multi_core_limit, CORE_NUM
    else:
        start_block = tl.load(sparse_start_idx + pid)
        end_block = tl.load(sparse_start_idx + pid + 1)
        step = 1
        multi_core_limit += 3
        last_third_loop = end_block
        last_second_loop = end_block + 1
        last_loop = end_block + 2
        end_block = end_block + preload

    with al.scope(core_mode="cube"):
        # cube keeps own taskInfo[4]
        # mark cur_task's batch_idx, head_num_idx, seq_q_idx, seq_kv_idx, seq_kv_loop_size
        # c1 always use task_info[cur_idx] as task producer
        # c2 always use task_info[cur_idx-2] as task consumer
        # the Deferred Consumption masks synchronization issues
        # task_id marks the task_idx as (task_id - consumer_stage)

        b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1 = 0, 0, 0, 0, 0  # task_1 V1
        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2 = 0, 0, 0, 0, 0  # task_2 C1
        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3 = 0, 0, 0, 0, 0
        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4 = 0, 0, 0, 0, 0
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_size = 0, 0, 0, 0
        taskId = 0

        cube_prefree_p_l1()
        # =================== Q sequence length loop ===================
        for sq_loop_idx in al.parallel(start_block, end_block, step):
            # =================== 判断是否cool down阶段 ===================
            is_last_loop = sq_loop_idx == last_loop
            is_last_second_loop = sq_loop_idx == last_second_loop
            is_last_third_loop = sq_loop_idx == last_third_loop
            not_last = not is_last_loop
            not_last_three = (not is_last_loop and not is_last_second_loop) and not is_last_third_loop

            # =================== 获取 producer 信息 ===================
            if not_last_three:
                (cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_size) = create_and_get_basic_pos(
                    sq_loop_idx, CORE_NUM, Z, H, pid, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N
                )
                # =================== 最后preload次循环解决cool down问题 所以发射次数固定 ===================
                if STAGE == 2:
                    cur_s2_size = TOPK
            if not not_last_three:
                cur_s2_size = 1

            # =================== l0c pingpong 需要写在for循环内 配合编译选项生效 ===================
            qk_l0c = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.L0C, is_mem_unique=True)
            pv_l0c = bl.alloc(tl.float32, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L0C, is_mem_unique=True)

            # =================== 常驻Q依赖 ===================
            q_l1_keep = bl.alloc(Q.dtype.element_ty, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L1)
            # =================== KV sequence length loop ===================
            for skv_loop_idx in range(0, cur_s2_size):
                # 记录topk中的index
                real_s2_idx = skv_loop_idx
                if STAGE == 2 and not_last_three:
                    topk_offset = ((cur_b_idx * H + cur_n_idx) * NUM_BLOCKS_M + cur_s1_idx) * TOPK + skv_loop_idx
                    real_s2_idx = tl.load(LUT + topk_offset).to(tl.int32)

                # create and push task to producer stack
                if not_last_three:
                    (
                        b_idx_1,
                        n_idx_1,
                        s1_idx_1,
                        s2_idx_1,
                        s2_size_1,
                        b_idx_2,
                        n_idx_2,
                        s1_idx_2,
                        s2_idx_2,
                        s2_size_2,
                        b_idx_3,
                        n_idx_3,
                        s1_idx_3,
                        s2_idx_3,
                        s2_size_3,
                        b_idx_4,
                        n_idx_4,
                        s1_idx_4,
                        s2_idx_4,
                        s2_size_4,
                    ) = create_task(
                        taskId,
                        cur_b_idx,
                        cur_n_idx,
                        cur_s1_idx,
                        skv_loop_idx,
                        cur_s2_size,
                        b_idx_1,
                        n_idx_1,
                        s1_idx_1,
                        s2_idx_1,
                        s2_size_1,
                        b_idx_2,
                        n_idx_2,
                        s1_idx_2,
                        s2_idx_2,
                        s2_size_2,
                        b_idx_3,
                        n_idx_3,
                        s1_idx_3,
                        s2_idx_3,
                        s2_size_3,
                        b_idx_4,
                        n_idx_4,
                        s1_idx_4,
                        s2_idx_4,
                        s2_size_4,
                    )

                q_rs = (
                    get_s_offset(taskId & 3, H, N_CTX, BLOCK_M, cur_b_idx, cur_n_idx, cur_s1_idx)
                    + tl.arange(0, BLOCK_M)[:, None]
                )
                q_cs = tl.arange(0, HEAD_DIM)[None, :]

                q_ptr = Q + q_rs * stride_qm + q_cs * stride_qk

                k_rs = (
                    get_s_offset(taskId & 3, H, N_CTX, BLOCK_N, cur_b_idx, cur_n_idx, real_s2_idx)
                    + tl.arange(0, BLOCK_N)[:, None]
                )

                k_cs = tl.arange(0, HEAD_DIM)[None, :]
                k_ptr = K + k_rs * stride_kn + k_cs * stride_kk

                if (not is_last_loop and not is_last_second_loop) and not is_last_third_loop:
                    # Q 常驻 L1 适配逻辑
                    if skv_loop_idx == 0:
                        q = tl.load(q_ptr)
                        # q = tl.load(q_ptr, mask=offs_m[:, None] < N_CTX)
                        bl.to_buffer(tensor=q, bind_buffer=q_l1_keep)
                    else:
                        q = bl.to_tensor(q_l1_keep)
                    _qk_matmul(q, k_ptr, qk_ub_ping, qk_ub_pong, qk_l0c, HEAD_DIM, BLOCK_N, taskId)

                # get c2 task
                c2_use_b_idx, c2_use_n_idx, c2_use_s1_idx, c2_use_s2_idx, c2_use_s2_size = get_cur_task(
                    (taskId + 2) & 3,
                    b_idx_1,
                    n_idx_1,
                    s1_idx_1,
                    s2_idx_1,
                    s2_size_1,
                    b_idx_2,
                    n_idx_2,
                    s1_idx_2,
                    s2_idx_2,
                    s2_size_2,
                    b_idx_3,
                    n_idx_3,
                    s1_idx_3,
                    s2_idx_3,
                    s2_size_3,
                    b_idx_4,
                    n_idx_4,
                    s1_idx_4,
                    s2_idx_4,
                    s2_size_4,
                )

                if taskId > 1 and not_last:
                    real_v_s2_idx = c2_use_s2_idx
                    if STAGE == 2:
                        c2_topk_offset = (
                            (c2_use_b_idx * H + c2_use_n_idx) * NUM_BLOCKS_M + c2_use_s1_idx
                        ) * TOPK + c2_use_s2_idx
                        real_v_s2_idx = tl.load(LUT + c2_topk_offset)

                    v_rs = (
                        get_s_offset((taskId + 2) & 3, H, N_CTX, BLOCK_N, c2_use_b_idx, c2_use_n_idx, real_v_s2_idx)
                        + tl.arange(0, BLOCK_N)[:, None]
                    )
                    v_cs = tl.arange(0, HEAD_DIM)[None, :]
                    v_ptr = V + v_rs * stride_vn + v_cs * stride_vk

                    _pv_matmul(
                        p_l1_ping,
                        p_l1_pong,
                        pv_ub_ping,
                        pv_ub_pong,
                        pv_l0c,
                        v_ptr,
                        HEAD_DIM,
                        BLOCK_M,
                        BLOCK_N,
                        taskId - 2,
                    )

                taskId += 1

        cube_postwait_s_ub()
        cube_postwait_pv_ub()

    with al.scope(core_mode="vector"):
        v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1 = 0, 0, 0, 0, 0  # task_1 V1
        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2 = 0, 0, 0, 0, 0  # task_2 C1
        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3 = 0, 0, 0, 0, 0
        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4 = 0, 0, 0, 0, 0
        v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_size = 0, 0, 0, 0

        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = 0, 0, 0, 0
        s1_task_cnt = 0
        vtaskId = 0

        # =================== use 3 buffer to keep data in UB ===================
        # softmax max
        m_i_tb0 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        m_i_tb1 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        m_i_tb2 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        # softmax sum
        l_i_tb0 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        l_i_tb1 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        l_i_tb2 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        # exp(max - maxi)
        alpha_tb0 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB, is_mem_unique=True)
        alpha_tb1 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB, is_mem_unique=True)
        alpha_tb2 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB, is_mem_unique=True)
        # flash update lse
        acc_buffer = bl.alloc(tl.float32, [BLOCK_M // 2, HEAD_DIM], al.ascend_address_space.UB, is_mem_unique=True)

        vec_prefree_s_ub()
        vec_prefree_pv_ub()

        # =================== Q sequence length loop ===================
        for sq_loop_idx in al.parallel(start_block, end_block, step):
            # =================== 循环控制和taskInfo[4]与Cube完全相同 等待支持后可以合并为一套 ===================
            v_is_last_loop = sq_loop_idx == last_loop
            v_is_last_second_loop = sq_loop_idx == last_second_loop
            v_is_last_third_loop = sq_loop_idx == last_third_loop
            v_not_last_two = not v_is_last_loop and not v_is_last_second_loop
            v_not_last_three = (not v_is_last_loop and not v_is_last_second_loop) and not v_is_last_third_loop

            if v_not_last_three:
                (v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_size) = create_and_get_basic_pos(
                    sq_loop_idx, CORE_NUM, Z, H, pid, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N
                )
                if STAGE == 2:
                    v_cur_s2_size = TOPK

            if not v_not_last_three:
                v_cur_s2_size = 1
            s1_task_cnt += 1

            # mask ptr
            if STAGE == 1:
                attn_mask_ptr = tl.make_block_ptr(
                    base=ATTEN_MASK,
                    shape=(N_CTX, N_CTX),
                    strides=(stride_am, 1),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M // 2, BLOCK_N),
                    order=(1, 0),
                )
            else:
                attn_mask_ptr = None

            # =================== KV sequence length loop ===================
            for skv_loop_idx in range(0, v_cur_s2_size):
                # create and push task to producer stack
                if v_not_last_three:
                    (
                        v_b_idx_1,
                        v_n_idx_1,
                        v_s1_idx_1,
                        v_s2_idx_1,
                        v_s2_size_1,
                        v_b_idx_2,
                        v_n_idx_2,
                        v_s1_idx_2,
                        v_s2_idx_2,
                        v_s2_size_2,
                        v_b_idx_3,
                        v_n_idx_3,
                        v_s1_idx_3,
                        v_s2_idx_3,
                        v_s2_size_3,
                        v_b_idx_4,
                        v_n_idx_4,
                        v_s1_idx_4,
                        v_s2_idx_4,
                        v_s2_size_4,
                    ) = create_task(
                        vtaskId,
                        v_cur_b_idx,
                        v_cur_n_idx,
                        v_cur_s1_idx,
                        skv_loop_idx,
                        v_cur_s2_size,
                        v_b_idx_1,
                        v_n_idx_1,
                        v_s1_idx_1,
                        v_s2_idx_1,
                        v_s2_size_1,
                        v_b_idx_2,
                        v_n_idx_2,
                        v_s1_idx_2,
                        v_s2_idx_2,
                        v_s2_size_2,
                        v_b_idx_3,
                        v_n_idx_3,
                        v_s1_idx_3,
                        v_s2_idx_3,
                        v_s2_size_3,
                        v_b_idx_4,
                        v_n_idx_4,
                        v_s1_idx_4,
                        v_s2_idx_4,
                        v_s2_size_4,
                    )
                    v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = update_task(
                        vtaskId, s1_task_cnt, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4
                    )

                # get v1 task
                v1_use_b_idx, v1_use_n_idx, v1_use_s1_idx, v1_use_s2_idx, v1_use_s2_size = get_cur_task(
                    (vtaskId - 1) & 3,
                    v_b_idx_1,
                    v_n_idx_1,
                    v_s1_idx_1,
                    v_s2_idx_1,
                    v_s2_size_1,
                    v_b_idx_2,
                    v_n_idx_2,
                    v_s1_idx_2,
                    v_s2_idx_2,
                    v_s2_size_2,
                    v_b_idx_3,
                    v_n_idx_3,
                    v_s1_idx_3,
                    v_s2_idx_3,
                    v_s2_size_3,
                    v_b_idx_4,
                    v_n_idx_4,
                    v_s1_idx_4,
                    v_s2_idx_4,
                    v_s2_size_4,
                )
                need_do_v1 = vtaskId > 0 and v_not_last_two
                if need_do_v1:
                    v1_last_skv = v1_use_s2_idx == v1_use_s2_size - 1
                    v1_need_update = v1_use_s2_idx != 0
                    v1_s1_task_mod3 = get_s_task(
                        vtaskId - 1, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4
                    )
                    process_v1(
                        qk_ub_ping,
                        qk_ub_pong,
                        p_l1_ping,
                        p_l1_pong,
                        attn_mask_ptr,
                        m_i_tb0,
                        m_i_tb1,
                        m_i_tb2,
                        l_i_tb0,
                        l_i_tb1,
                        l_i_tb2,
                        alpha_tb0,
                        alpha_tb1,
                        alpha_tb2,
                        sm_scale,
                        vtaskId,
                        v1_s1_task_mod3,
                        Q.dtype.element_ty,
                        v1_last_skv,
                        v1_need_update,
                        BLOCK_M,
                        BLOCK_N,
                        STAGE,
                    )

                # get v2 task
                v2_use_b_idx, v2_use_n_idx, v2_use_s1_idx, v2_use_s2_idx, v2_use_s2_size = get_cur_task(
                    (vtaskId + 1) & 3,
                    v_b_idx_1,
                    v_n_idx_1,
                    v_s1_idx_1,
                    v_s2_idx_1,
                    v_s2_size_1,
                    v_b_idx_2,
                    v_n_idx_2,
                    v_s1_idx_2,
                    v_s2_idx_2,
                    v_s2_size_2,
                    v_b_idx_3,
                    v_n_idx_3,
                    v_s1_idx_3,
                    v_s2_idx_3,
                    v_s2_size_3,
                    v_b_idx_4,
                    v_n_idx_4,
                    v_s1_idx_4,
                    v_s2_idx_4,
                    v_s2_size_4,
                )
                if vtaskId > 2:
                    update_acc = v2_use_s2_idx != 0
                    v2_s1_task_mod3 = get_s_task(
                        vtaskId - 3, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4
                    )
                    _flash_update(
                        pv_ub_ping,
                        pv_ub_pong,
                        alpha_tb0,
                        alpha_tb1,
                        alpha_tb2,
                        acc_buffer,
                        v2_s1_task_mod3,
                        BLOCK_M,
                        HEAD_DIM,
                        vtaskId,
                        update_acc,
                    )

                v2_last_skv_v2 = v2_use_s2_idx == v2_use_s2_size - 1
                # after sq row done, do flash softmax div sum
                if v2_last_skv_v2:
                    v2_s1_task_mod3 = get_s_task(
                        vtaskId - 3, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4
                    )
                    if v2_s1_task_mod3 == 0:
                        l_i = bl.to_tensor(l_i_tb0)
                        m_i = bl.to_tensor(m_i_tb0)
                    elif v2_s1_task_mod3 == 1:
                        l_i = bl.to_tensor(l_i_tb1)
                        m_i = bl.to_tensor(m_i_tb1)
                    else:
                        l_i = bl.to_tensor(l_i_tb2)
                        m_i = bl.to_tensor(m_i_tb2)
                    m_i += tl.math.log(l_i)
                    acc = bl.to_tensor(acc_buffer)
                    acc = acc / l_i[:, None]

                    sub_vec_id = al.sub_vec_id()
                    out_offset = get_s_offset(
                        (vtaskId + 1) & 3, H, N_CTX, BLOCK_M, v2_use_b_idx, v2_use_n_idx, v2_use_s1_idx
                    ) + sub_vec_id * (BLOCK_M // 2)

                    m_ptrs = M + out_offset + tl.arange(0, BLOCK_M // 2)
                    tl.store(m_ptrs, m_i)

                    o_rs = out_offset + tl.arange(0, BLOCK_M // 2)[:, None]  # [BM, 1]
                    o_cs = tl.arange(0, HEAD_DIM)[None, :]
                    o_ptrs = Out + o_rs * stride_om + o_cs * stride_on

                    tl.store(o_ptrs, acc.to(Out.type.element_ty))

                vtaskId += 1

        vec_postwait_p_l1()


@triton.jit
def _attn_fwd_A2(
    Q,
    K,
    V,
    qk_scale,
    topk: tl.constexpr,
    LUT,
    LSE,
    OS,
    Z: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    num_cores: tl.constexpr,
):
    M_factor = BLOCK_M2 // BLOCK_M
    NUM_BLOCKS_M = triton.cdiv(L, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        task_m_idx2 = task_m_idx // M_factor

        off_z = task_hz_idx // H
        off_h = task_hz_idx % H

        stride_qz = H * L * D
        stride_qh = L * D
        qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        lut_offset = (
            off_z.to(tl.int64) * H * M_BLOCKS * topk
            + off_h.to(tl.int64) * M_BLOCKS * topk
            + task_m_idx2.to(tl.int64) * topk
        )
        lse_offset = off_z.to(tl.int64) * H * L + off_h.to(tl.int64) * L

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)

        Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
        K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
        V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
        OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
        LUT_ptr = LUT + lut_offset
        LSE_ptrs = LSE + lse_offset + offs_m

        m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
        for m_block_idx in tl.range(topk):
            idx_n = tl.load(LUT_ptr + m_block_idx)
            n_mask = offs_n < L - idx_n * BLOCK_N

            k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])

            qk = tl.dot(q, tl.trans(k)) * (qk_scale * 1.4426950408889634)

            if L - idx_n * BLOCK_N < BLOCK_N:
                qk = tl.where(n_mask[None, :], qk, float("-inf"))

            v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(m_i, local_m)
            qk = qk - new_m[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - new_m)
            o_s = o_s * alpha[:, None]
            o_s += tl.dot(p.to(v.dtype), v)

            l_i = l_i * alpha + l_ij
            m_i = new_m

        o_s = o_s / l_i[:, None]
        tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)

        m_i += tl.math.log2(l_i)
        tl.store(LSE_ptrs, m_i, mask=offs_m < L)


@triton.jit
def _attn_bwd_preprocess(
    o_s,
    do_s,
    delta_s,
    BHL: tl.constexpr,
    D: tl.constexpr,
):
    """preprocess of attention backward

    Args:
        grid (Tuple[int]): nproc grids

        o_s (Tensor(B, H, L, D)): ptr to data 1
        do_s (Tensor(B, H, L, D)): ptr to data 2
        delta_s (Tensor(B, H, L)): ptr to target

    TODO:
        - is o_s and do_s contigous
    """
    pid = tl.program_id(0)
    nproc = tl.num_programs(0)

    BLOCK_M: tl.constexpr = 128

    idx_start, idx_step = pid * BLOCK_M, nproc * BLOCK_M
    for idx in tl.range(idx_start, BHL, idx_step):
        range_input = (
            idx * D
            + tl.arange(0, BLOCK_M)[:, None] * D  # row
            + tl.arange(0, D)[None, :]  # col
        )
        mask_input = (idx + tl.arange(0, BLOCK_M))[:, None] < BHL
        range_output = idx + tl.arange(0, BLOCK_M)
        mask_output = (idx + tl.arange(0, BLOCK_M)) < BHL

        in_1 = tl.load(o_s + range_input, mask=mask_input)
        in_2 = tl.load(do_s + range_input, mask=mask_input)
        out = tl.sum(in_1 * in_2, axis=1).to(delta_s.type.element_ty)

        tl.store(delta_s + range_output, out, mask=mask_output)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    LSE,
    DELTAS,
    DOS,
    DQ,
    LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sub_BN: tl.constexpr,  # BLOCK_N//2
):
    NUM_BLOCKS_M = M_BLOCKS
    NUM_BLOCKS = M_BLOCKS * B * H
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        qkv_offset = task_hz_idx.to(tl.int64) * L * D
        lse_offset = task_hz_idx.to(tl.int64) * L
        lut_offset = (task_hz_idx.to(tl.int64) * NUM_BLOCKS_M + task_m_idx.to(tl.int64)) * topk

        K_base = K + qkv_offset
        V_base = V + qkv_offset
        Q_base = Q + qkv_offset
        DOS_base = DOS + qkv_offset
        DQ_base = DQ + qkv_offset

        LSE_base = LSE + lse_offset
        DELTA_base = DELTAS + lse_offset
        LUT_ptr = LUT + lut_offset

        Q_ptrs = Q_base + offs_m[:, None] * D + offs_d[None, :]
        DOS_ptrs = DOS_base + offs_m[:, None] * D + offs_d[None, :]
        DQ_ptrs = DQ_base + offs_m[:, None] * D + offs_d[None, :]

        LSE_ptrs = LSE_base + offs_m
        DELTA_ptrs = DELTA_base + offs_m

        q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
        do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L)
        lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))
        delta_s = tl.load(DELTA_ptrs, mask=offs_m < L)

        dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for block_idx_topk in tl.range(topk, num_stages=2):
            idx_n = tl.load(LUT_ptr + block_idx_topk)

            # sub_block
            for n_part in tl.static_range(0, BLOCK_N, sub_BN):
                offs_n = n_part + tl.arange(0, sub_BN)
                n_mask = offs_n < (L - idx_n * BLOCK_N)

                K_ptrs2 = K_base + (idx_n * BLOCK_N + offs_n)[:, None] * D + offs_d[None, :]
                V_ptrs2 = V_base + (idx_n * BLOCK_N + offs_n)[:, None] * D + offs_d[None, :]

                k = tl.load(K_ptrs2, mask=n_mask[:, None])
                v = tl.load(V_ptrs2, mask=n_mask[:, None])

                qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)
                p = tl.math.exp2(qk - lse[:, None])

                p = tl.where(n_mask[None, :], p, 0.0)

                dp = tl.dot(do_s, v.T).to(tl.float32)
                ds = p * (dp - delta_s[:, None])

                dq += tl.dot(ds.to(k.dtype), k)

        tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)


@triton.jit
def _attn_bwd_dkdv_inner(
    Q_ptrs,
    k,
    v,
    DOS_ptrs,
    LSE_ptrs,
    DELTAS_ptrs,
    KBID_ptrs,
    DK_ptrs,
    DV_ptrs,
    qk_scale,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_BLOCKS: tl.constexpr,
):
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    for idx_m in tl.range(0, L, BLOCK_M):
        kbid = tl.load(KBID_ptrs)

        if kbid == 1:
            q = tl.load(Q_ptrs, boundary_check=(0,))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)
            lse = tl.load(LSE_ptrs, boundary_check=(0,))
            pT = tl.math.exp2(qkT - lse[None, :])

            do = tl.load(DOS_ptrs, boundary_check=(0,))
            dv += tl.dot(pT.to(do.dtype), do) + 1e-14

            dpT = tl.dot(v, tl.trans(do))
            delta = tl.load(DELTAS_ptrs, boundary_check=(0,))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q) + 1e-14

        Q_ptrs = tl.advance(Q_ptrs, (BLOCK_M, 0))
        DOS_ptrs = tl.advance(DOS_ptrs, (BLOCK_M, 0))
        LSE_ptrs = tl.advance(LSE_ptrs, (BLOCK_M,))
        DELTAS_ptrs = tl.advance(DELTAS_ptrs, (BLOCK_M,))
        KBID_ptrs += N_BLOCKS

    tl.store(DK_ptrs, (dk * qk_scale).to(DK_ptrs.dtype.element_ty), boundary_check=(0,))
    tl.store(DV_ptrs, dv.to(DV_ptrs.dtype.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_dkdv(
    Q,
    K,
    V,
    DOS,
    DK,
    DV,
    KBID,
    LSE,
    DELTAS,
    qk_scale,
    Z: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    num_cores: tl.constexpr,
):
    N_factor = BLOCK_N2 // BLOCK_N
    NUM_BLOCKS_N = triton.cdiv(L, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_N * Z * H

    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_hz_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        task_n_idx2 = task_n_idx // N_factor

        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        stride_qz = H * L * D
        stride_qh = L * D
        stride_kbz = H * M_BLOCKS * N_BLOCKS
        stride_kbh = M_BLOCKS * N_BLOCKS
        stride_lsz = H * L
        stride_lsh = L
        # offset
        qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        kbid_offset = off_z.to(tl.int64) * stride_kbz + off_h.to(tl.int64) * stride_kbh
        lse_offset = off_z.to(tl.int64) * stride_lsz + off_h.to(tl.int64) * stride_lsh
        # ptr
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qkv_offset, shape=(L, D), strides=(D, 1), offsets=(0, 0), block_shape=(BLOCK_M, D), order=(1, 0)
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qkv_offset,
            shape=(L, D),
            strides=(D, 1),
            offsets=(task_n_idx * BLOCK_N, 0),
            block_shape=(BLOCK_N, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qkv_offset,
            shape=(L, D),
            strides=(D, 1),
            offsets=(task_n_idx * BLOCK_N, 0),
            block_shape=(BLOCK_N, D),
            order=(1, 0),
        )
        DOS_block_ptr = tl.make_block_ptr(
            base=DOS + qkv_offset, shape=(L, D), strides=(D, 1), offsets=(0, 0), block_shape=(BLOCK_M, D), order=(1, 0)
        )
        DK_block_ptr = tl.make_block_ptr(
            base=DK + qkv_offset,
            shape=(L, D),
            strides=(D, 1),
            offsets=(task_n_idx * BLOCK_N, 0),
            block_shape=(BLOCK_N, D),
            order=(1, 0),
        )
        DV_block_ptr = tl.make_block_ptr(
            base=DV + qkv_offset,
            shape=(L, D),
            strides=(D, 1),
            offsets=(task_n_idx * BLOCK_N, 0),
            block_shape=(BLOCK_N, D),
            order=(1, 0),
        )

        LSE_block_ptr = tl.make_block_ptr(
            base=LSE + lse_offset, shape=(L,), strides=(1,), offsets=(0,), block_shape=(BLOCK_M,), order=(0,)
        )
        DELTAS_block_ptr = tl.make_block_ptr(
            base=DELTAS + lse_offset, shape=(L,), strides=(1,), offsets=(0,), block_shape=(BLOCK_M,), order=(0,)
        )
        KBID_ptr = KBID + kbid_offset + task_n_idx2

        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,))
        _attn_bwd_dkdv_inner(
            Q_ptrs=Q_block_ptr,
            k=k,
            v=v,
            DOS_ptrs=DOS_block_ptr,
            LSE_ptrs=LSE_block_ptr,
            DELTAS_ptrs=DELTAS_block_ptr,
            KBID_ptrs=KBID_ptr,
            DK_ptrs=DK_block_ptr,
            DV_ptrs=DV_block_ptr,
            qk_scale=qk_scale,
            L=L,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            N_BLOCKS=N_BLOCKS,
        )


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sparse_map, lut, topk, atten_mask, causal, sm_scale, BM, BN):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        extra_kern_args = {}

        num_cores = get_npu_aicore_num()
        NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
        NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]
        NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)
        grid = min(num_cores, NUM_BLOCKS)
        o = torch.zeros_like(q)

        sparse_start_idx = None

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if num_cores >= 28:
            _attn_fwd[(grid,)](
                q,
                k,
                v,
                atten_mask,
                M,
                o,
                sparse_start_idx,
                sm_scale,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[2],
                q.shape[0],
                q.shape[1],
                N_CTX=q.shape[2],
                HEAD_DIM=HEAD_DIM_K,
                BLOCK_M=BM,
                BLOCK_N=BN,
                STAGE=2,
                NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
                NUM_BLOCKS=NUM_BLOCKS,
                NUM_BLOCKS_M=NUM_BLOCKS_M,
                CORE_NUM=num_cores,
                TOPK=topk,
                LUT=lut,
                multibuffer=True,
                sync_solver=True,
                disable_auto_inject_block_sync=True,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
                **extra_kern_args,
            )
        else:
            _attn_fwd_A2[(grid,)](
                Q=q,
                K=k,
                V=v,
                qk_scale=sm_scale,
                topk=topk,
                LUT=lut,
                LSE=M,
                OS=o,
                Z=q.shape[0],
                H=q.shape[1],
                L=q.shape[2],
                D=q.shape[3],
                BLOCK_M=32 if BM >= 128 else BM,
                BLOCK_M2=BM,
                BLOCK_N=BN,
                M_BLOCKS=triton.cdiv(q.shape[2], BM),
                num_cores=num_cores,
            )

        ctx.save_for_backward(q, k, v, sparse_map, lut, M, o)
        ctx.sm_scale = sm_scale
        ctx.topk = topk
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.BLOCK_M = BM
        ctx.BLOCK_N = BN
        return o, M

    @staticmethod
    def backward(ctx, do_s):
        num_cube_cores = get_npu_aicore_num()
        num_vec_cores = num_cube_cores * 2
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, D = q.shape

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        BHL = B * H * L
        grid = (num_vec_cores,)
        _attn_bwd_preprocess[grid](o_s, do_s, delta_s, BHL, D)

        grid = (num_cube_cores,)
        _attn_bwd_dq[grid](
            q,
            k,
            v,
            lse,
            delta_s,
            do_s,
            dq,
            lut,
            ctx.sm_scale,
            ctx.topk,
            B,
            H,
            L,
            M_BLOCKS,
            D,
            BLOCK_M,
            BLOCK_N,
            sub_BN=BLOCK_N // 2,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5,
        )

        grid = (num_cube_cores,)
        _attn_bwd_dkdv[grid](
            Q=q,
            K=k,
            V=v,
            DOS=do_s,
            DK=dk,
            DV=dv,
            KBID=k_block_id,
            LSE=lse,
            DELTAS=delta_s,
            qk_scale=ctx.sm_scale,
            Z=q.shape[0],
            H=q.shape[1],
            L=q.shape[2],
            D=q.shape[3],
            BLOCK_M=BLOCK_M,
            BLOCK_N2=BLOCK_N,
            BLOCK_N=32,
            M_BLOCKS=M_BLOCKS,
            N_BLOCKS=N_BLOCKS,
            num_cores=num_cube_cores,
        )

        return dq, dk, dv, None, None, None, None, None, None
