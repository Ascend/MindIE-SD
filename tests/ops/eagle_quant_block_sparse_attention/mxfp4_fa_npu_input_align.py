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

"""MXFP4 packing and attention reference used by the FP4 golden test."""

import math

import torch
import torch.nn.functional as F
import torch_npu

# ==========================================
# 1. 基础量化工具定义
# ==========================================

_FP4_E2M1_POSITIVES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def pseudo_quant_fp4_e2m1(x: torch.Tensor, symmetric=True) -> torch.Tensor:
    """将任意形状的浮点 tensor 伪量化到 FP4 e2m1 格网。

    实现方式：对每个元素找到绝对值最近的可表示正数，再还原符号。
    返回与输入同形状、同 dtype 的 tensor（量化后的浮点值）。
    """
    fp4_vals = _FP4_E2M1_POSITIVES.to(device=x.device, dtype=x.dtype)
    sign = torch.sign(x)
    abs_x = x.abs()
    # [*shape, 8] 广播比较，找最近邻
    diff = (abs_x.unsqueeze(-1) - fp4_vals).abs()
    # 翻转最后一个维度，使得 argmin 遇到并列最小值时优先返回较大的索引（实现 Round-Half-Up）
    idx = 7 - diff.flip(dims=[-1]).argmin(dim=-1)
    quantized = fp4_vals[idx]
    # 0 的符号恢复为 +0（与硬件行为一致）
    return quantized * sign.where(sign != 0, torch.ones_like(sign))


def get_hadamard_matrix(n):
    """生成基础的 Hadamard 矩阵"""
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def get_block_diagonal_hadamard_matrix(d, block_size=32):
    """生成块对角 Hadamard 矩阵"""
    assert d % block_size == 0, "维度 d 必须能被 block_size 整除"
    H_block = get_hadamard_matrix(block_size)
    num_blocks = d // block_size
    return torch.block_diag(*([H_block] * num_blocks))


def quantize_mx_fp4_dim_d(x, block_size=32):
    """对 Q, K 矩阵沿 Head 维度(d) 进行 MXFP4 量化"""
    d = x.shape[-1]
    x_blocks = x.view(*x.shape[:-1], d // block_size, block_size)
    max_val = torch.max(torch.abs(x_blocks), dim=-1, keepdim=True)[0]

    scale = torch.pow(2.0, torch.ceil(torch.log2(torch.clamp(max_val / 6.0, min=1e-12))))
    x_scaled = x_blocks / scale
    x_fp4 = pseudo_quant_fp4_e2m1(x_scaled, symmetric=True)
    return (x_fp4 * scale).view_as(x)


def _quantize_and_pack_blocks(x_blocks: torch.Tensor, max_val: torch.Tensor):
    """对已被划分为块的 Tensor 和其 max_val 计算 E8M0 scale 并做 FP4e2m1 原地打包"""
    orig_device = x_blocks.device
    x_blocks = x_blocks.cpu()
    max_val = max_val.cpu()

    exp = torch.ceil(torch.log2(torch.clamp(max_val / 6.0, min=1e-12)))
    # OCP E8M0 scale with bias 127 (0 is reserved for NaN)
    scale_uint8 = (exp + 127).clamp(1, 255).to(torch.uint8)

    # Decode for actual quantization computation
    scale_float = torch.pow(2.0, scale_uint8.float() - 127)
    x_scaled = x_blocks / scale_float

    fp4_vals = _FP4_E2M1_POSITIVES.to(device=x_scaled.device, dtype=x_scaled.dtype)
    sign = torch.sign(x_scaled)
    abs_x = x_scaled.abs()
    diff = (abs_x.unsqueeze(-1) - fp4_vals).abs()
    idx = 7 - diff.flip(dims=[-1]).argmin(dim=-1)

    s_bit = torch.where(sign < 0, 8, 0).to(torch.uint8)
    fp4_idx = s_bit | idx.to(torch.uint8)

    fp4_idx_even = fp4_idx[..., 0::2]
    fp4_idx_odd = fp4_idx[..., 1::2]

    packed = (fp4_idx_even & 0x0F) | ((fp4_idx_odd & 0x0F) << 4)
    return packed.to(orig_device), scale_uint8.to(orig_device)


def quantize_mx_fp4_dim_d_packed(x, block_size=32):
    """对 Q, K 矩阵沿 Head 维度(d) 进行 MXFP4 量化，返回 packed uint8 和 scale_uint8"""
    d = x.shape[-1]
    x_blocks = x.view(*x.shape[:-1], d // block_size, block_size)
    max_val = torch.max(torch.abs(x_blocks), dim=-1, keepdim=True)[0]

    packed, scale_uint8 = _quantize_and_pack_blocks(x_blocks, max_val)

    packed = packed.view(*x.shape[:-1], d // 2)
    scale_uint8 = scale_uint8.view(*x.shape[:-1], d // block_size)

    return packed, scale_uint8


def quantize_mx_fp4_dim_seq_packed(V, block_size=32):
    """对 V 矩阵沿 Seq 维度进行 MXFP4 量化，返回 packed uint8, scale_uint8 和 V_max"""
    *batch_dims, seq_len_k, d = V.shape

    V_max = torch.max(torch.abs(V), dim=-2, keepdim=True)[0]
    V_norm = V / torch.clamp(V_max, min=1e-12)

    pad_len_v = (block_size - seq_len_k % block_size) % block_size
    if pad_len_v > 0:
        V_norm_padded = F.pad(V_norm, (0, 0, 0, pad_len_v), value=0.0)
    else:
        V_norm_padded = V_norm

    V_norm_blocks = V_norm_padded.view(*batch_dims, -1, block_size, d)
    V_block_max = torch.max(torch.abs(V_norm_blocks), dim=-2, keepdim=True)[0]

    packed, V_scale_uint8 = _quantize_and_pack_blocks(V_norm_blocks, V_block_max)

    packed_padded = packed.view(*batch_dims, -1, d // 2)
    V_q_uint8 = packed_padded[..., :seq_len_k, :]
    V_scale_uint8 = V_scale_uint8.squeeze(-2)

    return V_q_uint8, V_scale_uint8, V_max


def unpack_uint8_to_fp4e2m1_fp32(uint8_tensor: torch.Tensor, scale_uint8: torch.Tensor, block_size=32) -> torch.Tensor:
    """
    将包含打包FP4数据（两个FP4e2m1共用一个uint8）的tensor解码为FP32 tensor，并乘上scale。
    OCP standard scale uses bias 127.
    """
    orig_device = uint8_tensor.device
    uint8_tensor = uint8_tensor.cpu()
    scale_uint8 = scale_uint8.cpu()

    low_nibbles = (uint8_tensor & 0x0F).long()
    high_nibbles = ((uint8_tensor >> 4) & 0x0F).long()

    fp4_e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=uint8_tensor.device,
    )

    fp32_low = fp4_e2m1_lut[low_nibbles]
    fp32_high = fp4_e2m1_lut[high_nibbles]

    out_shape = list(uint8_tensor.shape)
    out_shape[-1] *= 2

    decoded = torch.stack((fp32_low, fp32_high), dim=-1).view(out_shape)

    scale_float = torch.pow(2.0, scale_uint8.float() - 127)
    scale_expanded = scale_float.unsqueeze(-1).expand(*scale_float.shape, block_size).contiguous().view(out_shape)

    return (decoded * scale_expanded).to(orig_device)


def unpack_uint8_to_fp4e2m1_fp32_seq(
    uint8_tensor: torch.Tensor, scale_uint8: torch.Tensor, block_size=32
) -> torch.Tensor:
    """
    解码按 Seq 维度量化的 V 矩阵。
    uint8_tensor: (*batch_dims, seq_len, d // 2)
    scale_uint8: (*batch_dims, pad_seq_len // block_size, d)
    """
    orig_device = uint8_tensor.device
    uint8_tensor = uint8_tensor.cpu()
    scale_uint8 = scale_uint8.cpu()

    low_nibbles = (uint8_tensor & 0x0F).long()
    high_nibbles = ((uint8_tensor >> 4) & 0x0F).long()

    fp4_e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=uint8_tensor.device,
    )

    fp32_low = fp4_e2m1_lut[low_nibbles]
    fp32_high = fp4_e2m1_lut[high_nibbles]

    out_shape = list(uint8_tensor.shape)
    out_shape[-1] *= 2

    decoded = torch.stack((fp32_low, fp32_high), dim=-1).view(out_shape)

    scale_float = torch.pow(2.0, scale_uint8.float() - 127)
    batch_dims = scale_float.shape[:-2]
    num_blocks = scale_float.shape[-2]
    d = scale_float.shape[-1]

    scale_expanded = scale_float.unsqueeze(-2).expand(*batch_dims, num_blocks, block_size, d).contiguous()
    scale_expanded = scale_expanded.view(*batch_dims, num_blocks * block_size, d)

    seq_len = out_shape[-2]
    return (decoded * scale_expanded[..., :seq_len, :]).to(orig_device)


# ==========================================
# 2. QKV 外部预量化逻辑
# ==========================================


def prepare_quantized_qkv(Q, K, V):
    """在输入 Flash Attention 之前完成所有硬件感知量化，并处理 V 的尾块 padding"""

    mxblock = 32

    # 1. Q, K 矩阵：块对角 Hadamard 旋转 + 沿 d 维度 MXFP4 量化
    # H_blk = get_block_diagonal_hadamard_matrix(d, block_size=32).to(Q.device)
    # Q_q = quantize_mx_fp4_dim_d(Q @ H_blk, block_size=mxblock)
    # K_q = quantize_mx_fp4_dim_d(K @ H_blk, block_size=mxblock)

    Q_q_uint8, Q_scale = quantize_mx_fp4_dim_d_packed(Q, block_size=mxblock)
    K_q_uint8, K_scale = quantize_mx_fp4_dim_d_packed(K, block_size=mxblock)

    # 2. V 矩阵: 按 channel 归一化后 沿 Seq 维度进行 MXFP4 量化
    V_q_uint8, V_scale_uint8, V_max = quantize_mx_fp4_dim_seq_packed(V, block_size=mxblock)

    return Q_q_uint8, Q_scale, K_q_uint8, K_scale, V_q_uint8, V_scale_uint8, V_max.float()


# ==========================================
# 3. Flash Attention 核心计算逻辑 (SRAM优化复用)
# ==========================================
# Packed Q/K/V inputs and scales may be prepared on CPU or NPU.


def simulate_mxfp4_flash_attention(
    Q_q_uint8,
    Q_scale,
    K_q_uint8,
    K_scale,
    V_q_uint8,
    V_scale,
    V_max,
    block_sparse_mask=None,
    always_update=False,
    useQMM=False,
):
    if Q_q_uint8.dim() == 4:
        B, N, S_q, _ = Q_q_uint8.shape
        d = Q_q_uint8.shape[-1] * 2
        N_kv = K_q_uint8.shape[1]
        O_out_all = torch.zeros((B, N, S_q, d), device=Q_q_uint8.device)
        for b in range(B):
            for n in range(N):
                n_kv = n * N_kv // N
                if block_sparse_mask is not None:
                    if block_sparse_mask.dim() == 4:
                        mask_bn = block_sparse_mask[b, n]
                    elif block_sparse_mask.dim() == 3:
                        mask_bn = block_sparse_mask[b]
                    else:
                        mask_bn = block_sparse_mask
                else:
                    mask_bn = None

                O_out_all[b, n] = simulate_mxfp4_flash_attention_sd(
                    Q_q_uint8[b, n],
                    Q_scale[b, n],
                    K_q_uint8[b, n_kv],
                    K_scale[b, n_kv],
                    V_q_uint8[b, n_kv],
                    V_scale[b, n_kv],
                    V_max[b, n_kv] if V_max.dim() >= 3 else V_max,
                    block_sparse_mask=mask_bn,
                    always_update=always_update,
                    useQMM=useQMM,
                )
        return O_out_all
    else:
        return simulate_mxfp4_flash_attention_sd(
            Q_q_uint8,
            Q_scale,
            K_q_uint8,
            K_scale,
            V_q_uint8,
            V_scale,
            V_max,
            block_sparse_mask=block_sparse_mask,
            always_update=always_update,
            useQMM=useQMM,
        )


def simulate_mxfp4_flash_attention_sd(
    Q_q_uint8,
    Q_scale,
    K_q_uint8,
    K_scale,
    V_q_uint8,
    V_scale,
    V_max,
    block_sparse_mask=None,
    always_update=False,
    useQMM=False,
):
    # 获取伪量化的结果
    Q_q = unpack_uint8_to_fp4e2m1_fp32(Q_q_uint8, Q_scale, block_size=32)
    K_q = unpack_uint8_to_fp4e2m1_fp32(K_q_uint8, K_scale, block_size=32)
    V_q = unpack_uint8_to_fp4e2m1_fp32_seq(V_q_uint8, V_scale, block_size=32)

    seq_len_q = Q_q.shape[0]
    seq_len_k, d = K_q.shape
    scaleValue = torch.tensor(1 / math.sqrt(d), dtype=torch.float32)
    bits = scaleValue.view(torch.int32)
    mask = torch.tensor((~((1 << 13) - 1)), dtype=torch.int32)
    scaleValue = (bits & mask).view(torch.float32).to(V_q.device)
    Br = 128  # Q block size
    Bc = 256  # K, V block size
    mxblock = 32
    LNC = 1.96
    LN2 = math.log(2.0)
    tLN2, tLNC = (
        torch.tensor(LN2, dtype=torch.float16, device=V_q.device),
        torch.tensor(LNC, dtype=torch.float16, device=V_q.device),
    )

    # 全局输出 HBM
    O_out = torch.zeros((seq_len_q, d), device=V_q.device)

    # 外层循环：遍历 Q (硬件上这使得 M, L, O 驻留在长度极小的 128 SRAM 中)
    for i in range(0, seq_len_q, Br):
        q_chunk_len = min(Br, seq_len_q - i)
        Q_i = Q_q[i : i + q_chunk_len]

        if useQMM:
            qblk_fp4 = Q_q_uint8[i : i + q_chunk_len].npu()
            qblk_scale_raw = Q_scale[i : i + q_chunk_len]
            qblk_scale = qblk_scale_raw.npu().reshape(qblk_scale_raw.shape[0], qblk_scale_raw.shape[1] // 2, 2)

        # 需求 1: M, L, O 只需要 128 长度，当前上下文中极致复用
        M_his = torch.full((q_chunk_len, 1), -60000, device=V_q.device).half()
        L_current = torch.zeros((q_chunk_len, 1), device=V_q.device)
        O_current = torch.zeros((q_chunk_len, d), device=V_q.device)  # O_UB
        O_L0c = torch.zeros((q_chunk_len, d + 8), device=V_q.device)

        K_q_uint8_valid = K_q_uint8
        K_scale_valid = K_scale

        # --- 新增: Block Sparse 动态 K Tile 拼装 ---
        if block_sparse_mask is not None:
            valid_seq_mask = block_sparse_mask[i // Br].repeat_interleave(128)[:seq_len_k]
            if not valid_seq_mask.any():
                O_out[i : i + q_chunk_len] = 0.0
                continue

            K_valid = K_q[valid_seq_mask]
            V_valid = V_q[valid_seq_mask]
            if useQMM:
                K_q_uint8_valid = K_q_uint8[valid_seq_mask]
                K_scale_valid = K_scale[valid_seq_mask]
        else:
            K_valid = K_q
            V_valid = V_q

        valid_seq_len_k = K_valid.shape[0]

        for j in range(0, valid_seq_len_k, Bc):
            k_chunk_len = min(Bc, valid_seq_len_k - j)
            K_j = K_valid[j : j + k_chunk_len]
            V_j = V_valid[j : j + k_chunk_len]

            # 需求 4 (延续上版): 拼接 1
            ones_col = torch.ones((k_chunk_len, 8), dtype=V_j.dtype, device=V_j.device)
            V_j_pad = torch.cat([V_j, ones_col], dim=-1)

            if useQMM:
                kblk_fp4_raw = K_q_uint8_valid[j : j + k_chunk_len]
                kblk_scale_raw = K_scale_valid[j : j + k_chunk_len]
                kblk_fp4 = kblk_fp4_raw.npu()
                kblk_scale = kblk_scale_raw.npu().reshape(kblk_scale_raw.shape[0], kblk_scale_raw.shape[1] // 2, 2)
                input_z_blk = torch.ops.npu.npu_quant_matmul(
                    qblk_fp4,
                    kblk_fp4.transpose(-1, -2),
                    kblk_scale.transpose(0, 1),
                    bias=None,
                    pertoken_scale=qblk_scale,
                    output_dtype=torch.float32,
                    x1_dtype=torch_npu.float4_e2m1fn_x2,
                    x2_dtype=torch_npu.float4_e2m1fn_x2,
                    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                    scale_dtype=torch_npu.float8_e8m0fnu,
                    group_sizes=[0, 0, 32],
                )
                S_ij = (input_z_blk * scaleValue).to(torch.float16).to(Q_i.device)
            else:
                S_ij = ((Q_i @ K_j.t()) * scaleValue).half()

            # 需求 2: 处理 Logits 计算时尾块不足 32 的 case (动态 padding)
            pad_k = (mxblock - k_chunk_len % mxblock) % mxblock
            if pad_k > 0:
                # S_ij 空白位补 -inf 确保 exp() 为 0; V_j 空白处补 0 不影响累加
                S_ij_padded = F.pad(S_ij, (0, pad_k), value=-float('inf'))
                V_j_pad_padded = F.pad(V_j_pad, (0, 0, 0, pad_k), value=0.0)
            else:
                S_ij_padded = S_ij
                V_j_pad_padded = V_j_pad

            S_ij_blocks = S_ij_padded.view(q_chunk_len, -1, mxblock)

            # 提取局部最大值并向 ln2 对齐
            m_local = torch.max(S_ij_blocks, dim=-1, keepdim=True)[0]
            inv_ln2_half = torch.tensor(1.4426950409, dtype=torch.float16, device=m_local.device)
            m_aligned = torch.ceil((m_local * inv_ln2_half).float())
            m_tile_max = torch.max(m_aligned, dim=1)[0]

            # ------------------------------------------
            # 需求 3: 延迟 Rescale & 整体 M 更新逻辑修正
            # 128行中任意一行满足 m - M >  100 时整体触发
            # ------------------------------------------
            updateO = (m_tile_max - M_his > 100.0).any()
            if always_update:
                updateO = True

            if updateO:
                M_new = torch.maximum(m_tile_max, M_his)

                # 去除第一次迭代(因M_his=-inf而导致)的无意义 Rescale 更新
                if j > 0:
                    scale_O = torch.pow(2.0, (M_his - M_new).float())
                    O_current = (O_current + O_L0c[:, :-8]) * scale_O
                    L_current = (L_current + O_L0c[:, -8 : -8 + 1]) * scale_O

                M_his = M_new  # 保存状态供 mx_scale 及下个Tile计算

            sub = m_aligned * tLN2 - tLNC
            P_base = torch.exp(S_ij_blocks - sub)  # 保持half 精度计算
            P_fp4 = pseudo_quant_fp4_e2m1(P_base).float()
            mx_scale = torch.pow(2.0, (m_aligned - M_his.unsqueeze(1)).float())

            P_eff = ((P_fp4) * mx_scale).view(q_chunk_len, -1)
            PxV = P_eff @ V_j_pad_padded.float()
            # PxV = ((PxV.view(torch.int32)) & 0xFFFFE000).view(torch.float32)
            O_L0c = PxV if updateO else O_L0c + PxV

        # move out last L0c data
        O_current += O_L0c[:, :-8]
        L_current += O_L0c[:, -8 : -8 + 1]
        O_current *= V_max
        O_out[i : i + q_chunk_len] = O_current / torch.clamp(L_current, min=1e-12)

    return O_out
