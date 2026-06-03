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

"""
Flash Attention Varlen CPU Golden (FP32 / MXFP4 统一)
=======================================================
Tiled FlashAttention v2 的 CPU golden, 仅 varlen (packed seq) 布局,
通过 `quantize` 开关切换 FP32 与 MXFP4 (Q/K/V + 可选 P) 两种模式:

    quantize=False  -> 全 FP32 (无量化), 用于校验 flash 算法骨架本身
    quantize=True   -> Q/K/V (可选 P) MXFP4 qdq, 用于推理量化建模

两种模式共享单一 kernel; MXFP4 只是在 kernel 之前 (Q/K/V) 与循环内
(P, 可选) 插入 qdq 步骤; 因此 ref vs flash 的算法一致性结论对两种
模式同样成立, 而 FP32 vs MXFP4 的差异即纯量化噪声。

两个入口:
  attention_cpu_golden_varlen        非分块 reference (flash 的对照)
  flash_attention_cpu_golden_varlen  Tiled flash v2

布局:
  q: [total_seq_q,  num_heads,    head_dim]
  k: [total_seq_kv, num_heads_kv, head_dim]
  v: [total_seq_kv, num_heads_kv, head_dim_v]
  cu_seqlens_q/kv: [batch+1] int, 存储边界. 例 [0,3,4,6,9] -> 存储 3/1/2/3
  seq_used_q/kv:   [batch] int, 有效长度 (左对齐, <= 存储长度); None = 满
  causal: K 可见于 Q 当且仅当 k <= q + (seq_used_kv - seq_used_q)
          (Q 对应 KV 末尾, 适配 decoding / prefix caching)
"""

import math
from typing import Optional

import torch

from mx_quant_fp4_tool import mx_quantize, mx_dequantize

# pylint: disable=too-many-lines,redefined-outer-name,not-callable,used-before-assignment


# ============================================================
# 计算 dtype 配置 (S/m/exp/P 的中间精度)
# ============================================================

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _resolve_s_dtype(dtype_str: str) -> torch.dtype:
    """把 s_dtype 字符串解析为 torch dtype, 非法值 raise ValueError."""
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"s_dtype must be one of {list(_DTYPE_MAP)}, got '{dtype_str}'")
    return _DTYPE_MAP[dtype_str]


# ============================================================
# MXFP4: 沿任意轴 qdq
# ============================================================


def _mxfp4_qdq_lastdim(t: torch.Tensor, block_size: int = 32, mode: str = "baseline") -> torch.Tensor:
    q, s = mx_quantize(t, block_size=block_size, mode=mode)
    return mx_dequantize(q, s, block_size=block_size)


def mxfp4_qdq(t: torch.Tensor, axis: int = -1, block_size: int = 32, mode: str = "baseline") -> torch.Tensor:
    """沿指定轴对张量做 MXFP4 quantize + dequantize."""
    ndim = t.dim()
    if axis < 0:
        axis += ndim
    if axis == ndim - 1:
        return _mxfp4_qdq_lastdim(t, block_size, mode)

    perm = list(range(ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    t_p = t.permute(perm).contiguous()
    dq_p = _mxfp4_qdq_lastdim(t_p, block_size, mode)

    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return dq_p.permute(inv_perm).contiguous()


# ============================================================
# FP4 E2M1 正向 grid (供 blockwise quantize_p 直接 round, 跳过 mxfp4_qdq)
# ============================================================
# Blockwise 方案 A 已知 P_local block 内 max = 1.0, MX scale 恒为 0.25,
# 所以 round 目标只是 P_local × 4 投到 FP4 正向 level 上.
# Levels = {0, 0.5, 1, 1.5, 2, 3, 4, 6}
# Midpoints (round 边界) = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}
# torch.bucketize 用 right=False 在 midpoint 平局时归入下方桶, 等价于
# round-half-down, 与 mx_quant_fp4_tool 的 argmin(|x - FP4_E2M1_VALUES|)
# 取首索引的行为一致.

_FP4_POS_LEVELS_FP32 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_POS_MIDPOINTS_FP32 = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


# ============================================================
# E2M1 / E8M0 binary dump (snap_local 路径专用; 路径硬编码, 不参数化)
# ============================================================
# kernel 在 snap_local 模式下把 P 的 FP4 raw data 与 per-block 反量化 scale
# 以 OCP MX 二进制格式写入下面两个文件:
#   _MXFP4_E2M1_DUMP_PATH - FP4 元素 nibble-packed (low nibble = element[2k],
#                           high nibble = element[2k+1], OCP MX spec).
#                           我们的 P 恒非负 (sign=0), idx ∈ [0, 7] 直接就是
#                           E2M1 的 4-bit 模式.
#   _MXFP4_E8M0_DUMP_PATH - 每 block 1 byte, 值 = (K_diff - 2) + 127 = K_diff + 125,
#                           clamp 到 [0, 254] (underflow -> 0, overflow -> 254).
# 内部布局: 每次 kernel 调用开始时 truncate 两文件; 每个 tile 把当前 P_eff
# 对应的 bytes append (seqk axis 在最后; head / q_tile / kv_tile 顺序串行).

_MXFP4_E2M1_DUMP_PATH = "mxfp4_p_e2m1.bin"
_MXFP4_E8M0_DUMP_PATH = "mxfp4_p_e8m0_scale.bin"


def _pack_fp4_e2m1_along_last(idx_tensor: torch.Tensor) -> torch.Tensor:
    """
    沿最后一轴把 FP4 element 索引 (0..7) 打包为 nibble-packed uint8.
    Per OCP MX: byte = (e[2k+1] << 4) | e[2k]. 奇数长度末尾 pad 0.
    sign 恒 0 (P >= 0), idx 直接就是 E2M1 4-bit 模式.
    """
    *batch, n = idx_tensor.shape
    if n % 2 == 1:
        pad = torch.zeros(*batch, 1, dtype=idx_tensor.dtype, device=idx_tensor.device)
        idx_tensor = torch.cat([idx_tensor, pad], dim=-1)
        n += 1
    pairs = idx_tensor.reshape(*batch, n // 2, 2)
    low = pairs[..., 0].to(torch.uint8) & 0xF
    high = pairs[..., 1].to(torch.uint8) & 0xF
    return (high << 4) | low


def _to_e8m0_uint8(exp_int: torch.Tensor) -> torch.Tensor:
    """
    整数指数 -> E8M0 (uint8, bias 127): byte = clamp(exp + 127, 0, 254).
    Underflow 饱和到 0 (= 2^-127), overflow 饱和到 254 (= 2^127).
    """
    biased = exp_int.to(torch.int64) + 127
    biased = torch.clamp(biased, min=0, max=254)
    return biased.to(torch.uint8)


def _append_dump_bytes(t: torch.Tensor, path: str) -> None:
    """把 uint8 张量 flatten 成 raw bytes 追加到文件 (不依赖 numpy)."""
    flat = t.contiguous().flatten().to(torch.uint8).tolist()
    with open(path, "ab") as f:
        f.write(bytes(flat))


def _truncate_dump_file(path: str) -> None:
    """清空文件 (kernel 调用开始时重置, 保证一次调用 = 一份完整 dump)."""
    with open(path, "wb"):
        pass


# ============================================================
# Blockwise quantize_p (方案 A) 核心计算
# ============================================================


def _blockwise_quantize_p_eff(
    S: torch.Tensor,
    m_running: Optional[torch.Tensor],
    seqk_dim: int,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
):
    """
    Blockwise quantize_p (方案 A): 沿 seqk_dim 每 mx_block_size 元素分 block
    计算 m_block, 用 m_block 在 block 内归一化 S, MX 量化 (scale 自动 = 0.25),
    再把 corr = 2^floor((m_block - m_new) / ln 2) 折叠为 E8M0 反量化系数.

    数学等价于把 standard global 量化 exp(S - m_new) 时的 MX scale 拆成
        scale = 2^(-2) × corr_floor,
    并把 FP4 dequant 因子 (2^-2) 和 corr_floor 一起折叠为单一 per-block E8M0 指数
    2^(delta - 2) (delta = floor((m_block - m_new) / ln 2)); 返回
        P_eff = FP4_raw_data × 2^(delta - 2)
    其中 FP4_raw_data = levels[idx] ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 6} (未乘 0.25).
    FP4 data 字段可能略不同于 global (差异 ≤ 1.5× quantum at block max element).

    参数:
        S:          任意维度 tensor, 在 s_dtype 中; seqk_dim 标识 seq_k 轴
        m_running:  FP32, 形状 = S 除去 seqk 轴, 之前 tile 的 running max
                    None 表示无历史 (用于非分块 ref, 等价 -inf)
        seqk_dim:   seq_k 轴的索引 (ND=-1 / DN=-2 在 3D ref; ND=-1 / DN=0 在 2D kernel)
        mx_block_size, mx_mode: 透传到 mxfp4_qdq

    返回:
        (P_eff, m_new)
            P_eff: 与 S 同 shape (s_dtype), 反量化后的有效 P (= P_q × corr_floor 广播)
            m_new: FP32, 形状 = S 除去 seqk 轴, 更新后的全局 max (供外部 alpha = exp(m_old - m_new) 使用)

    精度: m_running / m_new 在 FP32 (供外部 alpha 计算); 与 S 相关的算术
    (m_block, corr) 在 S 的 dtype 中进行, m_new 进入 corr 计算前 cast 回 S.dtype.
    """
    NEG_INF = float('-inf')
    device = S.device
    src_dtype = S.dtype
    ndim = S.dim()
    seqk_dim_pos = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[seqk_dim_pos]

    # 阶段 1: pad 到 mx_block_size 倍数 (-inf padding 不影响 max)
    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pad_shape = list(S.shape)
        pad_shape[seqk_dim_pos] = pad
        pad_tensor = torch.full(pad_shape, NEG_INF, dtype=src_dtype, device=device)
        S_padded = torch.cat([S, pad_tensor], dim=seqk_dim_pos)
    else:
        S_padded = S

    # 把 seqk 移到最后 -> reshape 为 [..., n_blk, mx_block_size]
    S_moved = S_padded.movedim(seqk_dim_pos, -1)
    other_shape = S_moved.shape[:-1]
    n_blk = S_moved.shape[-1] // mx_block_size
    S_reshape = S_moved.reshape(*other_shape, n_blk, mx_block_size)

    # 阶段 2: 每 block 内的 max, 跨 block 取 max -> 等价 S.max(seqk)
    m_block = S_reshape.max(dim=-1).values  # [..., n_blk], s_dtype
    m_block_safe = torch.where(
        torch.isinf(m_block) & (m_block < 0),
        torch.zeros_like(m_block),
        m_block,
    )
    m_ij = m_block.max(dim=-1).values  # [...], s_dtype

    # 阶段 3: 跨 tile running max (FP32). m_ij 从 s_dtype 提升到 FP32 再合并
    m_ij_fp32 = m_ij.float()
    if m_running is None:
        m_new = m_ij_fp32
    else:
        m_new = torch.maximum(m_running, m_ij_fp32)  # FP32
    m_safe_new_fp32 = torch.where(
        torch.isinf(m_new) & (m_new < 0),
        torch.zeros_like(m_new),
        m_new,
    )
    # ★ corr 与 m_block_safe (s_dtype) 同 dtype 计算, m_new 在此处 cast 回 src_dtype
    m_safe_new_s = m_safe_new_fp32.to(src_dtype)

    # 阶段 4: 局部归一化, P_local 每 block 内 max = 1.0
    P_local_r = torch.exp(S_reshape - m_block_safe.unsqueeze(-1))  # [..., n_blk, 32]
    # flatten 回 seqk 长度并 strip pad, 然后 move 回原 seqk_dim
    P_local_padded_moved = P_local_r.reshape(*other_shape, n_blk * mx_block_size)
    P_local_moved = P_local_padded_moved[..., :seqk_len]
    P_local = P_local_moved.movedim(-1, seqk_dim_pos)

    # 阶段 5: 显式 FP4 round (MX scale 恒 = 0.25 = 2^-2; dequant 因子折入 corr)
    # P_q = FP4 raw data = levels[idx], 未乘 0.25
    # 全程在 src_dtype, 不需要 FP32 round-trip
    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    P_local_x4 = P_local * 4.0
    idx = torch.bucketize(P_local_x4, midpoints, right=False)  # round-half-down
    P_q = levels[idx]  # FP4 raw data, 未乘 0.25

    # 阶段 6: corr = 0.25 × 2^floor((m_block - m_new) / ln 2) = 2^(delta - 2)
    # FP4 dequant 因子 (2^-2) 与 floor corr 合并为单一整数指数的 per-block E8M0 scale
    r = m_block_safe - m_safe_new_s.unsqueeze(-1)  # [..., n_blk], ≤ 0, s_dtype
    delta = torch.floor(r / math.log(2.0))
    delta = torch.clamp(delta, min=-127.0, max=127.0)
    # corr 显式保持 src_dtype (避免 torch.pow 在某些路径上升精度)
    corr = (2.0 ** (delta - 2.0)).to(src_dtype)  # [..., n_blk]

    # 阶段 7: corr 广播回 seqk_len, 并与 P_q 相乘得 P_eff
    corr_broad_moved = corr.repeat_interleave(mx_block_size, dim=-1)
    corr_broad_moved = corr_broad_moved[..., :seqk_len]
    corr_broad = corr_broad_moved.movedim(-1, seqk_dim_pos)
    P_eff = P_q * corr_broad

    return P_eff, m_new


# ============================================================
# Blockwise quantize_p (方案 B-snap-local) 核心计算
# ============================================================


def _blockwise_snap_local_quantize_p_eff(
    S: torch.Tensor,
    m_running: Optional[torch.Tensor],
    seqk_dim: int,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
):
    """
    方案 B-snap-local: m_block 在算出时立即 floor 到全局 log2 网格上,
    所有 m/corr/alpha 都落在该网格上, corr = 2^integer 精确 E8M0,
    online softmax 不变式得以保持 (与方案 A 的关键区别).

    与 _blockwise_quantize_p_eff (方案 A floor corr) 的对比:
      | 项目              | 方案 A          | 方案 B-snap-local  |
      | m_block 处理      | 用 raw 值       | floor 到 log2 网格 |
      | P_local block max | 恒 1.0          | ∈ [1, 2)           |
      | corr             | E8M0 floor 近似 | 2^integer 精确    |
      | 不变式           | ✗ 破坏          | ✓ 保持             |
      | pipelining       | ✓ 保留          | ✓ 保留             |
      | 主要误差         | per-block bias  | block max 元素饱和 |

    饱和分析: P_local block max = exp(m_block - m_block_snap) = exp(ε), ε ∈ [0, ln 2).
    FP4 grid 顶端 1.5 (= 6 × 0.25), exp(ε) > 1.5 (即 ε > ln 1.5 ≈ 0.405) 时 max
    元素饱和到 1.5. 假设 ε 均匀分布, 饱和率约 (ln 2 - ln 1.5) / ln 2 ≈ 41.5%.

    参数与 _blockwise_quantize_p_eff 一致, 但 m 的语义不同:
    m_running / m_new 为 FP32 整数值 K (即 floor(m_block_raw / ln 2)), **不是** K × ln 2.
    外部 alpha = torch.exp2(K_old - K_new), 整数次幂在 FP32 中位精确 (2^integer).
    内部 corr = 0.25 × 2^(K_block - K_new) = torch.exp2(K_diff - 2) (s_dtype),
    把 FP4 dequant 因子 (2^-2) 与 K_diff 合并为单一整数指数的 per-block E8M0 scale;
    返回 P_eff = FP4_raw_data × corr_broad, FP4_raw_data = levels[idx] (未乘 0.25).
    m_block_snap = K × LN2_s 仅本函数内用于 P_local = exp(S - m_block_snap) 的局部归一化.
    """
    NEG_INF = float('-inf')
    device = S.device
    src_dtype = S.dtype
    ndim = S.dim()
    seqk_dim_pos = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[seqk_dim_pos]
    LN2 = math.log(2.0)

    # 阶段 1: pad 到 mx_block_size 倍数 (-inf padding 不影响 max)
    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pad_shape = list(S.shape)
        pad_shape[seqk_dim_pos] = pad
        pad_tensor = torch.full(pad_shape, NEG_INF, dtype=src_dtype, device=device)
        S_padded = torch.cat([S, pad_tensor], dim=seqk_dim_pos)
    else:
        S_padded = S

    S_moved = S_padded.movedim(seqk_dim_pos, -1)
    other_shape = S_moved.shape[:-1]
    n_blk = S_moved.shape[-1] // mx_block_size
    S_reshape = S_moved.reshape(*other_shape, n_blk, mx_block_size)

    # 阶段 2: m_block_raw, 处理 -inf, 然后立即 snap 到 log2 网格 (无 m_new 依赖!)
    m_block_raw = S_reshape.max(dim=-1).values  # [..., n_blk]
    m_block_raw_safe = torch.where(
        torch.isinf(m_block_raw) & (m_block_raw < 0),
        torch.zeros_like(m_block_raw),
        m_block_raw,
    )
    # snap: floor(m / ln 2) × ln 2, 落到全局 log2 网格上
    K_block = torch.floor(m_block_raw_safe / LN2)
    K_block = torch.clamp(K_block, min=-127.0, max=127.0)

    ### local_group_max
    local_group_max = K_block.T
    local_group_max.numpy().tofile('dump_data/local_group_max.bin')
    print(f'local_group_max.shape={local_group_max.shape} K_block.dtype={local_group_max.dtype}')

    m_block_snap = K_block * LN2  # [..., n_blk], on log2 grid

    # 阶段 3: P_local = exp(S - m_block_snap), max per block ∈ [1, 2)
    # 保持 [..., n_blk, mx_block_size] 块对齐布局, FP4 round 在块对齐形态完成 (供 dump)
    P_local_r = torch.exp(S_reshape - m_block_snap.unsqueeze(-1))

    # 阶段 4: FP4 round (MX scale 恒 = 0.25 = 2^-2; dequant 因子折入 corr; bucketize 自动处理顶端饱和)
    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    P_local_x4_padded = P_local_r * 4.0  # [..., n_blk, mx_block_size], ∈ [0, 8)
    P_local_x4_padded_bf16 = P_local_x4_padded.to(torch.bfloat16)
    P_local_x4_padded = P_local_x4_padded_bf16.to(P_local_x4_padded.dtype)
    idx_padded_r = torch.bucketize(P_local_x4_padded.contiguous(), midpoints, right=True)
    P_q_padded_r = levels[idx_padded_r]  # FP4 raw data, 块对齐, 未乘 0.25

    ### P FP4
    p_packed_x4 = _pack_fp4_e2m1_along_last(idx_padded_r.reshape(*other_shape, n_blk * mx_block_size).T)
    p_packed_x4.numpy().tofile('dump_data/p_packed_x4.bin')
    p_packed_x4_nz = p_packed_x4.reshape(p_packed_x4.shape[0] // 256, 256, 2, 32).transpose(2, 1)
    p_packed_x4_nz.numpy().tofile('dump_data/p_packed_x4_nz.bin')
    torch.save(p_packed_x4_nz, 'p_packed_x4_nz.pt')

    # strip pad + 移回原 seqk 位置, 得到 [..., seqk_len] 后的 P_q (供 corr 乘)
    P_q_padded_moved = P_q_padded_r.reshape(*other_shape, n_blk * mx_block_size)
    P_q_moved = P_q_padded_moved[..., :seqk_len]
    P_q = P_q_moved.movedim(-1, seqk_dim_pos)

    # 阶段 5: K_new = max(K_running, K_ij), 直接在整数 K 域取最大
    # K_block (s_dtype, 整数值, 来自阶段 2 的 floor) -> K_ij = max -> 提 FP32 合并
    K_ij = K_block.max(dim=-1).values  # s_dtype, 整数
    K_ij_fp32 = K_ij.float()
    if m_running is None:
        m_new = K_ij_fp32  # FP32 K_new
    else:
        # m_running 为 FP32 整数 K (来自前一 tile 输出)
        m_new = torch.maximum(m_running, K_ij_fp32)  # FP32 K_new

    # 阶段 6: corr = 0.25 × 2^(K_block - K_new) = 2^(K_diff - 2), per-block E8M0 scale
    # FP4 dequant 因子 (2^-2) 与 K_diff 合并为单一整数指数; 在 s_dtype 中位精确
    # ★ K_new cast 回 src_dtype 与 K_block (s_dtype) 同 dtype 算
    K_new_s = m_new.to(src_dtype)
    ### global_max
    K_new_s.numpy().tofile('dump_data/global_max.bin')
    print(f'K_new_s.shape={K_new_s.shape} K_new_s.dtype={K_new_s.dtype}')

    K_diff = K_block - K_new_s.unsqueeze(-1)  # ≤ 0, s_dtype 整数
    corr = torch.exp2(K_diff - 2.0)  # 2^(K_diff - 2), 精确
    ### p_scale_e8m0
    p_scale_e8m0 = _to_e8m0_uint8(K_diff.to(torch.int32) - 2).T.contiguous()
    p_scale_e8m0.numpy().tofile('dump_data/p_scale_e8m0.bin')
    print(f'p_scale_e8m0.shape={p_scale_e8m0.shape} p_scale_e8m0.dtype={p_scale_e8m0.dtype}')
    p_scale_e8m0_nz = p_scale_e8m0.view(p_scale_e8m0.shape[0], p_scale_e8m0.shape[1] // 16, 16).transpose(1, 0)
    p_scale_e8m0_nz = p_scale_e8m0_nz.view(p_scale_e8m0.shape[1] // 16, p_scale_e8m0.shape[0] // 2, 2, 16).transpose(
        3, 2
    )
    print(f'p_scale_e8m0_nz.shape={p_scale_e8m0_nz.shape} p_scale_e8m0_nz.dtype={p_scale_e8m0_nz.dtype}')
    p_scale_e8m0_nz.numpy().tofile('dump_data/p_scale_e8m0_nz.bin')
    torch.save(p_scale_e8m0_nz, 'p_scale_e8m0_nz.pt')

    # 阶段 7: corr 广播回 seqk_len, 与 P_q 相乘得 P_eff
    corr_broad_moved = corr.repeat_interleave(mx_block_size, dim=-1)
    corr_broad_moved = corr_broad_moved[..., :seqk_len]
    corr_broad = corr_broad_moved.movedim(-1, seqk_dim_pos)
    P_eff = P_q * corr_broad

    return P_eff, m_new


# ============================================================
# Varlen 公共辅助
# ============================================================


def _normalize_cu_seqlens(cu_seqlens, seq_used, batch_size, tag=""):
    """Validate/convert cu_seqlens + seq_used -> int lists."""
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = cu_seqlens.tolist()
    cu_seqlens = [int(x) for x in cu_seqlens]
    if len(cu_seqlens) != batch_size + 1:
        raise ValueError(f"cu_seqlens_{tag} 长度应为 batch+1={batch_size + 1}, 实际 {len(cu_seqlens)}")
    storage = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(batch_size)]
    for i, s in enumerate(storage):
        if s < 0:
            raise ValueError(f"cu_seqlens_{tag}[{i + 1}]={cu_seqlens[i + 1]} < cu_seqlens_{tag}[{i}]={cu_seqlens[i]}")

    if seq_used is None:
        seq_used = storage
    else:
        if isinstance(seq_used, torch.Tensor):
            seq_used = seq_used.tolist()
        seq_used = [int(x) for x in seq_used]
        if len(seq_used) != batch_size:
            raise ValueError(f"seq_used_{tag} 长度应为 batch={batch_size}, 实际 {len(seq_used)}")
        for i in range(batch_size):
            if seq_used[i] < 0 or seq_used[i] > storage[i]:
                raise ValueError(f"batch {i}: seq_used_{tag}[{i}]={seq_used[i]} 越界 [0, {storage[i]}]")
    return cu_seqlens, seq_used


def _apply_input_quant(q, k, v, cu_kv, used_kv, mx_block_size, mx_mode, v_quant_axis):
    """
    Packed Q/K/V 整体 MXFP4 qdq.
    V@seq_k 时 per-batch 处理, 避免 32-block 跨 batch 边界污染 scale。
    """
    q_dq = mxfp4_qdq(q, axis=-1, block_size=mx_block_size, mode=mx_mode)
    k_dq = mxfp4_qdq(k, axis=-1, block_size=mx_block_size, mode=mx_mode)
    if v_quant_axis == "head_dim":
        v_dq = mxfp4_qdq(v, axis=-1, block_size=mx_block_size, mode=mx_mode)
    elif v_quant_axis == "seq_k":
        v_dq = v.clone()
        for off, sk in zip(cu_kv, used_kv):
            if sk == 0:
                continue
            v_dq[off : off + sk] = mxfp4_qdq(
                v[off : off + sk],
                axis=0,
                block_size=mx_block_size,
                mode=mx_mode,
            )
    else:
        raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")
    return q_dq, k_dq, v_dq


def _expand_kv_for_gqa_packed(k, v, num_heads_q):
    """Packed 布局 [seq, H, D]: K/V 沿 dim=1 repeat_interleave 到 num_heads_q."""
    num_heads_kv = k.shape[1]
    if num_heads_q == num_heads_kv:
        return k, v
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(f"GQA: num_heads_q ({num_heads_q}) 必须能被 num_heads_kv ({num_heads_kv}) 整除")
    if v.shape[1] != num_heads_kv:
        raise ValueError(f"K 头数 ({num_heads_kv}) 与 V 头数 ({v.shape[1]}) 不一致")
    gs = num_heads_q // num_heads_kv
    return k.repeat_interleave(gs, dim=1), v.repeat_interleave(gs, dim=1)


# ============================================================
# 单 batch tiled flash kernel (FA v2)
# ============================================================


def _flash_attn_single_batch_kernel(
    Q: torch.Tensor,  # [H, sq, D]
    K: torch.Tensor,  # [H, sk, D]
    V: torch.Tensor,  # [H, sk, D_v]
    softmax_scale: float,
    causal: bool,
    causal_offset: int,
    block_q: int,
    block_kv: int,
    quantize_p: bool,
    mx_block_size: int,
    mx_mode: str,
    s_layout: str = "ND",
    quantize_p_mode: str = "global",
    s_dtype: str = "fp32",
) -> torch.Tensor:
    """
    单 batch tiled FA v2; 输入已 dequant (FP32). 可选 P MXFP4 qdq.

    s_layout:
        "ND" - S_ij = Q_i @ K_j^T, shape [bq, bkv], softmax along last dim
        "DN" - S_ij = K_j @ Q_i^T, shape [bkv, bq], softmax along seq_k (dim 0)

    quantize_p_mode (仅 quantize_p=True 时生效):
        "global"             - 用全局 running max 归一化 P, MX scale 自适应
        "blockwise"          - 方案 A: 用 block 局部 max 归一化 P (MX scale 恒 0.25),
                               corr = exp(m_block - m_new) floor 到 2^k 折叠回 E8M0.
                               破坏 online softmax 不变式, 但保留 pipelining.
        "blockwise_snap_local" - 方案 B-snap-local: m_block 立即 snap 到 log2 网格,
                                 corr = 2^integer 精确, 不变式保持, pipelining 保留;
                                 代价是每 block max 元素可能饱和到 1.5 (~42% blocks).

    s_dtype: "fp32" / "fp16" / "bf16"
        Q@K^T matmul 累加在 FP32, 结果转为 s_dtype 后, S 相关的 max / exp / P /
        P_scale (含 blockwise helper 内部) 均在 s_dtype.
        sum(P) / PV matmul / O 更新 / m_i 跨 tile running state 与 alpha 更新因子均在 FP32.
        m_running 以 FP32 传入 helper, helper 内部在与 S 派生量 (m_block / corr) 算术时
        cast 回 s_dtype, 返回 m_new 仍为 FP32.

        m_i / alpha 的存储语义按 quantize_p_mode 分派:
          - "blockwise_snap_local": m_i 存 K (整数, FP32), alpha = torch.exp2(K_old - K_new),
            位精确 2^integer, 完全绕过 ln 2 精度噪声.
          - 其他模式 (global / blockwise / quantize_p=False): m_i 存实数 max,
            alpha = exp(m_old - m_new).
    """
    if s_layout not in ("ND", "DN"):
        raise ValueError(f"s_layout must be 'ND' or 'DN', got '{s_layout}'")
    if quantize_p_mode not in ("global", "blockwise", "blockwise_snap_local"):
        raise ValueError(
            f"quantize_p_mode must be 'global' / 'blockwise' / 'blockwise_snap_local', got '{quantize_p_mode}'"
        )
    s_torch_dtype = _resolve_s_dtype(s_dtype)
    H, sq, _ = Q.shape
    sk = K.shape[1]
    d_v = V.shape[-1]
    device = Q.device
    NEG_INF = float('-inf')

    out = torch.zeros(H, sq, d_v, dtype=torch.float32, device=device)
    num_q_tiles = (sq + block_q - 1) // block_q
    num_kv_tiles = (sk + block_kv - 1) // block_kv

    for h in range(H):
        Qh, Kh, Vh = Q[h], K[h], V[h]

        for i in range(num_q_tiles):
            q_lo = i * block_q
            q_hi = min(q_lo + block_q, sq)
            Q_i = Qh[q_lo:q_hi]
            bq = q_hi - q_lo

            m_i = torch.full((bq,), NEG_INF, dtype=torch.float32, device=device)
            l_i = torch.zeros(bq, dtype=torch.float32, device=device)
            O_i = torch.zeros(bq, d_v, dtype=torch.float32, device=device)

            for j in range(num_kv_tiles):
                k_lo = j * block_kv
                k_hi = min(k_lo + block_kv, sk)

                if causal and k_lo > (q_hi - 1) + causal_offset:
                    break

                K_j = Kh[k_lo:k_hi]
                V_j = Vh[k_lo:k_hi]

                # (1) S = Q @ K^T (ND) 或 K @ Q^T (DN), matmul 累加在 FP32
                if s_layout == "ND":
                    S_ij = Q_i @ K_j.t()  # FP32 [bq, bkv]
                else:
                    S_ij = K_j @ Q_i.t()  # FP32 [bkv, bq]

                # ★ S 转入 s_dtype, 后续 max/exp/P 都在 s_dtype 算
                S_ij = S_ij.to(s_torch_dtype)

                S_ij.numpy().tofile('dump_data/mm1Res.bin')
                print(f'mm1Res.shape={S_ij.shape} mm1Res.dtype={S_ij.dtype}')

                S_ij = S_ij * softmax_scale

                # (2) causal mask
                if causal:
                    q_pos_1d = torch.arange(q_lo, q_hi, device=device) + causal_offset
                    k_pos_1d = torch.arange(k_lo, k_hi, device=device)
                    if s_layout == "ND":
                        mask = k_pos_1d.unsqueeze(0) > q_pos_1d.unsqueeze(1)
                    else:
                        mask = k_pos_1d.unsqueeze(1) > q_pos_1d.unsqueeze(0)
                    S_ij = S_ij.masked_fill(mask, NEG_INF)

                seqk_dim = -1 if s_layout == "ND" else 0
                # m_i 全程 FP32 (running state); 与 S 相关的 max/exp 在 s_dtype 内部完成,
                # alpha = exp(m_old - m_new) 在 FP32 计算

                if quantize_p and quantize_p_mode in ("blockwise", "blockwise_snap_local"):
                    # 两种 blockwise 方案共享调用签名, dispatch 到不同 helper
                    if quantize_p_mode == "blockwise":
                        _helper_fn = _blockwise_quantize_p_eff
                    else:
                        _helper_fn = _blockwise_snap_local_quantize_p_eff
                    # helper 接收 FP32 m_running, 返回 FP32 m_new; P_eff_s 在 s_dtype
                    P_eff_s, m_new = _helper_fn(
                        S_ij,
                        m_running=m_i,
                        seqk_dim=seqk_dim,
                        mx_block_size=mx_block_size,
                        mx_mode=mx_mode,
                    )
                    P_fp32 = P_eff_s.float()
                else:
                    # global quantize_p 或 quantize_p=False, S 在 s_dtype
                    m_ij_s = S_ij.max(dim=seqk_dim).values  # s_dtype
                    # m_new 在 FP32 (供 alpha); 通过 m_ij.float() 与 m_i 合并
                    m_new = torch.maximum(m_i, m_ij_s.float())  # FP32
                    m_safe_fp32 = torch.where(
                        torch.isinf(m_new) & (m_new < 0),
                        torch.zeros_like(m_new),
                        m_new,
                    )
                    # ★ P 与 S 同 dtype 算: m_safe cast 回 s_dtype
                    m_safe_s = m_safe_fp32.to(s_torch_dtype)
                    if s_layout == "ND":
                        P_ij = torch.exp(S_ij - m_safe_s.unsqueeze(-1))  # s_dtype
                    else:
                        P_ij = torch.exp(S_ij - m_safe_s.unsqueeze(0))  # s_dtype

                    if quantize_p:
                        # qdq FP32 内部, round-trip 到 s_dtype
                        P_ij = mxfp4_qdq(
                            P_ij.float(),
                            axis=seqk_dim,
                            block_size=mx_block_size,
                            mode=mx_mode,
                        ).to(s_torch_dtype)

                    P_fp32 = P_ij.float()

                # ★ alpha 在 FP32 计算; snap_local 时 m_i 存 K (整数), 用 exp2(K_diff) 位精确
                m_i_safe_fp32 = torch.where(
                    torch.isinf(m_i) & (m_i < 0),
                    torch.zeros_like(m_i),
                    m_i,
                )
                m_new_safe_fp32 = torch.where(
                    torch.isinf(m_new) & (m_new < 0),
                    torch.zeros_like(m_new),
                    m_new,
                )
                if quantize_p and quantize_p_mode == "blockwise_snap_local":
                    alpha = torch.exp2(m_i_safe_fp32 - m_new_safe_fp32)  # 2^K_diff, 精确
                else:
                    alpha = torch.exp(m_i_safe_fp32 - m_new_safe_fp32)  # FP32

                # ★ FP32 累加: sum, PV matmul, O 更新
                mm2ReduceSum = P_fp32.sum(dim=seqk_dim)
                mm2ReduceSum.numpy().tofile('dump_data/mm2ReduceSum.bin')
                print(f'mm2ReduceSum.shape={mm2ReduceSum.shape} mm2ReduceSum.dtype={mm2ReduceSum.dtype}')
                l_i = alpha * l_i + P_fp32.sum(dim=seqk_dim)
                if s_layout == "ND":
                    O_i = alpha.unsqueeze(-1) * O_i + P_fp32 @ V_j
                else:
                    PV_i = V_j.t() @ P_fp32
                    PV_i.numpy().tofile('dump_data/mm2Res_1.bin')
                    print(f'mm2Res.shape={PV_i.shape} mm2Res.dtype={PV_i.dtype}')
                    O_i = alpha.unsqueeze(-1) * O_i + P_fp32.t() @ V_j

                # m_i 跨 tile 存 FP32 (running state, 来自 helper / global 路径均为 FP32)
                m_i = m_new

            l_safe = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
            O_i = O_i / l_safe.unsqueeze(-1)
            out[h, q_lo:q_hi] = O_i

    return out


# ============================================================
# 入口 1: 非分块 Reference (flash 的对照基线)
# ============================================================


def attention_cpu_golden_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_used_q=None,
    seq_used_kv=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    quantize: bool = False,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
    quantize_p: bool = False,
    s_layout: str = "ND",
    quantize_p_mode: str = "global",
    s_dtype: str = "fp32",
) -> torch.Tensor:
    """
    Varlen 非分块 attention reference, 用作 flash 的对照。
    `quantize=True` 时启用 Q/K/V (可选 P) MXFP4 qdq。
    `s_layout`: "ND" -> S = Q@K^T; "DN" -> S = K@Q^T (数学等价).
    `quantize_p_mode`: "global" (默认) / "blockwise" (方案 A floor corr) /
                      "blockwise_snap_local" (方案 B-snap-local, m_block 局部 snap 到
                      log2 网格, 不变式保持). 仅 quantize_p=True 生效.
    `s_dtype`: "fp32"/"fp16"/"bf16" — Q@K^T 后 S 转入该 dtype,
              max/exp/P/P_scale 在该 dtype 算; sum/PV matmul/O 更新仍 FP32.
    Unused slot (seq_used 到存储末尾) 填 0。
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("Varlen 期望 3D packed 输入 [total_seq, num_heads, head_dim]")
    if s_layout not in ("ND", "DN"):
        raise ValueError(f"s_layout must be 'ND' or 'DN', got '{s_layout}'")
    if quantize_p_mode not in ("global", "blockwise", "blockwise_snap_local"):
        raise ValueError(
            f"quantize_p_mode must be 'global' / 'blockwise' / 'blockwise_snap_local', got '{quantize_p_mode}'"
        )
    s_torch_dtype = _resolve_s_dtype(s_dtype)

    total_q, num_heads_q, head_dim = q.shape
    num_heads_kv = k.shape[1]
    d_v = v.shape[-1]
    device = q.device

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = cu_seqlens_q.numel() - 1 if isinstance(cu_seqlens_q, torch.Tensor) else len(cu_seqlens_q) - 1
    cu_q, used_q = _normalize_cu_seqlens(cu_seqlens_q, seq_used_q, batch_size, "q")
    cu_kv, used_kv = _normalize_cu_seqlens(cu_seqlens_kv, seq_used_kv, batch_size, "kv")

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(f"GQA: num_heads_q ({num_heads_q}) 必须能被 num_heads_kv ({num_heads_kv}) 整除")
    gs = num_heads_q // num_heads_kv

    out = torch.zeros(total_q, num_heads_q, d_v, dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]
        if sq == 0 or sk == 0:
            continue

        Qb = q[q_off : q_off + sq].float()  # [sq, Hq,  D]
        Kb = k[kv_off : kv_off + sk].float()  # [sk, Hkv, D]
        Vb = v[kv_off : kv_off + sk].float()  # [sk, Hkv, D_v]

        if quantize:
            Qb = mxfp4_qdq(Qb, axis=-1, block_size=mx_block_size, mode=mx_mode)
            Kb = mxfp4_qdq(Kb, axis=-1, block_size=mx_block_size, mode=mx_mode)
            if v_quant_axis == "head_dim":
                Vb = mxfp4_qdq(Vb, axis=-1, block_size=mx_block_size, mode=mx_mode)
            elif v_quant_axis == "seq_k":
                Vb = mxfp4_qdq(Vb, axis=0, block_size=mx_block_size, mode=mx_mode)
            else:
                raise ValueError(f"unknown v_quant_axis: {v_quant_axis}")

        if gs > 1:
            Kb = Kb.repeat_interleave(gs, dim=1)
            Vb = Vb.repeat_interleave(gs, dim=1)

        Qh = Qb.transpose(0, 1)  # [Hq, sq, D]
        Kh = Kb.transpose(0, 1)  # [Hq, sk, D]
        Vh = Vb.transpose(0, 1)  # [Hq, sk, D_v]

        # S 矩阵: matmul 累加在 FP32, ND = Q@K^T; DN = K@Q^T
        if s_layout == "ND":
            S = (Qh @ Kh.transpose(-1, -2)) * softmax_scale  # FP32 [Hq, sq, sk]
        else:
            S = (Kh @ Qh.transpose(-1, -2)) * softmax_scale  # FP32 [Hq, sk, sq]

        # ★ S 转入 s_dtype, 后续 max/exp/P 都在 s_dtype 算
        S = S.to(s_torch_dtype)

        if causal:
            offset = sk - sq
            q_1d = torch.arange(sq, device=device) + offset
            k_1d = torch.arange(sk, device=device)
            if s_layout == "ND":
                mask = k_1d.unsqueeze(0) > q_1d.unsqueeze(1)
            else:
                mask = k_1d.unsqueeze(1) > q_1d.unsqueeze(0)
            S = S.masked_fill(mask, float('-inf'))

        # softmax 沿 seq_k 轴: ND=-1, DN=-2
        seqk_dim = -1 if s_layout == "ND" else -2

        if quantize and quantize_p and quantize_p_mode in ("blockwise", "blockwise_snap_local"):
            # 两种 blockwise 方案共享调用签名, dispatch 到不同 helper
            if quantize_p_mode == "blockwise":
                _helper_fn = _blockwise_quantize_p_eff
            else:
                _helper_fn = _blockwise_snap_local_quantize_p_eff
            P_eff_s, _ = _helper_fn(
                S,
                m_running=None,
                seqk_dim=seqk_dim,
                mx_block_size=mx_block_size,
                mx_mode=mx_mode,
            )
            P_eff_s = torch.nan_to_num(P_eff_s, nan=0.0)
            # ★ FP32 做 sum 与 PV matmul
            P_fp32 = P_eff_s.float()
            prob_sum = P_fp32.sum(dim=seqk_dim, keepdim=True)
            l_safe = torch.where(prob_sum > 0, prob_sum, torch.ones_like(prob_sum))
            if s_layout == "ND":
                Ob = (P_fp32 @ Vh) / l_safe
            else:
                Ob = (P_fp32.transpose(-1, -2) @ Vh) / l_safe.transpose(-1, -2)
        elif quantize and quantize_p:
            # global quantize_p: max/exp/P 全在 s_dtype, qdq FP32 round-trip
            m = S.max(dim=seqk_dim, keepdim=True).values  # s_dtype
            m_safe = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
            P = torch.exp(S - m_safe)  # s_dtype
            P = torch.nan_to_num(P, nan=0.0)
            P = mxfp4_qdq(P.float(), axis=seqk_dim, block_size=mx_block_size, mode=mx_mode).to(s_torch_dtype)
            # ★ FP32 做 sum 与 PV matmul
            P_fp32 = P.float()
            prob_sum = P_fp32.sum(dim=seqk_dim, keepdim=True)
            l_safe = torch.where(prob_sum > 0, prob_sum, torch.ones_like(prob_sum))
            if s_layout == "ND":
                Ob = (P_fp32 @ Vh) / l_safe
            else:
                Ob = (P_fp32.transpose(-1, -2) @ Vh) / l_safe.transpose(-1, -2)
        else:
            # 不量化 P: 手动 softmax (max/exp/normalize 沿 s_dtype) 以便 s_dtype 生效
            # 注: 与 torch.softmax 在 FP32 下完全等价 (atol 1e-7), 但允许 s_dtype 透传
            m = S.max(dim=seqk_dim, keepdim=True).values  # s_dtype
            m_safe = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
            P = torch.exp(S - m_safe)  # s_dtype
            P = torch.nan_to_num(P, nan=0.0)
            # ★ FP32 做 sum 与归一化
            P_fp32 = P.float()
            prob_sum = P_fp32.sum(dim=seqk_dim, keepdim=True)
            l_safe = torch.where(prob_sum > 0, prob_sum, torch.ones_like(prob_sum))
            if s_layout == "ND":
                Ob = (P_fp32 @ Vh) / l_safe
            else:
                Ob = (P_fp32.transpose(-1, -2) @ Vh) / l_safe.transpose(-1, -2)

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out


# ============================================================
# 入口 2: Tiled Flash (主要的 golden)
# ============================================================


def flash_attention_cpu_golden_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_used_q=None,
    seq_used_kv=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
    quantize: bool = False,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    v_quant_axis: str = "head_dim",
    quantize_p: bool = False,
    s_layout: str = "ND",
    quantize_p_mode: str = "global",
    s_dtype: str = "fp32",
) -> torch.Tensor:
    """
    Varlen tiled FlashAttention v2, `quantize` 开关切换 FP32 / MXFP4。

    参数:
        quantize:       False -> 全程 FP32 无量化
                        True  -> Q/K/V (可选 P) MXFP4 qdq
        当 quantize=False 时, 后面所有 mx_* / v_quant_axis / quantize_p 被忽略

        s_layout:       "ND" (默认) -> kernel 内 S = Q @ K^T, [bq, bkv]
                        "DN"        -> kernel 内 S = K @ Q^T, [bkv, bq]
                        两种 layout 数学等价 (输出仅相差 FP 求和顺序), 外部
                        API 不变 (Q/K/V/O shape 一致, 只是中间 S/P 布局变).

        其余: 见模块 docstring。

    返回: [total_seq_q, num_heads, head_dim_v]
        valid 区域为 attention 输出, unused slot 全 0。
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"Varlen 期望 3D 输入, 实际 q.dim()={q.dim()}, k.dim()={k.dim()}, v.dim()={v.dim()}")

    total_q, num_heads_q, head_dim = q.shape
    total_kv, num_heads_kv, _ = k.shape
    d_v = v.shape[-1]
    device = q.device

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = cu_seqlens_q.numel() - 1 if isinstance(cu_seqlens_q, torch.Tensor) else len(cu_seqlens_q) - 1
    cu_q, used_q = _normalize_cu_seqlens(cu_seqlens_q, seq_used_q, batch_size, "q")
    cu_kv, used_kv = _normalize_cu_seqlens(cu_seqlens_kv, seq_used_kv, batch_size, "kv")
    if cu_q[-1] > total_q:
        raise ValueError(f"cu_seqlens_q[-1]={cu_q[-1]} > total_seq_q={total_q}")
    if cu_kv[-1] > total_kv:
        raise ValueError(f"cu_seqlens_kv[-1]={cu_kv[-1]} > total_seq_kv={total_kv}")

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(f"GQA: num_heads_q ({num_heads_q}) 必须能被 num_heads_kv ({num_heads_kv}) 整除")

    # ---- 可选 MXFP4 qdq (Q/K/V) ----
    if quantize:
        q_in, k_in, v_in = _apply_input_quant(
            q,
            k,
            v,
            cu_kv,
            used_kv,
            mx_block_size,
            mx_mode,
            v_quant_axis,
        )
    else:
        q_in, k_in, v_in = q, k, v

    # ---- GQA 展开 ----
    k_exp, v_exp = _expand_kv_for_gqa_packed(k_in, v_in, num_heads_q)

    # ---- 输出 buffer, unused slot 默认 0 ----
    out = torch.zeros(total_q, num_heads_q, d_v, dtype=torch.float32, device=device)

    # ---- 逐 batch kernel ----
    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]
        if sq == 0 or sk == 0:
            continue

        Qb = q_in[q_off : q_off + sq].transpose(0, 1).float().contiguous()
        Kb = k_exp[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()
        Vb = v_exp[kv_off : kv_off + sk].transpose(0, 1).float().contiguous()

        causal_offset = (sk - sq) if causal else 0

        Ob = _flash_attn_single_batch_kernel(
            Qb,
            Kb,
            Vb,
            softmax_scale=softmax_scale,
            causal=causal,
            causal_offset=causal_offset,
            block_q=block_q,
            block_kv=block_kv,
            quantize_p=(quantize and quantize_p),
            mx_block_size=mx_block_size,
            mx_mode=mx_mode,
            s_layout=s_layout,
            quantize_p_mode=quantize_p_mode,
            s_dtype=s_dtype,
        )

        out[q_off : q_off + sq] = Ob.transpose(0, 1)

    return out.to(q.dtype)


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # -------- 精度比较 helpers --------
    def _pass_rate(a, b, atol, rtol):
        """逐元素 |a-b| <= atol + rtol*|b| 的通过率."""
        diff = (a - b).abs()
        tol = atol + rtol * b.abs()
        passed = int((diff <= tol).sum().item())
        total = a.numel()
        pct = passed / total * 100.0
        return passed, total, pct, diff.max().item()

    def _check(name, a, b, atol=1e-5, rtol=1e-4):
        """算法不变式: 全部元素必须满足允差; pass% 必为 100%."""
        passed, total, pct, max_diff = _pass_rate(a, b, atol, rtol)
        ok = passed == total
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] {name}: pass {passed}/{total} ({pct:.2f}%), max diff {max_diff:.3e}")
        assert ok, f"{name}: only {pct:.2f}% pass, max diff {max_diff:.3e} > atol={atol:.0e}"
        return max_diff

    def _check_true(name, cond):
        ok = bool(cond)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {cond}")
        assert ok, f"{name} failed"

    def _info_cmp(name, a, b, atol=5e-2, rtol=1e-2):
        """量化噪声描述性指标 (不断言): pass@atol 百分比 + max/mean/cos."""
        passed, total, pct, max_diff = _pass_rate(a, b, atol, rtol)
        mean_err = (a - b).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        print(
            f"  [info] {name}: pass@{atol:.0e} {passed}/{total} ({pct:.2f}%), "
            f"max {max_diff:.3e}, mean {mean_err:.3e}, cos {cos:.4f}"
        )

    print("=" * 72)
    print("Flash Attention Varlen CPU Golden 自检 (FP32 / MXFP4)")
    print("=" * 72)

    # -------- 公共测试数据 --------
    # 主 varlen 配置: 4 batch, 不等长 + 含 unused slot + KV > Q
    CU_Q = [0, 3, 4, 6, 9]
    USED_Q = [2, 1, 2, 3]  # batch0 sq=2(<3 含 unused), batch1=1, b2=2, b3=3
    CU_KV = [0, 5, 7, 12, 16]
    USED_KV = [4, 1, 5, 3]  # KV 长度 >= Q (模拟 prefix cache / decoding)
    BATCH = len(USED_Q)
    NH_Q, NH_KV, HD = 8, 2, 64  # GQA 8/2
    TOTAL_Q = CU_Q[-1]
    TOTAL_KV = CU_KV[-1]

    Q = torch.randn(TOTAL_Q, NH_Q, HD)
    K = torch.randn(TOTAL_KV, NH_KV, HD)
    V = torch.randn(TOTAL_KV, NH_KV, HD)

    LAYOUTS = ["ND", "DN"]
    LAYOUT_ATOL = 1e-5  # ND vs DN 的允差: 数学等价, 仅相差 FP 求和顺序

    # ==========================================================
    # Group A: 算法骨架正确性 (ref vs flash) + ND/DN 双跑
    # ==========================================================
    # 测试意图:
    #   FP32 / MXFP4 两种 mode + ND / DN 两种 S 布局, 共 4 组组合下,
    #   tiled flash 与非分块 ref 走的是同一个数学路径, 仅差 FP 求和顺序.
    #   此外 ND vs DN 的输出也应数学等价 (atol=1e-5).
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group A | 算法骨架正确性 (ref vs flash) + ND/DN 双跑")
    print("意图: 每个 cfg 在 ND 和 DN 下分别 ref ≡ flash; 且 ND ≡ DN")
    print("=" * 72)

    a_cfgs = [
        ("A.1 FP32 / 非因果", False, False),
        ("A.2 FP32 / 因果  ", True, False),
        ("A.3 MXFP4 / 非因果", False, True),
        ("A.4 MXFP4 / 因果  ", True, True),
    ]
    for tag, causal_flag, q_flag in a_cfgs:
        print(f"\n--- {tag} ---")
        outs = {}
        for layout in LAYOUTS:
            ref = attention_cpu_golden_varlen(
                Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, causal=causal_flag, quantize=q_flag, s_layout=layout
            )
            fl = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                causal=causal_flag,
                quantize=q_flag,
                block_q=32,
                block_kv=32,
                s_layout=layout,
            )
            _check(f"[{layout}] ref vs flash", ref, fl, atol=1e-5)
            outs[layout] = fl
        _check("ND vs DN (flash)", outs["ND"], outs["DN"], atol=LAYOUT_ATOL)

    print("\n--- A.5 不同 tile 大小一致性 (ND/DN 双跑) ---")
    for layout in LAYOUTS:
        for mode_name, q_flag in [("FP32", False), ("MXFP4", True)]:
            print(f"  [{layout} / {mode_name}]")
            base = None
            for bq, bkv in [(32, 32), (64, 32), (128, 64), (256, 256)]:
                o = flash_attention_cpu_golden_varlen(
                    Q,
                    K,
                    V,
                    CU_Q,
                    CU_KV,
                    USED_Q,
                    USED_KV,
                    causal=True,
                    quantize=q_flag,
                    block_q=bq,
                    block_kv=bkv,
                    s_layout=layout,
                )
                if base is None:
                    base = o
                    print(f"     bq={bq:3d}, bkv={bkv:3d}: (基准)")
                else:
                    _check(f"     {layout} {mode_name} bq={bq} bkv={bkv} vs base", o, base, atol=1e-5)

    # ==========================================================
    # Group B: GQA / MQA 适配 (ND/DN 双跑)
    # ==========================================================
    # 测试意图:
    #   K/V 头数 < Q 头数时 repeat_interleave 展开发生在 kernel 进入之前,
    #   ND/DN 两种 S 布局都应正确处理. 不整除时无论 layout 都需 raise.
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group B | GQA / MQA 适配 (ND/DN 双跑)")
    print("意图: 各 GQA 比例在两种 layout 下都 ref ≡ flash, 且 ND ≡ DN")
    print("=" * 72)

    seq_each, batch_n = 64, 2
    cu_eq = [i * seq_each for i in range(batch_n + 1)]
    gqa_configs = [
        ("MHA(8/8)", 8, 8),
        ("GQA(8/4)", 8, 4),
        ("GQA(8/2)", 8, 2),
        ("MQA(8/1)", 8, 1),
    ]
    for tag, nh_q, nh_kv in gqa_configs:
        print(f"\n--- B.{tag} ---")
        qg = torch.randn(seq_each * batch_n, nh_q, HD)
        kg = torch.randn(seq_each * batch_n, nh_kv, HD)
        vg = torch.randn(seq_each * batch_n, nh_kv, HD)
        for mode_name, q_flag in [("FP32", False), ("MXFP4", True)]:
            outs = {}
            for layout in LAYOUTS:
                ref = attention_cpu_golden_varlen(qg, kg, vg, cu_eq, cu_eq, quantize=q_flag, s_layout=layout)
                fl = flash_attention_cpu_golden_varlen(
                    qg, kg, vg, cu_eq, cu_eq, quantize=q_flag, block_q=32, block_kv=32, s_layout=layout
                )
                _check_true(f"  [{layout}] {mode_name} 输出 shape == Q shape", fl.shape == qg.shape)
                _check(f"  [{layout}] {mode_name} ref vs flash", ref, fl, atol=1e-5)
                outs[layout] = fl
            _check(f"  {mode_name} ND vs DN", outs["ND"], outs["DN"], atol=LAYOUT_ATOL)

    print("\n--- B.整除性校验 (两种 layout 都应 raise) ---")
    for layout in LAYOUTS:
        raised = False
        try:
            flash_attention_cpu_golden_varlen(
                torch.randn(32, 7, HD),
                torch.randn(32, 4, HD),
                torch.randn(32, 4, HD),
                [0, 32],
                [0, 32],
                s_layout=layout,
            )
        except ValueError:
            raised = True
        _check_true(f"  [{layout}] heads_q=7 vs heads_kv=4 触发 ValueError", raised)

    # ==========================================================
    # Group C: Varlen 不变式 (ND/DN 双跑)
    # ==========================================================
    # 测试意图:
    #   varlen 关键不变式 — unused slot 输出 0; 输入不原地修改;
    #   空 batch / 空 KV 不崩溃; decoding (sq=1) 通过 — 在 ND/DN 下都满足.
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group C | Varlen 不变式 (ND/DN 双跑)")
    print("意图: unused=0 / 输入不变 / 空边界 / decoding 在两种 layout 下都成立")
    print("=" * 72)

    print("\n--- C.1 unused slot 全 0 (ND/DN 双跑) ---")
    Q_clone, K_clone, V_clone = Q.clone(), K.clone(), V.clone()
    out_main_by_layout = {}
    for layout in LAYOUTS:
        out_main = flash_attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, causal=True, quantize=True, block_q=32, block_kv=32, s_layout=layout
        )
        out_main_by_layout[layout] = out_main
        unused_max = 0.0
        for b in range(BATCH):
            valid_end = CU_Q[b] + USED_Q[b]
            slot_end = CU_Q[b + 1]
            if valid_end < slot_end:
                unused_max = max(
                    unused_max,
                    out_main[valid_end:slot_end].abs().max().item(),
                )
        _check_true(f"  [{layout}] unused slot max abs = {unused_max:.3e}", unused_max == 0.0)
    _check("  C.1 ND vs DN", out_main_by_layout["ND"], out_main_by_layout["DN"], atol=LAYOUT_ATOL)

    print("\n--- C.2 输入未被原地修改 ---")
    _check_true(
        "C.2 输入 Q/K/V 未变",
        torch.equal(Q, Q_clone) and torch.equal(K, K_clone) and torch.equal(V, V_clone),
    )

    print("\n--- C.3 空 batch (sq=0) / 空 KV (sk=0), ND/DN 双跑 ---")
    for layout in LAYOUTS:
        out_e = flash_attention_cpu_golden_varlen(
            torch.randn(5, NH_Q, HD),
            torch.randn(8, NH_KV, HD),
            torch.randn(8, NH_KV, HD),
            [0, 5, 5],
            [0, 4, 8],
            [3, 0],
            [4, 4],
            s_layout=layout,
        )
        _check_true(f"  [{layout}] 空 batch (sq=0): valid 部分非零", (out_e[:3].abs().sum() > 0).item())
        _check_true(f"  [{layout}] 空 batch (sq=0): unused 全 0", (out_e[3:5].abs().sum() == 0).item())

        out_ek = flash_attention_cpu_golden_varlen(
            torch.randn(5, NH_Q, HD),
            torch.randn(4, NH_KV, HD),
            torch.randn(4, NH_KV, HD),
            [0, 3, 5],
            [0, 4, 4],
            None,
            None,
            s_layout=layout,
        )
        _check_true(f"  [{layout}] 空 KV (sk=0) 对应 Q 输出全 0", (out_ek[3:5].abs().sum() == 0).item())

    print("\n--- C.4 Decoding (sq=1, sk=10), ND/DN 双跑 ---")
    Q_dec = torch.randn(1, NH_Q, HD)
    K_dec = torch.randn(10, NH_KV, HD)
    V_dec = torch.randn(10, NH_KV, HD)
    for mode_name, q_flag in [("FP32", False), ("MXFP4", True)]:
        outs = {}
        for layout in LAYOUTS:
            ref = attention_cpu_golden_varlen(
                Q_dec, K_dec, V_dec, [0, 1], [0, 10], causal=True, quantize=q_flag, s_layout=layout
            )
            fl = flash_attention_cpu_golden_varlen(
                Q_dec,
                K_dec,
                V_dec,
                [0, 1],
                [0, 10],
                causal=True,
                quantize=q_flag,
                block_q=1,
                block_kv=4,
                s_layout=layout,
            )
            _check(f"  [{layout}] {mode_name} decoding ref vs flash", ref, fl, atol=1e-5)
            outs[layout] = fl
        _check(f"  {mode_name} decoding ND vs DN", outs["ND"], outs["DN"], atol=LAYOUT_ATOL)

    # ==========================================================
    # Group D: MXFP4 专属 (ND/DN 双跑)
    # ==========================================================
    # 测试意图:
    #   验证 MXFP4 专有开关 (v_quant_axis, quantize_p) 在两种 S 布局下
    #   语义一致 — V@seq_k 在 DN 下 P 量化 axis 自动切到 -2, 但 seq_k
    #   的语义维度对齐.
    #   单 tile 时 flash ≡ ref; 多 tile 仅描述性观察; 路径必须确定性.
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group D | MXFP4 专属 (V 轴 / P 量化 / 确定性, ND/DN 双跑)")
    print("意图: 两种 layout 下 V/P 量化语义一致; 单tile≡ref; 路径确定性")
    print("=" * 72)

    print("\n--- D.1 V 量化轴 (head_dim vs seq_k), ND/DN 双跑 ---")
    for axis in ("head_dim", "seq_k"):
        outs = {}
        for layout in LAYOUTS:
            ref = attention_cpu_golden_varlen(
                Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=True, v_quant_axis=axis, s_layout=layout
            )
            fl = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                quantize=True,
                v_quant_axis=axis,
                block_q=32,
                block_kv=32,
                s_layout=layout,
            )
            _check(f"  [{layout}] V@{axis} ref vs flash", ref, fl, atol=1e-5)
            outs[layout] = fl
        _check(f"  V@{axis} ND vs DN", outs["ND"], outs["DN"], atol=LAYOUT_ATOL)

    print("\n--- D.2 P 量化单 tile (bkv >= max sk), ND/DN 双跑 ---")
    outs = {}
    for layout in LAYOUTS:
        ref = attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=True, quantize_p=True, s_layout=layout
        )
        fl = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            block_q=64,
            block_kv=64,
            s_layout=layout,
        )
        _check(f"  [{layout}] quantize_p 单 tile ref vs flash", ref, fl, atol=1e-5)
        outs[layout] = fl
    _check("  quantize_p 单 tile ND vs DN", outs["ND"], outs["DN"], atol=LAYOUT_ATOL)

    print("\n--- D.3 P 量化多 tile (描述性, ND/DN 各跑) ---")
    seq_big = 128
    cu_big = [0, seq_big]
    Qb_big = torch.randn(seq_big, NH_Q, HD)
    Kb_big = torch.randn(seq_big, NH_KV, HD)
    Vb_big = torch.randn(seq_big, NH_KV, HD)
    for layout in LAYOUTS:
        single = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big,
            cu_big,
            quantize=True,
            quantize_p=True,
            block_q=64,
            block_kv=seq_big,
            s_layout=layout,
        )
        for bkv in [16, 32, 64]:
            multi = flash_attention_cpu_golden_varlen(
                Qb_big,
                Kb_big,
                Vb_big,
                cu_big,
                cu_big,
                quantize=True,
                quantize_p=True,
                block_q=64,
                block_kv=bkv,
                s_layout=layout,
            )
            _info_cmp(f"  [{layout}] block_kv={bkv:3d} 多tile vs 单tile", single, multi, atol=5e-2)

    print("\n--- D.4 确定性 (ND/DN 各自 bit-identical) ---")
    for layout in LAYOUTS:
        o1 = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            causal=True,
            quantize=True,
            quantize_p=True,
            block_q=32,
            block_kv=32,
            s_layout=layout,
        )
        o2 = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            causal=True,
            quantize=True,
            quantize_p=True,
            block_q=32,
            block_kv=32,
            s_layout=layout,
        )
        _check_true(f"  [{layout}] 同配置两次 bit-identical", torch.equal(o1, o2))

    # ==========================================================
    # Group D.5: quantize_p_mode = "blockwise" (方案 A)
    # ==========================================================
    # 测试意图:
    #   blockwise 用 block 局部 max 归一化 P, MX scale 字段恒 0.25, corr 折叠
    #   到 E8M0 (floor 到 2^k). 验证:
    #   (a) blockwise 下 ref ≡ flash (算法骨架不依赖于 quantize_p_mode)
    #   (b) ND ≡ DN (布局对称)
    #   (c) 单 tile 下 blockwise ≠ global (FP4 data 字段差异, 描述性)
    #   (d) 多 tile 下 blockwise vs global 噪声水平描述性
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group D.5 | quantize_p_mode='blockwise' (方案 A, ND/DN 双跑)")
    print("意图: blockwise 模式 ref ≡ flash + ND ≡ DN; 与 global 量化路径噪声对比")
    print("=" * 72)

    print("\n--- D.5.a blockwise ref vs flash (ND/DN 双跑) ---")
    for layout in LAYOUTS:
        ref = attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise",
            s_layout=layout,
        )
        fl = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise",
            block_q=64,
            block_kv=64,
            s_layout=layout,
        )
        _check(f"  [{layout}] blockwise 单 tile ref vs flash", ref, fl, atol=1e-5)

    print("\n--- D.5.b blockwise ND vs DN (单 tile 与多 tile) ---")
    for bkv in [64, 32, 16]:
        outs = {}
        for layout in LAYOUTS:
            o = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                causal=True,
                quantize=True,
                quantize_p=True,
                quantize_p_mode="blockwise",
                block_q=32,
                block_kv=bkv,
                s_layout=layout,
            )
            outs[layout] = o
        _check(f"  block_kv={bkv:3d} blockwise ND vs DN", outs["ND"], outs["DN"], atol=LAYOUT_ATOL)

    print("\n--- D.5.c blockwise vs global (单 tile, 描述性) ---")
    # 单 tile 下 blockwise 与 global 的 effective MX scale 相同, 但 FP4 data 字段
    # 可能差 ≤ 1.5× quantum at max element. 描述性比较两者输出.
    for layout in LAYOUTS:
        out_global = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="global",
            block_q=64,
            block_kv=64,
            s_layout=layout,
        )
        out_blockwise = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise",
            block_q=64,
            block_kv=64,
            s_layout=layout,
        )
        _info_cmp(f"  [{layout}] 单 tile blockwise vs global", out_global, out_blockwise, atol=5e-2)

    print("\n--- D.5.d blockwise FP32 vs MXFP4 噪声 (大 seq 多 tile, 描述性) ---")
    # 用 D.3 的大 seq 数据复跑, 量化 P + blockwise mode 下的整体噪声水平
    cu_big = [0, seq_big]
    for layout in LAYOUTS:
        fp32_big = flash_attention_cpu_golden_varlen(
            Qb_big, Kb_big, Vb_big, cu_big, cu_big, quantize=False, block_q=64, block_kv=32, s_layout=layout
        )
        mx_blockwise = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big,
            cu_big,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise",
            block_q=64,
            block_kv=32,
            s_layout=layout,
        )
        _info_cmp(f"  [{layout}] FP32 vs MXFP4(P-blockwise) bkv=32", fp32_big, mx_blockwise, atol=5e-2)

    print("\n--- D.5.e blockwise vs global (多 MX-block per tile, 描述性) ---")
    # 注: 当 seq_k <= mx_block_size 时, 每 tile 只有一个 MX block, m_block ≡ m_new,
    # corr ≡ 1, blockwise ≡ global. 要观察差异需 seq_k > 32 (一个 tile 内多个 MX block).
    # 这里用 seq_big=128, bkv=128 (单 tile, 内含 4 个 MX block), 才真正体现 blockwise
    # 在非-winning block 上的 round-direction 差异.
    for layout in LAYOUTS:
        out_global = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big,
            cu_big,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="global",
            block_q=64,
            block_kv=seq_big,
            s_layout=layout,
        )
        out_blockwise = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big,
            cu_big,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise",
            block_q=64,
            block_kv=seq_big,
            s_layout=layout,
        )
        _info_cmp(f"  [{layout}] 多MX-block per tile blockwise vs global", out_global, out_blockwise, atol=5e-2)

    print("\n--- D.5.f quantize_p_mode 参数校验 ---")
    raised = False
    try:
        flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="invalid_mode",
        )
    except ValueError:
        raised = True
    _check_true("D.5.f 非法 quantize_p_mode 触发 ValueError", raised)

    # ==========================================================
    # Group D.6: quantize_p_mode='blockwise_snap_local' (方案 B-snap-local)
    # ==========================================================
    # 测试意图:
    #   方案 B-snap-local 把 m_block 立即 floor 到 log2 网格上, corr 精确 2^integer,
    #   E8M0 严格 + pipelining + 不变式保持. 代价是每 block max 元素可能饱和到 1.5.
    #
    #   (a) ref ≡ flash (算法骨架不变)
    #   (b) ND ≡ DN (布局对称)
    #   (c) 单 tile 下 snap_local vs global 有 FP4 grid 偏移 (描述性, 与 blockwise 不同)
    #   (d) 多 MX-block per tile 下 snap_local 应该 比 blockwise (方案 A) 更接近 global,
    #       因为不变式保持, per-row 偏移在 O/l 时 cancel
    #   (e) 与 s_dtype 组合可工作
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group D.6 | quantize_p_mode='blockwise_snap_local' (方案 B-snap-local)")
    print("意图: snap_local ref ≡ flash + ND ≡ DN; 多 block 下应比 blockwise 更接近 global")
    print("=" * 72)

    print("\n--- D.6.a snap_local ref vs flash (ND/DN 双跑) ---")
    for layout in LAYOUTS:
        ref = attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise_snap_local",
            s_layout=layout,
        )
        fl = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise_snap_local",
            block_q=64,
            block_kv=64,
            s_layout=layout,
        )
        _check(f"  [{layout}] snap_local 单 tile ref vs flash", ref, fl, atol=1e-5)

    print("\n--- D.6.b snap_local ND vs DN (多 tile) ---")
    for bkv in [64, 32, 16]:
        outs = {}
        for layout in LAYOUTS:
            o = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                causal=True,
                quantize=True,
                quantize_p=True,
                quantize_p_mode="blockwise_snap_local",
                block_q=32,
                block_kv=bkv,
                s_layout=layout,
            )
            outs[layout] = o
        _check(f"  block_kv={bkv:3d} snap_local ND vs DN", outs["ND"], outs["DN"], atol=LAYOUT_ATOL)

    print("\n--- D.6.c snap_local vs global (描述性) ---")
    # 单 tile (sk ≤ 32): snap_local 与 global 不同 (FP4 grid 偏移)
    # 多 MX-block per tile: snap_local 与 global 应更接近 (不变式保持)
    for layout in LAYOUTS:
        # 单 MX-block per tile (主测试数据 max sk = 5)
        out_global = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="global",
            block_q=64,
            block_kv=64,
            s_layout=layout,
        )
        out_snap = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise_snap_local",
            block_q=64,
            block_kv=64,
            s_layout=layout,
        )
        _info_cmp(f"  [{layout}] 单 MX-block snap_local vs global", out_global, out_snap, atol=5e-2)

    print("\n--- D.6.d snap_local vs blockwise (多 MX-block per tile, 关键对照) ---")
    # 用 D.5.e 同样配置: seq_big=128 单 tile 内含 4 个 MX block.
    # 关键比较: snap_local vs global 应远好于 blockwise vs global,
    # 因为 snap_local 保持不变式, per-row 偏移在 O/l 处 cancel.
    cu_big_d6 = [0, seq_big]
    for layout in LAYOUTS:
        out_global = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big_d6,
            cu_big_d6,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="global",
            block_q=64,
            block_kv=seq_big,
            s_layout=layout,
        )
        out_blockwise = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big_d6,
            cu_big_d6,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise",
            block_q=64,
            block_kv=seq_big,
            s_layout=layout,
        )
        out_snap = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big_d6,
            cu_big_d6,
            quantize=True,
            quantize_p=True,
            quantize_p_mode="blockwise_snap_local",
            block_q=64,
            block_kv=seq_big,
            s_layout=layout,
        )
        _info_cmp(f"  [{layout}] blockwise (方案A) vs global", out_global, out_blockwise, atol=5e-2)
        _info_cmp(f"  [{layout}] snap_local vs global", out_global, out_snap, atol=5e-2)

    print("\n--- D.6.e snap_local + s_dtype 组合 ---")
    for dt in ["fp32", "bf16", "fp16"]:
        atol_dt = {"fp32": 1e-5, "bf16": 5e-3, "fp16": 5e-3}[dt]
        for layout in LAYOUTS:
            ref = attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                quantize=True,
                quantize_p=True,
                quantize_p_mode="blockwise_snap_local",
                s_dtype=dt,
                s_layout=layout,
            )
            fl = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                quantize=True,
                quantize_p=True,
                quantize_p_mode="blockwise_snap_local",
                block_q=64,
                block_kv=64,
                s_dtype=dt,
                s_layout=layout,
            )
            _check(f"  [{layout}] snap_local s_dtype={dt} ref vs flash", ref, fl, atol=atol_dt)

    print("\n--- D.6.f 饱和率观察 (描述性, 验证 ~42% 块饱和的理论预测) ---")
    # 用一组随机 Q/K 计算 S, 看 snap_local 模式下 P_local 的 block max 分布
    torch.manual_seed(42)  # 固定种子, 结果可复现
    Qs_obs = torch.randn(64, 8, 64)
    Ks_obs = torch.randn(64, 8, 64)
    softmax_scale_obs = 1.0 / math.sqrt(64)
    S_obs = (Qs_obs.transpose(0, 1) @ Ks_obs.transpose(0, 1).transpose(-1, -2)) * softmax_scale_obs
    S_obs = S_obs[0]  # [sq=64, sk=64]
    m_block_raw = S_obs.reshape(64, 2, 32).max(-1).values  # [64, 2]
    LN2_obs = math.log(2.0)
    m_block_snap = torch.floor(m_block_raw / LN2_obs) * LN2_obs
    P_local_max_snap = torch.exp(m_block_raw - m_block_snap)  # = exp(ε) ∈ [1, 2)
    sat_rate = (P_local_max_snap > 1.5).float().mean().item()
    print(
        f"  [info] P_local_max @ snap_local: "
        f"min={P_local_max_snap.min().item():.3f}, "
        f"max={P_local_max_snap.max().item():.3f}, "
        f"mean={P_local_max_snap.mean().item():.3f}"
    )
    print(f"  [info] saturation rate (>1.5) = {sat_rate * 100:.1f}% (theory ~41.5%)")

    # ==========================================================
    # Group E: 量化噪声 (FP32 vs MXFP4, ND/DN 双跑)
    # ==========================================================
    # 测试意图:
    #   隔离纯量化噪声 — quantize=False 作 baseline, 对比 quantize=True
    #   的 pass% / 平均误差 / 余弦相似度.
    #   在 ND/DN 两种布局下各算一组, 验证布局选择不应显著改变量化噪声幅度
    #   (差异应来自 FP 求和顺序, 而非量化误差本身).
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group E | 量化噪声 (FP32 vs MXFP4, 描述性, ND/DN 双跑)")
    print("意图: 两种布局下量化噪声幅度应一致 (差异仅来自 FP 求和顺序)")
    print("=" * 72)

    print("\n--- E.1 非因果, ND/DN 双跑 ---")
    for layout in LAYOUTS:
        fp32_a = flash_attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=False, block_q=32, block_kv=32, s_layout=layout
        )
        mx_a = flash_attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, quantize=True, block_q=32, block_kv=32, s_layout=layout
        )
        _info_cmp(f"  [{layout}] FP32 vs MXFP4 非因果", fp32_a, mx_a, atol=5e-2)

    print("\n--- E.2 因果 + GQA(8/2), ND/DN 双跑 ---")
    fp32_b_by_layout = {}
    for layout in LAYOUTS:
        fp32_b = flash_attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, causal=True, quantize=False, block_q=32, block_kv=32, s_layout=layout
        )
        mx_b = flash_attention_cpu_golden_varlen(
            Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, causal=True, quantize=True, block_q=32, block_kv=32, s_layout=layout
        )
        fp32_b_by_layout[layout] = fp32_b
        _info_cmp(f"  [{layout}] FP32 vs MXFP4 因果+GQA(8/2)", fp32_b, mx_b, atol=5e-2)

    print("\n--- E.3 Decoding (sq=1, sk=10), ND/DN 双跑 ---")
    for layout in LAYOUTS:
        fp32_c = flash_attention_cpu_golden_varlen(
            Q_dec, K_dec, V_dec, [0, 1], [0, 10], causal=True, quantize=False, block_q=1, block_kv=4, s_layout=layout
        )
        mx_c = flash_attention_cpu_golden_varlen(
            Q_dec, K_dec, V_dec, [0, 1], [0, 10], causal=True, quantize=True, block_q=1, block_kv=4, s_layout=layout
        )
        _info_cmp(f"  [{layout}] FP32 vs MXFP4 decoding", fp32_c, mx_c, atol=5e-2)

    print("\n--- E.4 P 量化 + V@seq_k 组合, ND/DN 双跑 ---")
    for layout in LAYOUTS:
        mx_d = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            causal=True,
            quantize=True,
            quantize_p=True,
            v_quant_axis="seq_k",
            block_q=32,
            block_kv=32,
            s_layout=layout,
        )
        _info_cmp(f"  [{layout}] FP32 vs MXFP4(P+V@seq_k) 因果", fp32_b_by_layout[layout], mx_d, atol=1e-1)

    # ==========================================================
    # Group F: s_dtype 配置 (S/m/exp/P 精度)
    # ==========================================================
    # 测试意图:
    #   验证 s_dtype 参数:
    #   (a) "fp32" 输出与现有 baseline 完全一致 (no-op 退化)
    #   (b) 每种 dtype 下 ref ≡ flash (算法不变性, 因为 ref/flash 走的是
    #       同一种 s_dtype 数值路径, 仅相差 FP 求和顺序)
    #   (c) ND ≡ DN 在每种 dtype 下都成立
    #   (d) FP32 vs FP16 / FP32 vs BF16 的精度损失 (描述性)
    #   (e) 非法 dtype 触发 ValueError
    # ==========================================================
    print("\n" + "=" * 72)
    print("Group F | s_dtype 配置 (S/m/exp/P 精度: fp32/fp16/bf16)")
    print("意图: 各 dtype 下 ref ≡ flash, ND ≡ DN; FP32 是 baseline, FP16/BF16 描述性")
    print("=" * 72)

    DTYPES = ["fp32", "fp16", "bf16"]

    print("\n--- F.1 s_dtype='fp32' 与 baseline 一致 ---")
    # 对照: 旧代码路径 (s_dtype 默认 fp32) 在 causal+MXFP4 下的输出
    baseline = flash_attention_cpu_golden_varlen(
        Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, causal=True, quantize=True, block_q=32, block_kv=32
    )
    for layout in LAYOUTS:
        o = flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            causal=True,
            quantize=True,
            block_q=32,
            block_kv=32,
            s_layout=layout,
            s_dtype="fp32",
        )
        if layout == "ND":
            _check(f"  [{layout}] s_dtype=fp32 与默认 baseline 一致", o, baseline, atol=1e-6)
        else:
            # DN vs ND baseline 仅相差 FP 求和顺序
            _check(f"  [{layout}] s_dtype=fp32 vs baseline (ND)", o, baseline, atol=LAYOUT_ATOL)

    print("\n--- F.2 各 s_dtype 下 ref vs flash, ND/DN 双跑 ---")
    for dt in DTYPES:
        # dtype 越低, ref vs flash 的算法允差越松 (FP 求和顺序在 fp16 下放大)
        atol_dt = {"fp32": 1e-5, "bf16": 5e-3, "fp16": 5e-3}[dt]
        for layout in LAYOUTS:
            ref = attention_cpu_golden_varlen(
                Q, K, V, CU_Q, CU_KV, USED_Q, USED_KV, causal=True, quantize=True, s_dtype=dt, s_layout=layout
            )
            fl = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                causal=True,
                quantize=True,
                block_q=32,
                block_kv=32,
                s_dtype=dt,
                s_layout=layout,
            )
            _check(f"  [{layout}] s_dtype={dt} ref vs flash", ref, fl, atol=atol_dt)

    print("\n--- F.3 各 s_dtype 下 ND ≡ DN (flash) ---")
    for dt in DTYPES:
        atol_dt = {"fp32": LAYOUT_ATOL, "bf16": 5e-3, "fp16": 5e-3}[dt]
        outs = {}
        for layout in LAYOUTS:
            o = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                causal=True,
                quantize=True,
                block_q=32,
                block_kv=32,
                s_dtype=dt,
                s_layout=layout,
            )
            outs[layout] = o
        _check(f"  s_dtype={dt} ND vs DN", outs["ND"], outs["DN"], atol=atol_dt)

    print("\n--- F.4 FP32 vs FP16/BF16 噪声 (描述性) ---")
    # 用 D.3 的大 seq 数据, 体现 dtype 引入的精度损失
    cu_big_f = [0, seq_big]
    for layout in LAYOUTS:
        fp32_ref = flash_attention_cpu_golden_varlen(
            Qb_big,
            Kb_big,
            Vb_big,
            cu_big_f,
            cu_big_f,
            causal=True,
            quantize=True,
            block_q=64,
            block_kv=32,
            s_dtype="fp32",
            s_layout=layout,
        )
        for dt in ["fp16", "bf16"]:
            out_dt = flash_attention_cpu_golden_varlen(
                Qb_big,
                Kb_big,
                Vb_big,
                cu_big_f,
                cu_big_f,
                causal=True,
                quantize=True,
                block_q=64,
                block_kv=32,
                s_dtype=dt,
                s_layout=layout,
            )
            _info_cmp(f"  [{layout}] s_dtype=fp32 vs {dt}", fp32_ref, out_dt, atol=5e-3)

    print("\n--- F.5 quantize_p='blockwise' 与 s_dtype 组合 ---")
    # blockwise 路径下 helper 内部完整走 s_dtype, 也要验证 ref ≡ flash
    for dt in DTYPES:
        atol_dt = {"fp32": 1e-5, "bf16": 5e-3, "fp16": 5e-3}[dt]
        for layout in LAYOUTS:
            ref = attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                quantize=True,
                quantize_p=True,
                quantize_p_mode="blockwise",
                s_dtype=dt,
                s_layout=layout,
            )
            fl = flash_attention_cpu_golden_varlen(
                Q,
                K,
                V,
                CU_Q,
                CU_KV,
                USED_Q,
                USED_KV,
                quantize=True,
                quantize_p=True,
                quantize_p_mode="blockwise",
                block_q=64,
                block_kv=64,
                s_dtype=dt,
                s_layout=layout,
            )
            _check(f"  [{layout}] blockwise s_dtype={dt} ref vs flash", ref, fl, atol=atol_dt)

    print("\n--- F.6 非法 s_dtype 触发 ValueError ---")
    raised = False
    try:
        flash_attention_cpu_golden_varlen(
            Q,
            K,
            V,
            CU_Q,
            CU_KV,
            USED_Q,
            USED_KV,
            s_dtype="bogus",
        )
    except ValueError:
        raised = True
    _check_true("F.6 非法 s_dtype 触发 ValueError", raised)

    print("\n" + "=" * 72)
    print("自检完成")
    print("=" * 72)
