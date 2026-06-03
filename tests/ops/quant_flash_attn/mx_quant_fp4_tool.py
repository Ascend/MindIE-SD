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
MXFP4 (Microscaling FP4 E2M1) 完整量化与反量化实现
====================================================
包含: FP4 编码表 | MX 量化/反量化 | 位级打包/解包 | E8M0 缩放因子
支持: Baseline 和 OAS (溢出感知缩放) 两种策略

本版本修复了以下 bug:
  1. FP4 E2M1 编码表中 e=3 错误返回 inf/nan (应为 ±4.0/±6.0)
  2. compute_e8m0_scale_block 未按 OCP MX 规范扣除 emax_elem,
     且禁用了负指数 (scale<1 的 block 被错误量化)
  3. mx_quantize 去 pad 时存在多余的逻辑判断
  4. mx_dequantize_fp4_full 在 last dim 非 block_size 倍数时数据错位
  5. scale 通过 log2 反算并以 E8M0 biased 编码存储 (符合规范)
"""

import torch
from typing import Tuple, Dict

# pylint: disable=redefined-outer-name,not-callable,comparison-with-itself

# ============================================================
# 第一部分: FP4 (E2M1) 格式定义与编码表
# ============================================================
# FP4 E2M1 格式 (S-E-E-M):
#   Normal:    (-1)^S * 2^(E-1) * (1 + M/2)   bias=1
#   Subnormal: (-1)^S * 2^0 * (M/2)            E=0
# 注: OCP MX 规范的 E2M1 没有 inf/NaN, e=3 的两个码点是 ±4.0 / ±6.0
# ============================================================

# 所有 FP4 E2M1 可表示的 FP32 数值 (用于模拟量化)
FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)

# E8M0 偏置 (OCP MX 规范): biased = exp + 127
E8M0_BIAS = 127
# E2M1 元素格式的最大正规化指数: max=6.0=1.5*2^2, 所以 emax_elem=2
FP4_E2M1_EMAX = 2


def generate_fp4_e2m1_encoding_table():
    """
    生成完整的 FP4 E2M1 编码表:
      每个编码 4-bit, 格式: S-E-E-M (1位符号, 2位指数, 1位尾数)

    返回:
        fp32_values:       所有可表示的 FP32 值 (tensor, shape=[16])
        binary_encodings:  对应的 4-bit 编码 (tensor, shape=[16], dtype=uint8)
        fp32_to_encoding:  字典: {fp32_value: encoding}
        encoding_to_fp32:  字典: {encoding: fp32_value}
    """
    encodings = []
    values = []

    for s in [0, 1]:  # 符号位
        for e in range(4):  # 2 位指数 (0-3)
            for m in range(2):  # 1 位尾数 (0-1)
                encoding = (s << 3) | (e << 1) | m  # S-E-E-M
                if e == 0:
                    # 次正规数: (-1)^S * 2^0 * (M/2)
                    val = (-1) ** s * (2.0**0) * (m / 2.0)
                else:
                    # 正规数 (e=1,2,3): (-1)^S * 2^(E-1) * (1 + M/2)
                    # 注: E2M1 没有 inf/NaN, e=3 也是正规数
                    val = (-1) ** s * (2.0 ** (e - 1)) * (1.0 + m / 2.0)

                encodings.append(encoding)
                values.append(val)

    # 构建双向映射字典
    fp32_to_encoding = {}
    encoding_to_fp32 = {}
    for enc, val in zip(encodings, values):
        # 注: Python 中 0.0 == -0.0, 两者哈希相同, 后插入会覆盖前者
        # 这里用 setdefault 让正零的编码 (0b0000) 优先, 保证正零往返为正零
        if val not in fp32_to_encoding:
            fp32_to_encoding[val] = enc
        encoding_to_fp32[enc] = val

    fp32_values = torch.tensor(values, dtype=torch.float32)
    binary_encodings = torch.tensor(encodings, dtype=torch.uint8)

    return fp32_values, binary_encodings, fp32_to_encoding, encoding_to_fp32


# 全局编码表
FP4_VALUES, FP4_ENCODINGS, FP32_TO_FP4, FP4_TO_FP32 = generate_fp4_e2m1_encoding_table()


# ============================================================
# 第二部分: 基础量化/反量化模块 (FP32 模拟)
# ============================================================


def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """
    将张量量化到最近的 FP4 (E2M1) 值, 返回 FP32 模拟表示。

    输入: 任意形状 FP32 张量
    输出: 同形状 FP32 张量 (值是 FP4 可表示的值)
    """
    with torch.no_grad():
        expanded = tensor.unsqueeze(-1)  # [..., 1]
        expanded_abs = torch.abs(expanded - FP4_E2M1_VALUES.to(tensor.device))
        indices = torch.argmin(expanded_abs, dim=-1)
        quantized = FP4_E2M1_VALUES.to(tensor.device)[indices]
    return quantized


def dequantize_fp4(fp4_data: torch.Tensor) -> torch.Tensor:
    """FP4 反量化 (FP32 模拟值 -> FP32)"""
    return fp4_data.float()


def compute_e8m0_scale_block(
    block: torch.Tensor, mode: str = "baseline", emax_elem: int = FP4_E2M1_EMAX
) -> torch.Tensor:
    """
    为一个 block 计算 E8M0 缩放因子 (2 的整数次幂)。

    按 OCP MX 规范:
        scale_exp = floor(log2(max_abs(block))) - emax_elem   (baseline)
        scale_exp = ceil(log2(max_abs(block)))  - emax_elem   (oas, 防饱和)

    参数:
        block:     shape [..., block_size] 的张量
        mode:      "baseline" - floor, 最大化 FP4 范围利用, 允许少量饱和
                   "oas"      - ceil,  保证不饱和, 牺牲一点精度
        emax_elem: 元素格式最大正规化指数, FP4 E2M1 为 2

    返回:
        scale: 标量, 2 的整数次幂, 指数范围 [-127, 127]
    """
    max_abs = torch.max(torch.abs(block))

    # 整块为 0: 用 scale=1 (指数=0) 避免 log2(0)
    if max_abs.item() < 1e-30:
        return torch.tensor(1.0, device=block.device)

    log_max = torch.log2(max_abs)
    if mode == "baseline":
        exponent = torch.floor(log_max) - emax_elem
    elif mode == "oas":
        exponent = torch.ceil(log_max) - emax_elem
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # E8M0 with bias 127: 有效指数范围 [-127, 127]
    exponent = torch.clamp(exponent, -127.0, 127.0)
    return 2.0**exponent


def mx_quantize(
    tensor: torch.Tensor, block_size: int = 32, mode: str = "baseline"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MXFP4 量化 (FP32 模拟版本, 输出仍为 FP32)。

    将输入张量沿最后一维分组, 每组 block_size 个元素共享一个 E8M0 缩放因子,
    再各自独立量化至 FP4 (E2M1)。

    参数:
        tensor:     输入张量, 任意形状
        block_size: 分组大小, MX 规范固定为 32
        mode:       缩放因子计算策略 ("baseline" 或 "oas")

    返回:
        fp4_vals:    量化后的 FP32 模拟值, 形状同输入
        e8m0_scales: 每组共享的缩放因子, shape=[num_blocks]
    """
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)

    tensor_last = tensor.shape[-1]
    pad_size = 0
    if tensor_last % block_size != 0:
        pad_size = block_size - (tensor_last % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0.0)

    reshaped = tensor.reshape(-1, block_size)
    num_blocks = reshaped.shape[0]

    scales = torch.empty(num_blocks, device=tensor.device)
    fp4_quant = torch.empty_like(reshaped)

    for i in range(num_blocks):
        block = reshaped[i]
        scale = compute_e8m0_scale_block(block, mode=mode)
        scales[i] = scale
        normalized = block / scale
        fp4_quant[i] = quantize_fp4(normalized)

    fp4_quant = fp4_quant.view(tensor.shape)

    # 去除填充 (只要 pad_size > 0 就应当裁剪)
    if pad_size > 0:
        fp4_quant = fp4_quant[..., :-pad_size]

    return fp4_quant, scales


def mx_dequantize(fp4_vals: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """
    MXFP4 反量化 (从 FP32 模拟值恢复)。

    参数:
        fp4_vals:   量化后的 FP32 模拟值
        scales:     缩放因子, shape=[num_blocks]
        block_size: 分组大小

    返回:
        反量化后的 FP32 张量
    """
    original_shape = fp4_vals.shape

    # 若 last dim 非 block_size 倍数, 先 pad 再 unpad
    last_dim = original_shape[-1] if fp4_vals.dim() > 0 else 1
    pad_size = 0
    if fp4_vals.dim() > 0 and last_dim % block_size != 0:
        pad_size = block_size - (last_dim % block_size)
        fp4_vals = torch.nn.functional.pad(fp4_vals, (0, pad_size), value=0.0)

    reshaped = fp4_vals.reshape(-1, block_size)
    dequant = reshaped * scales.unsqueeze(1).to(reshaped.device)
    dequant = dequant.view(fp4_vals.shape)

    if pad_size > 0:
        dequant = dequant[..., :last_dim]

    return dequant


# ============================================================
# 第三部分: FP4 编码转换与位级打包 (真实 4-bit 存储)
# ============================================================


def convert_fp32_simulated_to_fp4_encoding(fp32_simulated: torch.Tensor) -> torch.Tensor:
    """
    将量化后的 FP32 模拟值转换为真实的 4-bit 二进制编码 (0-15)。

    输入: FP32 张量 (值必须是 FP4 可表示的, 如 1.0, 0.5, -0.5 等)
    输出: uint8 张量, 每个元素存储 0-15 的 4-bit 编码

    注: E2M1 没有 inf/NaN, 因此无需特殊值处理。
    """
    flat = fp32_simulated.flatten().cpu()
    encodings = torch.zeros_like(flat, dtype=torch.uint8)

    for fp32_val, encoding in FP32_TO_FP4.items():
        mask = flat == fp32_val
        encodings[mask] = encoding

    return encodings.to(fp32_simulated.device).reshape(fp32_simulated.shape)


def pack_fp4_to_uint8(fp4_encodings: torch.Tensor) -> torch.Tensor:
    """
    将 4-bit FP4 编码打包成 uint8 (每字节存储两个 FP4 值)。
    低 4 位存前者, 高 4 位存后者 (小端序)。

    输入: [..., block_size] uint8 张量 (每个元素 0-15)
    输出: [..., block_size//2] uint8 张量
    """
    shape = fp4_encodings.shape
    if shape[-1] % 2 != 0:
        raise ValueError(f"最后一维必须是偶数, 当前为 {shape[-1]}")

    reshaped = fp4_encodings.reshape(-1, 2)
    packed = (reshaped[:, 1] << 4) | (reshaped[:, 0] & 0x0F)

    return packed.reshape(*shape[:-1], shape[-1] // 2)


def pack_mxfp4_full(fp4_simulated: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> Dict:
    """
    完整的 MXFP4 打包: 将 FP32 模拟值和 scale 打包为紧凑字节格式。

    参数:
        fp4_simulated: mx_quantize() 输出的 FP32 模拟值
        scales:        mx_quantize() 输出的缩放因子 (2^exp)
        block_size:    分组大小

    返回打包字典:
        'fp4_data':       打包后的 uint8 数据
        'scales':         E8M0 biased 指数 (uint8, 已加 bias=127)
        'block_size':     int
        'original_shape': tuple (未填充的原始形状)
    """
    # Step 1: 转换为 4-bit 编码
    fp4_encodings = convert_fp32_simulated_to_fp4_encoding(fp4_simulated)

    # 确保最后一维是 block_size 的整数倍
    last_dim = fp4_encodings.shape[-1]
    if last_dim % block_size != 0:
        pad_size = block_size - (last_dim % block_size)
        fp4_encodings = torch.nn.functional.pad(fp4_encodings, (0, pad_size), mode='constant', value=0)

    # Step 2: 重排为 [num_blocks, block_size]
    flat_encodings = fp4_encodings.reshape(-1, block_size)

    # Step 3: 打包 FP4 数据 (32 个值 -> 16 字节)
    packed_data = pack_fp4_to_uint8(flat_encodings)

    # Step 4: 将 scales 转换为 E8M0 biased 指数 (bias=127)
    #   scales 必为 2 的整数次幂, log2 后 round 即可得到精确整数指数
    scale_exps_int = torch.log2(scales).round().long()
    scale_exponents = (scale_exps_int + E8M0_BIAS).clamp(0, 254).to(torch.uint8)

    return {
        'fp4_data': packed_data,
        'scales': scale_exponents,
        'block_size': block_size,
        'original_shape': tuple(fp4_simulated.shape),
    }


def unpack_uint8_to_fp4(packed: torch.Tensor) -> torch.Tensor:
    """
    从打包的 uint8 恢复 4-bit FP4 编码。

    输入: [..., packed_size] uint8 张量
    输出: [..., packed_size*2] uint8 张量 (每个元素 0-15)
    """
    shape = packed.shape
    flat = packed.flatten()
    unpacked = torch.zeros(len(flat) * 2, dtype=torch.uint8, device=packed.device)

    unpacked[0::2] = flat & 0x0F  # 低 4 位
    unpacked[1::2] = (flat >> 4) & 0x0F  # 高 4 位

    return unpacked.reshape(*shape[:-1], shape[-1] * 2)


def unpack_mxfp4_full(packed_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从打包字典解包, 恢复为 FP32 模拟值 ([num_blocks, block_size]) 和 scales。

    返回: (fp32_values_blocks, scales)
        fp32_values_blocks: shape [num_blocks, block_size], 尚未乘以 scale
        scales:              shape [num_blocks], 真实 scale 值 (2^exp)
    """
    # Step 1: 解包 FP4 编码 -> shape [num_blocks, block_size]
    fp4_encodings = unpack_uint8_to_fp4(packed_dict['fp4_data'])

    # Step 2: 将编码映射回 FP32 数值
    fp32_vals = torch.zeros_like(fp4_encodings, dtype=torch.float32)
    for encoding, fp32_val in FP4_TO_FP32.items():
        mask = fp4_encodings == encoding
        fp32_vals[mask] = fp32_val

    # Step 3: 从 E8M0 biased 指数恢复 scale
    scales = 2.0 ** (packed_dict['scales'].float() - E8M0_BIAS)

    return fp32_vals, scales


# ============================================================
# 第四部分: 完整的端到端 MXFP4 量化/反量化接口
# ============================================================


def mx_quantize_fp4_full(tensor: torch.Tensor, block_size: int = 32, mode: str = "baseline") -> Dict:
    """
    端到端 MXFP4 量化: FP32 输入 -> 打包的紧凑字节格式。

    参数:
        tensor:     输入 FP32 张量
        block_size: 分组大小 (默认 32)
        mode:       缩放策略 ("baseline" 或 "oas")

    返回:
        打包字典 (可直接存储或传输)
    """
    fp4_simulated, scales = mx_quantize(tensor, block_size, mode)
    return pack_mxfp4_full(fp4_simulated, scales, block_size)


def mx_dequantize_fp4_full(packed_dict: Dict) -> torch.Tensor:
    """
    端到端 MXFP4 反量化: 打包字节格式 -> FP32 张量。

    正确处理 last dim 非 block_size 倍数的情况:
      1. 先按 [num_blocks, block_size] 形状乘以 scale
      2. reshape 回 padded 形状
      3. 沿最后一维裁剪到 original_shape

    参数:
        packed_dict: mx_quantize_fp4_full() 返回的字典

    返回:
        反量化后的 FP32 张量, 形状与原输入一致
    """
    fp32_blocks, scales = unpack_mxfp4_full(packed_dict)
    # fp32_blocks shape: [num_blocks, block_size]

    block_size = packed_dict['block_size']
    original_shape = packed_dict['original_shape']

    # Step 1: 每 block 乘以对应的 scale
    dequant_flat = fp32_blocks * scales.unsqueeze(1).to(fp32_blocks.device)

    # Step 2: 计算 padded shape, reshape 回去
    if len(original_shape) == 0:
        # 标量: 直接取第一个元素
        return dequant_flat.flatten()[0]

    last_dim = original_shape[-1]
    if last_dim % block_size != 0:
        pad_size = block_size - (last_dim % block_size)
    else:
        pad_size = 0
    padded_last = last_dim + pad_size

    padded_shape = tuple(original_shape[:-1]) + (padded_last,)
    dequant = dequant_flat.reshape(padded_shape)

    # Step 3: 沿最后一维裁剪 pad
    if pad_size > 0:
        dequant = dequant[..., :last_dim]

    return dequant


def mxfp4_quantize_pack_last(
    tensor: torch.Tensor, quant_axis: int, block_size: int = 32, mode: str = "baseline"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    打包轴固定为最后一维, 量化轴任意 (可与打包轴同轴).

    参数:
        tensor:     任意形状 FP32 张量, 最后一维必须为偶数 (打包要求)
        quant_axis: 量化轴 (支持负索引). 若 == 最后一维, 行为等价于标准 OCP MXFP4.
        block_size: MX 分块大小 (规范固定 32)
        mode:       "baseline" (floor) 或 "oas" (ceil)

    返回:
        packed_fp4:  uint8, shape = 输入 shape, 但 last 维 = last // 2
                     OCP MX nibble 序: low = elem[2k], high = elem[2k+1]
        e8m0_scales: uint8 (biased, +127), shape = 输入 shape, 但 quant_axis 维
                     = ceil(quant_dim / block_size)
    """
    nd = tensor.dim()
    qax = quant_axis % nd
    last_dim = tensor.shape[-1]
    quant_dim = tensor.shape[qax]
    assert last_dim % 2 == 0, f"最后一维需为偶数 (打包要求), 当前 {last_dim}"

    # 量化阶段: qax 搬到 last (若 qax 已是 last, movedim 是 no-op)
    t_q_last = tensor.movedim(qax, -1).contiguous()
    fp4_sim, scales_flat = mx_quantize(t_q_last, block_size=block_size, mode=mode)

    # FP32 simulated -> 4-bit 编码 (逐元素, 形状不变)
    fp4_codes_q_last = convert_fp32_simulated_to_fp4_encoding(fp4_sim)

    # qax 搬回原位置 (同轴时 no-op)
    fp4_codes = fp4_codes_q_last.movedim(-1, qax).contiguous()

    # 打包阶段: 沿 last 打包 (同轴时, last 就是 quant 轴, 即标准 OCP MXFP4)
    packed_fp4 = pack_fp4_to_uint8(fp4_codes)

    # scales 还原 shape: 1D -> 原 shape 但 qax 维 = n_blk_q
    n_blk_q = (quant_dim + block_size - 1) // block_size
    rest_q_shape = list(t_q_last.shape[:-1])
    scales_q_last = scales_flat.reshape(*rest_q_shape, n_blk_q)
    scales_real = scales_q_last.movedim(-1, qax).contiguous()

    # 真实 scale (2^exp) -> E8M0 biased uint8
    scale_exp_int = torch.log2(scales_real).round().long()
    e8m0_scales = (scale_exp_int + E8M0_BIAS).clamp(0, 254).to(torch.uint8)

    return packed_fp4, e8m0_scales


# ============================================================
# 第五部分: 推理辅助函数
# ============================================================


def linear_mxfp4(x: torch.Tensor, w: torch.Tensor, block_size: int = 32, mode: str = "baseline") -> torch.Tensor:
    """
    使用 MXFP4 量化权重进行线性层推理 (模拟)。

    参数:
        x: 输入激活值 (FP32)
        w: 权重 (FP32)
        block_size: 分组大小
        mode: 量化策略

    返回:
        输出 = x @ dequantized(w)
    """
    qw, sw = mx_quantize(w, block_size, mode)
    with torch.no_grad():
        dqw = mx_dequantize(qw, sw, block_size)
    return x @ dqw


# ============================================================
# 第六部分: 测试与演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MXFP4 (E2M1) 完整量化与反量化测试")
    print("=" * 60)

    def _check(name, a, b, atol=1e-6, rtol=1e-5):
        """对不变式做 allclose 断言, 打印 max diff."""
        diff = (a - b).abs().max().item() if a.numel() > 0 else 0.0
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: max diff {diff:.3e}")
        assert ok, f"{name}: max diff {diff:.3e} > atol={atol:.0e}"

    def _check_true(name, cond):
        ok = bool(cond)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {cond}")
        assert ok, f"{name} failed"

    def _info(name, value):
        """仅打印, 不断言 (量化误差/描述性指标)."""
        print(f"  [info] {name}: {value}")

    # 测试数据
    torch.manual_seed(42)
    w = torch.randn(128, 256, dtype=torch.float32)

    # ========================================
    # 测试 1: 基础量化精度
    # ========================================
    print("\n--- 测试 1: 基础量化 (FP32 模拟) ---")
    q, s = mx_quantize(w, block_size=32, mode="baseline")
    dq = mx_dequantize(q, s, block_size=32)

    error = torch.abs(w - dq).mean().item()
    max_error = torch.abs(w - dq).max().item()
    cos_sim = torch.nn.functional.cosine_similarity(w.flatten(), dq.flatten(), dim=0)

    _info("Baseline 平均绝对误差", f"{error:.6f}")
    _info("Baseline 最大绝对误差", f"{max_error:.6f}")
    _info("Baseline 余弦相似度", f"{cos_sim.item():.6f}")

    # OAS 模式
    q_oas, s_oas = mx_quantize(w, block_size=32, mode="oas")
    dq_oas = mx_dequantize(q_oas, s_oas, block_size=32)
    error_oas = torch.abs(w - dq_oas).mean().item()
    _info("OAS      平均绝对误差", f"{error_oas:.6f}")

    # ========================================
    # 测试 2: 推理模拟
    # ========================================
    print("\n--- 测试 2: 线性层推理模拟 ---")
    x = torch.randn(16, 128, dtype=torch.float32)

    output_fp32 = x @ w
    output_mx = linear_mxfp4(x, w, mode="baseline")
    output_mx_oas = linear_mxfp4(x, w, mode="oas")

    cos_sim_out = torch.nn.functional.cosine_similarity(output_fp32.flatten(), output_mx.flatten(), dim=0)
    cos_sim_out_oas = torch.nn.functional.cosine_similarity(output_fp32.flatten(), output_mx_oas.flatten(), dim=0)

    _info("Baseline (推理) 余弦相似度", f"{cos_sim_out.item():.6f}")
    _info("OAS      (推理) 余弦相似度", f"{cos_sim_out_oas.item():.6f}")

    # ========================================
    # 测试 3: 完整打包流程
    # ========================================
    print("\n--- 测试 3: 完整打包流程 (FP32 -> FP4 字节) ---")

    packed = mx_quantize_fp4_full(w, block_size=32, mode="baseline")

    # 统计存储大小
    original_bytes = w.numel() * 4
    actual_compressed = w.numel() * 0.5 + (w.numel() // 32)

    print(f"  原始大小 (FP32):       {original_bytes:,} bytes")
    print(f"  压缩后大小 (MXFP4):    {actual_compressed:,.0f} bytes")
    print(f"  压缩比:                {actual_compressed / original_bytes:.2%}")
    print(f"  打包数据形状:          {packed['fp4_data'].shape}")
    print(f"  Scale 形状:            {packed['scales'].shape}")
    print(f"  每 block 字节数:       {32 * 0.5 + 1:.1f} (16 FP4 + 1 scale)")

    # 解包恢复
    dq_packed = mx_dequantize_fp4_full(packed)
    error_packed = torch.abs(w - dq_packed).mean().item()
    _info("打包后恢复误差 (量化噪声)", f"{error_packed:.6f}")

    # 打包路径 vs 模拟路径 应严格相等
    _check("打包路径 ≡ 模拟路径", dq, dq_packed, atol=0.0, rtol=0.0)

    # ========================================
    # 测试 4: FP4 编码验证
    # ========================================
    print("\n--- 测试 4: FP4 编码表验证 ---")
    print(f"  FP4 可表示值数量: {len(FP4_VALUES)}")
    print(f"  FP32->FP4 映射数量: {len(FP32_TO_FP4)}")
    print(f"  FP4->FP32 映射数量: {len(FP4_TO_FP32)}")

    # 验证不含 inf/NaN
    has_special = any((v != v) or (v == float('inf')) or (v == float('-inf')) for v in FP4_TO_FP32.values())
    _check_true("编码表不含 inf/NaN", not has_special)

    # 验证小张量
    test_tensor = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0, 2.0, -6.0, 4.0])
    enc = convert_fp32_simulated_to_fp4_encoding(test_tensor)
    print(f"\n  测试值: {test_tensor.tolist()}")
    print(f"  编码:   {[bin(e.item())[2:].zfill(4) for e in enc]}")

    # 全值往返验证
    test_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    enc_all = convert_fp32_simulated_to_fp4_encoding(test_vals)
    recovered = torch.zeros_like(test_vals)
    for i, e in enumerate(enc_all):
        recovered[i] = FP4_TO_FP32[e.item()]

    _check("全 16 值往返", test_vals, recovered, atol=0.0, rtol=0.0)
    _info("恢复值", f"{recovered.tolist()}")

    # ========================================
    # 测试 5: 任意形状张量支持
    # ========================================
    print("\n--- 测试 5: 不同形状张量支持 ---")
    shapes = [(32,), (64, 32), (16, 128, 64), (3, 50), (7, 33)]  # 后两个 last dim 非 32 倍数
    for shape in shapes:
        t = torch.randn(shape)
        q, s = mx_quantize(t, block_size=32)
        dq = mx_dequantize(q, s, block_size=32)
        err_sim = torch.abs(t - dq).mean().item()

        # 端到端打包路径
        packed = mx_quantize_fp4_full(t, block_size=32)
        dq_pk = mx_dequantize_fp4_full(packed)
        err_pk = torch.abs(t - dq_pk).mean().item()
        _check(f"形状 {list(shape)} 打包路径 ≡ 模拟路径", dq, dq_pk, atol=0.0, rtol=0.0)
        _info(f"形状 {list(shape)} 量化误差", f"sim={err_sim:.6f}, pack={err_pk:.6f}")

    # ========================================
    # 测试 6: 小值 (scale<1) 场景
    # ========================================
    print("\n--- 测试 6: 小值张量 (验证负指数支持) ---")
    w_small = torch.randn(64, 64) * 0.01  # 数量级 ~0.01
    q_s, s_s = mx_quantize(w_small, block_size=32)
    dq_s = mx_dequantize(q_s, s_s, block_size=32)
    err_small = torch.abs(w_small - dq_s).mean().item()
    rel_err = err_small / w_small.abs().mean().item()
    _info("输入 abs 均值", f"{w_small.abs().mean().item():.6f}")
    _info("反量化绝对误差", f"{err_small:.6f}")
    _info("相对误差", f"{rel_err:.2%}")

    # ========================================
    # 测试 7: mxfp4_quantize_pack_last (量化轴任意, 打包轴=last)
    # ========================================
    print("\n--- 测试 7: mxfp4_quantize_pack_last ---")

    def _dequant_from_packed(packed_fp4, e8m0_scales, quant_axis, orig_shape, block_size=32):
        """unpack + nibble->FP4 codes -> FP32 -> 沿 quant_axis 乘 scale, 还原 layout."""
        nd = len(orig_shape)
        qax = quant_axis % nd

        # last 维 unpack: [..., L/2] -> [..., L]
        codes = unpack_uint8_to_fp4(packed_fp4)

        # FP4 codes -> FP32 simulated
        fp4_sim = torch.zeros_like(codes, dtype=torch.float32)
        for enc, val in FP4_TO_FP32.items():
            fp4_sim[codes == enc] = val

        # E8M0 biased -> 真实 scale (位精确)
        scales_real = 2.0 ** (e8m0_scales.float() - E8M0_BIAS)

        # 沿 quant_axis 把 scale 广播到 block_size, 处理 pad
        fp4_sim_q_last = fp4_sim.movedim(qax, -1).contiguous()
        scales_q_last = scales_real.movedim(qax, -1).contiguous()
        quant_dim = orig_shape[qax]
        pad_size = (block_size - quant_dim % block_size) % block_size
        if pad_size > 0:
            fp4_sim_q_last = torch.nn.functional.pad(fp4_sim_q_last, (0, pad_size))
        scales_expanded = scales_q_last.repeat_interleave(block_size, dim=-1)
        dequant_padded = fp4_sim_q_last * scales_expanded
        if pad_size > 0:
            dequant_padded = dequant_padded[..., :quant_dim]
        return dequant_padded.movedim(-1, qax).contiguous()

    def _ref_quant_dequant_along(t, qax, block_size=32, mode="baseline"):
        """参考路径: movedim 量化轴到 last, 标准 MXFP4, 搬回."""
        t_q_last = t.movedim(qax, -1).contiguous()
        ref_q, ref_s = mx_quantize(t_q_last, block_size=block_size, mode=mode)
        ref_dq = mx_dequantize(ref_q, ref_s, block_size=block_size)
        return ref_dq.movedim(-1, qax).contiguous()

    # 同轴: qax=-1 (等价于标准 OCP MXFP4)
    t1 = torch.randn(4, 8, 32, 64)
    packed1, scales1 = mxfp4_quantize_pack_last(t1, quant_axis=-1)
    print(f"  [shape] 同轴 (qax=-1): in={list(t1.shape)} packed={list(packed1.shape)} scales={list(scales1.shape)}")
    ref1 = _ref_quant_dequant_along(t1, -1)
    new1 = _dequant_from_packed(packed1, scales1, -1, t1.shape)
    _check("同轴 (qax=-1) 反量化 ≡ 参考路径", ref1, new1, atol=0.0, rtol=0.0)

    # 异轴: qax=2 (BNSD 沿 S 量化, 沿 D 打包), S=33 非 32 倍数
    t2 = torch.randn(4, 8, 33, 64)
    packed2, scales2 = mxfp4_quantize_pack_last(t2, quant_axis=2)
    print(
        f"  [shape] 异轴 (qax=2,  S=33): in={list(t2.shape)} packed={list(packed2.shape)} scales={list(scales2.shape)}"
    )
    ref2 = _ref_quant_dequant_along(t2, 2)
    new2 = _dequant_from_packed(packed2, scales2, 2, t2.shape)
    _check("异轴 (qax=2) 反量化 ≡ movedim 参考路径", ref2, new2, atol=0.0, rtol=0.0)

    # 异轴: qax=0 (沿首轴量化)
    t3 = torch.randn(64, 16, 32)
    packed3, scales3 = mxfp4_quantize_pack_last(t3, quant_axis=0)
    print(f"  [shape] 异轴 (qax=0): in={list(t3.shape)} packed={list(packed3.shape)} scales={list(scales3.shape)}")
    ref3 = _ref_quant_dequant_along(t3, 0)
    new3 = _dequant_from_packed(packed3, scales3, 0, t3.shape)
    _check("异轴 (qax=0) 反量化 ≡ movedim 参考路径", ref3, new3, atol=0.0, rtol=0.0)

    # OAS 模式 + 异轴
    t4 = torch.randn(2, 100, 16) * 0.5
    packed4, scales4 = mxfp4_quantize_pack_last(t4, quant_axis=1, mode="oas")
    print(f"  [shape] OAS 异轴 (qax=1): in={list(t4.shape)} packed={list(packed4.shape)} scales={list(scales4.shape)}")
    ref4 = _ref_quant_dequant_along(t4, 1, mode="oas")
    new4 = _dequant_from_packed(packed4, scales4, 1, t4.shape)
    _check("OAS 异轴 (qax=1) 反量化 ≡ movedim 参考路径", ref4, new4, atol=0.0, rtol=0.0)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
