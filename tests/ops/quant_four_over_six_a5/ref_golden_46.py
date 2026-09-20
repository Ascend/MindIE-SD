#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
Golden reference for dynamic_mx_quant dst_type_max=4 (per-block 4/6 adaptive) — HOST model.

目的
----
为 dst_type_max=4 的 per-block 4/6 自适应量化提供一个独立、数值正确的主机端参考实现，
用来 (1) 在 host 上验证算法本身的精度，(2) 作为 NPU kernel 修复后逐 block 比对的 oracle。

忠实建模的 kernel 数据通路 (dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis.h):
  - ComputeMaxExpDynamicDtypeRange : 每 block  M = max|BF16(x)|  (Cast<half->bf16,CAST_RINT>, 保留尾数)
  - ComputeScaleDdr46              : 由 M + addValue 派生 scaleE8M0(E8M0) + recipScale(BF16)
                                      D=6 -> addValue=0x003f ; D=4 -> addValue=0x007f
  - ComputeVarForCand46            : 每 block  e = x - dequant  的 sumE / sumESq  (原始域, 非归一化域)
                                      dequant = fp4(x*recip) * 2^(scaleE8M0-127)
  - CompareSelect46                : better4 = (var4 < var6)，选 var 小者输出 mxScale/recipScale

精度验证 oracle
---------------
dst_type_max=6 (单一 D=6 候选, NPU 上已验证全对) 必须在标准输入 ROWS=1 COLS=128 下复现
mxScale = [127, 128, 129, 130]。该检查独立于 FP4 var 选优，专门校验
输入建模 + BF16 max + ComputeScaleDdr46(D=6) 的忠实度。

CLI / 环境变量与 NPU 测试 (examples/arch35/test_aclnn_dynamic_mx_quant_fp4_adaptive.cpp) 一致:
  ROWS, COLS, MODE (varied|uniform|rev), DST_TYPE_MAX (4.0 | 6.0)

参考文献:
  - tail_axis.h: ComputeMaxExpDynamicDtypeRange(395-453) ComputeScaleDdr46(747-796)
                 ComputeVarForCand46(807-875) CompareSelect46(882-922)
  - common.h: 常量 MAX_EXP_FOR_BF16=0x7f80 FP4_E2M1_BF16_MAX_EXP=0x0100 等
"""

import os
import sys
import math
import struct
import numpy as np

# ===========================================================================
# IEEE754 位运算助手 (精确, 不依赖 numpy 取整语义)
# ===========================================================================
def f32_bits(v):
    """python/numpy float -> IEEE754 float32 位 (uint32)。"""
    return struct.unpack('<I', struct.pack('<f', np.float32(v)))[0]

def bits_to_f32(u):
    """IEEE754 float32 位 -> python float。"""
    return float(struct.unpack('<f', struct.pack('<I', u & 0xffffffff))[0])

def f2h(v):
    """float -> IEEE754 float16 位 (截断 truncation, 非 RNE), 与测试 FloatToHalfBits 逐行一致。"""
    f = f32_bits(v)
    sign = (f >> 31) & 0x1
    exp = ((f >> 23) & 0xff) - 127
    man = f & 0x7fffff
    if exp >= 16:
        h = (sign << 15) | 0x7c00                       # inf/sat
    elif exp >= -14:
        h = (sign << 15) | ((exp + 15) << 10) | (man >> 13)
    elif exp >= -24:
        m = man | 0x800000
        sh = (-14 - exp)
        nm = (m >> sh) if sh < 32 else 0
        h = (sign << 15) | (nm >> 13)
    else:
        h = (sign << 15)
    return np.uint16(h)

def h2f(h):
    """IEEE754 float16 位 -> python float (精确, FP16 ⊂ float64)。

    正规/次正规/Inf/NaN 全覆盖 (旧 bit-shift 版在次正规上负移位崩溃, 已弃用)。
    """
    h = int(h)
    sign = (h >> 15) & 0x1
    exp = (h >> 10) & 0x1f
    man = h & 0x3ff
    if exp == 0x1f:                                   # Inf / NaN
        val = float('inf') if man == 0 else float('nan')
    elif exp == 0:                                    # 零 / 次正规
        val = man * (2.0 ** -24)                       # 次正规 = man * 2^-24
    else:                                             # 正规
        val = (1.0 + man / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val

def f2bf16_rne(v):
    """float -> BF16 位, ties-to-even (镜像 CAST_RINT)。
    u + 0x7FFF + lsb: 中点 (0x8000) 时 bit[16] 为偶不进, 为奇进 → ties-to-even。
    """
    u = f32_bits(v)
    lsb = (u >> 16) & 1
    return ((u + 0x7fff + lsb) >> 16) & 0xFFFF

def bf16_to_float(b):
    """BF16 位 -> float (位 reinterpret: 把 16 位放 float32 高 16 位)。"""
    return bits_to_f32((int(b) & 0xFFFF) << 16)

# ===========================================================================
# 常量 (镜像 dynamic_mx_quant_common.h)
# ===========================================================================
MAX_EXP_FOR_BF16        = 0x7f80   # BF16 指数域掩码 (亦 = BF16 INF 指数模式)
FP4_E2M1_BF16_MAX_EXP   = 0x0100   # clamp 值
EXP_BF16_BIAS           = 0x7f00
SHR_NUM_FOR_BF16        = 7
MAX_EXP_FOR_FP8         = 0x00ff   # E8M0 的 inf/nan 特判值
NAN_CUSTOMIZATION       = 0x7f81
SPECIAL_EXP_THRESHOLD   = 0x0040
ADD_VALUE_FOR_BF16_MAN1 = 0x003f   # D=6 ceil
ADD_VALUE_FOR_BF16_D4   = 0x007f   # D=4 ceil
# round-to-nearest 常量(替代 ceil,使 D6 有时饱和 → D4 有机会胜出)
ADD_VALUE_FOR_BF16_D6_ROUND = 0x0078   # D=6 round: carry @ mant≥8
ADD_VALUE_FOR_BF16_D4_ROUND = 0x004A   # D=4 round: carry @ mant≥54
SUB_OFFSET_FOR_D6_ROUND     = 0x0180   # round(log2(6))=3
SUB_OFFSET_FOR_D4_ROUND     = 0x0100   # round(log2(4))=2
BLOCKSIZE               = 32
N_BLOCK                 = 32       # 每 block 元素数

# ===========================================================================
# ComputeScaleDdr46 的逐元素复刻 (单个 block 的 M)
# ===========================================================================
def kernel_scale(M_bits, addValue, subOffset=FP4_E2M1_BF16_MAX_EXP):
    """复刻 ComputeScaleDdr46 (tail_axis.h)。

    入参: M_bits = |x|max 的 BF16 位 (符号已清, 保留尾数); addValue; subOffset(ceil=0x0100, round D=6=0x0180)。
    返回: (scaleE8M0, recip_bits)。
    """
    vdMaxExpOnly = M_bits & MAX_EXP_FOR_BF16
    cmp = (vdMaxExpOnly != MAX_EXP_FOR_BF16)             # True = 有效 (非 inf/nan)
    invalid = (vdMaxExpOnly < FP4_E2M1_BF16_MAX_EXP)     # True = 过小 -> clamp (检查仍用 0x0100)
    vdAdd = ((M_bits + addValue) & 0xFFFF) & MAX_EXP_FOR_BF16
    vdAdd = subOffset if invalid else vdAdd               # ★ clamp 用 subOffset(使 sharedExp=0)
    sharedExp = (vdAdd - subOffset) & 0xFFFF              # ★ subtract 用 subOffset
    scaleE8M0 = (sharedExp >> SHR_NUM_FOR_BF16) & 0xFF
    scaleE8M0 = scaleE8M0 if cmp else MAX_EXP_FOR_FP8
    # recipScale (BF16 halfScale)
    zeroMask = (sharedExp != 0)
    special = (sharedExp == EXP_BF16_BIAS)
    half = (EXP_BF16_BIAS - sharedExp) & 0xFFFF
    half = half if cmp else NAN_CUSTOMIZATION
    half = half if zeroMask else 0
    half = SPECIAL_EXP_THRESHOLD if special else half
    return scaleE8M0, half

def derive_d4_from_d6(M_bits, scale6, recip6):
    """kernel 1-bit 决策: 由 D6 候选推导 D4 scale/recip (匹配 ComputeScaleDdr46 尾段)。

    kernel:
      decision = (m < 8) OR (m >= 54)          // 1-bit
      scale4   = scale6 + decision              // E8M0 Add
      recip4   = Sub(recip6, decision << 7)     // bf16 Sub
    """
    m = M_bits & 0x007f
    decision = 1 if (m < 8) or (m >= 54) else 0
    scale4 = (scale6 + decision) & 0xFF
    recip4 = (recip6 - (decision << 7)) & 0xFFFF
    return scale4, recip4

# ===========================================================================
# FP4 E2M1 量化 (var 计算用 Cast<U,float> 的等价模型)
# ===========================================================================
FP4_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def fp4_quant(y, mode="rint"):
    """float -> FP4 E2M1 量化值: 就近网格, ties-to-even, clamp ±6。

    Ascend 硬件 Cast<fp4x2_e2m1_t,bf16> 对于 FP4 E2M1 (单尾数位) 表现为 ties-to-even。
    CAST_ROUND 的理论 ties-away 行为在 FP4 的 1-bit 尾数场景下退化为 ties-to-even。
    """
    a = abs(float(y))
    if mode == "floor":
        # 实测（真 kernel rm）：kernel floor = 向 -inf 整档后取网格（3.55→3.0、5.4→4.0、-4.78→-4.0）
        yv = float(y)
        cands = [0.0] + [g for g in FP4_GRID[1:]] + [-g for g in FP4_GRID[1:]]
        best = None
        for g in cands:
            if g <= yv + 1e-9 and (best is None or g > best):
                best = g
        return 0.0 if yv == 0.0 else best
    if mode == "round":
        # 实测（真 kernel 传 rm）：CAST_ROUND ties 为 ties-away（v=±5.0 tie→6.0）
        best = FP4_GRID[0]
        bd = abs(a - best)
        for g in FP4_GRID[1:]:
            d = abs(a - g)
            if d < bd:
                best, bd = g, d
            elif d == bd:
                best = g                          # ties → away
        return 0.0 if y == 0.0 else math.copysign(best, y)
    FE = {0.0, 1.0, 2.0, 4.0}
    best = FP4_GRID[0]
    bd = abs(a - best)
    for g in FP4_GRID[1:]:
        d = abs(a - g)
        if d < bd:
            best, bd = g, d
        elif d == bd and g in FE and best not in FE:
            best = g                              # ties → even (rint)
    if y == 0.0:
        return 0.0
    return math.copysign(best, y)

# ===========================================================================
# 每 block 的计算
# ===========================================================================
def block_M_bits(xf):
    """M = max|BF16(x)| 的 BF16 位 (镜像 ComputeMaxExpDynamicDtypeRange)。

    half Inf/NaN (FP16 exp field 0x7c00) 经 kernel invalidDataMask Select 置为
    MAX_EXP_FOR_BF16=0x7f80 (会主导 block max); 此处用 isfinite 等价复刻。
    """
    mb = 0
    for v in xf:
        if not math.isfinite(v):
            b = MAX_EXP_FOR_BF16                  # Inf/NaN -> 0x7f80
        else:
            b = f2bf16_rne(v) & 0x7fff            # |.|: 清符号, 保留指数+尾数
        if b > mb:
            mb = b
    return mb

def candidate_sumsq(xf, recip_bits, scaleE8M0, mode="rint"):
    """复刻 QuantVarCand46 的 per-block Σe² (缩放域, bf16→half 精度), 供 Σe² 选优。

    Kernel QuantVarCand46 数据通路:
      v_bf16 = bf16(x * recip)     # Mul: 两个 bf16 → bf16 RNE
      q      = fp4(v_bf16)         # Cast<U,bf16>: 量化到 FP4 网格 {0,±0.5,±1,±1.5,±2,±3,±4,±6}
      q_bf16 = bf16(q)             # Cast<bf16,U>: FP4 码点 → bf16 (bf16 RNE)
      e_bf16 = bf16(v - q_bf16)    # Sub: 缩放域误差 bf16 RNE
      e_half = half(e_bf16)        # Cast<half,bf16>: bf16 → half
      Σe²    = 配对(e_h² + e_h²) + 满二叉树 half reduce
             # Mul(combined, e0h, e0h) + MulAddDst(combined, e1h, e1h) 两两配对(各一次 half 取整),
             # 16 配对值经 ReduceSumWithDataBlock<half> 满二叉树 reduce

    T10 修复: 原实现为顺序 half 累加 (np.float16 链式 31 次 add)。硬件 ReduceSumWithDataBlock
    是「两两配对(一次取整) + 16 元素满二叉树」——顺序模型在 margin < 1 half-ULP 的近并列块
    与 kernel 选择相反 (实测 512x5120 r=69/301/445 三块: r=69 顺序模型判 D4 而 kernel 判 D6;
    r=301/445 顺序模型精确并列判 D6 而 kernel 判 D4)。树形模型 3/3 复现 kernel 选择。

    返回 (sumE=0, sumESq = Σe²_scaled 以 float 表示 half 精度值)。
    fp4x2_e2m1_t 硬件网格固定为 {0,0.5,1,1.5,2,3,4,6}; D4 候选通过 scale4 使 x/scale4≤~4,
    不改变量化网格本身。
    """
    recip_f = bf16_to_float(recip_bits)
    e2 = []
    for x in xf:
        x_bf16 = bf16_to_float(f2bf16_rne(float(x)))  # x → bf16 RNE
        r_bf16 = bf16_to_float(f2bf16_rne(recip_f))   # recip → bf16 RNE
        # Step 1: bf16 multiply (x_bf16 × r_bf16 → bf16 RNE)
        v = bf16_to_float(f2bf16_rne(x_bf16 * r_bf16))
        # Step 2: Cast<U,bf16> — FP4 量化到网格值 (float)
        q = fp4_quant(v, mode=mode)
        # Step 3: Cast<bf16,U> — FP4 码点 → bf16 (bf16 RNE)
        q_bf16 = bf16_to_float(f2bf16_rne(q))
        # Step 4: Sub — 缩放域误差 (bf16 RNE, 匹配硬件 bf16 Sub)
        e_bf16 = bf16_to_float(f2bf16_rne(v - q_bf16))
        # Step 5: Cast<half,bf16> — bf16 → half
        e_half = np.float16(e_bf16)
        # Step 6: e² — half 积 (Mul; 注意 e_h*e_h 为 half 单次取整, 与 float64 精确平方不同)
        e2.append(np.float16(e_half * e_half))
    # Step 7: 配对 (e0²+e1²) 一次 half 取整 (MulAddDst)
    paired = [np.float16(e2[2 * i] + e2[2 * i + 1]) for i in range(len(e2) // 2)]
    if len(e2) % 2:
        paired.append(e2[-1])
    # Step 8: 满二叉树 half reduce (ReduceSumWithDataBlock<half>: 16→8→4→2→1)
    vals = paired
    while len(vals) > 1:
        nxt = [np.float16(vals[2 * i] + vals[2 * i + 1]) for i in range(len(vals) // 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return 0.0, float(vals[0])

# ===========================================================================
# 输入生成 (与测试 harness 完全一致)
# ===========================================================================
def build_input(rows, cols, mode):
    blocksPerRow = cols // BLOCKSIZE
    xbits = np.zeros((rows, cols), dtype=np.uint16)
    dtype = os.environ.get('DTYPE', 'bf16')
    _f2 = f2bf16_rne if dtype == 'bf16' else f2h
    _x2f = bf16_to_float if dtype == 'bf16' else h2f
    for r in range(rows):
        for c in range(cols):
            blk = c // BLOCKSIZE
            if mode == 'uniform':
                base = 1.0
            elif mode == 'rev':
                base = math.ldexp(1.0, 5 - (blk % 6))
            else:                                      # varied (默认)
                base = math.ldexp(1.0, blk % 6)
            amp = base * float(os.environ.get('AMP', '5.5'))
            v = amp * math.sin(0.7 * c + 0.13 * r)
            xbits[r, c] = _f2(v)
    xf = np.vectorize(_x2f)(xbits).astype(np.float64)
    return xbits, xf, blocksPerRow

# ===========================================================================
# 主参考: 跑一遍, 逐 block 给出 scale_6/scale_4 候选 + 选优 + mxScale
# ===========================================================================
def run(rows, cols, mode, dst_type_max):
    xbits, xf, blocksPerRow = build_input(rows, cols, mode)
    out = []
    for r in range(rows):
        for blk in range(blocksPerRow):
            xs = xf[r, blk * BLOCKSIZE:(blk + 1) * BLOCKSIZE]
            M_bits = block_M_bits(xs)
            M_float = bf16_to_float(M_bits)
            s6, rc6 = kernel_scale(M_bits, ADD_VALUE_FOR_BF16_D6_ROUND, SUB_OFFSET_FOR_D6_ROUND)
            s4, rc4 = derive_d4_from_d6(M_bits, s6, rc6)  # kernel: 1-bit 派生, bf16 Sub
            if abs(dst_type_max - 6.0) < 1e-9:
                # D=6 单候选 (NPU 已验证路径): 直接输出 scale_6, 不做选优
                out.append(dict(r=r, blk=blk, M=M_float, M_bits=M_bits,
                                s6=s6, s4=s4, rc6=rc6, rc4=rc4,
                                ssq6=None, ssq4=None, sel='D6-only', mx=s6, rc=rc6))
                continue
            # 自适应 4/6 — 判据: 原始域 Σe² = 缩放域 Σ(x/scale−q)² × scale²
            #   kernel StoreScaleSelect: ss_bf + (scale<<8) (BF16 位运算 ×scale²)
            #   golden: float64 等价位运算验证一致 (mxScale 100% 实测确认)
            _, sesq6 = candidate_sumsq(xs, rc6, s6)
            _, sesq4 = candidate_sumsq(xs, rc4, s4)
            sesq6_orig = sesq6 * (2.0 ** (2 * (s6 - 127)))
            sesq4_orig = sesq4 * (2.0 ** (2 * (s4 - 127)))
            if sesq4_orig < sesq6_orig:
                sel, mx, rc = 'D4', s4, rc4
            else:
                sel, mx, rc = 'D6', s6, rc6
            out.append(dict(r=r, blk=blk, M=M_float, M_bits=M_bits,
                            s6=s6, s4=s4, rc6=rc6, rc4=rc4,
                            ssq6=sesq6_orig, ssq4=sesq4_orig, sel=sel, mx=mx, rc=rc))
    return out, blocksPerRow, xbits, xf

# ===========================================================================
# 入口: 验证 + 报告
# ===========================================================================
def validate_d6_baseline(rows, cols, mode):
    """D=6 必须复现 NPU 已验证的 mxScale 基线 (标准输入 127 128 129 130)。"""
    out, _, _, _ = run(rows, cols, mode, 6.0)
    got = [d['mx'] for d in out]
    if rows == 1 and cols == 128 and mode == 'varied':
        expect = [127, 128, 129, 130]
        ok = got == expect
        print('[validate] D=6 baseline (ROWS=1 COLS=128 varied):')
        print('           got    =', got)
        print('           expect =', expect)
        print('           ->', 'PASS ✅ (golden 忠实: 输入建模+BF16 max+kernel_scale D=6)' if ok else 'FAIL ❌')
        return ok
    print('[validate] D=6 mxScale (per block):', got)
    return True

def selftest_edges():
    """边界用例自检 (与 kernel ComputeScaleDdr46 特判一致)。"""
    ok = True
    # 全零 block: M_bits=0 -> sharedExp=0 -> scaleE8M0=0, recip=0
    s, rc = kernel_scale(0, ADD_VALUE_FOR_BF16_MAN1)
    ok &= (s == 0 and rc == 0)
    # 极小 block (e_field<2 -> invalidDataMask clamp): M_bits=0x0080 -> scale=0
    s, _ = kernel_scale(0x0080, ADD_VALUE_FOR_BF16_MAN1)
    ok &= (s == 0)
    # Inf/NaN max (vdMaxExpOnly==0x7f80): scale=0xff (MAX_EXP_FOR_FP8)
    s, _ = kernel_scale(MAX_EXP_FOR_BF16, ADD_VALUE_FOR_BF16_MAN1)
    ok &= (s == MAX_EXP_FOR_FP8)
    # D=6 已知点 (BF16 of 5.5/11/22/44 -> 127/128/129/130)
    for val, expect in [(5.5, 127), (11.0, 128), (22.0, 129), (44.0, 130)]:
        s, _ = kernel_scale(f2bf16_rne(val), ADD_VALUE_FOR_BF16_MAN1)
        ok &= (s == expect)
    # scale_4 >= scale_6 (D=4 不小于 D=6), 且 recip_4 = 1/2^... 一致
    for val in [5.5, 11.0, 22.0]:
        mb = f2bf16_rne(val)
        s6, rc6 = kernel_scale(mb, ADD_VALUE_FOR_BF16_MAN1)
        s4, rc4 = kernel_scale(mb, ADD_VALUE_FOR_BF16_D4)
        ok &= (s4 == s6 or s4 == s6 + 1)          # scale_4 ∈ {scale_6, scale_6+1}
        ok &= (abs(bf16_to_float(rc6) - 1.0 / math.ldexp(1.0, s6 - 127)) < 1e-12)
    print(f'[selftest] 边界 (零/极小/Inf/已知点/scale4∈{{s6,s6+1}}/recip): '
          f'{"PASS ✅" if ok else "FAIL ❌"}')
    return ok


def main():
    rows = int(os.environ.get('ROWS', '1'))
    cols = int(os.environ.get('COLS', '128'))
    mode = os.environ.get('MODE', 'varied')
    dst = float(os.environ.get('DST_TYPE_MAX', '4.0'))

    print('=' * 78)
    print(f'golden_fp4_adaptive : ROWS={rows} COLS={cols} MODE={mode} DST_TYPE_MAX={dst}')
    print('=' * 78)

    # 0) 边界用例自检
    ok = selftest_edges()
    # 1) 先用标准小输入跑 D=6 基线, 校验忠实度
    ok &= validate_d6_baseline(1, 128, 'varied')
    print()

    # 2) 跑目标配置
    out, blocksPerRow, xbits, xf = run(rows, cols, mode, dst)

    print(f'[result] dst_type_max={dst}  blocks={len(out)}  blocksPerRow={blocksPerRow}')
    print(f'{"r":>2} {"blk":>3} {"M":>10} {"scale6":>7} {"scale4":>7} {"sumSq6":>12} {"sumSq4":>12} {"sel":>7} {"mxScale":>8}')
    n4 = n6 = 0
    for d in out:
        v6s = f"{d['ssq6']:.4e}" if d['ssq6'] is not None else '-'
        v4s = f"{d['ssq4']:.4e}" if d['ssq4'] is not None else '-'
        print(f"{d['r']:>2} {d['blk']:>3} {d['M']:>10.4f} {d['s6']:>7} {d['s4']:>7} "
              f"{v6s:>12} {v4s:>12} {d['sel']:>7} {d['mx']:>8}")
        if d['sel'] == 'D4':
            n4 += 1
        elif d['sel'] == 'D6':
            n6 += 1

    # 3) mxScale E8M0 原始字节 (与 NPU CopyOut Compact 1B/scale 对齐)
    mx_bytes = [d['mx'] for d in out]
    print()
    print('[result] mxScale(E8M0) raw bytes:', ' '.join(str(b) for b in mx_bytes))
    # 可选: 写 mxScale 文件供外部比对 (env MXSCALE_OUT)
    mx_out = os.environ.get('MXSCALE_OUT')
    if mx_out:
        np.array(mx_bytes, dtype=np.uint8).tofile(mx_out)
        print(f'[result] mxScale written to: {mx_out} ({len(mx_bytes)} bytes)')

    # 3b) 可选: 与 NPU 实测 raw bytes 逐 block 对比 (env NPU_SCALE_RAW="b0 b1 ...")
    npu_raw = os.environ.get('NPU_SCALE_RAW')
    if npu_raw:
        npu = [int(x) for x in npu_raw.split()]
        print('[compare] NPU raw bytes       :', ' '.join(str(b) for b in npu))
        mism = [(d['r'], d['blk'], d['mx'], npu[i] if i < len(npu) else None)
                for i, d in enumerate(out) if i >= len(npu) or npu[i] != d['mx']]
        print(f'[compare] 逐 block 一致: {len(out) - len(mism)}/{len(out)}')
        if mism:
            print('[compare] 不一致 block (r, blk, golden, npu):')
            for r, blk, gm, nm in mism:
                print(f'           r={r} blk={blk} golden={gm} npu={nm}')
        else:
            print('[compare] -> golden 与 NPU 完全一致 ✅')

    if abs(dst - 4.0) < 1e-9:
        print(f'[result] 自适应选优 (Σe² 判据 = sumSq4<sumSq6; n=32 常数故等价 MSE; 非方差): '
              f'D4={n4}  D6={n6}  (混合={bool(n4 and n6)})')
        # 自洽校验: 所选候选须为 argmin(Σe²) —— 验证选优逻辑与 sumSq 一致
        bad = 0
        for d in out:
            if d['ssq6'] is None or d['ssq4'] is None:
                continue
            expect_d4 = d['ssq4'] < d['ssq6']
            if (d['sel'] == 'D4') != expect_d4:
                bad += 1
        status = '✅' if bad == 0 else f'({bad} 块不自洽 ❌)'
        print(f'[诊断] 选优 = argmin(Σe²) 自洽: {len(out) - bad}/{len(out)} {status}')

    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())


# ===========================================================================
# ===========================================================================
# T9d 扩展：任意输入 x 的 46 逐字节参考（复用上方全部 kernel 算法；供测试 oracle）
# x_f32_2d: (rows, cols) float64; 量化轴=cols（尾轴）须 32 对齐。
# ===========================================================================
def _s6s4(mb):
    """kernel 语义（与 verify_y_golden.py 一致，T5 实测 100%）：scale6/scale4 均用 ROUND 版常量
    （D6_ROUND 0x0078/0x0180、D4_ROUND 0x004A/0x0180)。"""
    s6, rc6 = kernel_scale(mb, ADD_VALUE_FOR_BF16_D6_ROUND, SUB_OFFSET_FOR_D6_ROUND)
    s4, rc4 = kernel_scale(mb, ADD_VALUE_FOR_BF16_D4_ROUND, SUB_OFFSET_FOR_D4_ROUND)
    return s6, rc6, s4, rc4


def _kernel_better4(xs, mode="rint"):
    """kernel-faithful winner：half 域判据 ss_adj = halfbits(Σe²) + ((2*scale)<<10)，D4 胜 ⟺
    half(4·Σe²4) < half(Σe²6)（StoreScaleSelectFunc，T10 修复：原 bf16 域 Cast 预取整翻转近并列块）。
    Σe² 为 half 精度值，×4/×2 缩放均精确——故原始域 float64 比较 orig4<orig6 与 half 域判据逐位等价，
    前提是 Σe² 累加模型与 kernel 一致（T10 修复后 candidate_sumsq 用配对+满二叉树模型，见其上注释）。"""
    mb = block_M_bits(xs)
    s6, rc6, s4, rc4 = _s6s4(mb)
    _, sq6 = candidate_sumsq(xs, rc6, s6, mode)
    _, sq4 = candidate_sumsq(xs, rc4, s4, mode)
    orig6 = sq6 * (2.0 ** (2 * (s6 - 127)))
    orig4 = sq4 * (2.0 ** (2 * (s4 - 127)))
    return (orig4 < orig6), s6, rc6, s4, rc4


def ref_46_scale(x_f32_2d, round_mode="rint"):
    """逐 block winner 序列经 DIST_PACK_B16 打包：(rows, ceil(nb/2), 2)
    （uint16 = [win(2k), win(2k+1)]；nb 奇数末对 pad=0，匹配 scaleColNum 偶数化）。"""
    rows, cols = x_f32_2d.shape
    assert cols % BLOCKSIZE == 0, "cols must be block-aligned"
    nb = cols // BLOCKSIZE
    wins = np.zeros((rows, nb), dtype=np.uint8)
    for r in range(rows):
        for blk in range(nb):
            xs = x_f32_2d[r, blk * BLOCKSIZE:(blk + 1) * BLOCKSIZE]
            better4, s6, _, s4, _ = _kernel_better4(xs, round_mode)
            wins[r, blk] = s4 if better4 else s6
    pair_n = (nb + 1) // 2
    out = np.zeros((rows, pair_n, 2), dtype=np.uint8)
    pairs = nb // 2
    if pairs:
        out[:, :pairs, :] = wins[:, :pairs * 2].reshape(rows, pairs, 2)
    if nb % 2:
        out[:, pairs, 0] = wins[:, -1]
    return out


FP4_CODE = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7}


def ref_46_y(x_f32_2d, pack="lohi", winners=None, round_mode="rint"):
    """逐 block 量化链参考：winner 可由 kernel 实测（npu mxscale 反解，decision 链已由 mx 验证）；
    无 winners 时用 kernel-faithful _kernel_better4。返回 (rows, cols//2) uint8。"""
    rows, cols = x_f32_2d.shape
    codes = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for blk in range(cols // BLOCKSIZE):
            xs = x_f32_2d[r, blk * BLOCKSIZE:(blk + 1) * BLOCKSIZE]
            better4, _, rc6, s4, rc4 = _kernel_better4(xs, round_mode)
            if winners is not None:
                better4 = (winners[r, blk] == s4)
            rc = rc4 if better4 else rc6
            recip_f = bf16_to_float(rc)
            for i, xv in enumerate(xs):
                x_bf16 = bf16_to_float(f2bf16_rne(float(xv)))
                r_bf16 = bf16_to_float(f2bf16_rne(recip_f))
                v = bf16_to_float(f2bf16_rne(x_bf16 * r_bf16))
                q = fp4_quant(v, mode=round_mode)
                # E2M1 3-bit 码：正值格点 1..7；负值 = 码值 | 0b1000（符号位）
                if q == 0.0:
                    # E2M1 负零（-0.0）独立编码 0b1000（符号位=1）；+0.0 -> 0b0000
                    codes[r, blk * BLOCKSIZE + i] = 0b1000 if math.copysign(1.0, q) < 0 else 0
                else:
                    codes[r, blk * BLOCKSIZE + i] = FP4_CODE[abs(q)] | (0b1000 if q < 0 else 0)
    out = np.zeros((rows, cols // 2), dtype=np.uint8)
    for r in range(rows):
        for k in range(cols // 2):
            if pack == "lohi":
                out[r, k] = codes[r, 2 * k] | (codes[r, 2 * k + 1] << 4)
            else:
                out[r, k] = (codes[r, 2 * k] << 4) | codes[r, 2 * k + 1]
    return out
