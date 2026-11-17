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
#
# This file is the CPU golden used by the qbmm_swiglu_mxquant operator tests.

"""Torch CPU/NPU reference for QuantBatchMatmulV3 + SwiGLU + DynamicMxQuant.

All tensors are byte tensors.  ``in_fp4`` / ``out_fp4`` are scalar switches
because Torch's public dtype set used by the benchmark tool has no fp8/fp4
entry.  FP8 bytes are E4M3FN, FP4 bytes are packed E2M1 nibbles (element 2*i
in the low nibble, element 2*i+1 in the high nibble).

FP8 mode
--------
a        uint8 [M, K]          E4M3FN values, row-major
scale_a  uint8 [M, K/64, 2]    E8M0 bytes; logical [M, K/32] row-major
b        uint8 [K, N]          E4M3FN values, row-major
scale_b  uint8 [K/64, N, 2]    E8M0 bytes; logical [K/32, N] row-major

FP4 mode
--------
a        uint8 [M, K/2]        packed E2M1 along K, row-major
b        uint8 [N, K/2]        packed E2M1 along K; logical [K, N] column-major

The fused result is
    C      = dequant(A) @ dequant(B)                  # fp32 accumulate
    act    = C[:, :N/2] ; gate = C[:, N/2:]
    S      = silu(act) * gate                         # fp32 activation
    S16    = round_to_bf16(S)                         # bf16 before MX quant
    y, sy  = dynamic_mx_quant(S16, emax(out_fp4))      # 32-element blocks

This is the fused-operator / official QuantMatmulActivationQuant semantics
(activation on the FP32 accumulator); call with ``matmul_bf16=False``.
``matmul_bf16=True`` (default) is the small-op stitching golden: QBMM writes a
BF16 intermediate before the activation.

Dynamic MX quant follows the device bit-exact rule used by the Ascend
implementation: the shared exponent is read from the bf16 exponent field of
the block maximum, the scale is applied as a power-of-two multiply, and the
value is rounded to the target element format with saturation.
"""

import torch

_MX_GROUP = 32
_FP8_MAX = 448.0
_FP8_EMAX = 8
_FP4_EMAX = 2

# E2M1 code order: bit3 sign, bits2:1 exponent, bit0 mantissa.
_FP4_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp4_lut_like(buf):
    values = torch.tensor(
        [_FP4_MAG[i] for i in range(8)] + [-_FP4_MAG[i] for i in range(8)],
        dtype=torch.float32,
        device=buf.device,
    )
    return values


def _decode_fp8(buf):
    return buf.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def _decode_fp4_row(buf, k):
    """Decode (rows, k/2) packed nibbles into (rows, k) values."""
    lo = (buf & 0x0F).to(torch.int32)
    hi = ((buf >> 4) & 0x0F).to(torch.int32)
    codes = torch.stack((lo, hi), dim=-1).reshape(buf.shape[0], k)
    return _fp4_lut_like(buf)[codes.long()]


def _decode_scale_a(scale):
    """[M, K/64, 2] E8M0 bytes -> [M, K/32] float32 powers of two."""
    e = scale.reshape(scale.shape[0], -1).to(torch.int32)
    return torch.exp2((e - 127).to(torch.float32))


def _decode_scale_b(scale):
    """[K/64, N, 2] E8M0 bytes -> [K/32, N] float32 powers of two.

    The two shared exponents of a 64-K block are adjacent in memory, so the
    logical K/32 row order is obtained by moving the trailing 2 ahead of N.
    """
    groups, n, two = scale.shape
    e = scale.to(torch.int32).permute(0, 2, 1).reshape(groups * two, n)
    return torch.exp2((e - 127).to(torch.float32))


def _round_f32_to_bf16_bits(x):
    """Round an f32 tensor to bf16 (ties-to-even) and return bf16 bit patterns."""
    bits = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).to(torch.int64)


def _bf16_bits_to_f32(bits):
    return (bits << 16).to(torch.int32).view(torch.float32)


def _silu(x):
    return x / (1.0 + torch.exp(-x))


def _quant_fp8_code(scaled):
    """E4M3FN encode with RNE, portable across CPU and NPU.

    The device float8 cast does not reliably use ties-to-even, so encode the
    format explicitly.  Layout follows the kernel's manual encoder.
    """
    bits = scaled.contiguous().view(torch.int32)
    sign = ((bits >> 31) & 1) << 7
    abs_bits = bits & 0x7FFFFFFF
    exp_field = (abs_bits >> 23) & 0xFF
    mant = abs_bits & 0x7FFFFF
    mant3 = (mant + 0x7FFFF + ((mant >> 20) & 1)) >> 20
    code_exp = exp_field - 120
    carry = mant3 >= 8
    code_exp = torch.where(carry, code_exp + 1, code_exp)
    mant3 = torch.where(carry, mant3 - 8, mant3)
    normal = (code_exp << 3) | mant3
    normal = torch.where(code_exp >= 16, torch.full_like(normal, 0x7E), normal)
    normal = torch.clamp(normal, max=0x7E)
    sub = torch.round(scaled.abs() * 512.0).to(torch.int32)
    sub = torch.clamp(sub, max=8)
    mag = torch.where(exp_field >= 121, normal, sub)
    return (sign | mag).to(torch.uint8)


def _quant_fp4_code(scaled):
    """Round scaled f32 values to E2M1 codes with ties-to-even on magnitude."""
    a = scaled.abs()
    mag = torch.zeros_like(a, dtype=torch.int32)
    mag = torch.where(a > 0.25, torch.ones_like(mag), mag)
    mag = torch.where(a >= 0.75, mag + 1, mag)
    mag = torch.where(a > 1.25, mag + 1, mag)
    mag = torch.where(a >= 1.75, mag + 1, mag)
    mag = torch.where(a > 2.5, mag + 1, mag)
    mag = torch.where(a >= 3.5, mag + 1, mag)
    mag = torch.where(a > 5.0, mag + 1, mag)
    bits = scaled.contiguous().view(torch.int32)
    sign = ((bits >> 31) & 1) << 3
    return mag | sign


def _dynamic_mx_quant(s16_bits, emax):
    """Quantize bf16 values, given as bf16 bit patterns, into MX format.

    Returns (scale_bytes [M, C/64, 2] uint8, fp8 bytes [M, C] uint8 or fp4
    packed bytes [M, C/2] uint8).
    """
    rows, cols = s16_bits.shape
    assert cols % _MX_GROUP == 0, "quantized axis must be a multiple of 32"
    groups = s16_bits.reshape(rows, cols // _MX_GROUP, _MX_GROUP)

    exp_bits = groups & 0x7F80
    max_exp = exp_bits.amax(dim=-1)  # bf16 exponent field << 7
    zero = max_exp == 0
    special = max_exp == 0x7F80

    exp_byte = (max_exp >> 7) - emax
    exp_byte = exp_byte.clamp(0, 254)
    exp_byte = torch.where(zero, torch.zeros_like(exp_byte), exp_byte)
    exp_byte = torch.where(special, torch.full_like(exp_byte, 255), exp_byte)

    # bf16 half scale = 2 ** -(exp_byte) = 2 ** (127 - exp_byte) in bf16.
    half_scale = torch.exp2((127 - exp_byte.clamp(0, 254)).to(torch.float32))
    half_scale = torch.where(zero, torch.zeros_like(half_scale), half_scale)
    half_scale = torch.where(special, torch.full_like(half_scale, float("nan")), half_scale)

    values = _bf16_bits_to_f32(s16_bits)
    scaled = values * half_scale.repeat_interleave(_MX_GROUP, dim=-1)
    # The fused kernel normalizes every zero (including -0) to +0; mirror it
    # so the byte-level comparison is well defined.
    scaled = torch.where(scaled == 0, torch.zeros_like(scaled), scaled)

    if emax == _FP4_EMAX:
        code = _quant_fp4_code(scaled)
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        q = packed
    else:
        q = _quant_fp8_code(scaled)

    sy = exp_byte.reshape(rows, cols // 64, 2).to(torch.uint8)
    return sy, q


def run(a, scale_a, b, scale_b, in_fp4, out_fp4, matmul_bf16=True):
    # Default True = small-op stitching golden (QBMM writes BF16, unchanged).
    # Pass matmul_bf16=False for the fused operator, which now matches the
    # official QuantMatmulActivationQuant semantics (activation on FP32).
    # The oracle is a full-FP32 CPU implementation.  Device float32 matmul
    # kernels use a different accumulation order, which flips MX scale
    # exponents at block boundaries; keeping the oracle on CPU makes the
    # comparison independent of the device schedule.
    out_device = a.device
    a = a.cpu()
    scale_a = scale_a.cpu()
    b = b.cpu()
    scale_b = scale_b.cpu()
    if in_fp4:
        k = a.shape[1] * 2
        n = b.shape[0]
        assert b.shape[1] * 2 == k
    else:
        k = a.shape[1]
        n = b.shape[1]
    n_out = n // 2

    if in_fp4:
        a_val = _decode_fp4_row(a, k)
        b_val = _decode_fp4_row(b, k).transpose(0, 1).contiguous()
    else:
        a_val = _decode_fp8(a)
        b_val = _decode_fp8(b)

    sa = _decode_scale_a(scale_a)  # [M, K/32]
    sb = _decode_scale_b(scale_b)  # [K/32, N]
    a_deq = a_val * sa.repeat_interleave(_MX_GROUP, dim=1)
    b_deq = b_val * sb.repeat_interleave(_MX_GROUP, dim=0)

    acc = torch.matmul(a_deq, b_deq)
    if matmul_bf16:
        # Small-op baseline semantics: QBMM writes a BF16 intermediate.
        acc_bits = _round_f32_to_bf16_bits(acc)
        acc_bf = _bf16_bits_to_f32(acc_bits)
    else:
        # Fused-operator semantics: activation consumes the FP32 accumulator.
        acc_bf = acc

    act = acc_bf[:, :n_out]
    gate = acc_bf[:, n_out:]
    s = _silu(act) * gate
    s_bits = _round_f32_to_bf16_bits(s)

    sy, y = _dynamic_mx_quant(s_bits, _FP4_EMAX if out_fp4 else _FP8_EMAX)
    y = y.to(out_device)
    sy = sy.to(out_device)
    if out_device.type == "npu":
        # The oracle itself runs on CPU; launch one small device kernel so the
        # benchmark tool has a device profile to cache.  It does not affect the
        # returned values or their comparison.
        torch.zeros(1, dtype=torch.float32, device=out_device)
    return y, sy
