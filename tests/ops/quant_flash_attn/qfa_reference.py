# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2. See LICENSE for details.

"""Portable NumPy MXFP4 codecs and independent, bounded-memory CPU references.

The MXFP4 attention mode models the kernel's FP16 scores, log2 group maxima,
BF16-to-FP4 probabilities and E8M0 scaling. It is NOT a bit-exact device simulator:
hardware Exp, reduction order, FlashDecode partitioning and rounding may differ.
"""

from dataclasses import dataclass

import numpy as np

LEVELS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
MIDPOINTS = (LEVELS[1:] + LEVELS[:-1]) / 2


@dataclass
class Quantized:
    packed: np.ndarray
    scales: np.ndarray
    axis: int
    shape: tuple


def fp4_codes(values):
    """Saturating nearest-even E2M1 encoding; low nibble is the even element."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("FP4 input must be finite")
    mag = np.abs(values)
    index = np.searchsorted(MIDPOINTS, mag, side="left")
    ties = (index < 7) & (mag == MIDPOINTS[np.minimum(index, 6)])
    index += ties & ((index & 1) != 0)
    return (index | (np.signbit(values).astype(np.uint8) << 3)).astype(np.uint8)


def bf16_round(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def quantize_mxfp4(values, axis=-1):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] % 2 or not all(values.shape):
        raise ValueError("MXFP4 requires nonempty dimensions and an even packing dimension")
    if not np.isfinite(values).all():
        raise ValueError("MXFP4 input must be finite")
    axis %= values.ndim
    moved = np.moveaxis(values, axis, -1)
    length = moved.shape[-1]
    padding = [(0, 0)] * moved.ndim
    padding[-1] = (0, (-length) % 32)
    grouped = np.pad(moved, padding).reshape(*moved.shape[:-1], -1, 32)
    peak = np.max(np.abs(grouped), axis=-1)
    exponent = np.floor(np.log2(np.maximum(peak, np.finfo(np.float32).tiny))) - 2
    exponent = np.where(peak == 0, 0, np.clip(exponent, -127, 127)).astype(np.int32)
    scale = np.ldexp(np.ones_like(peak), exponent)
    codes = fp4_codes(grouped / scale[..., None]).reshape(*moved.shape[:-1], -1)[..., :length]
    codes = np.moveaxis(codes, -1, axis)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    scales = np.moveaxis((exponent + 127).astype(np.uint8), -1, axis)
    return Quantized(np.ascontiguousarray(packed), np.ascontiguousarray(scales), axis, values.shape)


def decode_mxfp4(encoded):
    packed = encoded.packed
    if packed.dtype != np.uint8 or encoded.scales.dtype != np.uint8:
        raise ValueError("Packed data and E8M0 scales must be uint8")
    if packed.shape != (*encoded.shape[:-1], encoded.shape[-1] // 2):
        raise ValueError("Packed shape mismatch")
    expected = list(encoded.shape)
    expected[encoded.axis] = (expected[encoded.axis] + 31) // 32
    if tuple(expected) != encoded.scales.shape or (encoded.scales == 255).any():
        raise ValueError("Invalid scale shape or nonfinite E8M0 code")
    codes = np.empty(encoded.shape, dtype=np.uint8)
    codes[..., ::2], codes[..., 1::2] = packed & 15, packed >> 4
    values = LEVELS[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    scales = np.ldexp(np.ones(encoded.scales.shape, dtype=np.float64), encoded.scales.astype(np.int32) - 127)
    expanded = np.repeat(scales, 32, axis=encoded.axis)
    slices = [slice(None)] * values.ndim
    slices[encoded.axis] = slice(encoded.shape[encoded.axis])
    return (values * expanded[tuple(slices)]).astype(np.float32)


def _snap_probabilities(raw, scale):
    """Model one KV tile; return FP4 values and per-row/per-group log2 maxima."""
    raw = raw.astype(np.float16)
    if not np.isfinite(raw).all():
        raise ValueError("CPU model FP16 QK overflow; lower input scale or inspect this case")
    grouped = np.pad(raw, ((0, 0), (0, (-raw.shape[1]) % 32)), constant_values=-np.inf)
    grouped = grouped.reshape(raw.shape[0], -1, 32)
    scale = np.float16(scale)
    maxima = (grouped.max(axis=-1) * scale).astype(np.float16)
    exponent = np.floor((maxima * np.float16(1 / np.log(2))).astype(np.float16)).astype(np.float32)
    shift = ((exponent.astype(np.float16) - np.float16(2)) * np.float16(np.log(2))).astype(np.float16)
    logits = (grouped * scale).astype(np.float16)
    shifted = (logits - shift[..., None]).astype(np.float16)
    local = np.exp(shifted.astype(np.float32)).astype(np.float16)
    p = LEVELS[fp4_codes(bf16_round(local)) & 7]
    return p, exponent


def attention_rows(case, arrays, rows, kv_chunk=256, q_chunk=32):
    """Noncausal attention for explicit (batch, query-head, query-token) rows.

    Algorithm-aware P quantization on decoded Q/K/V, BF16 output model.
    Physical padding is excluded by case.q_used / case.kv_used.
    """
    if kv_chunk <= 0 or q_chunk <= 0 or kv_chunk % 32:
        raise ValueError("Invalid chunk size (KV chunk must be a multiple of 32)")
    q, k, v = (arrays[name] for name in ("q", "k", "v"))
    expected = [
        (case.batch, case.q_heads, case.physical_q, case.dim),
        (case.batch, case.kv_heads, case.physical_kv, case.dim),
    ]
    if q.shape != expected[0] or k.shape != expected[1] or v.shape != expected[1]:
        raise ValueError("Reference input shape mismatch")
    if any(not np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("Reference inputs must be finite")
    groups = {}
    for i, (b, h, token) in enumerate(rows):
        if not (0 <= b < case.batch and 0 <= h < case.q_heads and 0 <= token < case.q_used[b]):
            raise ValueError("Sample row is outside the effective query length")
        groups.setdefault((b, h), []).append((i, token))
    out = np.empty((len(rows), case.dim), dtype=np.float64)
    dtype = np.float32
    for (b, h), entries in groups.items():
        kh = h // (case.q_heads // case.kv_heads)
        for start in range(0, len(entries), q_chunk):
            selected = entries[start : start + q_chunk]
            qq = q[b, h, [token for _, token in selected]].astype(dtype)
            maximum = np.full(len(selected), -np.inf, dtype=dtype)
            denom = np.zeros(len(selected), dtype=dtype)
            acc = np.zeros((len(selected), case.dim), dtype=dtype)
            for ks in range(0, case.kv_used[b], kv_chunk):
                ke = min(ks + kv_chunk, case.kv_used[b])
                raw = qq @ k[b, kh, ks:ke].astype(dtype).T
                p4, exponents = _snap_probabilities(raw, case.scale)
                new_max = np.maximum(maximum, exponents.max(axis=-1))
                alpha = np.exp2(maximum - new_max)
                # E8M0 scale byte saturates at zero (2^-127), not at float zero.
                corr = np.exp2(np.maximum(exponents - new_max[:, None] - 2, -127))
                p = (p4 * corr[..., None]).reshape(len(selected), -1)[:, : ke - ks]
                acc = acc * alpha[:, None] + p @ v[b, kh, ks:ke].astype(dtype)
                denom = denom * alpha + p.sum(axis=-1, dtype=dtype)
                maximum = new_max
            if not (denom > 0).all() or not np.isfinite(acc).all():
                raise ValueError("Nonfinite or zero normalization in CPU reference")
            result = acc / denom[:, None]
            result = bf16_round(result)
            out[[index for index, _ in selected]] = result
    return out
