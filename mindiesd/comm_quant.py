# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Shared FP8 communication blocks; no device operators or distributed state."""

import math

import torch

COMM_DTYPES = ("none", "fp8_e4m3", "mxfp8")
FP8_MAX = 448.0


def encode_fp8_e4m3(value):
    """Encode finite E4M3FN, including signed zero and subnormal carry."""
    sign = torch.signbit(value).to(torch.uint8) << 7
    absolute = value.abs().float()
    exponent = torch.floor(torch.log2(absolute.clamp_min(2.0**-9)))
    normal = (absolute > 0) & (exponent >= -6)
    exponent_bits = (exponent + 7).clamp(1, 15).to(torch.int16)
    mantissa = torch.round((absolute / torch.pow(2.0, exponent) - 1.0) * 8.0).to(torch.int16)
    carry = normal & (mantissa >= 8)
    exponent_bits = exponent_bits + carry.to(torch.int16)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    max_finite = exponent_bits >= 15
    exponent_bits = exponent_bits.clamp(max=15)
    mantissa = torch.where(max_finite, mantissa.clamp(max=6), mantissa.clamp(0, 7))
    normal_code = ((exponent_bits << 3) | mantissa).to(torch.uint8)
    normal_code = torch.where(absolute >= FP8_MAX, 126, normal_code).to(torch.uint8)
    # Code 8 is the smallest normal value; rounding a subnormal can carry into it.
    subnormal = torch.round(absolute / (2.0**-9)).clamp(0, 8).to(torch.uint8)
    return sign | torch.where(normal, normal_code, subnormal)


def decode_fp8_e4m3(payload):
    """Decode finite E4M3FN bytes without requiring a device FP8 cast."""
    bits = payload.to(torch.int16)
    negative = (bits & 0x80) != 0
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    normal_value = torch.pow(2.0, exponent.float() - 7.0) * (1.0 + mantissa.float() / 8.0)
    subnormal_value = mantissa.float() * (2.0**-9)
    value = torch.where(exponent != 0, normal_value, subnormal_value)
    return torch.where(negative, -value, value)


def block_quantize(value, block_size, comm_dtype, *, scale_override=None, portable=False):
    """Quantize each leading communication chunk independently.

    ``mxfp8`` is the existing communication codec name for power-of-two FP32
    scales with a caller-selected block size; it does not imply native MXFP8 BSA.
    Native and portable encoders share the same scaling and padding contract.
    """
    if comm_dtype not in COMM_DTYPES[1:]:
        raise ValueError("communication dtype must be fp8_e4m3 or mxfp8")
    if block_size <= 0:
        raise ValueError("communication block size must be positive")
    chunks = value.shape[0]
    flat = value.float().reshape(chunks, -1)
    original_width = flat.shape[1]
    padded_width = math.ceil(original_width / block_size) * block_size
    if padded_width != original_width:
        flat = torch.nn.functional.pad(flat, (0, padded_width - original_width))
    blocks = flat.reshape(chunks, -1, block_size)
    scales = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / FP8_MAX
    if comm_dtype == "mxfp8":
        scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    if scale_override is not None:
        if not isinstance(scale_override, torch.Tensor):
            raise ValueError("communication scale overrides must be torch.Tensor values")
        if scale_override.numel() == 1 or tuple(scale_override.shape) == tuple(scales.shape):
            scales = scale_override.to(device=value.device, dtype=torch.float32).expand_as(scales)
        else:
            raise ValueError("scale override must be scalar or have shape %r" % (tuple(scales.shape),))
    normalized = (blocks / scales.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    payload = encode_fp8_e4m3(normalized) if portable else normalized.to(torch.float8_e4m3fn)
    return payload.reshape(chunks, padded_width), scales.contiguous(), original_width


def block_dequantize(payload, scales, original_width, shape, dtype):
    """Restore either native FP8 or portable bytes to the original layout."""
    block_size = payload.shape[1] // scales.shape[1]
    values = decode_fp8_e4m3(payload) if payload.dtype == torch.uint8 else payload.float()
    values = values.reshape(scales.shape[0], -1, block_size) * scales.unsqueeze(-1)
    return values.reshape(scales.shape[0], -1)[:, :original_width].reshape(shape).to(dtype)
