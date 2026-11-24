# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""A5 RNE INT8 scatter and FP8 send kernels using producer-reduced FP32 divisors.

The kernels preserve the one-head packet ownership map.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice


@triton.jit
def corrected_divide(x, y):
    """Correct reciprocal division using its FMA residual; preserve overflow."""
    initial = x / y
    residual = tl.fma(-initial, y, x)
    refined = initial + residual / y
    return tl.where(tl.abs(initial) == float('inf'), initial, refined)


@triton.jit
def zero_kernel(out, N: tl.constexpr, B: tl.constexpr):
    for i in range(tl.program_id(0), tl.cdiv(N, B), tl.num_programs(0)):
        off = i * B + tl.arange(0, B)
        tl.store(out + off, 0, off < N)


@triton.jit
def tiled_int8(
    padded,
    scales,
    src,
    padded_row,
    gblk,
    head_row,
    out,
    OWNED: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCKS: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    OUT_HEAD: tl.constexpr = 0,
):
    """Quantize four heads per tile and scatter into destination-major packets."""
    rowtiles = tl.cdiv(OWNED, R)
    headtiles = tl.cdiv(HEADS, H)
    for tile in range(tl.program_id(0).to(tl.int32), rowtiles * headtiles, tl.num_programs(0).to(tl.int32)):
        row = tile // headtiles * R + tl.arange(0, R)
        head = tile % headtiles * H + tl.arange(0, H)
        dim = tl.arange(0, D)
        pad = tl.load(padded_row + row, row < OWNED, other=0).to(tl.int32)
        token = tl.load(src + row, row < OWNED, other=0).to(tl.int32)
        block = tl.load(gblk + pad // 64, row < OWNED, other=0).to(tl.int32)
        sc = tl.load(
            scales + head[None, :] * BLOCKS + block[:, None], (row < OWNED)[:, None] & (head < HEADS)[None, :], other=1
        )
        sc = tl.where(sc == 0, 1.0, sc)
        mask = (row < OWNED)[:, None, None] & (head < HEADS)[None, :, None] & (dim < DIM)[None, None, :]
        x = tl.load(
            padded + (pad[:, None, None] * HEADS + head[None, :, None]) * DIM + dim[None, None, :], mask, other=0
        ).to(tl.float32)
        result = libdevice.rint(corrected_divide(x, sc[:, :, None])).to(tl.int8)
        dest = tl.load(head_row + head, head < HEADS, other=0).to(tl.int32)
        result = tl.trans(result, (1, 0, 2))
        mask = tl.trans(mask, (1, 0, 2))
        pitch = OUT_HEAD if OUT_HEAD else TOKENS * DIM
        tl.store(out + dest[:, None, None] * pitch + token[None, :, None] * DIM + dim[None, None, :], result, mask)


@triton.jit
def fp8_kernel(
    value,
    scales,
    out,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    CHUNKS: tl.constexpr,
    WORLD: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SD: tl.constexpr,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    OUT_HEAD: tl.constexpr = 0,
):
    """Write FP8 bytes directly in chunk/destination order using source strides."""
    tiles = tl.cdiv(TOKENS, ROWS)
    for tile in range(tl.program_id(0).to(tl.int32), tiles * HEADS, tl.num_programs(0).to(tl.int32)):
        head = tile // tiles
        token = tile % tiles * ROWS + tl.arange(0, ROWS)
        dim = tl.arange(0, D)
        scale = tl.load(scales + head * DIM + dim, dim < DIM, other=1)
        scale = tl.where(scale == 0, 1.0, scale)
        mask = (token < TOKENS)[:, None] & (dim < DIM)[None, :]
        x = tl.load(value + token[:, None] * ST + head * SH + dim[None, :] * SD, mask, other=0).to(tl.float32)
        y = corrected_divide(x, scale[None, :])
        y = tl.minimum(tl.maximum(y, -448.0), 448.0)
        y = tl.where(tl.abs(x) < 1.1754943508222875e-38, 0.0, y)
        dest = head // CHUNKS
        chunk = head % CHUNKS
        pitch = OUT_HEAD if OUT_HEAD else TOKENS * DIM
        offset = (chunk * WORLD + dest) * pitch + token[:, None] * DIM + dim[None, :]
        tl.store(out + offset, y.to(out.dtype.element_ty), mask)


def cores(device):
    p = triton.runtime.driver.active.utils.get_device_properties(device)
    return max(1, p.get('num_vectorcore', p.get('num_aicore', 1)))


def quantize(st):
    """Write the exact INT8 payload into the producer-owned buffer."""
    from .config import QUANT_BLOCK

    if QUANT_BLOCK != 64:
        raise ValueError("native INT8 payload ABI requires QUANT_BLOCK=64")
    c = cores(st.payload.device.index)
    # Valid full-coverage maps overwrite every head/token byte.
    # Partial/padded maps retain the existing zero-fill contract.
    if st.maps.n_owned != st.payload.shape[1]:
        if st.payload.is_contiguous():
            zero_kernel[c,](st.payload, st.payload.numel(), 4096)
        else:
            st.payload.zero_()
    tiled_int8[c,](
        st.payload_padded,
        st.scales,
        st.maps.src,
        st.maps.padded_row,
        st.maps.gblk,
        st.head_row,
        st.payload,
        st.maps.n_owned,
        st.payload.shape[1],
        st.hc,
        st.hs,
        st.nblk,
        2,
        32,
        triton.next_power_of_2(st.hs),
        st.payload.stride(0),
    )
    return st.payload


def v_send(value, divisor, chunks, world, out=None):
    """Allocate a contiguous one-head FP8 communication packet."""
    bs, tokens, heads, dim = value.shape
    assert bs == 1 and heads == chunks * world
    if out is None:
        out = torch.empty((chunks, world, tokens, bs, 1, dim), dtype=torch.float8_e4m3fn, device=value.device)
    else:
        if (
            out.shape != (chunks, world, tokens, bs, 1, dim)
            or out.dtype != torch.int8
            or out.device != value.device
            or out.stride(-1) != 1
            or out.stride(2) != dim
        ):
            raise ValueError('FP8 output must match the fixed one-head send layout')
        out = out.view(torch.float8_e4m3fn)
    fp8_kernel[cores(value.device.index),](
        value,
        divisor.contiguous(),
        out,
        tokens,
        heads,
        dim,
        chunks,
        world,
        value.stride(1),
        value.stride(2),
        value.stride(3),
        32,
        triton.next_power_of_2(dim),
        out.stride(1),
    )
    return out.view(torch.int8)
