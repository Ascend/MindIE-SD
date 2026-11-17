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
# This file is a CATLASS Python-DSL kernel: its layout and flag structure mirror
# the DSL examples, so it is excluded from line-length/format rewrites.
# ruff: noqa
# fmt: off
# pylint: disable=too-many-lines

"""QuantBatchMatmulV3 (MXFP8/MXFP4) + SwiGLU + DynamicMxQuant.

CATLASS Python DSL implementation.  The host entry point is ``run``; it binds
byte tensors to TLA tensors, compiles one mixed kernel per element-format
combination and launches it.

Logical contract (see docs/zh/features/qbmm_swiglu_mxquant.md):

    C32 = dequant(A) @ dequant(B)              fp32 accumulation
    S16 = round_bf16_rne(silu(C32[:, :N/2]) * C32[:, N/2:])
    Y, SY = dynamic_mx_quant(S16)

``in_fp4`` / ``out_fp4`` are scalar switches because the benchmark dtype set
has no fp8/fp4 entry; all tensors are byte tensors.  Host validation requires
``K % 64 == 0`` and ``N % 128 == 0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from catlass import tla
from catlass.tla.runtime import from_dlpack


_TM = 256
_TN = 128
_TK = 256
_HALF = _TM // 2

_UB_ACT_BYTES = _HALF * _TN * 4
_UB_Y_ELEMS = _HALF * _TN
_UB_SY_ELEMS = _HALF
_UB_CODES_ELEMS = _HALF * _TN

_BNONE = 0
_BF8 = 1
_BF4 = 2

# Optional two-kernel split-K path for tiny-M / long-K shapes.  Kept off by
# default: it was validated on targeted shapes only, while the default fused
# path carries the full 200-case + ATK acceptance.
_ENABLE_SPLITK = False


@tla.kernel
def fused_kernel(  # pragma: no cover - device-only CATLASS DSL body
    gm_a: tla.Tensor,
    gm_sa: tla.Tensor,
    gm_b: tla.Tensor,
    gm_sb: tla.Tensor,
    gm_y: tla.Tensor,
    gm_sy: tla.Tensor,
    TILE_M: tla.Constexpr[int],
    TILE_K: tla.Constexpr[int],
    SKIP_CHUNKS: tla.Constexpr[int],
    OUT_FP4: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1
    tm = TILE_M
    tn = _TN
    tk = TILE_K
    half = TILE_M // 2
    sk = tk // 32  # scale groups in one K tile

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    out_fp4 = tla.const_expr(OUT_FP4 == 1)
    emax = 2 if out_fp4 else 8
    emax_shifted = emax * 128

    m = gm_a.origin_shape[0]
    k = gm_a.origin_shape[1]
    n = gm_b.origin_shape[1]
    n_out = n // 2
    k_tiles = (k + tk - 1) // tk
    grid_m = (m + tm - 1) // tm
    grid_n = (n_out + tn - 1) // tn
    total_blocks = grid_m * grid_n

    # ---- flags ----
    l1_ready0 = tla.flag("l1_ready0", tla.arch.MTE2, tla.arch.MTE1)
    l1_ready1 = tla.flag("l1_ready1", tla.arch.MTE2, tla.arch.MTE1)
    l0_ready0 = tla.flag("l0_ready0", tla.arch.MTE1, tla.arch.CUBE)
    l0_ready1 = tla.flag("l0_ready1", tla.arch.MTE1, tla.arch.CUBE)
    l1_free0 = tla.flag("l1_free0", tla.arch.MTE1, tla.arch.MTE2)
    l1_free1 = tla.flag("l1_free1", tla.arch.MTE1, tla.arch.MTE2)
    l0_free0 = tla.flag("l0_free0", tla.arch.CUBE, tla.arch.MTE1)
    l0_free1 = tla.flag("l0_free1", tla.arch.CUBE, tla.arch.MTE1)
    l0c_ready = tla.flag("l0c_ready", tla.arch.CUBE, tla.arch.FIX)
    l0c_free = tla.flag("l0c_free", tla.arch.FIX, tla.arch.CUBE)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    mte3_done = tla.flag("mte3_done", tla.arch.MTE3, tla.arch.VECTOR)

    ub_ready = tla.cross_flag("ub_ready")
    ub_free = tla.cross_flag("ub_free")

    # ---- on-chip buffers ----
    l1a0 = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1a1 = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b_act0 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b_act1 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b_gate0 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b_gate1 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1sa0 = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sa1 = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_act0 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_act1 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_gate0 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_gate1 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)

    l0a0 = tla.allocate(tm * (tk // 2), dtype_a, tla.AddressSpace.l0a, 512)
    l0a1 = tla.allocate(tm * (tk // 2), dtype_a, tla.AddressSpace.l0a, 512)
    l0b_act0 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b_act1 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b_gate0 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b_gate1 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0c_act = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 512)
    l0c_gate = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 512)

    ub_act = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_gate = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    if tla.const_expr(out_fp4):
        ub_y = tla.allocate(half * tn // 2, tla.Int8, tla.AddressSpace.ub, 256)
        ub_codes = tla.allocate(half * tn, tla.Int32, tla.AddressSpace.ub, 256)
    else:
        ub_y = tla.allocate(half * tn // 4, tla.Int32, tla.AddressSpace.ub, 256)
        ub_codes = ub_y
    ub_sy = tla.allocate(half * 8, tla.Int32, tla.AddressSpace.ub, 256)

    c_ref = tla.make_tensor(
        gm_a.ptr,
        tla.make_layout(
            tla.make_shape(tm, tn), tla.make_stride(tn, 1), layoutTag=tla.arch.RowMajor
        ),
    )
    t_l0c_act = tla.make_tensor_like(l0c_act, c_ref, tla.arch.L0Clayout)
    t_l0c_gate = tla.make_tensor_like(l0c_gate, c_ref, tla.arch.L0Clayout)

    # =====================================================================
    # Cube: two MX matmuls per output tile (Act and Gate halves)
    # =====================================================================
    with tla.cube():
        tla.set_flag(l1_free0)
        tla.set_flag(l1_free1)
        tla.set_flag(l0_free0)
        tla.set_flag(l0_free1)
        tla.set_flag(l0c_free)

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_col = block_linear // grid_m
            block_row = block_linear % grid_m

            tla.wait_flag(l0c_free)

            # ---- prologue: K tile 0 -> L1 stage 0 ----
            gm_a_p = tla.get_tile(
                gm_a, tla.make_coord(block_row * tm, 0), tla.make_shape(tm, tk)
            )
            gm_b_act_p = tla.get_tile(
                gm_b, tla.make_coord(0, block_col * tn), tla.make_shape(tk, tn)
            )
            gm_b_gate_p = tla.get_tile(
                gm_b, tla.make_coord(0, n_out + block_col * tn), tla.make_shape(tk, tn)
            )
            gm_sa_p = tla.get_tile(
                gm_sa, tla.make_coord(block_row * tm, 0), tla.make_shape(tm, sk)
            )
            gm_sb_act_p = tla.get_tile(
                gm_sb, tla.make_coord(0, block_col * tn), tla.make_shape(sk, tn)
            )
            gm_sb_gate_p = tla.get_tile(
                gm_sb, tla.make_coord(0, n_out + block_col * tn), tla.make_shape(sk, tn)
            )
            # Tail handling follows the CATLASS mx_matmul reference: the
            # on-chip tile keeps the full L1 extent and only the valid part is
            # copied from GM, so a partial M/N edge needs no host padding.
            # Only K has to be masked -- the k-tile is widened to the whole
            # 64-K block the mmad consumes and the zero tail is filled on chip.
            # M/N pad lanes never reach GM (the epilogue drops them), so stale
            # L1 content there is harmless and is left untouched.
            m_valid = gm_a_p.origin_shape[0]
            k_valid = gm_a_p.origin_shape[1]
            n_valid = gm_b_act_p.origin_shape[1]
            l1b_tag = (
                tla.arch.zN
                if gm_b_act_p.layout_tag in ("RowMajor", "zN")
                else tla.arch.nZ
            )
            t_l1a0 = tla.make_tensor(
                l1a0, tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN)
            )
            t_l1b_act0 = tla.make_tensor(
                l1b_act0, tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag)
            )
            t_l1b_gate0 = tla.make_tensor(
                l1b_gate0, tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag)
            )
            t_l1sa0 = tla.make_tensor(
                l1sa0,
                tla.make_layout_with_tag((tm, sk), tla.Float8E8M0, tla.arch.zZMxScale),
            )
            t_l1sb_act0 = tla.make_tensor(
                l1sb_act0,
                tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale),
            )
            t_l1sb_gate0 = tla.make_tensor(
                l1sb_gate0,
                tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale),
            )
            tla.wait_flag(l1_free0)
            tla.copy(
                tla.get_tile(
                    t_l1a0, tla.make_coord(0, 0), tla.make_shape(m_valid, k_valid)
                ),
                gm_a_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1b_act0, tla.make_coord(0, 0), tla.make_shape(k_valid, n_valid)
                ),
                gm_b_act_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1b_gate0, tla.make_coord(0, 0), tla.make_shape(k_valid, n_valid)
                ),
                gm_b_gate_p,
            )
            tla.copy(
                tla.get_tile(t_l1sa0, tla.make_coord(0, 0), tla.make_shape(m_valid, sk)),
                gm_sa_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1sb_act0, tla.make_coord(0, 0), tla.make_shape(sk, n_valid)
                ),
                gm_sb_act_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1sb_gate0, tla.make_coord(0, 0), tla.make_shape(sk, n_valid)
                ),
                gm_sb_gate_p,
            )
            if k_valid < tk:
                tla.get_tile(
                    t_l1a0,
                    tla.make_coord(0, k_valid),
                    tla.make_shape(m_valid, tk - k_valid),
                ).fill(0)
                tla.get_tile(
                    t_l1b_act0,
                    tla.make_coord(k_valid, 0),
                    tla.make_shape(tk - k_valid, n_valid),
                ).fill(0)
                tla.get_tile(
                    t_l1b_gate0,
                    tla.make_coord(k_valid, 0),
                    tla.make_shape(tk - k_valid, n_valid),
                ).fill(0)
            tla.set_flag(l1_ready0)

            pair_range = tla.range(c0, (k_tiles + 1) // 2, c1)
            for pair in pair_range:
                even = pair * 2
                odd = even + 1

                # prefetch odd -> stage 1
                if odd < k_tiles:
                    gm_a_o = tla.get_tile(
                        gm_a,
                        tla.make_coord(block_row * tm, odd * tk),
                        tla.make_shape(tm, tk),
                    )
                    gm_b_act_o = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_b_gate_o = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, n_out + block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_sa_o = tla.get_tile(
                        gm_sa,
                        tla.make_coord(block_row * tm, odd * sk),
                        tla.make_shape(tm, sk),
                    )
                    gm_sb_act_o = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    gm_sb_gate_o = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, n_out + block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    m_valid_o = gm_a_o.origin_shape[0]
                    k_valid_o = gm_a_o.origin_shape[1]
                    n_valid_o = gm_b_act_o.origin_shape[1]
                    t_l1a1 = tla.make_tensor(
                        l1a1,
                        tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                    )
                    t_l1b_act1 = tla.make_tensor(
                        l1b_act1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1b_gate1 = tla.make_tensor(
                        l1b_gate1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1sa1 = tla.make_tensor(
                        l1sa1,
                        tla.make_layout_with_tag(
                            (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                        ),
                    )
                    t_l1sb_act1 = tla.make_tensor(
                        l1sb_act1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    t_l1sb_gate1 = tla.make_tensor(
                        l1sb_gate1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    tla.wait_flag(l1_free1)
                    tla.copy(
                        tla.get_tile(
                            t_l1a1,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_o, k_valid_o),
                        ),
                        gm_a_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_act1,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_o, n_valid_o),
                        ),
                        gm_b_act_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_gate1,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_o, n_valid_o),
                        ),
                        gm_b_gate_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sa1,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_o, sk),
                        ),
                        gm_sa_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_act1,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_o),
                        ),
                        gm_sb_act_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_gate1,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_o),
                        ),
                        gm_sb_gate_o,
                    )
                    if k_valid_o < tk:
                        tla.get_tile(
                            t_l1a1,
                            tla.make_coord(0, k_valid_o),
                            tla.make_shape(m_valid_o, tk - k_valid_o),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_act1,
                            tla.make_coord(k_valid_o, 0),
                            tla.make_shape(tk - k_valid_o, n_valid_o),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_gate1,
                            tla.make_coord(k_valid_o, 0),
                            tla.make_shape(tk - k_valid_o, n_valid_o),
                        ).fill(0)
                    tla.set_flag(l1_ready1)

                # compute even from stage 0
                gm_a_e = tla.get_tile(
                    gm_a,
                    tla.make_coord(block_row * tm, even * tk),
                    tla.make_shape(tm, tk),
                )
                gm_b_act_e = tla.get_tile(
                    gm_b,
                    tla.make_coord(even * tk, block_col * tn),
                    tla.make_shape(tk, tn),
                )
                gm_b_gate_e = tla.get_tile(
                    gm_b,
                    tla.make_coord(even * tk, n_out + block_col * tn),
                    tla.make_shape(tk, tn),
                )
                gm_sa_e = tla.get_tile(
                    gm_sa,
                    tla.make_coord(block_row * tm, even * sk),
                    tla.make_shape(tm, sk),
                )
                gm_sb_act_e = tla.get_tile(
                    gm_sb,
                    tla.make_coord(even * sk, block_col * tn),
                    tla.make_shape(sk, tn),
                )
                gm_sb_gate_e = tla.get_tile(
                    gm_sb,
                    tla.make_coord(even * sk, n_out + block_col * tn),
                    tla.make_shape(sk, tn),
                )
                t_l1a0_e = tla.make_tensor(
                    l1a0,
                    tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                )
                t_l1b_act0_e = tla.make_tensor(
                    l1b_act0,
                    tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                )
                t_l1b_gate0_e = tla.make_tensor(
                    l1b_gate0,
                    tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                )
                t_l1sa0_e = tla.make_tensor(
                    l1sa0,
                    tla.make_layout_with_tag(
                        (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                    ),
                )
                t_l1sb_act0_e = tla.make_tensor(
                    l1sb_act0,
                    tla.make_layout_with_tag(
                        (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                    ),
                )
                t_l1sb_gate0_e = tla.make_tensor(
                    l1sb_gate0,
                    tla.make_layout_with_tag(
                        (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                    ),
                )
                tla.wait_flag(l1_ready0)
                s_a0 = tla.get_tile(t_l1a0_e, tla.make_coord(0, 0), tla.make_shape(tm, tk // 2))
                s_ba0 = tla.get_tile(t_l1b_act0_e, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                s_bg0 = tla.get_tile(t_l1b_gate0_e, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                s_sa0 = tla.get_tile(t_l1sa0_e, tla.make_coord(0, 0), tla.make_shape(tm, sk // 2))
                s_sba0 = tla.get_tile(t_l1sb_act0_e, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                s_sbg0 = tla.get_tile(t_l1sb_gate0_e, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                t_l0a_0 = tla.make_tensor_like(l0a0, s_a0, tla.arch.zN)
                t_l0ba_0 = tla.make_tensor_like(l0b_act0, s_ba0, tla.arch.nZ)
                t_l0bg_0 = tla.make_tensor_like(l0b_gate0, s_bg0, tla.arch.nZ)
                tla.wait_flag(l0_free0)
                tla.copy(t_l0a_0, s_a0, scale=s_sa0)
                tla.copy(t_l0ba_0, s_ba0, scale=s_sba0)
                tla.copy(t_l0bg_0, s_bg0, scale=s_sbg0)
                tla.set_flag(l0_ready0)
                tla.wait_flag(l0_ready0)
                tla.mmad_mx(t_l0c_act, t_l0a_0, t_l0ba_0, init_c=(even == c0))
                tla.mmad_mx(t_l0c_gate, t_l0a_0, t_l0bg_0, init_c=(even == c0))
                tla.set_flag(l0_free0)

                s_a1 = tla.get_tile(t_l1a0_e, tla.make_coord(0, tk // 2), tla.make_shape(tm, tk // 2))
                s_ba1 = tla.get_tile(t_l1b_act0_e, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                s_bg1 = tla.get_tile(t_l1b_gate0_e, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                s_sa1 = tla.get_tile(t_l1sa0_e, tla.make_coord(0, sk // 2), tla.make_shape(tm, sk // 2))
                s_sba1 = tla.get_tile(t_l1sb_act0_e, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                s_sbg1 = tla.get_tile(t_l1sb_gate0_e, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                t_l0a_1 = tla.make_tensor_like(l0a1, s_a1, tla.arch.zN)
                t_l0ba_1 = tla.make_tensor_like(l0b_act1, s_ba1, tla.arch.nZ)
                t_l0bg_1 = tla.make_tensor_like(l0b_gate1, s_bg1, tla.arch.nZ)
                tla.wait_flag(l0_free1)
                tla.copy(t_l0a_1, s_a1, scale=s_sa1)
                tla.copy(t_l0ba_1, s_ba1, scale=s_sba1)
                tla.copy(t_l0bg_1, s_bg1, scale=s_sbg1)
                tla.set_flag(l1_free0)
                tla.set_flag(l0_ready1)
                tla.wait_flag(l0_ready1)
                tla.mmad_mx(t_l0c_act, t_l0a_1, t_l0ba_1, init_c=False)
                tla.mmad_mx(t_l0c_gate, t_l0a_1, t_l0bg_1, init_c=False)
                tla.set_flag(l0_free1)

                # prefetch next even -> stage 0
                if (even + 2) < k_tiles:
                    gm_a_n = tla.get_tile(
                        gm_a,
                        tla.make_coord(block_row * tm, (even + 2) * tk),
                        tla.make_shape(tm, tk),
                    )
                    gm_b_act_n = tla.get_tile(
                        gm_b,
                        tla.make_coord((even + 2) * tk, block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_b_gate_n = tla.get_tile(
                        gm_b,
                        tla.make_coord((even + 2) * tk, n_out + block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_sa_n = tla.get_tile(
                        gm_sa,
                        tla.make_coord(block_row * tm, (even + 2) * sk),
                        tla.make_shape(tm, sk),
                    )
                    gm_sb_act_n = tla.get_tile(
                        gm_sb,
                        tla.make_coord((even + 2) * sk, block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    gm_sb_gate_n = tla.get_tile(
                        gm_sb,
                        tla.make_coord((even + 2) * sk, n_out + block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    m_valid_n = gm_a_n.origin_shape[0]
                    k_valid_n = gm_a_n.origin_shape[1]
                    n_valid_n = gm_b_act_n.origin_shape[1]
                    t_l1a0_n = tla.make_tensor(
                        l1a0,
                        tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                    )
                    t_l1b_act0_n = tla.make_tensor(
                        l1b_act0,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1b_gate0_n = tla.make_tensor(
                        l1b_gate0,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1sa0_n = tla.make_tensor(
                        l1sa0,
                        tla.make_layout_with_tag(
                            (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                        ),
                    )
                    t_l1sb_act0_n = tla.make_tensor(
                        l1sb_act0,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    t_l1sb_gate0_n = tla.make_tensor(
                        l1sb_gate0,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    tla.wait_flag(l1_free0)
                    tla.copy(
                        tla.get_tile(
                            t_l1a0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_n, k_valid_n),
                        ),
                        gm_a_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_act0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_n, n_valid_n),
                        ),
                        gm_b_act_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_gate0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_n, n_valid_n),
                        ),
                        gm_b_gate_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sa0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_n, sk),
                        ),
                        gm_sa_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_act0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_n),
                        ),
                        gm_sb_act_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_gate0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_n),
                        ),
                        gm_sb_gate_n,
                    )
                    if k_valid_n < tk:
                        tla.get_tile(
                            t_l1a0_n,
                            tla.make_coord(0, k_valid_n),
                            tla.make_shape(m_valid_n, tk - k_valid_n),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_act0_n,
                            tla.make_coord(k_valid_n, 0),
                            tla.make_shape(tk - k_valid_n, n_valid_n),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_gate0_n,
                            tla.make_coord(k_valid_n, 0),
                            tla.make_shape(tk - k_valid_n, n_valid_n),
                        ).fill(0)
                    tla.set_flag(l1_ready0)

                # compute odd from stage 1
                if odd < k_tiles:
                    gm_a_o2 = tla.get_tile(
                        gm_a,
                        tla.make_coord(block_row * tm, odd * tk),
                        tla.make_shape(tm, tk),
                    )
                    gm_b_act_o2 = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_b_gate_o2 = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, n_out + block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_sa_o2 = tla.get_tile(
                        gm_sa,
                        tla.make_coord(block_row * tm, odd * sk),
                        tla.make_shape(tm, sk),
                    )
                    gm_sb_act_o2 = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    gm_sb_gate_o2 = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, n_out + block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    t_l1a1_o = tla.make_tensor(
                        l1a1,
                        tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                    )
                    t_l1b_act1_o = tla.make_tensor(
                        l1b_act1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1b_gate1_o = tla.make_tensor(
                        l1b_gate1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1sa1_o = tla.make_tensor(
                        l1sa1,
                        tla.make_layout_with_tag(
                            (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                        ),
                    )
                    t_l1sb_act1_o = tla.make_tensor(
                        l1sb_act1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    t_l1sb_gate1_o = tla.make_tensor(
                        l1sb_gate1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    tla.wait_flag(l1_ready1)
                    o_a0 = tla.get_tile(t_l1a1_o, tla.make_coord(0, 0), tla.make_shape(tm, tk // 2))
                    o_ba0 = tla.get_tile(t_l1b_act1_o, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                    o_bg0 = tla.get_tile(t_l1b_gate1_o, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                    o_sa0 = tla.get_tile(t_l1sa1_o, tla.make_coord(0, 0), tla.make_shape(tm, sk // 2))
                    o_sba0 = tla.get_tile(t_l1sb_act1_o, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                    o_sbg0 = tla.get_tile(t_l1sb_gate1_o, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                    t_l0a_o0 = tla.make_tensor_like(l0a0, o_a0, tla.arch.zN)
                    t_l0ba_o0 = tla.make_tensor_like(l0b_act0, o_ba0, tla.arch.nZ)
                    t_l0bg_o0 = tla.make_tensor_like(l0b_gate0, o_bg0, tla.arch.nZ)
                    tla.wait_flag(l0_free0)
                    tla.copy(t_l0a_o0, o_a0, scale=o_sa0)
                    tla.copy(t_l0ba_o0, o_ba0, scale=o_sba0)
                    tla.copy(t_l0bg_o0, o_bg0, scale=o_sbg0)
                    tla.set_flag(l0_ready0)
                    tla.wait_flag(l0_ready0)
                    tla.mmad_mx(t_l0c_act, t_l0a_o0, t_l0ba_o0, init_c=False)
                    tla.mmad_mx(t_l0c_gate, t_l0a_o0, t_l0bg_o0, init_c=False)
                    tla.set_flag(l0_free0)

                    o_a1 = tla.get_tile(t_l1a1_o, tla.make_coord(0, tk // 2), tla.make_shape(tm, tk // 2))
                    o_ba1 = tla.get_tile(t_l1b_act1_o, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                    o_bg1 = tla.get_tile(t_l1b_gate1_o, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                    o_sa1 = tla.get_tile(t_l1sa1_o, tla.make_coord(0, sk // 2), tla.make_shape(tm, sk // 2))
                    o_sba1 = tla.get_tile(t_l1sb_act1_o, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                    o_sbg1 = tla.get_tile(t_l1sb_gate1_o, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                    t_l0a_o1 = tla.make_tensor_like(l0a1, o_a1, tla.arch.zN)
                    t_l0ba_o1 = tla.make_tensor_like(l0b_act1, o_ba1, tla.arch.nZ)
                    t_l0bg_o1 = tla.make_tensor_like(l0b_gate1, o_bg1, tla.arch.nZ)
                    tla.wait_flag(l0_free1)
                    tla.copy(t_l0a_o1, o_a1, scale=o_sa1)
                    tla.copy(t_l0ba_o1, o_ba1, scale=o_sba1)
                    tla.copy(t_l0bg_o1, o_bg1, scale=o_sbg1)
                    tla.set_flag(l1_free1)
                    tla.set_flag(l0_ready1)
                    tla.wait_flag(l0_ready1)
                    tla.mmad_mx(t_l0c_act, t_l0a_o1, t_l0ba_o1, init_c=False)
                    tla.mmad_mx(t_l0c_gate, t_l0a_o1, t_l0bg_o1, init_c=False)
                    tla.set_flag(l0_free1)

            tla.set_flag(l0c_ready)
            tla.wait_flag(l0c_ready)

            ub_act_cube = tla.make_tensor_like(ub_act, t_l0c_act, tla.arch.RowMajor)
            ub_gate_cube = tla.make_tensor_like(ub_gate, t_l0c_gate, tla.arch.RowMajor)

            # One UB stage: the vector half must have consumed the previous
            # tile before the next fixpipe overwrites it.  The first tile of
            # each core has no predecessor.

            l0c2ub = tla.params.CopyL0C2DstParams(
                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
            )
            if block_linear != tla.arch.block_idx():
                tla.cross_core_wait_flag(ub_free, tla.arch.FIX)
            tla.copy(ub_act_cube, t_l0c_act, l0c2ub)
            tla.copy(ub_gate_cube, t_l0c_gate, l0c2ub)
            tla.set_flag(l0c_free)
            tla.cross_core_set_flag(ub_ready, tla.arch.FIX)

    # =====================================================================
    # Vector: bf16 RNE -> SwiGLU -> MX quant
    # =====================================================================
    with tla.vector():
        # Seed the MTE3->VECTOR credit for the first tile of each core.  The
        # UB output slots are reused by every tile, so the next tile must not
        # overwrite them before the previous tile's async UB->GM copy is done.
        tla.set_flag(mte3_done)
        m_out = gm_y.origin_shape[0]
        sub_idx = tla.arch.sub_block_idx()
        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_col = block_linear // grid_m
            block_row = block_linear % grid_m
            out_row = block_row * tm + sub_idx * half

            tla.cross_core_wait_flag(ub_ready, tla.arch.VECTOR)
            tla.wait_flag(mte3_done)

            gm_b_ref = tla.make_tensor(
                gm_b.ptr,
                tla.make_layout(
                    tla.make_shape(half, tn),
                    tla.make_stride(tn, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            codes_ref = tla.make_tensor(
                gm_sy.ptr,
                tla.make_layout(
                    tla.make_shape(half, tn),
                    tla.make_stride(tn, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            sy_ref = tla.make_tensor(
                gm_sy.ptr,
                tla.make_layout(
                    tla.make_shape(half, 8),
                    tla.make_stride(8, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            ub_act_half = tla.make_tensor_like(ub_act, gm_b_ref, tla.arch.RowMajor)
            ub_gate_half = tla.make_tensor_like(ub_gate, gm_b_ref, tla.arch.RowMajor)
            ub_sy_half = tla.make_tensor_like(ub_sy, sy_ref, tla.arch.RowMajor)
            if tla.const_expr(out_fp4):
                ub_codes_half = tla.make_tensor_like(
                    ub_codes, codes_ref, tla.arch.RowMajor
                )

            rows = m_out - out_row
            if rows > 0:
                rows = min(rows, half)
                gm_y_cols = tn // 2 if out_fp4 else tn // 4
                gm_y_t = tla.get_tile(
                    gm_y,
                    tla.make_coord(out_row, block_col * gm_y_cols),
                    tla.make_shape(half, gm_y_cols),
                )
                gm_sy_t = tla.get_tile(
                    gm_sy,
                    tla.make_coord(out_row, block_col),
                    tla.make_shape(half, 1),
                )
                ub_y_half = tla.make_tensor_like(ub_y, gm_y_t, tla.arch.RowMajor)
                # Skip the trailing 64-column chunk when this block's valid
                # output width is only 64 (e.g. n_out=64, or the last N block).
                cols_valid = n_out - block_col * tn
                cols_valid = min(cols_valid, tn)
                nchunks = (cols_valid + 63) // 64

                with tla.vec.func(mode="simd"):
                    m_src = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    m_lo = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
                    m_i32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
                    m_fp8 = tla.create_mask(
                        pattern=tla.mask.VL64, dtype=tla.Float8E4M3FN
                    )
                    m_i8 = tla.create_mask(pattern=tla.mask.VL32, dtype=tla.Int8)
                    m_i8_64 = tla.create_mask(pattern=tla.mask.VL64, dtype=tla.Int8)
                    v1 = tla.full(1, tla.Int32)
                    v7FFF = tla.full(0x7FFF, tla.Int32)
                    vMASK = tla.full(-65536, tla.Int32)
                    v7F80 = tla.full(0x7F80, tla.Int32)
                    v0F = tla.full(0x0F, tla.Int32)
                    v7FFFFFFF = tla.full(0x7FFFFFFF, tla.Int32)
                    vFF = tla.full(0xFF, tla.Int32)
                    v7FFFFF = tla.full(0x7FFFFF, tla.Int32)
                    v7FFFF = tla.full(0x7FFFF, tla.Int32)
                    v7E = tla.full(0x7E, tla.Int32)
                    v8 = tla.full(8, tla.Int32)
                    m_i32_16 = tla.create_mask(
                        pattern=tla.mask.VL16, dtype=tla.Int32
                    )

                    scale_zero = tla.full(0, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    v7F00 = tla.full(0x7F00, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    vFF_lane = tla.full(0xFF, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    v7F81_lane = tla.full(0x7F81, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    v0040_lane = tla.full(0x0040, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    for r in tla.range(c0, rows, c1):
                        scale_word = scale_zero
                        if tla.const_expr(SKIP_CHUNKS == 1):
                            for chunk in tla.range(c0, nchunks, c1):
                                col = chunk * 64
                                a_reg = tla.get_tile(
                                    ub_act_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()
                                g_reg = tla.get_tile(
                                    ub_gate_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()

                                # Official fused semantics: activation consumes
                                # the FP32 accumulator directly (no BF16
                                # rounding before SwiGLU).
                                a_bf = a_reg
                                g_bf = g_reg

                                den = tla.add(1.0, tla.exp(tla.neg(a_bf)))
                                s_val = tla.mul(tla.div(a_bf, den), g_bf)

                                s_bits = s_val.bitcast(tla.Int32)
                                s_round = tla.bitwise_and(
                                    tla.add(
                                        tla.add(s_bits, v7FFF),
                                        tla.bitwise_and(tla.shift_right(s_bits, 16), v1),
                                    ),
                                    vMASK,
                                )
                                s_bf16 = tla.shift_right(s_round, 16)
                                s_bf = s_round.bitcast(tla.Float32)

                                exp_bits = tla.bitwise_and(s_bf16, v7F80)
                                max_lo = exp_bits.reduce(tla.ReductionOp.MAX, mask=m_lo)
                                hi_only = tla.where(m_lo, tla.full(0, tla.Int32), exp_bits)
                                max_hi = hi_only.reduce(tla.ReductionOp.MAX, mask=m_src)

                                # Official DynamicMxQuant scale rules: clamp
                                # sub-minimum blocks, mark Inf/NaN blocks with
                                # the E8M0 0xFF byte, and use the special
                                # half-scale bit patterns.
                                special_lo = tla.cmp(max_lo, 0x7F80, "eq")
                                special_hi = tla.cmp(max_hi, 0x7F80, "eq")
                                zero_lo = tla.cmp(max_lo, 0, "eq")
                                zero_hi = tla.cmp(max_hi, 0, "eq")
                                invalid_lo = tla.cmp(max_lo, emax_shifted, "le")
                                invalid_hi = tla.cmp(max_hi, emax_shifted, "le")
                                shared_lo = tla.where(
                                    invalid_lo, scale_zero, tla.sub(max_lo, emax_shifted)
                                )
                                shared_hi = tla.where(
                                    invalid_hi, scale_zero, tla.sub(max_hi, emax_shifted)
                                )

                                byte_lo = tla.shift_right(shared_lo, 7)
                                byte_hi = tla.shift_right(shared_hi, 7)
                                byte_lo = tla.where(special_lo, vFF_lane, byte_lo)
                                byte_hi = tla.where(special_hi, vFF_lane, byte_hi)
                                byte_lo = tla.where(zero_lo, scale_zero, byte_lo)
                                byte_hi = tla.where(zero_hi, scale_zero, byte_hi)

                                half_lo_bits = tla.sub(v7F00, shared_lo)
                                half_hi_bits = tla.sub(v7F00, shared_hi)
                                half_lo_bits = tla.where(
                                    special_lo, v7F81_lane, half_lo_bits
                                )
                                half_hi_bits = tla.where(
                                    special_hi, v7F81_lane, half_hi_bits
                                )
                                half_lo_bits = tla.where(zero_lo, scale_zero, half_lo_bits)
                                half_hi_bits = tla.where(zero_hi, scale_zero, half_hi_bits)
                                half_lo_bits = tla.where(
                                    tla.cmp(shared_lo, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_lo_bits,
                                )
                                half_hi_bits = tla.where(
                                    tla.cmp(shared_hi, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_hi_bits,
                                )
                                half_lo = tla.full(
                                    tla.shift_left(half_lo_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )
                                half_hi = tla.full(
                                    tla.shift_left(half_hi_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )

                                scaled_lo = tla.mul(s_bf, half_lo, mask=m_src)
                                scaled_hi = tla.mul(s_bf, half_hi, mask=m_src)
                                scaled = tla.where(m_lo, scaled_lo, scaled_hi)
                                scaled = tla.where(
                                    tla.cmp(scaled, 0.0, "eq", mask=m_src),
                                    tla.full(0.0, tla.Float32),
                                    scaled,
                                )

                                if tla.const_expr(out_fp4):
                                    abs_v = tla.abs(scaled, mask=m_src)
                                    mag = tla.full(0, tla.Int32)
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.25, "gt", mask=m_src),
                                        tla.full(1, tla.Int32),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 3.5, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.25, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 2.5, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 5.0, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    code_bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(tla.shift_right(code_bits, 31), v1),
                                        3,
                                    )
                                    code = tla.bitwise_or(mag, sign)
                                    code_i32 = tla.get_tile(
                                        ub_codes_half,
                                        tla.make_coord(r, col),
                                        tla.make_shape(1, 64),
                                    )
                                    code_i32.store(code, mask=m_i32)
                                else:
                                    # Manual E4M3FN encode (the vector fp8 store packs
                                    # four bytes per 32-bit lane; encode and pack here).
                                    bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(
                                            tla.shift_right(bits, 31), v1
                                        ),
                                        7,
                                    )
                                    abs_bits = tla.bitwise_and(bits, v7FFFFFFF)
                                    exp_field = tla.bitwise_and(
                                        tla.shift_right(abs_bits, 23), vFF
                                    )
                                    mant = tla.bitwise_and(abs_bits, v7FFFFF)
                                    mant3 = tla.shift_right(
                                        tla.add(
                                            tla.add(mant, v7FFFF),
                                            tla.bitwise_and(
                                                tla.shift_right(mant, 20), v1
                                            ),
                                        ),
                                        20,
                                    )
                                    code_exp = tla.sub(exp_field, 120)
                                    carry = tla.cmp(mant3, 8, "ge", mask=m_src)
                                    code_exp = tla.where(
                                        carry, tla.add(code_exp, 1), code_exp
                                    )
                                    mant3 = tla.where(
                                        carry, tla.sub(mant3, 8), mant3
                                    )
                                    normal_code = tla.bitwise_or(
                                        tla.shift_left(code_exp, 3), mant3
                                    )
                                    normal_code = tla.where(
                                        tla.cmp(code_exp, 16, "ge", mask=m_src),
                                        v7E,
                                        normal_code,
                                    )
                                    normal_code = tla.min(normal_code, v7E)
                                    sub_f = tla.mul(
                                        tla.abs(scaled, mask=m_src), 512.0
                                    )
                                    sub_i = sub_f.to(
                                        tla.Int32,
                                        tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO,
                                            sat_mode=tla.params.SatMode.NOSAT,
                                            round_mode=tla.params.RoundMode.CAST_TRUNC,
                                        ),
                                        m_src,
                                    )
                                    frac = tla.sub(
                                        sub_f,
                                        sub_i.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.NOSAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            ),
                                            m_src,
                                        ),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "gt", mask=m_src),
                                        v1,
                                        tla.full(0, tla.Int32),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "eq", mask=m_src),
                                        tla.bitwise_and(sub_i, v1),
                                        up,
                                    )
                                    sub_code = tla.add(sub_i, up)
                                    sub_code = tla.min(sub_code, 8)
                                    mag = tla.where(
                                        tla.cmp(exp_field, 121, "ge", mask=m_src),
                                        normal_code,
                                        sub_code,
                                    )
                                    code = tla.bitwise_or(sign, mag)
                                    zero32 = tla.full(0, tla.Int32)
                                    even1, odd1 = tla.deinterleave(code, zero32)
                                    p0, p2 = tla.deinterleave(even1, zero32)
                                    p1, p3 = tla.deinterleave(odd1, zero32)
                                    word = tla.bitwise_or(
                                        tla.bitwise_or(
                                            p0, tla.shift_left(p1, 8)
                                        ),
                                        tla.bitwise_or(
                                            tla.shift_left(p2, 16),
                                            tla.shift_left(p3, 24),
                                        ),
                                    )
                                    y_chunk = tla.get_tile(
                                        ub_y_half,
                                        tla.make_coord(r, col // 4),
                                        tla.make_shape(1, 16),
                                    )
                                    y_chunk.store(word, mask=m_i32_16)

                                word = tla.bitwise_or(
                                    byte_lo, tla.shift_left(byte_hi, 8)
                                )
                                scale_word = tla.bitwise_or(
                                    scale_word, tla.shift_left(word, chunk * 16)
                                )
                        else:
                            for chunk in tla.range_constexpr(0, 2):
                                col = chunk * 64
                                a_reg = tla.get_tile(
                                    ub_act_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()
                                g_reg = tla.get_tile(
                                    ub_gate_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()

                                # Official fused semantics: activation consumes
                                # the FP32 accumulator directly (no BF16
                                # rounding before SwiGLU).
                                a_bf = a_reg
                                g_bf = g_reg

                                den = tla.add(1.0, tla.exp(tla.neg(a_bf)))
                                s_val = tla.mul(tla.div(a_bf, den), g_bf)

                                s_bits = s_val.bitcast(tla.Int32)
                                s_round = tla.bitwise_and(
                                    tla.add(
                                        tla.add(s_bits, v7FFF),
                                        tla.bitwise_and(tla.shift_right(s_bits, 16), v1),
                                    ),
                                    vMASK,
                                )
                                s_bf16 = tla.shift_right(s_round, 16)
                                s_bf = s_round.bitcast(tla.Float32)

                                exp_bits = tla.bitwise_and(s_bf16, v7F80)
                                max_lo = exp_bits.reduce(tla.ReductionOp.MAX, mask=m_lo)
                                hi_only = tla.where(m_lo, tla.full(0, tla.Int32), exp_bits)
                                max_hi = hi_only.reduce(tla.ReductionOp.MAX, mask=m_src)

                                # Official DynamicMxQuant scale rules: clamp
                                # sub-minimum blocks, mark Inf/NaN blocks with
                                # the E8M0 0xFF byte, and use the special
                                # half-scale bit patterns.
                                special_lo = tla.cmp(max_lo, 0x7F80, "eq")
                                special_hi = tla.cmp(max_hi, 0x7F80, "eq")
                                zero_lo = tla.cmp(max_lo, 0, "eq")
                                zero_hi = tla.cmp(max_hi, 0, "eq")
                                invalid_lo = tla.cmp(max_lo, emax_shifted, "le")
                                invalid_hi = tla.cmp(max_hi, emax_shifted, "le")
                                shared_lo = tla.where(
                                    invalid_lo, scale_zero, tla.sub(max_lo, emax_shifted)
                                )
                                shared_hi = tla.where(
                                    invalid_hi, scale_zero, tla.sub(max_hi, emax_shifted)
                                )

                                byte_lo = tla.shift_right(shared_lo, 7)
                                byte_hi = tla.shift_right(shared_hi, 7)
                                byte_lo = tla.where(special_lo, vFF_lane, byte_lo)
                                byte_hi = tla.where(special_hi, vFF_lane, byte_hi)
                                byte_lo = tla.where(zero_lo, scale_zero, byte_lo)
                                byte_hi = tla.where(zero_hi, scale_zero, byte_hi)

                                half_lo_bits = tla.sub(v7F00, shared_lo)
                                half_hi_bits = tla.sub(v7F00, shared_hi)
                                half_lo_bits = tla.where(
                                    special_lo, v7F81_lane, half_lo_bits
                                )
                                half_hi_bits = tla.where(
                                    special_hi, v7F81_lane, half_hi_bits
                                )
                                half_lo_bits = tla.where(zero_lo, scale_zero, half_lo_bits)
                                half_hi_bits = tla.where(zero_hi, scale_zero, half_hi_bits)
                                half_lo_bits = tla.where(
                                    tla.cmp(shared_lo, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_lo_bits,
                                )
                                half_hi_bits = tla.where(
                                    tla.cmp(shared_hi, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_hi_bits,
                                )
                                half_lo = tla.full(
                                    tla.shift_left(half_lo_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )
                                half_hi = tla.full(
                                    tla.shift_left(half_hi_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )

                                scaled_lo = tla.mul(s_bf, half_lo, mask=m_src)
                                scaled_hi = tla.mul(s_bf, half_hi, mask=m_src)
                                scaled = tla.where(m_lo, scaled_lo, scaled_hi)
                                scaled = tla.where(
                                    tla.cmp(scaled, 0.0, "eq", mask=m_src),
                                    tla.full(0.0, tla.Float32),
                                    scaled,
                                )

                                if tla.const_expr(out_fp4):
                                    abs_v = tla.abs(scaled, mask=m_src)
                                    mag = tla.full(0, tla.Int32)
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.25, "gt", mask=m_src),
                                        tla.full(1, tla.Int32),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 3.5, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.25, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 2.5, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 5.0, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    code_bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(tla.shift_right(code_bits, 31), v1),
                                        3,
                                    )
                                    code = tla.bitwise_or(mag, sign)
                                    code_i32 = tla.get_tile(
                                        ub_codes_half,
                                        tla.make_coord(r, col),
                                        tla.make_shape(1, 64),
                                    )
                                    code_i32.store(code, mask=m_i32)
                                else:
                                    # Manual E4M3FN encode (the vector fp8 store packs
                                    # four bytes per 32-bit lane; encode and pack here).
                                    bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(
                                            tla.shift_right(bits, 31), v1
                                        ),
                                        7,
                                    )
                                    abs_bits = tla.bitwise_and(bits, v7FFFFFFF)
                                    exp_field = tla.bitwise_and(
                                        tla.shift_right(abs_bits, 23), vFF
                                    )
                                    mant = tla.bitwise_and(abs_bits, v7FFFFF)
                                    mant3 = tla.shift_right(
                                        tla.add(
                                            tla.add(mant, v7FFFF),
                                            tla.bitwise_and(
                                                tla.shift_right(mant, 20), v1
                                            ),
                                        ),
                                        20,
                                    )
                                    code_exp = tla.sub(exp_field, 120)
                                    carry = tla.cmp(mant3, 8, "ge", mask=m_src)
                                    code_exp = tla.where(
                                        carry, tla.add(code_exp, 1), code_exp
                                    )
                                    mant3 = tla.where(
                                        carry, tla.sub(mant3, 8), mant3
                                    )
                                    normal_code = tla.bitwise_or(
                                        tla.shift_left(code_exp, 3), mant3
                                    )
                                    normal_code = tla.where(
                                        tla.cmp(code_exp, 16, "ge", mask=m_src),
                                        v7E,
                                        normal_code,
                                    )
                                    normal_code = tla.min(normal_code, v7E)
                                    sub_f = tla.mul(
                                        tla.abs(scaled, mask=m_src), 512.0
                                    )
                                    sub_i = sub_f.to(
                                        tla.Int32,
                                        tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO,
                                            sat_mode=tla.params.SatMode.NOSAT,
                                            round_mode=tla.params.RoundMode.CAST_TRUNC,
                                        ),
                                        m_src,
                                    )
                                    frac = tla.sub(
                                        sub_f,
                                        sub_i.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.NOSAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            ),
                                            m_src,
                                        ),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "gt", mask=m_src),
                                        v1,
                                        tla.full(0, tla.Int32),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "eq", mask=m_src),
                                        tla.bitwise_and(sub_i, v1),
                                        up,
                                    )
                                    sub_code = tla.add(sub_i, up)
                                    sub_code = tla.min(sub_code, 8)
                                    mag = tla.where(
                                        tla.cmp(exp_field, 121, "ge", mask=m_src),
                                        normal_code,
                                        sub_code,
                                    )
                                    code = tla.bitwise_or(sign, mag)
                                    zero32 = tla.full(0, tla.Int32)
                                    even1, odd1 = tla.deinterleave(code, zero32)
                                    p0, p2 = tla.deinterleave(even1, zero32)
                                    p1, p3 = tla.deinterleave(odd1, zero32)
                                    word = tla.bitwise_or(
                                        tla.bitwise_or(
                                            p0, tla.shift_left(p1, 8)
                                        ),
                                        tla.bitwise_or(
                                            tla.shift_left(p2, 16),
                                            tla.shift_left(p3, 24),
                                        ),
                                    )
                                    y_chunk = tla.get_tile(
                                        ub_y_half,
                                        tla.make_coord(r, col // 4),
                                        tla.make_shape(1, 16),
                                    )
                                    y_chunk.store(word, mask=m_i32_16)

                                word = tla.bitwise_or(
                                    byte_lo, tla.shift_left(byte_hi, 8)
                                )
                                scale_word = tla.bitwise_or(
                                    scale_word, tla.shift_left(word, chunk * 16)
                                )
                        sy_cell = tla.get_tile(
                            ub_sy_half,
                            tla.make_coord(r, 0),
                            tla.make_shape(1, 1),
                        )
                        sy_cell.store(scale_word)

                if tla.const_expr(out_fp4):
                    tla.pipe_barrier(tla.pipes.ALL)
                    valid_bytes = gm_y_t.origin_shape[1]
                    flat_codes = tla.make_tensor(
                        ub_codes,
                        tla.make_layout(
                            tla.make_shape(half * tn), tla.make_stride(1)
                        ),
                    )
                    flat_y = tla.make_tensor(
                        ub_y,
                        tla.make_layout(
                            tla.make_shape(half * tn // 2), tla.make_stride(1)
                        ),
                    )
                    with tla.vec.func(mode="simt", thread_block_dim=256):
                        tid, _, _ = tla.arch.thread_idx()
                        nthreads, _, _ = tla.arch.thread_block_dim()
                        for p in tla.range(tid, half * valid_bytes, nthreads):
                            row = p // valid_bytes
                            col = p - row * valid_bytes
                            c0v = flat_codes[row * tn + 2 * col]
                            c1v = flat_codes[row * tn + 2 * col + 1]
                            flat_y[p] = ((c0v & 0x0F) | ((c1v & 0x0F) << 4)).to(
                                tla.Int8
                            )
                    tla.pipe_barrier(tla.pipes.ALL)

                tla.set_flag(vec_done)
                tla.wait_flag(vec_done)
                tla.copy(gm_y_t, ub_y_half)
                tla.copy(gm_sy_t, ub_sy_half)
            tla.set_flag(mte3_done)
            tla.cross_core_set_flag(ub_free, tla.arch.VECTOR)


@tla.kernel
def mm_partial(  # pragma: no cover - device-only CATLASS DSL body
    gm_a: tla.Tensor,
    gm_sa: tla.Tensor,
    gm_b: tla.Tensor,
    gm_sb: tla.Tensor,
    gm_ca: tla.Tensor,
    gm_cg: tla.Tensor,
    TILE_M: tla.Constexpr[int],
    TILE_K: tla.Constexpr[int],
    SPLIT: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1
    tm = TILE_M
    tn = _TN
    tk = TILE_K
    sk = tk // 32
    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    m = gm_a.origin_shape[0]
    k = gm_a.origin_shape[1]
    n = gm_b.origin_shape[1]
    n_out = n // 2
    per = ((k + tk - 1) // tk) // SPLIT
    grid_m = (m + tm - 1) // tm
    grid_n = (n_out + tn - 1) // tn
    total_blocks = grid_m * grid_n * SPLIT

    l1_ready0 = tla.flag("l1_ready0", tla.arch.MTE2, tla.arch.MTE1)
    l1_ready1 = tla.flag("l1_ready1", tla.arch.MTE2, tla.arch.MTE1)
    l0_ready0 = tla.flag("l0_ready0", tla.arch.MTE1, tla.arch.CUBE)
    l0_ready1 = tla.flag("l0_ready1", tla.arch.MTE1, tla.arch.CUBE)
    l1_free0 = tla.flag("l1_free0", tla.arch.MTE1, tla.arch.MTE2)
    l1_free1 = tla.flag("l1_free1", tla.arch.MTE1, tla.arch.MTE2)
    l0_free0 = tla.flag("l0_free0", tla.arch.CUBE, tla.arch.MTE1)
    l0_free1 = tla.flag("l0_free1", tla.arch.CUBE, tla.arch.MTE1)
    l0c_ready = tla.flag("l0c_ready", tla.arch.CUBE, tla.arch.FIX)
    l0c_free = tla.flag("l0c_free", tla.arch.FIX, tla.arch.CUBE)

    l1a0 = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1a1 = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b_act0 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b_act1 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b_gate0 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b_gate1 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1sa0 = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sa1 = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_act0 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_act1 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_gate0 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_gate1 = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0a0 = tla.allocate(tm * (tk // 2), dtype_a, tla.AddressSpace.l0a, 512)
    l0a1 = tla.allocate(tm * (tk // 2), dtype_a, tla.AddressSpace.l0a, 512)
    l0b_act0 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b_act1 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b_gate0 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b_gate1 = tla.allocate((tk // 2) * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0c_act = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 512)
    l0c_gate = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 512)

    c_ref = tla.make_tensor(
        gm_a.ptr,
        tla.make_layout(
            tla.make_shape(tm, tn), tla.make_stride(tn, 1), layoutTag=tla.arch.RowMajor
        ),
    )
    t_l0c_act = tla.make_tensor_like(l0c_act, c_ref, tla.arch.L0Clayout)
    t_l0c_gate = tla.make_tensor_like(l0c_gate, c_ref, tla.arch.L0Clayout)

    with tla.cube():
        tla.set_flag(l1_free0)
        tla.set_flag(l1_free1)
        tla.set_flag(l0_free0)
        tla.set_flag(l0_free1)
        tla.set_flag(l0c_free)

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            split_idx = block_linear % SPLIT
            tile_lin = block_linear // SPLIT
            block_col = tile_lin // grid_m
            block_row = tile_lin % grid_m
            kb_local = split_idx * per

            tla.wait_flag(l0c_free)

            # ---- prologue: K tile 0 -> L1 stage 0 ----
            gm_a_p = tla.get_tile(
                gm_a, tla.make_coord(block_row * tm, kb_local * tk), tla.make_shape(tm, tk)
            )
            gm_b_act_p = tla.get_tile(
                gm_b, tla.make_coord(kb_local * tk, block_col * tn), tla.make_shape(tk, tn)
            )
            gm_b_gate_p = tla.get_tile(
                gm_b, tla.make_coord(kb_local * tk, n_out + block_col * tn), tla.make_shape(tk, tn)
            )
            gm_sa_p = tla.get_tile(
                gm_sa, tla.make_coord(block_row * tm, kb_local * sk), tla.make_shape(tm, sk)
            )
            gm_sb_act_p = tla.get_tile(
                gm_sb, tla.make_coord(kb_local * sk, block_col * tn), tla.make_shape(sk, tn)
            )
            gm_sb_gate_p = tla.get_tile(
                gm_sb, tla.make_coord(kb_local * sk, n_out + block_col * tn), tla.make_shape(sk, tn)
            )
            # Tail handling follows the CATLASS mx_matmul reference: the
            # on-chip tile keeps the full L1 extent and only the valid part is
            # copied from GM, so a partial M/N edge needs no host padding.
            # Only K has to be masked -- the k-tile is widened to the whole
            # 64-K block the mmad consumes and the zero tail is filled on chip.
            # M/N pad lanes never reach GM (the epilogue drops them), so stale
            # L1 content there is harmless and is left untouched.
            m_valid = gm_a_p.origin_shape[0]
            k_valid = gm_a_p.origin_shape[1]
            n_valid = gm_b_act_p.origin_shape[1]
            l1b_tag = (
                tla.arch.zN
                if gm_b_act_p.layout_tag in ("RowMajor", "zN")
                else tla.arch.nZ
            )
            t_l1a0 = tla.make_tensor(
                l1a0, tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN)
            )
            t_l1b_act0 = tla.make_tensor(
                l1b_act0, tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag)
            )
            t_l1b_gate0 = tla.make_tensor(
                l1b_gate0, tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag)
            )
            t_l1sa0 = tla.make_tensor(
                l1sa0,
                tla.make_layout_with_tag((tm, sk), tla.Float8E8M0, tla.arch.zZMxScale),
            )
            t_l1sb_act0 = tla.make_tensor(
                l1sb_act0,
                tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale),
            )
            t_l1sb_gate0 = tla.make_tensor(
                l1sb_gate0,
                tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale),
            )
            tla.wait_flag(l1_free0)
            tla.copy(
                tla.get_tile(
                    t_l1a0, tla.make_coord(0, 0), tla.make_shape(m_valid, k_valid)
                ),
                gm_a_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1b_act0, tla.make_coord(0, 0), tla.make_shape(k_valid, n_valid)
                ),
                gm_b_act_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1b_gate0, tla.make_coord(0, 0), tla.make_shape(k_valid, n_valid)
                ),
                gm_b_gate_p,
            )
            tla.copy(
                tla.get_tile(t_l1sa0, tla.make_coord(0, 0), tla.make_shape(m_valid, sk)),
                gm_sa_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1sb_act0, tla.make_coord(0, 0), tla.make_shape(sk, n_valid)
                ),
                gm_sb_act_p,
            )
            tla.copy(
                tla.get_tile(
                    t_l1sb_gate0, tla.make_coord(0, 0), tla.make_shape(sk, n_valid)
                ),
                gm_sb_gate_p,
            )
            if k_valid < tk:
                tla.get_tile(
                    t_l1a0,
                    tla.make_coord(0, k_valid),
                    tla.make_shape(m_valid, tk - k_valid),
                ).fill(0)
                tla.get_tile(
                    t_l1b_act0,
                    tla.make_coord(k_valid, 0),
                    tla.make_shape(tk - k_valid, n_valid),
                ).fill(0)
                tla.get_tile(
                    t_l1b_gate0,
                    tla.make_coord(k_valid, 0),
                    tla.make_shape(tk - k_valid, n_valid),
                ).fill(0)
            tla.set_flag(l1_ready0)

            pair_range = tla.range(c0, (per + 1) // 2, c1)
            for pair in pair_range:
                even = kb_local + pair * 2
                odd = even + 1

                # prefetch odd -> stage 1
                if odd < kb_local + per:
                    gm_a_o = tla.get_tile(
                        gm_a,
                        tla.make_coord(block_row * tm, odd * tk),
                        tla.make_shape(tm, tk),
                    )
                    gm_b_act_o = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_b_gate_o = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, n_out + block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_sa_o = tla.get_tile(
                        gm_sa,
                        tla.make_coord(block_row * tm, odd * sk),
                        tla.make_shape(tm, sk),
                    )
                    gm_sb_act_o = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    gm_sb_gate_o = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, n_out + block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    m_valid_o = gm_a_o.origin_shape[0]
                    k_valid_o = gm_a_o.origin_shape[1]
                    n_valid_o = gm_b_act_o.origin_shape[1]
                    t_l1a1 = tla.make_tensor(
                        l1a1,
                        tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                    )
                    t_l1b_act1 = tla.make_tensor(
                        l1b_act1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1b_gate1 = tla.make_tensor(
                        l1b_gate1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1sa1 = tla.make_tensor(
                        l1sa1,
                        tla.make_layout_with_tag(
                            (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                        ),
                    )
                    t_l1sb_act1 = tla.make_tensor(
                        l1sb_act1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    t_l1sb_gate1 = tla.make_tensor(
                        l1sb_gate1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    tla.wait_flag(l1_free1)
                    tla.copy(
                        tla.get_tile(
                            t_l1a1,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_o, k_valid_o),
                        ),
                        gm_a_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_act1,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_o, n_valid_o),
                        ),
                        gm_b_act_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_gate1,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_o, n_valid_o),
                        ),
                        gm_b_gate_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sa1,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_o, sk),
                        ),
                        gm_sa_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_act1,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_o),
                        ),
                        gm_sb_act_o,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_gate1,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_o),
                        ),
                        gm_sb_gate_o,
                    )
                    if k_valid_o < tk:
                        tla.get_tile(
                            t_l1a1,
                            tla.make_coord(0, k_valid_o),
                            tla.make_shape(m_valid_o, tk - k_valid_o),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_act1,
                            tla.make_coord(k_valid_o, 0),
                            tla.make_shape(tk - k_valid_o, n_valid_o),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_gate1,
                            tla.make_coord(k_valid_o, 0),
                            tla.make_shape(tk - k_valid_o, n_valid_o),
                        ).fill(0)
                    tla.set_flag(l1_ready1)

                # compute even from stage 0
                gm_a_e = tla.get_tile(
                    gm_a,
                    tla.make_coord(block_row * tm, even * tk),
                    tla.make_shape(tm, tk),
                )
                gm_b_act_e = tla.get_tile(
                    gm_b,
                    tla.make_coord(even * tk, block_col * tn),
                    tla.make_shape(tk, tn),
                )
                gm_b_gate_e = tla.get_tile(
                    gm_b,
                    tla.make_coord(even * tk, n_out + block_col * tn),
                    tla.make_shape(tk, tn),
                )
                gm_sa_e = tla.get_tile(
                    gm_sa,
                    tla.make_coord(block_row * tm, even * sk),
                    tla.make_shape(tm, sk),
                )
                gm_sb_act_e = tla.get_tile(
                    gm_sb,
                    tla.make_coord(even * sk, block_col * tn),
                    tla.make_shape(sk, tn),
                )
                gm_sb_gate_e = tla.get_tile(
                    gm_sb,
                    tla.make_coord(even * sk, n_out + block_col * tn),
                    tla.make_shape(sk, tn),
                )
                t_l1a0_e = tla.make_tensor(
                    l1a0,
                    tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                )
                t_l1b_act0_e = tla.make_tensor(
                    l1b_act0,
                    tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                )
                t_l1b_gate0_e = tla.make_tensor(
                    l1b_gate0,
                    tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                )
                t_l1sa0_e = tla.make_tensor(
                    l1sa0,
                    tla.make_layout_with_tag(
                        (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                    ),
                )
                t_l1sb_act0_e = tla.make_tensor(
                    l1sb_act0,
                    tla.make_layout_with_tag(
                        (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                    ),
                )
                t_l1sb_gate0_e = tla.make_tensor(
                    l1sb_gate0,
                    tla.make_layout_with_tag(
                        (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                    ),
                )
                tla.wait_flag(l1_ready0)
                s_a0 = tla.get_tile(t_l1a0_e, tla.make_coord(0, 0), tla.make_shape(tm, tk // 2))
                s_ba0 = tla.get_tile(t_l1b_act0_e, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                s_bg0 = tla.get_tile(t_l1b_gate0_e, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                s_sa0 = tla.get_tile(t_l1sa0_e, tla.make_coord(0, 0), tla.make_shape(tm, sk // 2))
                s_sba0 = tla.get_tile(t_l1sb_act0_e, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                s_sbg0 = tla.get_tile(t_l1sb_gate0_e, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                t_l0a_0 = tla.make_tensor_like(l0a0, s_a0, tla.arch.zN)
                t_l0ba_0 = tla.make_tensor_like(l0b_act0, s_ba0, tla.arch.nZ)
                t_l0bg_0 = tla.make_tensor_like(l0b_gate0, s_bg0, tla.arch.nZ)
                tla.wait_flag(l0_free0)
                tla.copy(t_l0a_0, s_a0, scale=s_sa0)
                tla.copy(t_l0ba_0, s_ba0, scale=s_sba0)
                tla.copy(t_l0bg_0, s_bg0, scale=s_sbg0)
                tla.set_flag(l0_ready0)
                tla.wait_flag(l0_ready0)
                tla.mmad_mx(t_l0c_act, t_l0a_0, t_l0ba_0, init_c=(even == kb_local))
                tla.mmad_mx(t_l0c_gate, t_l0a_0, t_l0bg_0, init_c=(even == kb_local))
                tla.set_flag(l0_free0)

                s_a1 = tla.get_tile(t_l1a0_e, tla.make_coord(0, tk // 2), tla.make_shape(tm, tk // 2))
                s_ba1 = tla.get_tile(t_l1b_act0_e, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                s_bg1 = tla.get_tile(t_l1b_gate0_e, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                s_sa1 = tla.get_tile(t_l1sa0_e, tla.make_coord(0, sk // 2), tla.make_shape(tm, sk // 2))
                s_sba1 = tla.get_tile(t_l1sb_act0_e, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                s_sbg1 = tla.get_tile(t_l1sb_gate0_e, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                t_l0a_1 = tla.make_tensor_like(l0a1, s_a1, tla.arch.zN)
                t_l0ba_1 = tla.make_tensor_like(l0b_act1, s_ba1, tla.arch.nZ)
                t_l0bg_1 = tla.make_tensor_like(l0b_gate1, s_bg1, tla.arch.nZ)
                tla.wait_flag(l0_free1)
                tla.copy(t_l0a_1, s_a1, scale=s_sa1)
                tla.copy(t_l0ba_1, s_ba1, scale=s_sba1)
                tla.copy(t_l0bg_1, s_bg1, scale=s_sbg1)
                tla.set_flag(l1_free0)
                tla.set_flag(l0_ready1)
                tla.wait_flag(l0_ready1)
                tla.mmad_mx(t_l0c_act, t_l0a_1, t_l0ba_1, init_c=False)
                tla.mmad_mx(t_l0c_gate, t_l0a_1, t_l0bg_1, init_c=False)
                tla.set_flag(l0_free1)

                # prefetch next even -> stage 0
                if (even + 2) < kb_local + per:
                    gm_a_n = tla.get_tile(
                        gm_a,
                        tla.make_coord(block_row * tm, (even + 2) * tk),
                        tla.make_shape(tm, tk),
                    )
                    gm_b_act_n = tla.get_tile(
                        gm_b,
                        tla.make_coord((even + 2) * tk, block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_b_gate_n = tla.get_tile(
                        gm_b,
                        tla.make_coord((even + 2) * tk, n_out + block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_sa_n = tla.get_tile(
                        gm_sa,
                        tla.make_coord(block_row * tm, (even + 2) * sk),
                        tla.make_shape(tm, sk),
                    )
                    gm_sb_act_n = tla.get_tile(
                        gm_sb,
                        tla.make_coord((even + 2) * sk, block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    gm_sb_gate_n = tla.get_tile(
                        gm_sb,
                        tla.make_coord((even + 2) * sk, n_out + block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    m_valid_n = gm_a_n.origin_shape[0]
                    k_valid_n = gm_a_n.origin_shape[1]
                    n_valid_n = gm_b_act_n.origin_shape[1]
                    t_l1a0_n = tla.make_tensor(
                        l1a0,
                        tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                    )
                    t_l1b_act0_n = tla.make_tensor(
                        l1b_act0,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1b_gate0_n = tla.make_tensor(
                        l1b_gate0,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1sa0_n = tla.make_tensor(
                        l1sa0,
                        tla.make_layout_with_tag(
                            (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                        ),
                    )
                    t_l1sb_act0_n = tla.make_tensor(
                        l1sb_act0,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    t_l1sb_gate0_n = tla.make_tensor(
                        l1sb_gate0,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    tla.wait_flag(l1_free0)
                    tla.copy(
                        tla.get_tile(
                            t_l1a0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_n, k_valid_n),
                        ),
                        gm_a_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_act0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_n, n_valid_n),
                        ),
                        gm_b_act_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1b_gate0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(k_valid_n, n_valid_n),
                        ),
                        gm_b_gate_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sa0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(m_valid_n, sk),
                        ),
                        gm_sa_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_act0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_n),
                        ),
                        gm_sb_act_n,
                    )
                    tla.copy(
                        tla.get_tile(
                            t_l1sb_gate0_n,
                            tla.make_coord(0, 0),
                            tla.make_shape(sk, n_valid_n),
                        ),
                        gm_sb_gate_n,
                    )
                    if k_valid_n < tk:
                        tla.get_tile(
                            t_l1a0_n,
                            tla.make_coord(0, k_valid_n),
                            tla.make_shape(m_valid_n, tk - k_valid_n),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_act0_n,
                            tla.make_coord(k_valid_n, 0),
                            tla.make_shape(tk - k_valid_n, n_valid_n),
                        ).fill(0)
                        tla.get_tile(
                            t_l1b_gate0_n,
                            tla.make_coord(k_valid_n, 0),
                            tla.make_shape(tk - k_valid_n, n_valid_n),
                        ).fill(0)
                    tla.set_flag(l1_ready0)

                # compute odd from stage 1
                if odd < kb_local + per:
                    gm_a_o2 = tla.get_tile(
                        gm_a,
                        tla.make_coord(block_row * tm, odd * tk),
                        tla.make_shape(tm, tk),
                    )
                    gm_b_act_o2 = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_b_gate_o2 = tla.get_tile(
                        gm_b,
                        tla.make_coord(odd * tk, n_out + block_col * tn),
                        tla.make_shape(tk, tn),
                    )
                    gm_sa_o2 = tla.get_tile(
                        gm_sa,
                        tla.make_coord(block_row * tm, odd * sk),
                        tla.make_shape(tm, sk),
                    )
                    gm_sb_act_o2 = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    gm_sb_gate_o2 = tla.get_tile(
                        gm_sb,
                        tla.make_coord(odd * sk, n_out + block_col * tn),
                        tla.make_shape(sk, tn),
                    )
                    t_l1a1_o = tla.make_tensor(
                        l1a1,
                        tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN),
                    )
                    t_l1b_act1_o = tla.make_tensor(
                        l1b_act1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1b_gate1_o = tla.make_tensor(
                        l1b_gate1,
                        tla.make_layout_with_tag((tk, tn), dtype_b, l1b_tag),
                    )
                    t_l1sa1_o = tla.make_tensor(
                        l1sa1,
                        tla.make_layout_with_tag(
                            (tm, sk), tla.Float8E8M0, tla.arch.zZMxScale
                        ),
                    )
                    t_l1sb_act1_o = tla.make_tensor(
                        l1sb_act1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    t_l1sb_gate1_o = tla.make_tensor(
                        l1sb_gate1,
                        tla.make_layout_with_tag(
                            (sk, tn), tla.Float8E8M0, tla.arch.nNMxScale
                        ),
                    )
                    tla.wait_flag(l1_ready1)
                    o_a0 = tla.get_tile(t_l1a1_o, tla.make_coord(0, 0), tla.make_shape(tm, tk // 2))
                    o_ba0 = tla.get_tile(t_l1b_act1_o, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                    o_bg0 = tla.get_tile(t_l1b_gate1_o, tla.make_coord(0, 0), tla.make_shape(tk // 2, tn))
                    o_sa0 = tla.get_tile(t_l1sa1_o, tla.make_coord(0, 0), tla.make_shape(tm, sk // 2))
                    o_sba0 = tla.get_tile(t_l1sb_act1_o, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                    o_sbg0 = tla.get_tile(t_l1sb_gate1_o, tla.make_coord(0, 0), tla.make_shape(sk // 2, tn))
                    t_l0a_o0 = tla.make_tensor_like(l0a0, o_a0, tla.arch.zN)
                    t_l0ba_o0 = tla.make_tensor_like(l0b_act0, o_ba0, tla.arch.nZ)
                    t_l0bg_o0 = tla.make_tensor_like(l0b_gate0, o_bg0, tla.arch.nZ)
                    tla.wait_flag(l0_free0)
                    tla.copy(t_l0a_o0, o_a0, scale=o_sa0)
                    tla.copy(t_l0ba_o0, o_ba0, scale=o_sba0)
                    tla.copy(t_l0bg_o0, o_bg0, scale=o_sbg0)
                    tla.set_flag(l0_ready0)
                    tla.wait_flag(l0_ready0)
                    tla.mmad_mx(t_l0c_act, t_l0a_o0, t_l0ba_o0, init_c=False)
                    tla.mmad_mx(t_l0c_gate, t_l0a_o0, t_l0bg_o0, init_c=False)
                    tla.set_flag(l0_free0)

                    o_a1 = tla.get_tile(t_l1a1_o, tla.make_coord(0, tk // 2), tla.make_shape(tm, tk // 2))
                    o_ba1 = tla.get_tile(t_l1b_act1_o, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                    o_bg1 = tla.get_tile(t_l1b_gate1_o, tla.make_coord(tk // 2, 0), tla.make_shape(tk // 2, tn))
                    o_sa1 = tla.get_tile(t_l1sa1_o, tla.make_coord(0, sk // 2), tla.make_shape(tm, sk // 2))
                    o_sba1 = tla.get_tile(t_l1sb_act1_o, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                    o_sbg1 = tla.get_tile(t_l1sb_gate1_o, tla.make_coord(sk // 2, 0), tla.make_shape(sk // 2, tn))
                    t_l0a_o1 = tla.make_tensor_like(l0a1, o_a1, tla.arch.zN)
                    t_l0ba_o1 = tla.make_tensor_like(l0b_act1, o_ba1, tla.arch.nZ)
                    t_l0bg_o1 = tla.make_tensor_like(l0b_gate1, o_bg1, tla.arch.nZ)
                    tla.wait_flag(l0_free1)
                    tla.copy(t_l0a_o1, o_a1, scale=o_sa1)
                    tla.copy(t_l0ba_o1, o_ba1, scale=o_sba1)
                    tla.copy(t_l0bg_o1, o_bg1, scale=o_sbg1)
                    tla.set_flag(l1_free1)
                    tla.set_flag(l0_ready1)
                    tla.wait_flag(l0_ready1)
                    tla.mmad_mx(t_l0c_act, t_l0a_o1, t_l0ba_o1, init_c=False)
                    tla.mmad_mx(t_l0c_gate, t_l0a_o1, t_l0bg_o1, init_c=False)
                    tla.set_flag(l0_free1)

            tla.set_flag(l0c_ready)
            tla.wait_flag(l0c_ready)
            ca_t = tla.get_tile(
                gm_ca,
                tla.make_coord(split_idx * tm + block_row * tm, block_col * tn),
                tla.make_shape(tm, n_valid),
            )
            cg_t = tla.get_tile(
                gm_cg,
                tla.make_coord(split_idx * tm + block_row * tm, block_col * tn),
                tla.make_shape(tm, n_valid),
            )
            tla.copy(
                ca_t,
                tla.get_tile(t_l0c_act, tla.make_coord(0, 0), tla.make_shape(tm, n_valid)),
            )
            tla.copy(
                cg_t,
                tla.get_tile(t_l0c_gate, tla.make_coord(0, 0), tla.make_shape(tm, n_valid)),
            )
            tla.set_flag(l0c_free)


@tla.kernel
def reduce_epi(  # pragma: no cover - device-only CATLASS DSL body
    gm_ca: tla.Tensor,
    gm_cg: tla.Tensor,
    gm_y: tla.Tensor,
    gm_sy: tla.Tensor,
    TILE_M: tla.Constexpr[int],
    SKIP_CHUNKS: tla.Constexpr[int],
    OUT_FP4: tla.Constexpr[int],
    SPLIT: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1
    tm = TILE_M
    tn = _TN
    half = tm
    out_fp4 = tla.const_expr(OUT_FP4 == 1)
    emax = 2 if out_fp4 else 8
    emax_shifted = emax * 128
    m_out = gm_y.origin_shape[0]
    n_out = gm_y.origin_shape[1] * (2 if out_fp4 else 4)
    grid_m = (m_out + tm - 1) // tm
    grid_n = (n_out + tn - 1) // tn
    total_blocks = grid_m * grid_n

    ub_act = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_gate = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_tmp_a = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_tmp_g = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    if tla.const_expr(out_fp4):
        ub_y = tla.allocate(half * tn // 2, tla.Int8, tla.AddressSpace.ub, 256)
        ub_codes = tla.allocate(half * tn, tla.Int32, tla.AddressSpace.ub, 256)
    else:
        ub_y = tla.allocate(half * tn // 4, tla.Int32, tla.AddressSpace.ub, 256)
        ub_codes = ub_y
    ub_sy = tla.allocate(half * 8, tla.Int32, tla.AddressSpace.ub, 256)

    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    mte3_done = tla.flag("mte3_done", tla.arch.MTE3, tla.arch.VECTOR)
    ub_ready = tla.flag("ub_ready", tla.arch.MTE2, tla.arch.VECTOR)
    ub_free = tla.flag("ub_free", tla.arch.VECTOR, tla.arch.MTE2)
    with tla.vector():
        # Seed the MTE3->VECTOR credit for the first tile of each core.  The
        # UB output slots are reused by every tile, so the next tile must not
        # overwrite them before the previous tile's async UB->GM copy is done.
        tla.set_flag(mte3_done)
        tla.set_flag(ub_free)
        m_out = gm_y.origin_shape[0]
        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_col = block_linear // grid_m
            block_row = block_linear % grid_m
            out_row = block_row * tm

            tla.wait_flag(mte3_done)

            gm_b_ref = tla.make_tensor(
                gm_ca.ptr,
                tla.make_layout(
                    tla.make_shape(half, tn),
                    tla.make_stride(tn, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            codes_ref = tla.make_tensor(
                gm_sy.ptr,
                tla.make_layout(
                    tla.make_shape(half, tn),
                    tla.make_stride(tn, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            sy_ref = tla.make_tensor(
                gm_sy.ptr,
                tla.make_layout(
                    tla.make_shape(half, 8),
                    tla.make_stride(8, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            ub_act_half = tla.make_tensor_like(ub_act, gm_b_ref, tla.arch.RowMajor)
            ub_gate_half = tla.make_tensor_like(ub_gate, gm_b_ref, tla.arch.RowMajor)
            ub_sy_half = tla.make_tensor_like(ub_sy, sy_ref, tla.arch.RowMajor)
            if tla.const_expr(out_fp4):
                ub_codes_half = tla.make_tensor_like(
                    ub_codes, codes_ref, tla.arch.RowMajor
                )

            rows = m_out - out_row
            if rows > 0:
                rows = min(rows, half)
                gm_y_cols = tn // 2 if out_fp4 else tn // 4
                gm_y_t = tla.get_tile(
                    gm_y,
                    tla.make_coord(out_row, block_col * gm_y_cols),
                    tla.make_shape(half, gm_y_cols),
                )
                gm_sy_t = tla.get_tile(
                    gm_sy,
                    tla.make_coord(out_row, block_col),
                    tla.make_shape(half, 1),
                )
                ub_y_half = tla.make_tensor_like(ub_y, gm_y_t, tla.arch.RowMajor)
                # Skip the trailing 64-column chunk when this block's valid
                # output width is only 64 (e.g. n_out=64, or the last N block).
                cols_valid = n_out - block_col * tn
                cols_valid = min(cols_valid, tn)
                nchunks = (cols_valid + 63) // 64

                # Sum the SPLIT fp32 partials for this sub-block's half.  The
                # partials are written in ascending split order, so the
                # reduction order is fixed and reproducible.
                ub_tmp_a_half = tla.make_tensor_like(
                    ub_tmp_a, gm_b_ref, tla.arch.RowMajor
                )
                ub_tmp_g_half = tla.make_tensor_like(
                    ub_tmp_g, gm_b_ref, tla.arch.RowMajor
                )
                with tla.vec.func(mode="simd"):
                    vzero = tla.full(0.0, tla.Float32)
                    for zr in tla.range(c0, half, c1):
                        for zc in tla.range(c0, tn // 64, c1):
                            tla.get_tile(
                                ub_act_half,
                                tla.make_coord(zr, zc * 64),
                                tla.make_shape(1, 64),
                            ).store(vzero)
                            tla.get_tile(
                                ub_gate_half,
                                tla.make_coord(zr, zc * 64),
                                tla.make_shape(1, 64),
                            ).store(vzero)
                for sp in tla.range(c0, SPLIT, c1):
                    tla.wait_flag(ub_free)
                    row0 = sp * tm + out_row
                    tla.copy(
                        ub_tmp_a_half,
                        tla.get_tile(
                            gm_ca,
                            tla.make_coord(row0, block_col * tn),
                            tla.make_shape(half, tn),
                        ),
                    )
                    tla.copy(
                        ub_tmp_g_half,
                        tla.get_tile(
                            gm_cg,
                            tla.make_coord(row0, block_col * tn),
                            tla.make_shape(half, tn),
                        ),
                    )
                    tla.set_flag(ub_ready)
                    tla.wait_flag(ub_ready)
                    with tla.vec.func(mode="simd"):
                        for sr in tla.range(c0, half, c1):
                            for sc in tla.range(c0, tn // 64, c1):
                                oa = tla.get_tile(
                                    ub_act_half,
                                    tla.make_coord(sr, sc * 64),
                                    tla.make_shape(1, 64),
                                )
                                va = tla.get_tile(
                                    ub_tmp_a_half,
                                    tla.make_coord(sr, sc * 64),
                                    tla.make_shape(1, 64),
                                )
                                oa.store(tla.add(oa.load(), va.load()))
                                og = tla.get_tile(
                                    ub_gate_half,
                                    tla.make_coord(sr, sc * 64),
                                    tla.make_shape(1, 64),
                                )
                                vg = tla.get_tile(
                                    ub_tmp_g_half,
                                    tla.make_coord(sr, sc * 64),
                                    tla.make_shape(1, 64),
                                )
                                og.store(tla.add(og.load(), vg.load()))
                    tla.set_flag(ub_free)

                with tla.vec.func(mode="simd"):
                    m_src = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    m_lo = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
                    m_i32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
                    m_fp8 = tla.create_mask(
                        pattern=tla.mask.VL64, dtype=tla.Float8E4M3FN
                    )
                    m_i8 = tla.create_mask(pattern=tla.mask.VL32, dtype=tla.Int8)
                    m_i8_64 = tla.create_mask(pattern=tla.mask.VL64, dtype=tla.Int8)
                    v1 = tla.full(1, tla.Int32)
                    v7FFF = tla.full(0x7FFF, tla.Int32)
                    vMASK = tla.full(-65536, tla.Int32)
                    v7F80 = tla.full(0x7F80, tla.Int32)
                    v0F = tla.full(0x0F, tla.Int32)
                    v7FFFFFFF = tla.full(0x7FFFFFFF, tla.Int32)
                    vFF = tla.full(0xFF, tla.Int32)
                    v7FFFFF = tla.full(0x7FFFFF, tla.Int32)
                    v7FFFF = tla.full(0x7FFFF, tla.Int32)
                    v7E = tla.full(0x7E, tla.Int32)
                    v8 = tla.full(8, tla.Int32)
                    m_i32_16 = tla.create_mask(
                        pattern=tla.mask.VL16, dtype=tla.Int32
                    )

                    scale_zero = tla.full(0, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    v7F00 = tla.full(0x7F00, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    vFF_lane = tla.full(0xFF, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    v7F81_lane = tla.full(0x7F81, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    v0040_lane = tla.full(0x0040, tla.Int32).reduce(
                        tla.ReductionOp.MAX, mask=m_lo
                    )
                    for r in tla.range(c0, rows, c1):
                        scale_word = scale_zero
                        if tla.const_expr(SKIP_CHUNKS == 1):
                            for chunk in tla.range(c0, nchunks, c1):
                                col = chunk * 64
                                a_reg = tla.get_tile(
                                    ub_act_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()
                                g_reg = tla.get_tile(
                                    ub_gate_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()

                                # Official fused semantics: activation consumes
                                # the FP32 accumulator directly (no BF16
                                # rounding before SwiGLU).
                                a_bf = a_reg
                                g_bf = g_reg

                                den = tla.add(1.0, tla.exp(tla.neg(a_bf)))
                                s_val = tla.mul(tla.div(a_bf, den), g_bf)

                                s_bits = s_val.bitcast(tla.Int32)
                                s_round = tla.bitwise_and(
                                    tla.add(
                                        tla.add(s_bits, v7FFF),
                                        tla.bitwise_and(tla.shift_right(s_bits, 16), v1),
                                    ),
                                    vMASK,
                                )
                                s_bf16 = tla.shift_right(s_round, 16)
                                s_bf = s_round.bitcast(tla.Float32)

                                exp_bits = tla.bitwise_and(s_bf16, v7F80)
                                max_lo = exp_bits.reduce(tla.ReductionOp.MAX, mask=m_lo)
                                hi_only = tla.where(m_lo, tla.full(0, tla.Int32), exp_bits)
                                max_hi = hi_only.reduce(tla.ReductionOp.MAX, mask=m_src)

                                # Official DynamicMxQuant scale rules: clamp
                                # sub-minimum blocks, mark Inf/NaN blocks with
                                # the E8M0 0xFF byte, and use the special
                                # half-scale bit patterns.
                                special_lo = tla.cmp(max_lo, 0x7F80, "eq")
                                special_hi = tla.cmp(max_hi, 0x7F80, "eq")
                                zero_lo = tla.cmp(max_lo, 0, "eq")
                                zero_hi = tla.cmp(max_hi, 0, "eq")
                                invalid_lo = tla.cmp(max_lo, emax_shifted, "le")
                                invalid_hi = tla.cmp(max_hi, emax_shifted, "le")
                                shared_lo = tla.where(
                                    invalid_lo, scale_zero, tla.sub(max_lo, emax_shifted)
                                )
                                shared_hi = tla.where(
                                    invalid_hi, scale_zero, tla.sub(max_hi, emax_shifted)
                                )

                                byte_lo = tla.shift_right(shared_lo, 7)
                                byte_hi = tla.shift_right(shared_hi, 7)
                                byte_lo = tla.where(special_lo, vFF_lane, byte_lo)
                                byte_hi = tla.where(special_hi, vFF_lane, byte_hi)
                                byte_lo = tla.where(zero_lo, scale_zero, byte_lo)
                                byte_hi = tla.where(zero_hi, scale_zero, byte_hi)

                                half_lo_bits = tla.sub(v7F00, shared_lo)
                                half_hi_bits = tla.sub(v7F00, shared_hi)
                                half_lo_bits = tla.where(
                                    special_lo, v7F81_lane, half_lo_bits
                                )
                                half_hi_bits = tla.where(
                                    special_hi, v7F81_lane, half_hi_bits
                                )
                                half_lo_bits = tla.where(zero_lo, scale_zero, half_lo_bits)
                                half_hi_bits = tla.where(zero_hi, scale_zero, half_hi_bits)
                                half_lo_bits = tla.where(
                                    tla.cmp(shared_lo, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_lo_bits,
                                )
                                half_hi_bits = tla.where(
                                    tla.cmp(shared_hi, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_hi_bits,
                                )
                                half_lo = tla.full(
                                    tla.shift_left(half_lo_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )
                                half_hi = tla.full(
                                    tla.shift_left(half_hi_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )

                                scaled_lo = tla.mul(s_bf, half_lo, mask=m_src)
                                scaled_hi = tla.mul(s_bf, half_hi, mask=m_src)
                                scaled = tla.where(m_lo, scaled_lo, scaled_hi)
                                scaled = tla.where(
                                    tla.cmp(scaled, 0.0, "eq", mask=m_src),
                                    tla.full(0.0, tla.Float32),
                                    scaled,
                                )

                                if tla.const_expr(out_fp4):
                                    abs_v = tla.abs(scaled, mask=m_src)
                                    mag = tla.full(0, tla.Int32)
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.25, "gt", mask=m_src),
                                        tla.full(1, tla.Int32),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 3.5, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.25, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 2.5, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 5.0, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    code_bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(tla.shift_right(code_bits, 31), v1),
                                        3,
                                    )
                                    code = tla.bitwise_or(mag, sign)
                                    code_i32 = tla.get_tile(
                                        ub_codes_half,
                                        tla.make_coord(r, col),
                                        tla.make_shape(1, 64),
                                    )
                                    code_i32.store(code, mask=m_i32)
                                else:
                                    # Manual E4M3FN encode (the vector fp8 store packs
                                    # four bytes per 32-bit lane; encode and pack here).
                                    bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(
                                            tla.shift_right(bits, 31), v1
                                        ),
                                        7,
                                    )
                                    abs_bits = tla.bitwise_and(bits, v7FFFFFFF)
                                    exp_field = tla.bitwise_and(
                                        tla.shift_right(abs_bits, 23), vFF
                                    )
                                    mant = tla.bitwise_and(abs_bits, v7FFFFF)
                                    mant3 = tla.shift_right(
                                        tla.add(
                                            tla.add(mant, v7FFFF),
                                            tla.bitwise_and(
                                                tla.shift_right(mant, 20), v1
                                            ),
                                        ),
                                        20,
                                    )
                                    code_exp = tla.sub(exp_field, 120)
                                    carry = tla.cmp(mant3, 8, "ge", mask=m_src)
                                    code_exp = tla.where(
                                        carry, tla.add(code_exp, 1), code_exp
                                    )
                                    mant3 = tla.where(
                                        carry, tla.sub(mant3, 8), mant3
                                    )
                                    normal_code = tla.bitwise_or(
                                        tla.shift_left(code_exp, 3), mant3
                                    )
                                    normal_code = tla.where(
                                        tla.cmp(code_exp, 16, "ge", mask=m_src),
                                        v7E,
                                        normal_code,
                                    )
                                    normal_code = tla.min(normal_code, v7E)
                                    sub_f = tla.mul(
                                        tla.abs(scaled, mask=m_src), 512.0
                                    )
                                    sub_i = sub_f.to(
                                        tla.Int32,
                                        tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO,
                                            sat_mode=tla.params.SatMode.NOSAT,
                                            round_mode=tla.params.RoundMode.CAST_TRUNC,
                                        ),
                                        m_src,
                                    )
                                    frac = tla.sub(
                                        sub_f,
                                        sub_i.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.NOSAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            ),
                                            m_src,
                                        ),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "gt", mask=m_src),
                                        v1,
                                        tla.full(0, tla.Int32),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "eq", mask=m_src),
                                        tla.bitwise_and(sub_i, v1),
                                        up,
                                    )
                                    sub_code = tla.add(sub_i, up)
                                    sub_code = tla.min(sub_code, 8)
                                    mag = tla.where(
                                        tla.cmp(exp_field, 121, "ge", mask=m_src),
                                        normal_code,
                                        sub_code,
                                    )
                                    code = tla.bitwise_or(sign, mag)
                                    zero32 = tla.full(0, tla.Int32)
                                    even1, odd1 = tla.deinterleave(code, zero32)
                                    p0, p2 = tla.deinterleave(even1, zero32)
                                    p1, p3 = tla.deinterleave(odd1, zero32)
                                    word = tla.bitwise_or(
                                        tla.bitwise_or(
                                            p0, tla.shift_left(p1, 8)
                                        ),
                                        tla.bitwise_or(
                                            tla.shift_left(p2, 16),
                                            tla.shift_left(p3, 24),
                                        ),
                                    )
                                    y_chunk = tla.get_tile(
                                        ub_y_half,
                                        tla.make_coord(r, col // 4),
                                        tla.make_shape(1, 16),
                                    )
                                    y_chunk.store(word, mask=m_i32_16)

                                word = tla.bitwise_or(
                                    byte_lo, tla.shift_left(byte_hi, 8)
                                )
                                scale_word = tla.bitwise_or(
                                    scale_word, tla.shift_left(word, chunk * 16)
                                )
                        else:
                            for chunk in tla.range_constexpr(0, 2):
                                col = chunk * 64
                                a_reg = tla.get_tile(
                                    ub_act_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()
                                g_reg = tla.get_tile(
                                    ub_gate_half,
                                    tla.make_coord(r, col),
                                    tla.make_shape(1, 64),
                                ).load()

                                # Official fused semantics: activation consumes
                                # the FP32 accumulator directly (no BF16
                                # rounding before SwiGLU).
                                a_bf = a_reg
                                g_bf = g_reg

                                den = tla.add(1.0, tla.exp(tla.neg(a_bf)))
                                s_val = tla.mul(tla.div(a_bf, den), g_bf)

                                s_bits = s_val.bitcast(tla.Int32)
                                s_round = tla.bitwise_and(
                                    tla.add(
                                        tla.add(s_bits, v7FFF),
                                        tla.bitwise_and(tla.shift_right(s_bits, 16), v1),
                                    ),
                                    vMASK,
                                )
                                s_bf16 = tla.shift_right(s_round, 16)
                                s_bf = s_round.bitcast(tla.Float32)

                                exp_bits = tla.bitwise_and(s_bf16, v7F80)
                                max_lo = exp_bits.reduce(tla.ReductionOp.MAX, mask=m_lo)
                                hi_only = tla.where(m_lo, tla.full(0, tla.Int32), exp_bits)
                                max_hi = hi_only.reduce(tla.ReductionOp.MAX, mask=m_src)

                                # Official DynamicMxQuant scale rules: clamp
                                # sub-minimum blocks, mark Inf/NaN blocks with
                                # the E8M0 0xFF byte, and use the special
                                # half-scale bit patterns.
                                special_lo = tla.cmp(max_lo, 0x7F80, "eq")
                                special_hi = tla.cmp(max_hi, 0x7F80, "eq")
                                zero_lo = tla.cmp(max_lo, 0, "eq")
                                zero_hi = tla.cmp(max_hi, 0, "eq")
                                invalid_lo = tla.cmp(max_lo, emax_shifted, "le")
                                invalid_hi = tla.cmp(max_hi, emax_shifted, "le")
                                shared_lo = tla.where(
                                    invalid_lo, scale_zero, tla.sub(max_lo, emax_shifted)
                                )
                                shared_hi = tla.where(
                                    invalid_hi, scale_zero, tla.sub(max_hi, emax_shifted)
                                )

                                byte_lo = tla.shift_right(shared_lo, 7)
                                byte_hi = tla.shift_right(shared_hi, 7)
                                byte_lo = tla.where(special_lo, vFF_lane, byte_lo)
                                byte_hi = tla.where(special_hi, vFF_lane, byte_hi)
                                byte_lo = tla.where(zero_lo, scale_zero, byte_lo)
                                byte_hi = tla.where(zero_hi, scale_zero, byte_hi)

                                half_lo_bits = tla.sub(v7F00, shared_lo)
                                half_hi_bits = tla.sub(v7F00, shared_hi)
                                half_lo_bits = tla.where(
                                    special_lo, v7F81_lane, half_lo_bits
                                )
                                half_hi_bits = tla.where(
                                    special_hi, v7F81_lane, half_hi_bits
                                )
                                half_lo_bits = tla.where(zero_lo, scale_zero, half_lo_bits)
                                half_hi_bits = tla.where(zero_hi, scale_zero, half_hi_bits)
                                half_lo_bits = tla.where(
                                    tla.cmp(shared_lo, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_lo_bits,
                                )
                                half_hi_bits = tla.where(
                                    tla.cmp(shared_hi, 0x7F00, "eq"),
                                    v0040_lane,
                                    half_hi_bits,
                                )
                                half_lo = tla.full(
                                    tla.shift_left(half_lo_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )
                                half_hi = tla.full(
                                    tla.shift_left(half_hi_bits, 16).bitcast(tla.Float32),
                                    tla.Float32,
                                )

                                scaled_lo = tla.mul(s_bf, half_lo, mask=m_src)
                                scaled_hi = tla.mul(s_bf, half_hi, mask=m_src)
                                scaled = tla.where(m_lo, scaled_lo, scaled_hi)
                                scaled = tla.where(
                                    tla.cmp(scaled, 0.0, "eq", mask=m_src),
                                    tla.full(0.0, tla.Float32),
                                    scaled,
                                )

                                if tla.const_expr(out_fp4):
                                    abs_v = tla.abs(scaled, mask=m_src)
                                    mag = tla.full(0, tla.Int32)
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.25, "gt", mask=m_src),
                                        tla.full(1, tla.Int32),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 0.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.75, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 3.5, "ge", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 1.25, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 2.5, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    mag = tla.where(
                                        tla.cmp(abs_v, 5.0, "gt", mask=m_src),
                                        tla.add(mag, 1),
                                        mag,
                                    )
                                    code_bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(tla.shift_right(code_bits, 31), v1),
                                        3,
                                    )
                                    code = tla.bitwise_or(mag, sign)
                                    code_i32 = tla.get_tile(
                                        ub_codes_half,
                                        tla.make_coord(r, col),
                                        tla.make_shape(1, 64),
                                    )
                                    code_i32.store(code, mask=m_i32)
                                else:
                                    # Manual E4M3FN encode (the vector fp8 store packs
                                    # four bytes per 32-bit lane; encode and pack here).
                                    bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(
                                        tla.bitwise_and(
                                            tla.shift_right(bits, 31), v1
                                        ),
                                        7,
                                    )
                                    abs_bits = tla.bitwise_and(bits, v7FFFFFFF)
                                    exp_field = tla.bitwise_and(
                                        tla.shift_right(abs_bits, 23), vFF
                                    )
                                    mant = tla.bitwise_and(abs_bits, v7FFFFF)
                                    mant3 = tla.shift_right(
                                        tla.add(
                                            tla.add(mant, v7FFFF),
                                            tla.bitwise_and(
                                                tla.shift_right(mant, 20), v1
                                            ),
                                        ),
                                        20,
                                    )
                                    code_exp = tla.sub(exp_field, 120)
                                    carry = tla.cmp(mant3, 8, "ge", mask=m_src)
                                    code_exp = tla.where(
                                        carry, tla.add(code_exp, 1), code_exp
                                    )
                                    mant3 = tla.where(
                                        carry, tla.sub(mant3, 8), mant3
                                    )
                                    normal_code = tla.bitwise_or(
                                        tla.shift_left(code_exp, 3), mant3
                                    )
                                    normal_code = tla.where(
                                        tla.cmp(code_exp, 16, "ge", mask=m_src),
                                        v7E,
                                        normal_code,
                                    )
                                    normal_code = tla.min(normal_code, v7E)
                                    sub_f = tla.mul(
                                        tla.abs(scaled, mask=m_src), 512.0
                                    )
                                    sub_i = sub_f.to(
                                        tla.Int32,
                                        tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO,
                                            sat_mode=tla.params.SatMode.NOSAT,
                                            round_mode=tla.params.RoundMode.CAST_TRUNC,
                                        ),
                                        m_src,
                                    )
                                    frac = tla.sub(
                                        sub_f,
                                        sub_i.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.NOSAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            ),
                                            m_src,
                                        ),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "gt", mask=m_src),
                                        v1,
                                        tla.full(0, tla.Int32),
                                    )
                                    up = tla.where(
                                        tla.cmp(frac, 0.5, "eq", mask=m_src),
                                        tla.bitwise_and(sub_i, v1),
                                        up,
                                    )
                                    sub_code = tla.add(sub_i, up)
                                    sub_code = tla.min(sub_code, 8)
                                    mag = tla.where(
                                        tla.cmp(exp_field, 121, "ge", mask=m_src),
                                        normal_code,
                                        sub_code,
                                    )
                                    code = tla.bitwise_or(sign, mag)
                                    zero32 = tla.full(0, tla.Int32)
                                    even1, odd1 = tla.deinterleave(code, zero32)
                                    p0, p2 = tla.deinterleave(even1, zero32)
                                    p1, p3 = tla.deinterleave(odd1, zero32)
                                    word = tla.bitwise_or(
                                        tla.bitwise_or(
                                            p0, tla.shift_left(p1, 8)
                                        ),
                                        tla.bitwise_or(
                                            tla.shift_left(p2, 16),
                                            tla.shift_left(p3, 24),
                                        ),
                                    )
                                    y_chunk = tla.get_tile(
                                        ub_y_half,
                                        tla.make_coord(r, col // 4),
                                        tla.make_shape(1, 16),
                                    )
                                    y_chunk.store(word, mask=m_i32_16)

                                word = tla.bitwise_or(
                                    byte_lo, tla.shift_left(byte_hi, 8)
                                )
                                scale_word = tla.bitwise_or(
                                    scale_word, tla.shift_left(word, chunk * 16)
                                )
                        sy_cell = tla.get_tile(
                            ub_sy_half,
                            tla.make_coord(r, 0),
                            tla.make_shape(1, 1),
                        )
                        sy_cell.store(scale_word)

                if tla.const_expr(out_fp4):
                    tla.pipe_barrier(tla.pipes.ALL)
                    valid_bytes = gm_y_t.origin_shape[1]
                    flat_codes = tla.make_tensor(
                        ub_codes,
                        tla.make_layout(
                            tla.make_shape(half * tn), tla.make_stride(1)
                        ),
                    )
                    flat_y = tla.make_tensor(
                        ub_y,
                        tla.make_layout(
                            tla.make_shape(half * tn // 2), tla.make_stride(1)
                        ),
                    )
                    with tla.vec.func(mode="simt", thread_block_dim=256):
                        tid, _, _ = tla.arch.thread_idx()
                        nthreads, _, _ = tla.arch.thread_block_dim()
                        for p in tla.range(tid, half * valid_bytes, nthreads):
                            row = p // valid_bytes
                            col = p - row * valid_bytes
                            c0v = flat_codes[row * tn + 2 * col]
                            c1v = flat_codes[row * tn + 2 * col + 1]
                            flat_y[p] = ((c0v & 0x0F) | ((c1v & 0x0F) << 4)).to(
                                tla.Int8
                            )
                    tla.pipe_barrier(tla.pipes.ALL)

                tla.set_flag(vec_done)
                tla.wait_flag(vec_done)
                tla.copy(gm_y_t, ub_y_half)
                tla.copy(gm_sy_t, ub_sy_half)
            tla.set_flag(mte3_done)


# ---------------------------------------------------------------------------
# Host entry
# ---------------------------------------------------------------------------


_ARTIFACTS: dict[tuple[int, int, int, int, int], object] = {}
_SPLITK_ARTIFACTS: dict[tuple[int, int, int, int, int], tuple] = {}


def _splitk_plan(m, k, n_out, tile_k):
    """Split-K factor for the two-kernel path, or 0 to stay fused.

    The fused kernel is already the better choice for most shapes: split-K
    only pays when the per-block K chain is the bottleneck, i.e. very small M,
    long K, and few N tiles.  Keep the gate conservative and make SPLIT a
    divisor of the K tile count so every split has the same work.
    """
    if not _ENABLE_SPLITK:
        return 0
    if m > 32 or k < 2048 or n_out <= 0 or n_out > 1024:
        return 0
    k_tiles = (k + tile_k - 1) // tile_k
    grid_n = (n_out + 127) // 128
    cap = min(8, max(1, 32 // grid_n))
    best = 0
    for split in range(1, cap + 1):
        if k_tiles % split == 0:
            best = split
    return best if best >= 2 else 0


def _splitk_artifacts(in_fp4, out_fp4, tile_m, splitk, skip_chunks, tensors):
    key = (1 if in_fp4 else 0, 1 if out_fp4 else 0, int(tile_m), int(splitk),
           int(skip_chunks))
    arts = _SPLITK_ARTIFACTS.get(key)
    if arts is None:
        a_t, sa_t, b_t, sb_t, ca_t, cg_t, y_t, sy_t = tensors
        k1 = tla.compile(
            mm_partial,
            a_t, sa_t, b_t, sb_t, ca_t, cg_t,
            int(tile_m), 256, int(splitk),
            options="--npu-arch 3510",
        )
        k2 = tla.compile(
            reduce_epi,
            ca_t, cg_t, y_t, sy_t,
            int(tile_m), int(skip_chunks), key[1], int(splitk),
            options="--npu-arch 3510",
        )
        arts = (k1, k2)
        _SPLITK_ARTIFACTS[key] = arts
    return arts



def _bind(value, tag, origin_shape, element_type):
    return from_dlpack(
        value,
        layout_tag=tag,
        origin_shape=origin_shape,
        element_type=element_type,
    ).mark_layout_dynamic()


def _compiled_artifact(
    in_fp4: int, out_fp4: int, tile_m: int, tile_k: int, skip_chunks: int,
    tensors,
):
    key = (1 if in_fp4 else 0, 1 if out_fp4 else 0, int(tile_m), int(tile_k),
           int(skip_chunks))
    artifact = _ARTIFACTS.get(key)
    if artifact is None:
        artifact = tla.compile(
            fused_kernel,
            *tensors,
            int(tile_m),
            int(tile_k),
            int(skip_chunks),
            key[1],
            options="--npu-arch 3510",
        )
        _ARTIFACTS[key] = artifact
    return artifact


def _validate_inputs(a, scale_a, b, scale_b, in_fp4, out_fp4):
    """Validate the public contract before binding tensors or launching."""
    if in_fp4 not in (0, 1) or out_fp4 not in (0, 1):
        raise ValueError(
            f"in_fp4 and out_fp4 must be 0 or 1, got {in_fp4} and {out_fp4}"
        )
    for name, value in (("a", a), ("scale_a", scale_a), ("b", b), ("scale_b", scale_b)):
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"{name} must be a torch.Tensor, got {type(value).__name__}"
            )
        if value.dtype != torch.uint8:
            raise ValueError(f"{name} must be uint8, got {value.dtype}")
        if value.device != a.device:
            raise ValueError(f"{name} must be on {a.device}, got {value.device}")
        if not value.is_contiguous():
            # ``from_dlpack(origin_shape=...)`` rebuilds dense strides, so a
            # strided view would be read as if it were contiguous.
            raise ValueError(f"{name} must be contiguous")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be rank-2 byte tensors")
    if scale_a.dim() != 3 or scale_b.dim() != 3:
        raise ValueError("scale_a and scale_b must be rank-3 byte tensors")

    m = a.shape[0]
    if in_fp4:
        if a.shape[1] % 2 != 0:
            raise ValueError("fp4 a must pack two E2M1 values per byte along K")
        k = a.shape[1] * 2
        n = b.shape[0]
        expected_b = (n, a.shape[1])
    else:
        k = a.shape[1]
        n = b.shape[1]
        expected_b = (k, n)
    if tuple(b.shape) != expected_b:
        mode = "fp4" if in_fp4 else "fp8"
        raise ValueError(f"b must be {expected_b} for {mode} input, got {tuple(b.shape)}")
    if k <= 0 or k % 64 != 0:
        raise ValueError(f"K must be a positive multiple of 64, got {k}")
    if n <= 0 or n % 128 != 0:
        raise ValueError(f"N must be a positive multiple of 128, got {n}")

    groups = k // 64
    expected_sa = (m, groups, 2)
    expected_sb = (groups, n, 2)
    if tuple(scale_a.shape) != expected_sa:
        raise ValueError(f"scale_a must be {expected_sa}, got {tuple(scale_a.shape)}")
    if tuple(scale_b.shape) != expected_sb:
        raise ValueError(f"scale_b must be {expected_sb}, got {tuple(scale_b.shape)}")
    return m, k, n


def run(a, scale_a, b, scale_b, in_fp4, out_fp4):
    in_fp4 = int(in_fp4)
    out_fp4 = int(out_fp4)
    m, k, n = _validate_inputs(a, scale_a, b, scale_b, in_fp4, out_fp4)

    n_out = n // 2
    device = a.device
    # Small-M shapes use a smaller M tile so the epilogue and cube work scale
    # with the real row count and more blocks can run in parallel.
    # TILE_M trades B re-read traffic (larger is better) against epilogue
    # parallelism/per-block overhead (smaller is better).  Model-scale work is
    # bandwidth-bound and wants 256; small work wants 64; the middle uses 128.
    work = m * k * n
    if work >= 100_000_000_000:
        tile_m = 256
    elif m <= 16:
        tile_m = 16
    elif m <= 32:
        # Tiny M: a 32-row tile still lands on a 16-row fractal boundary while
        # halving the cube/fixpipe work spent on pad rows.
        tile_m = 32
    elif work <= 2_000_000_000 or m <= 512:
        tile_m = 64
    else:
        tile_m = 128
    # The kernel copies only the valid M/N/K extent of each tile.  The E8M0
    # scale is the one exception: its K groups must stay finite, because a
    # stale/NaN exponent times the zero-filled K tail is still NaN
    # (0 * NaN = NaN).  127 (1.0) is the neutral exponent.  One F.pad per
    # tensor is a single device kernel, cheaper on small shapes than
    # empty + two slice copies.
    tile_k = 128 if k <= 128 else 256
    k_groups = (k + tile_k - 1) // tile_k * (tile_k // 32)
    # Only irregular N widths can have a 64-column tail chunk to skip; keep
    # the fully unrolled two-chunk epilogue for N % 128 == 0.
    skip_chunks = 1 if (n_out % 128) != 0 else 0
    # Split-K is only profitable for tiny M with a long K chain; it runs a
    # partial-matmul kernel plus a reduce+epilogue kernel.
    splitk = _splitk_plan(m, k, n_out, tile_k)
    if splitk:
        tile_m = 16 if m <= 16 else 32
    if k_groups != k // 32:
        dg = (k_groups - k // 32) // 2
        sa_pad = F.pad(scale_a, (0, 0, 0, dg, 0, 0), value=127)
        sb_pad = F.pad(scale_b, (0, 0, 0, 0, 0, dg), value=127)
    else:
        sa_pad, sb_pad = scale_a, scale_b
    a_pad, b_pad = a, b

    if in_fp4:
        a_t = _bind(a_pad, tla.arch.RowMajor, (m, k), tla.Float4E2M1)
        b_t = _bind(b_pad, tla.arch.ColumnMajor, (k, n), tla.Float4E2M1)
    else:
        a_t = _bind(a_pad, tla.arch.RowMajor, (m, k), tla.Float8E4M3FN)
        b_t = _bind(b_pad, tla.arch.RowMajor, (k, n), tla.Float8E4M3FN)
    y = torch.empty(
        (m, n_out // 2 if out_fp4 else n_out), dtype=torch.uint8, device=device
    )
    groups = n_out // 32
    padded_groups = ((groups + 3) // 4) * 4
    sy_pad = torch.empty((m, padded_groups), dtype=torch.uint8, device=device)

    sa_t = _bind(
        sa_pad, tla.arch.RowMajorMxScaleA, (m, k_groups), tla.Float8E8M0
    )
    sb_t = _bind(
        sb_pad, tla.arch.RowMajorMxScaleB, (k_groups, n), tla.Float8E8M0
    )
    if out_fp4:
        y_t = _bind(
            y, tla.arch.RowMajor, (m, n_out // 2), tla.Int8
        )
    else:
        y_i32 = y.view(torch.int32).view(m, n_out // 4)
        y_t = _bind(
            y_i32, tla.arch.RowMajor, (m, n_out // 4), tla.Int32
        )
    sy_i32 = sy_pad.view(torch.int32).view(m, padded_groups // 4)
    sy_t = _bind(sy_i32, tla.arch.RowMajor, (m, padded_groups // 4), tla.Int32)

    tensors = (a_t, sa_t, b_t, sb_t, y_t, sy_t)
    device_index = 0 if a.device.index is None else int(a.device.index)
    torch.npu.set_device(device_index)
    props = torch.npu.get_device_properties(device_index)
    grid_m = (m + tile_m - 1) // tile_m
    grid_n = (n_out + 127) // 128
    if splitk:
        # Two-kernel split-K: partial fp32 C per split, then a deterministic
        # reduce + SwiGLU + MX-quant epilogue.
        ca = torch.empty(
            (splitk * tile_m, n_out), dtype=torch.float32, device=device
        )
        cg = torch.empty(
            (splitk * tile_m, n_out), dtype=torch.float32, device=device
        )
        ca_t = _bind(ca, tla.arch.RowMajor, (splitk * tile_m, n_out), tla.Float32)
        cg_t = _bind(cg, tla.arch.RowMajor, (splitk * tile_m, n_out), tla.Float32)
        k1, k2 = _splitk_artifacts(
            in_fp4, out_fp4, tile_m, splitk, skip_chunks,
            (a_t, sa_t, b_t, sb_t, ca_t, cg_t, y_t, sy_t),
        )
        k1(
            a_t, sa_t, b_t, sb_t, ca_t, cg_t,
            block_num=max(1, min(int(props.cube_core_num), grid_m * grid_n * splitk)),
        )
        k2(
            ca_t, cg_t, y_t, sy_t,
            block_num=max(1, min(int(props.vector_core_num), grid_m * grid_n)),
        )
    else:
        artifact = _compiled_artifact(
            in_fp4, out_fp4, tile_m, tile_k, skip_chunks, tensors
        )
        # Do not launch idle cores for small shapes: one block needs exactly
        # one core, and the kernel's block-stride loops adapt to block_num.
        block_num = max(1, min(int(props.cube_core_num), grid_m * grid_n))
        artifact(*tensors, block_num=block_num)
    if padded_groups == groups:
        sy = sy_pad.view(m, groups // 2, 2)
    else:
        sy = sy_pad[:, :groups].contiguous().view(m, groups // 2, 2)
    return y, sy
