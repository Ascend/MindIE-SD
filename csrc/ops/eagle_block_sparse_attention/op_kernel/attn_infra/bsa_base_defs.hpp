/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file base_defs.hpp
 * \brief
 */

#ifndef BSA_BASE_DEFS_HPP
#define BSA_BASE_DEFS_HPP

#include <cstdint>

#include <kernel_operator.h>

#include "../attn_infra/detail/bsa_alignment.hpp"
#include "../attn_infra/detail/bsa_dependent_false.hpp"
#include "../attn_infra/detail/bsa_macros.hpp"

// [mx-scale] 启用 catlass 的 isMxScale* trait 块（tla/layout_bsa.hpp:325 的 #if CATLASS_ARCH_A5_ENABLED guard）。
// 仅 A5(310) 构建定义——fp8_e8m0_t 等 mx 类型在 A5 才有；A2(220) 不定义，避免编译 A2 无关的 fp8 类型。
#if (__CCE_AICORE__ == 310) && !defined(CATLASS_ARCH_A5_ENABLED)
#define CATLASS_ARCH_A5_ENABLED
#endif

namespace NpuArch {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

// [fp4 bit-granular] 元素的位宽。对齐 catlass 官方 copy_gm_to_l1.hpp 的 SizeOfBits 语义：
//   fp4x2_e2m1_t 是 2×4-bit 打包类型，但其语义元素是 fp4(4-bit)——GlobalTensor/LocalTensor
//   <fp4x2_e2m1_t>::operator[] 与 Nd2Nz 均按 fp4 元素(4-bit)寻址。故 fractal 的
//   ELE_NUM_PER_C0/ELE_NUM_PER_FRACTAL 须用 bit 粒度（fp4x2: 64/1024）而非 sizeof 字节
//   粒度（32/512），否则 nZ/zN fractal 的 crd2offset 会 2× 偏、Nd2Nz dstNz*Stride 派发也错位。
//   其余类型 8*sizeof(T)，与原 sizeof-based 算法等价（half/fp8/fp32/int8 零影响）。
template <class T>
struct SizeOfBits {
    static constexpr uint32_t value = sizeof(T) * 8;
};
template <>
struct SizeOfBits<fp4x2_e2m1_t> {
    static constexpr uint32_t value = 4;
};
constexpr uint32_t BytesToBits(uint32_t bytes) { return bytes * 8; }

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

constexpr uint32_t BYTE_PER_BLK_FP = 128; /// datablock size of A1->C2PiPE2GM

constexpr uint32_t MX_SCALE_COPY_GROUP_NUM = 2;
constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
constexpr uint32_t MX_BASEK_FACTOR = 64;

class EmptyClass {};

} // namespace NpuArch

#endif // HPP_HPP
