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

#ifndef HPP_HPP
#define HPP_HPP

#include <cstdint>

#include <kernel_operator.h>

#include "../attn_infra/detail/alignment.hpp"
#include "../attn_infra/detail/dependent_false.hpp"
#include "../attn_infra/detail/macros.hpp"
#ifndef DUMP_TENSOR
#define DEBUG_PRINT(str) do {  AscendC::printf("[DEBUG] Line: %d %s\n", __LINE__, str); } while(0)
#define DUMP_TENSOR(mmResUb, m, n ) \
    do { AscendC::printf("line %d, tensor: %s\n", __LINE__, #mmResUb); \
        uint32_t array_ ## __LINE__[] = {static_cast<uint32_t>(m), static_cast<uint32_t>(n)}; \
        AscendC::DumpTensor(mmResUb, 2, (m)*(n)); \
    } while(0)
#endif

namespace NpuArch {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

constexpr uint32_t BYTE_PER_BLK_FP = 128;  /// datablock size of A1->C2PiPE2GM

constexpr uint32_t MX_SCALE_COPY_GROUP_NUM = 2;
constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
constexpr uint32_t MX_BASEK_FACTOR = 64;

class EmptyClass {};

} // namespace NpuArch

#endif // HPP_HPP