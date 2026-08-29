/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_MMAD_ARCH35_OPT_HPP
#define GEMM_BLOCK_MMAD_ARCH35_OPT_HPP

#include "../../../attn_infra/base_defs.hpp"

namespace NpuArch::Gemm::Block {

struct Arch35MmadOpt {
    static constexpr int DYNAMIC_LOOP = -1;

    __aicore__ inline static bool UseM1N12(uint32_t mLoopNum, uint32_t nLoopNum)
    {
        return (mLoopNum == 1U) && (nLoopNum > 0U) && (nLoopNum <= 2U);
    }

    __aicore__ inline static uint32_t LoopNum(uint32_t total, uint32_t tile)
    {
        return CeilDiv(total, tile);
    }

    template <int STATIC_LOOP_NUM>
    __aicore__ inline static uint32_t LoopBound(uint32_t loopNum)
    {
        static_assert(STATIC_LOOP_NUM == DYNAMIC_LOOP || STATIC_LOOP_NUM > 0,
            "STATIC_LOOP_NUM must be -1 or positive");
        if constexpr (STATIC_LOOP_NUM == DYNAMIC_LOOP) {
            return loopNum;
        } else {
            return static_cast<uint32_t>(STATIC_LOOP_NUM);
        }
    }

    __aicore__ inline static uint32_t GetCurLoopCounter(uint32_t outerLoopItr, uint32_t loopNum)
    {
        return outerLoopItr * loopNum;
    }

    __aicore__ inline static uint32_t MainLoopNum(uint32_t total, uint32_t tile)
    {
        uint32_t loopNum = LoopNum(total, tile);
        return (loopNum > 0U) ? (loopNum - 1U) : 0U;
    }

    __aicore__ inline static uint32_t FinalTileSize(uint32_t total, uint32_t tile, uint32_t mainLoopNum)
    {
        return total - mainLoopNum * tile;
    }

    template <uint32_t STAGES>
    __aicore__ inline static uint32_t StageId(uint32_t counter)
    {
        static_assert(STAGES > 0U, "STAGES must not be 0");
        if constexpr ((STAGES & (STAGES - 1U)) == 0U) {
            return counter & (STAGES - 1U);
        } else {
            return counter % STAGES;
        }
    }

    template <uint32_t ALIGN>
    __aicore__ inline static uint32_t AlignUpPow2(uint32_t value)
    {
        static_assert(ALIGN > 0U && ((ALIGN & (ALIGN - 1U)) == 0U), "ALIGN must be power of 2");
        return (value + ALIGN - 1U) & ~(ALIGN - 1U);
    }
};

}  // namespace NpuArch::Gemm::Block

#endif
