/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_COMMON_HPP
#define EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_COMMON_HPP

#include "attn_infra/base_defs.hpp"
#include "attn_infra/arch/arch.hpp"
#include "attn_infra/layout/layout.hpp"

#include "attn_infra/gemm/block/block_mmad.hpp"
#include "attn_infra/gemm/dispatch_policy.hpp"
#include "attn_infra/gemm/gemm_type.hpp"

#include "attn_infra/arch/cross_core_sync.hpp"
#include "attn_infra/arch/resource.hpp"
#include "attn_infra/epilogue/block/block_epilogue.hpp"
#include "attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
using namespace matmul;

namespace RfaKenelCommon {
    constexpr uint32_t QK_READY_ID = 1;
    constexpr uint32_t SOFTMAX_READY_ID = 2;
    constexpr uint32_t PV_READY_ID = 3;
    constexpr uint32_t MASKTOIDX_READY_ID = 4;

    constexpr uint32_t BLOCK_SIZE = 16;
    constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
    constexpr uint32_t TMP_SIZE_DECODER = 32768;

    constexpr int32_t TILING_BATCH = 0;
    constexpr int32_t TILING_NUMHEADS = 1;
    constexpr int32_t TILING_HEADDIM = 2;
    constexpr int32_t TILING_NUMBLOKS = 3;
    constexpr int32_t TILING_BLOCKSIZE = 4;
    constexpr int32_t TILING_MAXBLOCKS = 5;
    constexpr int32_t TILING_TOR = 6;
    constexpr int32_t TILING_KVHEADS = 7;
    constexpr int32_t TILING_HEADSIZE = 8;
    constexpr int32_t TILING_PARASIZE = 9;
    constexpr int32_t TILING_HEAD_SPLIT_SIZE = 10;
    constexpr int32_t TILING_HEAD_SPLIT_NUM = 11;
    constexpr int32_t TILING_HEADDIM_ROPE = 13;
    constexpr int32_t TILING_MAX_KVSEQLEN = 14;
    constexpr int32_t TILING_KVSPLIT = 15;
    constexpr int32_t TILING_KVCORENUM = 16;
    constexpr int32_t TILING_TOTAL_QTOKENS = 18;
    constexpr int32_t TILING_FORMERTASKNUM = 19;
    constexpr int32_t TILING_TAILTASKNUM = 20;
    constexpr int32_t TILING_BLOCKSIZE_CALC = 25;
    constexpr int32_t TILING_HEADDIM_K_SPLIT = 38;
    constexpr int32_t TILING_HEADDIM_V_SPLIT = 39;
    constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
    constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;

    constexpr int32_t NUM1 = 1;
    constexpr int32_t NUM4 = 4;

    constexpr int32_t NUM64 = 64;
    constexpr int32_t NUM512 = 512;
    constexpr int32_t NUM576 = 576;
    constexpr uint32_t BASIC_BLOCK_SIZE = 128;
    constexpr int32_t Q_BLK = 128;
    constexpr int32_t MAX_STACK_LEN = 512;
    constexpr int32_t PRE_LAUNCH = 2;

    constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

    constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;

    constexpr uint32_t Q_TILE_CEIL = 128;
    constexpr uint32_t MAX_KV_STACK_LEN = 512; //可配置1024或512

    template <typename T>
    __aicore__ inline T AlignUp(T a, T b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b * b;
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return (a > b) ? b : a;
    }

    enum class cvPipeLineType {
        FAI_COMMON_NORMAL = 0,
        FAI_COMMON_CHUNK_MASK = 1,
        FAI_SPARSE_BLOCK = 2
    };

    enum class SparseMaskType {
        NO_MASK = 0,
        MASK_SPEC = 1,
        MASK_CAUSUAL = 2,
        SPARSE_BLOCK = 3
    };

    __aicore__ inline
    uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qNBlockTile = 1;
        return qNBlockTile;
    }

    __aicore__ inline
    uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_BLK;
        return qSBlockTile;
    }

    __aicore__ inline
    uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
    {
        uint32_t qBlocksInX = (x + BASIC_BLOCK_SIZE - 1) / BASIC_BLOCK_SIZE;
        uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
        uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
        uint32_t remainingBlocks = (remainingSeqlen + BASIC_BLOCK_SIZE - 1) / BASIC_BLOCK_SIZE;
        return qBlocksInX * completeXBlocks + remainingBlocks;
    }


    // EagleQuantBlockSparseAttention Kernel Parameters
    struct EagleQuantBlockSparseAttentionKernelParams {
        // 输入张量
        GM_ADDR q;                       // Query张量
        GM_ADDR k;                       // Key张量
        GM_ADDR v;                       // Value张量
        GM_ADDR blockSparseMask;         // 适配新增的张量
        GM_ADDR mask;                    // 掩码张量
        GM_ADDR blockTables;             // 块表
        GM_ADDR actualQseqlen;           // 实际Q序列长度
        GM_ADDR actualKvseqlen;          // 实际KV序列长度
        
        
        // 输出和工作空间
        GM_ADDR o;                       // 输出张量
        GM_ADDR lse;
        GM_ADDR workspace;
        GM_ADDR tiling;                  // Tiling数据

        // 默认构造函数
        __aicore__ inline
        EagleQuantBlockSparseAttentionKernelParams() {}

        // 带参数的构造函数(新增了参数的构造器)
        __aicore__ inline
        EagleQuantBlockSparseAttentionKernelParams(
            GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR blockSparseMask_, GM_ADDR mask_, GM_ADDR blockTables_,
            GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
            GM_ADDR o_, GM_ADDR lse_, GM_ADDR workspace_,
            GM_ADDR tiling_)
            : q(q_), k(k_), v(v_), blockSparseMask(blockSparseMask_), mask(mask_), blockTables(blockTables_),
            actualQseqlen(actualQseqlen_), actualKvseqlen(actualKvseqlen_),
            o(o_),
            lse(lse_),
            workspace(workspace_),
            tiling(tiling_)
        {}
    };

    // 为了兼容性，保留旧名称的别名（逐步废弃）
    using FASparseKernelParams = EagleQuantBlockSparseAttentionKernelParams;
}
#endif // EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_COMMON_HPP

