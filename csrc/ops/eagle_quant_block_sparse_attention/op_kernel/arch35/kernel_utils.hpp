/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BSA_ARCH35_KERNEL_UTILS
#define BSA_ARCH35_KERNEL_UTILS

#include "../attn_infra/base_defs.hpp"
#include "../attn_infra/arch/arch.hpp"
#include "../attn_infra/layout/layout.hpp"

#include "../attn_infra/gemm/block/block_mmad.hpp"
#include "../attn_infra/gemm/dispatch_policy.hpp"
#include "../attn_infra/gemm/gemm_type.hpp"

#include "../attn_infra/arch/cross_core_sync.hpp"
#include "../attn_infra/arch/resource.hpp"
#include "../attn_infra/epilogue/block/block_epilogue.hpp"
#include "../attn_infra/epilogue/dispatch_policy.hpp"
#include "../tla/tensor.hpp"
#include "../tla/layout.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BsaKernelArch35 {

enum class Format {
    TND = 0,
    BNSD = 1
};

struct BsaKernelParamsArch35 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR query_scale;
    GM_ADDR key_scale;
    GM_ADDR value_scale;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR blockSparseMask;
    GM_ADDR o;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline
    BsaKernelParamsArch35() {}
    __aicore__ inline
    BsaKernelParamsArch35(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
        GM_ADDR query_scale_, GM_ADDR key_scale_, GM_ADDR value_scale_,
        GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR blockSparseMask_, GM_ADDR o_,
        GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
        query_scale(query_scale_), key_scale(key_scale_), value_scale(value_scale_),
        actualKvseqlen(actualKvseqlen_), blockSparseMask(blockSparseMask_), o(o_),
        workSpace(workSpace_), tiling(tiling_) {}
};

constexpr uint32_t SPARSE_PATTERN_MODE_TABLE = 1;


template<class ArchTag>
__aicore__ inline void SparseTable2Count(
    NpuArch::Arch::Resource<ArchTag> &resource,
    AscendC::GlobalTensor<int32_t> sparseTableGM,
    AscendC::GlobalTensor<int32_t> sparseCountGM,
    uint32_t totalRowNumBlockMask,
    uint32_t yBlockNumAligned,
    uint32_t avgRowPerSubCore,
    uint32_t preActiveSubCoreNum)
{
    static constexpr uint32_t PRE_ROW_TILE = 128;
    static constexpr uint32_t PRE_COL_TILE = 128;
    static constexpr uint32_t PRE_ELEM_NUM_PER_LOOP = PRE_ROW_TILE * PRE_COL_TILE;
    static constexpr uint32_t TABLE_IN_INT32 = 0;
    static constexpr uint32_t TABLE_VALID_BIT = TABLE_IN_INT32 + PRE_ELEM_NUM_PER_LOOP * sizeof(int32_t);
    static constexpr uint32_t TABLE_ONE_VALUE = TABLE_VALID_BIT + PRE_COL_TILE * sizeof(uint8_t);
    static constexpr uint32_t TABLE_VALID_VALUE = TABLE_ONE_VALUE + PRE_COL_TILE * sizeof(float);
    static constexpr uint32_t TABLE_TILE_COUNT = TABLE_VALID_VALUE + PRE_ELEM_NUM_PER_LOOP * sizeof(float);
    static constexpr uint32_t TABLE_COUNT_FLOAT = TABLE_TILE_COUNT + PRE_ROW_TILE * sizeof(float);
    static constexpr uint32_t RSVD_SPARSE_COUNT = TABLE_COUNT_FLOAT + PRE_ROW_TILE * sizeof(float);
    // Pass Sum an explicit temp buffer so it will not pop hidden UB space outside this layout.
    static constexpr uint32_t TABLE_SUM_TMP = RSVD_SPARSE_COUNT + PRE_ROW_TILE * sizeof(int32_t);
    static constexpr uint32_t TABLE_SUM_TMP_SIZE = PRE_ROW_TILE * PRE_COL_TILE * sizeof(float);
    static constexpr uint32_t SPARSE_TABLE_UB_SIZE = TABLE_SUM_TMP + TABLE_SUM_TMP_SIZE;
    static constexpr uint32_t SPARSE_TABLE_UB_LIMIT = ArchTag::UB_SIZE - 8U * 1024U;
    static constexpr int32_t SPARSE_TABLE_END = -1;
    static constexpr int32_t SPARSE_TABLE_VALID_MIN = 0;
    static_assert(SPARSE_TABLE_UB_SIZE <= SPARSE_TABLE_UB_LIMIT,
        "SparseTable2Count UB layout exceeds the usable arch UB size.");

    AscendC::LocalTensor<int32_t> sparseTableUb =
        resource.ubBuf.template GetBufferByByte<int32_t>(TABLE_IN_INT32);//128*128*4Bytes
    AscendC::LocalTensor<uint8_t> validMaskUb =
        resource.ubBuf.template GetBufferByByte<uint8_t>(TABLE_VALID_BIT);//128Bytes
    AscendC::LocalTensor<float> oneValueUb =
        resource.ubBuf.template GetBufferByByte<float>(TABLE_ONE_VALUE);//128*4Bytes
    AscendC::LocalTensor<float> validValueUb =
        resource.ubBuf.template GetBufferByByte<float>(TABLE_VALID_VALUE);//128*128*4Bytes
    AscendC::LocalTensor<float> tileCountUb =
        resource.ubBuf.template GetBufferByByte<float>(TABLE_TILE_COUNT);//128*4Bytes
    AscendC::LocalTensor<float> countFloatUb =
        resource.ubBuf.template GetBufferByByte<float>(TABLE_COUNT_FLOAT);//128*4Bytes
    AscendC::LocalTensor<int32_t> sparseCountUb =
        resource.ubBuf.template GetBufferByByte<int32_t>(RSVD_SPARSE_COUNT);//128*4Bytes
    AscendC::LocalTensor<uint8_t> sumTmpUb =
        resource.ubBuf.template GetBufferByByte<uint8_t>(TABLE_SUM_TMP);//128*128*4Bytes

    uint32_t subCoreIdx = AscendC::GetBlockIdx();
    uint64_t curSubCoreRowOffset = static_cast<uint64_t>(subCoreIdx) * avgRowPerSubCore;
    uint32_t actDealtRow = (subCoreIdx == preActiveSubCoreNum - 1) ?
        static_cast<uint32_t>(totalRowNumBlockMask - curSubCoreRowOffset) : avgRowPerSubCore;
    if (subCoreIdx >= preActiveSubCoreNum) {
        return;
    }

    AscendC::Duplicate(oneValueUb, static_cast<float>(1.0), PRE_COL_TILE);
    AscendC::PipeBarrier<PIPE_V>();
    uint32_t rowLoop = (actDealtRow + PRE_ROW_TILE - 1) / PRE_ROW_TILE;
    for (uint32_t i = 0; i < rowLoop; i++) {
        uint32_t curLoopRowOffset = i * PRE_ROW_TILE;
        uint32_t actDealtRowCurLoop =
            (i == rowLoop - 1) ? (actDealtRow - curLoopRowOffset) : PRE_ROW_TILE;

        AscendC::Duplicate(countFloatUb, static_cast<float>(0.0), actDealtRowCurLoop);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t colLoop = (yBlockNumAligned + PRE_COL_TILE - 1) / PRE_COL_TILE;
        for (uint32_t j = 0; j < colLoop; j++) {
            uint32_t curLoopColOffset = j * PRE_COL_TILE;
            uint32_t actDealtColCurLoop =
                (j == colLoop - 1) ? (yBlockNumAligned - curLoopColOffset) : PRE_COL_TILE;
            // MTE copy only needs 32B alignment; CompareScalar keeps the 256B count alignment.
            uint32_t actDealtColCurLoopCopyAlign = ((actDealtColCurLoop + 7) / 8) * 8;
            uint32_t actDealtColCurLoopCompareAlign = ((actDealtColCurLoop + 63) / 64) * 64;
            uint32_t copyRightPadding = actDealtColCurLoopCopyAlign - actDealtColCurLoop;
            uint64_t sparseTableOffset =
                (curSubCoreRowOffset + curLoopRowOffset) * yBlockNumAligned + curLoopColOffset;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            if (actDealtColCurLoopCompareAlign > actDealtColCurLoopCopyAlign) {
                // Pre-fill the compare-only tail. This avoids large MTE right-padding configs.
                AscendC::Duplicate(sparseTableUb, SPARSE_TABLE_END, actDealtRowCurLoop * PRE_COL_TILE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            }
            // Load one int32 sparse-table tile. MTE pads only to 32B; any wider compare tail is already -1.
            AscendC::DataCopyPad(
                sparseTableUb,
                sparseTableGM[sparseTableOffset],
                AscendC::DataCopyExtParams(
                    actDealtRowCurLoop, //BlockCount 指定该指令包含的连续传输数据块的个数
                    actDealtColCurLoop * sizeof(int32_t), //指定该指令每个连续传输数据块长度，单位为字节
                    (yBlockNumAligned - actDealtColCurLoop) * sizeof(int32_t), //源操作数相邻连续数据块间隔（从前一个尾到后一个头），单位为字节
                    (PRE_COL_TILE - actDealtColCurLoopCopyAlign) * sizeof(int32_t) / 32,//目的操作数相邻连续数据块间隔（从前一个尾到后一个头），单位为32B
                    0),
                //isPad表示要填充 leftPadding，左侧需要填充的元素个数，字节数不要超过32B，rightPadding，右侧要填充元素个数，不超过32B，PaddingValue是要填充的数值
                AscendC::DataCopyPadExtParams<int32_t>(
                    true, 0, copyRightPadding, SPARSE_TABLE_END));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            for (uint32_t row = 0; row < actDealtRowCurLoop; row++) {
                // Sparse table rows are terminated by -1; valid table indices are non-negative.
                AscendC::CompareScalar(
                    validMaskUb,//输出，目的操作数
                    sparseTableUb[row * PRE_COL_TILE],//源操作数0
                    SPARSE_TABLE_VALID_MIN,
                    AscendC::CMPMODE::GE,
                    actDealtColCurLoopCompareAlign);//calCount，输入数据元素个数，设置CalCount时，需要保证calCount个元素所占空间256字节对齐
                AscendC::PipeBarrier<PIPE_V>();
                //Mask数值为0,从src0选取，否则从src1中选取
                AscendC::Select(
                    validValueUb[row * PRE_COL_TILE],//输出
                    validMaskUb,//mask输入
                    oneValueUb,//源操作数0
                    static_cast<float>(0.0),//源操作数1
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, //selMode
                    actDealtColCurLoopCompareAlign); //calCount
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Sum(
                tileCountUb,
                validValueUb,
                sumTmpUb,
                AscendC::SumParams{actDealtRowCurLoop, PRE_COL_TILE, actDealtColCurLoop});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(countFloatUb, countFloatUb, tileCountUb, actDealtRowCurLoop);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        AscendC::Cast(
            sparseCountUb,
            countFloatUb,
            AscendC::RoundMode::CAST_ROUND,
            actDealtRowCurLoop);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        uint64_t sparseCountOffset = curSubCoreRowOffset + curLoopRowOffset;
        AscendC::DataCopyPad(
            sparseCountGM[sparseCountOffset],
            sparseCountUb,
            AscendC::DataCopyExtParams(1, actDealtRowCurLoop * sizeof(int32_t), 0, 0, 0));
    }

}

__aicore__ inline
uint32_t GetCurQSTileNum(int64_t curQSeqlen, uint32_t blockShapeX, uint32_t qBaseTile)
{
    uint32_t fullXBlockNum = curQSeqlen / blockShapeX;
    uint32_t tailXBlockSize = curQSeqlen % blockShapeX;
    uint32_t qSTileNumPerFullXBlock = (blockShapeX + qBaseTile - 1) / qBaseTile;
    uint32_t qSTileNumTailXBlock = (tailXBlockSize + qBaseTile - 1) / qBaseTile;
    uint32_t curQSTileNum = qSTileNumPerFullXBlock * fullXBlockNum + qSTileNumTailXBlock;
    return curQSTileNum;
}

}

#endif
