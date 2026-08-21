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
 * \file eagle_quant_block_sparse_attention_kernel.h
 * \brief Block Sparse Attention Kernel Implementation
 */

#ifndef EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H
#define EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H

#include "kernel_common.hpp"

using namespace NpuArch;
using namespace RfaKenelCommon;

namespace BlockSparse {
    /**
     * @brief Block Sparse Attention Inference Kernel
     * This kernel implements block sparse attention where attention is computed only on
     * selected KV blocks specified by selectIdx. This reduces computation for long sequences
     * by focusing on relevant tokens.
     * @tparam BlockMmadQK Block-level QK matmul module
     * @tparam BlockMmadPV Block-level PV matmul module
     * @tparam EpilogueOnlineSoftmax Online softmax epilogue
     * @tparam EpilogueRescaleO Output rescaling epilogue
     * @tparam PAGED_CACHE_FLAG Whether to use paged KV cache
     * @tparam QUERY_LAYOUT Query tensor layout (0=TND, 1=BNSD)
     * @tparam KV_CACHE_LAYOUT KV cache layout (0=TND, 1=BNSD)
     */
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        bool PAGED_CACHE_FLAG,
        uint32_t QUERY_LAYOUT,
        uint32_t KV_CACHE_LAYOUT>
    class EagleQuantBlockSparseAttentionKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
    
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;
        static constexpr int32_t BASIC_BLOCK = 64;
        static constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        static constexpr uint32_t MASK_PATTERN_HALF_OFFSET = BASIC_BLOCK * 2 + LS_UB_TENSOR_OFFSET;
        static constexpr uint32_t MASK_PATTERN_FLOAT_OFFSET = BASIC_BLOCK * 2 + MASK_PATTERN_HALF_OFFSET;
        static constexpr uint32_t MASK_BIT_OFFSET = BASIC_BLOCK * 4 + MASK_PATTERN_FLOAT_OFFSET;
        static constexpr uint32_t MASK_IDX_OFFSET = BASIC_BLOCK  + MASK_BIT_OFFSET;
        static constexpr uint32_t SPARSE_IDX_OFFSET = BASIC_BLOCK * 4 + MASK_IDX_OFFSET;
        static constexpr uint32_t SELECT_NUM_IDX_OFFSET = BASIC_BLOCK * 4 + SPARSE_IDX_OFFSET;
        static constexpr uint32_t SYNC_OFFSET = BASIC_BLOCK * 4 + SELECT_NUM_IDX_OFFSET;
        
        __aicore__ inline
        EagleQuantBlockSparseAttentionKernel() {}

        __aicore__ inline void Mask2IdxAndCount(const AscendC::GlobalTensor<uint8_t> maskGM,
                                                AscendC::GlobalTensor<int32_t> selectIdxGM,
                                                AscendC::GlobalTensor<int32_t> selectNumGM)
        {
            static constexpr uint32_t PRE_ROW_TILE = 128;
            static constexpr uint32_t PRE_COL_TILE = 64;
            static constexpr uint32_t PRE_ELEM_NUM_PER_LOOP = PRE_ROW_TILE * PRE_COL_TILE;
            static constexpr uint32_t MASK_PAT_IN_UINT8 = 0;
            static constexpr uint32_t MASK_PAT_IN_FP16 = 2 * PRE_ELEM_NUM_PER_LOOP;
            static constexpr uint32_t MASK_PAT_IN_FP32 = 4 * PRE_ELEM_NUM_PER_LOOP;
            static constexpr uint32_t MASK_PAT_IN_BIT = 8 * PRE_ELEM_NUM_PER_LOOP;
            static constexpr uint32_t MASK_IDX = 9 * PRE_ELEM_NUM_PER_LOOP;
            static constexpr uint32_t RSVD_SPARSE_IDX = 13 * PRE_ELEM_NUM_PER_LOOP;
            static constexpr uint32_t RSVD_SPARSE_COUNT = 21 * PRE_ELEM_NUM_PER_LOOP;

            AscendC::LocalTensor<uint8_t> maskPatternUbLocal[2];
            AscendC::LocalTensor<int32_t> selectNumIdxUbLocal[2];
            AscendC::LocalTensor<uint8_t> maskPatternInBitUbLocal;
            AscendC::LocalTensor<uint32_t> maskPatternInBitUbLocalUint32;

            AscendC::LocalTensor<int32_t> maskIdxUbLocal;
            AscendC::LocalTensor<int32_t> sparseIdxUbLocal[2];
            AscendC::LocalTensor<half> maskPatternHalfLocal;
            AscendC::LocalTensor<float> maskPatternFloatLocal;

            for (uint32_t i = 0; i < 2; i++) {
                maskPatternUbLocal[i] = resource.ubBuf.template GetBufferByByte<uint8_t>(
                    MASK_PAT_IN_UINT8 + i * PRE_ELEM_NUM_PER_LOOP * sizeof(int8_t));
                sparseIdxUbLocal[i] = resource.ubBuf.template GetBufferByByte<int32_t>(
                    RSVD_SPARSE_IDX + i * PRE_ELEM_NUM_PER_LOOP * sizeof(int32_t));
                selectNumIdxUbLocal[i] = resource.ubBuf.template GetBufferByByte<int32_t>(
                    RSVD_SPARSE_COUNT + i * PRE_ROW_TILE * sizeof(int32_t));
            }
            maskPatternHalfLocal = resource.ubBuf.template GetBufferByByte<half>(MASK_PAT_IN_FP16);
            maskPatternFloatLocal = resource.ubBuf.template GetBufferByByte<float>(MASK_PAT_IN_FP32);
            maskPatternInBitUbLocal = resource.ubBuf.template GetBufferByByte<uint8_t>(MASK_PAT_IN_BIT);
            maskPatternInBitUbLocalUint32 = resource.ubBuf.template GetBufferByByte<uint32_t>(MASK_PAT_IN_BIT);
            maskIdxUbLocal = resource.ubBuf.template GetBufferByByte<int32_t>(MASK_IDX);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(1);

            uint32_t subCoreIdx = AscendC::GetBlockIdx();
            uint32_t curStartRowIdx = subCoreIdx * avgRowPerSubCore;
            uint32_t totalRowNumBlockMask = batch * qHeads * maxQBlockNum;
            uint32_t actDealtRow = (subCoreIdx == preActivateSubCoreNum - 1) ?
                (totalRowNumBlockMask - curStartRowIdx) : avgRowPerSubCore;
            if (subCoreIdx < preActivateSubCoreNum) {
                uint32_t rowLoop = CeilDiv(actDealtRow, PRE_ROW_TILE);
                uint32_t colLoop = CeilDiv(maxKvBlockNum, PRE_COL_TILE);
                uint32_t idxPingPongFlag = 0;
                uint32_t countPingPongFlag = 0;
                for (uint32_t i = 0; i < rowLoop; i++) {
                    uint32_t curLoopRowOffset = i * PRE_ROW_TILE;
                    int32_t actDealtRowCurLoop = (i == rowLoop - 1) ? (actDealtRow - curLoopRowOffset) : PRE_ROW_TILE;
                    uint64_t rsvdCountPerRow[PRE_ROW_TILE] = {0};
                    uint64_t rsvdCountPerRowCurColLoop[PRE_ROW_TILE] = {0};
                    for (uint32_t j = 0; j < colLoop; j++) {
                        uint32_t curLoopColOffset = j * PRE_COL_TILE;
                        uint32_t actDealtColCurLoop =
                            (j == colLoop - 1) ? (maxKvBlockNum - curLoopColOffset) : PRE_COL_TILE;
                        uint32_t actDealtColCurLoop32 = CeilDiv(actDealtColCurLoop, 32) * 32;
                        AscendC::CreateVecIndex(maskIdxUbLocal, static_cast<int32_t>(curLoopColOffset), PRE_COL_TILE);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(idxPingPongFlag);
                        int32_t maskOffset = curStartRowIdx * maxKvBlockNum +
                            curLoopRowOffset * maxKvBlockNum + curLoopColOffset;
                        AscendC::DataCopyPad(
                            maskPatternUbLocal[idxPingPongFlag],
                            maskGM[maskOffset],
                            AscendC::DataCopyExtParams(
                                actDealtRowCurLoop,
                                actDealtColCurLoop * sizeof(int8_t),
                                (maxKvBlockNum - actDealtColCurLoop) * sizeof(int8_t),
                                (PRE_COL_TILE - actDealtColCurLoop32) / 32, 0),
                            AscendC::DataCopyPadExtParams<uint8_t>(
                                true, 0, (actDealtColCurLoop32 - actDealtColCurLoop), 0));
                        
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                        AscendC::Cast(
                            maskPatternHalfLocal, maskPatternUbLocal[idxPingPongFlag],
                            AscendC::RoundMode::CAST_NONE, actDealtRowCurLoop * PRE_COL_TILE);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(idxPingPongFlag);
                        AscendC::Cast(
                            maskPatternFloatLocal, maskPatternHalfLocal,
                            AscendC::RoundMode::CAST_NONE, actDealtRowCurLoop * PRE_COL_TILE);
                        AscendC::PipeBarrier<PIPE_V>();
                        for (uint32_t k = 0; k < actDealtRowCurLoop; k++) {
                            if (actDealtColCurLoop32 != PRE_COL_TILE) {
                                AscendC::Duplicate(maskPatternFloatLocal[k * PRE_COL_TILE + actDealtColCurLoop32],
                                    (float)0, PRE_COL_TILE - actDealtColCurLoop32);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::CompareScalar(
                                maskPatternInBitUbLocal[k * PRE_COL_TILE],
                                maskPatternFloatLocal[k * PRE_COL_TILE],
                                (float)1.0, AscendC::CMPMODE::GE, PRE_COL_TILE);
                            AscendC::PipeBarrier<PIPE_V>();
                            if (k == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(idxPingPongFlag);
                            }
                            AscendC::GatherMask(
                                sparseIdxUbLocal[idxPingPongFlag][k * PRE_COL_TILE],
                                maskIdxUbLocal,
                                maskPatternInBitUbLocalUint32[k * PRE_COL_TILE / 4],
                                false,
                                (uint32_t)0, {1, 1, 0, 0}, rsvdCountPerRowCurColLoop[k]);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                            if (k == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                            }
                            AscendC::DataCopyPad(
                                selectIdxGM[curStartRowIdx * maxKvBlockNum +
                                    curLoopRowOffset * maxKvBlockNum + k * maxKvBlockNum + rsvdCountPerRow[k]],
                                sparseIdxUbLocal[idxPingPongFlag][k * PRE_COL_TILE],
                                AscendC::DataCopyExtParams(
                                    1, rsvdCountPerRowCurColLoop[k] * sizeof(int32_t), 0, 0, 0));
                            if (k == actDealtRowCurLoop - 1) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(idxPingPongFlag);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
                            rsvdCountPerRow[k] += rsvdCountPerRowCurColLoop[k];
                            if (k == actDealtRowCurLoop - 1) {
                                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                            }
                        }
                        idxPingPongFlag = 1 - idxPingPongFlag;
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(countPingPongFlag);
                    for (uint32_t k = 0; k < actDealtRowCurLoop; k++) {
                        selectNumIdxUbLocal[countPingPongFlag].SetValue(k, static_cast<int32_t>(rsvdCountPerRow[k]));
                    }
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(countPingPongFlag + 2);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(countPingPongFlag + 2);
                    AscendC::DataCopyPad(
                        selectNumGM[curStartRowIdx + curLoopRowOffset],
                        selectNumIdxUbLocal[countPingPongFlag],
                        AscendC::DataCopyExtParams(1, actDealtRowCurLoop * sizeof(int32_t), 0, 0, 0));
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(countPingPongFlag);
                    countPingPongFlag = 1 - countPingPongFlag;
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(1);
        }

        __aicore__ inline void operator()(EagleQuantBlockSparseAttentionKernelParams const &params)
        {
            __gm__ EagleQuantBlockSparseAttentionTilingData *blockSparseAttentionTilingData =
                reinterpret_cast<__gm__ EagleQuantBlockSparseAttentionTilingData *>(params.tiling);
            uint64_t mm1OutSize = blockSparseAttentionTilingData->mm1OutSize;
            uint64_t smOnlineOutSize = blockSparseAttentionTilingData->smOnlineOutSize;
            uint64_t mm2OutSize = blockSparseAttentionTilingData->mm2OutSize;
            uint64_t updateSize = blockSparseAttentionTilingData->updateSize;
            uint64_t selectNumIdxSize = blockSparseAttentionTilingData->selectNumIdxSize;
            uint64_t selectIdxSize = blockSparseAttentionTilingData->selectIdxSize;

            batch = blockSparseAttentionTilingData->batch;
            qHeads = blockSparseAttentionTilingData->numHeads;
            uint32_t kvHeads = blockSparseAttentionTilingData->kvHeads;

            uint32_t embed = blockSparseAttentionTilingData->embeddingSize;
            uint32_t pagedBlockSize = blockSparseAttentionTilingData->blockSize;
            uint32_t maxNumBlocksPerBatch = blockSparseAttentionTilingData->maxNumBlocksPerBatch;
            uint32_t firstBatchTaskNum = blockSparseAttentionTilingData->firstBatchTaskNum;
            uint32_t totalTaskNum = blockSparseAttentionTilingData->totalTaskNum;
            uint32_t maskType = blockSparseAttentionTilingData->maskType;
            ElementS scaleValue = static_cast<ElementS>(blockSparseAttentionTilingData->scaleValue);
            uint32_t totalQBlocks = blockSparseAttentionTilingData->totalQBlocks;
            maxKvBlockNum = blockSparseAttentionTilingData->maxKvBlockNum;
            uint32_t maxKvBlockNumPad = CeilDiv(maxKvBlockNum, 32) * 32;
            maxQBlockNum = blockSparseAttentionTilingData->maxQBlockNum;
            avgRowPerSubCore = blockSparseAttentionTilingData->avgRowNumPerSubCore;
            preActivateSubCoreNum = blockSparseAttentionTilingData->preActivateSubCoreNum;
            
            uint32_t qBlockX = blockSparseAttentionTilingData->blockShapeX;
            uint32_t qBlockY = blockSparseAttentionTilingData->blockShapeY;
            uint32_t qBlockNum = totalQBlocks / qBlockX;
            uint32_t qBlockInX = (qBlockX + BASIC_BLOCK_SIZE - 1) / BASIC_BLOCK_SIZE;
            uint32_t firstQBlockNum = blockSparseAttentionTilingData->firstQBlockNum;
            uint32_t maxQSeqlen = blockSparseAttentionTilingData->maxQSeqlen;
            uint32_t maxKvSeqlen = blockSparseAttentionTilingData->maxKvSeqlen;
            uint32_t useUniformQSeqlen = blockSparseAttentionTilingData->useUniformQSeqlen;
            uint32_t useUniformKvSeqlen = blockSparseAttentionTilingData->useUniformKvSeqlen;

            // Initialize global tensors
            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<int32_t> gSelectIdx;
            gSelectIdx.SetGlobalBuffer((__gm__ int32_t *)(params.workspace + mm1OutSize +
                smOnlineOutSize + mm2OutSize + updateSize + selectNumIdxSize));
            AscendC::GlobalTensor<int32_t> gSelectNumIdx;
            gSelectNumIdx.SetGlobalBuffer((__gm__ int32_t *)(params.workspace + mm1OutSize +
                smOnlineOutSize + mm2OutSize + updateSize));
            AscendC::GlobalTensor<uint8_t> gBlockSparseMask;
            gBlockSparseMask.SetGlobalBuffer((__gm__ uint8_t *)params.blockSparseMask);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)params.workspace);
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workspace + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workspace + mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workspace + mm1OutSize +
                smOnlineOutSize + mm2OutSize));
            
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_VEC__
            Mask2IdxAndCount(gBlockSparseMask, gSelectIdx, gSelectNumIdx);
#endif
            resource.pipe.Reset();
            AscendC::SyncAll<false>();

#ifdef __DAV_C220_CUBE__
            // Initialize hardware events for cube core
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            
            static constexpr uint32_t L1_QK_SIZE =
                BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
                BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
            BlockMmadQK blockMmadQK(resource);
            BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
#endif

#ifdef __DAV_C220_VEC__
            // Initialize hardware events for vector core
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
            EpilogueRescaleO epilogueRescaleO(resource);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            uint32_t bn = AscendC::GetSubBlockNum();
#endif
            // Calculate strides based on layout (compile-time optimization)
            // For TND: [T, N, D], stride = N * D
            // For BNSD: [B, N, S, D], strideB = N * S * D, strideN = S * D, strideS = D
            uint64_t strideQO = 0;
            uint64_t strideKV = 0;
            uint64_t strideQOB = 0;  // BNSD batch stride for Q
            uint64_t strideQON = 0;  // BNSD head stride for Q
            uint64_t strideQOS = 0;  // BNSD seq stride for Q
            uint64_t strideKVB = 0;  // BNSD batch stride for KV
            uint64_t strideKVN = 0;  // BNSD head stride for KV
            uint64_t strideKVS = 0;  // BNSD seq stride for KV
            
            if constexpr (QUERY_LAYOUT == 1) {  // BNSD_Q
                // BNSD: [B, N, S, D]
                // strideB = N * S * D, strideN = S * D, strideS = D
                // maxQSeqlen is the third dimension (S) of query shape, set in tiling
                strideQOB = qHeads * maxQSeqlen * embed;  // batch stride
                strideQON = maxQSeqlen * embed;  // head stride
                strideQOS = embed;  // seq stride
            } else {
                // TND: [T, N, D]
                strideQO = qHeads * embed;
            }
            
            if constexpr (KV_CACHE_LAYOUT == 1) {  // BNSD
                // BNSD: [B, N, S, D]
                // maxKvSeqlen is the third dimension (S) of value shape, set in tiling
                strideKVB = kvHeads * maxKvSeqlen * embed;  // batch stride
                strideKVN = maxKvSeqlen * embed;  // head stride
                strideKVS = embed;  // seq stride
            } else {
                // TND: [T, N, D]
                strideKV = kvHeads * embed;
            }
            
            uint32_t embedRound = AlignUp<uint32_t>(embed, BLOCK_SIZE);
            uint32_t groupSize = qHeads / kvHeads;

            uint64_t qBOffset = 0;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t oBOffset = 0;
            uint64_t blockBOffset = 0;
            uint64_t lseBOffset = 0;

            uint32_t preTotalTaskNum = 0;
            uint32_t preTotalQBlockNum = 0;
            uint32_t curBatch = 0;
            // 根据useUniformQSeqlen标志位决定使用actualSeqLengths数组还是maxQSeqlen
            uint32_t qSeqlen = useUniformQSeqlen ? maxQSeqlen :
                              static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(curBatch)));
            // 根据useUniformKvSeqlen标志位决定使用actualSeqLengthsKv数组还是maxKvSeqlen
            uint32_t kvSeqlen = useUniformKvSeqlen ? maxKvSeqlen :
                               static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch)));
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = curQNBlockTile == 0 ? 1 : (groupSize + curQNBlockTile - 1) / curQNBlockTile;
            uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = GetQBlocks(qSeqlen, qBlockX);
            uint32_t curTotalTaskNum = firstBatchTaskNum;
            uint32_t curQXBlockNum = (qSeqlen + qBlockX - 1) / qBlockX;
            uint32_t curTotalQBlockNum = firstQBlockNum;

            // Go through each task
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;
                    preTotalTaskNum = curTotalTaskNum;
                    preTotalQBlockNum = curTotalQBlockNum;
                    
                    // Update offsets based on layout (compile-time optimization)
                    if constexpr (QUERY_LAYOUT == 1) {  // BNSD_Q
                        // BNSD: [B, N, S, D], offset = batch * strideB
                        qBOffset = curBatch * strideQOB;
                        oBOffset = curBatch * strideQOB;
                        lseBOffset = curBatch * qHeads * maxQSeqlen;
                    } else {
                        // TND
                        qBOffset += qSeqlen * strideQO;
                        oBOffset += qSeqlen * strideQO;
                        lseBOffset += qSeqlen * qHeads;
                    }
                    
                    if constexpr (!PAGED_CACHE_FLAG) {
                        if constexpr (KV_CACHE_LAYOUT == 1) {  // BNSD
                            // BNSD: [B, N, S, D], offset = batch * strideB
                            kBOffset = curBatch * strideKVB;
                            vBOffset = curBatch * strideKVB;
                        } else {
                            // TND
                            kBOffset += kvSeqlen * strideKV;
                            vBOffset += kvSeqlen * strideKV;
                        }
                    } else {
                        blockBOffset += maxNumBlocksPerBatch;
                    }
                    // 根据useUniformQSeqlen标志位决定使用actualSeqLengths数组还是maxQSeqlen
                    qSeqlen = useUniformQSeqlen ? maxQSeqlen :
                             static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(curBatch)));
                    // 根据useUniformKvSeqlen标志位决定使用actualSeqLengthsKv数组还是maxKvSeqlen
                    kvSeqlen = useUniformKvSeqlen ? maxKvSeqlen :
                              static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch)));
                    curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    qNBlockNumPerGroup = curQNBlockTile == 0 ? 1 : (groupSize + curQNBlockTile - 1) / curQNBlockTile;
                    curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                    curQSBlockTile = GetQSBlockTile(kvSeqlen);
                    curQSBlockNum = GetQBlocks(qSeqlen, qBlockX);
                    curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                    curQXBlockNum = (qSeqlen + qBlockX - 1) / qBlockX;
                    curTotalQBlockNum += qHeads * curQXBlockNum;
                }

                // Q task splitting按照[qNBlockNum, qHead]
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
                uint32_t qXIdx = qSBlockIdx / qBlockInX;
                uint32_t qXInnerIdx = qSBlockIdx - qXIdx * qBlockInX;
                uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
                uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
                uint32_t xBlockNum = qSeqlen / qBlockX;
                uint32_t xTailNum = qSeqlen - xBlockNum * qBlockX;
                
                uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
                uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;

                uint32_t curSelectIdx = curBatch * qHeads * maxQBlockNum + qHeadIdx * maxQBlockNum + qXIdx;
                uint32_t curSelectNum = static_cast<uint32_t>(gSelectNumIdx.GetValue(curSelectIdx));
                if (curSelectNum == 0) {
                    continue;
                }
                uint32_t lastSelectIdx = static_cast<int32_t>(
                    gSelectIdx.GetValue(curSelectIdx * maxKvBlockNum + curSelectNum - 1));
                uint32_t kvYBlockNum = (kvSeqlen + qBlockY - 1) / qBlockY; // CeilDiv
                uint32_t curKvSeqLen = (lastSelectIdx == kvYBlockNum - 1 && kvSeqlen % qBlockY != 0) ?
                    qBlockY * (curSelectNum - 1) + kvSeqlen % qBlockY : qBlockY * curSelectNum;
                // Calculate offsets based on layout (compile-time optimization)
                uint64_t gmOffsetQ = 0;
                uint64_t gmOffsetK = 0;
                uint64_t gmOffsetV = 0;
                uint64_t gmOffsetO = 0;
                uint64_t gmOffsetLse = 0;
                if constexpr (QUERY_LAYOUT == 1) {  // BNSD_Q: [B, N, S, D]
                    // offset = batch * strideB + head * strideN + seq * strideS
                    uint32_t qSeqOffset = qXIdx * qBlockX + qXInnerIdx * BASIC_BLOCK_SIZE;
                    gmOffsetQ = qBOffset + qHeadIdx * strideQON + qSeqOffset * strideQOS;
                    gmOffsetO = oBOffset + qHeadIdx * strideQON + qSeqOffset * strideQOS;
                    // LSE format: [B, N, S] - strideN = maxQSeqlen
                    gmOffsetLse = lseBOffset + qHeadIdx * maxQSeqlen + qSeqOffset;
                } else {
                    // TND: [T, N, D]
                    uint32_t qSeqOffset = qXIdx * qBlockX + qXInnerIdx * BASIC_BLOCK_SIZE;
                    gmOffsetQ = qBOffset + qSeqOffset * strideQO + qHeadIdx * embed;
                    gmOffsetO = oBOffset + qSeqOffset * strideQO + qHeadIdx * embed;
                    // LSE format: [T, N] - same as Q/O but without D dimension
                    gmOffsetLse = lseBOffset + qSeqOffset * qHeads + qHeadIdx;
                }
                
                if constexpr (KV_CACHE_LAYOUT == 1) {  // BNSD: [B, N, S, D]
                    // offset = batch * strideB + head * strideN
                    // seq offset will be handled in blockMmadQK/blockMmadPV based on selectIdx
                    gmOffsetK = kBOffset + kvHeadIdx * strideKVN;
                    gmOffsetV = vBOffset + kvHeadIdx * strideKVN;
                } else {
                    // TND: [T, N, D]
                    gmOffsetK = kBOffset + kvHeadIdx * embed;
                    gmOffsetV = vBOffset + kvHeadIdx * embed;
                }

                uint32_t qSBlockSize = (qXIdx == xBlockNum) ?
                    (qXInnerIdx == xTailNum / curQSBlockTile ?
                        xTailNum - qXInnerIdx * curQSBlockTile : curQSBlockTile) :
                    ((qXInnerIdx == qBlockInX - 1) ? qBlockX - qXInnerIdx * curQSBlockTile : curQSBlockTile);

                uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1)) ?
                    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
                uint32_t rowNum = qSBlockSize * qNBlockSize;
                uint32_t rowNumRound = AlignUp<uint32_t>(rowNum, BLOCK_SIZE);

                uint32_t noSkipKvS = curKvSeqLen;
                uint32_t kvSLoopNumTotal = (noSkipKvS + pagedBlockSize - 1) / pagedBlockSize; // CeilDiv

                uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
                uint32_t stackSeqTile;
                uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
                uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
                int32_t stackSeqCount = 0;

#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                // For BNSD format, use strideKVS; for TND, use strideKV (compile-time)
                uint64_t actualStrideKV = 0;
                if constexpr (KV_CACHE_LAYOUT == 1) {
                    actualStrideKV = strideKVS;
                } else {
                    actualStrideKV = strideKV;
                }
                LayoutK layoutKTemp(actualStrideKV, blockStackNum * pagedBlockSize);
                LayoutV layoutVTemp(blockStackNum * pagedBlockSize, actualStrideKV);
                // Pass correct Q stride based on data format
                uint64_t qGmStride = 0;
                if constexpr (QUERY_LAYOUT == 1) {  // BNSD: [B, N, S, D]
                    qGmStride = strideQOS;  // embed
                } else {  // TND: [T, N, D]
                    qGmStride = strideQO;  // qHeads * embed
                }
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qGmStride);
#endif
                // Main computation loop: QK matmul -> Softmax -> PV matmul
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {
                    // Stage 1: QK matmul (computed on CUBE core)
                    if (kvSIdx < kvSLoopNumTotal) {
                        stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                        if (stackSeqTile >= pagedBlockSize * blockStackNum) {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1);
                        uint64_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                        GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                        // For BNSD format, pass strideKVS; for TND, pass strideKV (compile-time)
                        uint64_t actualStrideKVForQK = 0;
                        if constexpr (KV_CACHE_LAYOUT == 1) {
                            actualStrideKVForQK = strideKVS;
                        } else {
                            actualStrideKVForQK = strideKV;
                        }
                        blockMmadQK(gQ[gmOffsetQ],
                            gK[gmOffsetK],
                            gS[gmOffsetS],
                            gBlockTable[blockBOffset],
                            gSelectIdx[curSelectIdx * maxKvBlockNum],
                            layoutQTemp,
                            layoutKTemp,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            actualStrideKVForQK,
                            qBlockY,
                            curSelectNum,
                            kvYBlockNum,
                            kvSeqlen);
                        NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                        // Stage 2: Online softmax (computed on VECTOR core)
                        LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                        uint64_t gmOffsetP = gmOffsetS;

                        NpuArch::Arch::CrossCoreWaitFlag(qkReady);
                        // online softmax
                        epilogueOnlineSoftmax(gP[gmOffsetP],
                            gS[gmOffsetS],
                            layOutP,
                            layOutS,
                            actualBlockShapeQK,
                            (stackSeqCount == 0),
                            0,
                            qSBlockSize,
                            qNBlockSize,
                            curStackTileMod,
                            softmaxReady);
#endif
                    }
                    // Stage 3: PV matmul and output rescaling
                    if (kvSIdx >= preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;
                        stackSeqTile = noSkipKvS - nowkvSIdx * pagedBlockSize;
                        if (stackSeqTile >= pagedBlockSize * blockStackNum) {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                        uint64_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                        GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                        LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
#ifdef __DAV_C220_CUBE__
                        LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                        uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                        // For BNSD format, pass strideKVS; for TND, pass strideKV (compile-time)
                        uint64_t actualStrideKVForPV = 0;
                        if constexpr (KV_CACHE_LAYOUT == 1) {
                            actualStrideKVForPV = strideKVS;
                        } else {
                            actualStrideKVForPV = strideKV;
                        }
                        blockMmadPV(gP[gmOffsetP],
                            gV[gmOffsetV],
                            gOTmp[gmOffsetOTmp],
                            gBlockTable[blockBOffset],
                            gSelectIdx[curSelectIdx * maxKvBlockNum],
                            layoutPTemp,
                            layoutVTemp,
                            layoutOTmp,
                            actualBlockShapePV,
                            nowkvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            kvSeqlen,
                            actualStrideKVForPV,
                            blockStackNum,
                            softmaxReady,
                            qBlockY,
                            curSelectNum,
                            kvYBlockNum);
                        NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                        // Setup layoutO based on data format
                        LayoutO layoutO;
                        LayoutLse layoutLse;
                        if constexpr (QUERY_LAYOUT == 1) {  // BNSD: [B, N, S, D]
                            // BNSD format: stride[0] = embed (strideQOS)
                            layoutO = LayoutO(qSeqlen, embed);
                            layoutLse = LayoutLse(qSeqlen, 1);
                        } else {  // TND: [T, N, D]
                            // TND format: stride[0] = qHeads * embed (strideQO)
                            layoutO = LayoutO(qSeqlen, qHeads * embed);
                            layoutLse = LayoutLse(qSeqlen, qHeads);
                        }
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                        NpuArch::Arch::CrossCoreWaitFlag(pvReady);
                        // rescale O
                        epilogueRescaleO(
                            gO[gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            qNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),
                            nowkvSIdx + blockStackNum >= kvSLoopNumTotal,
                            curStackTileMod);
#endif
                    }
                    stackSeqCount++;
                }
            }
#ifdef __DAV_C220_CUBE__
            // Wait for all CUBE core events
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            // Wait for all VECTOR core events
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();
        }

    private:
        NpuArch::Arch::Resource<ArchTag> resource;
        NpuArch::Arch::CrossCoreFlag qkReady{QK_READY_ID};
        NpuArch::Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        NpuArch::Arch::CrossCoreFlag pvReady{PV_READY_ID};

        uint32_t batch{0};
        uint32_t qHeads{0};
        uint32_t maxQBlockNum{0};
        uint32_t maxKvBlockNum{0};
        uint32_t avgRowPerSubCore{0};
        uint32_t preActivateSubCoreNum{0};
    };

} // namespace BlockSparse

#endif // EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H

