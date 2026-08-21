/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_ARCH35_KERNEL_REGULAR
#define BSA_ARCH35_KERNEL_REGULAR

#include "../arch35/kernel_utils.hpp"


using namespace NpuArch;
using namespace tla;

namespace BsaKernelArch35 {

template <
    class EpilogueMask2Idx,
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO,
    Format qFormat,
    Format kvFormat>
class BsaRegularKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;
    
    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementSparseMask = typename EpilogueMask2Idx::ElementSparseMask;
    using ElementSparseIdx = typename EpilogueMask2Idx::ElementSparseIdx;
    using ElementSparseCount = typename EpilogueMask2Idx::ElementSparseCount;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
    // Methods
    __aicore__ inline
    BsaRegularKernelArch35() {}

    __aicore__ inline
    int GetQkStaticLoopNum()
    {
        return ((qBaseTile_ == BlockMmadQK::L0_TILE_M) &&
                (embed_ == BlockMmadQK::L0_TILE_K) &&
                ((kvBaseTile_ == BlockMmadQK::L0_TILE_N) ||
                 (kvBaseTile_ == BlockMmadQK::L0_TILE_N * 2U))) ?
            1 : Gemm::Block::Arch35MmadOpt::DYNAMIC_LOOP;
    }

    __aicore__ inline
    int GetPvStaticNLoopNum()
    {
        if (qBaseTile_ != BlockMmadPV::L0_TILE_M) {
            return Gemm::Block::Arch35MmadOpt::DYNAMIC_LOOP;
        }
        if (embed_ == BlockMmadPV::L0_TILE_N) {
            return 1;
        }
        if (embed_ == BlockMmadPV::L0_TILE_N * 2U) {
            return 2;
        }
        return Gemm::Block::Arch35MmadOpt::DYNAMIC_LOOP;
    }

    template<int quant_mode=0>
    __aicore__ inline
    void operator()(BsaKernelParamsArch35 const &params)
    {
        static_assert(quant_mode == 0);
        __gm__ EagleQuantBlockSparseAttentionTilingData *bsaTilingData =
            reinterpret_cast<__gm__ EagleQuantBlockSparseAttentionTilingData *>(params.tiling);
        FetchBaseShapeInfo(bsaTilingData);
        CalcOnChipBufTileInfo(bsaTilingData);
        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<uint8_t> gBlockSparseMask;
        gBlockSparseMask.SetGlobalBuffer((__gm__ uint8_t *)params.blockSparseMask);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx;
        if (bsaTilingData->sparsePatternMode == SPARSE_PATTERN_MODE_TABLE) {
            gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx *)params.blockSparseMask);
        } else {
            gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx *)params.workSpace);
        }
        AscendC::GlobalTensor<ElementSparseCount> gSparseCount;
        gSparseCount.SetGlobalBuffer((__gm__ ElementSparseCount *)(params.workSpace + sparseIdxSize_));

        // cross core data move dst buffers
        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // set reverse sync flags
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_VEC__
        uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
        if (bsaTilingData->sparsePatternMode == SPARSE_PATTERN_MODE_TABLE) {
            SparseTable2Count(resource, gSparseIdx, gSparseCount,
                totalRowNumBlockMask, yBlockNumAligned_, avgRowPerSubCore_, preActiveSubCoreNum_);
        } else {
            EpilogueMask2Idx epilogueMask2Idx(resource);
            epilogueMask2Idx(
                gBlockSparseMask, gSparseIdx, gSparseCount,
                totalRowNumBlockMask, yBlockNumAligned_, avgRowPerSubCore_, preActiveSubCoreNum_);
        }
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);
        // Calculate strides based on layout
        // For TND: [T, N, D], stride = N * D
        // For BNSD: [B, N, S, D], strideB = N * S * D, strideN = S * D, strideS = D
        int64_t strideQO = 0;
        int64_t strideKV = 0;
        int64_t strideQOB = 0;  // BNSD batch_ stride for Q
        int64_t strideQON = 0;  // BNSD head stride for Q
        int64_t strideQOS = 0;  // BNSD seq stride for Q
        int64_t strideKVB = 0;  // BNSD batch_ stride for KV
        int64_t strideKVN = 0;  // BNSD head stride for KV
        int64_t strideKVS = 0;  // BNSD seq stride for KV
        if constexpr (qFormat == Format::BNSD) {
            strideQOB = qHeads_ * qSeqlenAligned_ * embed_;  // batch_ stride
            strideQON = qSeqlenAligned_ * embed_;  // head stride
            strideQOS = embed_;  // seq stride
        } else if constexpr (qFormat == Format::TND) {
            strideQO = qHeads_ * embed_;
        }
        if constexpr (kvFormat == Format::BNSD) {
            strideKVB = kvHeads_ * kvSeqlenAligned_ * embed_;  // batch_ stride
            strideKVN = kvSeqlenAligned_ * embed_;  // head stride
            strideKVS = embed_;  // seq stride
        } else if constexpr (kvFormat == Format::TND) {
            strideKV = kvHeads_ * embed_;
        }
        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        int qkStaticLoopNum = GetQkStaticLoopNum();
        int pvStaticNLoopNum = GetPvStaticNLoopNum();
        int64_t qBOffset = 0;
        int64_t kBOffset = 0;
        int64_t vBOffset = 0;
        int64_t oBOffset = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        int64_t qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
        int64_t kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        // Go through each task
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                if constexpr (qFormat == Format::TND) {
                    qBOffset += qSeqlen * strideQO;
                    oBOffset += qSeqlen * strideQO;
                }
                if constexpr (kvFormat == Format::TND) {
                    kBOffset += kvSeqlen * strideKV;
                    vBOffset += kvSeqlen * strideKV;
                }
                qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
                kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
                curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
                curTotalTaskNum += curQSTileNum * qHeads_;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qHeadIdx = taskIdxCurBatch / curQSTileNum;
            uint32_t kvHeadIdx = qHeadIdx / groupSize;
            uint32_t qSTileIdx = taskIdxCurBatch - qHeadIdx * curQSTileNum;
            // corresponding xBlock index of cur task
            uint32_t xBlockIdx = qSTileIdx / qSTileNumPerFullXBlock;
            // the q base tile index within the corrsponding xBlock of cur task
            uint32_t qSTileIdxCurXBlock = qSTileIdx - xBlockIdx * qSTileNumPerFullXBlock;
            // corresponding head index of cur task
            // corresponding blockSparseMask gm offset of cur task
            // gmBlockSparseMask has the shape [B, qN, xBlockNumAligned, yBlockNumAligned]
            int64_t sparseMaskBOffset = curBatch * qHeads_ * xBlockNumAligned_ * yBlockNumAligned_;
            int64_t sparseMaskNOffset = qHeadIdx * xBlockNumAligned_ * yBlockNumAligned_;
            int64_t sparseMaskXOffset = xBlockIdx * yBlockNumAligned_;
            int64_t gmOffsetSparseMask = sparseMaskBOffset + sparseMaskNOffset + sparseMaskXOffset;
            // corresponding Q/K/V/O gm offset of cur task
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t qSOffset = xBlockIdx * blockShapeX_ + qSTileIdxCurXBlock * qBaseTile_;
            if constexpr (qFormat == Format::BNSD) {
                qBOffset = curBatch * strideQOB;
                oBOffset = curBatch * strideQOB;
                gmOffsetQ = qBOffset + qHeadIdx * strideQON + qSOffset * strideQOS;
                gmOffsetO = oBOffset + qHeadIdx * strideQON + qSOffset * strideQOS;
            } else if constexpr (qFormat == Format::TND) {
                gmOffsetQ = qBOffset + qSOffset * strideQO + qHeadIdx * embed_;
                gmOffsetO = oBOffset + qSOffset * strideQO + qHeadIdx * embed_;
            }
            if constexpr (kvFormat == Format::BNSD) {
                kBOffset = curBatch * strideKVB;
                vBOffset = curBatch * strideKVB;
                gmOffsetK = kBOffset + kvHeadIdx * strideKVN;
                gmOffsetV = vBOffset + kvHeadIdx * strideKVN;
            } else if constexpr (kvFormat == Format::TND) {
                gmOffsetK = kBOffset + kvHeadIdx * embed_;
                gmOffsetV = vBOffset + kvHeadIdx * embed_;
            }
            // the actual x block num of cur batch_, calc by actual qseqlen
            uint32_t xBlockNumAval = static_cast<uint32_t>(CeilDiv(qSeqlen, static_cast<int64_t>(blockShapeX_)));
            uint32_t xBlockSize = (xBlockIdx == xBlockNumAval - 1) ?
                (qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
            uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
            uint32_t qSTileSizeAct = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1) ?
                (xBlockSize - qSTileIdxCurXBlock * qBaseTile_) : qBaseTile_;
            // calc the gathered kvS from sparse mask
            uint32_t gmOffsetSparseCount =
                curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
            uint32_t yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);
            if (yBlockNumRsvd == 0) {
 	            continue;
 	        }
            uint32_t gmOffsetSparseIdx = gmOffsetSparseCount * yBlockNumAligned_;
            uint32_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
            uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);

            uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
            uint32_t lastYBlockSize = (lastSparseIdx == yBlockNumAval - 1) ?
                kvSeqlen - lastSparseIdx * blockShapeY_ : blockShapeY_;
            int64_t gatheredKvSeqlen = (yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;
            // the rowNum of cur task
            // no qS*qN combination even in GQA/MQA senario, since each qN has a different sparse pattern
            uint32_t rowNum = qSTileSizeAct;
            uint32_t rowNumRound = RoundUp(rowNum, 16);
            uint32_t kvSLoopNum = static_cast<uint32_t>(CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));
            uint32_t kvSTileSizeAct = kvBaseTile_;
#ifdef __DAV_CUBE__
            uint32_t kvShapeCol = 0;
            uint32_t qShapeCol = 0;
            if constexpr (qFormat == Format::BNSD) {
                qShapeCol = strideQOS;
            } else if constexpr (qFormat == Format::TND) {
                qShapeCol = strideQO;
            }
            if constexpr (kvFormat == Format::BNSD) {
                kvShapeCol = strideKVS;
            } else if constexpr (kvFormat == Format::TND) {
                kvShapeCol = strideKV;
            }

            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            if constexpr (std::is_same_v<LayoutQ, layout::RowMajor>) {
                auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, qShapeCol);
                auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
                blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
            } else {
                auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qShapeCol, qBaseTile_);
                auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
                blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
            }

            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kvShapeCol, kvBaseTile_);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});

            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, kvShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});
#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = 0;
            if constexpr (qFormat == Format::BNSD) {
                oShapeCol = strideQOS;
            } else if constexpr (qFormat == Format::TND) {
                oShapeCol = strideQO;
            }
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
#endif
            for (uint32_t gatheredKvSTileIdx = 0; gatheredKvSTileIdx < kvSLoopNum + PRE_LAUNCH; gatheredKvSTileIdx++) {
                if (gatheredKvSTileIdx < kvSLoopNum) {
                    if (gatheredKvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    // QK
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = gatheredKvSTileIdx % UB_S_OTMP_BUF_STAGES;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId],
                    ubSLayoutTla, Arch::PositionUB{});
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);
#ifdef __DAV_CUBE__
                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdx, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdx, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, true);
                    if (qkStaticLoopNum == 1) {
                        blockMmadQK.template operator()<0, 1, 1>(
                            gmKTensorTla, ubSTensorTla, gSparseIdx[gmOffsetSparseIdx],
                            actualBlockShapeQK,
                            gatheredKvSTileIdx, kvSeqlen,
                            kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                            prefixSumL0AStages, prefixSumL0BStages,
                            mm1ToSmFlag, 1.0f);
                    } else {
                        blockMmadQK.template operator()<0, -1, -1>(
                            gmKTensorTla, ubSTensorTla, gSparseIdx[gmOffsetSparseIdx],
                            actualBlockShapeQK,
                            gatheredKvSTileIdx, kvSeqlen,
                            kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                            prefixSumL0AStages, prefixSumL0BStages,
                            mm1ToSmFlag, 1.0f);
                    }
                    if (gatheredKvSTileIdx == kvSLoopNum - 1)
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
#endif
                    // SM
                    uint32_t l1PBufId = gatheredKvSTileIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId],
                    l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    epilogueOnlineSoftmax(
                        l1PTensorTla,
                        actualBlockShapeQK,
                        (gatheredKvSTileIdx == 0),
                        ubSBufId,
                        l1PBufId,
                        mm1ToSmFlag,
                        smToMm2Flag);
#endif
                }
                if (gatheredKvSTileIdx >= PRE_LAUNCH) {
                    uint32_t gatheredKvSTileIdxDe = gatheredKvSTileIdx - PRE_LAUNCH;
                    if (gatheredKvSTileIdxDe == kvSLoopNum - 1) {
                        kvSTileSizeAct = gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    // PV
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = gatheredKvSTileIdxDe % UB_S_OTMP_BUF_STAGES;
                    // 核间同步flagId规律：每份dst对应一个id，从小到大顺序依次为qk->sm, sm->pv, pv->re
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = gatheredKvSTileIdxDe % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                    auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor[ubOTmpBufId],
                        ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdxDe, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdxDe, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, false);
                    if (pvStaticNLoopNum == 1) {
                        blockMmadPV.template operator()<1, 1>(
                            gmVTensorTla, ubOTmpTensorTla, gSparseIdx[gmOffsetSparseIdx],
                            actualBlockShapePV,
                            gatheredKvSTileIdxDe, kvSeqlen,
                            kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                            prefixSumL0AStages, prefixSumL0BStages,
                            smToMm2Flag, mm2ToReFlag);
                    } else if (pvStaticNLoopNum == 2) {
                        blockMmadPV.template operator()<1, 2>(
                            gmVTensorTla, ubOTmpTensorTla, gSparseIdx[gmOffsetSparseIdx],
                            actualBlockShapePV,
                            gatheredKvSTileIdxDe, kvSeqlen,
                            kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                            prefixSumL0AStages, prefixSumL0BStages,
                            smToMm2Flag, mm2ToReFlag);
                    } else {
                        blockMmadPV.template operator()<-1, -1>(
                            gmVTensorTla, ubOTmpTensorTla, gSparseIdx[gmOffsetSparseIdx],
                            actualBlockShapePV,
                            gatheredKvSTileIdxDe, kvSeqlen,
                            kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                            prefixSumL0AStages, prefixSumL0BStages,
                            smToMm2Flag, mm2ToReFlag);
                    }
#endif
#ifdef __DAV_VEC__
                    // rescale O
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = gatheredKvSTileIdxDe % (PRE_LAUNCH + 1);
                    epilogueRescaleO(
                        gO[gmOffsetO], actualBlockShapePV,
                        curTileMod, gatheredKvSTileIdxDe,
                        (gatheredKvSTileIdxDe == 0),
                        (gatheredKvSTileIdxDe == kvSLoopNum - 1),
                        mm2ToReFlag);
#endif
                }
            }
        }
        // release reverse sync flags
        ReleaseSyncFlags<4, 4, 4>();
    }

    __aicore__ inline
    void FetchBaseShapeInfo(__gm__ EagleQuantBlockSparseAttentionTilingData *bsaTilingData)
    {
        batch_ = bsaTilingData->batch;
        qHeads_ = bsaTilingData->numHeads;
        kvHeads_ = bsaTilingData->kvHeads;
        embed_ = bsaTilingData->embeddingSize;
        firstBatchTaskNum_ = bsaTilingData->firstBatchTaskNum;
        totalTaskNum_ = bsaTilingData->totalTaskNum;
        blockShapeX_ = bsaTilingData->blockShapeX;
        blockShapeY_ = bsaTilingData->blockShapeY;
        scaleValue_ = bsaTilingData->scaleValue;
        // mask2idx tile info
        xBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.xBlockNumAligned;
        yBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.yBlockNumAligned;
        avgRowPerSubCore_ = bsaTilingData->BsaMask2IdxTileInfo.avgRowPerSubCore;
        preActiveSubCoreNum_ = bsaTilingData->BsaMask2IdxTileInfo.preActiveSubCoreNum;
        // base tile info
        qBaseTile_ = bsaTilingData->BsaBaseTileInfo.qBaseTile;
        kvBaseTile_ = bsaTilingData->BsaBaseTileInfo.kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = (!bsaTilingData->useUniformQSeqlen) && (!bsaTilingData->useUniformKvSeqlen);
        sparseIdxSize_ = bsaTilingData->selectIdxSize;
        // aligned seqlen q & kv
        qSeqlenAligned_ = bsaTilingData->maxQSeqlen;
        kvSeqlenAligned_ = bsaTilingData->maxKvSeqlen;
    }

    __aicore__ inline
    void CalcOnChipBufTileInfo(__gm__ EagleQuantBlockSparseAttentionTilingData *bsaTilingData)
    {
        mm1L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileM;
        mm1L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileN;
        mm1L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKLeft;
        mm1L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKRight;
        mm2L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileM;
        mm2L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileN;
        mm2L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKLeft;
        mm2L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKRight;
        qL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.qL1BufNum;
        kL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.kL1BufNum;
        vL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.vL1BufNum;
        pL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.pL1BufNum;
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
            qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
            pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
            mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        mm1L0ATotalStages_ = (qBaseTile_ / BlockMmadQK::L0_TILE_M) * (embed_ / BlockMmadQK::L0_TILE_K);
        mm1L0BTotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        mm2L0ATotalStages_ = (qBaseTile_ / BlockMmadPV::L0_TILE_M) * (kvBaseTile_ / BlockMmadPV::L0_TILE_K);
        mm2L0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (embed_ / BlockMmadPV::L0_TILE_N);
    }

    __aicore__ inline
    uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(
        uint32_t gatheredKvSTileIdx, uint32_t singleMm1L0Stages,
        uint32_t singleMm2L0Stages, uint32_t kvSLoopNum,
        bool isCurPhaseMm1)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseMm1) {
            prefixSumStages = (gatheredKvSTileIdx <= PRE_LAUNCH) ?
                gatheredKvSTileIdx * singleMm1L0Stages :
                gatheredKvSTileIdx * singleMm1L0Stages + (gatheredKvSTileIdx - PRE_LAUNCH) * singleMm2L0Stages;
        } else {
            prefixSumStages = (gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH) ?
                (gatheredKvSTileIdx + 1 + PRE_LAUNCH) * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages:
                kvSLoopNum * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages;
        }
        return prefixSumStages;
    }

    __aicore__ inline
    void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
        AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        uint32_t rowNumPerSubCore = EpilogueOnlineSoftmax::SM_ROW_MAX_ELEM_NUM;
        uint32_t colNumPerSubCore = EpilogueOnlineSoftmax::SM_COL_MAX_ELEM_NUM;
        uint32_t rescaleCol = EpilogueRescaleO::RESCALE_COL_MAX_ELEM_NUM;
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * i);
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
                rowNumPerSubCore * rescaleCol * sizeof(ElementOTmp) * i);
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline
    void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        // L0A
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // L0B
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // VSDB (L1 Value Scale Double Buffer)
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        // cross core sync
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // mask2index
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(1);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(5);
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(6);
        }
#endif
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline
    void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
         // VSDB (L1 Value Scale Double Buffer)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    /*
    tiling info, which are const in each kernel launch
    */
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    uint32_t blockShapeX_;
    uint32_t blockShapeY_;
    float scaleValue_;
    // mask2idx tile info
    uint32_t xBlockNumAligned_;
    uint32_t yBlockNumAligned_;
    uint32_t avgRowPerSubCore_;
    uint32_t preActiveSubCoreNum_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // whether actual seqlen is provided
    uint32_t actSeqAval_;
    // workspace size
    uint64_t sparseIdxSize_;
    // aligned seqlen q & kv
    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;
    // L1 tile info
    uint32_t mm1L1TileM_;
    uint32_t mm1L1TileN_;
    uint32_t mm1L1TileKLeft_;
    uint32_t mm1L1TileKRight_;
    uint32_t mm2L1TileM_;
    uint32_t mm2L1TileN_;
    uint32_t mm2L1TileKLeft_;
    uint32_t mm2L1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;
    uint32_t mm1L0ATotalStages_;
    uint32_t mm1L0BTotalStages_;
    uint32_t mm2L0ATotalStages_;
    uint32_t mm2L0BTotalStages_;
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
};

}

#endif
