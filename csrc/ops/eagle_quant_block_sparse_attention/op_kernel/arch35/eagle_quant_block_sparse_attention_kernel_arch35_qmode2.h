/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_ARCH35_KERNEL_QMODE2
#define BSA_ARCH35_KERNEL_QMODE2

#include "eagle_quant_block_sparse_attention_kernel_arch35_base.h"

using namespace NpuArch;
using namespace tla;

namespace BsaKernelArch35 {

template <class EpilogueMask2Idx, class BlockMmadQK_, class EpilogueOnlineSoftmax, class BlockMmadPV_,
    class EpilogueRescaleO, Format qFormat, Format kvFormat>
class BsaRegularKernelArch35<BsaRegularKernelArch35QMode2, EpilogueMask2Idx, BlockMmadQK_, EpilogueOnlineSoftmax,
    BlockMmadPV_, EpilogueRescaleO, qFormat, kvFormat> {
  public:
    using BlockMmadQK = BlockMmadQK_;
    using BlockMmadPV = BlockMmadPV_;
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

    using LayoutQ = layout::ColumnMajor;
    using LayoutK = layout::RowMajor;
    // qmode2 uses RowMajor because QK is calculated as (K * QT)T.
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint32_t PRE_LAUNCH = 3;
    static constexpr uint32_t FINAL = 1;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_BUF_STAGES = 3;
    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t QUANT_BLOCK_SIZE = 32;
    static constexpr uint32_t KVTILE_ALIGN_IN_SOFTMAX = 128;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;
    static constexpr uint32_t MM2_L1_ADDR_START = ArchTag::L1_SIZE / 2;

    // Methods
    __aicore__ inline BsaRegularKernelArch35() {}

    __aicore__ inline void operator()(BsaKernelParamsArch35 const &params) {
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
        AscendC::GlobalTensor<float8_e8m0_t> queryScale;
        queryScale.SetGlobalBuffer((__gm__ float8_e8m0_t *)(params.query_scale));
        AscendC::GlobalTensor<float8_e8m0_t> keyScale;
        keyScale.SetGlobalBuffer((__gm__ float8_e8m0_t *)(params.key_scale));
        AscendC::GlobalTensor<float8_e8m0_t> valueScale;
        valueScale.SetGlobalBuffer((__gm__ float8_e8m0_t *)(params.value_scale));
        AscendC::GlobalTensor<float> queryLevel0Scale;
        if (params.query_level0_scale != nullptr) {
            queryLevel0Scale.SetGlobalBuffer((__gm__ float *)(params.query_level0_scale));
        }
        AscendC::GlobalTensor<float> keyLevel0Scale;
        if (params.key_level0_scale != nullptr) {
            keyLevel0Scale.SetGlobalBuffer((__gm__ float *)(params.key_level0_scale));
        }
        AscendC::GlobalTensor<float> valueLevel0Scale;
        if (params.value_level0_scale != nullptr) {
            valueLevel0Scale.SetGlobalBuffer((__gm__ float *)(params.value_level0_scale));
        }

        // cross core data move dst buffers
        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<float8_e8m0_t> l1PSTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, l1PSTensor, ubSTensor, ubOTmpTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        bool isSparseIndexGiven = bsaTilingData->sparsePatternMode == SPARSE_PATTERN_MODE_TABLE;
        // set reverse sync flags
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_VEC__
        uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
        if (isSparseIndexGiven) {
            SparseTable2Count(resource, gSparseIdx, gSparseCount, totalRowNumBlockMask, tableShapeY, avgRowPerSubCore_,
                preActiveSubCoreNum_);
        } else {
            EpilogueMask2Idx epilogueMask2Idx(resource);
            epilogueMask2Idx(gBlockSparseMask, gSparseIdx, gSparseCount, totalRowNumBlockMask, yBlockNumAligned_,
                avgRowPerSubCore_, preActiveSubCoreNum_);
        }
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
        blockMmadPV.InitVAppendColumns(BlockMmadQK::L0A_STAGES);
        const uint64_t fixpipeScale = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scaleValue_));
        AscendC::SetFixpipePreQuantFlag(fixpipeScale);
        AscendC::PipeBarrier<PIPE_FIX>();
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource);
        EpilogueRescaleO epilogueRescaleO(resource, embed_);
        const auto &vLv0ScaleTensor = epilogueRescaleO.vLv0ScaleTensor;

#endif
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);
        // Calculate strides based on layout
        // For TND: [T, N, D], stride = N * D
        // For BNSD: [B, N, S, D], strideB = N * S * D, strideN = S * D, strideS = D
        int64_t strideQO = 0;
        int64_t strideKV = 0;
        int64_t strideQOB = 0; // BNSD batch_ stride for Q
        int64_t strideQON = 0; // BNSD head stride for Q
        int64_t strideQOS = 0; // BNSD seq stride for Q
        int64_t strideKVB = 0; // BNSD batch_ stride for KV
        int64_t strideKVN = 0; // BNSD head stride for KV
        int64_t strideKVS = 0; // BNSD seq stride for KV
        uint32_t strideVScaleHead = 0;
        if constexpr (qFormat == Format::BNSD) {
            strideQOB = qHeads_ * qSeqlenAligned_ * embed_; // batch_ stride
            strideQON = qSeqlenAligned_ * embed_; // head stride
            strideQOS = embed_; // seq stride
        } else if constexpr (qFormat == Format::TND) {
            strideQO = qHeads_ * embed_;
        }
        if constexpr (kvFormat == Format::BNSD) {
            strideKVB = kvHeads_ * kvSeqlenAligned_ * embed_; // batch_ stride
            strideKVN = kvSeqlenAligned_ * embed_; // head stride
            strideKVS = embed_; // seq stride
        } else if constexpr (kvFormat == Format::TND) {
            strideKV = kvHeads_ * embed_;
        }
        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;

        if constexpr (kvFormat == Format::BNSD) {
            strideVScaleHead = CeilDiv(kvSeqlenAligned_, 64U) * embed_ * 2U;
        }
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
            // int64_t sparseMaskBOffset = curBatch * qHeads_ * xBlockNumAligned_ * yBlockNumAligned_;
            // int64_t sparseMaskNOffset = qHeadIdx * xBlockNumAligned_ * yBlockNumAligned_;
            // int64_t sparseMaskXOffset = xBlockIdx * yBlockNumAligned_;
            // int64_t gmOffsetSparseMask = sparseMaskBOffset + sparseMaskNOffset + sparseMaskXOffset;
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
            uint32_t xBlockSize =
                (xBlockIdx == xBlockNumAval - 1) ? (qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
            uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
            uint32_t qSTileSizeAct = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1)
                ? (xBlockSize - qSTileIdxCurXBlock * qBaseTile_)
                : qBaseTile_;
            // calc the gathered kvS from sparse mask
            uint32_t gmOffsetSparseCount =
                curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
            uint32_t yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);
            if (yBlockNumRsvd == 0) {
                continue;
            }
            // uint32_t gmOffsetSparseIdx = gmOffsetSparseCount * (isSparseIndexGiven ? yBlockNumAligned_ : RoundUp(yBlockNumAligned_, 32)); // new mask2idx 每行 32对齐
            uint32_t gmOffsetSparseIdx = gmOffsetSparseCount * tableShapeY;
            uint32_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
            uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);

            uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
            uint32_t lastYBlockSize =
                (lastSparseIdx == yBlockNumAval - 1) ? kvSeqlen - lastSparseIdx * blockShapeY_ : blockShapeY_;
            int64_t gatheredKvSeqlen = (yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;
            // the rowNum of cur task
            // no qS*qN combination even in GQA/MQA senario, since each qN has a different sparse pattern
            uint32_t rowNum = qSTileSizeAct;
            uint32_t rowNumRound = RoundUp(rowNum, std::is_same_v<LayoutQ, layout::RowMajor> ? 16 : 64);
            uint32_t kvSLoopNum = static_cast<uint32_t>(CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));
            uint32_t kvSTileSizeAct = kvBaseTile_;
            const uint32_t pvRowNumAligned = RoundUp(rowNum, 128);
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
            uint32_t qsOffset = gmOffsetQ / QUANT_BLOCK_SIZE;

            //MakeMxScaleLayout 参数包含<Element LayoutTage isMxScaleB>(rows, cols)
            auto gmQSLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, layout::ColumnMajor, true>(
                CeilDiv(embed_, MX_SCALE_GROUP_NUM), rowNum);
            auto gmQSTensorTla = tla::MakeTensor(queryScale[qsOffset], gmQSLayoutTla, Arch::PositionGM{});

            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qShapeCol, qBaseTile_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            blockMmadQK.loadQGM(gmQTensorTla, gmQSTensorTla, actualBlockShapeQ);

            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kvSeqlenAligned_, kvShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});

            uint32_t ksOffset = gmOffsetK / QUANT_BLOCK_SIZE;
            auto gmKSLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, layout::RowMajor, false>(
                kvSeqlen, CeilDiv(embed_, MX_SCALE_GROUP_NUM));
            auto gmKSTensorTla = tla::MakeTensor(keyScale[ksOffset], gmKSLayoutTla, Arch::PositionGM{});

            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, kvShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});

            uint32_t vsOffset = gmOffsetV / MX_SCALE_GROUP_NUM;
            if constexpr (kvFormat == Format::BNSD) {
                vsOffset = (curBatch * kvHeads_ + kvHeadIdx) * strideVScaleHead;
            }
            auto gmVSLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, layout::RowMajor, true>(
                CeilDiv(kvSeqlen, MX_SCALE_GROUP_NUM), embed_);
            auto gmVSTensorTla = tla::MakeTensor(valueScale[vsOffset], gmVSLayoutTla, Arch::PositionGM{});

            const uint32_t tailKvTile =
                static_cast<uint32_t>(gatheredKvSeqlen - static_cast<int64_t>(kvSLoopNum - 1) * kvBaseTile_);
            blockMmadQK.InitL1ToL0LoadParams(kvBaseTile_, RoundUp(tailKvTile, 64), rowNum, embed_);
            blockMmadPV.InitL1ToL0LoadParams(
                kvBaseTile_, RoundUp(tailKvTile, 128), pvRowNumAligned, embed_ + BlockMmadPV::V_APPEND_COLS);

            uint32_t qkL0BufId = 0;
            uint32_t pvL0BufId = 0;
            uint32_t pvL0EventId = BlockMmadQK::L0A_STAGES;
            for (uint32_t preloadIdx = 0; preloadIdx < PRE_LAUNCH && preloadIdx < kvSLoopNum; ++preloadIdx) {
                const uint32_t preloadSize = (preloadIdx == kvSLoopNum - 1) ? tailKvTile : kvBaseTile_;
                GemmCoord preloadShape{rowNum, preloadSize, embed_};
                blockMmadQK.IssueKTileLoad(gmKTensorTla, gmKSTensorTla, gSparseIdx[gmOffsetSparseIdx], preloadShape,
                    preloadIdx, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd, preloadIdx == 0);
            }
#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = 0;
            if constexpr (qFormat == Format::BNSD) {
                oShapeCol = strideQOS;
            } else if constexpr (qFormat == Format::TND) {
                oShapeCol = strideQO;
            }
#endif
            uint32_t rescaleCount = 0;
            for (uint32_t gatheredKvSTileIdx = 0; gatheredKvSTileIdx < kvSLoopNum + PRE_LAUNCH + FINAL;
                 gatheredKvSTileIdx++) {
#ifdef __DAV_VEC__
                constexpr uint32_t SM_PRELUNCH = PRE_LAUNCH - 1;
                if (gatheredKvSTileIdx >= SM_PRELUNCH && gatheredKvSTileIdx - SM_PRELUNCH < kvSLoopNum) {
                    uint32_t idxRes = gatheredKvSTileIdx - SM_PRELUNCH;
                    uint32_t kvSTileSizeActRes =
                        (idxRes == kvSLoopNum - 1) ? gatheredKvSeqlen - idxRes * kvBaseTile_ : kvBaseTile_;
                    uint32_t ubSBufIdRes = idxRes % UB_S_BUF_STAGES;
                    uint32_t l1PBufIdRes = idxRes % pL1BufNum_;
                    Arch::CrossCoreFlag mm1ToSmFlagRes(ubSBufIdRes);
                    Arch::CrossCoreFlag smToMm2FlagRes(l1PBufIdRes + UB_S_BUF_STAGES);

                    epilogueOnlineSoftmax.ResolveSoftmax(l1PTensor[l1PBufIdRes], l1PSTensor[l1PBufIdRes],
                        kvSTileSizeActRes, rowNum, (idxRes == 0), ubSBufIdRes, l1PBufIdRes, mm1ToSmFlagRes,
                        smToMm2FlagRes);
                }
#endif
                if (gatheredKvSTileIdx < kvSLoopNum) {
                    kvSTileSizeAct = gatheredKvSTileIdx == kvSLoopNum - 1
                        ? gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_
                        : kvBaseTile_;
                    // QK
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t qkMAligned = RoundUp(rowNum, 64);
                    uint32_t kvSTileSizeAligned = RoundUp(kvSTileSizeAct, KVTILE_ALIGN_IN_SOFTMAX);
                    uint32_t ubSBufId = gatheredKvSTileIdx % UB_S_BUF_STAGES;
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);
                    // The 64-column physical shape is required by the future FP4 zN Softmax layout.
                    uint32_t qkUbRows = CeilDiv(kvSTileSizeAligned * qkMAligned, 64U);
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(qkUbRows, 64);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
#ifdef __DAV_CUBE__
                    blockMmadQK.ComputeLoadedKTile(ubSTensorTla, actualBlockShapeQK, gatheredKvSTileIdx, kvSeqlen,
                        kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd, mm1ToSmFlag, qkL0BufId);

                    const uint32_t nextKTileIdx = gatheredKvSTileIdx + PRE_LAUNCH;
                    if (nextKTileIdx < kvSLoopNum) {
                        const uint32_t nextKTileSize = (nextKTileIdx == kvSLoopNum - 1) ? tailKvTile : kvBaseTile_;
                        GemmCoord nextKShape{rowNum, nextKTileSize, embed_};
                        blockMmadQK.IssueKTileLoad(gmKTensorTla, gmKSTensorTla, gSparseIdx[gmOffsetSparseIdx],
                            nextKShape, nextKTileIdx, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                            false);
                    }

                    GemmCoord preloadVShape{pvRowNumAligned, embed_, kvSTileSizeAct};
                    blockMmadPV.IssueVTileLoad(gmVTensorTla, gmVSTensorTla, gSparseIdx[gmOffsetSparseIdx],
                        preloadVShape, gatheredKvSTileIdx, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval,
                        yBlockNumRsvd);

#endif
                    // SM
                    uint32_t l1PBufId = gatheredKvSTileIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_BUF_STAGES;

                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::nZ>(rowNumRound, kvSTileSizeAligned);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
                    uint32_t pScaleRowAligned = RoundUp(rowNum, 128);
                    auto l1PSLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, layout::zZ, false>(
                        pScaleRowAligned, CeilDiv(kvSTileSizeAligned, MX_SCALE_GROUP_NUM));
                    auto l1PSTensorTla = tla::MakeTensor(l1PSTensor[l1PBufId], l1PSLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    uint32_t v0ScaleOffset = curBatch * kvHeads_ * embed_ + kvHeadIdx * embed_;
                    if (gatheredKvSTileIdx == kvSLoopNum - 1) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
                        AscendC::DataCopy(vLv0ScaleTensor, valueLevel0Scale[v0ScaleOffset], embed_);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                    }
                    // int32_t sparseTableStartOffset = gmOffsetSparseIdx + (gatheredKvSTileIdx) * (kvBaseTile_/blockShapeY_);
                    epilogueOnlineSoftmax.IssueSoftmax(
                        actualBlockShapeQK, (gatheredKvSTileIdx == 0), ubSBufId, l1PBufId, mm1ToSmFlag, smToMm2Flag);
                    AscendC::PipeBarrier<PIPE_V>();
#endif
                }
                if (gatheredKvSTileIdx >= PRE_LAUNCH) {
                    uint32_t gatheredKvSTileIdxDe = gatheredKvSTileIdx - PRE_LAUNCH;
                    kvSTileSizeAct = gatheredKvSTileIdxDe == kvSLoopNum - 1
                        ? gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_
                        : kvBaseTile_;
                    // PV
                    uint32_t pvRowNumAligned = RoundUp(rowNum, 128);
                    GemmCoord actualBlockShapePV{pvRowNumAligned, embed_, kvSTileSizeAct};

                    uint32_t ubOTmpBufId = rescaleCount % UB_OTMP_BUF_STAGES;
                    // 核间同步flagId规律：每份dst对应一个id，从小到大顺序依次为qk->sm, sm->pv, pv->re
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = gatheredKvSTileIdxDe % pL1BufNum_;
                    uint32_t pvOutputCols = embedRound + BlockMmadPV::V_APPEND_COLS - BlockMmadPV::V_COPY_OUT_SKIP_COLS;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(pvRowNumAligned, pvOutputCols);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_BUF_STAGES;

                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t pvPStorageK = RoundUp(kvSTileSizeAct, KVTILE_ALIGN_IN_SOFTMAX);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, layout::nZ>(pvRowNumAligned, pvPStorageK);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
                    auto l1PSLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, layout::zZ, false>(
                        pvRowNumAligned, CeilDiv(pvPStorageK, MX_SCALE_GROUP_NUM));
                    auto l1PSTensorTla = tla::MakeTensor(l1PSTensor[l1PBufId], l1PSLayoutTla, Arch::PositionL1{});

                    bool isUpdateLastO = false;
                    if (gatheredKvSTileIdxDe < kvSLoopNum) {
                        constexpr uint16_t ssbufReadyFlagId = EpilogueOnlineSoftmax::SSBUF_READY_FLAG_ID;
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(ssbufReadyFlagId);
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(ssbufReadyFlagId + V0_V1_FLAG_ID_OFFSET);
                        auto ssbufBase = reinterpret_cast<__ssbuf__ uint8_t *>(AscendC::GetSsbufBaseAddr());
                        const uint32_t vec0Slot = 2U * l1PBufId;
                        const uint32_t vec1Slot = vec0Slot + 1U;
                        auto vec0UpdateMsg = reinterpret_cast<__ssbuf__ uint32_t *>(
                            ssbufBase + vec0Slot * EpilogueOnlineSoftmax::SSBUF_SLOT_BYTES);
                        auto vec1UpdateMsg = reinterpret_cast<__ssbuf__ uint32_t *>(
                            ssbufBase + vec1Slot * EpilogueOnlineSoftmax::SSBUF_SLOT_BYTES);
                        // l1PBufId 和 l1PBufId-1的max比较得出isUpdateLastO vec0 | vec1
                        isUpdateLastO = (*vec0UpdateMsg | *vec1UpdateMsg) > 0;
                    } else { // Final Loop
                        isUpdateLastO = true;
                    }

                    blockMmadPV.ComputeLoadedVTile(l1PTensorTla, l1PSTensorTla, ubOTmpTensorTla, actualBlockShapePV,
                        gatheredKvSTileIdxDe, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                        gatheredKvSTileIdxDe == 0U || isUpdateLastO, isUpdateLastO, smToMm2Flag, mm2ToReFlag,
                        kvSLoopNum, //! if gatheredKvSTileIdxDe == kvsloopnum, only fixpipe, no mmad
                        pvL0BufId, pvL0EventId);
                    if (isUpdateLastO) {
                        rescaleCount++;
                    }
#endif
#ifdef __DAV_VEC__
                    uint32_t curTileMod = gatheredKvSTileIdxDe % (PRE_LAUNCH + 1);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    bool isUpdateLastO = false;
                    if (gatheredKvSTileIdxDe < kvSLoopNum) {
                        AscendC::CrossCoreWaitFlag<1, PIPE_MTE3>(EpilogueOnlineSoftmax::AIV_UPDATE_SYNC_FLAG_ID);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EpilogueOnlineSoftmax::SSBUF_SYNC_EVENT_ID);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EpilogueOnlineSoftmax::SSBUF_SYNC_EVENT_ID);

                        auto ssbufBase = reinterpret_cast<__ssbuf__ uint8_t *>(AscendC::GetSsbufBaseAddr());
                        const uint32_t vec0Slot = 2U * curTileMod;
                        const uint32_t vec1Slot = vec0Slot + 1U;
                        auto vec0UpdateMsg = reinterpret_cast<__ssbuf__ uint32_t *>(
                            ssbufBase + vec0Slot * EpilogueOnlineSoftmax::SSBUF_SLOT_BYTES);
                        auto vec1UpdateMsg = reinterpret_cast<__ssbuf__ uint32_t *>(
                            ssbufBase + vec1Slot * EpilogueOnlineSoftmax::SSBUF_SLOT_BYTES);
                        isUpdateLastO = (*vec0UpdateMsg | *vec1UpdateMsg) > 0U;
                    }
                    if (isUpdateLastO) { //  rescale only if need
                        epilogueRescaleO(gO[gmOffsetO], (GemmCoord){rowNum, embed_, kvSTileSizeAct}, curTileMod,
                            ubOTmpBufId, rescaleCount == 0, false, mm2ToReFlag);
                        rescaleCount++;
                    } else if (gatheredKvSTileIdxDe == kvSLoopNum) { // div & output only
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                        epilogueRescaleO(gO[gmOffsetO], (GemmCoord){rowNum, embed_, kvSTileSizeAct}, curTileMod,
                            ubOTmpBufId, rescaleCount == 0, true, mm2ToReFlag);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
                    }
#endif
                }
            }
        }
        // release reverse sync flags
        ReleaseSyncFlags<4, 4, 4>();
    }

    __aicore__ inline void FetchBaseShapeInfo(__gm__ EagleQuantBlockSparseAttentionTilingData *bsaTilingData) {
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
        tableShapeY = bsaTilingData->maxKvBlockNum;
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

    __aicore__ inline void CalcOnChipBufTileInfo(__gm__ EagleQuantBlockSparseAttentionTilingData *bsaTilingData) {
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
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(
            mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_, qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(
            mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_, pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        // Keep Q/K and their scales in the lower 256 KiB. P/V starts from
        // the upper half of L1 so the two matmul phases never alias in L1.
        mm2L1AddrStart_ = MM2_L1_ADDR_START;
    }

    __aicore__ inline void InitCrossCoreDstBuf(AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
        AscendC::LocalTensor<float8_e8m0_t> (&l1PSTensor)[MAX_CROSS_CORE_BUF_STAGES],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_BUF_STAGES],
        AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_OTMP_BUF_STAGES]) {
        uint32_t pScaleBufSize = CeilDiv(mm2L1TileKLeft_, MX_SCALE_GROUP_NUM) * mm2L1TileM_;
        uint32_t pDataSlotSize = CeilDiv<2>(mm2L1TileM_ * mm2L1TileKLeft_);
        uint32_t pDataBufSize = pDataSlotSize * pL1BufNum_;
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(mm2L1AddrStart_ + pDataSlotSize * i);
            l1PSTensor[i] = resource.l1Buf.template GetBufferByByte<float8_e8m0_t>(
                mm2L1AddrStart_ + pDataBufSize + pScaleBufSize * i);
        }
        uint32_t rowNumPerSubCore = EpilogueOnlineSoftmax::SM_ROW_MAX_ELEM_NUM;
        uint32_t colNumPerSubCore = EpilogueOnlineSoftmax::SM_COL_MAX_ELEM_NUM;
        uint32_t pvOutputCol =
            EpilogueRescaleO::RESCALE_COL_MAX_ELEM_NUM + BlockMmadPV::V_APPEND_COLS - BlockMmadPV::V_COPY_OUT_SKIP_COLS;
        for (uint32_t i = 0; i < UB_S_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * i);
        }
        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_BUF_STAGES +
                rowNumPerSubCore * pvOutputCol * sizeof(ElementOTmp) * i);
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE> __aicore__ inline void InitSyncFlags() {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query/key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        // QK L0A/L0B pairs
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // PV L0A/L0B pairs
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);

        // cross core sync
        if constexpr (SM_MM2_MODE == 4U) {
            constexpr uint32_t smToMm2FlagId0 = UB_S_BUF_STAGES; // 3
            constexpr uint32_t smToMm2FlagId1 = UB_S_BUF_STAGES + 1; // 4
            constexpr uint32_t smToMm2FlagId2 = UB_S_BUF_STAGES + 2; // 5
            constexpr uint32_t smToMm2FlagId3 = UB_S_BUF_STAGES + 3; // 6
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId0); // 3
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId0 + V0_V1_FLAG_ID_OFFSET); // 19
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId1); // 4
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId1 + V0_V1_FLAG_ID_OFFSET); // 20
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId2); // 5
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId2 + V0_V1_FLAG_ID_OFFSET); // 21
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId3); // 6
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(smToMm2FlagId3 + V0_V1_FLAG_ID_OFFSET); // 22
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
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
            constexpr uint32_t mm1ToSmFlagId0 = 0; // 0
            constexpr uint32_t mm1ToSmFlagId1 = 1; // 1
            constexpr uint32_t mm1ToSmFlagId2 = 2; // 2
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(mm1ToSmFlagId0); // 0
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(mm1ToSmFlagId1); // 1
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(mm1ToSmFlagId2); // 2
        }
        if constexpr (MM2_RE_MODE == 4U) {
            constexpr uint32_t mm2ToReFlagId0 = UB_S_BUF_STAGES + MAX_CROSS_CORE_BUF_STAGES; // 7
            constexpr uint32_t mm2ToReFlagId1 = UB_S_BUF_STAGES + MAX_CROSS_CORE_BUF_STAGES + 1; // 8
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(mm2ToReFlagId0); // 7
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(mm2ToReFlagId1); // 8
        }
#endif
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void ReleaseSyncFlags() {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);

        if constexpr (MM1_SM_MODE == 4U) {
            constexpr uint32_t mm1ToSmFlagId0 = 0; // 0
            constexpr uint32_t mm1ToSmFlagId1 = 1; // 1
            constexpr uint32_t mm1ToSmFlagId2 = 2; // 2
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(mm1ToSmFlagId0); // 0
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(mm1ToSmFlagId1); // 1
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(mm1ToSmFlagId2); // 2
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(mm1ToSmFlagId0 + V0_V1_FLAG_ID_OFFSET); // 16
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(mm1ToSmFlagId1 + V0_V1_FLAG_ID_OFFSET); // 17
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(mm1ToSmFlagId2 + V0_V1_FLAG_ID_OFFSET); // 18
        }
        if constexpr (MM2_RE_MODE == 4U) {
            constexpr uint32_t mm2ToReFlagId0 = UB_S_BUF_STAGES + MAX_CROSS_CORE_BUF_STAGES; // 7
            constexpr uint32_t mm2ToReFlagId1 = UB_S_BUF_STAGES + MAX_CROSS_CORE_BUF_STAGES + 1; // 8
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(mm2ToReFlagId0); // 7
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(mm2ToReFlagId0 + V0_V1_FLAG_ID_OFFSET); // 23
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(mm2ToReFlagId1); // 8
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(mm2ToReFlagId1 + V0_V1_FLAG_ID_OFFSET); // 24
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_MM2_MODE == 4U) {
            constexpr uint32_t smToMm2FlagId0 = UB_S_BUF_STAGES; // 3
            constexpr uint32_t smToMm2FlagId1 = UB_S_BUF_STAGES + 1; // 4
            constexpr uint32_t smToMm2FlagId2 = UB_S_BUF_STAGES + 2; // 5
            constexpr uint32_t smToMm2FlagId3 = UB_S_BUF_STAGES + 3; // 6
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(smToMm2FlagId0); // 3
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(smToMm2FlagId1); // 4
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(smToMm2FlagId2); // 5
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(smToMm2FlagId3); // 6
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
    uint32_t tableShapeY;
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
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
};

}

#endif
