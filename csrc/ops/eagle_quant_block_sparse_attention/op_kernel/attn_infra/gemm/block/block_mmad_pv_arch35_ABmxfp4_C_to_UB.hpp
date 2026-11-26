/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief MXFP4 PV computes P * V.
 * P is produced in L1 and consumed as the A matrix. V is sparsely gathered
 * from GM to L1 and consumed as the B matrix. The result is row-major O in UB.
 */
#ifndef GEMM_BLOCK_PV_ARCH35_ABMXFP4_C2UB_HPP
#define GEMM_BLOCK_PV_ARCH35_ABMXFP4_C2UB_HPP

#include "c_api/asc_simd.h"
#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/arch/cross_core_sync.hpp"
#include "../../../attn_infra/coord.hpp"
#include "../../../attn_infra/gemm/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/helper.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_arch35_opt.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_mmad.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

template <class L1TileShape_, class L0TileShape_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaPVQMode2, L1TileShape_, L0TileShape_, float4_e2m1x2_t, float4_e2m1x2_t, ElementC_,
    ElementBias_, TileCopy_, TileMmad_> {
  public:
    using DispatchPolicy = MmadAtlasA5BsaPVQMode2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = float4_e2m1x2_t;
    using ElementB = float4_e2m1x2_t;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagL1AS = typename TileCopy::LayoutTagL1MxScaleA;
    using LayoutTagL1BS = typename TileCopy::LayoutTagL1MxScaleB;

    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static_assert(L0C_STAGES == 1, "MXFP4 PV requires exactly one L0C stage");
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t L0A_BUF_SIZE = 128U * 256U / 2U;
    static constexpr uint32_t L0B_BUF_SIZE = 256U * 144U / 2U;
    static constexpr uint32_t L0A_BUF_OFFSET = 2U * (256U * 128U / 2U);
    static constexpr uint32_t L0B_BUF_OFFSET = 2U * (128U * 128U / 2U);
    static_assert(L0A_BUF_OFFSET + L0A_BUF_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE,
        "MXFP4 QK/PV L0A buffers exceed hardware capacity");
    static_assert(L0B_BUF_OFFSET + L0B_BUF_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE,
        "MXFP4 QK/PV L0B buffers exceed hardware capacity");
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_BUF_SIZE / L0C_STAGES;
    static constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
    static constexpr uint32_t MAX_L1_STAGES = 4;
    static constexpr uint32_t L1B_EVENT_ID_OFFSET = 4;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;
    static constexpr uint32_t L0C_BUF_ID = 0;
    static constexpr uint32_t L0C_EVENT_ID = 2;
    // The last 16 columns are initialized in L0B once. Regular V loads only
    // overwrite the first 128 columns.
    static constexpr uint32_t V_SUM_COLS = 8;
    static constexpr uint32_t V_PADDING_COLS = 8;
    static constexpr uint32_t V_APPEND_COLS = V_SUM_COLS + V_PADDING_COLS;
    static constexpr uint32_t V_COPY_OUT_SKIP_COLS = V_PADDING_COLS;
    static constexpr uint16_t MXFP4_ONE = 0x2222;
    static constexpr uint16_t MX_SCALE_2_POW_NEG_50 = 0x4d4d;
    static constexpr uint16_t MX_SCALE_ONE = 0x7f7f;
    struct L1ToL0LoadParams {
        AscendC::LoadData2DParamsV2 pData;
        AscendC::LoadData2DMxParams pScale;
        AscendC::LoadData2DParamsV2 vData;
        AscendC::LoadData2DMxParams vScale;
    };

    __aicore__ inline BlockMmadTla(
        Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart, Mm2L1TileHelper &mm2L1TileHelper) {
        l1ATileM = mm2L1TileHelper.mm2L1TileM;
        l1BTileN = mm2L1TileHelper.mm2L1TileN;
        l1ATileK = mm2L1TileHelper.mm2L1TileKLeft;
        l1BTileK = mm2L1TileHelper.mm2L1TileKRight;
        l1ABufNum = mm2L1TileHelper.pL1BufNum;
        l1BBufNum = mm2L1TileHelper.vL1BufNum;

        const uint32_t pDataSize = CeilDiv<2>(mm2L1TileHelper.mm2L1TileM * mm2L1TileHelper.mm2L1TileKLeft);
        const uint32_t pScaleSize =
            CeilDiv<MX_SCALE_GROUP_NUM>(mm2L1TileHelper.mm2L1TileKLeft) * mm2L1TileHelper.mm2L1TileM;
        const uint32_t pSlotsSize = (pDataSize + pScaleSize) * mm2L1TileHelper.pL1BufNum;
        const uint32_t vStorageN = l1BTileN + V_APPEND_COLS;
        const uint32_t vDataSize = CeilDiv<2>(l1BTileK * vStorageN);
        const uint32_t vScaleSize = CeilDiv<MX_SCALE_GROUP_NUM>(l1BTileK) * (l1BTileN + V_APPEND_COLS);
        const uint32_t vDataOffset = l1BufAddrStart + pSlotsSize;
        const uint32_t vScaleOffset = vDataOffset + vDataSize * l1BBufNum;
        for (uint32_t i = 0; i < l1BBufNum; ++i) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(vDataOffset + vDataSize * i);
            l1BSTensor[i] = resource.l1Buf.template GetBufferByByte<float8_e8m0_t>(vScaleOffset + vScaleSize * i);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_BUF_OFFSET + L0A_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_BUF_OFFSET + L0B_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_HALF_BUF_SIZE + L0C_BUF_SIZE * i);
        }
    }

    __aicore__ inline void InitVAppendColumns(uint32_t l0EventIdStart) {
        static_assert(L0B_STAGES == 2, "MXFP4 PV append initialization requires two L0B stages");
        const uint32_t outputCols = l1BTileN + V_APPEND_COLS;
        const uint32_t kvFractals64 = CeilDiv<64>(l1BTileK);
        const uint32_t outputFractals16 = CeilDiv<16>(outputCols);
        const uint32_t l1BEventId = L1B_EVENT_ID_OFFSET;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        auto l1BFillTensor = l1BTensor[0].template ReinterpretCast<uint16_t>();
        AscendC::InitConstValueParams<uint16_t> vFillParams;
        vFillParams.repeatTimes = 1;
        vFillParams.blockNum = l1BTileK * outputCols / 64;
        vFillParams.dstGap = 0;
        vFillParams.initValue = MXFP4_ONE;
        AscendC::Fill(l1BFillTensor, vFillParams);

        AscendC::PipeBarrier<PIPE_MTE2>();
        auto l1BSFillTensor = l1BSTensor[0].template ReinterpretCast<uint16_t>();
        AscendC::InitConstValueParams<uint16_t> vPaddingScaleFillParams;
        vPaddingScaleFillParams.repeatTimes = 1;
        vPaddingScaleFillParams.blockNum = kvFractals64 * outputCols / 16;
        vPaddingScaleFillParams.dstGap = 0;
        vPaddingScaleFillParams.initValue = MX_SCALE_ONE;
        AscendC::Fill(l1BSFillTensor, vPaddingScaleFillParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);

        AscendC::LoadData2DParamsV2 vInitData(
            0, 0, outputFractals16, kvFractals64, outputFractals16, outputFractals16, false, 0);
        AscendC::LoadData2DMxParams vInitScale(0, 0, outputFractals16, kvFractals64, kvFractals64, kvFractals64);
        for (uint32_t l0BufId = 0; l0BufId < L0B_STAGES; ++l0BufId) {
            const uint32_t l0EventId = l0EventIdStart + l0BufId;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0EventId);
            AscendC::LoadData(l0BTensor[l0BufId], l1BTensor[0], l1BSTensor[0], vInitData, vInitScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0EventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0EventId);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0EventId);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
    }

    __aicore__ inline ~BlockMmadTla() {}

    __aicore__ inline void InitL1ToL0LoadParams(
        uint32_t normalKvTile, uint32_t tailKvTile, uint32_t queryRows, uint32_t outputCols) {
        InitL1ToL0LoadParamsForTile(normalLoadParams, normalKvTile, queryRows, outputCols);
        InitL1ToL0LoadParamsForTile(tailLoadParams, tailKvTile, queryRows, outputCols);
    }

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <class TensorB, class TensorBS, class TensorL1B, class TensorL1BS>
    __aicore__ inline void SparseVBaseTileL1FullLoad(TensorB &gBTensor, TensorBS &gBSTensor, TensorL1B &l1BTensorTla,
        TensorL1BS &l1BSTensorTla, AscendC::GlobalTensor<int32_t> gSparseBlockIdx, uint32_t gatheredKvSTileIdx,
        uint32_t kvSeqlen, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t curBaseTileSize, uint32_t embed) {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        using CopyGmToL1BS = typename TileCopy_::template CopyGmToL1MxScaleB<TensorBS>;
        CopyGmToL1B copyGmToL1B;
        CopyGmToL1BS copyGmToL1BS;
        const uint32_t gatheredYBlockIdx = gatheredKvSTileIdx * kvSBaseTile / blockShapeY;
        const uint32_t blockNum = CeilDiv(curBaseTileSize, blockShapeY);
        uint32_t dealtLenAccum = 0;
        const uint32_t block0Idx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t block1Idx = block0Idx;
        if (likely(blockNum == 2U)) {
            block1Idx = gSparseBlockIdx.GetValue(gatheredYBlockIdx + 1U);
        }
        constexpr uint32_t EMBED = 128;
        constexpr uint32_t FP4_PER_FP32 = 8;
        constexpr uint32_t FP32_COLS = EMBED / FP4_PER_FP32;
        const uint64_t block0SrcDataOffset = static_cast<uint64_t>(block0Idx) * blockShapeY * EMBED;
        const uint64_t block1SrcDataOffset = static_cast<uint64_t>(block1Idx) * blockShapeY * EMBED;
        const uint64_t srcDataMatrixStride = (block1SrcDataOffset - block0SrcDataOffset) / FP4_PER_FP32;
        const bool useTwoBlockCopy =
            curBaseTileSize == 2U * blockShapeY && srcDataMatrixStride > 0U && srcDataMatrixStride <= 65535U;

        if (likely(useTwoBlockCopy)) {
            const uint64_t block0SrcScaleOffset =
                static_cast<uint64_t>(block0Idx) * blockShapeY / MX_SCALE_GROUP_NUM * EMBED;
            const uint64_t block1SrcScaleOffset =
                static_cast<uint64_t>(block1Idx) * blockShapeY / MX_SCALE_GROUP_NUM * EMBED;
            const uint64_t srcScaleMatrixStride = (block1SrcScaleOffset - block0SrcScaleOffset) / 2;
            const uint32_t alignedBaseTileSize = (curBaseTileSize + 63U) & ~63U;

            AscendC::Nd2NzParams dataParams;
            dataParams.ndNum = 2;
            dataParams.nValue = blockShapeY;
            dataParams.dValue = FP32_COLS;
            dataParams.srcNdMatrixStride = static_cast<uint32_t>(srcDataMatrixStride);
            dataParams.srcDValue = FP32_COLS;
            dataParams.dstNzC0Stride = alignedBaseTileSize;
            dataParams.dstNzNStride = 1;
            dataParams.dstNzMatrixStride = blockShapeY * FP32_COLS / 2;

            AscendC::Nd2NzParams scaleParams;
            scaleParams.ndNum = 2;
            scaleParams.nValue = blockShapeY / MX_SCALE_GROUP_NUM / 2;
            scaleParams.dValue = EMBED;
            scaleParams.srcNdMatrixStride = static_cast<uint32_t>(srcScaleMatrixStride);
            scaleParams.srcDValue = EMBED;
            scaleParams.dstNzC0Stride = alignedBaseTileSize / 64;
            scaleParams.dstNzNStride = 1;
            scaleParams.dstNzMatrixStride = blockShapeY / 4;

            auto dstData = l1BTensorTla.data()[0].template ReinterpretCast<float>();
            auto srcData = gBTensor.data()[block0SrcDataOffset].template ReinterpretCast<float>();
            AscendC::DataCopy(dstData, srcData, dataParams);
            auto dstScale = l1BSTensorTla.data()[0].template ReinterpretCast<half>();
            auto srcScale = gBSTensor.data()[block0SrcScaleOffset].template ReinterpretCast<half>();
            AscendC::DataCopy(dstScale, srcScale, scaleParams);
            return;
        }

        if (blockNum == 2U) {
            const uint32_t oriStartOffset = block0Idx * blockShapeY;
            auto l1BTensorTlaTile =
                GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(blockShapeY, embed));
            auto gBTensorTlaTile =
                GetTile(gBTensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(blockShapeY, embed));
            copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
            auto l1BSTensorTlaTile = GetTile(l1BSTensorTla, tla::MakeCoord(dealtLenAccum / MX_SCALE_GROUP_NUM, 0),
                tla::MakeShape(blockShapeY / MX_SCALE_GROUP_NUM, embed));
            auto gBSTensorTlaTile = GetTile(gBSTensor, tla::MakeCoord(oriStartOffset / MX_SCALE_GROUP_NUM, 0),
                tla::MakeShape(blockShapeY / MX_SCALE_GROUP_NUM, embed));
            copyGmToL1BS(l1BSTensorTlaTile, gBSTensorTlaTile);
            dealtLenAccum += blockShapeY;
        }

        const uint32_t lastBlockSize = curBaseTileSize - dealtLenAccum;
        const uint32_t oriStartOffset = block1Idx * blockShapeY;
        auto l1BTensorTlaTile =
            GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(lastBlockSize, embed));
        auto gBTensorTlaTile =
            GetTile(gBTensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(lastBlockSize, embed));
        copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
        auto l1BSTensorTlaTile = GetTile(l1BSTensorTla, tla::MakeCoord(dealtLenAccum / MX_SCALE_GROUP_NUM, 0),
            tla::MakeShape(CeilDiv(lastBlockSize, MX_SCALE_GROUP_NUM), embed));
        auto gBSTensorTlaTile = GetTile(gBSTensor, tla::MakeCoord(oriStartOffset / MX_SCALE_GROUP_NUM, 0),
            tla::MakeShape(CeilDiv(lastBlockSize, MX_SCALE_GROUP_NUM), embed));
        copyGmToL1BS(l1BSTensorTlaTile, gBSTensorTlaTile);
    }

    template <class TensorB, class TensorBS>
    __aicore__ inline void IssueVTileLoad(TensorB &gBTensor, TensorBS &gBSTensor,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape, uint32_t gatheredKvSTileIdx,
        uint32_t kvSeqlen, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd) {
        const uint32_t embed = actualOriShape[1];
        const uint32_t curBaseTileSize = actualOriShape[2];
        const uint32_t curBaseTileSizeAligned = Arch35MmadOpt::AlignUpPow2<128>(curBaseTileSize);
        const uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        const uint32_t l1BEventId = l1BBufId + L1B_EVENT_ID_OFFSET;
        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(curBaseTileSizeAligned, embed);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
        auto l1BSLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, LayoutTagL1BS, true>(
            CeilDiv(curBaseTileSizeAligned, MX_SCALE_GROUP_NUM), embed);
        auto l1BSTensorTla = tla::MakeTensor(l1BSTensor[l1BBufId], l1BSLayoutTla, Arch::PositionL1{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        SparseVBaseTileL1FullLoad(gBTensor, gBSTensor, l1BTensorTla, l1BSTensorTla, gSparseBlockIdx, gatheredKvSTileIdx,
            kvSeqlen, kvSBaseTile, blockShapeY, yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embed);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
    }

    template <class TensorA, class TensorAS, class TensorC>
    __aicore__ inline void ComputeLoadedVTile(TensorA &l1ATensorTla, TensorAS &l1ASTensorTla, TensorC &ubCTensor,
        GemmCoord actualOriShape, uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile,
        uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd, bool InitValue, bool needCopyOut,
        Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag, uint32_t kvSLoopNum, uint32_t &l0BufId,
        uint32_t &l0EventId) {
        static_assert(L0A_STAGES == 2 && L0B_STAGES == 2, "QMode2 PV requires two paired L0A/L0B stages");
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
        const uint32_t rowNum = actualOriShape[0];
        const uint32_t embed = actualOriShape[1];
        const uint32_t outputCols = embed + V_APPEND_COLS;
        const uint32_t curBaseTileSize = actualOriShape[2];
        const uint32_t curBaseTileSizeAligned = Arch35MmadOpt::AlignUpPow2<128>(curBaseTileSize);
        const bool isFinalDrain = gatheredKvSTileIdx == kvSLoopNum;
        const bool isLastRealTile = gatheredKvSTileIdx + 1U == kvSLoopNum;

        if (gatheredKvSTileIdx > 0 && (needCopyOut || isFinalDrain)) {
            WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            if (!isFinalDrain) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0C_EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0C_EVENT_ID);
            }
            auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, outputCols);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor[L0C_BUF_ID], l0CLayoutTla, Arch::PositionL0C{});
            const uint32_t copyOutCols = outputCols - V_COPY_OUT_SKIP_COLS;
            auto ubCTensorTlaTile = GetTile(ubCTensor, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, copyOutCols));
            copyL0CToDst(ubCTensorTlaTile, l0CTensorTla, isFinalDrain ? 3U : 0U);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);
            SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            l0CNeedsSync = true;
        }

        if (unlikely(isFinalDrain)) {
            // The final iteration only drains the previous PV result.
            return;
        }

        const uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        const uint32_t l1BEventId = l1BBufId + L1B_EVENT_ID_OFFSET;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);

        const auto &loadParams = (curBaseTileSizeAligned == l1BTileK) ? normalLoadParams : tailLoadParams;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0EventId);
        AscendC::LoadData(
            l0BTensor[l0BufId], l1BTensor[l1BBufId], l1BSTensor[l1BBufId], loadParams.vData, loadParams.vScale);
        if (curBaseTileSizeAligned != l1BTileK) {
            const uint64_t appendMxOffset =
                static_cast<uint64_t>(loadParams.vScale.xStep) * loadParams.vScale.dstStride * 32U;
            const uint64_t appendMxDst =
                reinterpret_cast<uint64_t>(l0BTensor[l0BufId].GetPhyAddr()) / 16U + appendMxOffset;
            asc_copy_l12l0b_mx(appendMxDst, reinterpret_cast<__cbuf__ fp8_e8m0_t *>(l1BSTensor[0].GetPhyAddr()),
                loadParams.vScale.xStep, 0U, 1U, loadParams.vScale.yStep, normalLoadParams.vScale.srcStride,
                loadParams.vScale.dstStride);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        AscendC::LoadData(
            l0ATensor[l0BufId], l1ATensorTla.data(), l1ASTensorTla.data(), loadParams.pData, loadParams.pScale);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0EventId);
        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0EventId);

        if (l0CNeedsSync) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);
            l0CNeedsSync = false;
        }
        const uint8_t pvMmadUnitFlag = isLastRealTile ? 3U : 0U;
        asc_mmad_mx(reinterpret_cast<__cc__ float *>(l0CTensor[L0C_BUF_ID].GetPhyAddr()),
            reinterpret_cast<__ca__ fp4x2_e2m1_t *>(l0ATensor[l0BufId].GetPhyAddr()),
            reinterpret_cast<__cb__ fp4x2_e2m1_t *>(l0BTensor[l0BufId].GetPhyAddr()),
            Arch35MmadOpt::AlignUpPow2<16>(rowNum), curBaseTileSizeAligned, outputCols, pvMmadUnitFlag, true, false,
            InitValue);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0EventId);
        l0BufId ^= 1U;
        l0EventId ^= 1U;
    }

  protected:
    __aicore__ inline static void InitL1ToL0LoadParamsForTile(
        L1ToL0LoadParams &params, uint32_t kvTile, uint32_t queryRows, uint32_t outputCols) {
        const uint32_t kvFractals16 = (kvTile + 15) / 16;
        const uint32_t kvFractals64 = (kvTile + 63) / 64;
        const uint32_t queryFractals16 = (queryRows + 15) / 16;
        const uint32_t queryFractals64 = (queryRows + 63) / 64;
        const uint32_t outputFractals16 = (outputCols + 15) / 16;
        const uint32_t validOutputCols = outputCols - V_APPEND_COLS;
        const uint32_t validOutputFractals16 = (validOutputCols + 15) / 16;
        const uint32_t validOutputFractals64 = (validOutputCols + 63) / 64;

        params.pData =
            AscendC::LoadData2DParamsV2(0, 0, kvFractals16, queryFractals64, kvFractals16, queryFractals16, true, 0);
        params.pScale = AscendC::LoadData2DMxParams(0, 0, queryFractals16, kvFractals64, kvFractals64, kvFractals64);
        params.vData = AscendC::LoadData2DParamsV2(
            0, 0, kvFractals16, validOutputFractals64, kvFractals16, outputFractals16, true, 0);
        params.vScale =
            AscendC::LoadData2DMxParams(0, 0, validOutputFractals16, kvFractals64, kvFractals64, kvFractals64);
    }

    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<float8_e8m0_t> l1BSTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0A_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0C_STAGES];
    L1ToL0LoadParams normalLoadParams;
    L1ToL0LoadParams tailLoadParams;
    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;
    // InitSyncFlags() gives M one initial FIX_M token. A PV accumulation
    // chain consumes it once, keeps ownership across all real tiles, and the
    // FINAL Fixpipe returns it for the next task (or ReleaseSyncFlags()).
    bool l0CNeedsSync = true;
};

} // namespace NpuArch::Gemm::Block
#endif
