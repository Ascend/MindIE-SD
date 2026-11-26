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
 * @brief matmul implementation for single q&k^t base tile
 * This implementation is designed for the following senario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_QK_ARCH35_ABMXFP4_C2UB_HPP
#define GEMM_BLOCK_QK_ARCH35_ABMXFP4_C2UB_HPP

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

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

#ifndef GEMM_BLOCK_QK_ARCH35_COMMON_HELPER_HPP
#define GEMM_BLOCK_QK_ARCH35_COMMON_HELPER_HPP
template <class T> struct HasParams {
    template <class U> static char test(typename U::Params *);
    template <class U> static int test(...);
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(char);
};

struct Mm1L1TileHelper {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;

    __aicore__ inline Mm1L1TileHelper() {}

    __aicore__ inline Mm1L1TileHelper(uint32_t m, uint32_t n, uint32_t kl, uint32_t kr, uint32_t pbn, uint32_t vbn)
        : mm1L1TileM(m), mm1L1TileN(n), mm1L1TileKLeft(kl), mm1L1TileKRight(kr), qL1BufNum(pbn), kL1BufNum(vbn) {}
};
#endif

template <class L1TileShape_, class L0TileShape_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaQKQMode2, L1TileShape_, L0TileShape_, float4_e2m1x2_t, float4_e2m1x2_t, ElementC_,
    ElementBias_, TileCopy_, TileMmad_> {
  public:
    using DispatchPolicy = MmadAtlasA5BsaQKQMode2;
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
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static_assert(tla::get<1>(L1TileShape_{}) <= 256U, "QMode2 QK L1 N tile supports at most two sparse blocks");
    static constexpr uint32_t L0A_BUF_SIZE = 256U * 128U / 2U;
    static constexpr uint32_t L0B_BUF_SIZE = 128U * 128U / 2U;
    static constexpr uint32_t L0A_BUF_OFFSET = 0U;
    static constexpr uint32_t L0B_BUF_OFFSET = 0U;
    static_assert(L0A_BUF_OFFSET + L0A_BUF_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE,
        "MXFP4 QK L0A buffers exceed hardware capacity");
    static_assert(L0B_BUF_OFFSET + L0B_BUF_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE,
        "MXFP4 QK L0B buffers exceed hardware capacity");
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_HALF_BUF_SIZE / L0C_STAGES;
    static constexpr uint32_t MMAD_M = 64;
    static constexpr uint32_t MX_SCALE_GROUP_NUM = 32;

    struct L1ToL0LoadParams {
        AscendC::LoadData2DParamsV2 kData;
        AscendC::LoadData2DMxParams kScale;
        AscendC::LoadData2DParamsV2 qData;
        AscendC::LoadData2DMxParams qScale;
    };

    static constexpr uint32_t MAX_L1_STAGES = 4;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16; // 核间同步mode4，AIC侧需要两个flagId分别对应两个AIV

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, Mm1L1TileHelper &mm1L1TileHelper) {
        l1ATileN = mm1L1TileHelper.mm1L1TileN;
        l1BTileN = mm1L1TileHelper.mm1L1TileM;
        l1ATileK = mm1L1TileHelper.mm1L1TileKRight;
        l1BTileK = mm1L1TileHelper.mm1L1TileKLeft;
        l1ABufNum = mm1L1TileHelper.kL1BufNum;
        l1BBufNum = mm1L1TileHelper.qL1BufNum;

        const uint32_t l1ADataBufSize = CeilDiv<2>(l1ATileN * l1ATileK);
        const uint32_t l1BDataBufSize = CeilDiv<2>(l1BTileK * l1BTileN);
        const uint32_t l1AScaleBufSize = l1ATileN * CeilDiv<MX_SCALE_GROUP_NUM>(l1ATileK);
        const uint32_t l1BScaleBufSize = l1BTileN * CeilDiv<MX_SCALE_GROUP_NUM>(l1BTileK);
        const uint32_t l1BDataOffset = l1ADataBufSize * l1ABufNum;
        const uint32_t l1AScaleOffset = l1BDataOffset + l1BDataBufSize * l1BBufNum;
        const uint32_t l1BScaleOffset = l1AScaleOffset + l1AScaleBufSize * l1ABufNum;

        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1ADataBufSize * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BDataOffset + l1BDataBufSize * i);
        }
        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ASTensor[i] =
                resource.l1Buf.template GetBufferByByte<float8_e8m0_t>(l1AScaleOffset + l1AScaleBufSize * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BSTensor[i] =
                resource.l1Buf.template GetBufferByByte<float8_e8m0_t>(l1BScaleOffset + l1BScaleBufSize * i);
        }
        for (uint32_t i = 0; i < L0A_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_BUF_OFFSET + L0A_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L0B_STAGES; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_BUF_OFFSET + L0B_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

    __aicore__ inline void InitL1ToL0LoadParams(
        uint32_t normalKvTile, uint32_t tailKvTile, uint32_t queryRows, uint32_t embed) {
        InitL1ToL0LoadParamsForTile(normalLoadParams, normalKvTile, queryRows, embed);
        InitL1ToL0LoadParamsForTile(tailLoadParams, tailKvTile, queryRows, embed);
    }

    template <class TensorB, class TensorBS>
    __aicore__ inline void loadQGM(TensorB &gBTensor, TensorBS &gBSTensor, GemmCoord actualOriShape) {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        using CopyGmToL1BS = typename TileCopy_::template CopyGmToL1MxScaleB<TensorBS>;
        CopyGmToL1B copyGmToL1B;
        CopyGmToL1BS copyGmToL1BS;
        const uint32_t rowNum = actualOriShape[0];
        const uint32_t embed = actualOriShape[1];
        const uint32_t qBlock0Rows = rowNum > MMAD_M ? MMAD_M : rowNum;
        const uint32_t qBlock1Rows = rowNum - qBlock0Rows;
        const uint32_t scaleCols = CeilDiv(embed, MX_SCALE_GROUP_NUM);
        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(embed * 2U, MMAD_M);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[0], l1BLayoutTla, Arch::PositionL1{});
        auto l1BSLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, LayoutTagL1BS, true>(scaleCols * 2U, MMAD_M);
        auto l1BSTensorTla = tla::MakeTensor(l1BSTensor[0], l1BSLayoutTla, Arch::PositionL1{});
        auto l1BBlock0TensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(embed, qBlock0Rows));
        auto l1BSBlock0TensorTlaTile =
            GetTile(l1BSTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(scaleCols, qBlock0Rows));
        auto gBBlock0TensorTlaTile = GetTile(gBTensor, tla::MakeCoord(0, 0), tla::MakeShape(embed, qBlock0Rows));
        auto gBSBlock0TensorTlaTile = GetTile(gBSTensor, tla::MakeCoord(0, 0), tla::MakeShape(scaleCols, qBlock0Rows));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1B(l1BBlock0TensorTlaTile, gBBlock0TensorTlaTile);
        copyGmToL1BS(l1BSBlock0TensorTlaTile, gBSBlock0TensorTlaTile);
        if (qBlock1Rows != 0U) {
            auto l1BBlock1TensorTlaTile =
                GetTile(l1BTensorTla, tla::MakeCoord(embed, 0), tla::MakeShape(embed, qBlock1Rows));
            auto l1BSBlock1TensorTlaTile =
                GetTile(l1BSTensorTla, tla::MakeCoord(scaleCols, 0), tla::MakeShape(scaleCols, qBlock1Rows));
            auto gBBlock1TensorTlaTile =
                GetTile(gBTensor, tla::MakeCoord(0, MMAD_M), tla::MakeShape(embed, qBlock1Rows));
            auto gBSBlock1TensorTlaTile =
                GetTile(gBSTensor, tla::MakeCoord(0, MMAD_M), tla::MakeShape(scaleCols, qBlock1Rows));
            copyGmToL1B(l1BBlock1TensorTlaTile, gBBlock1TensorTlaTile);
            copyGmToL1BS(l1BSBlock1TensorTlaTile, gBSBlock1TensorTlaTile);
        }
    }

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <class TensorA, class TensorAS, class TensorL1A, class TensorL1AS>
    __aicore__ inline void SparseKL1TileNLoad(TensorA &gATensor, TensorAS &gASTensor, TensorL1A &l1ATensorTla,
        TensorL1AS &l1ASTensorTla, AscendC::GlobalTensor<int32_t> gSparseBlockIdx, uint32_t gatheredKvSTileIdx,
        uint32_t kvSeqlen, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t l1BTileNAct, uint32_t embed, uint32_t kvSBaseTileInnerOffset) {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;
        using CopyGmToL1AS = typename TileCopy_::template CopyGmToL1MxScaleA<TensorAS>;
        CopyGmToL1AS copyGmToL1AS;
        uint32_t gatheredStartOffset = gatheredKvSTileIdx * kvSBaseTile + kvSBaseTileInnerOffset;
        const uint32_t gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
        const uint32_t blockNum = CeilDiv(l1BTileNAct, blockShapeY);
        uint32_t dealtLenAccum = 0;
        const uint32_t block0Idx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t block1Idx = block0Idx;
        if (likely(blockNum == 2U)) {
            block1Idx = gSparseBlockIdx.GetValue(gatheredYBlockIdx + 1U);
        }
        constexpr uint32_t EMBED = 128;
        constexpr uint32_t FP4_PER_FP32 = 8;
        constexpr uint32_t FP32_COLS = EMBED / FP4_PER_FP32;
        constexpr uint32_t SCALE_COLS = EMBED / MX_SCALE_GROUP_NUM;
        constexpr uint32_t SCALE_HALF_COLS = SCALE_COLS / 2;
        const uint64_t block0SrcDataOffset = static_cast<uint64_t>(block0Idx) * blockShapeY * EMBED;
        const uint64_t block1SrcDataOffset = static_cast<uint64_t>(block1Idx) * blockShapeY * EMBED;
        const uint64_t srcDataMatrixStride = (block1SrcDataOffset - block0SrcDataOffset) / FP4_PER_FP32;
        const bool useTwoBlockCopy =
            l1BTileNAct == 2U * blockShapeY && srcDataMatrixStride > 0U && srcDataMatrixStride <= 65535U;

        if (likely(useTwoBlockCopy)) {
            const uint64_t block0SrcScaleOffset = static_cast<uint64_t>(block0Idx) * blockShapeY * SCALE_COLS;
            const uint64_t block1SrcScaleOffset = static_cast<uint64_t>(block1Idx) * blockShapeY * SCALE_COLS;
            const uint64_t srcScaleMatrixStride = (block1SrcScaleOffset - block0SrcScaleOffset) / 2;
            AscendC::Dn2NzParams scaleParams;
            scaleParams.dnNum = 2;
            scaleParams.nValue = SCALE_HALF_COLS;
            scaleParams.dValue = blockShapeY;
            scaleParams.srcDnMatrixStride = srcScaleMatrixStride;
            scaleParams.srcDValue = SCALE_HALF_COLS;
            scaleParams.dstNzC0Stride = SCALE_HALF_COLS;
            scaleParams.dstNzNStride = 1;
            scaleParams.dstNzMatrixStride = blockShapeY * SCALE_HALF_COLS;

            AscendC::Nd2NzParams dataParams;
            dataParams.ndNum = 2;
            dataParams.nValue = blockShapeY;
            dataParams.dValue = FP32_COLS;
            dataParams.srcNdMatrixStride = static_cast<uint32_t>(srcDataMatrixStride);
            dataParams.srcDValue = FP32_COLS;
            dataParams.dstNzC0Stride = (l1BTileNAct + 15U) & ~15U;
            dataParams.dstNzNStride = 1;
            dataParams.dstNzMatrixStride = blockShapeY * FP4_PER_FP32;

            auto dstScale = l1ASTensorTla.data()[0].template ReinterpretCast<half>();
            auto srcScale = gASTensor.data()[block0SrcScaleOffset].template ReinterpretCast<half>();
            AscendC::DataCopy(dstScale, srcScale, scaleParams);
            auto dstData = l1ATensorTla.data()[0].template ReinterpretCast<float>();
            auto srcData = gATensor.data()[block0SrcDataOffset].template ReinterpretCast<float>();
            AscendC::DataCopy(dstData, srcData, dataParams);
            return;
        }

        if (blockNum == 2U) {
            const uint32_t oriStartOffset = block0Idx * blockShapeY;
            auto l1ATensorTlaTile =
                GetTile(l1ATensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(blockShapeY, embed));
            auto gATensorTlaTile =
                GetTile(gATensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(blockShapeY, embed));

            auto l1ASTensorTlaTile = GetTile(l1ASTensorTla, tla::MakeCoord(dealtLenAccum, 0),
                tla::MakeShape(blockShapeY, CeilDiv(embed, MX_SCALE_GROUP_NUM)));
            auto gASTensorTlaTile = GetTile(gASTensor, tla::MakeCoord(oriStartOffset, 0),
                tla::MakeShape(blockShapeY, CeilDiv(embed, MX_SCALE_GROUP_NUM)));
            copyGmToL1AS(l1ASTensorTlaTile, gASTensorTlaTile);
            copyGmToL1A(l1ATensorTlaTile, gATensorTlaTile);
            dealtLenAccum += blockShapeY;
        }

        const uint32_t lastBlockSize = l1BTileNAct - dealtLenAccum;
        const uint32_t oriStartOffset = block1Idx * blockShapeY;
        auto l1ATensorTlaTile =
            GetTile(l1ATensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(lastBlockSize, embed));
        auto gATensorTlaTile =
            GetTile(gATensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(lastBlockSize, embed));
        auto l1ASTensorTlaTile = GetTile(l1ASTensorTla, tla::MakeCoord(dealtLenAccum, 0),
            tla::MakeShape(lastBlockSize, CeilDiv(embed, MX_SCALE_GROUP_NUM)));
        auto gASTensorTlaTile = GetTile(gASTensor, tla::MakeCoord(oriStartOffset, 0),
            tla::MakeShape(lastBlockSize, CeilDiv(embed, MX_SCALE_GROUP_NUM)));
        copyGmToL1AS(l1ASTensorTlaTile, gASTensorTlaTile);
        copyGmToL1A(l1ATensorTlaTile, gATensorTlaTile);
    }

    template <class TensorA, class TensorAS>
    __aicore__ inline void IssueKTileLoad(TensorA &gATensor, TensorAS &gASTensor,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape, uint32_t gatheredKvSTileIdx,
        uint32_t kvSeqlen, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        bool firstLoad) {
        const uint32_t embed = actualOriShape[2];
        const uint32_t curBaseTileSize = actualOriShape[1];
        const uint32_t curBaseTileSizeAligned = Arch35MmadOpt::AlignUpPow2<64>(curBaseTileSize);
        // QMode2 constrains one KV tile to at most the 256-row L1 K tile, so a
        // gathered tile always occupies exactly one L1 stage.
        const uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;
        const uint32_t l1AEventId = l1ABufId;
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(curBaseTileSizeAligned, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        auto l1ASLayoutTla = tla::MakeMxScaleLayout<float8_e8m0_t, LayoutTagL1AS, false>(
            curBaseTileSizeAligned, CeilDiv(embed, MX_SCALE_GROUP_NUM));
        auto l1ASTensorTla = tla::MakeTensor(l1ASTensor[l1ABufId], l1ASLayoutTla, Arch::PositionL1{});
        if (!firstLoad) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
        }
        SparseKL1TileNLoad(gATensor, gASTensor, l1ATensorTla, l1ASTensorTla, gSparseBlockIdx, gatheredKvSTileIdx,
            kvSeqlen, kvSBaseTile, blockShapeY, yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embed, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);
    }

    template <class TensorC>
    __aicore__ inline void ComputeLoadedKTile(TensorC &ubCTensor, GemmCoord actualOriShape, uint32_t gatheredKvSTileIdx,
        uint32_t kvSeqlen, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        Arch::CrossCoreFlag mm1ToSmFlag, uint32_t &l0BufId) {
        static_assert(
            L0A_STAGES == 2 && L0B_STAGES == 2 && L0C_STAGES == 2, "QMode2 QK requires two L0A/L0B/L0C stages");
        const uint32_t rowNum = Arch35MmadOpt::AlignUpPow2<64>(actualOriShape[0]);
        const uint32_t embed = actualOriShape[2];
        const uint32_t curBaseTileSize = actualOriShape[1];
        const uint32_t curBaseTileSizeAligned = Arch35MmadOpt::AlignUpPow2<64>(curBaseTileSize);
        const uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;
        const uint32_t l1AEventId = l1ABufId;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);

        // N <= 256 and K == 128 are loaded once. Only the Q/M dimension is
        // split into at most two 64-column MMADs.
        const uint32_t l0TileNAct = curBaseTileSizeAligned;
        const uint32_t l0TileKAct = embed;
        const uint32_t l0TileNFix = l0TileNAct > 64 ? RoundUp(l0TileNAct, 128) / 2 : l0TileNAct;
        const uint64_t loop3Para = (static_cast<uint64_t>(MMAD_M) << 32) | (static_cast<uint64_t>(l0TileNFix) << 16) |
            static_cast<uint64_t>(l0TileNAct > 64 ? 2 : 1);
        set_loop3_para(loop3Para);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        const auto &loadParams = (curBaseTileSize == l1ATileN) ? normalLoadParams : tailLoadParams;
        AscendC::LoadData(
            l0ATensor[l0BufId], l1ATensor[l1ABufId], l1ASTensor[l1ABufId], loadParams.kData, loadParams.kScale);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
        AscendC::LoadData(
            l0BTensor[l0BufId], l1BTensor[0], l1BSTensor[0], normalLoadParams.qData, normalLoadParams.qScale);
        auto qBlock1DataParams = normalLoadParams.qData;
        auto qBlock1ScaleParams = normalLoadParams.qScale;
        qBlock1DataParams.kStartPosition += embed / 64U;
        qBlock1ScaleParams.yStartPosition += embed / 64U;
        const uint32_t qM64Bytes = l0TileKAct * MMAD_M / 2U;
        auto qL0Block1 =
            l0BTensor[l0BufId].template ReinterpretCast<uint8_t>()[qM64Bytes].template ReinterpretCast<ElementB>();
        AscendC::LoadData(qL0Block1, l1BTensor[0], l1BSTensor[0], qBlock1DataParams, qBlock1ScaleParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

        auto qL0Base = reinterpret_cast<__cb__ uint8_t *>(l0BTensor[l0BufId].GetPhyAddr());
        auto qMmad0 = reinterpret_cast<__cb__ fp4x2_e2m1_t *>(qL0Base);
        auto qMmad1 = reinterpret_cast<__cb__ fp4x2_e2m1_t *>(qL0Base + qM64Bytes);

        constexpr uint32_t mMmadAct = MMAD_M;
        constexpr uint32_t mMmadAligned = MMAD_M;
        const uint32_t l0CBufId = l0CPingPongFlag;
        const uint32_t l0CEventId = l0CBufId;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        asc_mmad_mx(reinterpret_cast<__cc__ float *>(l0CTensor[l0CBufId].GetPhyAddr()),
            reinterpret_cast<__ca__ fp4x2_e2m1_t *>(l0ATensor[l0BufId].GetPhyAddr()), qMmad0, l0TileNAct, l0TileKAct,
            mMmadAligned, 0, true, false, true);

        // Reverse cross-core sync: do the first Fixpipe only after ubCTensor is released.
        WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

        copy_matrix_cc_to_ub(
            reinterpret_cast<__ubuf__ half *>(ubCTensor.data().GetPhyAddr()), // dst_addr: UB中的FP16 RowMajor QK
            reinterpret_cast<__cc__ float *>(l0CTensor[l0CBufId].GetPhyAddr()), // src_addr: L0C中的FP32 QK
            0, // sid: 固定为0
            mMmadAct, // n_size: Q/M方向64行
            l0TileNFix, // m_size: 单个LOOP3分片的KV宽度
            MMAD_M * 2, // loop_dst_stride: UB中相邻分片跨度
            l0TileNAct, // loop_src_stride: L0C中相邻分片跨度
            0, // dual_dst_ctl: 单目的地址
            false, // sub_blockid: 第一个Q64子块
            0, // clip_relu_pre: 关闭
            0, // unit_flag_ctl: 关闭
            static_cast<uint64_t>(QuantMode_t::QF322F16_PRE), // quant_pre: FP32乘scaleValue后转FP16
            false, // relu_pre: 关闭
            false, // split_en: 关闭
            true, // nz2nd_en: L0C NZ转UB ND
            static_cast<uint64_t>(QuantMode_post::NoConv), // quant_post: 不做二次转换
            0, // relu_post: 关闭
            false, // clip_relu_post: 关闭
            false, // loop_enhance_en: 关闭
            0, // eltwise_op: 关闭
            false, // eltwise_antq_en: 关闭
            false, // loop_enhance_merge_en: 关闭
            false, // c0_pad_en: 关闭
            false, // wino_post_en: 关闭
            false, // broadcast_en: 关闭
            false); // nz2dn_en: 关闭；由前面的nz2nd_en选择所需转换
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        l0CPingPongFlag ^= 1U;

        if (rowNum > MMAD_M) {
            constexpr uint32_t mMmadAct = MMAD_M;
            constexpr uint32_t mMmadAligned = MMAD_M;
            const uint32_t l0CBufId = l0CPingPongFlag;
            const uint32_t l0CEventId = l0CBufId;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            asc_mmad_mx(reinterpret_cast<__cc__ float *>(l0CTensor[l0CBufId].GetPhyAddr()),
                reinterpret_cast<__ca__ fp4x2_e2m1_t *>(l0ATensor[l0BufId].GetPhyAddr()), qMmad1, l0TileNAct,
                l0TileKAct, mMmadAligned, 0, true, false, true);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

            copy_matrix_cc_to_ub(reinterpret_cast<__ubuf__ half *>(ubCTensor.data().GetPhyAddr()),
                reinterpret_cast<__cc__ float *>(l0CTensor[l0CBufId].GetPhyAddr()), 0, mMmadAct, l0TileNFix, MMAD_M * 2,
                l0TileNAct, 0, true, 0, 0, static_cast<uint64_t>(QuantMode_t::QF322F16_PRE), false, false, true,
                static_cast<uint64_t>(QuantMode_post::NoConv), 0, false, false, 0, false, false, false, false, false,
                false);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            l0CPingPongFlag ^= 1U;
        }

        // Release the paired L0A/L0B stage only after every M slice consumes it.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0BufId ^= 1U;
        AscendC::PipeBarrier<PIPE_FIX>();
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
    }

  protected:
    __aicore__ inline static void InitL1ToL0LoadParamsForTile(
        L1ToL0LoadParams &params, uint32_t kvTile, uint32_t queryRows, uint32_t embed) {
        const uint32_t kvFractals16 = (kvTile + 15) / 16;
        const uint32_t embedFractals64 = (embed + 63) / 64;
        constexpr uint32_t packedMFractals16 = MMAD_M / 16U;
        const uint32_t packedKFractals64 = embed * 2U / 64U;
        const uint32_t qM64KFractals64 = embed / 64U;

        params.kData =
            AscendC::LoadData2DParamsV2(0, 0, kvFractals16, embedFractals64, kvFractals16, kvFractals16, false, 0);
        params.kScale =
            AscendC::LoadData2DMxParams(0, 0, kvFractals16, embedFractals64, embedFractals64, embedFractals64);
        // Q is reinterpreted as ColumnMajor [K, M] in GM and copied to L1 nZ,
        // so L1-to-L0B is the native nZ-to-nZ path used for Q^T.
        params.qData = AscendC::LoadData2DParamsV2(
            0, 0, packedMFractals16, qM64KFractals64, packedMFractals16, packedMFractals16, false, 0);
        params.qScale =
            AscendC::LoadData2DMxParams(0, 0, packedMFractals16, qM64KFractals64, packedKFractals64, qM64KFractals64);
    }

    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<float8_e8m0_t> l1ASTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<float8_e8m0_t> l1BSTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0A_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0C_STAGES];

    L1ToL0LoadParams normalLoadParams;
    L1ToL0LoadParams tailLoadParams;

    uint32_t l1ATileN;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;
};
////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block
#endif
