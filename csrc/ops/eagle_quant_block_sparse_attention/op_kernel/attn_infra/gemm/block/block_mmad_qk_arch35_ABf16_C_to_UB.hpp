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
#ifndef GEMM_BLOCK_QK_ARCH35_ABF16_C2UB_HPP
#define GEMM_BLOCK_QK_ARCH35_ABF16_C2UB_HPP

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
template <class T>
struct HasParams {
    template <class U>
    static char test(typename U::Params*);
    template <class U>
    static int test(...);
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(char);
};

struct Mm1L1TileHelper {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;

    __aicore__ inline
    Mm1L1TileHelper() {}

    __aicore__ inline
    Mm1L1TileHelper(
        uint32_t m,
        uint32_t n,
        uint32_t kl,
        uint32_t kr,
        uint32_t pbn,
        uint32_t vbn) :
        mm1L1TileM(m),
        mm1L1TileN(n),
        mm1L1TileKLeft(kl),
        mm1L1TileKRight(kr),
        qL1BufNum(pbn),
        kL1BufNum(vbn) {}
};
#endif

template <
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5BsaQK,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_>
{
public:
    using DispatchPolicy = MmadAtlasA5BsaQK;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_HALF_BUF_SIZE / L0_STAGES;

    static constexpr uint32_t MAX_L1_STAGES = 3; // 编译期常量，为静态L1Tensor数组开辟准备。取一个buffer份数的极大值
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16; // 核间同步mode4，AIC侧需要两个flagId分别对应两个AIV

    __aicore__ inline
    BlockMmadTla(Arch::Resource<ArchTag> &resource, Mm1L1TileHelper &mm1L1TileHelper)
    {
        l1ATileM = mm1L1TileHelper.mm1L1TileM;
        l1BTileN = mm1L1TileHelper.mm1L1TileN;
        l1ATileK = mm1L1TileHelper.mm1L1TileKLeft;
        l1BTileK = mm1L1TileHelper.mm1L1TileKRight;
        l1ABufNum = mm1L1TileHelper.qL1BufNum;
        l1BBufNum = mm1L1TileHelper.kL1BufNum;

        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1ATileM * l1ATileK * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
                L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline
    ~BlockMmadTla() {}

    template <class TensorA>
    __aicore__ inline
    void loadQGM(TensorA &gATensor, GemmCoord actualOriShape)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});
        auto l1ATensorTlaTile = GetTile(l1ATensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        auto gATensorTlaTile = GetTile(gATensor,
            tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1A(l1ATensorTlaTile, gATensorTlaTile);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }
    
    __aicore__ inline
    uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
    {
        return outterLoopItr * curLoopNum + curLoopItr;
    }
    
    template <class TensorB, class TensorL1B>
    __aicore__ inline
    void SparseKL1TileNLoad(TensorB &gBTensor, TensorL1B &l1BTensorTla,
                            AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                            uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                            uint32_t kvSBaseTile, uint32_t blockShapeY,
                            uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                            uint32_t l1BTileNAct, uint32_t embed,
                            uint32_t kvSBaseTileInnerOffset)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile + kvSBaseTileInnerOffset;
        uint32_t baseTileEndOffset = baseTileStartOffset + l1BTileNAct;
        // 稀疏情况下对实际选中的部分gather后进行基本块切分
        // 当前处理的Yblock在gather的序列中的起始偏移，初始值为当前基块的起始偏移
        uint32_t gatheredStartOffset = baseTileStartOffset;
        // 当前处理的Yblock gather后的下标，初始值为基本块起始偏移对应的按Y方向稀疏block的gather后起始下标
        uint32_t gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
        // 当前基本块起始偏移对应的按Y方向稀疏后的block内起始偏移
        uint32_t yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
        // 当前处理的Yblock原始的下标，初始值为基本块对应的按Y方向稀疏block的原始起始下标
        uint32_t oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        // 当前处理的Yblock起始位置在原始序列中的偏移，初始值为基本块在原始序列中的起始偏移
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
        // 逐稀疏block搬移填充基本块过程中，已处理的累积序列长度
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < l1BTileNAct && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = blockShapeY;
            if (oriYBlockIdx == yBlockNumAval - 1) {
                curYBlockSize = kvSeqlen - oriYBlockIdx * blockShapeY;
            }
            uint32_t gatheredEndOffset =
                min(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                    baseTileEndOffset);
            // 当前循环处理的序列长度
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) {
                break;
            }

            auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, curDealtLen));
            auto gBTensorTlaTile = GetTile(gBTensor,
                tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, curDealtLen));
            copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
            // 为下一次循环刷新循环变量
            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < l1BTileNAct) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
            }
        }
    }

    template <class TensorB, class TensorC>
    __aicore__ inline
    void operator()(TensorB &gBTensor, TensorC &ubCTensor,
                    AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                    GemmCoord actualOriShape,
                    uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                    uint32_t kvSBaseTile, uint32_t blockShapeY,
                    uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                    uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
                    Arch::CrossCoreFlag mm1ToSmFlag,
                    float scaleValue = 0.0f){
        this->operator()<0>(gBTensor, ubCTensor, gSparseBlockIdx,
                actualOriShape, gatheredKvSTileIdx, kvSeqlen,
                kvSBaseTile, blockShapeY, yBlockNumAval, yBlockNumRsvd,
                prefixSumL0AStages, prefixSumL0BStages,
               mm1ToSmFlag, scaleValue);
    }

    template <
        int quant_mode,
        int staticML0LoopNum = Arch35MmadOpt::DYNAMIC_LOOP,
        int staticKL0LoopNum = Arch35MmadOpt::DYNAMIC_LOOP,
        class TensorB,
        class TensorC>
    __aicore__ inline
    void operator()(TensorB &gBTensor, TensorC &ubCTensor,
                    AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                    GemmCoord actualOriShape,
                    uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                    uint32_t kvSBaseTile, uint32_t blockShapeY,
                    uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                    uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
                    Arch::CrossCoreFlag mm1ToSmFlag,
                    float scaleValue = 0.0f)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDstSub0;
        CopyL0CToDst copyL0CToDstSub1;
        /*
        if constexpr (HasParams<CopyL0CToDst>::value && quant_mode == 1) {
            using DstParams = typename CopyL0CToDst::Params;
            CopyL0CToDstSub0 = CopyL0CToDst(DstParams{scaleValue});
            CopyL0CToDstSub1 = CopyL0CToDst(DstParams{scaleValue});
        }*/

        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t curBaseTileSize = actualOriShape[1];

        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});

        // P full base tile already on L1
        uint32_t nL1LoopNum = Arch35MmadOpt::LoopNum(curBaseTileSize, l1BTileN);
        uint32_t mL0LoopNum = Arch35MmadOpt::LoopNum(rowNum, L0_TILE_M);
        uint32_t kL0LoopNum = Arch35MmadOpt::LoopNum(embed, L0_TILE_K);
        uint32_t mFixPAligned8 = Arch35MmadOpt::AlignUpPow2<8>(rowNum);
        uint32_t mPerSubCore = mFixPAligned8 / 2;
        bool l1ToL0ANoRepeatFlag = (mL0LoopNum == 1U) && (kL0LoopNum <= L0_STAGES);

        // while splitting the base tile S to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        uint32_t nLoopCounterL1 = Arch35MmadOpt::GetCurLoopCounter(gatheredKvSTileIdx, nL1LoopNum);
        uint32_t nL1Offset = 0;
        for (uint32_t nL1Itr = 0; nL1Itr < nL1LoopNum; nL1Itr++) {
            uint32_t l1TileNAct = (nL1Itr == nL1LoopNum - 1) ? (curBaseTileSize - nL1Offset) : l1BTileN;
            uint32_t l1BBufId = nLoopCounterL1 % l1BBufNum;
            uint32_t l1BEventId = l1BBufId + 1;
            auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(embed, l1TileNAct);
            auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
            SparseKL1TileNLoad(
                gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
                yBlockNumAval, yBlockNumRsvd, l1TileNAct, embed, nL1Offset);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
            uint32_t nL0LoopNum = Arch35MmadOpt::LoopNum(l1TileNAct, L0_TILE_N);
            uint32_t lNLoopCounterL0 = Arch35MmadOpt::GetCurLoopCounter(nL1Itr, nL0LoopNum);
            uint32_t l0CLoopCounter = Arch35MmadOpt::GetCurLoopCounter(nLoopCounterL1, nL0LoopNum);
            uint32_t nL0Offset = 0;
            for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (l1TileNAct - nL0Offset) : L0_TILE_N;
                // l0C nbuffer chunked only in n loop
                uint32_t l0CBufId = Arch35MmadOpt::StageId<L0_STAGES>(l0CLoopCounter);
                uint32_t l0CEventId = l0CBufId;
                auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
                auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
                uint32_t mOffset = 0;
                uint32_t l0ALoopBase = static_cast<uint32_t>(prefixSumL0AStages);
                uint32_t l0BLoopBase = static_cast<uint32_t>(prefixSumL0BStages) + lNLoopCounterL0 * kL0LoopNum;
                for (uint32_t mL0Itr = 0; mL0Itr < Arch35MmadOpt::LoopBound<staticML0LoopNum>(mL0LoopNum); mL0Itr++) {
                    uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mOffset) : L0_TILE_M;
                    // uint32_t mLoopCounter = GetCurLoopCounter(gatheredKvSTileIdx, mL0LoopNum, mL0Itr);
                    // different m chunks will be concated in the same piece of l0C buffer
                    auto l0CTensorTlaTile = GetTile(l0CTensorTla,
                        tla::MakeCoord(mOffset, 0), tla::MakeShape(l0TileMAct, l0TileNAct));
                    uint32_t l0ALoopCounter = l0ALoopBase;
                    uint32_t l0BLoopCounter = l0BLoopBase;
                    uint32_t kOffset = 0;
                    for (uint32_t kL0Itr = 0; kL0Itr < Arch35MmadOpt::LoopBound<staticKL0LoopNum>(kL0LoopNum); kL0Itr++) {
                        uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ?
                            (embed - kOffset) : L0_TILE_K;
                        uint32_t l0ABufId = Arch35MmadOpt::StageId<L0_STAGES>(l0ALoopCounter);
                        uint32_t l0BBufId = Arch35MmadOpt::StageId<L0_STAGES>(l0BLoopCounter);
                        uint32_t l0AEventId = l0ABufId;
                        uint32_t l0BEventId = l0BBufId + 2;
                        // when L0B buffers wouldn't be reused across the k loop
                        // redundant L0B load caused by m loop can be avoided
                        auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(kOffset, nL0Offset),
                            tla::MakeShape(l0TileKAct, l0TileNAct));
                        auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                        auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});
                        
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                        copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                        if ((nL0Itr == nL0LoopNum - 1) && (mL0Itr == mL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                        }

                        auto l1ATensorTlaTile = GetTile(l1ATensorTla, tla::MakeCoord(mOffset, kOffset),
                            tla::MakeShape(l0TileMAct, l0TileKAct));
                        auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                        auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                        if (l1ToL0ANoRepeatFlag && (nL1Itr == 0) && (nL0Itr == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                            copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                        }

                        bool initMmad = (kL0Itr == 0);
                        uint32_t l0TileMAligned = Arch35MmadOpt::AlignUpPow2<16>(l0TileMAct);
                        if (l1ToL0ANoRepeatFlag && (nL1Itr == 0) && (nL0Itr == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                        if (mL0Itr == 0 && kL0Itr == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                        }
                        tileMmad(
                            l0CTensorTlaTile,
                            l0ATensorTla,
                            l0BTensorTla,
                            l0TileMAligned,
                            l0TileNAct,
                            l0TileKAct,
                            initMmad);
                        if (l1ToL0ANoRepeatFlag && (nL1Itr == nL1LoopNum - 1) && (nL0Itr == nL0LoopNum - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                        l0ALoopCounter++;
                        l0BLoopCounter++;
                        kOffset += L0_TILE_K;
                    }
                    mOffset += L0_TILE_M;
                    l0ALoopBase += kL0LoopNum;
                }
                // fixpipe
                if (nL0Itr == 0) {
                    // reverse crossCoreSync, do fixPipe only after ubCTensor is fully released
                    WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                // 需要kernel传输ubCTensor的时候确保其shape的m，n是满足32B（8个32位元素）对齐的
                // rounded up by 8 and splited in half to each AIV
                // valid rows in AIV0: [0, mFixPAligned8 / 2 - 1]
                // valid rows in AIV1: [mFixPAligned8 / 2, rowNum - 1]
                uint32_t nFixPAligned16 = Arch35MmadOpt::AlignUpPow2<16>(l0TileNAct);
                auto ubCTensorTlaTile = GetTile(ubCTensor,
                    tla::MakeCoord(0, nL0Offset), tla::MakeShape(mPerSubCore, nFixPAligned16));
                auto l0CTensorTlaTileSub0 = GetTile(l0CTensorTla,
                        tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, l0TileNAct));
                auto l0CTensorTlaTileSub1 = GetTile(l0CTensorTla,
                        tla::MakeCoord(mPerSubCore, 0), tla::MakeShape(mPerSubCore, l0TileNAct));
                copyL0CToDstSub0(ubCTensorTlaTile, l0CTensorTlaTileSub0, false);
                copyL0CToDstSub1(ubCTensorTlaTile, l0CTensorTlaTileSub1, true);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                lNLoopCounterL0++;
                l0CLoopCounter++;
                nL0Offset += L0_TILE_N;
            }
            nLoopCounterL1++;
            nL1Offset += l1BTileN;
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
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

}  // namespace NpuArch::Gemm::Block
#endif
