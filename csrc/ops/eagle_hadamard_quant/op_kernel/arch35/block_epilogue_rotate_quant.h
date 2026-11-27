/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_rotate_quant.h
 * \brief Block epilogue for rotate quantization with MX FP4/FP8 support
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/tile/arch35/fill_ub.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/tile/arch35/mx_quant.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_> class BlockEpilogueRotateQuant {
  public:
    using DataTypeIn = DataTypeIn_;
    using DataTypeOut = DataTypeOut_;
    using DataTypeScale = DataTypeScale_;

    static_assert(AscendC::IsSameType<DataTypeIn, bfloat16_t>::value,
        "BlockEpilogueRotateQuant requires BF16 input from Fixpipe.");
    static_assert(
        sizeof(DataTypeScale) == sizeof(int8_t), "BlockEpilogueRotateQuant requires one-byte MX scale storage.");

    static_assert(AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
            AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
            AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value,
        "BlockEpilogueRotateQuant only supports FP4-E2M1 / FP8-E4M3 / FP8-E5M2 output.");

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t> problemShape{};
        bfloat16_t alpha{0};
        bool needClamp{false};
        int64_t axis{0};
        int64_t roundMode{0};
        int64_t scaleAlg{0};
        float dstTypeMax{0.0f};
        float invDstTypeMax{0.0f};
    };

    __aicore__ inline BlockEpilogueRotateQuant() {}

    __aicore__ inline void Init(Params const &params);

    __aicore__ static inline bool CanDoubleBufferOutputs(uint32_t rows, uint32_t columns) {
        return rows <= MAX_K128_ROWS && static_cast<uint64_t>(rows) * columns <= MN_SIZE &&
            SCALE_BLOCK_PONG_OFFSET + K128_SCALE_BLOCK_BYTES <= AscendC::TOTAL_UB_SIZE;
    }

    __aicore__ static inline bool CanDoubleBufferInputs(uint32_t rows, uint32_t columns) {
        return rows <= MAX_K128_ROWS && static_cast<uint64_t>(rows) * columns <= MN_SIZE &&
            X_PONG_OFFSET + MN_SIZE * sizeof(bfloat16_t) <= AscendC::TOTAL_UB_SIZE;
    }

    __aicore__ static inline uint32_t InputOffset(uint16_t slot) { return slot == 0 ? X_UB_OFFSET : X_PONG_OFFSET; }

    // The paired path keeps one shared BF16 conversion tile at X_UB_OFFSET.
    // Its FP32 Fixpipe destinations start beyond both output slots and vector scratch.
    __aicore__ static inline uint32_t PairInputOffset(uint16_t slot) {
        return X_PONG_OFFSET + slot * MN_SIZE * sizeof(float);
    }

    template <uint16_t SyncMode, bool paired = false>
    __aicore__ inline void ProcessTile(const BlockShape &blockShape, const BlockCoord &blockCoord,
        bool releaseBeforeWriteback, uint16_t inputFreeFlag, uint32_t inputOffset, uint16_t outputSlot);

    template <uint16_t SyncMode>
    __aicore__ inline void ProcessPanelPair(const BlockCoord &blockCoord, uint16_t pairColumn, bool lastPair,
        bool reuseInput, uint16_t inputFreeFlag, uint32_t inputOffset);

    static constexpr uint16_t RING_GROUPS = 2;
    static constexpr bool RING_FP4 = AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value;
    static constexpr uint32_t RING_TILE_ROW_BYTES = RING_FP4 ? 64 : 128;
    static constexpr uint32_t RING_TILE_BYTES = 128 * RING_TILE_ROW_BYTES;
    static constexpr uint32_t RING_ROW_BYTES = 4 * RING_TILE_ROW_BYTES;
    // FP8 needs twice the output bytes per row; halve the rows to retain two UB groups.
    static constexpr uint16_t RING_ROWS = RING_FP4 ? 256 : 128;
    static constexpr uint32_t INPUT_SLOT_BYTES = 128 * 128 * sizeof(bfloat16_t);
    static constexpr uint32_t PAIR_Y = 2 * INPUT_SLOT_BYTES;
    static constexpr uint32_t PAIR_SCALE = PAIR_Y + 2 * RING_TILE_BYTES;
    static constexpr uint32_t PAIR_SCRATCH = PAIR_SCALE + 2 * 512;
    static constexpr uint32_t RING_Y = PAIR_SCRATCH + 3 * 1024;
    static constexpr uint32_t RING_SCALE = RING_Y + RING_GROUPS * RING_ROWS * RING_ROW_BYTES;
    static constexpr uint32_t RING_END = RING_SCALE + RING_GROUPS * RING_ROWS * 32;
    static_assert(RING_END <= 248 * 1024);

  private:
    static __simd_vf__ inline void CastPairInputVf(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint32_t count) {
        using namespace AscendC::Reg;
        // LocalTensor Cast uses SAT on DAV-3510. Preserve Inf/NaN, as BF16 Fixpipe does.
        static constexpr CastTrait trait = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr uint16_t elements = AscendC::GetVecLen() / sizeof(float);
        const uint16_t repeats = (count + elements - 1) / elements;
        RegTensor<float> raw;
        RegTensor<bfloat16_t> bf16;
        for (uint16_t i = 0; i < repeats; ++i) {
            auto mask = UpdateMask<float>(count);
            LoadAlign(raw, src + i * elements);
            Cast<bfloat16_t, float, trait>(bf16, raw, mask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dst + i * elements, bf16, mask);
        }
    }

    uint16_t ringGroup_ = 0;
    uint16_t ringRows_ = 0;
    uint64_t ringGm_ = 0;

    static __simd_vf__ inline void Pack128(
        __ubuf__ int8_t *y, __ubuf__ int8_t *scales, __ubuf__ int8_t *yDst, __ubuf__ int8_t *scaleDst) {
        using namespace AscendC::Reg;
        uint32_t count = RING_TILE_ROW_BYTES;
        auto mask = UpdateMask<int8_t>(count);
        for (uint16_t row = 0; row < 128; ++row) {
            RegTensor<int8_t> left, right;
            LoadAlign(left, y + row * RING_TILE_ROW_BYTES);
            LoadAlign(right, y + (row + 128) * RING_TILE_ROW_BYTES);
            StoreAlign(yDst + row * RING_ROW_BYTES, left, mask);
            StoreAlign(yDst + row * RING_ROW_BYTES + RING_TILE_ROW_BYTES, right, mask);
            auto *lo = scales + row * 4;
            auto *hi = lo + 512;
            auto *dst = scaleDst + row * 32;
            UnalignReg a, b, c;
            LoadUnAlignPre(a, lo);
            LoadUnAlign(left, a, lo);
            LoadUnAlignPre(b, hi);
            LoadUnAlign(right, b, hi);
            StoreUnAlign<int8_t, PostLiteral::POST_MODE_UPDATE>(dst, left, c, 4);
            StoreUnAlign<int8_t, PostLiteral::POST_MODE_UPDATE>(dst, right, c, 4);
            StoreUnAlignPost<int8_t, PostLiteral::POST_MODE_UPDATE>(dst, c, 0);
        }
    }

  public:
    template <uint16_t SyncMode> __aicore__ inline void RingQuant(uint16_t lane, uint16_t freeFlag, bool reuse) {
        ComputeMxQuant(
            16384, lane * INPUT_SLOT_BYTES, PAIR_Y + lane * RING_TILE_BYTES, PAIR_SCALE + lane * 512, PAIR_SCRATCH);
        if (reuse) {
            Gemm::Sync::NotifyCube<SyncMode, PIPE_V>(freeFlag);
        }
    }
    __aicore__ inline void RingPack(uint64_t gm, uint16_t pairColumn, uint16_t issuePair, bool lastRow) {
        const uint32_t yBase = RING_Y + ringGroup_ * RING_ROWS * RING_ROW_BYTES;
        const uint32_t sBase = RING_SCALE + ringGroup_ * RING_ROWS * 32;
        if (issuePair == 0 && ringRows_ == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ringGroup_);
            ringGm_ = gm;
        }
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<Pack128>(reinterpret_cast<__ubuf__ int8_t *>(PAIR_Y),
            reinterpret_cast<__ubuf__ int8_t *>(PAIR_SCALE),
            reinterpret_cast<__ubuf__ int8_t *>(yBase + ringRows_ * RING_ROW_BYTES + pairColumn * RING_TILE_ROW_BYTES),
            reinterpret_cast<__ubuf__ int8_t *>(sBase + ringRows_ * 32 + pairColumn * 4));
        AscendC::PipeBarrier<PIPE_V>();
        if (issuePair == 1) {
            ringRows_ += 128;
            if (ringRows_ == RING_ROWS || lastRow) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ringGroup_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ringGroup_);
                CopyOutputFromUbToGm(ringGm_, ringRows_, 512, shape_.n, yBase);
                CopyScaleFromUbToGm(ringGm_ / 32, sBase, ringRows_, 16, shape_.n / 32, 32);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ringGroup_);
                ringGroup_ = (ringGroup_ + 1) % RING_GROUPS;
                ringRows_ = 0;
            }
        }
    }

  private:
    static constexpr int64_t MODE_RINT = 4;
    static constexpr int64_t MODE_ROUND = 0;
    static constexpr int64_t MODE_FLOOR = 1;

    static constexpr int32_t GROUP_SIZE = 32;
    static constexpr int32_t MN_SIZE = 16 * 1024;
    static constexpr int32_t EMAX_SIZE = 512;
    // Slot zero keeps the original UB layout. The extra K=128 slots follow all shared scratch.
    static constexpr uint32_t X_UB_OFFSET = 0;
    static constexpr uint32_t Y_UB_OFFSET = X_UB_OFFSET + MN_SIZE * sizeof(bfloat16_t);
    static constexpr uint32_t LIMIT_UB_OFFSET = Y_UB_OFFSET + MN_SIZE * sizeof(int8_t);
    static constexpr uint32_t MAX_EXP_UB_OFFSET = LIMIT_UB_OFFSET + EMAX_SIZE * sizeof(bfloat16_t);
    static constexpr uint32_t RECIPROCAL_UB_OFFSET = MAX_EXP_UB_OFFSET + EMAX_SIZE * sizeof(uint16_t);
    static constexpr uint32_t SCALE_UB_OFFSET = RECIPROCAL_UB_OFFSET + EMAX_SIZE * sizeof(uint16_t);
    static constexpr uint32_t SCALE_BLOCK_UB_OFFSET = SCALE_UB_OFFSET + EMAX_SIZE * sizeof(int8_t);

    static constexpr float SIX_FLOAT = 6.0f;
    static constexpr float SEVEN_FLOAT = 7.0f;
    static constexpr uint32_t SCALE_STORE_STRIDE = 32;
    static constexpr uint32_t MAX_K128_ROWS = MN_SIZE / 128;
    static constexpr uint32_t K128_SCALE_BLOCK_BYTES = MAX_K128_ROWS * SCALE_STORE_STRIDE;
    // Reserve the same full Y tensor extent as slot zero. Both slots reuse the vector-only scratch.
    static constexpr uint32_t Y_PONG_OFFSET = SCALE_BLOCK_UB_OFFSET + K128_SCALE_BLOCK_BYTES;
    static constexpr uint32_t SCALE_BLOCK_PONG_OFFSET = Y_PONG_OFFSET + MN_SIZE * sizeof(int8_t);
    static constexpr uint32_t X_PONG_OFFSET = SCALE_BLOCK_PONG_OFFSET + K128_SCALE_BLOCK_BYTES;

    static constexpr uint16_t PANEL_ROWS = 64; // One AIV's half of the 128-row Fixpipe tile.
    static constexpr uint16_t PANEL_COLUMNS = 512;
    static constexpr uint16_t PAIR_COLUMNS = PANEL_COLUMNS / 2;
    static constexpr uint16_t PAIR_Y_BYTES = PAIR_COLUMNS / 2;
    static constexpr uint16_t PAIR_SCALE_BYTES = PAIR_COLUMNS / GROUP_SIZE;
    static_assert(2 * PANEL_ROWS * PAIR_Y_BYTES <= LIMIT_UB_OFFSET - Y_UB_OFFSET);
    static_assert(2 * PANEL_ROWS * PAIR_SCALE_BYTES <= K128_SCALE_BLOCK_BYTES);
    static_assert(PANEL_ROWS * SCALE_STORE_STRIDE <= K128_SCALE_BLOCK_BYTES);

    static constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
    static constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
    static constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;

    __aicore__ inline void InitGlobalBuffer(Params const &params);
    __aicore__ inline void InitProblemShape(const ProblemShape &problemShape);
    __aicore__ inline void InitQuantParams(bfloat16_t alpha, bool needClamp, int64_t axis, int64_t roundMode,
        int64_t scaleAlg, float dstTypeMax, float invDstTypeMax);
    __aicore__ inline void CopyUbToGm(__gm__ int8_t *dst, uint32_t srcUbOffset, uint16_t rows, uint32_t columns,
        uint32_t srcRowPitch, uint32_t dstRowPitch);
    __aicore__ inline void CopyOutputFromUbToGm(uint64_t offset, uint16_t blockCount, uint32_t dataSize,
        uint32_t dstStride, uint32_t srcUbOffset = Y_UB_OFFSET);
    __aicore__ inline void CopyScaleFromUbToGm(uint64_t offset, uint32_t srcUbOffset, uint16_t blockCount,
        uint32_t dataSize, uint32_t dstStride, uint32_t srcStride);
    __aicore__ inline void CopyPadScaleFromUbToGm(
        uint64_t offset, uint32_t nKRatio, uint32_t nkIdx, uint16_t firstNum, uint16_t rowLoopNum, uint16_t lastNum);

    __aicore__ inline void CastPairInput(uint32_t inputOffset, uint32_t count);
    __aicore__ inline void ComputeMxQuant(uint32_t totalDataInUB, uint32_t inputOffset, uint32_t yUbOffset,
        uint32_t scaleUbOffset = SCALE_UB_OFFSET, uint32_t scratch = LIMIT_UB_OFFSET);
    static __simd_vf__ inline void PackPanelVf(
        __ubuf__ int8_t *y, __ubuf__ int8_t *scales, __ubuf__ int8_t *panelY, __ubuf__ int8_t *panelScale);
    __aicore__ inline void ClearScaleBlock(uint32_t blkM, uint32_t scaleBlockOffset = SCALE_BLOCK_UB_OFFSET);
    __aicore__ inline void ComputeTransLayout(uint16_t mSize, uint16_t scaleBlockN, uint32_t scaleBlockOffset);
    __aicore__ inline void TransFp4Output(uint16_t mSize, uint16_t kSize, uint32_t srcOffset, uint32_t dstOffset);
    __aicore__ inline void PadScale(
        uint32_t nKRatio, uint32_t blkM, uint16_t firstNum, uint16_t rowLoopNum, uint16_t lastNum);

    static __simd_vf__ inline void PadScaleVf(__ubuf__ int8_t *scaleAddr, __ubuf__ int8_t *scaleBlkAddr,
        uint32_t nKRatio, uint16_t firstNum, uint16_t lastNum, uint16_t rowLoopNum);
    static __simd_callee__ inline void DataCopyByCount(
        __ubuf__ int8_t *dstAddr, __ubuf__ int8_t *srcAddr, uint32_t count);

    // ---- Shape ----
    struct RotateQuantShapeInfo {
        int64_t m;
        int64_t n;
        int64_t b;
    };

    RotateQuantShapeInfo shape_;

    // ---- GM ----
    __gm__ int8_t *cGlobal_{nullptr};
    __gm__ int8_t *scaleGlobal_{nullptr};

    // ---- Quant Params ----
    bfloat16_t alpha_ = 0.0f;
    bool needClamp_ = false;
    int64_t axis_ = 0;
    int64_t roundMode_ = 0;
    int64_t scaleAlg_ = 0;
    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t fpEmax_ = 0;
    uint16_t addValueBits_ = 0;
};

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::Init(Params const &params) {
    // 饱和模式由单指令设置生效
    InitGlobalBuffer(params);
    InitProblemShape(params.problemShape);
    InitQuantParams(params.alpha, params.needClamp, params.axis, params.roundMode, params.scaleAlg, params.dstTypeMax,
        params.invDstTypeMax);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
template <uint16_t SyncMode, bool paired>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::ProcessTile(
    const BlockShape &blockShape, const BlockCoord &blockCoord, bool releaseBeforeWriteback, uint16_t inputFreeFlag,
    uint32_t inputOffset, uint16_t outputSlot) {
    const int64_t blkM = asc::te::get<Gemm::MNK_M>(blockShape) / (paired ? 2 : 1);
    const int64_t groupWidth = asc::te::get<Gemm::MNK_K>(blockShape);
    const int64_t k = (paired ? 2 : 1) * groupWidth;
    const uint64_t firstRow = asc::te::get<Gemm::MNK_M>(blockCoord) + (paired ? AscendC::GetSubBlockIdx() * blkM : 0);
    const uint64_t offset = firstRow * shape_.b + asc::te::get<Gemm::MNK_B>(blockCoord);
    uint32_t mnSize = blkM * k;
    uint64_t yOffset = offset * static_cast<uint64_t>(groupWidth);
    const uint32_t yUbOffset = outputSlot == 0 ? Y_UB_OFFSET : Y_PONG_OFFSET;
    uint32_t writebackOffset = yUbOffset;
    const uint32_t scaleBlockOffset = outputSlot == 0 ? SCALE_BLOCK_UB_OFFSET : SCALE_BLOCK_PONG_OFFSET;
    if constexpr (paired) {
        CastPairInput(inputOffset, mnSize);
        inputOffset = X_UB_OFFSET;
    }
    ComputeMxQuant(mnSize, inputOffset, yUbOffset);

    uint32_t nScaleSize = Gemm::CeilDiv<int64_t>(shape_.n, GROUP_SIZE);
    uint32_t isOdd = nScaleSize & 1;
    nScaleSize += isOdd;
    if (shape_.b > 1) {
        uint64_t mIdx = offset / shape_.b;
        uint64_t bIdx = offset % shape_.b;
        uint64_t scaleOffset = mIdx * nScaleSize + Gemm::CeilDiv<uint64_t>(bIdx * groupWidth, GROUP_SIZE);
        uint16_t scaleBlockN = Gemm::CeilDiv<int64_t>(k, GROUP_SIZE);
        uint16_t dataSize = scaleBlockN + ((bIdx == shape_.b - 1) ? isOdd : 0);

        // CopyScaleFromUbToGm 按行距 SCALE_STORE_STRIDE(32B) 读取, 清零必须覆盖每行完整行距,
        // 否则最后一个 bIdx 的 pad 字节(行 m 的 m*32+1 处)读到未初始化 UB, 污染 scale 输出
        ClearScaleBlock(blkM, scaleBlockOffset);
        ComputeTransLayout(blkM, scaleBlockN, scaleBlockOffset);
        if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
            // A packed K=32 row occupies only 16 bytes. Give MTE3 a 32-byte UB row pitch.
            if (k < 64 && blkM > 1) {
                AscendC::PipeBarrier<PIPE_V>();
                // A <=128-row K=32 tile uses <=2 KiB packed / 4 KiB padded.
                // Keep padding in the upper 8 KiB of this 16 KiB output slot,
                // so MTE3 no longer reads X and Fixpipe may safely reuse it.
                writebackOffset = blkM <= MAX_K128_ROWS ? yUbOffset + MN_SIZE / 2 : X_UB_OFFSET;
                TransFp4Output(blkM, k, yUbOffset, writebackOffset);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputSlot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputSlot);
        if (releaseBeforeWriteback) {
            // Both GM stores read output slots; Fixpipe may now reuse X.
            Gemm::Sync::NotifyCube<SyncMode, PIPE_MTE3>(inputFreeFlag);
        }
        CopyOutputFromUbToGm(yOffset, blkM, k, shape_.n, writebackOffset);
        CopyScaleFromUbToGm(scaleOffset, scaleBlockOffset, blkM, dataSize, nScaleSize, SCALE_STORE_STRIDE);
    } else if (isOdd == 1) {
        uint32_t nKRatio = shape_.n / k;
        uint64_t mIdx = offset / nKRatio;
        uint32_t nkIdx = offset % nKRatio;
        uint64_t scaleOffset = mIdx * nScaleSize + nkIdx;
        ClearScaleBlock(blkM);
        const uint16_t firstNum = (nKRatio - nkIdx <= blkM) ? nKRatio - nkIdx : blkM;
        const uint16_t rowLoopNum = (firstNum < blkM) ? (blkM - firstNum) / nKRatio : 0;
        const uint16_t lastNum = (firstNum < blkM) ? (blkM - firstNum) % nKRatio : 0;
        PadScale(nKRatio, blkM, firstNum, rowLoopNum, lastNum);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        CopyOutputFromUbToGm(yOffset, 1, mnSize, 0);
        CopyPadScaleFromUbToGm(scaleOffset, nKRatio, nkIdx, firstNum, rowLoopNum, lastNum);
    } else {
        uint64_t scaleOffset = offset * Gemm::CeilDiv<int64_t>(k, GROUP_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        CopyOutputFromUbToGm(yOffset, 1, mnSize, 0);
        uint32_t scaleSize = Gemm::CeilDiv<uint32_t>(mnSize, GROUP_SIZE);
        CopyScaleFromUbToGm(scaleOffset, SCALE_UB_OFFSET, 1, scaleSize, 0, scaleSize);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputSlot);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::CastPairInput(
    uint32_t inputOffset, uint32_t count) {
    asc_vf_call<CastPairInputVf>(
        reinterpret_cast<__ubuf__ bfloat16_t *>(X_UB_OFFSET), reinterpret_cast<__ubuf__ float *>(inputOffset), count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
template <uint16_t SyncMode>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::ProcessPanelPair(
    const BlockCoord &blockCoord, uint16_t pairColumn, bool lastPair, bool reuseInput, uint16_t inputFreeFlag,
    uint32_t inputOffset) {
    CastPairInput(inputOffset, PANEL_ROWS * PAIR_COLUMNS);
    if (reuseInput) {
        Gemm::Sync::NotifyCube<SyncMode, PIPE_V>(inputFreeFlag);
    }
    // pairColumn is the physical group (0 or 2), independent of issue order.
    const uint16_t half = pairColumn / 2;
    ComputeMxQuant(PANEL_ROWS * PAIR_COLUMNS, X_UB_OFFSET, Y_UB_OFFSET + half * PANEL_ROWS * PAIR_Y_BYTES,
        SCALE_BLOCK_UB_OFFSET + half * PANEL_ROWS * PAIR_SCALE_BYTES);
    if (!lastPair) {
        return;
    }
    // Only the final packed panel is shared with MTE3. Both quantizations can overlap its previous store.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<PackPanelVf>(reinterpret_cast<__ubuf__ int8_t *>(Y_UB_OFFSET),
        reinterpret_cast<__ubuf__ int8_t *>(SCALE_BLOCK_UB_OFFSET), reinterpret_cast<__ubuf__ int8_t *>(Y_PONG_OFFSET),
        reinterpret_cast<__ubuf__ int8_t *>(SCALE_BLOCK_PONG_OFFSET));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const uint64_t row = asc::te::get<Gemm::MNK_M>(blockCoord) + AscendC::GetSubBlockIdx() * PANEL_ROWS;
    const uint64_t group = asc::te::get<Gemm::MNK_B>(blockCoord) - pairColumn;
    const uint64_t offset = row * shape_.n + group * (PANEL_COLUMNS / 4);
    CopyOutputFromUbToGm(offset, PANEL_ROWS, PANEL_COLUMNS, shape_.n, Y_PONG_OFFSET);
    CopyScaleFromUbToGm(offset / GROUP_SIZE, SCALE_BLOCK_PONG_OFFSET, PANEL_ROWS, PANEL_COLUMNS / GROUP_SIZE,
        shape_.n / GROUP_SIZE, SCALE_STORE_STRIDE);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__simd_vf__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::PackPanelVf(
    __ubuf__ int8_t *y, __ubuf__ int8_t *scales, __ubuf__ int8_t *panelY, __ubuf__ int8_t *panelScale) {
    using namespace AscendC::Reg;
    uint32_t count = PAIR_Y_BYTES;
    MaskReg yMask = UpdateMask<int8_t>(count);
    for (uint16_t row = 0; row < PANEL_ROWS; ++row) {
        RegTensor<int8_t> left, right;
        LoadAlign(left, y + row * PAIR_Y_BYTES);
        LoadAlign(right, y + (PANEL_ROWS + row) * PAIR_Y_BYTES);
        StoreAlign(panelY + row * 2 * PAIR_Y_BYTES, left, yMask);
        StoreAlign(panelY + (row * 2 + 1) * PAIR_Y_BYTES, right, yMask);

        auto *lo = scales + row * PAIR_SCALE_BYTES;
        auto *hi = lo + PANEL_ROWS * PAIR_SCALE_BYTES;
        auto *dst = panelScale + row * SCALE_STORE_STRIDE;
        UnalignReg loLoad, hiLoad, store;
        LoadUnAlignPre(loLoad, lo);
        LoadUnAlign(left, loLoad, lo);
        LoadUnAlignPre(hiLoad, hi);
        LoadUnAlign(right, hiLoad, hi);
        StoreUnAlign<int8_t, PostLiteral::POST_MODE_UPDATE>(dst, left, store, PAIR_SCALE_BYTES);
        StoreUnAlign<int8_t, PostLiteral::POST_MODE_UPDATE>(dst, right, store, PAIR_SCALE_BYTES);
        StoreUnAlignPost<int8_t, PostLiteral::POST_MODE_UPDATE>(dst, store, 0);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::InitGlobalBuffer(
    Params const &params) {
    cGlobal_ = reinterpret_cast<__gm__ int8_t *>(params.outGmAddr);
    scaleGlobal_ = reinterpret_cast<__gm__ int8_t *>(params.scaleGmAddr);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::InitProblemShape(
    const ProblemShape &problemShape) {
    shape_.m = asc::te::get<Gemm::MNK_M>(problemShape);
    shape_.n = asc::te::get<Gemm::MNK_N>(problemShape);
    shape_.b = asc::te::get<Gemm::MNK_B>(problemShape);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::InitQuantParams(
    bfloat16_t alpha, bool needClamp, int64_t axis, int64_t roundMode, int64_t scaleAlg, float dstTypeMax,
    float invDstTypeMax) {
    alpha_ = alpha;
    needClamp_ = needClamp;
    axis_ = axis;
    roundMode_ = roundMode;
    scaleAlg_ = scaleAlg;
    dstTypeMax_ = dstTypeMax;
    invDstTypeMax_ = invDstTypeMax;

    // 进位附加值规则统一为 gelu_tanh 语义（DYN 路径使用）
    addValueBits_ = dstTypeMax_ == SEVEN_FLOAT ? ADD_VALUE_FOR_BF16_MAN2 : ADD_VALUE_FOR_BF16_MAN1;

    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
        // FP8 时 host 强制 dstTypeMax=0（下刷的 invDstTypeMax 为 0），cuBLAS 路径按输出 dtype 取 1/dstTypeMax
        invDstTypeMax_ = 1.0f / 448.0f;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        invDstTypeMax_ = 1.0f / 57344.0f;
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::CopyUbToGm(
    __gm__ int8_t *dst, uint32_t srcUbOffset, uint16_t rows, uint32_t columns, uint32_t srcRowPitch,
    uint32_t dstRowPitch) {
    auto ubLayout = Gemm::MakeNDExtLayout<int8_t>(rows, columns, srcRowPitch);
    auto gmLayout = Gemm::MakeNDExtLayout<int8_t>(rows, columns, dstRowPitch);
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(srcUbOffset), ubLayout);
    auto output = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(dst), gmLayout);
    asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), output, src);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::CopyOutputFromUbToGm(
    uint64_t offset, uint16_t blockCount, uint32_t dataSize, uint32_t dstStride, uint32_t srcUbOffset) {
    uint32_t srcStride = dataSize;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        offset >>= 1;
        dataSize >>= 1;
        dstStride >>= 1;
        srcStride = dataSize;
        if (blockCount > 1 && dataSize < SCALE_STORE_STRIDE) {
            srcStride = SCALE_STORE_STRIDE;
        }
    }
    CopyUbToGm(cGlobal_ + offset, srcUbOffset, blockCount, dataSize, srcStride, dstStride == 0 ? dataSize : dstStride);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::CopyScaleFromUbToGm(
    uint64_t offset, uint32_t srcUbOffset, uint16_t blockCount, uint32_t dataSize, uint32_t dstStride,
    uint32_t srcStride) {
    CopyUbToGm(
        scaleGlobal_ + offset, srcUbOffset, blockCount, dataSize, srcStride, dstStride == 0 ? dataSize : dstStride);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::CopyPadScaleFromUbToGm(
    uint64_t offset, uint32_t nKRatio, uint32_t nkIdx, uint16_t firstNum, uint16_t rowLoopNum, uint16_t lastNum) {
    uint32_t dataSize = firstNum + ((nkIdx + firstNum == nKRatio) ? 1 : 0);
    uint32_t srcOffset = SCALE_BLOCK_UB_OFFSET;
    CopyScaleFromUbToGm(offset, srcOffset, 1, dataSize, dataSize, dataSize);
    offset += dataSize;
    srcOffset += Gemm::CeilAlign<uint32_t>(firstNum + 1, SCALE_STORE_STRIDE);

    if (rowLoopNum > 0) {
        dataSize = nKRatio + 1;
        uint32_t srcStride = Gemm::CeilAlign<uint32_t>(dataSize, SCALE_STORE_STRIDE);
        CopyScaleFromUbToGm(offset, srcOffset, rowLoopNum, dataSize, dataSize, srcStride);
        offset += rowLoopNum * dataSize;
        srcOffset += rowLoopNum * srcStride;
    }

    if (lastNum > 0) {
        CopyScaleFromUbToGm(offset, srcOffset, 1, lastNum, lastNum, lastNum);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::ComputeMxQuant(
    uint32_t totalDataInUB, uint32_t inputOffset, uint32_t yUbOffset, uint32_t scaleUbOffset, uint32_t scratch) {
    const uint32_t totalScale = Gemm::CeilDiv<uint32_t>(totalDataInUB, GROUP_SIZE);
    auto dataLayout =
        Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalDataInUB), static_cast<int64_t>(totalDataInUB));
    auto scaleLayout =
        Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalScale), static_cast<int64_t>(totalScale));
    auto srcTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(inputOffset), dataLayout);
    auto quantYTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(yUbOffset), dataLayout);
    auto limitTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(scratch + LIMIT_UB_OFFSET - LIMIT_UB_OFFSET),
        scaleLayout);
    auto maxExpTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(scratch + MAX_EXP_UB_OFFSET - LIMIT_UB_OFFSET),
        scaleLayout);
    auto yScaleTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(scaleUbOffset), scaleLayout);
    auto reciprocalTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(scratch + RECIPROCAL_UB_OFFSET - LIMIT_UB_OFFSET),
        scaleLayout);

    // MxQuantConfig 由 Init 阶段解析的普通成员在调用点组装（tile 类型不得出现在类作用域）
    // zeroScaleOnZeroExp=false: rotate/gelu_tanh 阵营（sharedExp 判定，仅清零 reciprocal）
    Tile::MxQuantConfig cfg{Tile::MxScaleAlg::OCP, fpEmax_, invDstTypeMax_, addValueBits_, false};
    // The original RotateQuant only rounds subnormal scales up for FP8.
    cfg.roundSubnormalScaleUp = !AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value;
    if (scaleAlg_ == 1) { // per-channel cuBLAS（仅 FP8）
        cfg.alg = Tile::MxScaleAlg::CUBLAS;
    } else if (scaleAlg_ == 0) {
        cfg.alg = Tile::MxScaleAlg::OCP;
    } else if (dstTypeMax_ == SIX_FLOAT || dstTypeMax_ == SEVEN_FLOAT) {
        cfg.alg = Tile::MxScaleAlg::DYN_DTYPE_RANGE;
    } else { // FP4 cuBLAS: dstTypeMax in [6, 12], excluding 6 and 7; reciprocal comes from host.
        cfg.alg = Tile::MxScaleAlg::CUBLAS;
    }

    Tile::MxQuant<DataTypeOut> mx;
    if (needClamp_) {
        // 钳位链路: limit = absMax * alpha；useExp: OCP 取 limit 指数字段，其余取 limit 全位
        mx.GroupMaxExp(srcTensor, maxExpTensor, limitTensor, totalDataInUB, alpha_, scaleAlg_ == 0);
    } else {
        // useAbs: OCP 取指数字段，cuBLAS/Dyn 取绝对值位
        mx.GroupMaxExp(srcTensor, maxExpTensor, totalDataInUB, scaleAlg_ != 0);
    }
    mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScale);

    Tile::MxQuantFp4RoundMode fp4RoundMode = Tile::MxQuantFp4RoundMode::RINT;
    if (roundMode_ == MODE_FLOOR) {
        fp4RoundMode = Tile::MxQuantFp4RoundMode::FLOOR;
    } else if (roundMode_ == MODE_ROUND) {
        fp4RoundMode = Tile::MxQuantFp4RoundMode::ROUND;
    }
    if (needClamp_) {
        mx.Quantize(srcTensor, reciprocalTensor, limitTensor, quantYTensor, totalDataInUB, fp4RoundMode);
    } else {
        mx.Quantize(srcTensor, reciprocalTensor, quantYTensor, totalDataInUB, fp4RoundMode);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::ClearScaleBlock(
    uint32_t blkM, uint32_t scaleBlockOffset) {
    uint32_t count = blkM * SCALE_STORE_STRIDE;
    auto layout = Gemm::MakeNDExtLayout<int8_t>(1, count, count);
    auto scaleBlock =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(scaleBlockOffset), layout);
    Gemm::Tile::FillUb<int8_t>::FillWithValue(scaleBlock, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::ComputeTransLayout(
    uint16_t mSize, uint16_t scaleBlockN, uint32_t scaleBlockOffset) {
    // scale 布局: (M, scaleBlockN) -> (M, 32B 行块)，供 MTE 按行距 SCALE_STORE_STRIDE 搬运
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(
        static_cast<int64_t>(mSize), static_cast<int64_t>(scaleBlockN), static_cast<int64_t>(scaleBlockN));
    auto srcTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(SCALE_UB_OFFSET), srcLayout);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize),
        static_cast<int64_t>(AscendC::ONE_BLK_SIZE), static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
    auto dstTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(scaleBlockOffset), dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransScaleLayout(srcTensor, dstTensor, mSize, scaleBlockN);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::TransFp4Output(
    uint16_t mSize, uint16_t kSize, uint32_t srcOffset, uint32_t dstOffset) {
    uint32_t packedBytes = kSize / 2;
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, packedBytes, Gemm::Align16(packedBytes));
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, packedBytes, Gemm::Align32(packedBytes));
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(srcOffset), srcLayout);
    auto dst = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(dstOffset), dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransFp4OutLayout(src, dst, mSize, kSize);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::PadScale(
    uint32_t nKRatio, uint32_t blkM, uint16_t firstNum, uint16_t rowLoopNum, uint16_t lastNum) {
    if ASCEND_IS_AIC {
        return;
    }
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(1, blkM, blkM);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(1, blkM * SCALE_STORE_STRIDE, blkM * SCALE_STORE_STRIDE);
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(SCALE_UB_OFFSET), srcLayout);
    auto dst =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(SCALE_BLOCK_UB_OFFSET), dstLayout);
    asc_vf_call<PadScaleVf>(reinterpret_cast<__ubuf__ int8_t *>(src.data().get()),
        reinterpret_cast<__ubuf__ int8_t *>(dst.data().get()), nKRatio, firstNum, lastNum, rowLoopNum);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__simd_vf__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::PadScaleVf(
    __ubuf__ int8_t *scaleAddr, __ubuf__ int8_t *scaleBlkAddr, uint32_t nKRatio, uint16_t firstNum, uint16_t lastNum,
    uint16_t rowLoopNum) {
    // ClearScaleBlock provides the zero padding; each staged row starts at a 32-byte boundary.
    DataCopyByCount(scaleBlkAddr, scaleAddr, firstNum);
    scaleAddr += firstNum;
    scaleBlkAddr += (firstNum + SCALE_STORE_STRIDE) / SCALE_STORE_STRIDE * SCALE_STORE_STRIDE;

    for (uint16_t mIdx = 0; mIdx < rowLoopNum; ++mIdx) {
        DataCopyByCount(scaleBlkAddr, scaleAddr, nKRatio);
        scaleAddr += nKRatio;
        scaleBlkAddr += (nKRatio + SCALE_STORE_STRIDE) / SCALE_STORE_STRIDE * SCALE_STORE_STRIDE;
    }

    if (lastNum > 0) {
        DataCopyByCount(scaleBlkAddr, scaleAddr, lastNum);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_>
__simd_callee__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_>::DataCopyByCount(
    __ubuf__ int8_t *dstAddr, __ubuf__ int8_t *srcAddr, uint32_t count) {
    constexpr uint32_t VF_LEN_8 = asc_get_vf_len() / sizeof(uint8_t);
    uint16_t loopNum = (count + VF_LEN_8 - 1) / VF_LEN_8;
    AscendC::Reg::RegTensor<int8_t> value;
    AscendC::Reg::UnalignReg unalign;
    for (uint16_t i = 0; i < loopNum; ++i) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int8_t>(count);
        AscendC::Reg::LoadUnAlignPre(unalign, srcAddr + i * VF_LEN_8);
        AscendC::Reg::LoadUnAlign(value, unalign, srcAddr + i * VF_LEN_8);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstAddr + i * VF_LEN_8, value, mask);
    }
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
