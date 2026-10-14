/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef SPARSE_BLOCK_ESTIMATE_ARCH35_QK_H
#define SPARSE_BLOCK_ESTIMATE_ARCH35_QK_H

#include "kernel_operator.h"
// Included by sparse_block_estimate_kernel.h after its layout constants.

namespace SparseEstimateArch35 {

constexpr AscendC::FixpipeConfig QK_FIX_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

// Only the two supported Q group sizes share this implementation. Addresses
// are bytes; the SDK tensor sizes are elements. Keep the M64 stage spacing.
template <typename InputT, uint32_t GroupTiles> class PackedQK {
    static_assert(GroupTiles == 2U || GroupTiles == 4U, "Only two/four Q tiles are supported");
    static constexpr uint32_t M = GroupTiles * Q_PACKED_TILE;
    static constexpr uint32_t Q_BYTES = M * PACKED_DIM * sizeof(InputT);
    static constexpr uint32_t K_BYTES = K_PACKED_TILE * PACKED_DIM * sizeof(InputT);
    static constexpr uint32_t L0_AB_BYTES = 32U * 1024U;
    static constexpr uint32_t L0_C_BYTES = 64U * 1024U;

  public:
    __aicore__ inline PackedQK() {
        qL1_ = AscendC::LocalTensor<InputT>(AscendC::TPosition::A1, 0, M * PACKED_DIM);
        for (uint32_t i = 0; i < 2U; ++i) {
            kL1_[i] =
                AscendC::LocalTensor<InputT>(AscendC::TPosition::A1, Q_BYTES + i * K_BYTES, K_BYTES / sizeof(InputT));
            aL0_[i] =
                AscendC::LocalTensor<InputT>(AscendC::TPosition::A2, i * L0_AB_BYTES, L0_AB_BYTES / sizeof(InputT));
            bL0_[i] =
                AscendC::LocalTensor<InputT>(AscendC::TPosition::B2, i * L0_AB_BYTES, L0_AB_BYTES / sizeof(InputT));
            cL0_[i] = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, i * L0_C_BYTES, L0_C_BYTES / sizeof(float));
        }
    }

    // SparseBlockEstimate packs `stride` adjacent raw Q rows into one MMAD row.
    // Each positive-stride copy writes one lane directly to its final L1 NZ
    // coordinates, so the legacy GM -> UB -> GM reorder workspace is avoided.
    __aicore__ inline void loadQPackedQGM(AscendC::GlobalTensor<InputT> gATensor, uint32_t packedRows,
        uint32_t validPackedRows, uint32_t dim, uint32_t stride) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        if (validPackedRows < packedRows) {
            AscendC::InitConstValueParams<half> zeroParams(1, packedRows * dim * stride * sizeof(InputT) / 32U, 0, 0);
            AscendC::Fill(qL1_.template ReinterpretCast<half>(), zeroParams);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::Nd2NzParams copyParams;
        copyParams.ndNum = 1U;
        copyParams.nValue = validPackedRows;
        copyParams.dValue = dim;
        copyParams.srcNdMatrixStride = 0U;
        copyParams.srcDValue = dim * stride;
        copyParams.dstNzC0Stride = packedRows;
        copyParams.dstNzNStride = 1U;
        copyParams.dstNzMatrixStride = 0U;
        for (uint32_t dstLane = 0; dstLane < stride; ++dstLane) {
            uint32_t srcLane = stride - 1U - dstLane;
            uint32_t dstOffset = dstLane * packedRows * dim;
            uint32_t srcOffset = srcLane * dim;
            AscendC::DataCopy(qL1_[dstOffset], gATensor[srcOffset], copyParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    // Heads may end anywhere within a Q256 tile. Place each
    // tile at its padded L1 row offset while keeping GM heads unpadded.
    __aicore__ inline void loadQPackedQGMSegmented(AscendC::GlobalTensor<InputT> gATensor, uint32_t packedRows,
        uint32_t dim, uint32_t stride, uint32_t rawRowsPerHead, uint32_t firstFlatTile, uint32_t tilePackedRows,
        uint32_t validTiles) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::InitConstValueParams<half> zeroParams(1, packedRows * dim * stride * sizeof(InputT) / 32U, 0, 0);
        AscendC::Fill(qL1_.template ReinterpretCast<half>(), zeroParams);
        AscendC::PipeBarrier<PIPE_MTE2>();

        uint32_t tileRawRows = tilePackedRows * stride;
        uint32_t tilesPerHead = (rawRowsPerHead + tileRawRows - 1U) / tileRawRows;
        AscendC::Nd2NzParams copyParams;
        copyParams.ndNum = 1U;
        copyParams.dValue = dim;
        copyParams.srcNdMatrixStride = 0U;
        copyParams.srcDValue = dim * stride;
        copyParams.dstNzC0Stride = packedRows;
        copyParams.dstNzNStride = 1U;
        copyParams.dstNzMatrixStride = 0U;
        for (uint32_t tile = 0; tile < validTiles; ++tile) {
            uint32_t flatTile = firstFlatTile + tile;
            uint32_t head = flatTile / tilesPerHead;
            uint32_t rowInHead = (flatTile % tilesPerHead) * tileRawRows;
            uint32_t validRawRows = min(tileRawRows, rawRowsPerHead - rowInHead);
            uint64_t rawRow = static_cast<uint64_t>(head) * rawRowsPerHead + rowInHead;
            for (uint32_t dstLane = 0; dstLane < stride; ++dstLane) {
                uint32_t srcLane = stride - 1U - dstLane;
                // Each lane may have a different row count in the final pack.
                // The buffer was cleared above; never read beyond this head.
                if (srcLane >= validRawRows) {
                    continue;
                }
                copyParams.nValue = (validRawRows - srcLane + stride - 1U) / stride;
                // NZ row displacement is one C0, not one full embedding.
                uint32_t dstOffset = dstLane * packedRows * dim + tile * tilePackedRows * (32U / sizeof(InputT));
                uint64_t srcOffset = (rawRow + srcLane) * dim;
                AscendC::DataCopy(qL1_[dstOffset], gATensor[srcOffset], copyParams);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    // tile is relative to the current K segment. Its last tile must not
    // prefetch into the next segment or the next Q task.
    __aicore__ inline void Compute(AscendC::GlobalTensor<InputT> key, AscendC::LocalTensor<float> score, uint32_t tile,
        uint32_t rawTokens, uint32_t nAct, uint16_t scoreFlag) {
        uint32_t slot = tile & 1U;
        uint32_t kEvent = slot + 1U;
        if (tile == 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEvent);
            LoadK(key, slot, 0U, rawTokens, nAct);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvent);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvent);
        uint32_t packedCols = (rawTokens + PACK_STRIDE - 1U) / PACK_STRIDE;
        uint32_t nextStart = (tile + 1U) * K_PACKED_TILE;
        if (nextStart < packedCols) {
            uint32_t nextSlot = slot ^ 1U;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(nextSlot + 1U);
            LoadK(key, nextSlot, nextStart, rawTokens, min(K_PACKED_TILE, packedCols - nextStart));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(nextSlot + 1U);
        }

        uint32_t nFractals = (nAct + 15U) / 16U;
        for (uint32_t slice = 0; slice < K_L0_SLICES; ++slice) {
            // Each preceding tile consumed eight stages, so both A/B stage
            // counters have even bases, including at every 64K boundary.
            uint32_t abSlot = slice & 1U;
            uint32_t bEvent = abSlot + 2U;
            AscendC::LoadData2DParamsV2 load;
            load.mStartPosition = 0;
            load.kStartPosition = slice * (L0_K_TILE / 16U);
            load.mStep = nFractals;
            load.kStep = L0_K_TILE / 16U;
            load.srcStride = nFractals;
            load.dstStride = nFractals;
            load.ifTranspose = false;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bEvent);
            AscendC::LoadData(bL0_[abSlot], kL1_[slot], load);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bEvent);
            if (slice + 1U == K_L0_SLICES) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEvent);
            }
            load.mStep = M / 16U;
            load.srcStride = M / 16U;
            load.dstStride = M / 16U;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(abSlot);
            AscendC::LoadData(aL0_[abSlot], qL1_, load);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(abSlot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(abSlot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bEvent);
            if (slice == 0U) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
            }
            AscendC::MmadParams mm;
            mm.m = M;
            mm.n = nAct;
            mm.k = L0_K_TILE;
            mm.cmatrixInitVal = slice == 0U;
            mm.disableGemv = true;
            mm.unitFlag = 0;
            AscendC::Mmad(cL0_[slot], aL0_[abSlot], bL0_[abSlot], mm);
            // Preserve integer division: rounding N up changes the barrier.
            if ((M / 16U) * (nAct / 16U) < 10U) {
                AscendC::PipeBarrier<PIPE_M>();
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(abSlot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bEvent);
        }
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(scoreFlag);
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(scoreFlag + 16U);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(slot);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fix;
        fix.mSize = M / 2U;
        fix.nSize = nFractals * 16U;
        fix.srcStride = M;
        fix.dstStride = K_PACKED_TILE;
        fix.quantPre = ::QuantMode_t::NoQuant;
        fix.reluEn = false;
        fix.unitFlag = 0;
        fix.dualDstCtl = 0;
        fix.subBlockId = false;
        AscendC::Fixpipe<float, float, QK_FIX_UB>(score, cL0_[slot], fix);
        fix.subBlockId = true;
        AscendC::Fixpipe<float, float, QK_FIX_UB>(score, cL0_[slot][(M / 2U) * 16U], fix);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(scoreFlag);
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(scoreFlag + 16U);
    }

  private:
    __aicore__ inline void LoadK(
        AscendC::GlobalTensor<InputT> key, uint32_t slot, uint32_t start, uint32_t rawTokens, uint32_t nAct) {
        uint32_t fullCols = rawTokens / PACK_STRIDE;
        uint32_t tailTokens = rawTokens % PACK_STRIDE;
        bool containsTail = tailTokens != 0U && start <= fullCols && fullCols < start + nAct;
        uint32_t alignedN = (nAct + 15U) / 16U * 16U;
        if (containsTail) {
            AscendC::InitConstValueParams<half> zero(1, PACKED_DIM * alignedN * sizeof(InputT) / 32U, 0, 0);
            AscendC::Fill(kL1_[slot].template ReinterpretCast<half>(), zero);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.dValue = PACKED_DIM;
        copy.srcNdMatrixStride = 0U;
        copy.srcDValue = PACKED_DIM;
        copy.dstNzC0Stride = alignedN;
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        if (start < fullCols) {
            copy.nValue = min(nAct, fullCols - start);
            AscendC::DataCopy(kL1_[slot], key[static_cast<uint64_t>(start) * PACKED_DIM], copy);
        }
        if (containsTail) {
            copy.nValue = 1U;
            copy.dValue = tailTokens * HEAD_DIM;
            // nZ column offset is 16 elements; keep the full tile's stride
            // for both copies, including a tail that crosses a C0 boundary.
            AscendC::DataCopy(
                kL1_[slot][(fullCols - start) * 16U], key[static_cast<uint64_t>(fullCols) * PACKED_DIM], copy);
        }
    }

    AscendC::LocalTensor<InputT> qL1_;
    AscendC::LocalTensor<InputT> kL1_[2];
    AscendC::LocalTensor<InputT> aL0_[2];
    AscendC::LocalTensor<InputT> bL0_[2];
    AscendC::LocalTensor<float> cL0_[2];
};

} // namespace SparseEstimateArch35
#endif // SPARSE_BLOCK_ESTIMATE_ARCH35_QK_H
