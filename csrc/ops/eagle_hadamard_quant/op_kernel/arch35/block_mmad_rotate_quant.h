/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_rotate_quant.h
 * \brief
 */

#pragma once

#include "rotate_quant_tiling_data.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Block {
#if MINDIESD_EAGLE_PAIR_FIXPIPE
// Keep the required Fixpipe trait local: unrelated Blaze tile traits may require
// a newer CANN API even though split-M L0C-to-UB is supported by this SDK.
struct RotateQuantCopyL0C2UBSplitM {
    using TraitType = asc::te::l0c_to_ub_trait;
    static constexpr TraitType value{asc::te::round_mode::default_round, false, false, asc::te::dual_dst_mode::split_m};
};
#endif
// One L1 bank holds one NZ matrix. Its 128-row M tiles share the bank's NZ pitch.
struct RotateQuantL1Batch {
    static constexpr int64_t TILE = 128;
    static constexpr int64_t TILES = 8;
    static constexpr int64_t ROWS = TILE * TILES;
    static constexpr uint64_t BANK_BYTES = ROWS * TILE * sizeof(bfloat16_t);
    static constexpr uint16_t ROTATION_BANK = 1; // Batch zero uses bank zero.
    static constexpr uint64_t FOUR_ROTATION_BASE = BANK_BYTES + BANK_BYTES / 2;
    static constexpr uint64_t ROTATION_BYTES = TILE * TILE * sizeof(bfloat16_t);
    static constexpr uint16_t ROTATION_EVENT = 2; // X bank handshakes use 0 and 1.

    uint16_t bank{};
    int64_t rows{}; // Zero selects the original single-tile path.
    int64_t row{};

    template <int columns = 1> __aicore__ static inline RotateQuantL1Batch ForTile(int64_t tile, int64_t m) {
        if constexpr (columns == 4) {
            // The four-column scheduler only accepts full row tiles.
            return {static_cast<uint16_t>(tile & 1), TILE, 0};
        }
        constexpr int64_t tiles = TILES / columns;
        constexpr int64_t rows = tiles * TILE;
        const int64_t batch = tile / tiles;
        const int64_t remaining = m - batch * rows;
        return {static_cast<uint16_t>(batch % 2), remaining < rows ? remaining : rows, (tile % tiles) * TILE};
    }
};

template <typename Input, bool BlockDiagonal = false> class RotateQuantMmad {
  public:
    using L1Batch = RotateQuantL1Batch;
    using TupleL1L0Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

    static_assert(AscendC::Std::is_same_v<Input, half> || AscendC::Std::is_same_v<Input, bfloat16_t>,
        "RotateQuant requires FP16 or BF16 inputs.");

    __aicore__ inline void Init(
        const RotateQuantAptOpt::RotateQuantAptTilingData &tiling, bool residentRotation, bool sharedRotation) {
        b_ = tiling.B;
        residentRotation_ = residentRotation;
        sharedRotation_ = sharedRotation;
        rotationLayout_ = tiling.rotationIsNz;
        baseK_ = tiling.baseK;
        const uint64_t mL1 = tiling.mL1;
        const uint64_t nL1 = tiling.nL1;
        const uint64_t kL1 = tiling.kL1;
        uint64_t aL1OneBuffer = Gemm::CeilAlign(mL1, Gemm::BLOCK_CUBE) * Gemm::CeilAlign(kL1, Gemm::BLOCK_CUBE);
        uint64_t aL1OneBufferBytes = aL1OneBuffer * sizeof(Input);
        uint64_t rotL1OneBufferBytes =
            Gemm::CeilAlign(kL1, Gemm::BLOCK_CUBE) * Gemm::CeilAlign(nL1, Gemm::BLOCK_CUBE) * sizeof(Input);
        uint64_t rotL1Base = aL1OneBufferBytes * Gemm::DOUBLE_BUFFER_COUNT;
        for (uint32_t i = 0; i < Gemm::DOUBLE_BUFFER_COUNT; ++i) {
            aL1Buffer_[i] = aL1OneBufferBytes * i;
            rotL1Buffer_[i] = rotL1Base + rotL1OneBufferBytes * i;
        }
        l0PingPong_ = 0;
        blockPingPong_ = 0;
        subBlockId_ = 0;
    }

    template <int columns = 1, typename TensorA>
    __aicore__ inline void PrefetchBatch(const TensorA &gmA, int64_t column, int64_t firstTile, const L1Batch &batch) {
        constexpr int64_t width = columns * L1Batch::TILE;
        auto src = asc::te::squeeze<0>(
            gmA.slice(asc::te::make_coord(column, asc::te::make_coord(firstTile * L1Batch::TILE, 0L)),
                asc::te::make_shape(1L, asc::te::make_shape(batch.rows, width))));
        auto layout =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<Input>>(batch.rows, width);
        auto dst = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, Input>(batch.bank * L1Batch::BANK_BYTES), layout);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(batch.bank);
        asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_l1{}), dst, src);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(batch.bank);
    }

    // A(blkM x K) @ rot(K x K) -> L0C(float) -> Fixpipe(F322BF16) -> UB(bfloat16).
    // BlockDiagonal issues independent small MMADs and shares only the 128-column output.
    // gmA carries the row pitch in its GM layout (B > 1: b_ * k_, otherwise k_), CopyGM2L1 derives Nd2Nz from it.
    template <bool paired = false, int columns = paired ? 2 : 1, typename TensorA>
    __aicore__ inline void operator()(TensorA gmA, __gm__ Input *gmRot, TupleL1L0Shape blockShape, bool rotationChanged,
        uint32_t rotationK = 0, L1Batch batch = {}, uint32_t inputUbOffset = 0, uint16_t lane = 0,
        uint16_t panelColumn = 0, uint16_t issueColumn = 0, bool firstRow = false, uint16_t panel = 0,
        int16_t owner = -1) {
        uint64_t curML1 = asc::te::get<Gemm::MNK_M>(blockShape);
        uint64_t curNL1 = asc::te::get<Gemm::MNK_N>(blockShape);
        uint64_t curKL1 = asc::te::get<Gemm::MNK_K>(blockShape);

        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / Gemm::DOUBLE_BUFFER_COUNT;
        const uint16_t blockSlot = blockPingPong_ & 0x1;
        const bool streamRotation = b_ > 1 && !residentRotation_;
        const uint16_t rotL1BufferId = streamRotation ? blockSlot : 0;
        const uint64_t bankOffset = HALF_L0C_SIZE * blockSlot;
        // Adjacent full tiles along N share the NZ pitch of their 128x256 parent.
        const uint64_t l0cOffset = bankOffset + (paired ? lane * curML1 * curNL1 * sizeof(float) : 0);

        const uint64_t al1OffsetBytes = batch.rows ? batch.bank * L1Batch::BANK_BYTES : aL1Buffer_[blockSlot];
        const uint64_t al1Rows = batch.rows ? static_cast<uint64_t>(batch.rows) : curML1;

        auto layoutAL1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<Input>>(
            al1Rows, columns * curKL1);
        auto tensorAL1 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, Input>(al1OffsetBytes), layoutAL1);
        if (!batch.rows) {
            CopyGM2L1(tensorAL1, gmA, blockSlot + Gemm::L1_BUFFER_ID_BASE);
        }

        // Resident rotations use L1 only to stage the first load of a column.
        const uint64_t rotL1Offset = columns == 4
            ? (L1Batch::FOUR_ROTATION_BASE - panel * L1Batch::BANK_BYTES) + panelColumn * L1Batch::ROTATION_BYTES
            : batch.rows ? L1Batch::ROTATION_BANK * L1Batch::BANK_BYTES
                         : rotL1Buffer_[rotL1BufferId];
        auto layoutRotL1 =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<Input>>(curKL1, curNL1);
        auto tensorRotL1 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, Input>(rotL1Offset), layoutRotL1);
        if (columns == 4 ? firstRow : (rotationChanged || streamRotation)) {
            if constexpr (columns == 4) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1Batch::ROTATION_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1Batch::ROTATION_EVENT);
            } else if (batch.rows) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1Batch::ROTATION_BANK);
            }
            const uint16_t mutexId = rotL1BufferId + ROT_L1_BUFFER_ID_BASE;
            if constexpr (BlockDiagonal) {
                CopyDiagonalRotation(rotL1Offset, gmRot, rotationK, mutexId);
            } else if (rotationLayout_ != RotateQuantAptOpt::ROTATION_ND) {
                CopyPackedRotation(rotL1Offset, gmRot, curKL1, curNL1, mutexId);
            } else {
                auto layoutND =
                    asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<Input>>(
                        curKL1, curNL1);
                auto rotND = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(gmRot), layoutND);
                CopyGM2L1(tensorRotL1, rotND, mutexId);
            }
            if constexpr (columns == 4) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1Batch::ROTATION_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1Batch::ROTATION_EVENT);
            } else if (batch.rows) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1Batch::ROTATION_BANK);
            }
        }

        // Never multiply the structural zeros between rotations: NaN/Inf in A
        // would otherwise contaminate unrelated blocks through NaN/Inf * 0.
        const uint64_t stepK = BlockDiagonal ? rotationK : baseK_;
        uint64_t kL0Iter = Gemm::CeilDiv(curKL1, stepK);
        uint64_t curNL0 = asc::te::get<5>(blockShape); // baseN
        for (uint64_t iter = 0; iter < kL0Iter; ++iter) {
            uint64_t curK0 = (iter + 1 == kL0Iter) ? (curKL1 - iter * stepK) : stepK;
            const uint64_t sliceN = BlockDiagonal ? curK0 : curNL0;
            const uint64_t columnOffset = BlockDiagonal ? iter * stepK : 0;
            const uint16_t l0BufferId = l0PingPong_ & 0x1;

            auto l1Tensors = AscendC::Std::make_tuple(tensorAL1, tensorRotL1);
            auto l0Shape = AscendC::Std::make_tuple(curML1, sliceN, curK0, iter * stepK, columnOffset);
            auto l0Tensors = CopyL1ToL0<paired, columns>(l1Tensors, l0Shape, l0BufferId, blockSlot, rotL1BufferId,
                rotationChanged, batch, panelColumn, issueColumn, lane, !BlockDiagonal || iter == 0,
                !BlockDiagonal || iter + 1 == kL0Iter);

            // NZ packs whole 16-row slabs, including an M tail. Each small
            // MMAD initializes a disjoint N slice of the same L0C output bank.
            const uint64_t sliceOffset = columnOffset * Gemm::CeilAlign(curML1, Gemm::BLOCK_CUBE) * sizeof(float);
            auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(
                curML1, sliceN);
            auto tensorL0C = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset + sliceOffset), layoutL0C);
            auto mnkShape = AscendC::Std::make_tuple(curML1, sliceN, curK0);
            Mmad(tensorL0C, l0Tensors, mnkShape, asc::te::unit_flag_mode::disable, BlockDiagonal || iter == 0,
                l0BufferId, blockSlot);
            l0PingPong_++;
        }

        // Fixpipe reads the same L0C buffer that Mmad wrote. Explicit M/FIX locks order
        // producer/consumer access and buffer reuse; the other L0C buffer permits overlap.
        // Cross-core PIPE_FIX waits protect the destination AIV UB, not the M pipeline.
        if (owner >= 0) {
            auto srcLayout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(
                curML1, curNL1);
            auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset), srcLayout);
            auto dstLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(curML1, curNL1);
            auto dst = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(inputUbOffset), dstLayout);
            Gemm::LockFix(blockSlot + Gemm::L0C_BUFFER_ID_BASE);
            auto copy = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
            asc::te::l0c_to_ub_params params{asc::te::unit_flag_mode::disable, static_cast<uint8_t>(owner)};
            asc::te::copy(copy.with(params), dst, src);
            Gemm::UnLockFix(blockSlot + Gemm::L0C_BUFFER_ID_BASE);
            if (lane == 1) {
                ++blockPingPong_;
            }
            return;
        }
        if constexpr (paired) {
            if (lane == 0) {
                return;
            }
        }
        const uint64_t outN = (paired ? 2 : 1) * curNL0;
        uint64_t nAlign = Gemm::CeilAlign(outN, Gemm::BLOCK_CUBE);
        auto layoutL0COut =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(curML1, outN);
        auto tensorL0COut =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(bankOffset), layoutL0COut);
        auto layoutUB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(curML1, nAlign);
        // DAV-3510 dual destinations require NoQuant. Each AIV narrows its FP32 half locally.
        using FixOutput = AscendC::Std::conditional_t<paired, float, bfloat16_t>;
        auto tensorUB =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, FixOutput>(inputUbOffset), layoutUB);
        auto ubTensors = AscendC::Std::make_tuple(tensorUB, tensorL0COut);
        FixpipeL0CToUB<paired>(ubTensors, blockSlot);
        ++blockPingPong_;
    }

  private:
    static constexpr uint16_t ROT_L1_BUFFER_ID_BASE = Gemm::L1_BUFFER_ID_BASE + Gemm::DOUBLE_BUFFER_COUNT;

    // Stage GM [128/k,k,k] on the diagonal of NZ [128,128] in L1.
    // Only diagonal slices are loaded into L0B; the gaps are never read.
    // NZ(r,c) = (c/16)*128*16 + r*16 + c%16 for a 16-bit element.
    __aicore__ inline void CopyDiagonalRotation(uint64_t l1Offset, __gm__ Input *gm, uint32_t k, uint16_t mutexId) {
        constexpr uint16_t width = 128;
        constexpr uint16_t c0 = 32 / sizeof(Input);
        AscendC::LocalTensor<Input> dst(AscendC::TPosition::B1, l1Offset, width * width);
        AscendC::GlobalTensor<Input> src;
        src.SetGlobalBuffer(gm);
        Gemm::LockMte2(mutexId);
        if (rotationLayout_ == RotateQuantAptOpt::ROTATION_ND) {
            AscendC::Nd2NzParams copy{};
            copy.ndNum = width / k;
            copy.nValue = copy.dValue = copy.srcDValue = k;
            copy.srcNdMatrixStride = k * k;
            copy.dstNzC0Stride = width;
            copy.dstNzNStride = 1;
            copy.dstNzMatrixStride = k * (width + c0);
            AscendC::DataCopy(dst, src, copy);
        } else {
            // Compact NZ rotations retain their original k-row pitch in GM.
            // Scatter only real diagonal slices into the 128-row L1 parent.
            for (uint16_t matrix = 0; matrix < width / k; ++matrix) {
                const uint32_t target = matrix * k * (width + c0);
                const uint32_t source = matrix * k * k;
                if (rotationLayout_ == RotateQuantAptOpt::ROTATION_L1_NZ) {
                    AscendC::DataCopyParams copy{
                        static_cast<uint16_t>(k / c0), static_cast<uint16_t>(k), 0, static_cast<uint16_t>(width - k)};
                    AscendC::DataCopy(dst[target], src[source], copy);
                } else {
                    constexpr uint16_t tileBursts = c0 * c0 * sizeof(Input) / 32;
                    AscendC::DataCopyParams copy{
                        static_cast<uint16_t>(k / c0), tileBursts, static_cast<uint16_t>((k / c0 - 1) * tileBursts), 0};
                    for (uint16_t col = 0; col < k / c0; ++col) {
                        AscendC::DataCopy(dst[target + col * width * c0], src[source + col * c0 * c0], copy);
                    }
                }
            }
        }
        Gemm::UnLockMte2(mutexId);
    }

    // L1 order [n1,k1,k0,n0] is contiguous. Legacy [k1,n1,k0,n0] needs
    // one gather per tile column, with strides measured in 32-byte bursts.
    __aicore__ inline void CopyPackedRotation(
        uint64_t l1Offset, __gm__ Input *gm, uint64_t k, uint64_t n, uint16_t mutexId) {
        constexpr uint32_t tile = 16;
        constexpr uint16_t burstsPerTile = tile * tile * sizeof(Input) / 32;
        const uint16_t kTiles = k / tile;
        const uint16_t nTiles = n / tile;
        AscendC::GlobalTensor<Input> src;
        src.SetGlobalBuffer(gm);
        AscendC::LocalTensor<Input> dst(AscendC::TPosition::B1, l1Offset, k * n);
        Gemm::LockMte2(mutexId);
        if (rotationLayout_ == RotateQuantAptOpt::ROTATION_L1_NZ) {
            AscendC::DataCopy(dst, src, static_cast<uint32_t>(k * n));
        } else {
            AscendC::DataCopyParams params{
                kTiles, burstsPerTile, static_cast<uint16_t>((nTiles - 1) * burstsPerTile), 0};
            for (uint16_t col = 0; col < nTiles; ++col) {
                AscendC::DataCopy(dst[col * kTiles * tile * tile], src[col * tile * tile], params);
            }
        }
        Gemm::UnLockMte2(mutexId);
    }

    template <typename TensorDst, typename TensorSrc>
    __aicore__ inline void CopyGM2L1(TensorDst &dst, const TensorSrc &src, uint16_t bufferId) {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        Gemm::LockMte2(bufferId);
        asc::te::copy(copyGM2L1, dst, src);
        Gemm::UnLockMte2(bufferId);
    }

    template <bool paired, int columns, typename L1Tensors, typename L0Shape>
    __aicore__ inline auto CopyL1ToL0(const L1Tensors &l1Tensors, const L0Shape &l0Shape, uint16_t l0BufferId,
        uint16_t aL1BufferId, uint16_t rotL1BufferId, bool rotationChanged, const L1Batch &batch, uint16_t panelColumn,
        uint16_t issueColumn, uint16_t lane, bool firstSlice, bool lastSlice) {
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / Gemm::DOUBLE_BUFFER_COUNT;
        uint64_t l0Offset = HALF_L0_SIZE * l0BufferId;
        const auto &tensorAL1 = asc::te::get<0>(l1Tensors);
        const auto &tensorBL1 = asc::te::get<1>(l1Tensors);
        auto curML0 = asc::te::get<0>(l0Shape);
        auto curNL0 = asc::te::get<1>(l0Shape);
        auto curK0 = asc::te::get<2>(l0Shape);
        auto kOffset = asc::te::get<3>(l0Shape);
        auto bColumnOffset = asc::te::get<4>(l0Shape);

        auto layoutL0A =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<Input>>(curML0, curK0);
        auto tensorL0A =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, Input>(l0Offset), layoutL0A);
        auto tensorBlockAL1 = tensorAL1.slice(
            asc::te::make_coord(batch.row, kOffset + panelColumn * L1Batch::TILE), asc::te::make_shape(curML0, curK0));
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        if (!batch.rows && firstSlice) {
            Gemm::LockMte1(aL1BufferId + Gemm::L1_BUFFER_ID_BASE);
        } else if (batch.rows && batch.row == 0 && issueColumn == 0 && firstSlice) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(batch.bank);
        }
        Gemm::LockMte1(l0BufferId + Gemm::L0_BUFFER_ID_BASE);
        asc::te::copy(copyL12L0A, tensorL0A, tensorBlockAL1);
        if (!batch.rows && lastSlice) {
            Gemm::UnLockMte1(aL1BufferId + Gemm::L1_BUFFER_ID_BASE);
        } else if (batch.rows && batch.row + static_cast<int64_t>(curML0) == batch.rows && issueColumn + 1 == columns &&
            lastSlice) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(batch.bank);
        }

        auto layoutL0B =
            asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<Input>>(curK0, curNL0);
        // Each paired column keeps B under the same L0 mutex as its A operand.
        // Compact resident rotations occupy separate L0B ranges (8/16 KiB total
        // for K=32/64), so later row tiles can reuse every diagonal slice.
        const uint64_t residentOffset = BlockDiagonal ? bColumnOffset * curK0 * sizeof(Input) : 0;
        const uint64_t bOffset = residentRotation_ && BlockDiagonal
            ? (paired ? lane * (AscendC::TOTAL_L0B_SIZE / 2) : 0) + residentOffset
            : (residentRotation_ && !paired ? residentOffset : l0Offset);
        auto tensorL0B = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, Input>(bOffset), layoutL0B);
        if (!residentRotation_ || rotationChanged) {
            // Dense resident B is read under either A slot. Compact diagonal
            // slices instead keep a fixed A-slot parity: each column issues
            // four K=32 or two K=64 MMADs, and every slice has its own B range.
            // The current A lock therefore also protects its compact B slice.
            const uint16_t otherSlotMutex = (l0BufferId ^ 1U) + Gemm::L0_BUFFER_ID_BASE;
            if (residentRotation_ && !paired && !BlockDiagonal) {
                Gemm::LockMte1(otherSlotMutex);
            }
            auto tensorBlockBL1 =
                tensorBL1.slice(asc::te::make_coord(kOffset, bColumnOffset), asc::te::make_shape(curK0, curNL0));
            auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
            if (columns != 4 && batch.rows && firstSlice) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1Batch::ROTATION_BANK);
            }
            if (firstSlice) {
                Gemm::LockMte1(rotL1BufferId + ROT_L1_BUFFER_ID_BASE);
            }
            // A shared full tile stays at the same address in each of the two L0B slots.
            if (!sharedRotation_ || !rotationLoaded_[l0BufferId]) {
                asc::te::copy(copyL12L0B, tensorL0B, tensorBlockBL1);
                if (sharedRotation_) {
                    rotationLoaded_[l0BufferId] = true;
                }
            }
            if (lastSlice) {
                Gemm::UnLockMte1(rotL1BufferId + ROT_L1_BUFFER_ID_BASE);
            }
            if (columns != 4 && batch.rows && lastSlice) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1Batch::ROTATION_BANK);
            }
            if (residentRotation_ && !paired && !BlockDiagonal) {
                Gemm::UnLockMte1(otherSlotMutex);
            }
        }
        Gemm::UnLockMte1(l0BufferId + Gemm::L0_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorL0A, tensorL0B);
    }

    template <typename TensorL0C, typename L0Tensors, typename MnkShape>
    __aicore__ inline void Mmad(TensorL0C &tensorL0C, const L0Tensors &l0Tensors, const MnkShape &mnkShape,
        asc::te::unit_flag_mode unitFlag, bool cmatrixInitVal, uint16_t l0BufferId, uint16_t l0cBufferId) {
        const auto &tensorL0A = asc::te::get<0>(l0Tensors);
        const auto &tensorL0B = asc::te::get<1>(l0Tensors);
        auto curM = asc::te::get<0>(mnkShape);
        auto curN = asc::te::get<1>(mnkShape);
        auto curK = asc::te::get<2>(mnkShape);

        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
            static_cast<uint16_t>(curK), unitFlag, cmatrixInitVal};
        Gemm::LockM(l0BufferId + Gemm::L0_BUFFER_ID_BASE);
        Gemm::LockM(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
        asc::te::mmad(asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation>>{}.with(mmadParams), tensorL0C,
            tensorL0A, tensorL0B);
        Gemm::UnLockM(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
        Gemm::UnLockM(l0BufferId + Gemm::L0_BUFFER_ID_BASE);
    }

    template <bool paired, typename TensorsTuple>
    __aicore__ inline void FixpipeL0CToUB(const TensorsTuple &tensors, uint16_t l0cBufferId) {
        const auto &dst = asc::te::get<0>(tensors);
        const auto &src = asc::te::get<1>(tensors);

        Gemm::LockFix(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
#if MINDIESD_EAGLE_PAIR_FIXPIPE
        if constexpr (paired) {
            auto copy = asc::te::make_copy(asc::te::copy_l0c_to_ub{}, RotateQuantCopyL0C2UBSplitM{});
            asc::te::copy(copy.with(asc::te::l0c_to_ub_params{}), dst, src);
        } else
#endif
        {
            asc::te::l0c_to_ub_params params{
                asc::te::unit_flag_mode::disable, static_cast<uint8_t>((subBlockId_++) & 1)};
            auto copy = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
            asc::te::copy(copy.with(params), dst, src);
        }
        Gemm::UnLockFix(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
    }

    uint8_t rotationLayout_{RotateQuantAptOpt::ROTATION_ND};
    bool residentRotation_{false};
    bool sharedRotation_{false};
    bool rotationLoaded_[2]{};
    uint64_t b_{1};
    uint64_t baseK_{16};
    uint64_t l0PingPong_{1};
    uint64_t blockPingPong_{0};
    uint64_t subBlockId_{0};

    uint64_t aL1Buffer_[Gemm::DOUBLE_BUFFER_COUNT] = {0};
    uint64_t rotL1Buffer_[Gemm::DOUBLE_BUFFER_COUNT] = {0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
