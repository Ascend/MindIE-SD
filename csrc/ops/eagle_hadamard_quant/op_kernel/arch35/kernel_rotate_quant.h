/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_rotate_quant.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"
#include "basic_api/kernel_operator_scalar_intf.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "rotate_quant_tiling_data.h"
// DAV-3510 full-tile paths share batching and the paired/panel scheduler.
// Explicit zero definitions retain the corresponding fallback for comparisons.
#ifndef MINDIESD_EAGLE_BATCH_L1
#define MINDIESD_EAGLE_BATCH_L1 1
#endif
#ifndef MINDIESD_EAGLE_PAIR_FIXPIPE
#define MINDIESD_EAGLE_PAIR_FIXPIPE 1
#endif
#ifndef MINDIESD_EAGLE_FOUR_COLUMNS
#define MINDIESD_EAGLE_FOUR_COLUMNS 1
#endif
#include "block_mmad_rotate_quant.h"
#include "blaze/gemm/block/block_scheduler_rotate_quant.h"
#include "block_epilogue_rotate_quant.h"
#include "tensor_api/tensor.h"

#ifndef MINDIESD_EAGLE_REUSE_L0B
#define MINDIESD_EAGLE_REUSE_L0B 1
#endif

namespace Blaze {
namespace Gemm {
namespace Kernel {

// Whole waves keep their columns. The last incomplete wave is partitioned in
// column-major tile order, so two neighboring cores may share a rotation.
struct RotateQuantCoreWork {
    int64_t core, cores, columns, tilesPerColumn, wholeRounds, tailBegin, rounds;

    __aicore__ static inline RotateQuantCoreWork Make(
        int64_t core, int64_t cores, int64_t columns, int64_t blocks, bool resident) {
        if (!resident) {
            const int64_t rounds = core < blocks ? (blocks - 1 - core) / cores + 1 : 0;
            return {core, cores, 0, 0, 0, 0, rounds};
        }
        const int64_t rows = blocks / columns;
        const int64_t whole = (columns / cores) * rows;
        const int64_t tail = (columns % cores) * rows;
        const int64_t begin = core * tail / cores;
        const int64_t end = (core + 1) * tail / cores;
        return {core, cores, columns, rows, whole, begin, whole + end - begin};
    }

    struct Span {
        int64_t block, firstRowTile, tiles;
    };

    // Only called at a span boundary, with round < rounds.
    __aicore__ inline Span GetSpan(int64_t round) const {
        if (columns == 0) {
            return {core, 0, rounds};
        }
        if (round < wholeRounds) {
            return {core + (round / tilesPerColumn) * cores, 0, tilesPerColumn};
        }
        const int64_t tile = tailBegin + (round - wholeRounds);
        const int64_t column = columns - columns % cores + tile / tilesPerColumn;
        const int64_t row = tile % tilesPerColumn;
        const int64_t left = tilesPerColumn - row;
        return {row * columns + column, row, left < rounds - round ? left : rounds - round};
    }
};

template <typename Input, typename Output, typename Scale, bool Grouped, bool BlockDiagonal = false>
class RotateQuantKernel {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Block::BlockSchedulerRotateQuant<ProblemShape, Grouped>;
    using L1Batch = Block::RotateQuantL1Batch;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<bfloat16_t, Output, Scale>;

  public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR y, GM_ADDR scale,
        const RotateQuantAptOpt::RotateQuantAptTilingData &t, GM_ADDR groups = nullptr, uint32_t rotationK = 0) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if constexpr (!Grouped) {
            // Both halves of a pair must be full. Partial rows/columns use the existing tile scheduler.
            if (CanUsePairedTiles(t)) {
                if (CanUseRingTiles(t)) {
                    RunTiles<true, 4, true>(x, rot, alpha, y, scale, t, groups, rotationK);
                    return;
                }
#if MINDIESD_EAGLE_PAIR_FIXPIPE && MINDIESD_EAGLE_BATCH_L1
#if MINDIESD_EAGLE_FOUR_COLUMNS
                if constexpr (AscendC::IsSameType<Output, fp4x2_e2m1_t>::value) {
                    if (t.B % 4 == 0) {
                        RunTiles<true, 4>(x, rot, alpha, y, scale, t, groups, rotationK);
                        return;
                    }
                }
#endif
                RunTiles<true>(x, rot, alpha, y, scale, t, groups, rotationK);
                return;
#endif
            }
        }
#endif
        RunTiles<false>(x, rot, alpha, y, scale, t, groups, rotationK);
    }

  private:
    __aicore__ static inline bool CanUsePairedTiles(const RotateQuantAptOpt::RotateQuantAptTilingData &t) {
        return MINDIESD_EAGLE_REUSE_L0B && t.B >= 2 && t.B % 2 == 0 && t.K == 128 &&
            t.N == static_cast<int64_t>(t.B) * t.K && t.mL1 == 128 && t.M > t.mL1 && t.M % t.mL1 == 0 && t.kL1 == 128 &&
            t.nL1 == 128 && t.baseK == 128 && t.baseN == 128 && AscendC::TOTAL_L1_SIZE >= 2 * L1Batch::BANK_BYTES &&
            AscendC::TOTAL_L0C_SIZE >= 256 * 1024 && AscendC::TOTAL_L0B_SIZE >= 64 * 1024 &&
            BlockEpilogue::PairInputOffset(1) <= AscendC::TOTAL_UB_SIZE;
    }

    // A ring owns eight full 128-column tiles and flushes each span's last row.
    // Called only after CanUsePairedTiles checks full tiles and on-chip capacity.
    __aicore__ static inline bool CanUseRingTiles(const RotateQuantAptOpt::RotateQuantAptTilingData &t) {
        return MINDIESD_EAGLE_BATCH_L1 && MINDIESD_EAGLE_PAIR_FIXPIPE && MINDIESD_EAGLE_FOUR_COLUMNS && t.B % 8 == 0 &&
            t.M >= 256 && !t.hasAlpha && AscendC::IsSameType<Input, bfloat16_t>::value &&
            AscendC::TOTAL_UB_SIZE >= BlockEpilogue::RING_END;
    }

    template <bool paired, int columns = paired ? 2 : 1, bool ring = false>
    __aicore__ inline void RunTiles(GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR y, GM_ADDR scale,
        const RotateQuantAptOpt::RotateQuantAptTilingData &t, GM_ADDR groups, uint32_t rotationK) {
        static_assert(!BlockDiagonal || !Grouped);
        Block::RotateQuantMmad<Input, BlockDiagonal> blockMmadOp;
        BlockEpilogue epilogueOp;
        int64_t curBlockIdx = Gemm::GetCurrentBlockIdx();
        int64_t coreNums = AscendC::GetBlockNum();
        if (coreNums <= 0) {
            return;
        }
        const ProblemShape shape{t.M, t.N, t.K, t.B};
        typename BlockScheduler::Params schedulerParams{t.mL1, t.nL1, t.kL1, t.baseM, t.baseN, t.baseK, t.nKRatio,
            t.tailML1, t.dstTypeMax, t.invDstTypeMax, t.axis, t.roundMode, t.scaleAlg, t.hasAlpha, t.trans};
        typename BlockScheduler::GroupParams groupParams{groups, t.groupNum, t.groupListType};
        BlockScheduler bs(shape, coreNums, schedulerParams, groupParams);
        int64_t blockNums = bs.GetBlockNums();

        // A complete 128x128 BF16/FP16 rotation fits in one 32 KiB L0B slot.
        const bool residentRotation = MINDIESD_EAGLE_REUSE_L0B && !Grouped && t.B > 1 && t.M > t.mL1 && t.K == 128 &&
            t.kL1 == t.K && t.nL1 == t.K && t.baseK == t.K && t.baseN == t.K;
        const bool batchX = (ring || MINDIESD_EAGLE_BATCH_L1) && residentRotation && t.mL1 == L1Batch::TILE &&
            AscendC::TOTAL_L1_SIZE >= 2 * L1Batch::BANK_BYTES;
        constexpr int issuedColumns = paired ? 2 : 1;
        constexpr int pairsPerRow = columns / issuedColumns;
        constexpr int panels = ring ? 2 : 1;
        const auto work = RotateQuantCoreWork::Make(
            curBlockIdx, coreNums, t.B / (columns * panels), blockNums / (columns * panels), residentRotation);
        const int64_t localRounds = work.rounds * pairsPerRow * panels;
        if (localRounds == 0) {
            return;
        }

        const bool sharedRotation = MINDIESD_EAGLE_REUSE_L0B && !Grouped && !BlockDiagonal && !paired && t.B == 1 &&
            t.K == 128 && t.kL1 == t.K && t.nL1 == t.K && t.baseK == t.K && t.baseN == t.K &&
            AscendC::TOTAL_L0B_SIZE >= 64 * 1024;
        blockMmadOp.Init(t, residentRotation, sharedRotation);

        int64_t oriOverflowMode =
            AscendC::GetCtrlSpr<ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL, ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL>();
        if ASCEND_IS_AIV {
            AscendC::SetCtrlSpr<ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL, ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL>(0);
        }

        bfloat16_t alphaValue = 0;
        bool needClamp = false;
        if (t.hasAlpha) {
            auto layout = asc::te::make_layout(asc::te::make_shape(1L), asc::te::make_shape(1L));
            auto gmAlpha = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t *>(alpha)), layout);
            alphaValue = gmAlpha[0];
            float value = AscendC::Cast(alphaValue);
            needClamp = value > 0.0f && value < 1.0f;
        }
        epilogueOp.Init(
            {y, scale, shape, alphaValue, needClamp, t.axis, t.roundMode, t.scaleAlg, t.dstTypeMax, t.invDstTypeMax});
        auto gmA =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ Input *>(x)),
                MakeLayoutAGm(t.M, t.N, columns * t.K, t.B / columns));
        auto *rotation = reinterpret_cast<__gm__ Input *>(rot);
        int64_t loadedRotationOffset = -1;
        const int64_t blockStride = residentRotation ? t.B : coreNums;
        const bool releaseBeforeWriteback =
            !Grouped && t.B > 1 && t.mL1 <= 128 && static_cast<uint64_t>(t.mL1) * t.K <= 16384;
        const bool doubleBufferOutputs = columns != 4 && releaseBeforeWriteback && localRounds > 2 &&
            BlockEpilogue::CanDoubleBufferOutputs(t.mL1, t.K);
        const bool doubleBufferInputs = releaseBeforeWriteback && localRounds > 2 &&
            (paired ? BlockEpilogue::PairInputOffset(2) <= AscendC::TOTAL_UB_SIZE
                    : BlockEpilogue::CanDoubleBufferInputs(t.mL1, t.K));
        const int64_t inputReuseDistance = (doubleBufferInputs ? 4 : 2) / issuedColumns;
        RotateQuantCoreWork::Span span{};
        int64_t blockIndex = 0;
        int64_t spanEnd = 0;
        int64_t spanRows = 0;
        int64_t rowTile = 0;
        if ASCEND_IS_AIC {
            if (batchX) {
                // One writable credit per physical L1 bank, carried across column changes.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            }
        }
        if ASCEND_IS_AIV {
            if constexpr (ring) {
                for (uint16_t g = 0; g < BlockEpilogue::RING_GROUPS; ++g) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(g);
                }
            } else if constexpr (columns == 4) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            } else if (doubleBufferOutputs) {
                // One writable credit per output slot. MTE3 returns it after both GM stores.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            }
        }
        for (int64_t roundIdx = 0; roundIdx < localRounds; ++roundIdx) {
            if (roundIdx == spanEnd) {
                span = work.GetSpan(roundIdx / (pairsPerRow * panels));
                spanEnd += span.tiles * pairsPerRow * panels;
                rowTile = 0;
                blockIndex = span.block * columns * panels;
                if (batchX) {
                    const int64_t available = t.M - span.firstRowTile * L1Batch::TILE;
                    const int64_t capacity = span.tiles * L1Batch::TILE;
                    spanRows = available < capacity ? available : capacity;
                }
            }
            const uint16_t pair = roundIdx % pairsPerRow;
            // Reverse the pair order on alternate row tiles so their common pair stays in L0B.
            const uint16_t pairColumn = columns == 4 ? (pair ^ (rowTile & 1)) * issuedColumns : 0;

            const uint16_t issuePanel = (roundIdx / pairsPerRow) % panels;
            const uint16_t panel = ring ? issuePanel ^ (rowTile & 1) : 0;
            auto blockInfo = bs.GetBlockInfo(blockIndex + panel * columns + pairColumn);
            auto blockShape = blockInfo.blockShape;
            auto blockCoord = blockInfo.blockCoord;
            // A paired round feeds both AIVs. A single-column round feeds one alternating AIV.
            const uint16_t localSlot = (paired ? roundIdx : roundIdx / 2) & 1;
            const uint16_t inputSlot = doubleBufferInputs ? localSlot : 0;
            const uint32_t inputOffset =
                paired ? BlockEpilogue::PairInputOffset(inputSlot) : BlockEpilogue::InputOffset(inputSlot);
            const uint16_t inputFreeFlag = ROTATE_QUANT_SYNC_AIV_AIC_FLAG + 2 * inputSlot;
            const uint16_t inputReadyFlag = ROTATE_QUANT_SYNC_AIC_AIV_FLAG + 2 * inputSlot;
            if ASCEND_IS_AIC {
                const uint16_t subBlockFlagOffset = paired ? 0 : (roundIdx & 1) * ROTATE_QUANT_FLAG_ID_MAX;
                if (!ring && roundIdx >= inputReuseDistance) {
                    Gemm::Sync::WaitForVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(
                        inputFreeFlag + subBlockFlagOffset, paired);
                }
                int64_t coordM = asc::te::get<Gemm::MNK_M>(blockCoord);
                int64_t coordB = asc::te::get<Gemm::MNK_B>(blockCoord);
                int64_t blkM = asc::te::get<Gemm::MNK_M>(blockShape);
                auto gmBlockA = asc::te::squeeze<0>(
                    gmA.slice(asc::te::make_coord(coordB / columns, asc::te::make_coord(coordM, 0L)),
                        asc::te::make_shape(1L, asc::te::make_shape(blkM, static_cast<int64_t>(t.K)))));
                int64_t rotationOffset = blockInfo.rotationOffset;
                if constexpr (BlockDiagonal) {
                    // A column tile consumes (128/rotationK) compact rotation matrices.
                    rotationOffset = coordB * t.K * rotationK;
                }
                auto *gmBlockRot = rotation + rotationOffset;
                // Rotation changes do not reset the block buffers or the AIC/AIV round counter.
                L1Batch batch{};
                if (batchX) {
                    batch = L1Batch::ForTile<columns>(ring ? issuePanel : rowTile, spanRows);
                    if (rowTile == 0 && issuePanel == 0 && pair == 0) {
                        blockMmadOp.template PrefetchBatch<columns>(gmA, coordB / columns, span.firstRowTile, batch);
                    }
                }
                for (uint16_t lane = 0; lane < issuedColumns; ++lane) {
                    const uint16_t delta = panel * ROTATE_QUANT_FLAG_ID_MAX;
                    if constexpr (ring) {
                        if (roundIdx / (pairsPerRow * panels) != 0 || pair != 0) {
                            Gemm::Sync::WaitForVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(
                                ROTATE_QUANT_SYNC_AIV_AIC_FLAG + lane * 2 + delta, false);
                        }
                    }
                    blockMmadOp.template operator()<paired, columns>(gmBlockA,
                        gmBlockRot + static_cast<int64_t>(lane) * t.K * (BlockDiagonal ? rotationK : t.K), blockShape,
                        loadedRotationOffset != rotationOffset, rotationK, batch,
                        ring ? lane * BlockEpilogue::INPUT_SLOT_BYTES : inputOffset, lane, pairColumn + lane,
                        pair * issuedColumns + lane, rowTile == 0, panel, ring ? static_cast<int16_t>(panel) : -1);
                    if constexpr (ring) {
                        Gemm::Sync::NotifyVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(
                            ROTATE_QUANT_SYNC_AIC_AIV_FLAG + lane * 2 + delta, false);
                    }
                }
                loadedRotationOffset = rotationOffset;
                if constexpr (!ring) {
                    Gemm::Sync::NotifyVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(
                        inputReadyFlag + subBlockFlagOffset, paired);
                }
                if constexpr (ring) {
                    const int64_t nextRow = rowTile + issuePanel;
                    if (pair == 0 && nextRow < span.tiles) {
                        const uint16_t nextBank = issuePanel ^ 1;
                        const uint16_t nextPanel = nextBank ^ (nextRow & 1);
                        blockMmadOp.template PrefetchBatch<columns>(gmA, blockIndex % t.B / columns + nextPanel,
                            span.firstRowTile + nextRow, L1Batch::ForTile<columns>(nextBank, spanRows));
                    }
                } else if constexpr (columns == 4) {
                    const int64_t next = rowTile + 1;
                    if (pair == 0 && next < span.tiles) {
                        blockMmadOp.template PrefetchBatch<columns>(
                            gmA, coordB / columns, span.firstRowTile + next, L1Batch::ForTile<columns>(next, spanRows));
                    }
                } else if (batchX) {
                    // Stages 0/1 issue the next two batches. MTE2 waits on the target bank;
                    // MTE1 can continue consuming the current batch through its final tile.
                    constexpr int64_t tilesPerBank = L1Batch::TILES / columns;
                    constexpr int64_t stages = 2 * tilesPerBank;
                    const int64_t stage = rowTile % stages;
                    if (stage < 2) {
                        const int64_t firstTile = rowTile - stage + (stage + 1) * tilesPerBank;
                        if (firstTile < span.tiles) {
                            blockMmadOp.template PrefetchBatch<columns>(gmA, coordB / columns,
                                span.firstRowTile + firstTile, L1Batch::ForTile<columns>(firstTile, spanRows));
                        }
                    }
                }
            }
            if ASCEND_IS_AIV {
                if constexpr (ring) {
                    if (panel == AscendC::GetSubBlockIdx()) {
                        const bool reuse = roundIdx / (pairsPerRow * panels) + 1 < work.rounds || pair == 0;
                        for (uint16_t lane = 0; lane < 2; ++lane) {
                            Gemm::Sync::WaitForCube<ROTATE_QUANT_SYNC_MODE, PIPE_V>(
                                ROTATE_QUANT_SYNC_AIC_AIV_FLAG + 2 * lane);
                            epilogueOp.template RingQuant<ROTATE_QUANT_SYNC_MODE>(
                                lane, ROTATE_QUANT_SYNC_AIV_AIC_FLAG + 2 * lane, reuse);
                        }
                        const uint64_t gm = asc::te::get<Gemm::MNK_M>(blockCoord) * t.N +
                            (asc::te::get<Gemm::MNK_B>(blockCoord) - pairColumn) * t.K;
                        epilogueOp.RingPack(gm, pairColumn, pair, rowTile + 1 == span.tiles);
                    }
                } else if (paired || (roundIdx & 1) == AscendC::GetSubBlockIdx()) {
                    Gemm::Sync::WaitForCube<ROTATE_QUANT_SYNC_MODE, PIPE_V>(inputReadyFlag);
                    const bool reuseInput = roundIdx + inputReuseDistance < localRounds;
                    if constexpr (columns == 4) {
                        epilogueOp.template ProcessPanelPair<ROTATE_QUANT_SYNC_MODE>(
                            blockCoord, pairColumn, pair == 1, reuseInput, inputFreeFlag, inputOffset);
                    } else {
                        const uint16_t outputSlot = doubleBufferOutputs ? localSlot : 0;
                        if (doubleBufferOutputs) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputSlot);
                        }
                        epilogueOp.template ProcessTile<ROTATE_QUANT_SYNC_MODE, paired>(blockShape, blockCoord,
                            reuseInput && releaseBeforeWriteback, inputFreeFlag, inputOffset, outputSlot);
                        if (!doubleBufferOutputs) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                        }
                        if (reuseInput && !releaseBeforeWriteback) {
                            Gemm::Sync::NotifyCube<ROTATE_QUANT_SYNC_MODE, PIPE_MTE3>(inputFreeFlag);
                        }
                    }
                }
            }
            if (pair + 1 == pairsPerRow && issuePanel + 1 == panels) {
                ++rowTile;
                blockIndex += blockStride;
            }
        }

        if ASCEND_IS_AIC {
            if (batchX) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            }
        }
        if ASCEND_IS_AIV {
            if constexpr (ring) {
                for (uint16_t g = 0; g < BlockEpilogue::RING_GROUPS; ++g) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(g);
                }
            } else if constexpr (columns == 4) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            } else if (doubleBufferOutputs) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            }
            AscendC::SetCtrlSpr<ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL, ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL>(
                oriOverflowMode);
        }
    }

  private:
    // B == 1 views x as (1, M * N/K, K); B > 1 views x as (B, M, K).
    // In the latter case batch B advances by K, while adjacent M rows advance by B * K.
    __aicore__ inline auto MakeLayoutAGm(int64_t m, int64_t n, int64_t k, int64_t b) {
        int64_t rows = (b > 1) ? m : m * (n / k);
        int64_t rowStride = b * k;
        auto matrixShape =
            asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, rows), asc::te::make_shape(asc::te::_1{}, k));
        auto matrixStride = asc::te::make_stride(
            asc::te::make_stride(asc::te::_0{}, rowStride), asc::te::make_stride(asc::te::_0{}, asc::te::_1{}));
        auto shape = asc::te::make_shape(b, matrixShape);
        auto stride = asc::te::make_stride(k, matrixStride);
        return asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<Input>>(
            shape, stride);
    }

    static constexpr uint16_t ROTATE_QUANT_SYNC_MODE = 4;
    static constexpr uint16_t ROTATE_QUANT_SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint16_t ROTATE_QUANT_SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint16_t ROTATE_QUANT_FLAG_ID_MAX = 16;
    static constexpr uint16_t ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL = 60;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
