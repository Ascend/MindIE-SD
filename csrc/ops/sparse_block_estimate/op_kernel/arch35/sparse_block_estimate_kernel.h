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

#ifndef SPARSE_BLOCK_ESTIMATE_ARCH35_KERNEL_H
#define SPARSE_BLOCK_ESTIMATE_ARCH35_KERNEL_H

#include "kernel_operator.h"
#include <cstdint>

namespace SparseEstimateArch35 {

constexpr uint32_t PACK_STRIDE = 8;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t PACKED_DIM = PACK_STRIDE * HEAD_DIM;
constexpr uint32_t Q_RAW_TILE = 256;
constexpr uint32_t K_RAW_TILE = 512;
constexpr uint32_t K_SEGMENT_TILES = 65536U / K_RAW_TILE;
constexpr uint32_t Q_PACKED_TILE = Q_RAW_TILE / PACK_STRIDE;
constexpr uint32_t Q_TILES_PER_GROUP = 4;
constexpr uint32_t Q_GROUP_PACKED_ROWS = Q_TILES_PER_GROUP * Q_PACKED_TILE;
constexpr uint32_t K_PACKED_TILE = K_RAW_TILE / PACK_STRIDE;
constexpr uint32_t L0_K_TILE = 128;
constexpr uint32_t K_L0_SLICES = PACKED_DIM / L0_K_TILE;
constexpr uint32_t Q_BLOCK = 128;
constexpr uint32_t KV_BLOCK = 128;
constexpr uint32_t Q_ROWS_PER_BLOCK = Q_BLOCK / PACK_STRIDE;
constexpr uint32_t KV_COLS_PER_BLOCK = KV_BLOCK / PACK_STRIDE;
constexpr uint32_t Q_BLOCKS_PER_TILE = Q_RAW_TILE / Q_BLOCK;
constexpr uint32_t KV_BLOCKS_PER_TILE = K_RAW_TILE / KV_BLOCK;
constexpr uint32_t SCORE_PIECE_SIZE = 16;
constexpr uint32_t AIV_Q_ROWS = Q_GROUP_PACKED_ROWS / 2U;
constexpr uint32_t Q_BLOCKS_PER_AIV = AIV_Q_ROWS / Q_ROWS_PER_BLOCK;
constexpr uint32_t S_ELEMS = AIV_Q_ROWS * K_PACKED_TILE;
constexpr uint32_t S_BYTES = S_ELEMS * sizeof(float);
constexpr uint32_t MAX_KV_BLOCKS_ALIGNED = 256;
constexpr uint32_t MAX_LONG_KV_BLOCKS_ALIGNED = 3328;
constexpr uint32_t MAX_K_SEQUENCE = MAX_LONG_KV_BLOCKS_ALIGNED * KV_BLOCK;
constexpr uint32_t MAX_K_TILES = MAX_KV_BLOCKS_ALIGNED / KV_BLOCKS_PER_TILE;
constexpr uint32_t FIRST_REDUCE_HISTORY_BYTES = MAX_K_TILES * AIV_Q_ROWS * KV_BLOCKS_PER_TILE * sizeof(float);
constexpr uint32_t MAX_HISTORY_BYTES = MAX_K_TILES * AIV_Q_ROWS * sizeof(float);
constexpr uint32_t PARTIAL_SCORE_BYTES = AIV_Q_ROWS * MAX_KV_BLOCKS_ALIGNED * sizeof(float);
constexpr uint32_t L1_Q_BYTES = Q_GROUP_PACKED_ROWS * PACKED_DIM * sizeof(uint16_t);
constexpr uint32_t L1_K_BYTES = 2U * PACKED_DIM * K_PACKED_TILE * sizeof(uint16_t);
constexpr uint32_t L1_USED_BYTES = L1_Q_BYTES + L1_K_BYTES;
constexpr uint32_t UB_LOCAL_MAX_OFFSET = 2U * S_BYTES;
constexpr uint32_t UB_LOCAL_SUM_OFFSET = UB_LOCAL_MAX_OFFSET + 256U;
constexpr uint32_t UB_EXP_INPUT_OFFSET = UB_LOCAL_SUM_OFFSET + 256U;
constexpr uint32_t UB_EXP_OUTPUT_OFFSET = UB_EXP_INPUT_OFFSET + 256U;
constexpr uint32_t UB_PARTIAL_OFFSET = 34U * 1024U;
constexpr uint32_t UB_FIRST_HISTORY_OFFSET = UB_PARTIAL_OFFSET + PARTIAL_SCORE_BYTES;
constexpr uint32_t UB_MAX_HISTORY_OFFSET = UB_FIRST_HISTORY_OFFSET + FIRST_REDUCE_HISTORY_BYTES;
constexpr uint32_t UB_HISTORY_END = UB_MAX_HISTORY_OFFSET + MAX_HISTORY_BYTES;
// The online-softmax histories and the final mask workspaces have disjoint
// lifetimes.  Reuse the 64 KiB partial-score region after BuildPartialScore so
// the A2-compatible 256-column mask still fits in the 248 KiB usable UB.
// Ascend 950 Sort stores one FP32 score and one uint32 index per item.
// Both the result and full-sort scratch need N * 8 bytes (GetSortTmpSize).
constexpr uint32_t SORT_ITEM_BYTES = sizeof(float) + sizeof(uint32_t);
constexpr uint32_t SHORT_SORT_BYTES = MAX_KV_BLOCKS_ALIGNED * SORT_ITEM_BYTES;
constexpr uint32_t UB_SORT_TMP_OFFSET = UB_PARTIAL_OFFSET;
constexpr uint32_t UB_SORT_PROPOSAL_OFFSET = UB_SORT_TMP_OFFSET + SHORT_SORT_BYTES;
constexpr uint32_t UB_SORTED_SCORE_OFFSET = UB_SORT_PROPOSAL_OFFSET + SHORT_SORT_BYTES;
constexpr uint32_t UB_SORT_INDEX_OFFSET = UB_SORTED_SCORE_OFFSET + 1024U;
constexpr uint32_t UB_SORTED_INDEX_OFFSET = UB_SORT_INDEX_OFFSET + 1024U;
constexpr uint32_t UB_MASK_OFFSET = UB_SORTED_INDEX_OFFSET + 1024U;
// Score output aliases the history region only after every K tile has been
// rescaled into partial0Ub_.
constexpr uint32_t UB_SCORE_OFFSET = UB_FIRST_HISTORY_OFFSET;
constexpr uint32_t UB_SCORE_WORK_OFFSET = UB_SCORE_OFFSET + 4U * 1024U;
constexpr uint32_t UB_SCORE_DENOM_OFFSET = UB_SCORE_WORK_OFFSET + 1024U;
constexpr uint32_t UB_MASK_HALF_OFFSET = UB_SCORE_DENOM_OFFSET + 1024U;
constexpr uint32_t UB_COUNT_HALF_OFFSET = UB_MASK_HALF_OFFSET + 1024U;
constexpr uint32_t UB_COUNT_OFFSET = UB_COUNT_HALF_OFFSET + 1024U;
constexpr uint32_t UB_SUM_TMP_OFFSET = UB_HISTORY_END;
constexpr uint32_t SUM_SCRATCH_BYTES = 32U * 1024U;
constexpr uint32_t UB_USED_BYTES = UB_SUM_TMP_OFFSET + SUM_SCRATCH_BYTES;
// Long-K postprocess aliases all online-history storage after the history has
// been converted into compact score rows in the segment workspace.
constexpr uint32_t LONG_SORT_BYTES = MAX_LONG_KV_BLOCKS_ALIGNED * SORT_ITEM_BYTES;
constexpr uint32_t UB_LONG_SORT_TMP_OFFSET = UB_PARTIAL_OFFSET;
constexpr uint32_t UB_LONG_SORT_PROPOSAL_OFFSET = UB_LONG_SORT_TMP_OFFSET + LONG_SORT_BYTES;
constexpr uint32_t UB_LONG_SCORE_OFFSET = UB_LONG_SORT_PROPOSAL_OFFSET + LONG_SORT_BYTES;
constexpr uint32_t UB_LONG_SCORE_WORK_OFFSET = UB_LONG_SCORE_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(float);
constexpr uint32_t UB_LONG_SORTED_SCORE_OFFSET = UB_LONG_SCORE_WORK_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(float);
constexpr uint32_t UB_LONG_SORT_INDEX_OFFSET = UB_LONG_SORTED_SCORE_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(float);
constexpr uint32_t UB_LONG_SORTED_INDEX_OFFSET =
    UB_LONG_SORT_INDEX_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(uint32_t);
constexpr uint32_t UB_LONG_MASK_OFFSET = UB_LONG_SORTED_INDEX_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(uint32_t);
constexpr uint32_t UB_LONG_MASK_HALF_OFFSET = UB_LONG_MASK_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(int8_t);
constexpr uint32_t UB_LONG_COUNT_HALF_OFFSET = UB_LONG_MASK_HALF_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(uint16_t);
constexpr uint32_t UB_LONG_SCORE_DENOM_OFFSET =
    UB_LONG_COUNT_HALF_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(uint16_t);
constexpr uint32_t UB_LONG_COUNT_OFFSET = UB_LONG_SCORE_DENOM_OFFSET + MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(float);
constexpr uint32_t UB_LONG_MASK_END = UB_LONG_COUNT_OFFSET + 32U;
// Keep each completed output row live until this AIV finishes the whole group.
// Counts occupy separate 32-byte blocks for aligned Vector writes and DMA.
constexpr uint32_t OUTPUT_COUNT_STRIDE = 32U / sizeof(int32_t);
constexpr uint32_t UB_BATCH_MASK_OFFSET = UB_USED_BYTES;
constexpr uint32_t UB_BATCH_COUNT_OFFSET = UB_BATCH_MASK_OFFSET + Q_BLOCKS_PER_AIV * MAX_KV_BLOCKS_ALIGNED;
constexpr uint32_t UB_BATCH_END = UB_BATCH_COUNT_OFFSET + Q_BLOCKS_PER_AIV * 32U;
constexpr uint32_t UB_LONG_BATCH_MASK_OFFSET = UB_LONG_MASK_END;
constexpr uint32_t UB_LONG_BATCH_COUNT_OFFSET =
    UB_LONG_BATCH_MASK_OFFSET + Q_BLOCKS_PER_AIV * MAX_LONG_KV_BLOCKS_ALIGNED;
constexpr uint32_t UB_LONG_BATCH_END = UB_LONG_BATCH_COUNT_OFFSET + Q_BLOCKS_PER_AIV * 32U;
// During the K sweep/reconstruction, keep an entire K segment's history in
// UB. Sorting begins only after these histories have become compact UB scores.
constexpr uint32_t UB_SEGMENT_FIRST_OFFSET = UB_PARTIAL_OFFSET;
constexpr uint32_t UB_SEGMENT_MAX_OFFSET =
    UB_SEGMENT_FIRST_OFFSET + K_SEGMENT_TILES * AIV_Q_ROWS * KV_BLOCKS_PER_TILE * sizeof(float);
constexpr uint32_t UB_SEGMENT_SUM_OFFSET = UB_SEGMENT_MAX_OFFSET + K_SEGMENT_TILES * AIV_Q_ROWS * sizeof(float);
constexpr uint32_t LONG_SUM_SCRATCH_BYTES = 2U * 1024U;
constexpr uint32_t UB_SEGMENT_SCORE_OFFSET = UB_SEGMENT_SUM_OFFSET + LONG_SUM_SCRATCH_BYTES;
constexpr uint32_t UB_SEGMENT_SCORE_END =
    UB_SEGMENT_SCORE_OFFSET + Q_BLOCKS_PER_AIV * MAX_LONG_KV_BLOCKS_ALIGNED * sizeof(float);
// Final reconstruction and mask generation start after the last S tile has
// been consumed. Cube waits for TASK_DONE before writing S for another task.
// Reuse the dead S region for reconstructed scores. Keep one vector of read
// padding: arch950 Sum loads a full 256-byte vector even when reducing 16 floats.
constexpr uint32_t UB_SEGMENT_PARTIAL_OFFSET = 0U;
constexpr uint32_t LONG_SUM_READ_PADDING = 256U;
constexpr uint32_t LONG_PARTIAL_BYTES = UB_LOCAL_MAX_OFFSET - UB_SEGMENT_PARTIAL_OFFSET - LONG_SUM_READ_PADDING;
constexpr uint32_t LongScoreBatchTiles(uint32_t rows) {
    // Two K512 tiles produce eight FP32 scores, preserving 32-byte output alignment.
    return LONG_PARTIAL_BYTES / (KV_BLOCKS_PER_TILE * rows * sizeof(float)) / 2U * 2U;
}
// Piece sums are only live during mask selection, after exp scratch is dead.
// 3328 scores require 208 FP32 piece sums in the expired exp/partial scratch.
constexpr uint32_t UB_PIECE_SUM_OFFSET = UB_EXP_INPUT_OFFSET;
constexpr uint32_t PIECE_SUM_BYTES = MAX_LONG_KV_BLOCKS_ALIGNED / SCORE_PIECE_SIZE * sizeof(float);

} // namespace SparseEstimateArch35

#include "sparse_block_estimate_qk.h"

namespace SparseEstimateArch35 {

using namespace AscendC;

constexpr uint16_t FLAG_TASK_DONE = 4;

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y) { return (x + y - 1U) / y; }

__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t y) { return CeilDiv(x, y) * y; }

// Mode 4 addresses the two AIV flag banks separately from the AIC side.
// Each AIV waits on local flag 5, but the AIC must set both 5 and 21.
__aicore__ inline uint16_t PhaseDoneFlag(unsigned aiv) { return static_cast<uint16_t>(5U + aiv * 16U); }

// Return the full-wave prefix. A partial wave is split only when its two-Q
// children fit in one wave. Whole four-Q groups must fit within each KV head
// so that multiplying a task ID by two preserves its Q and workspace range.
__aicore__ inline uint32_t FullQTaskPrefix(uint32_t total, uint32_t cores, uint32_t tilesPerKv, bool enabled) {
    if (!enabled || cores == 0U || total < cores || total > 0x7fffffffU || tilesPerKv % 4U != 0U) {
        return total;
    }
    uint32_t remainder = total % cores;
    if (remainder == 0U || remainder > cores / 2U) {
        return total;
    }
    return total - remainder;
}

constexpr uint32_t MASK_SCORE_PIECE_SIZE = 64U;

// Fixed-shape online softmax for one AIV-owned 32/64 x 64 FP32 QK tile. The implementation
// follows the arch35 FIA VF path: first obtain every row's new running max,
// then reload the row, exponentiate against that max and reduce it. A packed
// K tile contains four 16-column logical KV128 blocks.
template <uint32_t RowCount>
__simd_vf__ inline void OnlineSoftmaxReduceVF(__ubuf__ float *src, __ubuf__ float *localMax, __ubuf__ float *localSum,
    __ubuf__ float *tileMax, __ubuf__ float *firstReduce, __ubuf__ float *rescaledSumScratch, uint32_t validColumns,
    float scale) {
    using namespace AscendC::MicroAPI;

    RegTensor<float> input;
    RegTensor<float> currentMax;
    RegTensor<float> oldMax;
    RegTensor<float> newMax;
    RegTensor<float> oldSum;
    RegTensor<float> oldMaxScale;
    RegTensor<float> rescaledOldSum;
    RegTensor<float> maxBroadcast;
    RegTensor<float> expValue;
    RegTensor<float> prefixSum0;
    RegTensor<float> prefixSum1;
    RegTensor<float> prefixSum2;
    RegTensor<float> blockSum0;
    RegTensor<float> blockSum1;
    RegTensor<float> blockSum2;
    RegTensor<float> blockSum3;
    RegTensor<float> firstThreeSum;
    RegTensor<float> tileSum;
    RegTensor<float> newSum;

    UnalignRegForStore maxStore;
    UnalignRegForStore reduceStore0;
    UnalignRegForStore reduceStore1;
    UnalignRegForStore reduceStore2;

    // CANN 9.1 UpdateMask takes uint32_t&, so every mask length must be a
    // writable scalar register rather than a literal, constexpr, or expression.
    uint32_t scalarColumns = 1U;
    uint32_t rowCount = RowCount;
    // UpdateMask consumes its uint32_t argument (POST_UPDATE).  Give every
    // mask an independent scalar copy.
    uint32_t validMaskColumns = validColumns;
    uint32_t prefix0Columns = validColumns < KV_COLS_PER_BLOCK ? validColumns : KV_COLS_PER_BLOCK;
    uint32_t prefix1Limit = 2U * KV_COLS_PER_BLOCK;
    uint32_t prefix1Columns = validColumns < prefix1Limit ? validColumns : prefix1Limit;
    uint32_t prefix2Limit = 3U * KV_COLS_PER_BLOCK;
    uint32_t prefix2Columns = validColumns < prefix2Limit ? validColumns : prefix2Limit;
    MaskReg scalarMask = UpdateMask<float>(scalarColumns);
    MaskReg rowMask = UpdateMask<float>(rowCount);
    MaskReg validMask = UpdateMask<float>(validMaskColumns);
    MaskReg prefix0Mask = UpdateMask<float>(prefix0Columns);
    MaskReg prefix1Mask = UpdateMask<float>(prefix1Columns);
    MaskReg prefix2Mask = UpdateMask<float>(prefix2Columns);

    // Pass 1: scale QK in registers and collect the current tile maximum.
    __ubuf__ float *tileMaxOut = tileMax;
    for (uint16_t row = 0; row < RowCount; ++row) {
        LoadAlign(input, src + row * K_PACKED_TILE);
        Muls(input, input, scale, validMask);
        Reduce<AscendC::MicroAPI::ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(currentMax, input, validMask);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(tileMaxOut, currentMax, maxStore, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(tileMaxOut, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // Merge the 64 current maxima with the running state in one vector.
    LoadAlign(currentMax, tileMax);
    LoadAlign(oldMax, localMax);
    Max(newMax, currentMax, oldMax, rowMask);
    ExpSub(oldMaxScale, oldMax, newMax, rowMask);
    LoadAlign(oldSum, localSum);
    Mul(rescaledOldSum, oldSum, oldMaxScale, rowMask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(localMax, newMax, rowMask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(tileMax, newMax, rowMask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(rescaledSumScratch, rescaledOldSum, rowMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // Pass 2: exp(S - newMax), reduce the four logical KV blocks independently,
    // and finish the online running sum update.
    // Store all four logical blocks in four streams, then sum those aligned
    // vectors to update localSum. This preserves tileMax as the max-history
    // snapshot while keeping UnalignRegForStore at the arch950 limit of four.
    __ubuf__ float *reduceOut0 = firstReduce;
    __ubuf__ float *reduceOut1 = firstReduce + RowCount;
    __ubuf__ float *reduceOut2 = firstReduce + 2U * RowCount;
    __ubuf__ float *reduceOut3 = firstReduce + 3U * RowCount;
    for (uint16_t row = 0; row < RowCount; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(maxBroadcast, localMax + row);

        LoadAlign(input, src + row * K_PACKED_TILE);
        Muls(input, input, scale, validMask);
        ExpSub(expValue, input, maxBroadcast, validMask);
        Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
            prefixSum0, expValue, prefix0Mask);
        Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
            prefixSum1, expValue, prefix1Mask);
        Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
            prefixSum2, expValue, prefix2Mask);
        Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(tileSum, expValue, validMask);
        Adds(blockSum0, prefixSum0, 0.0f, scalarMask);
        Sub(blockSum1, prefixSum1, prefixSum0, scalarMask);
        Sub(blockSum2, prefixSum2, prefixSum1, scalarMask);
        Add(firstThreeSum, blockSum0, blockSum1, scalarMask);
        Add(firstThreeSum, firstThreeSum, blockSum2, scalarMask);
        Sub(blockSum3, tileSum, firstThreeSum, scalarMask);

        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(reduceOut0, blockSum0, reduceStore0, 1);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(reduceOut1, blockSum1, reduceStore1, 1);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(reduceOut2, blockSum2, reduceStore2, 1);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(reduceOut3, blockSum3, maxStore, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(reduceOut0, reduceStore0, 0);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(reduceOut1, reduceStore1, 0);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(reduceOut2, reduceStore2, 0);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(reduceOut3, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    LoadAlign(blockSum0, firstReduce);
    LoadAlign(blockSum1, firstReduce + RowCount);
    LoadAlign(blockSum2, firstReduce + 2U * RowCount);
    LoadAlign(blockSum3, firstReduce + 3U * RowCount);
    Add(firstThreeSum, blockSum0, blockSum1, rowMask);
    Add(firstThreeSum, firstThreeSum, blockSum2, rowMask);
    Add(tileSum, firstThreeSum, blockSum3, rowMask);
    LoadAlign(oldSum, rescaledSumScratch);
    Add(newSum, oldSum, tileSum, rowMask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(localSum, newSum, rowMask);
}

// Reduce the initialized, aligned score range in 16/32/64-score pieces.
// Sort/Extract only writes alignedColumns elements; the rest of the allocated
// score buffer is uninitialized beyond that range for shorter sequences.
template <uint32_t PieceSize>
__simd_vf__ inline void ReduceScorePiecesVF(
    __ubuf__ float *src, __ubuf__ float *pieceSums, __ubuf__ float *total, uint32_t alignedColumns) {
    using namespace AscendC::MicroAPI;

    RegTensor<float> input;
    RegTensor<float> pieceSum;
    RegTensor<float> totalSum;
    UnalignRegForStore pieceStore;
    UnalignRegForStore totalStore;

    static_assert(PieceSize == 16U || PieceSize == 32U || PieceSize == 64U, "Unsupported score piece size");
    uint32_t pieceColumns = PieceSize;
    uint32_t scalarColumns = 1U;
    MaskReg pieceMask = UpdateMask<float>(pieceColumns);
    MaskReg scalarMask = UpdateMask<float>(scalarColumns);
    Duplicate(totalSum, 0.0f);
    __ubuf__ float *pieceOut = pieceSums;
    uint16_t pieceCount = static_cast<uint16_t>((alignedColumns + PieceSize - 1U) / PieceSize);
    for (uint16_t piece = 0; piece < pieceCount; ++piece) {
        uint32_t remaining = alignedColumns - piece * PieceSize;
        if (remaining < PieceSize) {
            pieceMask = UpdateMask<float>(remaining);
        }
        LoadAlign(input, src + piece * PieceSize);
        Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(pieceSum, input, pieceMask);
        Add(totalSum, totalSum, pieceSum, scalarMask);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(pieceOut, pieceSum, pieceStore, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(pieceOut, pieceStore, 0);
    StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(total, totalSum, totalStore, 1);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(total, totalStore, 0);
}

template <typename InputT, uint32_t GroupTiles = 4U> class Kernel {
    // Keep the baseline UB offsets/capacities. M64 uses only half of each
    // history region, leaving the proven sort/compact-score aliases intact.
    static_assert(GroupTiles == 2U || GroupTiles == 4U, "Only M64/M128 Q groups are supported");
    static constexpr uint32_t Q_TILES_PER_GROUP = GroupTiles;
    static constexpr uint32_t Q_GROUP_PACKED_ROWS = GroupTiles * Q_PACKED_TILE;
    static constexpr uint32_t AIV_Q_ROWS = Q_GROUP_PACKED_ROWS / 2U;
    static constexpr uint32_t Q_BLOCKS_PER_AIV = AIV_Q_ROWS / Q_ROWS_PER_BLOCK;
    static constexpr uint32_t S_BYTES = AIV_Q_ROWS * K_PACKED_TILE * sizeof(float);
    static constexpr uint32_t SCORE_BATCH_TILES = LongScoreBatchTiles(AIV_Q_ROWS);
    static_assert(SCORE_BATCH_TILES * KV_BLOCKS_PER_TILE * AIV_Q_ROWS * sizeof(float) + LONG_SUM_READ_PADDING <=
            UB_LOCAL_MAX_OFFSET,
        "Reconstruction batch must not overlap normalization state");
    static_assert(AIV_Q_ROWS == 32U || AIV_Q_ROWS == 64U, "VF row state must fit in one FP32 register");
    static_assert(2U * S_BYTES <= UB_LOCAL_MAX_OFFSET, "S ping/pong must not overlap row state");
    static_assert(Q_GROUP_PACKED_ROWS * PACKED_DIM * sizeof(uint16_t) + L1_K_BYTES <= L1_USED_BYTES,
        "Q and K buffers must fit in L1");

  public:
    using BlockMmadQK = PackedQK<InputT, GroupTiles>;

    __aicore__ inline Kernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR sparseMask, GM_ADDR sparseCount, GM_ADDR workspace,
        const SparseBlockEstimateTilingData &tiling) {
        tiling_ = &tiling;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(query));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(key));
        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(sparseMask));
        countGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sparseCount));

        qTilesPerHead_ = CeilDiv(tiling.seqLenQ, Q_RAW_TILE);
        qBlocksPerHead_ = CeilDiv(tiling.seqLenQ, Q_BLOCK);
        gqa_ = tiling.headNumQ / tiling.headNumKV;
        // Treat all contiguous BNSD Q tiles sharing one KV head as one range.
        // A four-tile M128 task may therefore cross a Q-head boundary while
        // retaining the same K head.
        qTilesPerKvHead_ = qTilesPerHead_ * gqa_;
        qGroupsPerKvHead_ = CeilDiv(qTilesPerKvHead_, Q_TILES_PER_GROUP);
        kPackedCols_ = CeilDiv(tiling.seqLenK, PACK_STRIDE);
        kTiles_ = CeilDiv(kPackedCols_, K_PACKED_TILE);
        kvBlocks_ = CeilDiv(tiling.seqLenK, tiling.sparseSize);
        kvBlocksAligned_ = AlignUp(kvBlocks_, 32U);
        totalTasks_ = tiling.batchSize * tiling.headNumKV * qGroupsPerKvHead_;
        longK_ = kvBlocksAligned_ > MAX_KV_BLOCKS_ALIGNED;
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        workspaceMaxOffset_ = kTiles_ * AIV_Q_ROWS * KV_BLOCKS_PER_TILE;
        workspaceStateOffset_ = workspaceMaxOffset_ + kTiles_ * AIV_Q_ROWS;
        workspaceStrideElems_ = workspaceStateOffset_ + 2U * AIV_Q_ROWS;
    }

    __aicore__ inline void InitBuffers() {
        // The two-Q tail shrinks Q and S buffers, while all UB workspaces
        // retain their baseline offsets. Each AIV owns whole Q128 blocks.

        sUb_[0] = ub_[0].template ReinterpretCast<float>();
        sUb_[1] = ub_[S_BYTES].template ReinterpretCast<float>();
        localMaxUb_ = ub_[UB_LOCAL_MAX_OFFSET].template ReinterpretCast<float>();
        localSumUb_ = ub_[UB_LOCAL_SUM_OFFSET].template ReinterpretCast<float>();
        expInputUb_ = ub_[UB_EXP_INPUT_OFFSET].template ReinterpretCast<float>();
        expOutputUb_ = ub_[UB_EXP_OUTPUT_OFFSET].template ReinterpretCast<float>();
        pieceSumsUb_ = ub_[UB_PIECE_SUM_OFFSET].template ReinterpretCast<float>();
        partial0Ub_ = ub_[UB_PARTIAL_OFFSET].template ReinterpretCast<float>();
        // All Sort operands are disjoint and 32-byte aligned as required by
        // the arch950 API.
        if (longK_) {
            firstReduceHistoryUb_ = ub_[UB_SEGMENT_FIRST_OFFSET].template ReinterpretCast<float>();
            maxHistoryUb_ = ub_[UB_SEGMENT_MAX_OFFSET].template ReinterpretCast<float>();
            sortTmpUb_ = ub_[UB_LONG_SORT_TMP_OFFSET].template ReinterpretCast<float>();
            sortProposalUb_ = ub_[UB_LONG_SORT_PROPOSAL_OFFSET].template ReinterpretCast<float>();
            scoreUb_ = ub_[UB_LONG_SCORE_OFFSET].template ReinterpretCast<float>();
            scoreWorkUb_ = ub_[UB_LONG_SCORE_WORK_OFFSET].template ReinterpretCast<float>();
            sortedScoreUb_ = ub_[UB_LONG_SORTED_SCORE_OFFSET].template ReinterpretCast<float>();
            sortIndexUb_ = ub_[UB_LONG_SORT_INDEX_OFFSET].template ReinterpretCast<uint32_t>();
            sortedIndexUb_ = ub_[UB_LONG_SORTED_INDEX_OFFSET].template ReinterpretCast<uint32_t>();
            maskUb_ = ub_[UB_LONG_MASK_OFFSET].template ReinterpretCast<int8_t>();
            maskHalfUb_ = ub_[UB_LONG_MASK_HALF_OFFSET].template ReinterpretCast<half>();
            scoreDenomUb_ = ub_[UB_LONG_SCORE_DENOM_OFFSET].template ReinterpretCast<float>();
            countUb_ = ub_[UB_LONG_COUNT_OFFSET].template ReinterpretCast<int32_t>();
            batchMaskUb_ = ub_[UB_LONG_BATCH_MASK_OFFSET].template ReinterpretCast<int8_t>();
            batchCountUb_ = ub_[UB_LONG_BATCH_COUNT_OFFSET].template ReinterpretCast<int32_t>();
            // Scratch stays separate from reconstructed scores and batched mask outputs.
            sumTmpUb_ = ub_[UB_SEGMENT_SUM_OFFSET].template ReinterpretCast<uint8_t>();
        } else {
            firstReduceHistoryUb_ = ub_[UB_FIRST_HISTORY_OFFSET].template ReinterpretCast<float>();
            maxHistoryUb_ = ub_[UB_MAX_HISTORY_OFFSET].template ReinterpretCast<float>();
            sortTmpUb_ = ub_[UB_SORT_TMP_OFFSET].template ReinterpretCast<float>();
            sortProposalUb_ = ub_[UB_SORT_PROPOSAL_OFFSET].template ReinterpretCast<float>();
            sortedScoreUb_ = ub_[UB_SORTED_SCORE_OFFSET].template ReinterpretCast<float>();
            sortIndexUb_ = ub_[UB_SORT_INDEX_OFFSET].template ReinterpretCast<uint32_t>();
            sortedIndexUb_ = ub_[UB_SORTED_INDEX_OFFSET].template ReinterpretCast<uint32_t>();
            sumTmpUb_ = ub_[UB_SUM_TMP_OFFSET].template ReinterpretCast<uint8_t>();
            maskUb_ = ub_[UB_MASK_OFFSET].template ReinterpretCast<int8_t>();
            scoreUb_ = ub_[UB_SCORE_OFFSET].template ReinterpretCast<float>();
            scoreWorkUb_ = ub_[UB_SCORE_WORK_OFFSET].template ReinterpretCast<float>();
            scoreDenomUb_ = ub_[UB_SCORE_DENOM_OFFSET].template ReinterpretCast<float>();
            maskHalfUb_ = ub_[UB_MASK_HALF_OFFSET].template ReinterpretCast<half>();
            countUb_ = ub_[UB_COUNT_OFFSET].template ReinterpretCast<int32_t>();
            batchMaskUb_ = ub_[UB_BATCH_MASK_OFFSET].template ReinterpretCast<int8_t>();
            batchCountUb_ = ub_[UB_BATCH_COUNT_OFFSET].template ReinterpretCast<int32_t>();
        }
    }

    __aicore__ inline void Process(uint32_t taskBegin, uint32_t taskEnd, bool phaseFence = false) {
        taskBegin_ = taskBegin;
#ifdef __DAV_CUBE__
        uint32_t core = GetBlockIdx();
#else
        uint32_t core = GetBlockIdx() / GetSubBlockNum();
#endif
        // In the two-Q tail, some launched cores have no work. Both AIVs and
        // their AIC must return before seeding any cross-core credits.
        if (taskEnd <= taskBegin || core >= taskEnd - taskBegin) {
            return;
        }
        if (tiling_->dim != HEAD_DIM || tiling_->stride != PACK_STRIDE || tiling_->sparseSize != Q_BLOCK ||
            tiling_->causal || kvBlocksAligned_ > MAX_LONG_KV_BLOCKS_ALIGNED) {
            return;
        }

        InitPipeEvents();
#ifdef __DAV_CUBE__
        uint32_t tasksOnCore = CeilDiv(taskEnd - taskBegin - core, tiling_->coreNumAic);
        residentQTask_ = totalTasks_;
        // Segment-major scheduling is local to each core. There
        // is no global barrier: cores may progress to different K segments.
        for (segmentStart_ = 0; segmentStart_ < kTiles_; segmentStart_ = segmentEnd_) {
            segmentEnd_ = longK_ ? min(segmentStart_ + K_SEGMENT_TILES, kTiles_) : kTiles_;
            for (uint32_t ordinal = 0; ordinal < tasksOnCore; ++ordinal) {
                uint32_t task = TaskInSegment(core, ordinal, tasksOnCore);
                RunTask(task);
            }
        }
        // Drain the reverse credits returned by both AIVs for both S buffers.
        uint16_t sFree0 = 0U;
        uint16_t sFree0V1 = 16U;
        uint16_t sFree1 = 1U;
        uint16_t sFree1V1 = 17U;
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(sFree0);
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(sFree0V1);
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(sFree1);
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(sFree1V1);
#else
        uint32_t tasksOnCore = CeilDiv(taskEnd - taskBegin - core, tiling_->coreNumAic);
        subBlockIdx_ = GetSubBlockIdx();
        // Both AIVs consume both ping/pong S buffers. Seed one reverse credit
        // for each buffer; mode 4 routes it to this AIV's physical flag bank.
        uint16_t initialSFree0 = 0U;
        uint16_t initialSFree1 = 1U;
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(initialSFree0);
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(initialSFree1);
        for (segmentStart_ = 0; segmentStart_ < kTiles_; segmentStart_ = segmentEnd_) {
            segmentEnd_ = longK_ ? min(segmentStart_ + K_SEGMENT_TILES, kTiles_) : kTiles_;
            for (uint32_t ordinal = 0; ordinal < tasksOnCore; ++ordinal) {
                uint32_t task = TaskInSegment(core, ordinal, tasksOnCore);
                RunTask(task);
            }
        }
#endif
        ReleasePipeEvents();
        if (phaseFence) {
            // Synchronize only this AIC and its two AIVs before reusing phase flags.
#ifdef __DAV_CUBE__
            // Drain both TASK_DONE waits on MTE1, S-free waits on FIX, and
            // ReleasePipeEvents before notifying AIVs or reusing local event IDs.
            PipeBarrier<PIPE_ALL>();
            // Mode 4 targets one AIV flag bank at a time.
            for (uint32_t aiv = 0U; aiv < 2U; ++aiv) {
                uint16_t phaseDone = PhaseDoneFlag(aiv);
                AscendC::CrossCoreSetFlag<4, PIPE_FIX>(phaseDone);
            }
            // Subsequent FIX operations follow these notifications on the same pipe.
#else
            // Next-phase S-free signals also use MTE3, so they cannot pass this wait.
            uint16_t phaseDone = PhaseDoneFlag(0U);
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(phaseDone);
            // Order next-phase V/MTE2 buffer accesses after the MTE3 wait.
            PipeBarrier<PIPE_ALL>();
#endif
        }
    }

  private:
    __aicore__ inline uint32_t TaskInSegment(uint32_t aicIdx, uint32_t ordinal, uint32_t tasksOnCore) const {
        // Reverse local traversal on alternating K segments. The last Q group
        // of one segment is then the first of the next, still resident in L1.
        uint32_t index = ((segmentStart_ / K_SEGMENT_TILES) & 1U) != 0U ? tasksOnCore - 1U - ordinal : ordinal;
        return taskBegin_ + aicIdx + index * tiling_->coreNumAic;
    }

    __aicore__ inline void DecodeTask(uint32_t task, uint32_t &batch, uint32_t &kvHead, uint32_t &flatQTile) const {
        flatQTile = (task % qGroupsPerKvHead_) * Q_TILES_PER_GROUP;
        uint32_t bkv = task / qGroupsPerKvHead_;
        kvHead = bkv % tiling_->headNumKV;
        batch = bkv / tiling_->headNumKV;
    }

    __aicore__ inline bool DecodeOutputBlock(
        uint32_t kvHead, uint32_t flatQTile, uint32_t localQBlock, uint32_t &qHead, uint32_t &qBlockInHead) const {
        uint32_t flatQBlock = flatQTile * Q_BLOCKS_PER_TILE + localQBlock;
        if (flatQBlock >= qTilesPerKvHead_ * Q_BLOCKS_PER_TILE) {
            return false;
        }
        // Each head occupies whole Q256 tiles in L1, even when its final
        // Q128 block uses only the first half of a tile.
        uint32_t paddedBlocksPerHead = qTilesPerHead_ * Q_BLOCKS_PER_TILE;
        qHead = kvHead * gqa_ + flatQBlock / paddedBlocksPerHead;
        qBlockInHead = flatQBlock % paddedBlocksPerHead;
        return qHead < tiling_->headNumQ && qBlockInHead < qBlocksPerHead_;
    }

    __aicore__ inline void RunTask(uint32_t task) {
        uint32_t batch = 0;
        uint32_t kvHead = 0;
        uint32_t flatQTile = 0;
        DecodeTask(task, batch, kvHead, flatQTile);

#ifdef __DAV_CUBE__
        RunCube(mm_, batch, kvHead, flatQTile, task == residentQTask_);
        residentQTask_ = task;
#elif defined(__DAV_VEC__)
        currentTask_ = task;
        RunVector(batch, kvHead, flatQTile);
#endif
    }

    __aicore__ inline void LoadQ(BlockMmadQK &mm, uint32_t batch, uint32_t kvHead, uint32_t flatQTile) {
        uint32_t firstQHead = kvHead * gqa_;
        if (tiling_->seqLenQ % Q_RAW_TILE != 0U) {
            uint64_t headOffset =
                (static_cast<uint64_t>(batch) * tiling_->headNumQ + firstQHead) * tiling_->seqLenQ * HEAD_DIM;
            auto qTensor = qGm_[headOffset];
            uint32_t validTiles = min(Q_TILES_PER_GROUP, qTilesPerKvHead_ - flatQTile);
            mm.loadQPackedQGMSegmented(qTensor, Q_GROUP_PACKED_ROWS, HEAD_DIM, PACK_STRIDE, tiling_->seqLenQ, flatQTile,
                Q_PACKED_TILE, validTiles);
            return;
        }
        uint64_t qOffset = ((static_cast<uint64_t>(batch) * tiling_->headNumQ + firstQHead) * tiling_->seqLenQ +
                               static_cast<uint64_t>(flatQTile) * Q_RAW_TILE) *
            HEAD_DIM;
        auto qTensor = qGm_[qOffset];
        uint32_t validQTiles = min(Q_TILES_PER_GROUP, qTilesPerKvHead_ - flatQTile);
        uint32_t validPackedRows = validQTiles * Q_PACKED_TILE;
        mm.loadQPackedQGM(qTensor, Q_GROUP_PACKED_ROWS, validPackedRows, HEAD_DIM, PACK_STRIDE);
    }

    __aicore__ inline void RunCube(BlockMmadQK &mm, uint32_t batch, uint32_t kvHead, uint32_t flatQTile, bool reuseQ) {
        if (reuseQ) {
            // Consume the previous sweep's Q-buffer credit even without a
            // reload, so the following SetFlag cannot duplicate that credit.
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        } else {
            LoadQ(mm, batch, kvHead, flatQTile);
        }

        uint64_t kOffset = ((static_cast<uint64_t>(batch) * tiling_->headNumKV + kvHead) * tiling_->seqLenK +
                               static_cast<uint64_t>(segmentStart_) * K_RAW_TILE) *
            HEAD_DIM;
        uint32_t segmentSeqLen =
            min(tiling_->seqLenK - segmentStart_ * K_RAW_TILE, (segmentEnd_ - segmentStart_) * K_RAW_TILE);

        auto kTensor = kGm_[kOffset];
        for (uint32_t kt = segmentStart_; kt < segmentEnd_; ++kt) {
            uint32_t nAct = min(K_PACKED_TILE, kPackedCols_ - kt * K_PACKED_TILE);
            uint32_t ubId = kt & 1U;
            // Restart K prefetch at each segment; S parity remains global.
            uint32_t localKt = kt - segmentStart_;
            mm.Compute(kTensor, sUb_[ubId], localKt, segmentSeqLen, nAct, static_cast<uint16_t>(ubId));
        }
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        // Histories, output buffers, and the single Q L1 buffer are reused by
        // the next task. Wait for both AIV owners.
        uint16_t taskDone0 = FLAG_TASK_DONE;
        uint16_t taskDone1 = FLAG_TASK_DONE + 16U;
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(taskDone0);
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(taskDone1);
    }

    __aicore__ inline void RunVector(uint32_t batch, uint32_t kvHead, uint32_t flatQTile) {
        if (segmentStart_ == 0U) {
            Duplicate(localMaxUb_, -3.4e38f, AIV_Q_ROWS);
            Duplicate(localSumUb_, 0.0f, AIV_Q_ROWS);
            PipeBarrier<PIPE_V>();
        } else {
            // State belongs to a logical Q task, not to the physical core.
            // Order earlier workspace writes and the previous UB consumer.
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            uint64_t state = WorkspaceAivBase() + workspaceStateOffset_;
            DataCopy(localMaxUb_, workspaceGm_[state], AIV_Q_ROWS);
            DataCopy(localSumUb_, workspaceGm_[state + AIV_Q_ROWS], AIV_Q_ROWS);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
        }

        for (uint32_t kt = segmentStart_; kt < segmentEnd_; ++kt) {
            uint32_t nAct = min(K_PACKED_TILE, kPackedCols_ - kt * K_PACKED_TILE);
            uint16_t readyFlag = kt & 1U;
            AscendC::CrossCoreWaitFlag<4, PIPE_V>(readyFlag);
            ProcessSTile(kt, nAct, sUb_[kt & 1U]);
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(readyFlag);
        }

        if (segmentEnd_ < kTiles_) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            uint64_t base = WorkspaceAivBase();
            uint32_t segmentTiles = segmentEnd_ - segmentStart_;
            DataCopy(workspaceGm_[base + segmentStart_ * AIV_Q_ROWS * KV_BLOCKS_PER_TILE], firstReduceHistoryUb_,
                segmentTiles * AIV_Q_ROWS * KV_BLOCKS_PER_TILE);
            DataCopy(workspaceGm_[base + workspaceMaxOffset_ + segmentStart_ * AIV_Q_ROWS], maxHistoryUb_,
                segmentTiles * AIV_Q_ROWS);
            uint64_t state = WorkspaceAivBase() + workspaceStateOffset_;
            DataCopy(workspaceGm_[state], localMaxUb_, AIV_Q_ROWS);
            DataCopy(workspaceGm_[state + AIV_Q_ROWS], localSumUb_, AIV_Q_ROWS);
            // The next Q task may immediately overwrite these UB vectors.
            SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
            uint16_t taskDone = FLAG_TASK_DONE;
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(taskDone);
            return;
        }

        if (tiling_->seqLenQ % Q_BLOCK != 0U) {
            WeightQueryTail(flatQTile);
        }
        if (longK_) {
            // Order prior workspace writes on MTE3 before reconstruction
            // reads earlier segments through MTE2; the final segment stays in UB.
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
            BuildPartialScoreLong();
        } else {
            BuildPartialScore();
        }
        uint32_t outputRows = 0U;
        uint64_t firstOutputRow = 0U;
        uint32_t firstLocalQBlock = subBlockIdx_ * Q_BLOCKS_PER_AIV;
        for (uint32_t qBlock = 0; qBlock < Q_BLOCKS_PER_AIV; ++qBlock) {
            uint32_t qHead = 0;
            uint32_t qBlockInHead = 0;
            if (DecodeOutputBlock(kvHead, flatQTile, firstLocalQBlock + qBlock, qHead, qBlockInHead)) {
                if (outputRows == 0U) {
                    firstOutputRow =
                        (static_cast<uint64_t>(batch) * tiling_->headNumQ + qHead) * qBlocksPerHead_ + qBlockInHead;
                }
                maskUb_ = batchMaskUb_[outputRows * kvBlocksAligned_];
                countUb_ = batchCountUb_[outputRows * OUTPUT_COUNT_STRIDE];
                ++outputRows;
                if (longK_) {
                    LoadLongScore(qBlock);
                    FinalizeMask(qBlockInHead, 0U);
                } else {
                    FinalizeMask(qBlockInHead, qBlock);
                }
            }
        }
        if (outputRows != 0U) {
            WriteMaskGroup(firstOutputRow, outputRows);
        }
        uint16_t taskDone = FLAG_TASK_DONE;
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(taskDone);
    }

    // Apply Q-tail weights once, through the normalization denominator.
    // A partial pack contributes validTokens/8; fully padded packs contribute
    // zero via an infinite denominator. Online max/sum history stays unchanged.
    __aicore__ inline void WeightQueryTail(uint32_t flatQTile) {
        constexpr uint32_t TILES_PER_AIV = Q_TILES_PER_GROUP / 2U;
        uint32_t firstTile = flatQTile + subBlockIdx_ * TILES_PER_AIV;
        for (uint32_t tile = 0; tile < TILES_PER_AIV; ++tile) {
            uint32_t flatTile = firstTile + tile;
            if (flatTile >= qTilesPerKvHead_ || flatTile % qTilesPerHead_ != qTilesPerHead_ - 1U) {
                continue;
            }
            uint32_t validRawRows = tiling_->seqLenQ - (qTilesPerHead_ - 1U) * Q_RAW_TILE;
            uint32_t validPackedRows = CeilDiv(validRawRows, PACK_STRIDE);
            uint32_t rowBase = tile * Q_PACKED_TILE;
            uint32_t tailTokens = validRawRows % PACK_STRIDE;
            // The first padded row need not be 32-byte aligned (Q300 starts
            // padding at packed row 6). Use scalar stores for this short suffix
            // instead of passing an unaligned tensor to vector Duplicate.
            SetFlag<HardEvent::V_S>(EVENT_ID5);
            WaitFlag<HardEvent::V_S>(EVENT_ID5);
            if (tailTokens != 0U) {
                uint32_t row = rowBase + validPackedRows - 1U;
                localSumUb_.SetValue(row,
                    localSumUb_.GetValue(row) * (static_cast<float>(PACK_STRIDE) / static_cast<float>(tailTokens)));
            }
            auto sumBits = localSumUb_.ReinterpretCast<uint32_t>();
            for (uint32_t row = validPackedRows; row < Q_PACKED_TILE; ++row) {
                // IEEE FP32 +infinity: finite numerator / infinity is zero.
                sumBits.SetValue(rowBase + row, 0x7f800000U);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID5);
            WaitFlag<HardEvent::S_V>(EVENT_ID5);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessSTile(uint32_t kt, uint32_t nAct, LocalTensor<float> s) {
        // Each K tile owns a history slot until this segment is complete.
        // No history DMA is issued on the steady-state K512 critical path.
        uint32_t historyIndex = longK_ ? kt - segmentStart_ : kt;
        uint32_t firstHistoryOffset = historyIndex * AIV_Q_ROWS * KV_BLOCKS_PER_TILE;
        uint32_t maxHistoryOffset = historyIndex * AIV_Q_ROWS;
        auto firstReduceHistory = firstReduceHistoryUb_[firstHistoryOffset];
        auto maxHistory = maxHistoryUb_[maxHistoryOffset];
        OnlineSoftmaxReduceVF<AIV_Q_ROWS>((__ubuf__ float *)s.GetPhyAddr(), (__ubuf__ float *)localMaxUb_.GetPhyAddr(),
            (__ubuf__ float *)localSumUb_.GetPhyAddr(), (__ubuf__ float *)maxHistory.GetPhyAddr(),
            (__ubuf__ float *)firstReduceHistory.GetPhyAddr(), (__ubuf__ float *)expInputUb_.GetPhyAddr(), nAct,
            tiling_->scaleFactor);
        // The S-buffer credit is sent on MTE3 immediately after this function.
        // Preserve the V->MTE3 dependency even though the history no longer
        // leaves UB; otherwise the cube may overwrite S before VF finishes.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
    }

    __aicore__ inline void BuildPartialScore() {
        uint32_t matrixElems = AIV_Q_ROWS * kvBlocksAligned_;
        Duplicate(partial0Ub_, 0.0f, matrixElems);
        PipeBarrier<PIPE_V>();
        for (uint32_t kt = 0; kt < kTiles_; ++kt) {
            uint32_t firstHistoryOffset = kt * AIV_Q_ROWS * KV_BLOCKS_PER_TILE;
            uint32_t maxHistoryOffset = kt * AIV_Q_ROWS;
            auto firstReduceHistory = firstReduceHistoryUb_[firstHistoryOffset];
            auto maxHistory = maxHistoryUb_[maxHistoryOffset];
            uint32_t kvBlockStart = kt * KV_BLOCKS_PER_TILE;
            Sub(expInputUb_, maxHistory, localMaxUb_, AIV_Q_ROWS);
            PipeBarrier<PIPE_V>();
            Exp(expOutputUb_, expInputUb_, AIV_Q_ROWS);
            PipeBarrier<PIPE_V>();
            Div(expOutputUb_, expOutputUb_, localSumUb_, AIV_Q_ROWS);
            PipeBarrier<PIPE_V>();
            for (uint32_t b = 0; b < KV_BLOCKS_PER_TILE && kvBlockStart + b < kvBlocks_; ++b) {
                Mul(partial0Ub_[(kvBlockStart + b) * AIV_Q_ROWS], firstReduceHistory[b * AIV_Q_ROWS], expOutputUb_,
                    AIV_Q_ROWS);
            }
            PipeBarrier<PIPE_V>();
        }

        // Each AIV owns four complete groups of 16 packed rows, equivalent to
        // two complete raw Q256 tiles.
        for (uint32_t qBlock = 0; qBlock < Q_BLOCKS_PER_AIV; ++qBlock) {
            uint32_t rowStart = qBlock * Q_ROWS_PER_BLOCK;
            Sum(scoreUb_[qBlock * kvBlocksAligned_], partial0Ub_[rowStart], sumTmpUb_,
                SumParams{kvBlocksAligned_, AIV_Q_ROWS, Q_ROWS_PER_BLOCK});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline uint64_t WorkspaceAivBase() const {
        uint32_t aivOwner = currentTask_ * 2U + subBlockIdx_;
        return static_cast<uint64_t>(aivOwner) * workspaceStrideElems_;
    }

    __aicore__ inline void BuildPartialScoreLong() {
        // Drain the last online-softmax reads before reconstruction overwrites S storage.
        // All final K-tile ready events have been consumed; TASK_DONE remains
        // withheld until reconstruction and every mask write are complete.
        PipeBarrier<PIPE_V>();
        uint64_t base = WorkspaceAivBase();
        auto partial = ub_[UB_SEGMENT_PARTIAL_OFFSET].template ReinterpretCast<float>();
        auto work = ub_[UB_SEGMENT_SCORE_OFFSET].template ReinterpretCast<float>();
        auto scratch = ub_[UB_SEGMENT_SUM_OFFSET].template ReinterpretCast<uint8_t>();
        Duplicate(work, 0.0f, Q_BLOCKS_PER_AIV * kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        // Consume the last segment directly from UB before loading any earlier
        // history. It was never written to GM. Earlier segments use bulk DMA.
        for (uint32_t pass = 0; pass < kTiles_; pass += K_SEGMENT_TILES) {
            uint32_t chunkStart = pass == 0U ? segmentStart_ : pass - K_SEGMENT_TILES;
            uint32_t chunkTiles = min(K_SEGMENT_TILES, kTiles_ - chunkStart);
            if (pass != 0U) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
                DataCopy(firstReduceHistoryUb_, workspaceGm_[base + chunkStart * AIV_Q_ROWS * KV_BLOCKS_PER_TILE],
                    chunkTiles * AIV_Q_ROWS * KV_BLOCKS_PER_TILE);
                DataCopy(maxHistoryUb_, workspaceGm_[base + workspaceMaxOffset_ + chunkStart * AIV_Q_ROWS],
                    chunkTiles * AIV_Q_ROWS);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            }
            for (uint32_t batchOffset = 0; batchOffset < chunkTiles; batchOffset += SCORE_BATCH_TILES) {
                uint32_t batchStart = chunkStart + batchOffset;
                uint32_t batchTiles = min(SCORE_BATCH_TILES, chunkTiles - batchOffset);
                for (uint32_t batchTile = 0; batchTile < batchTiles; ++batchTile) {
                    uint32_t localTile = batchOffset + batchTile;
                    auto tileMax = maxHistoryUb_[localTile * AIV_Q_ROWS];
                    auto tileHistory = firstReduceHistoryUb_[localTile * AIV_Q_ROWS * KV_BLOCKS_PER_TILE];
                    Sub(expInputUb_, tileMax, localMaxUb_, AIV_Q_ROWS);
                    PipeBarrier<PIPE_V>();
                    Exp(expOutputUb_, expInputUb_, AIV_Q_ROWS);
                    PipeBarrier<PIPE_V>();
                    Div(expOutputUb_, expOutputUb_, localSumUb_, AIV_Q_ROWS);
                    PipeBarrier<PIPE_V>();
                    uint32_t partialBlockStart = batchTile * KV_BLOCKS_PER_TILE;
                    for (uint32_t b = 0; b < KV_BLOCKS_PER_TILE; ++b) {
                        Mul(partial[(partialBlockStart + b) * AIV_Q_ROWS], tileHistory[b * AIV_Q_ROWS], expOutputUb_,
                            AIV_Q_ROWS);
                    }
                    PipeBarrier<PIPE_V>();
                }

                uint32_t batchBlocks = batchTiles * KV_BLOCKS_PER_TILE;
                for (uint32_t qBlock = 0; qBlock < Q_BLOCKS_PER_AIV; ++qBlock) {
                    uint32_t rowStart = qBlock * Q_ROWS_PER_BLOCK;
                    Sum(work[qBlock * kvBlocksAligned_ + batchStart * KV_BLOCKS_PER_TILE], partial[rowStart], scratch,
                        SumParams{batchBlocks, AIV_Q_ROWS, Q_ROWS_PER_BLOCK});
                    // Finish source reads and score writes before reusing the batch storage.
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    __aicore__ inline void LoadLongScore(uint32_t qBlock) {
        auto compact = ub_[UB_SEGMENT_SCORE_OFFSET].template ReinterpretCast<float>();
        Adds(scoreUb_, compact[qBlock * kvBlocksAligned_], 0.0f, kvBlocksAligned_);
        SetFlag<HardEvent::V_S>(EVENT_ID5);
        WaitFlag<HardEvent::V_S>(EVENT_ID5);
        for (uint32_t col = kvBlocks_; col < kvBlocksAligned_; ++col) {
            scoreUb_.SetValue(col, 0.0f);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID5);
        WaitFlag<HardEvent::S_V>(EVENT_ID5);
    }

    __aicore__ inline void FinalizeMask(uint32_t qBlockGlobal, uint32_t qBlockInAiv) {
        LocalTensor<float> score = scoreUb_[qBlockInAiv * kvBlocksAligned_];

        // Keep an untouched copy for final threshold comparison.  The working
        // copy excludes protected columns from both the denominator and Sort,
        // matching the original scalar selection semantics.
        Adds(scoreWorkUb_, score, 0.0f, kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID5);
        WaitFlag<HardEvent::V_S>(EVENT_ID5);
        if (tiling_->setFirstCol && kvBlocks_ > 0) {
            scoreWorkUb_.SetValue(0, 0.0f);
        }
        bool protectLast = tiling_->setDiag || !tiling_->causal;
        if (protectLast && kvBlocks_ > 0) {
            scoreWorkUb_.SetValue(kvBlocks_ - 1U, 0.0f);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID5);
        WaitFlag<HardEvent::S_V>(EVENT_ID5);

        // Follow A2's mask path: sort on Vector, then reduce adaptive pieces of sorted
        // values and the total in one VF launch. Scalar only scans the pieces
        // and the selected piece's values.
        ArithProgression<int32_t>(sortIndexUb_.ReinterpretCast<int32_t>(), 0, 1, kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(sortProposalUb_, scoreWorkUb_, sortIndexUb_, sortTmpUb_, kvBlocksAligned_ / 32U);
        PipeBarrier<PIPE_V>();
        Extract<float>(sortedScoreUb_, sortedIndexUb_, sortProposalUb_, kvBlocksAligned_ / 32U);
        PipeBarrier<PIPE_V>();
        ReduceScorePiecesVF<MASK_SCORE_PIECE_SIZE>((__ubuf__ float *)sortedScoreUb_.GetPhyAddr(),
            (__ubuf__ float *)pieceSumsUb_.GetPhyAddr(), (__ubuf__ float *)scoreDenomUb_.GetPhyAddr(),
            kvBlocksAligned_);
        SetFlag<HardEvent::V_S>(EVENT_ID5);
        WaitFlag<HardEvent::V_S>(EVENT_ID5);

        float total = scoreDenomUb_.GetValue(0);
        uint32_t keepCap = kvBlocks_;
        if (tiling_->rowSparse > 0.0f && tiling_->rowSparse < 1.0f) {
            keepCap = max(1U, static_cast<uint32_t>(kvBlocks_ * tiling_->rowSparse));
        }
        float target = total * tiling_->threshold;
        float cumulative = 0.0f;
        float guard = 0.0f;
        if (tiling_->threshold < 1.0f) {
            uint32_t pieceStart = 0U;
            uint32_t completePieces = keepCap / MASK_SCORE_PIECE_SIZE;
            for (uint32_t piece = 0; piece < completePieces; ++piece) {
                float pieceSum = pieceSumsUb_.GetValue(piece);
                if (cumulative + pieceSum >= target) {
                    pieceStart = piece * MASK_SCORE_PIECE_SIZE;
                    break;
                }
                cumulative += pieceSum;
                pieceStart = (piece + 1U) * MASK_SCORE_PIECE_SIZE;
            }
            uint32_t searchEnd = min(pieceStart + MASK_SCORE_PIECE_SIZE, keepCap);
            if (pieceStart == keepCap) {
                guard = sortedScoreUb_.GetValue(keepCap - 1U);
            } else {
                for (uint32_t index = pieceStart; index < searchEnd; ++index) {
                    guard = sortedScoreUb_.GetValue(index);
                    cumulative += guard;
                    if (cumulative >= target) {
                        break;
                    }
                }
                // A Vector piece reduction and the Scalar in-piece scan can
                // differ by one FP32 rounding step. Continue only in that rare
                // case so the threshold is never stopped before it is reached.
                for (uint32_t index = searchEnd; index < keepCap && cumulative < target; ++index) {
                    guard = sortedScoreUb_.GetValue(index);
                    cumulative += guard;
                }
            }
        }
        guard -= 1.0e-6f;

        // Vectorized score >= guard conversion, following A2's stable
        // Relu/(x+eps)/ceil sequence.  Padding scores are zeroed explicitly by
        // the aligned reduction and the output mask is initialized to zero.
        SetFlag<HardEvent::S_V>(EVENT_ID5);
        WaitFlag<HardEvent::S_V>(EVENT_ID5);
        Adds(scoreWorkUb_, score, -guard, kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        Relu(scoreWorkUb_, scoreWorkUb_, kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        Adds(scoreDenomUb_, scoreWorkUb_, 0.001f, kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        Div(scoreWorkUb_, scoreWorkUb_, scoreDenomUb_, kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        Duplicate(maskUb_, static_cast<int8_t>(0), kvBlocksAligned_);
        PipeBarrier<PIPE_V>();
        Cast(maskHalfUb_, scoreWorkUb_, RoundMode::CAST_NONE, kvBlocks_);
        PipeBarrier<PIPE_V>();
        Cast(maskUb_, maskHalfUb_, RoundMode::CAST_CEIL, kvBlocks_);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_S>(EVENT_ID5);
        WaitFlag<HardEvent::V_S>(EVENT_ID5);
        if (tiling_->setFirstCol && kvBlocks_ > 0) {
            maskUb_.SetValue(0, static_cast<int8_t>(1));
        }
        if (protectLast && kvBlocks_ > 0) {
            maskUb_.SetValue(kvBlocks_ - 1U, static_cast<int8_t>(1));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID5);
        WaitFlag<HardEvent::S_V>(EVENT_ID5);
        Cast(maskHalfUb_, maskUb_, RoundMode::CAST_NONE, kvBlocks_);
        PipeBarrier<PIPE_V>();
        // Counts above 2048 may not be exactly representable in FP16.
        // Reuse expired score workspaces for an exact FP32 mask sum.
        Cast(scoreWorkUb_, maskHalfUb_, RoundMode::CAST_NONE, kvBlocks_);
        PipeBarrier<PIPE_V>();
        Sum(scoreDenomUb_, scoreWorkUb_, sumTmpUb_, SumParams{1U, kvBlocksAligned_, kvBlocks_});
        PipeBarrier<PIPE_V>();
        Cast(countUb_, scoreDenomUb_, RoundMode::CAST_ROUND, 1U);
        // The next row reuses scoreDenomUb_, but has its own mask/count slot.
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteMaskGroup(uint64_t firstRow, uint32_t rows) {
        // Valid rows are contiguous in BNSD output even across a Q-head boundary:
        // DecodeOutputBlock skips padded Q blocks, and UB slots are packed likewise.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        DataCopy(maskGm_[firstRow * kvBlocksAligned_], batchMaskUb_, rows * kvBlocksAligned_);
        DataCopyExtParams params{};
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = sizeof(int32_t);
        // Normal padding reads one 32-byte UB block per count and writes 4 bytes
        // to GM. Zero strides pack the counts without copying the UB padding.
        DataCopyPad(countGm_[firstRow], batchCountUb_, params);
        // Both scalar and Vector may reuse this storage in the next Q task.
        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    }

    __aicore__ inline void InitPipeEvents() {
#ifdef __DAV_CUBE__
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
#endif
    }

    __aicore__ inline void ReleasePipeEvents() {
#ifdef __DAV_CUBE__
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
#endif
    }

  private:
#ifdef __DAV_CUBE__
    BlockMmadQK mm_;
    uint32_t residentQTask_ = 0;
#endif
    LocalTensor<uint8_t> ub_{TPosition::VECCALC, 0, 256U * 1024U};
    const SparseBlockEstimateTilingData *tiling_ = nullptr;
    GlobalTensor<InputT> qGm_;
    GlobalTensor<InputT> kGm_;
    GlobalTensor<int8_t> maskGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<float> workspaceGm_;
    LocalTensor<float> sUb_[2];
    LocalTensor<float> localMaxUb_;
    LocalTensor<float> localSumUb_;
    LocalTensor<float> firstReduceHistoryUb_;
    LocalTensor<float> maxHistoryUb_;
    LocalTensor<float> expInputUb_;
    LocalTensor<float> expOutputUb_;
    LocalTensor<float> pieceSumsUb_;
    LocalTensor<float> partial0Ub_;
    LocalTensor<float> sortTmpUb_;
    LocalTensor<float> sortProposalUb_;
    LocalTensor<float> sortedScoreUb_;
    LocalTensor<uint32_t> sortIndexUb_;
    LocalTensor<uint32_t> sortedIndexUb_;
    LocalTensor<uint8_t> sumTmpUb_;
    LocalTensor<int8_t> maskUb_;
    LocalTensor<int8_t> batchMaskUb_;
    LocalTensor<int32_t> batchCountUb_;
    LocalTensor<float> scoreUb_;
    LocalTensor<float> scoreWorkUb_;
    LocalTensor<float> scoreDenomUb_;
    LocalTensor<half> maskHalfUb_;
    LocalTensor<int32_t> countUb_;
    uint32_t qTilesPerHead_ = 0;
    uint32_t qBlocksPerHead_ = 0;
    uint32_t gqa_ = 0;
    uint32_t qTilesPerKvHead_ = 0;
    uint32_t qGroupsPerKvHead_ = 0;
    uint32_t kPackedCols_ = 0;
    uint32_t kTiles_ = 0;
    uint32_t kvBlocks_ = 0;
    uint32_t kvBlocksAligned_ = 0;
    uint32_t totalTasks_ = 0;
    uint32_t taskBegin_ = 0;
    uint32_t subBlockIdx_ = 0;
    uint32_t workspaceMaxOffset_ = 0;
    uint32_t workspaceStateOffset_ = 0;
    uint32_t workspaceStrideElems_ = 0;
    uint32_t currentTask_ = 0;
    uint32_t segmentStart_ = 0;
    uint32_t segmentEnd_ = 0;
    bool longK_ = false;
};

// Initialize SDK state once around both phases, never between them.
// Workspace IDs remain absolute: two half tasks occupy exactly the workspace
// range of the original full task. No inter-core merge or allocation is needed.
template <typename InputT> class SparseBlockEstimateKernel {
  public:
    __aicore__ inline SparseBlockEstimateKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR mask, GM_ADDR count, GM_ADDR workspace,
        const SparseBlockEstimateTilingData &tiling) {
        uint32_t tilesPerKv = CeilDiv(tiling.seqLenQ, Q_RAW_TILE) * (tiling.headNumQ / tiling.headNumKV);
        total_ = tiling.batchSize * tiling.headNumKV * CeilDiv(tilesPerKv, 4U);
        prefix_ =
            FullQTaskPrefix(total_, tiling.coreNumAic, tilesPerKv, tiling.seqLenK > MAX_KV_BLOCKS_ALIGNED * KV_BLOCK);
        GM_ADDR user = AscendC::GetUserWorkspace(workspace);
        full_.Init(query, key, mask, count, user, tiling);
        if (prefix_ != total_) {
            tail_.Init(query, key, mask, count, user, tiling);
        }
    }

    __aicore__ inline void InitBuffers() {
        InitSocState();
        full_.InitBuffers();
        if (prefix_ != total_) {
            tail_.InitBuffers();
        }
    }

    __aicore__ inline void Process() {
        full_.Process(0U, prefix_, prefix_ != total_);
        if (prefix_ != total_) {
            tail_.Process(prefix_ * 2U, total_ * 2U);
        }
        InitSocState();
    }

  private:
    Kernel<InputT, 4U> full_;
    Kernel<InputT, 2U> tail_;
    uint32_t total_ = 0;
    uint32_t prefix_ = 0;
};

} // namespace SparseEstimateArch35

#endif // SPARSE_BLOCK_ESTIMATE_ARCH35_KERNEL_H
