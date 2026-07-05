/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_COLUMN_HPP
#define EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_COLUMN_HPP

#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    class TileShape_>
struct TileBroadcastInplaceByColumn {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    __aicore__ inline TileBroadcastInplaceByColumn() {}

    __aicore__ inline void operator()(AscendC::LocalTensor<ElementCompute> const &ubInOut) {
        constexpr uint32_t eleNumPerBlk =
            static_cast<uint32_t>(BYTE_PER_BLK) / static_cast<uint32_t>(sizeof(ElementCompute));
        constexpr uint32_t blkNumPerRow = TileShape::COLUMN / eleNumPerBlk;

        constexpr uint64_t defaultMask = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);
        constexpr uint64_t tailMask = (TileShape::ROW % BLK_NUM_PER_VECTOR_FRACTAL) * eleNumPerBlk;

        constexpr uint8_t repeatTimes = 1;

        AscendC::CopyRepeatParams repeatParams;
        repeatParams.dstStride = blkNumPerRow;
        repeatParams.srcStride = blkNumPerRow;
        repeatParams.dstRepeatSize = 1;
        repeatParams.srcRepeatSize = 1;

        for (uint32_t rowOffset = 0; rowOffset < TileShape::ROW; rowOffset += BLK_NUM_PER_VECTOR_FRACTAL) {
            uint64_t mask = ((TileShape::ROW - rowOffset) >= BLK_NUM_PER_VECTOR_FRACTAL) ? defaultMask : tailMask;
            for (uint32_t colOffset = eleNumPerBlk; colOffset < TileShape::COLUMN; colOffset += eleNumPerBlk) {
                AscendC::Copy(ubInOut[rowOffset * TileShape::COLUMN + colOffset],
                    ubInOut[rowOffset * TileShape::COLUMN], mask, 1, repeatParams);
            }
        }
    }
};

} // namespace NpuArch::Epilogue::Tile

#endif
