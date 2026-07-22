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

#ifndef EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_ROW_HPP
#define EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_ROW_HPP

#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    class TileShape_>
struct TileBroadcastInplaceByRow {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    __aicore__ inline TileBroadcastInplaceByRow() {}

    __aicore__ inline void operator()(AscendC::LocalTensor<ElementCompute> const &ubInOut) {
        constexpr uint32_t eleNumPerVectorFractal =
            static_cast<uint32_t>(BYTE_PER_VECTOR_FRACTAL) / static_cast<uint32_t>(sizeof(ElementCompute));

        constexpr uint64_t mask = eleNumPerVectorFractal;
        constexpr uint8_t repeatTimes = TileShape::COLUMN / eleNumPerVectorFractal;

        AscendC::CopyRepeatParams repeatParams;
        repeatParams.dstStride = 1;
        repeatParams.srcStride = 1;
        repeatParams.dstRepeatSize = BLK_NUM_PER_VECTOR_FRACTAL;
        repeatParams.srcRepeatSize = BLK_NUM_PER_VECTOR_FRACTAL;

        for (uint32_t rowOffset = 1; rowOffset < TileShape::ROW; ++rowOffset) {
            AscendC::Copy(ubInOut[rowOffset * TileShape::COLUMN], ubInOut, mask, repeatTimes, repeatParams);
        }
    }
};

} // namespace NpuArch::Epilogue::Tile

#endif
