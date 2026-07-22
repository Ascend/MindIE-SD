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

#ifndef EPILOGUE_TILE_TILE_CAST_HPP
#define EPILOGUE_TILE_TILE_CAST_HPP

#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class DstType_, class SrcType_,
    /// Length of the compute buffer
    class TileShape_>
struct TileCast {
    using ArchTag = ArchTag_;
    using ElementDst = typename DstType_::Element;
    using ElementSrc = typename SrcType_::Element;
    using TileShape = TileShape_;

    __aicore__ inline TileCast() {}

    __aicore__ inline void operator()(
        AscendC::LocalTensor<ElementDst> const &ubOut, AscendC::LocalTensor<ElementSrc> const &ubIn) {
        AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
    }
};

} // namespace NpuArch::Epilogue::Tile

#endif
