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

#include <algorithm>
#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tiling_data.h"

namespace optiling {
namespace {
constexpr int64_t GROUP = 32;
constexpr int64_t VECTOR_TILE = 256;

int64_t CeilDiv(int64_t n, int64_t d) { return n / d + (n % d != 0); }

ge::graphStatus Tile(gert::TilingContext *context) {
    const auto *x = context->GetInputShape(0);
    const auto *gate = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const auto *mode = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *scaleAlg = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    if (!x || !gate || !mode || (*mode != 0 && *mode != 1) || !scaleAlg || (*scaleAlg != 0 && *scaleAlg != 1) ||
        !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = x->GetStorageShape();
    const auto &gateShape = gate->GetStorageShape();
    if (shape.GetDimNum() != 2 || gateShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = shape.GetDim(0), cols = shape.GetDim(1);
    if (rows <= 0 || cols <= 0 || rows != gateShape.GetDim(0) || cols != gateShape.GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t aiv = platform.GetCoreNumAiv();
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (aiv <= 0) {
        return ge::GRAPH_FAILED;
    }
    SigmoidMulMxTilingData tile{};
    tile.rows = rows;
    tile.cols = cols;
    // All cores start on 32-byte boundaries in both output tensors.
    const int64_t scaleStride = 2 * CeilDiv(cols, 64);
    tile.rowQuantum = 1;
    while ((cols % GROUP * tile.rowQuantum) % GROUP || (scaleStride % GROUP * tile.rowQuantum) % GROUP) {
        tile.rowQuantum *= 2;
    }
    const auto cores = std::min(aiv, CeilDiv(rows, tile.rowQuantum));
    int64_t cap = *mode == 0 ? 16384 : 18432;
    for (; cap >= VECTOR_TILE; cap -= VECTOR_TILE) {
        tile.tileCols = std::min(cap, cols);
        tile.tileRows = cols % 64 == 0 ? std::min(rows, cap / tile.tileCols) : 1;
        tile.bufferElements = CeilDiv(tile.tileRows * tile.tileCols, VECTOR_TILE) * VECTOR_TILE;
        tile.scaleCapacity = CeilDiv(tile.bufferElements / GROUP, 128) * 128;
        // Each group needs two scale output slots, one BF16 reciprocal and one group maximum.
        const uint32_t maxBytes = *mode == 0 && *scaleAlg == 1 ? sizeof(uint32_t) : sizeof(uint16_t);
        uint64_t bytes = tile.bufferElements * (*mode == 0 ? 14U : 12U) + (4U + maxBytes) * tile.scaleCapacity;
        if (bytes <= ub) {
            break;
        }
    }
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (cap < VECTOR_TILE || !raw || raw->GetCapacity() < sizeof(tile) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &tile, sizeof(tile));
    raw->SetDataSize(sizeof(tile));
    workspace[0] = 0;
    context->SetBlockDim(cores);
    context->SetTilingKey(*mode + 2 * *scaleAlg);
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_OPTILING(SigmoidMulDynamicMxQuant).Tiling(Tile);
} // namespace optiling
