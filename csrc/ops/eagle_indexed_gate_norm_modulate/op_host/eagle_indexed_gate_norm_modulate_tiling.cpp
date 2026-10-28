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
#include "eagle_indexed_gate_norm_modulate_tiling.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr const char *OP_NAME = "EagleIndexedGateNormModulate";
// Keep 8 KiB API reserve and 256 B system area; count explicit scratch separately.
constexpr uint64_t UB_RESERVE = 8192 + 256;
constexpr uint64_t BYTES_PER_COLUMN = 46;
constexpr uint64_t TILED_BYTES_PER_COLUMN = 24;

bool HasShape(gert::TilingContext *context, size_t input, int64_t first, int64_t second = -1) {
    const auto *storage = context->GetInputShape(input);
    if (storage == nullptr) {
        return false;
    }
    const auto &shape = storage->GetStorageShape();
    return shape.GetDimNum() == (second < 0 ? 1U : 2U) && shape.GetDim(0) == first &&
        (second < 0 || shape.GetDim(1) == second);
}
} // namespace

static ge::graphStatus Tiling(gert::TilingContext *context) {
    if (context == nullptr || context->GetPlatformInfo() == nullptr || context->GetAttrs() == nullptr ||
        context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &residual = context->GetInputShape(0)->GetStorageShape();
    const auto &gate = context->GetInputShape(1)->GetStorageShape();
    if (residual.GetDimNum() != 2 || gate.GetDimNum() != 2) {
        OP_LOGE(OP_NAME, "residual and gate must be rank 2.");
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = residual.GetDim(0);
    const int64_t hidden = residual.GetDim(1);
    const int64_t indexedRows = gate.GetDim(0);
    if (rows < 0 || hidden < 1 || hidden > 7168 || indexedRows < 1 ||
        rows > std::numeric_limits<int64_t>::max() / hidden ||
        indexedRows > std::numeric_limits<int64_t>::max() / hidden || !HasShape(context, 1, indexedRows, hidden) ||
        !HasShape(context, 2, rows, hidden) || !HasShape(context, 3, hidden) ||
        !HasShape(context, 4, indexedRows, hidden) || !HasShape(context, 5, indexedRows, hidden) ||
        !HasShape(context, 6, rows)) {
        OP_LOGE(OP_NAME,
            "Expected residual/branch[N,D], gate/shift/scale[K,D], weight[D], indices[N]; "
            "N >= 0, K >= 1 and 1 <= D <= 7168.");
        return ge::GRAPH_FAILED;
    }
    const auto *eps = context->GetAttrs()->GetAttrPointer<float>(0);
    const auto *outputDtype = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    if (eps == nullptr || !std::isfinite(*eps) || *eps <= 0 || outputDtype == nullptr ||
        (*outputDtype != 0 && *outputDtype != 1) || context->GetInputDesc(0) == nullptr) {
        OP_LOGE(OP_NAME, "eps must be finite and positive; output_dtype must be 0 (FP16) or 1 (BF16).");
        return ge::GRAPH_FAILED;
    }
    const auto inputDtype = context->GetInputDesc(0)->GetDataType();
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 6; ++i) {
        if (context->GetInputDesc(i) == nullptr || context->GetInputDesc(i)->GetDataType() != inputDtype) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *indexDesc = context->GetInputDesc(6);
    if (indexDesc == nullptr ||
        (indexDesc->GetDataType() != ge::DT_INT32 && indexDesc->GetDataType() != ge::DT_INT64)) {
        return ge::GRAPH_FAILED;
    }
    const auto expectedOutput = *outputDtype == 0 ? ge::DT_FLOAT16 : ge::DT_BF16;
    for (size_t i = 0; i < 2; ++i) {
        if (context->GetOutputDesc(i) == nullptr || context->GetOutputDesc(i)->GetDataType() != expectedOutput) {
            return ge::GRAPH_FAILED;
        }
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto coreCount = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t aligned = static_cast<uint32_t>((hidden + 127) / 128 * 128);
    const bool tiled = hidden > 3968;
    // Five full-row raw inputs with branch/scale and residual/shift FP32 tile reuse.
    const uint64_t tileSize = hidden <= 6144 ? 3072 : (hidden <= 6656 ? 2048 : 1024);
    const uint64_t scratchBytes = tiled ? 1056 : 288;
    const uint64_t tileBytes = 3 * sizeof(float) * tileSize;
    const uint64_t vectorBytes = tiled ? TILED_BYTES_PER_COLUMN * aligned + tileBytes : BYTES_PER_COLUMN * aligned;
    if (coreCount == 0 || vectorBytes + scratchBytes + UB_RESERVE > ubBytes) {
        OP_LOGE(OP_NAME, "Insufficient AIV or UB resources for the selected single-buffer path.");
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCores = static_cast<uint32_t>(std::min<int64_t>(std::max<int64_t>(rows, 1), coreCount));
    EagleIndexedGateNormModulateTilingData data;
    data.set_rows(rows);
    data.set_indexedRows(indexedRows);
    data.set_hiddenSize(static_cast<uint32_t>(hidden));
    data.set_alignedSize(aligned);
    data.set_usedCores(usedCores);
    data.set_eps(*eps);
    data.set_inverseHidden(1.0f / static_cast<float>(hidden));
    context->SetBlockDim(usedCores);
    context->SetTilingKey(!tiled ? 0 : (hidden <= 6144 ? 3 : (hidden <= 6656 ? 1 : 2)));
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (raw == nullptr || raw->GetCapacity() < data.GetDataSize() || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EagleIndexedGateNormModulate).Tiling(Tiling);
} // namespace optiling
