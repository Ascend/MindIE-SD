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

#include "indexed_adaln_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

namespace {
constexpr int64_t kFp32Size = 4;
constexpr int64_t kMaxHidden = 16384;
constexpr int64_t kUbReserve = 8192; // leave headroom for TPipe bookkeeping

ge::graphStatus CheckAndGetShapes(gert::TilingContext *context, int64_t &batch, int64_t &seqLen, int64_t &hidden,
    int64_t &tableK, int64_t &dtypeSize) {
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *scaleShape = context->GetInputShape(1);
    const gert::StorageShape *shiftShape = context->GetInputShape(2);
    const gert::StorageShape *indexShape = context->GetInputShape(3);
    if (xShape == nullptr || scaleShape == nullptr || shiftShape == nullptr || indexShape == nullptr) {
        OP_LOGE("IndexedAdaLayerNorm", "Failed to get input shapes.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &x = xShape->GetStorageShape();
    const gert::Shape &scale = scaleShape->GetStorageShape();
    const gert::Shape &shift = shiftShape->GetStorageShape();
    const gert::Shape &index = indexShape->GetStorageShape();
    if (x.GetDimNum() != 3 || scale.GetDimNum() != 3 || shift.GetDimNum() != 3 || index.GetDimNum() != 1) {
        OP_LOGE("IndexedAdaLayerNorm", "Unexpected input rank.");
        return ge::GRAPH_FAILED;
    }

    batch = x.GetDim(0);
    seqLen = x.GetDim(1);
    hidden = x.GetDim(2);
    tableK = scale.GetDim(1);
    if (batch <= 0 || seqLen <= 0 || hidden <= 0 || tableK <= 0) {
        OP_LOGE("IndexedAdaLayerNorm", "Shapes must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (scale.GetDim(0) != batch || shift.GetDim(0) != batch || scale.GetDim(2) != hidden ||
        shift.GetDim(2) != hidden || shift.GetDim(1) != tableK) {
        OP_LOGE("IndexedAdaLayerNorm", "scale_table/shift_table shape mismatch.");
        return ge::GRAPH_FAILED;
    }
    if (index.GetDim(0) != seqLen) {
        OP_LOGE("IndexedAdaLayerNorm", "mod_index length must equal T.");
        return ge::GRAPH_FAILED;
    }

    auto xDesc = context->GetInputDesc(0);
    auto scaleDesc = context->GetInputDesc(1);
    auto shiftDesc = context->GetInputDesc(2);
    auto indexDesc = context->GetInputDesc(3);
    if (xDesc == nullptr || scaleDesc == nullptr || shiftDesc == nullptr || indexDesc == nullptr) {
        OP_LOGE("IndexedAdaLayerNorm", "Failed to get input descriptors.");
        return ge::GRAPH_FAILED;
    }

    const ge::DataType xDtype = xDesc->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT) {
        OP_LOGE("IndexedAdaLayerNorm", "x must be FP16, BF16 or FP32, got %d.", static_cast<int32_t>(xDtype));
        return ge::GRAPH_FAILED;
    }
    if (scaleDesc->GetDataType() != xDtype || shiftDesc->GetDataType() != xDtype) {
        OP_LOGE("IndexedAdaLayerNorm", "x/scale_table/shift_table must use the same dtype.");
        return ge::GRAPH_FAILED;
    }
    if (indexDesc->GetDataType() != ge::DT_INT64) {
        OP_LOGE("IndexedAdaLayerNorm", "mod_index must be INT64.");
        return ge::GRAPH_FAILED;
    }

    dtypeSize = (xDtype == ge::DT_FLOAT) ? 4 : 2;
    const int64_t rowBytes = hidden * dtypeSize;
    if (hidden > kMaxHidden || (rowBytes % 32) != 0) {
        OP_LOGE("IndexedAdaLayerNorm",
            "H * dtype_size must be a multiple of 32 bytes (K-row DataCopyPad) "
            "and <= %ld, got H=%ld, dtype_size=%ld.",
            kMaxHidden, hidden, dtypeSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus IndexedAdaLayerNormTilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t hidden = 0;
    int64_t tableK = 0;
    int64_t dtypeSize = 2;
    if (CheckAndGetShapes(context, batch, seqLen, hidden, tableK, dtypeSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(platformInfo.GetCoreNumAiv());
    if (coreNum <= 0) {
        coreNum = 40;
    }
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const int64_t totalRows = batch * seqLen;
    const int64_t usedCoreNum = std::min<int64_t>(coreNum, totalRows);
    const int64_t rowsPerCore = totalRows / usedCoreNum;
    const int64_t tailCoreNum = totalRows % usedCoreNum;

    // UB plan (bytes):
    //   x/out (input dtype)                  2 * H * dtypeSize
    //   table (K rows, input dtype)          K * H * dtypeSize
    //   x/y/tmp fp32                         3 * H * 4
    //   scale/shift fp32 caches (K rows)     2 * K * H * 4
    //   mod_index int32 low-word tile        2 * 256 * 4
    const int64_t ubNeed = 2 * hidden * dtypeSize + tableK * hidden * dtypeSize + 3 * hidden * kFp32Size +
        2 * tableK * hidden * kFp32Size + 64 * static_cast<int64_t>(sizeof(float)) +
        2 * 256 * static_cast<int64_t>(sizeof(int32_t));
    if (static_cast<uint64_t>(ubNeed + kUbReserve) > ubSize) {
        OP_LOGE("IndexedAdaLayerNorm", "UB capacity exceeded: need=%ld, ub=%lu. Target shape H=3072, K=3, T=4096.",
            ubNeed + kUbReserve, static_cast<unsigned long>(ubSize));
        return ge::GRAPH_FAILED;
    }

    const auto attrs = context->GetAttrs();
    float epsilon = 1e-6f;
    if (attrs != nullptr) {
        const float *epsPtr = attrs->GetAttrPointer<float>(0);
        if (epsPtr != nullptr) {
            epsilon = *epsPtr;
        }
    }

    IndexedAdaLayerNormTilingData tiling;
    tiling.set_batchSize(batch);
    tiling.set_seqLen(seqLen);
    tiling.set_hiddenSize(hidden);
    tiling.set_tableK(tableK);
    tiling.set_totalRows(totalRows);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_epsilon(epsilon);
    tiling.set_aveFactor(1.0f / static_cast<float>(hidden));

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(0);

    auto *rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr) {
        OP_LOGE("IndexedAdaLayerNorm", "Failed to get raw tiling data.");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        OP_LOGE("IndexedAdaLayerNorm", "Failed to get workspace sizes.");
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForIndexedAdaLayerNorm(gert::TilingParseContext *context) {
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexedAdaLayerNorm)
    .Tiling(IndexedAdaLayerNormTilingFunc)
    .TilingParse<IndexedAdaLayerNormCompileInfo>(TilingPrepareForIndexedAdaLayerNorm);

} // namespace optiling
