/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "mul_add_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

ge::graphStatus MulAddTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *aShape = context->GetInputShape(0);
    if (aShape == nullptr) {
        OP_LOGE("MulAdd", "Failed to get input shape of MulAdd.");
        return ge::GRAPH_FAILED;
    }

    int64_t batchSize = aShape->GetStorageShape().GetDim(0);
    int64_t seqLen = aShape->GetStorageShape().GetDim(1);
    int64_t hiddenSize = aShape->GetStorageShape().GetDim(2);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(platformInfo.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubSizeLimit = static_cast<int64_t>(ubSize);

    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE("MulAdd", "Failed to get input descriptor of MulAdd.");
        return ge::GRAPH_FAILED;
    }
    auto dataType = inputDesc->GetDataType();
    int64_t dtypeSize = (dataType == ge::DT_BF16) ? 2 : 2; // BF16/FP16 both 2 bytes
    int64_t dtypeFlag = (dataType == ge::DT_BF16) ? 0 : 1;

    int64_t alignElements = 32 / dtypeSize;
    int64_t hiddenSizeAlign = ((hiddenSize + alignElements - 1) / alignElements) * alignElements;

    int64_t residentBBytes = hiddenSizeAlign * static_cast<int64_t>(sizeof(float));
    int64_t availableUb = ubSizeLimit - residentBBytes;

    int64_t perRowBytes = hiddenSizeAlign * dtypeSize;
    int64_t bufferCoeffPerRow = 6;
    int64_t fixedOverhead = 3 * static_cast<int64_t>(sizeof(float)) * hiddenSizeAlign;

    int64_t rowsPerTile = (availableUb - fixedOverhead) / (bufferCoeffPerRow * perRowBytes);
    rowsPerTile = std::max<int64_t>(1L, rowsPerTile);
    rowsPerTile = std::min<int64_t>(rowsPerTile, 4095L);

    int64_t oneBatchLength = seqLen * hiddenSize;
    int64_t oneBatchCore = (oneBatchLength + coreNum - 1) / coreNum;
    int64_t rowsPerCore = (oneBatchCore + hiddenSize - 1) / hiddenSize;
    int64_t oneBatchCoreAlign = rowsPerCore * hiddenSize;

    int64_t cacheLineElements = 512 / dtypeSize;
    oneBatchCoreAlign = ((oneBatchCoreAlign + cacheLineElements - 1) / cacheLineElements) * cacheLineElements;
    oneBatchCoreAlign = ((oneBatchCoreAlign + hiddenSize - 1) / hiddenSize) * hiddenSize;

    if (oneBatchCoreAlign == 0) {
        oneBatchCoreAlign = hiddenSize;
    }

    int64_t usedCoreNum = (oneBatchLength + oneBatchCoreAlign - 1) / oneBatchCoreAlign;
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    int64_t formerNum = 0;
    int64_t formerLength = 0;
    int64_t tailLength = 0;

    if (usedCoreNum == 1) {
        formerNum = 0;
        formerLength = 0;
        tailLength = oneBatchLength;
    } else {
        formerNum = usedCoreNum - 1;
        formerLength = oneBatchCoreAlign;
        tailLength = oneBatchLength - formerNum * formerLength;
    }

    MulAddTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_hiddenSizeAlign(hiddenSizeAlign);
    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_tailLength(tailLength);
    tiling.set_rowsPerTile(rowsPerTile);
    tiling.set_dtypeFlag(dtypeFlag);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(0);

    auto *tilingData = context->GetRawTilingData();
    if (tilingData == nullptr) {
        OP_LOGE("MulAdd", "Failed to get raw tiling data of MulAdd.");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE("MulAdd", "Failed to get workspace sizes of MulAdd.");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMulAdd(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MulAdd)
    .Tiling(MulAddTilingFunc)
    .TilingParse<MulAddCompileInfo>(TilingPrepareForMulAdd);

} // namespace optiling
