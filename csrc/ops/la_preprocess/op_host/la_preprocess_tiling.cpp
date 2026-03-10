/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "la_preprocess_tiling.h"

#include <string>
#include <cinttypes>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


using namespace std;


namespace optiling {

ge::graphStatus LaPreprocessTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* qShape = context->GetInputShape(0);
    const gert::StorageShape* kShape = context->GetInputShape(1);
    const gert::StorageShape* vShape = context->GetInputShape(2);

    if (qShape == nullptr || kShape == nullptr || vShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
    uint32_t qSeqLen = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(1));
    uint32_t kSeqLen = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(1));
    uint32_t vSeqLen = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(1));
    uint32_t headNum = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(2));
    uint32_t headDim = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(3));

    if (context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto alignLen = *(context->GetAttrs()->GetAttrPointer<int32_t>(0));
    if (context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dataType = context->GetInputDesc(0)->GetDataType();

    uint32_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = 1;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivecNum = platformInfo.GetCoreNumAiv();

    LaPreprocessTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_qSeqLen(qSeqLen);
    tiling.set_kSeqLen(kSeqLen);
    tiling.set_vSeqLen(vSeqLen);
    tiling.set_headNum(headNum);
    tiling.set_headDim(headDim);
    tiling.set_alignLen(alignLen);
    tiling.set_ubSize(ubSize);

    context->SetBlockDim(aivecNum);
    context->SetTilingKey(tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForLaPreprocess(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(LaPreprocess)
    .Tiling(LaPreprocessTilingFunc)
    .TilingParse<LaPreprocessCompileInfo>(TilingPrepareForLaPreprocess);

} // namespace optiling
